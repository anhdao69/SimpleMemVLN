"""Complete episodes and deterministic single-shard length bucketing."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler


class EpisodeDataset(Dataset):
    def __init__(self, manifest, serializer, limit=None, selection="first"):
        self.episodes = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        if selection == "shortest":
            self.episodes.sort(key=lambda e: (len(e["steps"]), e["episode_uid"]))
        if limit:
            self.episodes = self.episodes[:limit]
        self.serializer = serializer
        # Exact token lengths from native 300-token image expansion, validated
        # against actual processor output by the serializer and data audit.
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (640, 480))
        step_length = serializer.encode_observation(image)["input_ids"].numel()
        self.encoded_lengths = []
        for ep in self.episodes:
            length = serializer.encode_prefix(ep["instruction"])["input_ids"].numel()
            length += len(ep["steps"]) * step_length
            if serializer.mode in ("qwen_text", "candidate_logits"):
                from qwen_vl.contracts import ACTIONS

                length += sum(
                    len(serializer.feedback_ids(ACTIONS.index(s["action_name"])))
                    for s in ep["steps"]
                )
            if length > serializer.config["training"]["model_max_length"]:
                raise ValueError(
                    f'{ep["episode_uid"]}: encoded length {length} exceeds cap'
                )
            self.encoded_lengths.append(length)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        result = self.serializer.encode_episode(self.episodes[index])
        if result["input_ids"].numel() != self.encoded_lengths[index]:
            raise ValueError("Audited encoded length disagrees with actual processor")
        if self.serializer.config["runtime"].get("stage_pixels_bf16", False):
            # Trainer/DeepSpeed casts floating inputs to BF16 on transfer anyway;
            # round-to-nearest-even here is bitwise identical and halves staging RAM.
            result["pixel_values"] = result["pixel_values"].to(torch.bfloat16)
        if self.serializer.config.get('research'):
            from qwen_vl.research.vggt_sampling import sample_frames,collection_trajectory_key,sampled_visual_positions
            from qwen_vl.data.vggt_online import preprocess_teacher_images
            episode=self.episodes[index]
            spec=self.serializer.config['research']
            if spec['run_id'] in ('G3','G3_M5'):
                from qwen_vl.research.vggt_g3 import sample_pairs,observation_anchors
                pairs=sample_pairs(collection_trajectory_key(episode),len(episode['steps']),spec.get('sample_seed',429))
                frames=sorted({s for s,t in pairs})
                anchors=observation_anchors(result['response_target_positions'],result['response_action_index'],result['num_actions'])
                result['aux_anchors']=anchors[torch.tensor([t for s,t in pairs],dtype=torch.long)]
                result['aux_lags']=torch.tensor([t-s for s,t in pairs],dtype=torch.long)
                result['aux_source_indices']=torch.tensor([frames.index(s) for s,t in pairs],dtype=torch.long)
                result['aux_pair_count']=len(pairs)
            else:
                frames=sample_frames(collection_trajectory_key(episode),len(episode['steps']),spec.get('sample_seed',429))
                result['aux_visual_positions']=sampled_visual_positions(result['input_ids'],
                    self.serializer.image_id,result['step_plan'][1],frames)
                result['aux_frame_count']=len(frames)
            result['teacher_images']=preprocess_teacher_images([episode['steps'][i]['rgb_path'] for i in frames])
        return result


def episode_collator(items):
    if len(items) != 1:
        raise ValueError("Exactly one unpadded episode per rank/microbatch")
    return items[0]


class EpisodeLengthSampler(Sampler):
    def __init__(self, lengths, seed=429, group_size=64, pool_size=64, world_size=4, epoch_offset=0):
        if pool_size % group_size or group_size % world_size:
            raise ValueError(
                "Pools must contain complete update groups and rank assignments"
            )
        if type(epoch_offset) is not int or epoch_offset < 0:
            raise ValueError("epoch_offset must be a nonnegative integer")
        self.lengths, self.seed, self.group_size, self.pool_size, self.world_size = (
            lengths,
            seed,
            group_size,
            pool_size,
            world_size,
        )
        # A continuation run that starts from an earlier run's weights draws the
        # order that run's next epoch would draw (Trainer epoch 0 -> data epoch offset).
        self.epoch_offset = epoch_offset
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch_offset + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist()
        groups, tail = [], []
        for offset in range(0, len(order), self.pool_size):
            pool = sorted(
                order[offset : offset + self.pool_size],
                key=lambda i: self.lengths[i],
                reverse=True,
            )
            full = len(pool) // self.group_size * self.group_size
            groups.extend(
                pool[i : i + self.group_size] for i in range(0, full, self.group_size)
            )
            tail.extend(pool[full:])
        result = []
        for index in torch.randperm(len(groups), generator=g).tolist():
            group = groups[index]
            # Each consecutive world_size block is one distributed microstep.
            # Balanced alternating ordering avoids one rank always drawing longest.
            ranks = torch.randperm(self.world_size, generator=g).tolist()
            for start in range(0, len(group), self.world_size):
                chunk = group[start : start + self.world_size]
                result.extend(chunk[r] for r in ranks)
                ranks.reverse()
        return iter(result + tail)  # Accelerate alone repeats the final tail.
