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
        return result


def episode_collator(items):
    if len(items) != 1:
        raise ValueError("Exactly one unpadded episode per rank/microbatch")
    return items[0]


class EpisodeLengthSampler(Sampler):
    def __init__(self, lengths, seed=429, group_size=64, pool_size=64, world_size=4):
        if pool_size % group_size or group_size % world_size:
            raise ValueError(
                "Pools must contain complete update groups and rank assignments"
            )
        self.lengths, self.seed, self.group_size, self.pool_size, self.world_size = (
            lengths,
            seed,
            group_size,
            pool_size,
            world_size,
        )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
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
