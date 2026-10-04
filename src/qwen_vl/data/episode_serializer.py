"""Immutable fragments shared verbatim by full-episode training and serving."""

import hashlib
import torch
from PIL import Image
from qwen_vl.contracts import (
    ACTIONS,
    HABITAT_IDS,
    CANDIDATES,
    serializer_version,
    validate_episode,
)
from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
from qwen_vl.data.lane_metadata import build_lane_roles
from qwen_vl.research.lane_contract import parse_step_lane_spec

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM = (
    "You control a navigation agent. Follow the instruction using the current RGB view.\n"
    "Reply with exactly one action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.\n"
    "Instruction: "
)


class EpisodeSerializer:
    def __init__(self, processor, config):
        self.processor, self.config = processor, config
        self.tokenizer = processor.tokenizer
        self.mode = config["model"]["output_mode"]
        self.tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        for token, expected in [
            ("<|im_end|>", 248046),
            ("<|im_start|>", 248045),
            ("<|vision_start|>", 248053),
            ("<|vision_end|>", 248054),
        ]:
            if self.tokenizer.convert_tokens_to_ids(token) != expected:
                raise ValueError(f"Unexpected tokenizer: {token}")
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.image_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.separator = self.ids("\n")
        self.action_ids = [self.ids(a) + [self.eos] for a in ACTIONS]
        if max(map(len, self.action_ids)) > 16:
            raise ValueError("Canonical action exceeds response budget")
        # Preserve the existing template bytes, including its Jinja escapes.
        self.observation_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": IMAGE}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.mode == "candidate_logits":
            from qwen_vl.data.candidates import DECISION_CUE, validate_candidates

            self.observation_text += DECISION_CUE
            self.candidate_token_ids = validate_candidates(
                self.tokenizer, CANDIDATES, self.observation_text
            )
        self.template_hash = hashlib.sha256(
            QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()
        ).hexdigest()

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def feedback_ids(self, class_id):
        if class_id not in range(len(ACTIONS)):
            raise ValueError("Invalid feedback class")
        if (
            self.mode == "candidate_logits"
            and self.config["observations"].get("feedback_format") == "none"
        ):
            # Close the assistant turn without revealing the target/prediction.
            # Fixed boundaries remain in the same observation step group.
            return [self.eos] + self.separator
        if (
            self.mode == "candidate_logits"
            and self.config["observations"].get("feedback_format", "candidate_token")
            == "candidate_token"
        ):
            return [self.candidate_token_ids[class_id], self.eos] + self.separator
        return self.action_ids[class_id] + self.separator

    def text_block(self, ids):
        ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)}

    def encode_prefix(self, instruction, output_mode=None):
        if self.mode == "classification":
            text = (
                "<|im_start|>system\n"
                + SYSTEM
                + instruction
                + "<|im_end|>\n<|im_start|>user\n"
            )
        else:
            system = SYSTEM
            if self.mode == "candidate_logits":
                from qwen_vl.data.candidates import CANDIDATE_SYSTEM

                system = CANDIDATE_SYSTEM
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system + instruction},
                    {"role": "user", "content": IMAGE},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if self.mode == "candidate_logits":
                from qwen_vl.data.candidates import DECISION_CUE

                rendered += DECISION_CUE
            if not rendered.endswith(self.observation_text):
                raise ValueError("Chat renderer rewrites historical fragments")
            text = rendered[: -len(self.observation_text)]
        result = self.text_block(self.ids(text))
        if (
            result["input_ids"].numel()
            > self.config["observations"]["max_prefix_tokens"]
        ):
            raise ValueError("Instruction prefix exceeds admission cap")
        return result

    def encode_observation(self, rgb, step_id=0, output_mode=None):
        image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
        if image.mode != "RGB" or image.size != (640, 480):
            raise ValueError(f"Expected RGB 640x480, got {image.mode} {image.size}")
        text = (
            IMAGE + "\nAction:"
            if self.mode == "classification"
            else self.observation_text
        )
        block = dict(
            self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                add_special_tokens=False,
            )
        )
        if block["image_grid_thw"].tolist() != [[1, 30, 40]]:
            raise ValueError(f"Unexpected image grid {block['image_grid_thw']}")
        ids = block["input_ids"]
        if int(ids.eq(self.image_id).sum()) != 300:
            raise ValueError("Expected 300 actual expanded image tokens")
        block["mm_token_type_ids"] = ids.eq(self.image_id).long()
        block.pop("attention_mask", None)
        cap = self.config["observations"]["max_step_group_tokens"]
        reserve = 0 if self.mode == "classification" else 16 + len(self.separator)
        if self.mode == "candidate_logits":
            reserve = max(len(self.feedback_ids(c)) for c in range(len(ACTIONS)))
        if ids.numel() + reserve > cap:
            raise ValueError(
                f"Step {step_id} exceeds cap including response reservation"
            )
        return block

    def encode_episode(self, episode, output_mode=None):
        validate_episode(episode)
        blocks = [self.encode_prefix(episode["instruction"])]
        prefix = blocks[0]["input_ids"].numel()
        cursor = prefix
        lane_spec = parse_step_lane_spec(self.config)
        observation_spans = []
        spans, reads, target_positions, action_index, classes = [], [], [], [], []
        for t, step in enumerate(episode["steps"]):
            with Image.open(step["rgb_path"]) as image:
                block = self.encode_observation(image, t)
            blocks.append(block)
            start = cursor
            cursor += block["input_ids"].numel()
            observation_spans.append((start, cursor))
            cls = ACTIONS.index(step["action_name"])
            classes.append(cls)
            if self.mode == "classification":
                reads.append(cursor - 1)
            elif self.mode == "candidate_logits":
                reads.append(cursor - 1)
                feedback = self.feedback_ids(cls)
                blocks.append(self.text_block(feedback))
                cursor += len(feedback)
            else:
                ids = self.action_ids[cls]
                target_positions.extend(range(cursor, cursor + len(ids)))
                action_index.extend([t] * len(ids))
                blocks.append(self.text_block(ids + self.separator))
                cursor += len(ids) + len(self.separator)
            spans.append((start, cursor))
        if cursor > self.config["training"]["model_max_length"]:
            raise ValueError(
                f"{episode['episode_uid']}: {cursor} tokens exceeds episode admission cap"
            )
        result = {
            key: torch.cat(
                [b[key] for b in blocks if key in b],
                dim=(1 if key in ("input_ids", "mm_token_type_ids") else 0),
            )
            for key in (
                "input_ids",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
            )
        }
        result.update(
            step_plan=(prefix, tuple(spans)),
            action_class_ids=torch.tensor(classes),
            read_positions=torch.tensor(reads, dtype=torch.long),
            response_target_positions=torch.tensor(target_positions, dtype=torch.long),
            response_action_index=torch.tensor(action_index, dtype=torch.long),
            num_actions=len(classes),
            episode_uid=episode["episode_uid"],
        )
        if lane_spec is not None:
            roles = build_lane_roles(prefix, tuple(observation_spans), tuple(spans), cursor)
            for t, (_, end) in enumerate(spans):
                if self.mode == "qwen_text":
                    # Derive independently from the supervised first response target.
                    first_target = next(pos for pos, action in zip(target_positions, action_index) if action == t)
                    decision = first_target - 1
                else:
                    decision = reads[t]
                if end - 1 <= decision:
                    raise ValueError("Lane writer must follow the action decision")
            result["step_lane_roles"] = roles
        return result

    def metadata(self):
        result = dict(
            serializer=serializer_version(self.config),
            template_sha256=self.template_hash,
            actions=list(ACTIONS),
            habitat_ids=list(HABITAT_IDS),
            action_token_ids=self.action_ids,
            eos=self.eos,
            separator_ids=self.separator,
            rgb_shape=[480, 640, 3],
            position_convention="logical_text_plus_native_mrope_v1",
            rotary_precision="fp32_preserved_across_module_casts_v1",
            config=self.config,
        )
        lane_spec = parse_step_lane_spec(self.config)
        if lane_spec is not None:
            result["step_lane"] = lane_spec.to_dict()
        if self.mode == "candidate_logits":
            from qwen_vl.data.candidates import DECISION_CUE

            result.update(
                candidate_tokens=list(CANDIDATES),
                candidate_token_ids=self.candidate_token_ids,
                decision_cue=DECISION_CUE,
                feedback_format=self.config["observations"].get(
                    "feedback_format", "candidate_token"
                ),
                feedback_token_ids=[self.feedback_ids(c) for c in range(len(ACTIONS))],
                action_head_mode=self.config["model"].get(
                    "action_head_mode", "lm_rows_trainable"
                ),
            )
        return result
