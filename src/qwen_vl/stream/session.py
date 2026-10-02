"""Single-episode, idempotent streaming policy with strict failure semantics."""
from collections import deque
import time
import torch
from qwen_vl.contracts import ACTIONS, CANDIDATES, action_result, parse_action
from qwen_vl.stream.cache import (
    make_stream_cache_fp32,
    evict_kv,
    kv_length,
    assert_state_dtypes,
)
from qwen_vl.stream.positions import PositionLedger


class StreamSession:
    def __init__(self, model, serializer):
        self.model, self.serializer = model.eval(), serializer
        self.cfg = model.navigation_config
        self.episode_uid = None
        self.valid = False
        self.busy = False

    @torch.inference_mode()
    def reset(self, episode_uid, instruction):
        self.valid = False
        self.episode_uid = episode_uid
        self.cache = make_stream_cache_fp32(self.model.config.text_config)
        self.positions = PositionLedger()
        self.resident = deque()
        self.next_step = 0
        self.last_result = None
        self.model.backbone.model.rope_deltas = None
        self.busy = False
        self._append(self.serializer.encode_prefix(instruction))
        self.prefix_length = self.positions.logical_token_count
        self.valid = True

    def _append(self, block):
        device = next(self.model.parameters()).device
        block = {k: v.to(device) if torch.is_tensor(v) else v for k, v in block.items()}
        length = block["input_ids"].numel()
        if (
            self.positions.logical_token_count + length
            > self.cfg["runtime"]["max_logical_context_tokens"]
        ):
            raise ValueError("Logical serving context exhausted")
        positions = self.positions.append(
            self.model.backbone.model,
            block["input_ids"],
            block["mm_token_type_ids"],
            block.get("image_grid_thw"),
        )
        hidden = self.model.hidden(
            **block, position_ids=positions, cache=self.cache, stream_append=True
        )
        assert_state_dtypes(self.cache)
        if kv_length(self.cache) > self.cfg["memory"]["kv_guard_tokens"]:
            raise ValueError("Resident KV exceeds declared guard")
        return hidden

    def _begin_step(self, step_id, block):
        reserve = block["input_ids"].numel()
        if self.serializer.mode == "qwen_text":
            reserve += 16 + len(self.serializer.separator)
        elif self.serializer.mode == 'candidate_logits':
            reserve += max(len(self.serializer.feedback_ids(c)) for c in range(len(ACTIONS)))
        if (
            self.positions.logical_token_count + reserve
            > self.cfg["runtime"]["max_logical_context_tokens"]
        ):
            raise ValueError("Complete step cannot fit serving context")
        if self.cfg["memory"]["mode"] == "window8":
            while self.resident and self.resident[0][0] < step_id - 7:
                _, length = self.resident.popleft()
                evict_kv(self.cache, self.prefix_length, length)

    @torch.inference_mode()
    def observe(self, episode_uid, step_id, rgb):
        if not self.valid or self.busy or episode_uid != self.episode_uid:
            raise ValueError("Invalid, busy, or mismatched session")
        if self.last_result is not None and step_id == self.next_step - 1:
            return self.last_result
        if step_id != self.next_step:
            raise ValueError("Out-of-order observation")
        self.busy = True
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            block = self.serializer.encode_observation(rgb, step_id)
            self._begin_step(step_id, block)
            start = self.positions.logical_token_count
            hidden = self._append(block)
            extra = {}
            if self.serializer.mode == 'candidate_logits':
                logits=self.model.action_logits(hidden[0,-1])
                if logits.shape != (len(ACTIONS),) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid candidate logits')
                cls=int(logits.argmax())
                feedback=self.serializer.feedback_ids(cls)
                self._append(self.serializer.text_block(feedback))
                extra.update(generated_tokens=0, feedback_token_ids=feedback)
                if self.cfg['runtime'].get('action_diagnostics',False):
                    scores=logits.float()
                    top=scores.topk(2).values
                    extra.update(action_logits=scores.cpu().tolist(),
                        action_probabilities=scores.softmax(-1).cpu().tolist(),
                        candidate_token_ids=list(self.serializer.candidate_token_ids),
                        candidate_tokens=list(CANDIDATES), margin_top1_top2=float(top[0]-top[1]))
            elif self.serializer.mode == "classification":
                logits = self.model.classifier(hidden[0, -1])
                cls = int(logits.argmax())
                extra["logits"] = logits.float().cpu().tolist()
            else:
                response = []
                forbidden = set(self.serializer.tokenizer.all_special_ids) - {
                    self.serializer.eos
                }
                for _ in range(16):
                    token = int(self.model.backbone.lm_head(hidden[0, -1]).argmax())
                    if token == self.serializer.eos:
                        self._append(
                            self.serializer.text_block(
                                [token] + self.serializer.separator
                            )
                        )
                        text = self.serializer.tokenizer.decode(
                            response,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        cls = parse_action(text)
                        extra.update(
                            response_token_ids=response + [token],
                            response_text=text,
                            decode_tokens=len(response) + 1,
                        )
                        break
                    if token in forbidden:
                        raise ValueError(f"Forbidden structural token {token}")
                    response.append(token)
                    hidden = self._append(self.serializer.text_block([token]))
                else:
                    raise ValueError(
                        "Response limit reached without assistant terminator"
                    )
            if self.cfg["memory"]["mode"] == "window8":
                self.resident.append(
                    (step_id, self.positions.logical_token_count - start)
                )
            torch.cuda.synchronize()
            self.last_result = action_result(
                cls,
                episode_uid=episode_uid,
                step_id=step_id,
                model_seconds=time.perf_counter() - started,
                retained_kv_tokens=kv_length(self.cache),
                status="ok",
                **extra,
            )
            self.next_step += 1
            return self.last_result
        except Exception:
            self.valid = False
            raise
        finally:
            self.busy = False
