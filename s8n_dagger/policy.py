"""Transactional recovery adapter around the immutable evaluated StreamSession.

The frozen decoder runs verbatim. Its post-observation state is saved before
its first response token, so recovery can replace only the assistant response.
Model modules/weights are shared; all mutable session state is copied.
"""
from copy import deepcopy
import hashlib
import time

import numpy as np
import torch

from qwen_vl.contracts import ACTIONS, HABITAT_IDS, action_result
from qwen_vl.stream.cache import kv_length
from qwen_vl.stream.session import StreamSession


class RecoverySession(StreamSession):
    """Single-episode proposal/commit protocol with idempotent exact retries.

    ``propose(uid, step, rgb)`` leaves one pending learner response. ``commit``
    keeps its exact generated tokens if its action is selected, otherwise it
    restores post-observation state and appends the executed canonical action.
    ``force`` ingests an expert-only observation without running greedy decode.
    Reset is the only way to recover from a model/append/parse failure.
    """

    _STATE_FIELDS = (
        "cache", "positions", "resident", "episodic", "step_lane_cache",
        "prefix_length", "next_step", "last_result",
    )

    def __init__(self, model, serializer):
        if serializer.mode != "qwen_text":
            raise ValueError("Recovery adapter requires the S8N qwen_text serializer")
        super().__init__(model, serializer)
        self._clear_transactions()

    def _clear_transactions(self):
        self._pending_proposal = None
        self._last_transaction = None
        self._observation_snapshot = None
        self._capture_observation = False
        self._step_start = None

    @torch.inference_mode()
    def reset(self, episode_uid, instruction):
        if not isinstance(episode_uid, str) or not episode_uid:
            raise ValueError("Nonempty episode UID required")
        self._clear_transactions()
        super().reset(episode_uid, instruction)
        return {"status": "ok", "episode_uid": episode_uid}

    def observe(self, *args, **kwargs):
        raise ValueError("Use propose/commit or force for recovery collection")

    def _begin_step(self, step_id, block):
        super()._begin_step(step_id, block)
        self._step_start = self.positions.logical_token_count

    def _append_with_role(self, block, role, *, complete=False, kind=None):
        hidden = super()._append_with_role(block, role, complete=complete, kind=kind)
        if kind == "observation" and self._capture_observation:
            self._observation_snapshot = self._snapshot()
        return hidden

    def _snapshot(self):
        # Episodic scorers point directly at decoder layers. A plain deepcopy
        # would duplicate the backbone weights through these references.
        memo = {id(self.model): self.model, id(self.serializer): self.serializer}
        for obj in (*self.model.modules(), *self.model.parameters(), *self.model.buffers()):
            memo[id(obj)] = obj
        values = {name: getattr(self, name) for name in self._STATE_FIELDS}
        values["rope_deltas"] = self.model.backbone.model.rope_deltas
        return deepcopy(values, memo)

    def _restore(self, snapshot):
        for name in self._STATE_FIELDS:
            setattr(self, name, snapshot[name])
        self.model.backbone.model.rope_deltas = snapshot["rope_deltas"]

    @staticmethod
    def _rgb_digest(rgb):
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.shape != (480, 640, 3):
            raise ValueError("Expected uint8 RGB 640x480")
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    def _validate_session(self, uid, step):
        if not self.valid or self.busy or uid != self.episode_uid:
            raise ValueError("Invalid, busy, or mismatched session")
        if type(step) is not int or step < 0:
            raise ValueError("Step must be a nonnegative integer")

    @staticmethod
    def _validate_action(action):
        if type(action) is not int or action not in HABITAT_IDS:
            raise ValueError("Invalid Habitat action ID")
        return HABITAT_IDS.index(action)

    def _retry(self, operation, step, digest=None, action=None):
        previous = self._last_transaction
        if previous is None or previous["step"] != step:
            return None
        if previous["operation"] != operation:
            raise ValueError("Step already completed through another operation")
        if digest is not None and previous["digest"] != digest:
            raise ValueError("Retry observation changed")
        if action is not None and previous["action"] != action:
            raise ValueError("Retry executed action changed")
        return deepcopy(previous["result"])

    @torch.inference_mode()
    def propose(self, uid, step, rgb):
        self._validate_session(uid, step)
        digest = self._rgb_digest(rgb)
        pending = self._pending_proposal
        if pending is not None:
            if pending["step"] != step or pending["digest"] != digest:
                raise ValueError("Pending proposal requires its exact retry or commit")
            return deepcopy(pending["result"])
        if step != self.next_step:
            raise ValueError("Out-of-order proposal or already committed step")
        self._capture_observation = True
        try:
            result = super().observe(uid, step, rgb)
            if self._observation_snapshot is None:
                raise RuntimeError("Missing post-observation rollback state")
            self._pending_proposal = dict(step=step, digest=digest, result=deepcopy(result))
            return deepcopy(result)
        except Exception:
            self.valid = False
            self._observation_snapshot = None
            raise
        finally:
            self._capture_observation = False

    def _append_canonical(self, cls):
        response = list(self.serializer.action_ids[cls])
        if not response or response[-1] != self.serializer.eos:
            raise ValueError("Canonical response lacks assistant terminator")
        for token in response[:-1]:
            self._append_with_role(self.serializer.text_block([token]), 2, kind="response_token")
        self._append_with_role(
            self.serializer.text_block([response[-1]] + list(self.serializer.separator)),
            2, complete=True, kind="response_close",
        )
        return dict(response_token_ids=response, response_text=ACTIONS[cls], decode_tokens=0)

    def _close_group(self):
        if self.cfg["memory"]["mode"] == "window8":
            self.resident.append((self.next_step, self.positions.logical_token_count - self._step_start))
        if self.episodic is not None:
            self.episodic.close_group(self.positions.logical_token_count)

    def _record(self, operation, step, digest, action, result):
        self.last_result = deepcopy(result)
        self._last_transaction = dict(operation=operation, step=step, digest=digest,
                                      action=action, result=deepcopy(result))
        self._pending_proposal = None
        self._observation_snapshot = None
        return deepcopy(result)

    @torch.inference_mode()
    def commit(self, uid, step, habitat_action_id):
        self._validate_session(uid, step)
        cls = self._validate_action(habitat_action_id)
        if self._pending_proposal is None:
            retry = self._retry("commit", step, action=habitat_action_id)
            if retry is not None:
                return retry
            raise ValueError("Commit requires a pending proposal")
        pending = self._pending_proposal
        if step != pending["step"]:
            raise ValueError("Commit step differs from pending proposal")
        self.busy = True
        try:
            result = deepcopy(pending["result"])
            overridden = habitat_action_id != result["habitat_action_id"]
            if overridden:
                self._restore(self._observation_snapshot)
                extra = self._append_canonical(cls)
                self._close_group()
                self.next_step = step + 1
                # Original metadata belongs to the learner proposal, returned by
                # propose. Commit metadata describes only the executed response.
                result = action_result(cls, episode_uid=uid, step_id=step, status="ok",
                                       retained_kv_tokens=kv_length(self.cache), **extra)
            result.update(committed=True, overridden=overridden, response_source="canonical_override" if overridden else "learner")
            torch.cuda.synchronize()
            return self._record("commit", step, pending["digest"], habitat_action_id, result)
        except Exception:
            self.valid = False
            raise
        finally:
            self.busy = False

    @torch.inference_mode()
    def force(self, uid, step, rgb, habitat_action_id):
        self._validate_session(uid, step)
        cls = self._validate_action(habitat_action_id)
        digest = self._rgb_digest(rgb)
        if self._pending_proposal is not None:
            raise ValueError("Cannot force a step with a pending proposal")
        retry = self._retry("force", step, digest, habitat_action_id)
        if retry is not None:
            return retry
        if step != self.next_step:
            raise ValueError("Out-of-order forced observation")
        self.busy = True
        started = time.perf_counter()
        try:
            block = self.serializer.encode_observation(rgb, step)
            self._begin_step(step, block)
            if self.episodic is not None:
                image = block["input_ids"][0].eq(self.serializer.image_id).nonzero().reshape(-1)
                self.episodic.open_group(step, self._step_start, self._step_start + int(image[0]))
            self._append_with_role(block, 1, kind="observation")
            extra = self._append_canonical(cls)
            self._close_group()
            self.next_step = step + 1
            torch.cuda.synchronize()
            result = action_result(cls, episode_uid=uid, step_id=step, status="ok",
                                   retained_kv_tokens=kv_length(self.cache), model_seconds=time.perf_counter() - started,
                                   committed=True, overridden=False, response_source="expert", **extra)
            return self._record("force", step, digest, habitat_action_id, result)
        except Exception:
            self.valid = False
            raise
        finally:
            self.busy = False
