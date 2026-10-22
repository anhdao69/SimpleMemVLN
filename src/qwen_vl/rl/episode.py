"""Full-episode training tensors for sampled rollouts.

Identical layout to ``EpisodeSerializer.encode_episode`` for text policies, but
built from the in-memory frames the policy saw and the response tokens it
actually sampled (which need not be the canonical action tokenization, and whose
last response may be an invalid, unterminated format error). For episodic-memory
policies (M5) it also carries the same ``episodic_layout`` metadata as SFT, so the
training forward rebuilds the bank selection from the sampled token stream.
"""
import torch

from qwen_vl.data.episodic_metadata import build_episodic_layout
from qwen_vl.research.episodic_contract import parse_episodic_spec


def encode_rollout(serializer, instruction, frames, responses, model_max_length=None, episode_uid=None,
                   memory_banks=None):
    """frames[t]: uint8 HxWx3; responses[t]: dict(tokens=[...], complete=bool).

    ``memory_banks`` (MemFork): the realised frame banks per step (``FrameBank.banks``); the
    training forward then uses exactly that memory (``stream/retention.injected_plans``).
    """
    if serializer.mode != 'qwen_text':
        raise ValueError('Rollout encoding is defined for text-action policies only')
    if not frames or len(frames) != len(responses):
        raise ValueError('Every observation needs exactly one sampled response')
    blocks = [serializer.encode_prefix(instruction)]
    prefix = blocks[0]['input_ids'].numel()
    cursor = prefix
    spans, observation_spans, targets, action_index = [], [], [], []
    for t, (rgb, response) in enumerate(zip(frames, responses)):
        tokens = list(response['tokens'])
        if not tokens or len(tokens) > 16:
            raise ValueError('Responses must contain 1..16 sampled tokens')
        if response['complete'] != (tokens[-1] == serializer.eos):
            raise ValueError('Completed responses end with the assistant terminator')
        if not response['complete'] and t != len(frames) - 1:
            raise ValueError('Only the final response may be unterminated')
        block = serializer.encode_observation(rgb, t)
        blocks.append(block)
        start = cursor
        cursor += block['input_ids'].numel()
        observation_spans.append((start, cursor))
        targets.extend(range(cursor, cursor + len(tokens)))
        action_index.extend([t] * len(tokens))
        closure = serializer.separator if response['complete'] else []
        blocks.append(serializer.text_block(tokens + closure))
        cursor += len(tokens) + len(closure)
        spans.append((start, cursor))
    if model_max_length is not None and cursor > model_max_length:
        raise ValueError(f'{cursor} tokens exceeds episode admission cap')
    result = {
        key: torch.cat([b[key] for b in blocks if key in b],
                       dim=(1 if key in ('input_ids', 'mm_token_type_ids') else 0))
        for key in ('input_ids', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw')
    }
    result.update(step_plan=(prefix, tuple(spans)),
                  response_target_positions=torch.tensor(targets, dtype=torch.long),
                  response_action_index=torch.tensor(action_index, dtype=torch.long),
                  num_actions=len(frames))
    if parse_episodic_spec(serializer.config) is not None:
        # Metadata only, as in SFT: identical tokens, images and targets.
        result['episodic_layout'] = build_episodic_layout(
            result['input_ids'], serializer.image_id, prefix, tuple(observation_spans), tuple(spans))
        result['episode_uid'] = episode_uid
        if memory_banks is not None:
            from qwen_vl.stream.retention import injected_plans
            spec = parse_episodic_spec(serializer.config)
            result['episodic_plans'] = injected_plans(spec, result['episodic_layout'], memory_banks)
    elif memory_banks is not None:
        raise ValueError('Memory banks need an episodic_window_v1 policy')
    return result


def response_log_probs(model, batch, temperature, with_entropy=False, probe_token=None):
    """Per-token log pi_T(token) from the full-episode training forward.

    Matches verl's actor: logits are divided by the rollout temperature before
    the full-vocabulary log-softmax.
    """
    episodic = {}
    if 'episodic_layout' in batch:
        episodic = dict(episodic_layout=batch['episodic_layout'], episode_uid=batch['episode_uid'])
        if 'episodic_plans' in batch:
            episodic['episodic_plans'] = batch['episodic_plans']
    hidden = model.hidden(batch['input_ids'], batch['mm_token_type_ids'], batch['pixel_values'],
                          batch['image_grid_thw'], batch['step_plan'], **episodic)
    positions = batch['response_target_positions']
    logits = model.backbone.lm_head(hidden[0, positions - 1]).float() / temperature
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, batch['input_ids'][0, positions].unsqueeze(-1)).squeeze(-1)
    entropy = None
    if with_entropy:  # diagnostic only: keep it out of the autograd graph
        with torch.no_grad():
            entropy = -(logp_all.exp() * logp_all).sum(-1)
    if probe_token is not None:  # tempered log-prob of one token at every response position
        return logp, entropy, logp_all[:, probe_token]
    return logp, entropy
