"""Versioned episode and navigation contracts (no model/CUDA imports)."""
from copy import deepcopy
from pathlib import Path

ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
HABITAT_IDS = (1, 2, 3, 0)
CANDIDATES = ("A", "B", "C", "D")
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERIALIZERS = {
    "classification": "vln_observation_stream_v3",
    "qwen_text": "vln_append_only_chat_v1",
    "candidate_logits": "vln_candidate_logits_v1",
}


def parse_action(text):
    text = text.strip()
    if text not in ACTIONS:
        raise ValueError(f"Invalid navigation response: {text!r}")
    return ACTIONS.index(text)


def action_result(class_id, **metadata):
    if class_id not in range(4):
        raise ValueError(f"Invalid class: {class_id}")
    return dict(
        class_id=class_id,
        action_name=ACTIONS[class_id],
        habitat_action_id=HABITAT_IDS[class_id],
        **metadata,
    )


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            deep_merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def validate_resume_contract(saved, navigation, data):
    if saved.get("navigation") != navigation:
        raise ValueError("Resume navigation/serializer/visibility contract differs")
    if saved.get("data") != data:
        raise ValueError("Resume manifest/episode order/sampler contract differs")


def select_episode_ids(episodes, requested):
    lookup = {ep["episode_uid"]: ep for ep in episodes}
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Episode selection must be nonempty and unique")
    missing = set(requested) - lookup.keys()
    if missing:
        raise ValueError(f"Requested episodes not in manifest: {sorted(missing)}")
    return [lookup[uid] for uid in requested]


def validate_episode(episode, root=None, check_images=True):
    required = (
        "episode_uid",
        "dataset",
        "dataset_version",
        "split",
        "scene_id",
        "episode_id",
        "instruction_id",
        "instruction",
    )
    for key in required:
        if not str(episode.get(key, "")).strip():
            raise ValueError(f'Missing {key}: {episode.get("episode_uid")}')
    if episode.get("observation_action_alignment") != "observation_before_action":
        raise ValueError("Episode must explicitly declare observation_before_action")
    steps = episode.get("steps", [])
    if not steps:
        raise ValueError("Empty episode")
    for index, step in enumerate(steps):
        if step["step_id"] != index or step.get("is_valid") is not True:
            raise ValueError(
                f'Invalid chronology/target: {episode["episode_uid"]}:{index}'
            )
        parse_action(step["action_name"])
        if (step["action_name"] == "STOP") != (index == len(steps) - 1):
            raise ValueError("Exactly one final STOP observation is required")
        path = Path(step["rgb_path"])
        if root is not None:
            path = Path(root) / path
        if check_images and not path.is_file():
            raise FileNotFoundError(path)


def validate_config(cfg, world_size=None):
    mode = cfg["model"]["output_mode"]
    memory = cfg["memory"]["mode"]
    if mode not in SERIALIZERS or memory not in ("full_context", "window8"):
        raise ValueError("Unsupported output or memory mode")
    obs, train = cfg["observations"], cfg["training"]
    if obs["serializer_version"] != SERIALIZERS[mode]:
        raise ValueError("Serializer/output conflict")
    if obs["append_action_tokens"] != (mode in ("qwen_text", "candidate_logits")):
        raise ValueError("Action-history/output conflict")
    if mode == 'candidate_logits':
        if cfg['model'].get('action_head_mode', 'lm_rows_trainable') not in ('lm_rows_trainable', 'lm_rows_frozen', 'copied_linear'):
            raise ValueError('Unknown candidate action head')
        if obs.get('feedback_format', 'candidate_token') not in ('candidate_token', 'canonical_action_text'):
            raise ValueError('Unknown candidate feedback format')
        if train.get('class_weighting', 'none') not in ('none', 'sqrt_inverse_frequency', 'effective_number'):
            raise ValueError('Unknown class weighting')
    if mode == "qwen_text" and not train.get("supervise_assistant_terminator"):
        raise ValueError("Text actions must supervise assistant terminator")
    if train["microbatch_episodes_per_rank"] != 1 or train["use_cache"]:
        raise ValueError("Training requires uncached complete episodes, batch one")
    if memory == "window8":
        if (
            cfg["runtime"]["text_attention"] != "simplememvln_step_attention"
            or cfg["memory"]["kv_window_steps_including_current"] != 8
            or not cfg["memory"]["evict_before_new_step"]
        ):
            raise ValueError("Window8 requires matching attention and FIFO policy")
    elif cfg["memory"]["evict_before_new_step"]:
        raise ValueError("FullContext cannot evict")
    if (
        world_size is not None
        and train["nominal_episodes_per_update"]
        != world_size * train["gradient_accumulation_steps"]
    ):
        raise ValueError("Global episode batch must equal world size × accumulation")
    return cfg
