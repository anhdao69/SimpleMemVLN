"""Pinned candidate codec: validate actual boundaries, not just labels."""

from qwen_vl.contracts import CANDIDATES

DECISION_CUE = "Action:\n"
CANDIDATE_SYSTEM = (
    "You control a navigation agent. Follow the instruction using the current RGB view.\n"
    "Choose exactly one label:\nA = MOVE_FORWARD\nB = TURN_LEFT\nC = TURN_RIGHT\nD = STOP\n"
    "Reply with the label only. Previous actions may be recorded as labels or action names.\n"
    "Instruction: "
)


def validate_candidates(
    tokenizer, candidates=CANDIDATES, observation_text=DECISION_CUE
):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    if len(candidates) != 4 or len(set(candidates)) != 4:
        raise ValueError("Exactly four distinct candidate labels required")
    ids = []
    prefix = encode(observation_text)
    suffix = encode("<|im_end|>\n")
    for label in candidates:
        tokens = encode(label)
        if len(tokens) != 1 or tokens[0] in tokenizer.all_special_ids:
            raise ValueError(f"Candidate must be one nonspecial token: {label!r}")
        if tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != label:
            raise ValueError("Candidate does not round-trip")
        if (
            encode(observation_text + label + "<|im_end|>\n")
            != prefix + tokens + suffix
        ):
            raise ValueError("Candidate merges across the decision/feedback boundary")
        ids.append(tokens[0])
    if len(set(ids)) != 4:
        raise ValueError("Candidate IDs must be distinct")
    return ids
