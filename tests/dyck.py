import random
from typing import List, Tuple


# ----------------------------
# 1.  Core Dyck generator
# ----------------------------
def _random_balanced_string(max_len: int, max_depth: int) -> str:
    """
    Produce a Dyck word (balanced parentheses) with length ≤ max_len
    and nesting depth ≤ max_depth.  Algorithm: random walk with stack
    and hard constraints to guarantee eventual closure.
    """
    if max_len % 2 == 1:
        max_len -= 1  # must be even
    seq, depth = [], 0
    while len(seq) < max_len:
        open_possible = depth < max_depth and len(seq) < max_len - (depth + 1)
        close_possible = depth > 0
        if not open_possible and not close_possible:
            break  # no legal moves left

        # Biased coin: favour closing when deep to avoid overshooting max_len
        if open_possible and close_possible:
            p_open = 0.5 * (max_depth - depth) / max_depth
            step = "(" if random.random() < p_open else ")"
        elif open_possible:
            step = "("
        else:
            step = ")"

        seq.append(step)
        depth += 1 if step == "(" else -1

    # Close any remaining opens
    seq.extend([")"] * depth)
    return "".join(seq)


# ----------------------------
# 2.  Assemble (tokens, label)
# ----------------------------
def generate_dyck_example(
    max_len: int = 128,
    max_depth: int = 12,
    pad_to_len: bool = True,
) -> Tuple[str, int, int]:
    """
    Returns
    -------
    tokens : str
        ID-encoded sequence, with exactly one sentinel 'Q'.
        If pad_to_len is True, it is right-padded to `max_len+1`
        (Dyck string plus sentinel).
    label : int
        Stack depth immediately **before** 'Q'.
    """
    dyck = _random_balanced_string(max_len, max_depth)

    # Choose insertion point and compute depth label
    q_pos = random.randint(0, len(dyck))  # inclusive of ends
    depth = 0
    for tok in dyck[:q_pos]:
        depth += 1 if tok == "(" else -1

    # Build final token list
    tokens = dyck[:q_pos] + "Q" + dyck[q_pos:]
    if pad_to_len:
        pad_len = (max_len + 1) - len(tokens)
        tokens += " " * max(0, pad_len)

    return tokens, q_pos, depth
