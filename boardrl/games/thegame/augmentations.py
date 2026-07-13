from copy import copy
import random


def shuffle_hand(samples):
    """Return copies of samples with The Game's hand cards reordered."""

    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)

        lines = sample.state.splitlines(keepends=True)
        for position, line in enumerate(lines):
            if not line.startswith("Hand:"):
                continue

            content = line[len("Hand:") :]
            for ending in ("\r\n", "\n", "\r"):
                if content.endswith(ending):
                    content = content[: -len(ending)]
                    break
            else:
                ending = ""

            cards = content.split()
            if len(cards) > 1:
                random.shuffle(cards)
            lines[position] = "Hand: " + " ".join(cards) + ending
            break

        sample.state = "".join(lines)
    return augmented
