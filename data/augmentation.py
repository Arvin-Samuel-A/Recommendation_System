import math
import random
from typing import List


def random_crop(seq: List[int], crop_ratio: float, rng: random.Random) -> List[int]:
    min_len = max(1, math.ceil(len(seq) * crop_ratio))
    crop_len = rng.randint(min_len, len(seq))
    start = rng.randint(0, len(seq) - crop_len)
    return seq[start : start + crop_len]


def random_mask(
    seq: List[int], mask_ratio: float, mask_token: int, rng: random.Random
) -> List[int]:
    n = len(seq)
    num_to_mask = max(1, min(n - 1, int(n * mask_ratio)))
    indices = list(range(n))
    rng.shuffle(indices)
    mask_indices = set(indices[:num_to_mask])
    return [mask_token if i in mask_indices else item for i, item in enumerate(seq)]


def subsequence_sample(seq: List[int], rng: random.Random) -> List[int]:
    length = rng.randint(1, len(seq) - 1) if len(seq) > 1 else 1
    return seq[:length]


def augment_sequence(
    seq: List[int],
    mask_token: int,
    rng: random.Random,
    crop_ratio: float = 0.6,
    mask_ratio: float = 0.2,
) -> List[int]:
    if len(seq) <= 1:
        return list(seq)

    choice = rng.randint(0, 2)
    if choice == 0:
        return random_crop(seq, crop_ratio, rng)
    elif choice == 1:
        return random_mask(seq, mask_ratio, mask_token, rng)
    else:
        return subsequence_sample(seq, rng)
