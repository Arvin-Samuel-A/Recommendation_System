import math
import random

import pytest

from data.augmentation import (
    random_crop,
    random_mask,
    subsequence_sample,
    augment_sequence,
)


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def seq():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestRandomCrop:
    def test_returns_contiguous_subsequence(self, seq, rng):
        cropped = random_crop(seq, 0.6, rng)
        assert len(cropped) >= math.ceil(len(seq) * 0.6)
        start = seq.index(cropped[0])
        assert seq[start : start + len(cropped)] == cropped

    def test_min_length_respected(self, seq, rng):
        for _ in range(50):
            cropped = random_crop(seq, 0.8, rng)
            assert len(cropped) >= math.ceil(len(seq) * 0.8)

    def test_never_empty(self, rng):
        result = random_crop([5], 0.5, rng)
        assert len(result) >= 1


class TestRandomMask:
    def test_uses_mask_token(self, seq, rng):
        mask_token = 999
        masked = random_mask(seq, 0.2, mask_token, rng)
        assert mask_token in masked

    def test_mask_token_not_num_items(self, seq, rng):
        num_items = 10
        mask_token = num_items + 1
        masked = random_mask(seq, 0.2, mask_token, rng)
        mask_count = sum(1 for x in masked if x == mask_token)
        assert mask_count >= 1
        assert mask_token == num_items + 1

    def test_at_least_one_unmasked(self, seq, rng):
        mask_token = 999
        for _ in range(100):
            masked = random_mask(seq, 0.5, mask_token, rng)
            unmasked = [x for x in masked if x != mask_token]
            assert len(unmasked) >= 1

    def test_same_length(self, seq, rng):
        masked = random_mask(seq, 0.3, 999, rng)
        assert len(masked) == len(seq)


class TestSubsequenceSample:
    def test_shorter_than_original(self, seq, rng):
        for _ in range(50):
            sub = subsequence_sample(seq, rng)
            assert len(sub) < len(seq)
            assert len(sub) >= 1

    def test_is_prefix(self, seq, rng):
        sub = subsequence_sample(seq, rng)
        assert seq[: len(sub)] == sub


class TestAugmentSequence:
    def test_all_strategies_hit(self, seq):
        mask_token = 11
        counts = {0: 0, 1: 0, 2: 0}
        for seed in range(300):
            rng = random.Random(seed)
            result = augment_sequence(seq, mask_token, rng)
            if mask_token in result:
                counts[1] += 1
            elif len(result) < len(seq) and result == seq[: len(result)]:
                counts[2] += 1
            else:
                counts[0] += 1
        assert all(c > 0 for c in counts.values()), f"Not all strategies hit: {counts}"

    def test_single_item_sequence(self):
        rng = random.Random(42)
        result = augment_sequence([5], 999, rng)
        assert result == [5]
