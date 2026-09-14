import os
import tempfile

import numpy as np
import pytest
import torch

from configs.config import Config
from data.sequence_generator import (
    RecDataset,
    _left_pad,
    generate_sliding_window_examples,
    generate_sequences,
    tensorise_examples,
    build_dataloader,
    GlobalNegativePool,
    filter_negatives,
    generate_eval_candidates,
)


class TestLeftPad:
    def test_short_sequence(self):
        result = _left_pad([1, 2, 3], 5)
        assert result == [0, 0, 1, 2, 3]

    def test_exact_length(self):
        result = _left_pad([1, 2, 3], 3)
        assert result == [1, 2, 3]

    def test_long_sequence_truncated(self):
        result = _left_pad([1, 2, 3, 4, 5], 3)
        assert result == [3, 4, 5]


class TestSlidingWindow:
    def test_basic_examples(self):
        items = [1, 2, 3, 4, 5]
        seqs, targets, _ = generate_sliding_window_examples(items, max_seq_length=5)
        assert len(seqs) == 4
        assert targets == [2, 3, 4, 5]

    def test_left_padding_alignment(self):
        items = [1, 2, 3, 4, 5]
        seqs, _, _ = generate_sliding_window_examples(items, max_seq_length=5)
        assert seqs[0] == [0, 0, 0, 0, 1]
        assert seqs[0][-1] == 1

    def test_multi_step_no_cross_split(self):
        items = [1, 2, 3, 4, 5]
        split_end = 4
        _, _, targets_k = generate_sliding_window_examples(
            items, max_seq_length=5, multi_step_k=2, split_end_idx=split_end
        )
        for i, tk in enumerate(targets_k):
            for t in tk:
                if t != 0:
                    assert t in items


class TestGenerateSequences:
    def test_separate_history_dicts(self):
        config = Config()
        config.max_seq_length = 10
        config.multi_step_k = 2
        user_seqs = {0: list(range(1, 21))}
        train = {0: list(range(1, 17))}
        val = {0: [17, 18]}
        test = {0: [19, 20]}

        _, _, _, train_hist, eval_hist = generate_sequences(
            user_seqs, train, val, test, config
        )
        assert train_hist is not eval_hist
        assert train_hist[0] != eval_hist[0]
        assert len(eval_hist[0]) > len(train_hist[0])

    def test_split_before_truncate(self):
        config = Config()
        config.max_seq_length = 5
        config.multi_step_k = 0
        items = list(range(1, 21))
        user_seqs = {0: items}
        train = {0: items[:16]}
        val = {0: items[16:18]}
        test = {0: items[18:]}

        train_ex, val_ex, test_ex, _, _ = generate_sequences(
            user_seqs, train, val, test, config
        )
        for seq in train_ex["sequences"]:
            assert len(seq) == config.max_seq_length

    def test_generates_examples(self):
        config = Config()
        config.max_seq_length = 10
        config.multi_step_k = 2
        user_seqs = {0: list(range(1, 11)), 1: list(range(11, 21))}
        train = {0: list(range(1, 9)), 1: list(range(11, 19))}
        val = {0: [9], 1: [19]}
        test = {0: [10], 1: [20]}

        train_ex, val_ex, test_ex, _, _ = generate_sequences(
            user_seqs, train, val, test, config
        )
        assert len(train_ex["sequences"]) > 0
        assert len(val_ex["sequences"]) > 0
        assert len(test_ex["sequences"]) > 0


class TestRecDataset:
    def test_getitem_returns_tensors(self):
        n = 10
        S = 5
        seqs = torch.randint(0, 100, (n, S))
        targets = torch.randint(1, 100, (n,))
        user_ids = torch.arange(n)
        lengths = torch.full((n,), S)
        ds = RecDataset(seqs, targets, user_ids, lengths)

        item = ds[0]
        assert isinstance(item["sequences"], torch.Tensor)
        assert isinstance(item["targets"], torch.Tensor)
        assert len(ds) == n

    def test_optional_fields(self):
        n = 5
        S = 3
        ds = RecDataset(
            torch.zeros(n, S, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            targets_k=torch.zeros(n, 2, dtype=torch.long),
        )
        item = ds[0]
        assert "targets_k" in item


class TestDataLoader:
    def test_construction(self):
        config = Config()
        config.batch_size = 4
        config.num_workers = 0
        ds = RecDataset(
            torch.zeros(10, 5, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
        )
        loader = build_dataloader(ds, config, shuffle=False)
        batch = next(iter(loader))
        assert batch["sequences"].shape[0] == 4


class TestGlobalNegativePool:
    def test_warm_only(self):
        warm = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        pool_obj = GlobalNegativePool(warm, seed=42)
        pool = pool_obj.build_once(pool_size=20)
        for item in pool:
            assert int(item) in warm

    def test_deterministic(self):
        warm = set(range(1, 101))
        p1 = GlobalNegativePool(warm, seed=42)
        p1.build_once(50)
        p2 = GlobalNegativePool(warm, seed=42)
        p2.build_once(50)
        assert np.array_equal(p1.pool, p2.pool)
        assert p1.fingerprint == p2.fingerprint

    def test_pad_mask_excluded(self):
        warm = {1, 2, 3, 4, 5}
        pool_obj = GlobalNegativePool(warm, seed=42)
        pool = pool_obj.build_once(10)
        assert 0 not in pool

    def test_save_load_roundtrip(self):
        warm = set(range(1, 51))
        pool_obj = GlobalNegativePool(warm, seed=42)
        pool_obj.build_once(30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pool.npy")
            pool_obj.save(path)
            loaded = GlobalNegativePool.load(path, warm)
            assert np.array_equal(pool_obj.pool, loaded.pool)
            assert pool_obj.fingerprint == loaded.fingerprint


class TestFilterNegatives:
    def test_excludes_target_and_history(self):
        pool = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        negs = filter_negatives(pool, target=5, history_set={1, 2}, num_negatives=5)
        assert 5 not in negs
        assert 1 not in negs
        assert 2 not in negs
        assert len(negs) == 5

    def test_excludes_pad_and_mask(self):
        pool = np.array([0, 1, 2, 3, 11, 4, 5])
        negs = filter_negatives(pool, target=1, history_set=set(), num_negatives=3, padding_id=0, mask_id=11)
        assert 0 not in negs
        assert 11 not in negs
        assert 1 not in negs

    def test_exhaustion_raises(self):
        pool = np.array([1, 2, 3])
        with pytest.raises(RuntimeError, match="Negative pool exhausted"):
            filter_negatives(pool, target=1, history_set={2, 3}, num_negatives=5)


class TestEvalCandidates:
    def test_target_always_first(self):
        pool = np.arange(1, 101)
        examples = {
            "targets": [50, 75],
            "user_ids": [0, 1],
            "sequences": [[1, 2, 3], [4, 5, 6]],
        }
        history = {0: [1, 2, 3], 1: [4, 5, 6]}
        candidates = generate_eval_candidates(
            examples, pool, history, num_negatives=10, padding_id=0, mask_id=101
        )
        assert len(candidates) == 2
        assert candidates[0][0] == 50
        assert candidates[1][0] == 75
        assert len(candidates[0]) == 11

    def test_identical_across_configs(self):
        pool = np.arange(1, 101)
        examples = {"targets": [10], "user_ids": [0], "sequences": [[1, 2]]}
        history = {0: [1, 2]}
        c1 = generate_eval_candidates(examples, pool, history, 10, 0, 101)
        c2 = generate_eval_candidates(examples, pool, history, 10, 0, 101)
        assert np.array_equal(c1[0], c2[0])
