import hashlib
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

    def test_train_lengths_are_per_example(self):
        """REV-088: `lengths` must be each example's own non-pad count, not the user's."""
        config = Config()
        config.max_seq_length = 10
        config.multi_step_k = 0
        items = [1, 2, 3, 4, 5, 6]
        train_ex, _, _, _, _ = generate_sequences(
            {0: items}, {0: items}, {0: []}, {0: []}, config
        )
        assert len(train_ex["sequences"]) == len(items) - 1
        for i, seq in enumerate(train_ex["sequences"]):
            assert train_ex["lengths"][i] == int((seq != 0).sum()), (
                f"example {i}: lengths={train_ex['lengths'][i]} "
                f"but non-pad count is {int((seq != 0).sum())}"
            )

    def test_train_lengths_saturate_at_max_seq_length(self):
        """Contexts longer than S are truncated, so `lengths` caps at S."""
        config = Config()
        config.max_seq_length = 4
        config.multi_step_k = 0
        items = list(range(1, 11))
        train_ex, _, _, _, _ = generate_sequences(
            {0: items}, {0: items}, {0: []}, {0: []}, config
        )
        for i, seq in enumerate(train_ex["sequences"]):
            assert train_ex["lengths"][i] == int((seq != 0).sum())
        assert train_ex["lengths"].max() == config.max_seq_length

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

    def test_no_list_of_arrays_or_matrix_copy(self, tmp_path):
        """REV-087: rows go into one preallocated array, optionally an on-disk memmap.

        The old form returned N separate `(101,)` arrays (~0.9 GB of per-array overhead
        for the test split) which `save_eval_candidates` then copied into a contiguous
        matrix -- another 0.86 GB, at the same point in the pipeline as the other peaks.
        """
        import numpy as np
        from data.sequence_generator import save_eval_candidates, load_eval_candidates

        pool = np.arange(1, 201)
        examples = {"targets": [50, 75, 12], "user_ids": [0, 1, 2]}
        history = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7]}

        in_memory = generate_eval_candidates(examples, pool, history, 10, 0, 999)
        assert isinstance(in_memory, np.ndarray) and not isinstance(in_memory, list)
        assert in_memory.shape == (3, 11) and in_memory.dtype == np.int64

        path = str(tmp_path / "test_candidates.npy")
        on_disk = generate_eval_candidates(
            examples, pool, history, 10, 0, 999, out_path=path,
        )
        assert isinstance(on_disk, np.memmap), "out_path must allocate an on-disk memmap"
        assert np.array_equal(np.asarray(on_disk), in_memory)

        # The memmap is already backed by `path`, so saving must not rewrite it, and the
        # fingerprint contract must be byte-identical to the whole-matrix digest.
        fingerprint = save_eval_candidates(on_disk, path)
        assert fingerprint == hashlib.sha256(in_memory.tobytes()).hexdigest()[:16]
        assert np.array_equal(load_eval_candidates(path), in_memory)

        # REV-119: the memory-mapped read must be the SAME rows, verified the same way.
        # `run_local_personalization` addresses a few thousand rows of an 856 MB matrix,
        # so it reads this way; FR-1's "one candidate protocol" depends on the two paths
        # being indistinguishable in content.
        mapped = load_eval_candidates(path, mmap=True)
        assert isinstance(mapped, np.memmap), "mmap=True must not materialise the matrix"
        assert np.array_equal(np.asarray(mapped), in_memory)
        assert np.array_equal(mapped[1], in_memory[1]), "row slicing must agree"
        # Fingerprint verification must run -- and pass -- over the memmap too, otherwise
        # the mapped path would silently skip the integrity check the eager path enforces.
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            import json as _json
            meta_path = path + ".meta.json"
            with open(meta_path) as f:
                meta = _json.load(f)
            good = meta["fingerprint"]
            meta["fingerprint"] = "0" * 16
            with open(meta_path, "w") as f:
                _json.dump(meta, f)
            try:
                load_eval_candidates(path, mmap=True)
            finally:
                meta["fingerprint"] = good
                with open(meta_path, "w") as f:
                    _json.dump(meta, f)

    def test_block_fingerprint_matches_whole_matrix_digest(self):
        """REV-087: block-wise hashing must reproduce the digest byte-for-byte."""
        import numpy as np
        from data.sequence_generator import _fingerprint_rows

        matrix = np.arange(100 * 101, dtype=np.int64).reshape(100, 101)
        expected = hashlib.sha256(matrix.tobytes()).hexdigest()[:16]
        for block_rows in (1, 7, 100, 4096):
            assert _fingerprint_rows(matrix, block_rows=block_rows) == expected


class TestLazyMetadataGather:
    """REV-058: `RecDataset` must never materialise a dense `(N, S, F)` metadata block.

    The dense form is 60.8 GB for the MovieLens-20M train split against a 30 GB host
    budget, so no real-scale entry point could start. What the dataset holds is the
    `(vocab, F)` matrix; the per-example `(S, F)` block is gathered in `__getitem__` and
    collated by the DataLoader.
    """

    def _dataset(self, n=5000, s=50, f=39, vocab=200):
        sequences = torch.randint(0, vocab, (n, s), dtype=torch.long)
        targets = torch.randint(1, vocab, (n,), dtype=torch.long)
        user_ids = torch.zeros(n, dtype=torch.long)
        lengths = torch.full((n,), s, dtype=torch.long)
        matrix = np.random.rand(vocab, f).astype(np.float32)
        return RecDataset(sequences, targets, user_ids, lengths, None, matrix), matrix

    def test_resident_tensor_is_the_vocab_matrix_not_n_s_f(self):
        n, s, f, vocab = 5000, 50, 39, 200
        ds, matrix = self._dataset(n, s, f, vocab)

        assert ds.has_metadata
        assert tuple(ds.metadata_matrix.shape) == (vocab, f), (
            "the dataset must retain the (vocab, feature_dim) matrix"
        )
        # The dense block would be n*s*f elements; the matrix is vocab*f.
        assert ds.metadata_matrix.numel() == vocab * f
        assert ds.metadata_matrix.numel() < n * s * f

        # No attribute holds an (N, S, F) tensor.
        for name, value in vars(ds).items():
            if isinstance(value, torch.Tensor):
                assert value.dim() <= 2, (
                    f"RecDataset.{name} is {tuple(value.shape)}; a dense per-example "
                    f"metadata block is exactly what REV-058 forbids"
                )

    def test_getitem_gathers_the_correct_rows(self):
        ds, matrix = self._dataset(n=32, s=50, f=39, vocab=200)
        item = ds[7]
        assert tuple(item["metadata"].shape) == (50, 39)
        expected = torch.as_tensor(matrix[ds.sequences[7].numpy()], dtype=torch.float32)
        assert torch.allclose(item["metadata"], expected)

    def test_dataloader_collates_b_s_f(self):
        ds, _ = self._dataset(n=64, s=50, f=39, vocab=200)
        cfg = Config()
        cfg.batch_size = 8
        cfg.num_workers = 0
        cfg.pin_memory = False
        loader = build_dataloader(ds, cfg, shuffle=False)
        batch = next(iter(loader))
        assert tuple(batch["metadata"].shape) == (8, 50, 39), (
            "the DataLoader must collate the gathered rows into the (B, S, F) block "
            "train_one_step already expects from batch['metadata']"
        )

    def test_mask_id_row_gathers_zeros(self):
        """REV-072: MASK_ID has a real, all-zero row, so no clamp is needed."""
        vocab, f = 200, 39
        matrix = np.random.rand(vocab, f).astype(np.float32)
        matrix[vocab - 1] = 0.0  # MASK_ID row
        sequences = torch.full((4, 10), vocab - 1, dtype=torch.long)
        ds = RecDataset(
            sequences,
            torch.ones(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.full((4,), 10, dtype=torch.long),
            None,
            matrix,
        )
        assert torch.all(ds[0]["metadata"] == 0)

    def test_short_matrix_is_padded_without_load_metadata_matrix(self):
        """REV-090b: RecDataset aligns the matrix itself, not only via the loader.

        A pre-REV-072 `item_metadata.npy` has `num_items + 1` rows and therefore no
        MASK_ID row. Handing that array straight to RecDataset used to IndexError on the
        first masked sequence; the alignment must not depend on the caller.
        """
        num_items, f = 199, 39
        mask_id = num_items + 1
        short_matrix = np.random.rand(num_items + 1, f).astype(np.float32)  # no MASK row
        sequences = torch.full((4, 10), mask_id, dtype=torch.long)
        ds = RecDataset(
            sequences,
            torch.ones(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.full((4,), 10, dtype=torch.long),
            None,
            short_matrix,
        )
        assert ds.metadata_matrix.shape[0] == mask_id + 1
        assert torch.all(ds[0]["metadata"] == 0)
