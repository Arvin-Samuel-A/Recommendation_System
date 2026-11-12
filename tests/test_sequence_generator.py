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
    chunk_user_history,
    chunk_training_histories,
    generate_sequences,
    tensorise_examples,
    build_dataloader,
    GlobalNegativePool,
    filter_negatives,
    load_tensorised,
    load_train_dataset,
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


class TestChunking:
    """LLM-style training rows: whole histories, a next-item target at every position."""

    def test_short_history_is_one_chunk_of_shifted_targets(self):
        assert chunk_user_history([1, 2, 3, 4, 5], 8) == [([1, 2, 3, 4], [2, 3, 4, 5])]

    def test_long_history_is_cut_from_the_end(self):
        chunks = chunk_user_history(list(range(1, 11)), 4)
        assert chunks == [
            ([1], [2]),
            ([2, 3, 4, 5], [3, 4, 5, 6]),
            ([6, 7, 8, 9], [7, 8, 9, 10]),
        ]

    def test_every_item_after_the_first_is_a_target_exactly_once(self):
        items = list(range(1, 38))
        for width in (1, 3, 5, 36, 50):
            targets = [t for _, tgts in chunk_user_history(items, width) for t in tgts]
            assert targets == items[1:], width

    def test_single_item_history_yields_nothing(self):
        assert chunk_user_history([7], 4) == []

    def test_rows_are_left_padded_and_aligned(self):
        out = chunk_training_histories({0: [1, 2, 3], 1: list(range(10, 16))}, 4)
        assert out["sequences"].tolist() == [
            [0, 0, 1, 2],
            [0, 0, 0, 10],
            [11, 12, 13, 14],
        ]
        assert out["targets"].tolist() == [
            [0, 0, 2, 3],
            [0, 0, 0, 11],
            [12, 13, 14, 15],
        ]
        assert out["user_ids"].tolist() == [0, 1, 1]
        assert out["lengths"].tolist() == [2, 1, 4]
        # A position is real exactly when it has a target.
        assert np.array_equal(out["sequences"] != 0, out["targets"] != 0)


class TestGenerateSequences:
    def test_separate_history_dicts(self):
        config = Config()
        config.max_seq_length = 10
        user_seqs = {0: list(range(1, 21))}
        train = {0: list(range(1, 19))}
        val = {0: [19]}
        test = {0: [20]}

        _, _, train_hist, eval_hist = generate_sequences(
            user_seqs, train, val, test, config
        )
        assert train_hist is not eval_hist
        assert train_hist[0] == list(range(1, 19))
        assert eval_hist[0] == list(range(1, 20))

    def test_leave_one_out_rows(self):
        """One val and one test row per user, each with its full preceding context."""
        config = Config()
        config.max_seq_length = 5
        user_seqs = {0: list(range(1, 11)), 1: [11, 12, 13]}
        train = {0: list(range(1, 9)), 1: [11]}
        val = {0: [9], 1: [12]}
        test = {0: [10], 1: [13]}

        val_ex, test_ex, _, _ = generate_sequences(user_seqs, train, val, test, config)
        assert val_ex["targets"].tolist() == [9, 12]
        assert test_ex["targets"].tolist() == [10, 13]
        assert val_ex["sequences"].tolist() == [[4, 5, 6, 7, 8], [0, 0, 0, 0, 11]]
        assert test_ex["sequences"].tolist() == [[5, 6, 7, 8, 9], [0, 0, 0, 11, 12]]
        assert val_ex["lengths"].tolist() == [5, 1]
        assert test_ex["lengths"].tolist() == [5, 2]


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

    def test_per_position_targets(self):
        n = 5
        S = 3
        ds = RecDataset(
            torch.zeros(n, S, dtype=torch.long),
            torch.ones(n, S, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.long),
        )
        item = ds[0]
        assert item["targets"].shape == (S,)


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


class TestEpochPermutationSampler:
    def test_one_draw_per_epoch_however_far_it_is_read(self):
        """A capped epoch must leave the generator where a full epoch would, so data
        order never depends on how far worker prefetch read."""
        from data.sequence_generator import EpochPermutationSampler

        partial, full = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
        it = iter(EpochPermutationSampler(10, partial))
        first = [next(it) for _ in range(3)]
        everything = list(EpochPermutationSampler(10, full))
        assert first == everything[:3]
        assert sorted(everything) == list(range(10))
        assert torch.equal(partial.get_state(), full.get_state())


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
        return RecDataset(sequences, targets, user_ids, lengths, matrix), matrix

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
            short_matrix,
        )
        assert ds.metadata_matrix.shape[0] == mask_id + 1
        assert torch.all(ds[0]["metadata"] == 0)


class TestStoredWidth:
    """specs/ablation-run T2.2 (AC-16, AC-19): stored at the widest grid length, narrowed
    exactly, and never silently narrower than requested."""

    STORED = 16
    MODEL = 8
    SPLITS = ("val", "test")

    @staticmethod
    def _pickle(processed, name):
        import pickle

        with open(os.path.join(processed, name), "rb") as f:
            return pickle.load(f)

    def test_every_split_is_stored_at_the_widest_grid_length(self, tiny_processed_dir):
        import json

        processed = tiny_processed_dir()
        with open(os.path.join(processed, "preprocessing_manifest.json")) as f:
            assert json.load(f)["stored_seq_length"] == self.STORED
        for split in self.SPLITS:
            data = torch.load(
                os.path.join(processed, f"{split}_tensorised.pt"), weights_only=True,
            )
            assert data["sequences"].shape[1] == self.STORED, split
            examples = self._pickle(processed, f"{split}_examples.pkl")
            assert examples["sequences"].shape[1] == self.STORED, split

    def test_narrowing_equals_generating_at_the_narrow_width(self, tiny_processed_dir):
        from data.sequence_generator import generate_sequences, load_examples
        from tests.conftest import tiny_config

        processed = tiny_processed_dir()
        splits = self._pickle(processed, "splits.pkl")
        user_sequences = self._pickle(processed, "user_sequences.pkl")
        cfg = tiny_config(processed, "/nonexistent-unused")
        generated = generate_sequences(
            user_sequences, splits["train"], splits["val"], splits["test"], cfg,
            width=self.MODEL,
        )
        for split, expected in zip(self.SPLITS, generated[:2]):
            narrowed = load_tensorised(
                os.path.join(processed, f"{split}_tensorised.pt"),
                max_seq_length=self.MODEL,
            )
            assert torch.equal(
                narrowed.sequences, torch.from_numpy(expected["sequences"]).long()
            ), split
            assert torch.equal(
                narrowed.lengths, torch.from_numpy(expected["lengths"]).long()
            ), split
            ex = load_examples(
                os.path.join(processed, f"{split}_examples.pkl"), self.MODEL,
            )
            assert np.array_equal(ex["sequences"], expected["sequences"]), split
            assert np.array_equal(ex["lengths"], expected["lengths"]), split
            assert np.array_equal(ex["targets"], expected["targets"]), split

    def test_training_split_is_chunked_at_the_reader_width(self, tiny_processed_dir):
        """The training split is never narrowed: each width gets its own chunks."""
        processed = tiny_processed_dir()
        history = self._pickle(processed, "train_user_history.pkl")
        for width in (4, self.MODEL, self.STORED):
            ds = load_train_dataset(processed, width)
            expected = chunk_training_histories(history, width)
            assert ds.sequences.shape[1] == width
            assert torch.equal(ds.sequences, torch.from_numpy(expected["sequences"]).long())
            assert torch.equal(ds.targets, torch.from_numpy(expected["targets"]).long())
            n_targets = sum(max(0, len(v) - 1) for v in history.values())
            assert int((ds.targets != 0).sum()) == n_targets

    def test_requesting_more_than_is_stored_raises(self, tiny_processed_dir):
        from data.sequence_generator import load_examples

        processed = tiny_processed_dir()
        for split in self.SPLITS:
            with pytest.raises(ValueError, match="--stage preprocess") as exc:
                load_tensorised(
                    os.path.join(processed, f"{split}_tensorised.pt"),
                    max_seq_length=self.STORED + 1,
                )
            assert "sequence_length_grid" in str(exc.value)
            with pytest.raises(ValueError, match="sequence_length_grid"):
                load_examples(
                    os.path.join(processed, f"{split}_examples.pkl"), self.STORED + 1,
                )

    def test_exact_stored_width_is_returned_unchanged(self, tiny_processed_dir):
        processed = tiny_processed_dir()
        path = os.path.join(processed, "val_tensorised.pt")
        raw = torch.load(path, weights_only=True)
        same = load_tensorised(path, max_seq_length=self.STORED)
        assert torch.equal(same.sequences, raw["sequences"])


def _tracked_source_files():
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True,
    ).stdout.split()
    return [p for p in out if not p.startswith("tests/") and os.path.exists(p)]


def _string_constants(node):
    import ast

    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def _call_name(call):
    import ast

    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def find_unnarrowed_readers(paths):
    """Every `load_tensorised(...)` without `max_seq_length=`, and every direct `open()`
    of an `*_examples.pkl` outside the writer -- specs/ablation-run D7 / AC-17."""
    import ast

    offenders = []
    for path in paths:
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "load_tensorised":
                if not any(kw.arg == "max_seq_length" for kw in node.keywords):
                    offenders.append(f"{path}:{node.lineno} load_tensorised without max_seq_length")
            elif name == "open" and path != os.path.join("data", "preprocessing.py"):
                args = list(node.args) + [kw.value for kw in node.keywords]
                if any(
                    c.endswith("_examples.pkl")
                    for arg in args for c in _string_constants(arg)
                ):
                    offenders.append(f"{path}:{node.lineno} opens *_examples.pkl directly")
    return offenders


class TestEveryReaderNarrows:
    """specs/ablation-run T2.3 / AC-17: no reader may see stored-width sequences."""

    def test_static_every_reader_goes_through_narrowing(self):
        offenders = find_unnarrowed_readers(_tracked_source_files())
        assert not offenders, "readers that do not narrow:\n" + "\n".join(offenders)

    def test_the_audit_catches_both_defect_shapes(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text(
            "import os\n"
            "ds = load_tensorised(os.path.join(p, 'val_tensorised.pt'), meta)\n"
            "with open(os.path.join(p, 'test_examples.pkl'), 'rb') as f: pass\n"
            "with open(os.path.join(p, f'{s}_examples.pkl'), 'rb') as f: pass\n"
            "ok = load_tensorised(path, meta, max_seq_length=8)\n"
        )
        assert len(find_unnarrowed_readers([str(bad)])) == 3


class TestNarrowedReadersRunEndToEnd:
    """specs/ablation-run T2.3 / AC-17 (runtime half): every downstream stage runs on data
    stored at width 16 with a model built at width 8."""

    def test_every_stage_runs_on_wider_stored_data(self, tiny_processed_dir, tmp_path):
        import argparse

        import main as cli
        from evaluation.cold_start import run_cold_start_evaluation
        from evaluation.longtail import run_longtail_evaluation
        from training.pretrain import build_model, load_checkpoint, run_pretrain
        from tests.conftest import tiny_config

        processed = tiny_processed_dir()
        stored = torch.load(
            os.path.join(processed, "test_tensorised.pt"), weights_only=True,
        )["sequences"].shape[1]
        cfg = tiny_config(
            processed, str(tmp_path), pretrain_epochs=1, max_steps_per_epoch=2,
        )
        assert stored == 16 and cfg.max_seq_length == 8

        run_pretrain(cfg, "t-small")
        args = argparse.Namespace(model="t-small")

        cli.run_evaluate_stage(cfg, args)
        assert os.path.exists(os.path.join(cfg.log_dir, "test_metrics_t_small.json"))
        assert cli.run_routing_diagnostics_stage(cfg, args)
        assert cli.run_gradient_diagnostics_stage(cfg, args)

        model, moe = build_model(cfg)
        load_checkpoint(os.path.join(cfg.model_checkpoint_dir(), "best.pt"), model, moe)
        assert run_cold_start_evaluation(model, moe, cfg, "t-small")
        assert run_longtail_evaluation(model, moe, cfg, "t-small")
