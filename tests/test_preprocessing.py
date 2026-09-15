import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from configs.config import Config
from data.preprocessing import (
    _validate_files,
    _compute_fingerprint,
    process_ratings,
    compute_chronological_split,
    identify_cold_items,
    extract_metadata,
    EXPECTED_FILES,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def config(temp_dir):
    cfg = Config()
    cfg.raw_data_dir = os.path.join(temp_dir, "raw")
    cfg.processed_data_dir = os.path.join(temp_dir, "processed")
    os.makedirs(cfg.raw_data_dir, exist_ok=True)
    os.makedirs(cfg.processed_data_dir, exist_ok=True)
    return cfg


def _create_minimal_csvs(raw_dir: str):
    pd.DataFrame({
        "userId": [1], "movieId": [1], "rating": [5.0], "timestamp": [100],
    }).to_csv(os.path.join(raw_dir, "rating.csv"), index=False)
    pd.DataFrame({
        "movieId": [1], "title": ["Test"], "genres": ["Action"],
    }).to_csv(os.path.join(raw_dir, "movie.csv"), index=False)
    pd.DataFrame({
        "userId": [1], "movieId": [1], "tag": ["fun"], "timestamp": [100],
    }).to_csv(os.path.join(raw_dir, "tag.csv"), index=False)
    pd.DataFrame({
        "movieId": [1], "tagId": [1], "relevance": [0.9],
    }).to_csv(os.path.join(raw_dir, "genome_scores.csv"), index=False)
    pd.DataFrame({
        "tagId": [1], "tag": ["fun"],
    }).to_csv(os.path.join(raw_dir, "genome_tags.csv"), index=False)
    pd.DataFrame({
        "movieId": [1], "imdbId": [1], "tmdbId": [1],
    }).to_csv(os.path.join(raw_dir, "link.csv"), index=False)


class TestFileValidation:
    def test_missing_file_raises(self, config):
        with pytest.raises(FileNotFoundError, match="Missing required file"):
            _validate_files(config.raw_data_dir)

    def test_missing_column_raises(self, config):
        _create_minimal_csvs(config.raw_data_dir)
        pd.DataFrame({"userId": [1], "movieId": [1]}).to_csv(
            os.path.join(config.raw_data_dir, "rating.csv"), index=False
        )
        with pytest.raises(ValueError, match="missing required column"):
            _validate_files(config.raw_data_dir)

    def test_valid_files_pass(self, config):
        _create_minimal_csvs(config.raw_data_dir)
        _validate_files(config.raw_data_dir)

    def test_fingerprint_deterministic(self, config):
        _create_minimal_csvs(config.raw_data_dir)
        fp1 = _compute_fingerprint(config.raw_data_dir)
        fp2 = _compute_fingerprint(config.raw_data_dir)
        assert fp1 == fp2
        assert len(fp1) == 16


class TestRatingsPipeline:
    def _make_ratings(self, raw_dir: str, config: Config):
        users = []
        movies = []
        ratings = []
        timestamps = []
        for u in range(1, 11):
            for i in range(10):
                users.append(u)
                movies.append(100 + (u * 10 + i) % 30)
                ratings.append(4.0 + (i % 2) * 0.5)
                timestamps.append(1000 + u * 100 + i)

        df = pd.DataFrame({
            "userId": users,
            "movieId": movies,
            "rating": ratings,
            "timestamp": timestamps,
        })
        df.to_csv(os.path.join(raw_dir, "rating.csv"), index=False)

    def test_contiguous_item_ids(self, config):
        self._make_ratings(config.raw_data_dir, config)
        seqs, stats = process_ratings(config)

        all_items = set()
        for items in seqs.values():
            all_items.update(items)

        assert 0 not in all_items, "Padding ID 0 must not be a real item"
        assert min(all_items) == 1, "Item IDs must start at 1"
        assert max(all_items) == stats["num_items"], "Item IDs must be contiguous to num_items"
        assert all_items == set(range(1, stats["num_items"] + 1))

    def test_padding_and_mask_ids(self, config):
        self._make_ratings(config.raw_data_dir, config)
        _, stats = process_ratings(config)

        assert stats["PADDING_ID"] == 0
        assert stats["MASK_ID"] == stats["num_items"] + 1
        assert stats["VOCAB_SIZE"] == stats["num_items"] + 2

    def test_min_interactions_filter(self, config):
        config.min_interactions = 8
        users = []
        movies = []
        ratings = []
        timestamps = []
        for i in range(10):
            users.append(1)
            movies.append(100 + i)
            ratings.append(5.0)
            timestamps.append(1000 + i)
        for i in range(3):
            users.append(2)
            movies.append(200 + i)
            ratings.append(5.0)
            timestamps.append(2000 + i)
        df = pd.DataFrame({
            "userId": users,
            "movieId": movies,
            "rating": ratings,
            "timestamp": timestamps,
        })
        df.to_csv(os.path.join(config.raw_data_dir, "rating.csv"), index=False)
        seqs, stats = process_ratings(config)
        assert stats["num_users"] == 1
        assert 0 in seqs or len(seqs) == 1

    def test_chronological_order(self, config):
        self._make_ratings(config.raw_data_dir, config)
        seqs, _ = process_ratings(config)
        rating_df = pd.read_csv(os.path.join(config.raw_data_dir, "rating.csv"))
        rating_df = rating_df.sort_values(["userId", "timestamp"])

    def test_implicit_filter(self, config):
        config.implicit_threshold = 4.5
        users = []
        movies = []
        ratings = []
        timestamps = []
        for i in range(10):
            users.append(1)
            movies.append(100 + i)
            ratings.append(4.0 if i < 5 else 5.0)
            timestamps.append(1000 + i)
        df = pd.DataFrame({
            "userId": users,
            "movieId": movies,
            "rating": ratings,
            "timestamp": timestamps,
        })
        df.to_csv(os.path.join(config.raw_data_dir, "rating.csv"), index=False)
        seqs, stats = process_ratings(config)
        assert stats["num_items"] == 5

    def test_dataset_stats_saved(self, config):
        self._make_ratings(config.raw_data_dir, config)
        process_ratings(config)
        stats_path = os.path.join(config.processed_data_dir, "dataset_stats.json")
        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            stats = json.load(f)
        assert "num_items" in stats
        assert "PADDING_ID" in stats
        assert "MASK_ID" in stats
        assert "VOCAB_SIZE" in stats


class TestChronologicalSplit:
    def test_split_ratios(self):
        config = Config()
        config.train_ratio = 0.8
        config.val_ratio = 0.1
        seqs = {0: list(range(1, 21))}
        ts = {0: list(range(100, 120))}
        train, val, test, cutoff = compute_chronological_split(seqs, ts, config)
        assert len(train[0]) == 16
        assert len(val[0]) == 2
        assert len(test[0]) == 2

    def test_cutoff_is_max_train_timestamp(self):
        config = Config()
        seqs = {
            0: list(range(1, 11)),
            1: list(range(11, 21)),
        }
        ts = {
            0: list(range(100, 110)),
            1: list(range(200, 210)),
        }
        _, _, _, cutoff = compute_chronological_split(seqs, ts, config)
        assert cutoff == 207


class TestColdStartHoldout:
    """REV-011: the holdout protocol is config-driven.

    `timestamp_cutoff` keeps the original "first seen strictly after the training
    cutoff" semantics; `first_appearance_rank` (the new default) holds out the
    latest-appearing fraction of the catalogue and is non-empty by construction.
    Both branches are covered, plus the loud failure on an empty holdout.
    """

    USER_SEQUENCES = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    TS_PER_USER = {0: [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]}
    TRAIN = {0: [1, 2, 3, 4, 5, 6, 7, 8]}
    VAL = {0: [9]}
    TEST = {0: [10]}
    CUTOFF = 170.0

    def _run(self, strategy, ratio):
        config = Config()
        config.cold_start_strategy = strategy
        config.cold_start_ratio = ratio
        config.processed_data_dir = tempfile.mkdtemp()
        try:
            return identify_cold_items(
                self.TRAIN, self.VAL, self.TEST, self.TS_PER_USER,
                self.USER_SEQUENCES, self.CUTOFF, config,
            )
        finally:
            import shutil
            shutil.rmtree(config.processed_data_dir, ignore_errors=True)

    def _first_seen(self, item):
        return min(
            ts for uid, items in self.USER_SEQUENCES.items()
            for item2, ts in zip(items, self.TS_PER_USER[uid])
            if item2 == item
        )

    def _assert_invariants(self, cold, warm, filtered):
        train_items = set()
        for items in filtered.values():
            train_items.update(items)
        assert cold.isdisjoint(train_items), "cold items must be stripped from training"

        all_items = set()
        for items in self.USER_SEQUENCES.values():
            all_items.update(items)
        assert cold | warm == all_items
        assert cold.isdisjoint(warm)
        assert cold, "the holdout must be non-empty"

    def test_timestamp_cutoff_selects_post_cutoff_items(self):
        cold, warm, filtered = self._run("timestamp_cutoff", 1.0)
        for item in cold:
            first_ts = self._first_seen(item)
            assert first_ts > self.CUTOFF, (
                f"Cold item {item} first seen at {first_ts} <= cutoff {self.CUTOFF}"
            )
        assert cold == {9, 10}
        self._assert_invariants(cold, warm, filtered)

    def test_first_appearance_rank_selects_latest_items(self):
        cold, warm, filtered = self._run("first_appearance_rank", 0.3)
        # The three latest-appearing items in the catalogue.
        assert cold == {8, 9, 10}
        latest_warm = max(self._first_seen(i) for i in warm)
        assert min(self._first_seen(i) for i in cold) > latest_warm
        self._assert_invariants(cold, warm, filtered)

    def test_first_appearance_rank_is_non_empty_where_cutoff_is_not(self):
        """The defect REV-011 describes: on a catalogue whose items all predate the
        cutoff, `timestamp_cutoff` yields an empty holdout and must now RAISE rather
        than log-and-continue, while the default strategy still produces one."""
        config = Config()
        config.cold_start_strategy = "timestamp_cutoff"
        config.cold_start_ratio = 0.3
        config.processed_data_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(RuntimeError):
                identify_cold_items(
                    self.TRAIN, self.VAL, self.TEST, self.TS_PER_USER,
                    self.USER_SEQUENCES, 1_000.0, config,
                )
        finally:
            import shutil
            shutil.rmtree(config.processed_data_dir, ignore_errors=True)

        cold, _, _ = self._run("first_appearance_rank", 0.3)
        assert cold

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="cold_start_strategy"):
            self._run("not_a_strategy", 0.3)


class TestMetadataExtraction:
    def test_matrix_shape_and_padding_row(self, config):
        num_items = 5
        item_map = {100: 1, 200: 2, 300: 3, 400: 4, 500: 5}

        pd.DataFrame({
            "movieId": [100, 100, 200, 200, 300, 300, 400, 400, 500, 500],
            "tagId": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "relevance": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        }).to_csv(os.path.join(config.raw_data_dir, "genome_scores.csv"), index=False)

        pd.DataFrame({
            "movieId": [100, 200, 300, 400, 500],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Action|Drama", "Comedy", "Action", "(no genres listed)", "Drama|Comedy"],
        }).to_csv(os.path.join(config.raw_data_dir, "movie.csv"), index=False)

        meta = extract_metadata(config, item_map, num_items)

        # REV-072: the matrix spans the full embedding vocabulary -- pad=0,
        # items 1..num_items, MASK_ID=num_items+1 -- so a masked sequence can be
        # gathered without clamping MASK_ID onto the last real item's features.
        assert meta.shape[0] == num_items + 2
        assert np.all(meta[0] == 0), "Row 0 (padding) must be all zeros"
        assert np.all(meta[num_items + 1] == 0), "MASK_ID row must be all zeros"

        for remapped_id in range(1, num_items + 1):
            assert meta.shape[1] > 0

    def test_zero_metadata_items(self, config):
        num_items = 3
        item_map = {100: 1, 200: 2, 999: 3}

        pd.DataFrame({
            "movieId": [100, 200],
            "tagId": [1, 1],
            "relevance": [0.9, 0.8],
        }).to_csv(os.path.join(config.raw_data_dir, "genome_scores.csv"), index=False)

        pd.DataFrame({
            "movieId": [100, 200],
            "title": ["A", "B"],
            "genres": ["Action", "Drama"],
        }).to_csv(os.path.join(config.raw_data_dir, "movie.csv"), index=False)

        meta = extract_metadata(config, item_map, num_items)
        assert np.all(meta[3] == 0), "Item 999 (remapped=3) has no data, should be zeros"


PROCESSED_DIR = os.path.join("data", "processed")

#: The whole point of T9.1 -- a clean checkout must SKIP these, not error.
requires_processed_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, "dataset_stats.json")),
    reason="data/processed/ not built; run `python main.py --stage preprocess`",
)


@requires_processed_data
class TestEndToEndPreprocessing:
    """T9.1: Verify all output files from run_preprocessing on real data.

    REV-042: `EXPECTED_FILES` omitted `test_examples.pkl` and the candidate sets -- the
    two artifacts that were actually missing (REV-002, REV-041) -- and nothing asserted
    `cold_items.pkl` was non-empty, so REV-011's vacuous holdout passed unnoticed. The
    class also errored rather than skipped on a clean checkout.
    """

    EXPECTED_FILES = [
        "user_sequences.pkl",
        "item_mapping.pkl",
        "user_mapping.pkl",
        "reverse_item_mapping.pkl",
        "item_metadata.npy",
        "cold_items.pkl",
        "warm_items.pkl",
        "dataset_stats.json",
        "dataset_manifest.json",
        "train_tensorised.pt",
        "val_tensorised.pt",
        "test_tensorised.pt",
        "global_negative_pool.npy",
        "global_negative_pool.npy.meta.json",
        "train_user_history.pkl",
        "eval_user_history.pkl",
        "splits.pkl",
        # REV-002: consumed by cold_start.py, longtail.py, local_train.py, federated.py
        "train_examples.pkl",
        "val_examples.pkl",
        "test_examples.pkl",
        # REV-041: the fixed candidate sets every model must be scored against
        "val_candidates.npy",
        "val_candidates.npy.meta.json",
        "test_candidates.npy",
        "test_candidates.npy.meta.json",
        # REV-007: long-tail popularity buckets
        "popularity_buckets.pkl",
        "preprocessing_manifest.json",
    ]

    def test_all_output_files_present(self):
        missing = [
            f for f in self.EXPECTED_FILES
            if not os.path.exists(os.path.join(PROCESSED_DIR, f))
        ]
        assert not missing, f"Missing output files: {missing}"

    def test_cold_items_non_empty_and_disjoint_from_training(self):
        """REV-011 shipped a cold-start holdout of 0 items, which made every target
        classify as warm and the entire cold-start question vacuous. File existence
        could not detect that; this can."""
        with open(os.path.join(PROCESSED_DIR, "cold_items.pkl"), "rb") as f:
            cold_items = set(pickle.load(f))
        assert len(cold_items) > 0, (
            "cold_items.pkl is empty -- the cold-start holdout is vacuous and every "
            "target will classify as warm"
        )

        with open(os.path.join(PROCESSED_DIR, "train_user_history.pkl"), "rb") as f:
            train_user_history = pickle.load(f)
        train_interacted = set()
        for items in train_user_history.values():
            train_interacted.update(int(i) for i in items)

        leaked = cold_items & train_interacted
        assert not leaked, (
            f"{len(leaked)} cold items appear in training interactions "
            f"(e.g. {sorted(leaked)[:10]}) -- the holdout is not disjoint from training"
        )

    def test_candidate_sets_match_their_splits_and_fingerprints(self):
        """REV-041: the stored sets must be loadable, fingerprint-verified, and sized to
        their split -- otherwise cross-model comparability is incidental."""
        import torch

        from data.sequence_generator import load_eval_candidates

        for split in ("val", "test"):
            path = os.path.join(PROCESSED_DIR, f"{split}_candidates.npy")
            matrix = load_eval_candidates(path)  # raises on fingerprint mismatch

            data = torch.load(
                os.path.join(PROCESSED_DIR, f"{split}_tensorised.pt"),
                weights_only=True, mmap=True,
            )
            n_examples = data["sequences"].shape[0]
            assert matrix.shape[0] == n_examples, (
                f"{split}_candidates has {matrix.shape[0]} rows but the {split} split "
                f"has {n_examples} examples"
            )
            # column 0 is the target, the rest are negatives
            assert matrix.shape[1] > 1
            targets = data["targets"].numpy()
            assert (matrix[:, 0] == targets).all(), (
                f"{split}_candidates column 0 is not the target column"
            )

    def test_examples_pickles_align_with_tensorised_splits(self):
        """REV-002: these were never written at all, so every consumer crashed."""
        import torch

        for split in ("train", "val", "test"):
            with open(os.path.join(PROCESSED_DIR, f"{split}_examples.pkl"), "rb") as f:
                examples = pickle.load(f)
            for key in ("sequences", "targets", "user_ids"):
                assert key in examples, f"{split}_examples.pkl missing '{key}'"

            data = torch.load(
                os.path.join(PROCESSED_DIR, f"{split}_tensorised.pt"),
                weights_only=True, mmap=True,
            )
            assert len(examples["targets"]) == data["sequences"].shape[0], (
                f"{split}_examples.pkl has {len(examples['targets'])} rows but "
                f"{split}_tensorised.pt has {data['sequences'].shape[0]}"
            )

    def test_popularity_buckets_partition_the_catalogue(self):
        """REV-007: the 20/20/60 split, checked on the real catalogue."""
        with open(os.path.join(PROCESSED_DIR, "popularity_buckets.pkl"), "rb") as f:
            buckets = pickle.load(f)

        head, mid, tail = buckets["head"], buckets["mid"], buckets["tail"]
        assert head.isdisjoint(mid) and head.isdisjoint(tail) and mid.isdisjoint(tail)

        # The 20/20/60 split is over items that HAVE interactions; `tail` additionally
        # absorbs every zero-interaction item, so the full catalogue is the wrong
        # denominator.
        with open(os.path.join(PROCESSED_DIR, "train_user_history.pkl"), "rb") as f:
            train_user_history = pickle.load(f)
        interacted = set()
        for items in train_user_history.values():
            interacted.update(int(i) for i in items)
        n_ranked = len(interacted)
        assert n_ranked > 0

        assert abs(len(head) / n_ranked - 0.20) < 0.01, (
            f"head is {len(head) / n_ranked:.1%} of ranked items, expected ~20%"
        )
        assert abs(len(mid) / n_ranked - 0.20) < 0.01, (
            f"mid is {len(mid) / n_ranked:.1%} of ranked items, expected ~20%"
        )
        assert abs(len(tail & interacted) / n_ranked - 0.60) < 0.01, (
            f"tail is {len(tail & interacted) / n_ranked:.1%} of ranked items, "
            f"expected ~60%"
        )
        # every zero-interaction item lands in tail
        assert (tail - interacted), "expected some zero-interaction items in tail"
        assert head.isdisjoint(tail - interacted)

    def test_dataset_stats_valid(self):
        stats_path = os.path.join("data", "processed", "dataset_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["num_users"] > 0
        assert stats["num_items"] > 0
        assert stats["VOCAB_SIZE"] == stats["num_items"] + 2
        assert stats["PADDING_ID"] == 0
        assert stats["MASK_ID"] == stats["num_items"] + 1

    def test_tensorised_datasets_loadable(self):
        import torch
        processed_dir = os.path.join("data", "processed")
        for split in ["train", "val", "test"]:
            path = os.path.join(processed_dir, f"{split}_tensorised.pt")
            data = torch.load(path, weights_only=True)
            assert "sequences" in data
            assert "targets" in data
            assert data["sequences"].shape[0] > 0

    def test_negative_pool_valid(self):
        processed_dir = os.path.join("data", "processed")
        pool = np.load(os.path.join(processed_dir, "global_negative_pool.npy"))
        assert pool.shape[0] > 0
        assert pool.min() >= 1

    def test_user_sequences_consistent_with_stats(self):
        processed_dir = os.path.join("data", "processed")
        with open(os.path.join(processed_dir, "user_sequences.pkl"), "rb") as f:
            seqs = pickle.load(f)
        with open(os.path.join(processed_dir, "dataset_stats.json")) as f:
            stats = json.load(f)
        assert len(seqs) == stats["num_users"]


def _write_minimal_ratings(config):
    """A small rating.csv with enough per-user history to survive the filters."""
    os.makedirs(config.raw_data_dir, exist_ok=True)
    rows = []
    for uid in range(1, 4):
        for j in range(6):
            rows.append(
                {
                    "userId": uid,
                    "movieId": 100 + j,
                    "rating": 5.0,
                    "timestamp": f"2015-01-0{j + 1} 00:00:00",
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(config.raw_data_dir, "rating.csv"), index=False
    )


class TestSingleRatingPass:
    """REV-085: `rating.csv` must be read once, with narrow dtypes and parsed timestamps.

    It used to be read in full twice -- `process_ratings` and then
    `get_timestamps_per_user` repeating the identical read/filter/sort to recover a
    column the first pass had already loaded and thrown away. The `timestamp` column
    came back as `object` (Python strings) at 1.52 GB of a 2.00 GB frame.
    """

    def test_process_ratings_returns_aligned_timestamps(self, config):
        _write_minimal_ratings(config)

        seqs, stats, ts = process_ratings(config, return_timestamps=True)

        assert set(ts.keys()) == set(seqs.keys()), (
            "timestamps must cover exactly the users that produced sequences"
        )
        for uid, items in seqs.items():
            assert len(ts[uid]) == len(items), (
                f"user {uid}: {len(ts[uid])} timestamps for {len(items)} items"
            )
            assert ts[uid] == sorted(ts[uid]), "timestamps must be chronological"

    def test_default_call_shape_is_unchanged(self, config):
        _write_minimal_ratings(config)
        result = process_ratings(config)
        assert len(result) == 2, "the two-value return must stay the default"

    def test_timestamps_are_numeric_not_strings(self, config):
        _write_minimal_ratings(config)
        _, _, ts = process_ratings(config, return_timestamps=True)
        sample = next(iter(ts.values()))
        assert all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in sample), (
            "timestamps must be parsed to epoch numbers, not left as object strings"
        )

    def test_reader_is_called_once_per_preprocessing_pass(self, config, monkeypatch):
        """The defining property: one pass over the file, not two."""
        import data.preprocessing as prep

        _write_minimal_ratings(config)
        calls = {"n": 0}
        real = prep._read_ratings

        def counting(cfg):
            calls["n"] += 1
            return real(cfg)

        monkeypatch.setattr(prep, "_read_ratings", counting)
        prep.process_ratings(config, return_timestamps=True)
        assert calls["n"] == 1, (
            f"rating.csv was read {calls['n']} times; sequences and timestamps must "
            f"come from a single pass"
        )
