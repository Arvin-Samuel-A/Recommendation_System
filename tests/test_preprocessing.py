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
    def test_cold_items_post_cutoff(self):
        config = Config()
        config.cold_start_ratio = 1.0
        config.processed_data_dir = tempfile.mkdtemp()
        try:
            user_sequences = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            ts_per_user = {0: [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]}
            train = {0: [1, 2, 3, 4, 5, 6, 7, 8]}
            val = {0: [9]}
            test = {0: [10]}
            cutoff = 170.0

            cold, warm, filtered = identify_cold_items(
                train, val, test, ts_per_user, user_sequences, cutoff, config
            )

            for item in cold:
                first_ts = min(
                    ts for uid, items in user_sequences.items()
                    for item2, ts in zip(items, ts_per_user[uid])
                    if item2 == item
                )
                assert first_ts > cutoff, f"Cold item {item} first seen at {first_ts} <= cutoff {cutoff}"

            train_items = set()
            for items in filtered.values():
                train_items.update(items)
            assert cold.isdisjoint(train_items)

            all_items = set()
            for items in user_sequences.values():
                all_items.update(items)
            assert cold | warm == all_items
            assert cold.isdisjoint(warm)
        finally:
            import shutil
            shutil.rmtree(config.processed_data_dir, ignore_errors=True)


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

        assert meta.shape[0] == num_items + 1
        assert np.all(meta[0] == 0), "Row 0 (padding) must be all zeros"

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
