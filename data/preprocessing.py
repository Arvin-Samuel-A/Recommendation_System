import hashlib
import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from configs.config import Config

logger = logging.getLogger(__name__)

EXPECTED_FILES = {
    "rating.csv": {
        "columns": ["userId", "movieId", "rating", "timestamp"],
        "min_rows": 19_000_000,
    },
    "movie.csv": {
        "columns": ["movieId", "title", "genres"],
        "min_rows": 25_000,
    },
    "tag.csv": {
        "columns": ["userId", "movieId", "tag", "timestamp"],
        "min_rows": 400_000,
    },
    "genome_scores.csv": {
        "columns": ["movieId", "tagId", "relevance"],
        "min_rows": 11_000_000,
    },
    "genome_tags.csv": {
        "columns": ["tagId", "tag"],
        "min_rows": 1_000,
    },
    "link.csv": {
        "columns": ["movieId", "imdbId", "tmdbId"],
        "min_rows": 25_000,
    },
}


def _compute_fingerprint(data_dir: str) -> str:
    h = hashlib.sha256()
    for fname in sorted(EXPECTED_FILES.keys()):
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            h.update(fname.encode())
            h.update(str(os.path.getsize(fpath)).encode())
    return h.hexdigest()[:16]


def _validate_files(data_dir: str) -> None:
    for fname, spec in EXPECTED_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required file: {fpath}")
        df = pd.read_csv(fpath, nrows=5)
        for col in spec["columns"]:
            if col not in df.columns:
                raise ValueError(
                    f"File '{fname}' missing required column '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )


def _validate_row_counts(data_dir: str) -> Dict[str, int]:
    counts = {}
    for fname, spec in EXPECTED_FILES.items():
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            n = sum(1 for _ in f) - 1
        counts[fname] = n
        if n < spec["min_rows"]:
            logger.warning(
                "%s has %d rows (expected >= %d)", fname, n, spec["min_rows"]
            )
    return counts


def acquire_dataset(config: Config) -> str:
    raw_dir = config.raw_data_dir
    os.makedirs(raw_dir, exist_ok=True)

    all_present = all(
        os.path.exists(os.path.join(raw_dir, f)) for f in EXPECTED_FILES
    )

    if all_present and not config.force_download:
        logger.info("All raw files already present in %s, skipping download", raw_dir)
        _validate_files(raw_dir)
        fingerprint = _compute_fingerprint(raw_dir)
        logger.info("Dataset fingerprint: %s", fingerprint)
        return fingerprint

    if not config.auto_download_dataset:
        raise RuntimeError(
            f"Raw data files missing from {raw_dir} and auto_download_dataset=False. "
            "Set auto_download_dataset=True or download manually."
        )

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        )

    logger.info("Downloading dataset '%s' via Kaggle API...", config.dataset)
    try:
        dataset_path = kagglehub.dataset_download(config.dataset)
    except Exception as e:
        raise RuntimeError(
            f"Kaggle dataset download failed: {e}\n"
            "Ensure your Kaggle credentials are configured:\n"
            "  1. Create ~/.kaggle/kaggle.json with your API token\n"
            "  2. chmod 600 ~/.kaggle/kaggle.json\n"
            "  See https://www.kaggle.com/docs/api"
        ) from e

    logger.info("Downloaded to: %s", dataset_path)
    dataset_path = str(dataset_path)

    for fname in EXPECTED_FILES:
        found = False
        for root, _, files in os.walk(dataset_path):
            if fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(raw_dir, fname)
                shutil.copy2(src, dst)
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find '{fname}' in downloaded dataset at {dataset_path}"
            )

    _validate_files(raw_dir)
    row_counts = _validate_row_counts(raw_dir)
    fingerprint = _compute_fingerprint(raw_dir)

    manifest = {
        "dataset": config.dataset,
        "fingerprint": fingerprint,
        "files": {f: {"rows": row_counts.get(f, 0)} for f in EXPECTED_FILES},
    }
    os.makedirs(config.processed_data_dir, exist_ok=True)
    manifest_path = os.path.join(config.processed_data_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Dataset fingerprint: %s", fingerprint)
    return fingerprint


def process_ratings(config: Config) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    rating_path = os.path.join(config.raw_data_dir, "rating.csv")
    if not os.path.exists(rating_path):
        raise FileNotFoundError(f"rating.csv not found at {rating_path}")

    logger.info("Loading rating.csv...")
    df = pd.read_csv(rating_path)

    required_cols = ["userId", "movieId", "rating", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"rating.csv missing column '{col}'. Found: {list(df.columns)}"
            )

    original_count = len(df)
    df = df[df["rating"] >= config.implicit_threshold].copy()
    logger.info(
        "Implicit filter (>= %.1f): %d → %d rows",
        config.implicit_threshold, original_count, len(df),
    )

    user_counts = df.groupby("userId").size()
    valid_users = user_counts[user_counts >= config.min_interactions].index
    df = df[df["userId"].isin(valid_users)].copy()
    logger.info(
        "Min interactions filter (>= %d): %d valid users, %d rows",
        config.min_interactions, len(valid_users), len(df),
    )

    df = df.sort_values(["userId", "timestamp"])

    user_ids_orig = sorted(df["userId"].unique())
    user_map = {orig: idx for idx, orig in enumerate(user_ids_orig)}

    item_ids_orig = sorted(df["movieId"].unique())
    item_map = {orig: idx + 1 for idx, orig in enumerate(item_ids_orig)}

    df["user_id"] = df["userId"].map(user_map)
    df["item_id"] = df["movieId"].map(item_map)

    user_sequences: Dict[int, List[int]] = {}
    skipped_users = 0
    for user_id, group in df.groupby("user_id"):
        items = group["item_id"].tolist()
        if len(items) < 3:
            skipped_users += 1
            continue
        user_sequences[user_id] = items

    if skipped_users > 0:
        logger.info("Skipped %d users with < 3 items after filtering", skipped_users)

    num_items = len(item_ids_orig)
    stats = {
        "num_users": len(user_sequences),
        "num_items": num_items,
        "PADDING_ID": 0,
        "MASK_ID": num_items + 1,
        "VOCAB_SIZE": num_items + 2,
        "total_interactions": sum(len(v) for v in user_sequences.values()),
        "skipped_users": skipped_users,
    }

    os.makedirs(config.processed_data_dir, exist_ok=True)

    stats_path = os.path.join(config.processed_data_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    seq_path = os.path.join(config.processed_data_dir, "user_sequences.pkl")
    with open(seq_path, "wb") as f:
        pickle.dump(user_sequences, f)

    user_map_path = os.path.join(config.processed_data_dir, "user_mapping.pkl")
    with open(user_map_path, "wb") as f:
        pickle.dump(user_map, f)

    item_map_path = os.path.join(config.processed_data_dir, "item_mapping.pkl")
    with open(item_map_path, "wb") as f:
        pickle.dump(item_map, f)

    reverse_item_map = {v: k for k, v in item_map.items()}
    rev_path = os.path.join(config.processed_data_dir, "reverse_item_mapping.pkl")
    with open(rev_path, "wb") as f:
        pickle.dump(reverse_item_map, f)

    logger.info(
        "Ratings processed: %d users, %d items, vocab_size=%d",
        stats["num_users"], stats["num_items"], stats["VOCAB_SIZE"],
    )
    return user_sequences, stats


def compute_chronological_split(
    user_sequences: Dict[int, List[int]],
    timestamps_per_user: Dict[int, List[float]],
    config: Config,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], float]:
    train_seqs: Dict[int, List[int]] = {}
    val_seqs: Dict[int, List[int]] = {}
    test_seqs: Dict[int, List[int]] = {}
    all_train_timestamps: List[float] = []

    for user_id, items in user_sequences.items():
        ts = timestamps_per_user[user_id]
        n = len(items)
        train_end = max(1, int(n * config.train_ratio))
        val_end = max(train_end + 1, int(n * (config.train_ratio + config.val_ratio)))
        val_end = min(val_end, n)

        train_items = items[:train_end]
        val_items = items[train_end:val_end]
        test_items = items[val_end:]

        train_seqs[user_id] = train_items
        if val_items:
            val_seqs[user_id] = val_items
        if test_items:
            test_seqs[user_id] = test_items

        all_train_timestamps.extend(ts[:train_end])

    global_training_cutoff = max(all_train_timestamps)
    logger.info("Global training cutoff timestamp: %.0f", global_training_cutoff)
    return train_seqs, val_seqs, test_seqs, global_training_cutoff


def identify_cold_items(
    train_seqs: Dict[int, List[int]],
    val_seqs: Dict[int, List[int]],
    test_seqs: Dict[int, List[int]],
    timestamps_per_user: Dict[int, List[float]],
    user_sequences: Dict[int, List[int]],
    global_training_cutoff: float,
    config: Config,
) -> Tuple[Set[int], Set[int], Dict[int, List[int]]]:
    item_first_seen: Dict[int, float] = {}
    item_first_seen_orig_id: Dict[int, int] = {}

    for user_id, items in user_sequences.items():
        ts = timestamps_per_user[user_id]
        for item, t in zip(items, ts):
            if item not in item_first_seen or t < item_first_seen[item]:
                item_first_seen[item] = t

    all_items: Set[int] = set()
    for items in user_sequences.values():
        all_items.update(items)

    cold_candidates = []
    for item in all_items:
        if item_first_seen.get(item, 0) > global_training_cutoff:
            cold_candidates.append((item_first_seen[item], item))

    cold_candidates.sort()

    max_cold = max(1, int(len(cold_candidates) * config.cold_start_ratio))
    if len(cold_candidates) <= max_cold:
        selected_cold = [item for _, item in cold_candidates]
        logger.info(
            "Cold-start: only %d candidates available (requested ratio would give %d). Using all.",
            len(cold_candidates), max_cold,
        )
    else:
        selected_cold = [item for _, item in cold_candidates[:max_cold]]

    cold_item_set = set(selected_cold)
    warm_item_set = all_items - cold_item_set

    filtered_train: Dict[int, List[int]] = {}
    for user_id, items in train_seqs.items():
        filtered = [i for i in items if i not in cold_item_set]
        if filtered:
            filtered_train[user_id] = filtered

    train_interacted_items: Set[int] = set()
    for items in filtered_train.values():
        train_interacted_items.update(items)

    assert cold_item_set.isdisjoint(train_interacted_items), (
        "Cold items found in filtered training data!"
    )
    assert cold_item_set | warm_item_set == all_items, (
        "Cold + warm items don't equal all items"
    )
    assert cold_item_set.isdisjoint(warm_item_set), (
        "Cold and warm items overlap"
    )

    os.makedirs(config.processed_data_dir, exist_ok=True)
    cold_path = os.path.join(config.processed_data_dir, "cold_items.pkl")
    with open(cold_path, "wb") as f:
        pickle.dump(cold_item_set, f)
    warm_path = os.path.join(config.processed_data_dir, "warm_items.pkl")
    with open(warm_path, "wb") as f:
        pickle.dump(warm_item_set, f)

    logger.info(
        "Cold-start holdout: %d cold items, %d warm items, %d total",
        len(cold_item_set), len(warm_item_set), len(all_items),
    )
    return cold_item_set, warm_item_set, filtered_train


def extract_metadata(
    config: Config,
    item_map: Dict[int, int],
    num_items: int,
) -> np.ndarray:
    genome_path = os.path.join(config.raw_data_dir, "genome_scores.csv")
    movie_path = os.path.join(config.raw_data_dir, "movie.csv")

    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"genome_scores.csv not found at {genome_path}")
    if not os.path.exists(movie_path):
        raise FileNotFoundError(f"movie.csv not found at {movie_path}")

    logger.info("Loading genome scores...")
    genome_df = pd.read_csv(genome_path)
    num_genome_tags = genome_df["tagId"].nunique()
    top_k = config.genome_top_tags

    logger.info("Loading movies for genre extraction...")
    movie_df = pd.read_csv(movie_path)

    all_genres: Set[str] = set()
    for genres_str in movie_df["genres"].dropna():
        if genres_str != "(no genres listed)":
            for g in genres_str.split("|"):
                all_genres.add(g.strip())
    genre_list = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    num_genres = len(genre_list)

    feature_dim = top_k + num_genres
    metadata_matrix = np.zeros((num_items + 1, feature_dim), dtype=np.float32)

    genome_grouped = genome_df.groupby("movieId")
    zero_metadata_count = 0

    for orig_id, remapped_id in item_map.items():
        genome_features = np.zeros(top_k, dtype=np.float32)
        if orig_id in genome_grouped.groups:
            group = genome_grouped.get_group(orig_id)
            top_tags = group.nlargest(top_k, "relevance")
            relevances = top_tags["relevance"].values
            genome_features[: len(relevances)] = relevances

        genre_features = np.zeros(num_genres, dtype=np.float32)
        movie_row = movie_df[movie_df["movieId"] == orig_id]
        if not movie_row.empty:
            genres_str = movie_row.iloc[0]["genres"]
            if pd.notna(genres_str) and genres_str != "(no genres listed)":
                for g in genres_str.split("|"):
                    g = g.strip()
                    if g in genre_to_idx:
                        genre_features[genre_to_idx[g]] = 1.0

        row_features = np.concatenate([genome_features, genre_features])
        if row_features.sum() == 0:
            zero_metadata_count += 1
        metadata_matrix[remapped_id] = row_features

    assert np.all(metadata_matrix[0] == 0), "Row 0 (padding) must be all zeros"

    os.makedirs(config.processed_data_dir, exist_ok=True)
    meta_path = os.path.join(config.processed_data_dir, "item_metadata.npy")
    np.save(meta_path, metadata_matrix)

    logger.info(
        "Metadata extracted: shape=%s, feature_dim=%d (genome=%d, genres=%d), "
        "zero-metadata items=%d",
        metadata_matrix.shape, feature_dim, top_k, num_genres, zero_metadata_count,
    )
    return metadata_matrix


def get_timestamps_per_user(config: Config) -> Dict[int, List[float]]:
    rating_path = os.path.join(config.raw_data_dir, "rating.csv")
    df = pd.read_csv(rating_path)
    df = df[df["rating"] >= config.implicit_threshold].copy()

    user_counts = df.groupby("userId").size()
    valid_users = user_counts[user_counts >= config.min_interactions].index
    df = df[df["userId"].isin(valid_users)].copy()
    df = df.sort_values(["userId", "timestamp"])

    user_map_path = os.path.join(config.processed_data_dir, "user_mapping.pkl")
    with open(user_map_path, "rb") as f:
        user_map = pickle.load(f)

    timestamps_per_user: Dict[int, List[float]] = {}
    for user_id_orig, group in df.groupby("userId"):
        remapped = user_map.get(user_id_orig)
        if remapped is not None:
            timestamps_per_user[remapped] = group["timestamp"].tolist()

    return timestamps_per_user
