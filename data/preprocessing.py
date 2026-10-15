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


RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]

#: specs/ablation-run FR-6: recorded in dataset_stats.json and the preprocessing manifest.
SAMPLING_FIELDS = (
    "user_sample_fraction",
    "user_sample_seed",
    "num_users_before_sampling",
    "num_users_after_sampling",
    "user_sample_fingerprint",
)


def _read_ratings(config: Config, sampling_info: Optional[dict] = None) -> "pd.DataFrame":
    """Read, filter and chronologically sort `rating.csv` -- ONCE.

    REV-085: the file (20,000,263 rows, 690 MB) was read in full twice with no
    `usecols`, no dtype map and no timestamp parsing. The `timestamp` column alone came
    back as `object` -- Python strings -- at 1.52 GB of the 2.00 GB frame, and each read
    was followed by three more full-width `.copy()` materialisations. Reading once, with
    narrow dtypes and epoch-second timestamps, removes a whole pass and ~1.5 GB of it.
    """
    rating_path = os.path.join(config.raw_data_dir, "rating.csv")
    if not os.path.exists(rating_path):
        raise FileNotFoundError(f"rating.csv not found at {rating_path}")

    logger.info("Loading rating.csv...")
    header = pd.read_csv(rating_path, nrows=0)
    for col in RATING_COLUMNS:
        if col not in header.columns:
            raise ValueError(
                f"rating.csv missing column '{col}'. Found: {list(header.columns)}"
            )

    df = pd.read_csv(
        rating_path,
        usecols=RATING_COLUMNS,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    # Epoch seconds as int64 rather than a column of Python strings.
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9
    )

    original_count = len(df)
    df = df[df["rating"] >= config.implicit_threshold]
    logger.info(
        "Implicit filter (>= %.1f): %d -> %d rows",
        config.implicit_threshold, original_count, len(df),
    )

    user_counts = df.groupby("userId").size()
    valid_users = user_counts[user_counts >= config.min_interactions].index
    df = df[df["userId"].isin(valid_users)]
    logger.info(
        "Min interactions filter (>= %d): %d valid users, %d rows",
        config.min_interactions, len(valid_users), len(df),
    )

    # specs/ablation-run FR-6 / D6: a seeded sample of USERS, drawn AFTER both filters so
    # each kept user's full filtered history survives. Everything downstream (ID maps,
    # splits, cold/warm items, pool, candidates, histories) derives from this frame, so
    # it is rebuilt from the subset by construction. `fraction == 1.0` skips the block
    # entirely, so the full-data output is unchanged.
    fraction = getattr(config, "user_sample_fraction", 1.0)
    if fraction < 1.0:
        users = np.sort(df["userId"].unique())
        n_before = len(users)
        n_keep = max(1, int(round(fraction * n_before)))
        keep = np.random.RandomState(config.user_sample_seed).choice(
            users, size=n_keep, replace=False,
        )
        df = df[df["userId"].isin(keep)]
        kept_sorted = np.sort(keep).astype(np.int64)
        logger.info(
            "User sample (fraction=%g, seed=%d): %d -> %d users, %d rows",
            fraction, config.user_sample_seed, n_before, n_keep, len(df),
        )
        if sampling_info is not None:
            sampling_info.update({
                "user_sample_fraction": float(fraction),
                "user_sample_seed": int(config.user_sample_seed),
                "num_users_before_sampling": int(n_before),
                "num_users_after_sampling": int(n_keep),
                "user_sample_fingerprint": hashlib.sha256(
                    kept_sorted.tobytes()
                ).hexdigest()[:16],
            })
    elif sampling_info is not None:
        n_users = int(df["userId"].nunique())
        sampling_info.update({
            "user_sample_fraction": 1.0,
            "user_sample_seed": int(getattr(config, "user_sample_seed", 42)),
            "num_users_before_sampling": n_users,
            "num_users_after_sampling": n_users,
            "user_sample_fingerprint": None,
        })

    return df.sort_values(["userId", "timestamp"], kind="stable")


def process_ratings(
    config: Config, return_timestamps: bool = False
):
    """Build per-user item sequences, and optionally their timestamps from the same pass.

    REV-085: `get_timestamps_per_user` repeated the identical read/filter/sort purely to
    recover the timestamp column this function had already loaded and discarded.
    `return_timestamps=True` returns both from one pass.
    """
    sampling_info: dict = {}
    df = _read_ratings(config, sampling_info)

    user_ids_orig = sorted(df["userId"].unique())
    user_map = {orig: idx for idx, orig in enumerate(user_ids_orig)}

    item_ids_orig = sorted(df["movieId"].unique())
    item_map = {orig: idx + 1 for idx, orig in enumerate(item_ids_orig)}

    df["user_id"] = df["userId"].map(user_map)
    df["item_id"] = df["movieId"].map(item_map)

    user_sequences: Dict[int, List[int]] = {}
    timestamps_per_user: Dict[int, List[float]] = {}
    skipped_users = 0
    for user_id, group in df.groupby("user_id"):
        items = group["item_id"].tolist()
        if len(items) < 3:
            skipped_users += 1
            continue
        user_sequences[user_id] = items
        if return_timestamps:
            timestamps_per_user[user_id] = group["timestamp"].tolist()

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
        # specs/ablation-run FR-6: the sample this dataset was built from.
        **sampling_info,
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
    if return_timestamps:
        return user_sequences, stats, timestamps_per_user
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

    for user_id, items in user_sequences.items():
        ts = timestamps_per_user[user_id]
        for item, t in zip(items, ts):
            if item not in item_first_seen or t < item_first_seen[item]:
                item_first_seen[item] = t

    all_items: Set[int] = set()
    for items in user_sequences.values():
        all_items.update(items)

    # REV-137 (owner decision 2026-09-18). Three strategies, of which the DEFAULT is
    # "unseen_in_training". The distinction that matters is whether the strategy
    # ACHIEVES the disjointness guarantee or CONSTRUCTS it:
    #
    #   "unseen_in_training" -- cold = all_items - train_interacted_items. An item is
    #       cold exactly when the model never saw it. The removal step below is a no-op
    #       by construction (asserted, not assumed) and `cold_start_ratio` does not
    #       apply: the size is data-determined.
    #   "first_appearance_rank" -- holds out the latest-appearing `cold_start_ratio`
    #       fraction, then DELETES those items' training interactions. Non-empty by
    #       construction but INDUCED: measured on MovieLens-20M, 927 of 2,072 cold items
    #       had 6,634 real training interactions removed across 2,192 users.
    #   "timestamp_cutoff" -- items first seen strictly after `global_training_cutoff`.
    #       Empty on this dataset, because that cutoff is the MAXIMUM over every user's
    #       own training end and so lands 0.1 days before the final interaction -- not
    #       because items appear early. An empty holdout is a hard error below.
    strategy = getattr(config, "cold_start_strategy", "unseen_in_training")

    ranked = sorted(
        ((item_first_seen.get(item, 0.0), item) for item in all_items),
        key=lambda pair: (pair[0], pair[1]),
    )

    # The union over the UNFILTERED training split -- i.e. what the model would see if no
    # holdout were applied. This is the quantity "unseen_in_training" is defined against.
    pre_holdout_train_items: Set[int] = set()
    for items in train_seqs.values():
        pre_holdout_train_items.update(items)

    if strategy == "unseen_in_training":
        selected_cold = sorted(all_items - pre_holdout_train_items)
        # FR-3's optional first-appearance threshold, applied only when configured.
        threshold = getattr(config, "cold_start_first_seen_after", None)
        if threshold is not None:
            selected_cold = [
                item
                for item in selected_cold
                if item_first_seen.get(item, 0.0) > threshold
            ]
            logger.info(
                "Applied first-appearance threshold %.0f: %d items remain cold",
                threshold, len(selected_cold),
            )
    elif strategy == "timestamp_cutoff":
        cold_candidates = [
            (t, item) for t, item in ranked if t > global_training_cutoff
        ]
        max_cold = max(1, int(len(cold_candidates) * config.cold_start_ratio))
        selected_cold = [item for _, item in cold_candidates[:max_cold]]
    elif strategy == "first_appearance_rank":
        n_cold = max(1, int(len(ranked) * config.cold_start_ratio))
        # The latest-appearing items are the tail of the first-appearance ordering.
        selected_cold = [item for _, item in ranked[-n_cold:]]
    else:
        raise ValueError(
            f"Unknown cold_start_strategy '{strategy}'. "
            "Choose from 'unseen_in_training', 'first_appearance_rank', "
            "'timestamp_cutoff'"
        )

    cold_item_set = set(selected_cold)

    if not cold_item_set:
        # REV-011: an empty holdout must fail loudly, not proceed with an INFO log.
        raise RuntimeError(
            f"Cold-start holdout is EMPTY under strategy '{strategy}' with "
            f"cold_start_ratio={config.cold_start_ratio} over {len(all_items)} items. "
            "Cold-start evaluation would be vacuous. Use "
            "cold_start_strategy='unseen_in_training' (the default), which selects "
            "every item the training split never contains."
        )
    warm_item_set = all_items - cold_item_set

    filtered_train: Dict[int, List[int]] = {}
    for user_id, items in train_seqs.items():
        filtered = [i for i in items if i not in cold_item_set]
        if filtered:
            filtered_train[user_id] = filtered

    train_interacted_items: Set[int] = set()
    for items in filtered_train.values():
        train_interacted_items.update(items)

    # AC-03 (amended 2026-09-18): under "unseen_in_training" no training interaction may
    # be deleted to produce the cold set. Assert the no-op property rather than assuming
    # it -- this is what distinguishes a natural cold set from an induced one, and it is
    # the clause `first_appearance_rank` violates (6,634 deletions on the real data).
    if strategy == "unseen_in_training":
        assert filtered_train == train_seqs, (
            "cold_start_strategy='unseen_in_training' must not delete any training "
            "interaction, but the filtered training split differs from the input split. "
            f"users before={len(train_seqs)} after={len(filtered_train)}, "
            f"interactions before={sum(len(v) for v in train_seqs.values())} "
            f"after={sum(len(v) for v in filtered_train.values())}"
        )

    assert cold_item_set.isdisjoint(train_interacted_items), (
        "Cold items found in filtered training data!"
    )
    assert cold_item_set | warm_item_set == all_items, (
        "Cold + warm items don't equal all items"
    )
    assert cold_item_set.isdisjoint(warm_item_set), (
        "Cold and warm items overlap"
    )
    assert len(cold_item_set) > 0, "Cold item set must be non-empty"

    os.makedirs(config.processed_data_dir, exist_ok=True)
    cold_path = os.path.join(config.processed_data_dir, "cold_items.pkl")
    with open(cold_path, "wb") as f:
        pickle.dump(cold_item_set, f)
    warm_path = os.path.join(config.processed_data_dir, "warm_items.pkl")
    with open(warm_path, "wb") as f:
        pickle.dump(warm_item_set, f)

    removed_interactions = (
        sum(len(v) for v in train_seqs.values())
        - sum(len(v) for v in filtered_train.values())
    )
    logger.info(
        "Cold-start holdout (%s): %d cold items, %d warm items, %d total; "
        "%d training interactions removed",
        strategy, len(cold_item_set), len(warm_item_set), len(all_items),
        removed_interactions,
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
    # REV-072: the matrix must span the whole embedding vocabulary -- pad=0,
    # items 1..num_items, MASK_ID=num_items+1 -- not just `num_items + 1` rows.
    # Without the mask row, gathering metadata for a masked sequence is an IndexError,
    # and callers papered over it by clamping MASK_ID onto the last real item's genome
    # vector. Both the pad row and the mask row stay all-zero.
    metadata_matrix = np.zeros((num_items + 2, feature_dim), dtype=np.float32)

    # REV-089: both lookups are built once. Previously each of the ~20,720 items ran a
    # full linear scan of movie.csv (`movie_df[movie_df["movieId"] == orig_id]`) plus a
    # per-item `get_group` and `nlargest` -- hundreds of millions of element comparisons
    # and one intermediate frame per item, in the phase that is already the pipeline's
    # memory peak. The results are identical: ties in `relevance` yield equal values
    # whichever row wins, and movieId is unique in movie.csv.
    top_relevance = (
        genome_df.groupby("movieId")["relevance"].nlargest(top_k).reset_index(level=1, drop=True)
    )
    genome_by_id: Dict[int, np.ndarray] = {
        int(movie_id): values.to_numpy(dtype=np.float32)
        for movie_id, values in top_relevance.groupby(level=0)
    }
    genres_by_id: Dict[int, str] = (
        movie_df.drop_duplicates("movieId", keep="first")
        .set_index("movieId")["genres"]
        .to_dict()
    )
    zero_metadata_count = 0

    for orig_id, remapped_id in item_map.items():
        genome_features = np.zeros(top_k, dtype=np.float32)
        relevances = genome_by_id.get(orig_id)
        if relevances is not None:
            genome_features[: len(relevances)] = relevances

        genre_features = np.zeros(num_genres, dtype=np.float32)
        genres_str = genres_by_id.get(orig_id)
        if genres_str is not None and pd.notna(genres_str) and genres_str != "(no genres listed)":
            for g in genres_str.split("|"):
                g = g.strip()
                if g in genre_to_idx:
                    genre_features[genre_to_idx[g]] = 1.0

        row_features = np.concatenate([genome_features, genre_features])
        if row_features.sum() == 0:
            zero_metadata_count += 1
        metadata_matrix[remapped_id] = row_features

    assert np.all(metadata_matrix[0] == 0), "Row 0 (padding) must be all zeros"
    # REV-072: AC-09 extended to the mask row.
    assert np.all(metadata_matrix[num_items + 1] == 0), (
        f"Row {num_items + 1} (MASK_ID) must be all zeros"
    )

    os.makedirs(config.processed_data_dir, exist_ok=True)
    meta_path = os.path.join(config.processed_data_dir, "item_metadata.npy")
    np.save(meta_path, metadata_matrix)

    logger.info(
        "Metadata extracted: shape=%s, feature_dim=%d (genome=%d, genres=%d), "
        "zero-metadata items=%d",
        metadata_matrix.shape, feature_dim, top_k, num_genres, zero_metadata_count,
    )
    return metadata_matrix


def run_preprocessing(config: Config) -> None:
    from data.sequence_generator import (
        generate_sequences, tensorise_examples, save_tensorised,
        GlobalNegativePool, generate_eval_candidates, save_eval_candidates,
        stored_seq_length,
    )

    # specs/ablation-run D6: an out-of-range sample fraction must fail before any work.
    config.validate()

    logger.info("=== Step 1/7: Acquire dataset ===")
    fingerprint = acquire_dataset(config)

    logger.info("=== Step 2/7: Process ratings (sequences + timestamps, one pass) ===")
    # REV-085: sequences and timestamps come from the SAME read of rating.csv.
    user_sequences, stats, timestamps_per_user = process_ratings(
        config, return_timestamps=True,
    )
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = stats["VOCAB_SIZE"]

    logger.info("=== Step 3/7: Timestamps taken from step 2 ===")

    logger.info("=== Step 4/7: Chronological split ===")
    train_seqs, val_seqs, test_seqs, global_cutoff = compute_chronological_split(
        user_sequences, timestamps_per_user, config,
    )

    logger.info("=== Step 5/7: Identify cold items ===")
    train_interactions_before = sum(len(v) for v in train_seqs.values())
    cold_item_set, warm_item_set, filtered_train = identify_cold_items(
        train_seqs, val_seqs, test_seqs,
        timestamps_per_user, user_sequences, global_cutoff, config,
    )
    # AC-03 (amended 2026-09-18): the before/after comparison is recorded in the manifest
    # so "was any training interaction deleted to make this cold set?" is answerable from
    # the artifact rather than from the code that wrote it.
    train_interactions_removed = train_interactions_before - sum(
        len(v) for v in filtered_train.values()
    )

    logger.info("=== Step 6/7: Extract metadata ===")
    item_map_path = os.path.join(config.processed_data_dir, "item_mapping.pkl")
    with open(item_map_path, "rb") as f:
        item_map = pickle.load(f)
    metadata_matrix = extract_metadata(config, item_map, stats["num_items"])
    config.feature_dim = metadata_matrix.shape[-1]

    logger.info("=== Step 7/7: Generate sequences & tensorise ===")
    # specs/ablation-run FR-7 / D7: store at the widest width any configured arm reads;
    # every reader narrows to its own `max_seq_length`.
    stored_width = stored_seq_length(config)
    logger.info(
        "Storing sequences at width %d (max_seq_length=%d, sequence_length_grid=%s)",
        stored_width, config.max_seq_length, list(config.sequence_length_grid),
    )
    train_examples, val_examples, test_examples, train_user_history, eval_user_history = (
        generate_sequences(
            user_sequences, filtered_train, val_seqs, test_seqs, config,
            width=stored_width,
        )
    )

    train_dataset = tensorise_examples(train_examples, metadata_matrix)
    val_dataset = tensorise_examples(val_examples, metadata_matrix)
    test_dataset = tensorise_examples(test_examples, metadata_matrix)

    processed = config.processed_data_dir
    save_tensorised(train_dataset, os.path.join(processed, "train_tensorised.pt"))
    save_tensorised(val_dataset, os.path.join(processed, "val_tensorised.pt"))
    save_tensorised(test_dataset, os.path.join(processed, "test_tensorised.pt"))

    with open(os.path.join(processed, "train_user_history.pkl"), "wb") as f:
        pickle.dump(train_user_history, f)
    with open(os.path.join(processed, "eval_user_history.pkl"), "wb") as f:
        pickle.dump(eval_user_history, f)

    # REV-002: the example dicts were tensorised but never persisted. Four downstream
    # modules (cold_start, longtail, local_train, federated) open
    # `test_examples.pkl` unconditionally, so all four raised FileNotFoundError on
    # their first line of work and `--stage all` aborted at the personalize stage.
    for name, examples in (
        ("train_examples.pkl", train_examples),
        ("val_examples.pkl", val_examples),
        ("test_examples.pkl", test_examples),
    ):
        with open(os.path.join(processed, name), "wb") as f:
            pickle.dump(examples, f)

    neg_pool = GlobalNegativePool(warm_item_set, seed=config.seed)
    # REV-025: derived from config rather than hardcoded, so the manifest and the
    # artifact agree and an operator tuning `pool_size` sees an effect.
    pool_size = config.resolve_pool_size()
    neg_pool.build_once(pool_size)
    neg_pool.save(os.path.join(processed, "global_negative_pool.npy"))

    # REV-041 / T8.3: generate the fixed evaluation candidate sets ONCE and persist
    # them. Previously every evaluation site rebuilt them, so cross-model identity held
    # only because the procedure happened to be deterministic.
    logger.info("Generating fixed evaluation candidate sets...")
    # REV-087: each split is generated directly into its own on-disk `.npy` memmap and
    # fingerprinted in row blocks, so neither the list-of-arrays form nor the contiguous
    # copy `np.asarray` used to build is ever resident.
    val_cand_path = os.path.join(processed, "val_candidates.npy")
    val_fp = save_eval_candidates(
        generate_eval_candidates(
            val_examples, neg_pool.pool, eval_user_history,
            config.num_sampled_negatives,
            padding_id=0, mask_id=stats["MASK_ID"], seed=config.seed,
            out_path=val_cand_path,
        ),
        val_cand_path,
    )
    test_cand_path = os.path.join(processed, "test_candidates.npy")
    test_fp = save_eval_candidates(
        generate_eval_candidates(
            test_examples, neg_pool.pool, eval_user_history,
            config.num_sampled_negatives,
            padding_id=0, mask_id=stats["MASK_ID"], seed=config.seed,
            out_path=test_cand_path,
        ),
        test_cand_path,
    )

    # Popularity buckets are a pure function of the training history; computing them
    # here means long-tail evaluation loads rather than derives them.
    from evaluation.longtail import compute_popularity_buckets
    compute_popularity_buckets(
        train_user_history,
        set(range(1, stats["num_items"] + 1)),
        os.path.join(processed, "popularity_buckets.pkl"),
    )

    with open(os.path.join(processed, "splits.pkl"), "wb") as f:
        pickle.dump({
            "train": filtered_train, "val": val_seqs, "test": test_seqs,
            "global_cutoff": global_cutoff,
        }, f)

    manifest = {
        "fingerprint": fingerprint,
        "num_users": stats["num_users"],
        "num_items": stats["num_items"],
        "vocab_size": stats["VOCAB_SIZE"],
        "feature_dim": int(config.feature_dim),
        "pool_size": pool_size,
        "pool_fingerprint": neg_pool.fingerprint,
        "cold_start_strategy": getattr(config, "cold_start_strategy", "unseen_in_training"),
        "cold_start_ratio": config.cold_start_ratio,
        "num_cold_items": len(cold_item_set),
        "num_warm_items": len(warm_item_set),
        "train_interactions_before_holdout": int(train_interactions_before),
        "train_interactions_removed_by_holdout": int(train_interactions_removed),
        "num_sampled_negatives": config.num_sampled_negatives,
        "val_candidates_fingerprint": val_fp,
        "test_candidates_fingerprint": test_fp,
        "seed": config.seed,
        "stored_seq_length": int(stored_width),
    }
    # specs/ablation-run FR-6: the same five sampling fields as dataset_stats.json.
    for key in SAMPLING_FIELDS:
        manifest[key] = stats.get(key)
    with open(os.path.join(processed, "preprocessing_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Preprocessing complete. %d users, %d items, vocab=%d, %d cold items, "
        "pool_size=%d, fingerprint=%s",
        stats["num_users"], stats["num_items"], stats["VOCAB_SIZE"],
        len(cold_item_set), pool_size, fingerprint,
    )


def check_processed_dir(config: Config) -> None:
    """Refuse a processed directory built from a different user sample (FR-7, D6).

    Reads `preprocessing_manifest.json`; a field the manifest lacks is read as the
    pre-feature default (fraction 1.0). The seed is compared only for a real sample.
    A directory with no manifest is warned about, not refused.
    """
    path = os.path.join(config.processed_data_dir, "preprocessing_manifest.json")
    if not os.path.exists(path):
        logger.warning(
            "%s not found -- cannot verify that %s was built with "
            "user_sample_fraction=%g / user_sample_seed=%d",
            path, config.processed_data_dir, config.user_sample_fraction,
            config.user_sample_seed,
        )
        return
    with open(path) as f:
        manifest = json.load(f)
    stored_fraction = float(manifest.get("user_sample_fraction", 1.0))
    stored_seed = manifest.get("user_sample_seed")
    remedy = (
        f"Re-run `python main.py --stage preprocess --user_sample_fraction "
        f"{config.user_sample_fraction:g} --user_sample_seed {config.user_sample_seed}` "
        f"or point processed_data_dir at the matching directory."
    )
    if abs(stored_fraction - float(config.user_sample_fraction)) > 1e-12:
        raise ValueError(
            f"{config.processed_data_dir} was preprocessed with user_sample_fraction="
            f"{stored_fraction:g} but this run asks for {config.user_sample_fraction:g}. "
            + remedy
        )
    if stored_fraction < 1.0 and stored_seed is not None and (
        int(stored_seed) != int(config.user_sample_seed)
    ):
        raise ValueError(
            f"{config.processed_data_dir} was preprocessed with user_sample_seed="
            f"{stored_seed} but this run asks for {config.user_sample_seed}. " + remedy
        )


def get_timestamps_per_user(config: Config) -> Dict[int, List[float]]:
    """Per-user timestamps, aligned with `process_ratings`' item sequences.

    REV-085: this used to repeat the full read/filter/sort of `rating.csv` -- a second
    ~4-5 GB transient peak and a second sort of 20M rows -- purely to recover a column
    the first pass had already loaded. `run_preprocessing` now takes both from one call
    to `process_ratings(return_timestamps=True)`; this entry point is kept for callers
    that only want the timestamps, and shares the same single-pass reader.
    """
    df = _read_ratings(config)

    user_map_path = os.path.join(config.processed_data_dir, "user_mapping.pkl")
    with open(user_map_path, "rb") as f:
        user_map = pickle.load(f)

    timestamps_per_user: Dict[int, List[float]] = {}
    for user_id_orig, group in df.groupby("userId"):
        remapped = user_map.get(user_id_orig)
        if remapped is not None:
            timestamps_per_user[remapped] = group["timestamp"].tolist()

    return timestamps_per_user
