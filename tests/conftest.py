"""Shared test fixtures.

`tiny_processed_dir` (specs/ablation-run T0.1, design D14): synthetic raw MovieLens-shaped
CSVs run through the REAL `data.preprocessing.run_preprocessing`, so tests exercise the
artifacts the pipeline actually writes without touching `data/` (steering/testing.md,
"Dev-Host Test Hygiene": never copy `data/processed/` into a test).
"""
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import pytest

from configs.config import Config

#: Items 1..TINY_WARM_ITEMS are the ordinary catalogue; TINY_WARM_ITEMS+1..TINY_NUM_ITEMS
#: occur ONLY as some users' final two interactions, so they never reach a training split
#: and the default `unseen_in_training` cold set is non-empty.
TINY_NUM_USERS = 60
TINY_WARM_ITEMS = 140
TINY_NUM_ITEMS = 150
TINY_MOVIE_ID_OFFSET = 1000  # original ids differ from remapped ids on purpose
TINY_GENOME_TAGS = 24
TINY_GENRES = ("Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi")

#: The preprocessing parameters D14 pins for the fixture.
TINY_PREPROCESS = {
    "max_seq_length": 8,
    "sequence_length_grid": [4, 8, 16],
    "num_sampled_negatives": 10,
    "seed": 42,
    "device": "cpu",
}

#: The fixture must stay tiny (D14): a directory above this fails the factory.
TINY_MAX_BYTES = 20 * 1024 * 1024


def write_tiny_raw_csvs(raw_dir: str, seed: int = 0) -> None:
    """Write the six raw CSVs `run_preprocessing` validates, for ~60 users x 150 items."""
    os.makedirs(raw_dir, exist_ok=True)
    rnd = random.Random(seed)
    start = datetime(2005, 1, 1)

    rating_rows = []
    cold_items = list(range(TINY_WARM_ITEMS + 1, TINY_NUM_ITEMS + 1))
    for user in range(1, TINY_NUM_USERS + 1):
        n = rnd.randint(12, 30)
        items = rnd.sample(range(1, TINY_WARM_ITEMS + 1), n)
        if user % 3 == 0:
            # The two final interactions are cold items: with n >= 12 and the default
            # 0.8 / 0.1 split they land in val/test, never in training.
            k = (user // 3) % len(cold_items)
            items[-2:] = [cold_items[k], cold_items[(k + 1) % len(cold_items)]]
        base = start + timedelta(days=user)
        for i, item in enumerate(items):
            ts = base + timedelta(minutes=10 * i)
            rating_rows.append(
                (user, TINY_MOVIE_ID_OFFSET + item, 5.0, ts.strftime("%Y-%m-%d %H:%M:%S"))
            )
    pd.DataFrame(
        rating_rows, columns=["userId", "movieId", "rating", "timestamp"],
    ).to_csv(os.path.join(raw_dir, "rating.csv"), index=False)

    movie_ids = [TINY_MOVIE_ID_OFFSET + i for i in range(1, TINY_NUM_ITEMS + 1)]
    pd.DataFrame({
        "movieId": movie_ids,
        "title": [f"Movie {m} (2000)" for m in movie_ids],
        "genres": [
            "|".join(sorted({TINY_GENRES[m % 6], TINY_GENRES[(m // 6) % 6]}))
            for m in movie_ids
        ],
    }).to_csv(os.path.join(raw_dir, "movie.csv"), index=False)

    pd.DataFrame({
        "userId": [1, 2, 3],
        "movieId": movie_ids[:3],
        "tag": ["fun", "dark", "slow"],
        "timestamp": [start.strftime("%Y-%m-%d %H:%M:%S")] * 3,
    }).to_csv(os.path.join(raw_dir, "tag.csv"), index=False)

    genome_rows = [
        (m, t, round(((m * 31 + t * 17) % 100) / 100.0, 3))
        for m in movie_ids for t in range(1, TINY_GENOME_TAGS + 1)
    ]
    pd.DataFrame(
        genome_rows, columns=["movieId", "tagId", "relevance"],
    ).to_csv(os.path.join(raw_dir, "genome_scores.csv"), index=False)

    pd.DataFrame({
        "tagId": list(range(1, TINY_GENOME_TAGS + 1)),
        "tag": [f"tag{t}" for t in range(1, TINY_GENOME_TAGS + 1)],
    }).to_csv(os.path.join(raw_dir, "genome_tags.csv"), index=False)

    pd.DataFrame({
        "movieId": movie_ids,
        "imdbId": [100000 + m for m in movie_ids],
        "tmdbId": [200000 + m for m in movie_ids],
    }).to_csv(os.path.join(raw_dir, "link.csv"), index=False)


def dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


def build_tiny_processed_dir(root: str, **overrides) -> str:
    """Raw CSVs under `root/raw`, real preprocessing into `root/processed`."""
    from data.preprocessing import run_preprocessing

    raw = os.path.join(root, "raw")
    processed = os.path.join(root, "processed")
    write_tiny_raw_csvs(raw)

    cfg = Config()
    cfg.raw_data_dir = raw
    cfg.processed_data_dir = processed
    cfg.auto_download_dataset = False
    for key, value in {**TINY_PREPROCESS, **overrides}.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"tiny_processed_dir: unknown Config field '{key}'")
        setattr(cfg, key, value)
    run_preprocessing(cfg)

    size = dir_size_bytes(processed)
    assert size < TINY_MAX_BYTES, (
        f"tiny processed dir is {size / 1e6:.1f} MB; the fixture must stay < 20 MB"
    )
    return processed


@pytest.fixture(scope="session")
def tiny_processed_dir(tmp_path_factory):
    """Factory: `tiny_processed_dir(**overrides) -> processed_dir`, cached per override set.

    Overrides are `Config` fields applied before preprocessing (e.g.
    `user_sample_fraction=0.5`). Directories live under pytest's session temp root and
    are built once per distinct override set.
    """
    cache: Dict[Tuple, str] = {}

    def _factory(**overrides) -> str:
        key = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in overrides.items()))
        if key not in cache:
            root = str(tmp_path_factory.mktemp("tiny_data"))
            cache[key] = build_tiny_processed_dir(root, **overrides)
        return cache[key]

    return _factory


def tiny_config(processed_dir: str, out_dir: str, **overrides) -> Config:
    """A CPU t-small config over a tiny processed dir, with every output under `out_dir`.

    The sequence/negative parameters match the ones the processed dir was built with
    (TINY_PREPROCESS); training is capped so a run takes a few seconds.
    """
    cfg = Config()
    cfg.apply_model_config("t-small")
    for key, value in TINY_PREPROCESS.items():
        setattr(cfg, key, value)
    cfg.processed_data_dir = processed_dir
    # The fixture's own raw dir (never data/raw). It is the same for every run over one
    # processed dir, because `raw_data_dir` is part of the training fingerprint.
    cfg.raw_data_dir = os.path.join(os.path.dirname(processed_dir), "raw")
    cfg.output_dir = out_dir
    cfg.checkpoint_dir = os.path.join(out_dir, "checkpoints")
    cfg.log_dir = os.path.join(out_dir, "logs")
    cfg.plot_dir = os.path.join(out_dir, "plots")
    cfg.ablation_dir = os.path.join(out_dir, "ablations")
    cfg.batch_size = 16
    cfg.eval_batch_size = 64
    cfg.pretrain_epochs = 3
    cfg.max_steps_per_epoch = 4
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.warmup_runs = 1
    cfg.timed_runs = 2
    cfg.personalization_eval_users = 2
    cfg.local_epochs = 1
    cfg.federated_rounds = 1
    cfg.clients_per_round = 2
    cfg.federated_eval_examples = 32
    cfg.ablation_eval_examples = 32
    cfg.grad_diag_num_batches = 1
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"tiny_config: unknown Config field '{key}'")
        setattr(cfg, key, value)
    return cfg
