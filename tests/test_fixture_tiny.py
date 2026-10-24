"""specs/ablation-run T0.1: the tiny synthetic processed-data fixture."""
import json
import os
import pickle

from tests.conftest import TINY_MAX_BYTES, dir_size_bytes, tiny_config

#: Everything `run_pretrain` opens from the processed directory.
RUN_PRETRAIN_ARTIFACTS = (
    "dataset_stats.json",
    "item_metadata.npy",
    "val_tensorised.pt",
    "train_user_history.pkl",
    "eval_user_history.pkl",
    "preprocessing_manifest.json",
)

#: Read by evaluation, cold-start, long-tail, personalization and federated stages.
DOWNSTREAM_ARTIFACTS = (
    "test_tensorised.pt",
    "global_negative_pool.npy",
    "global_negative_pool.npy.meta.json",
    "val_examples.pkl",
    "test_examples.pkl",
    "cold_items.pkl",
    "warm_items.pkl",
    "popularity_buckets.pkl",
    "splits.pkl",
    "user_mapping.pkl",
    "item_mapping.pkl",
)


def test_fixture_builds_every_artifact_run_pretrain_reads(tiny_processed_dir):
    processed = tiny_processed_dir()
    for name in RUN_PRETRAIN_ARTIFACTS + DOWNSTREAM_ARTIFACTS:
        assert os.path.exists(os.path.join(processed, name)), f"missing {name}"

    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    assert stats["num_users"] > 0
    assert stats["num_items"] > 0

    with open(os.path.join(processed, "cold_items.pkl"), "rb") as f:
        cold = pickle.load(f)
    assert len(cold) > 0, "the unseen_in_training cold set must be non-empty"

    size = dir_size_bytes(processed)
    assert size < TINY_MAX_BYTES, f"{size} bytes"


def test_fixture_never_touches_the_repository_data_dir(tiny_processed_dir, tmp_path_factory):
    processed = os.path.realpath(tiny_processed_dir())
    temp_root = os.path.realpath(str(tmp_path_factory.getbasetemp()))
    raw = os.path.join(os.path.dirname(processed), "raw")
    repo_data = os.path.realpath("data")

    for path in (processed, raw):
        assert path.startswith(temp_root + os.sep), f"{path} is not under {temp_root}"
        assert not path.startswith(repo_data + os.sep), f"{path} is under data/"


def test_fixture_is_cached_per_override_set(tiny_processed_dir):
    assert tiny_processed_dir() == tiny_processed_dir()


def test_tiny_config_points_every_output_at_the_temp_dir(tiny_processed_dir, tmp_path):
    processed = tiny_processed_dir()
    cfg = tiny_config(processed, str(tmp_path))
    assert cfg.processed_data_dir == processed
    for field in ("output_dir", "checkpoint_dir", "log_dir", "plot_dir", "ablation_dir"):
        assert getattr(cfg, field).startswith(str(tmp_path)), field
    assert cfg.device == "cpu"
    assert cfg.max_seq_length == 8
