"""Amazon Reviews 2014 (5-core) as a second dataset: loader, features, preset, fingerprints.

Everything runs on the tiny gzip fixture in `tests/conftest.py`
(`tiny_amazon_processed_dir`), never on `data/raw_beauty`.
"""
import copy
import gzip
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict

import numpy as np
import pytest

from configs.config import DATASET_PRESETS, Config
from tests.conftest import (
    TINY_AMAZON_NO_META_ASIN, TINY_AMAZON_UNREVIEWED_ASIN, tiny_amazon_config,
    write_tiny_amazon_raw,
)

#: `_training_fingerprint(Config())` computed BEFORE the Amazon loader existed. Finished
#: MovieLens arms are reused by fingerprint, so this must never move.
ML_DEFAULT_FINGERPRINT = "f5576d4fe175"

#: Config fields added with the Amazon loader.
AMAZON_FIELDS = (
    "dataset_format", "amazon_category", "amazon_title_dim", "amazon_top_brands",
    "amazon_price_bins", "amazon_top_categories", "epoch_snapshot_interval",
)


def _amazon_cfg(raw_dir, processed_dir="unused"):
    cfg = Config()
    cfg.apply_dataset_preset("amazon_beauty_2014")
    cfg.raw_data_dir = str(raw_dir)
    cfg.processed_data_dir = str(processed_dir)
    return cfg


def _sasrec_sequences(raw_dir):
    """Per-reviewer asin lists the way the standard SASRec script builds them: append in
    file order, then Python's stable sort by time (ties keep file order)."""
    users = {}
    with gzip.open(os.path.join(raw_dir, "reviews_Beauty_5.json.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            users.setdefault(r["reviewerID"], []).append((r["unixReviewTime"], r["asin"]))
    return {u: [a for _, a in sorted(v, key=lambda x: x[0])] for u, v in users.items()}


def _load(processed, name):
    with open(os.path.join(processed, name), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- parsing


class TestParsing:
    def test_reviews_map_onto_the_shared_frame_in_file_order(self, tmp_path):
        from data.amazon import REVIEW_ORDER_COLUMN, read_reviews

        write_tiny_amazon_raw(str(tmp_path))
        df = read_reviews(_amazon_cfg(tmp_path))
        assert list(df.columns) == [
            "userId", "movieId", "rating", "timestamp", REVIEW_ORDER_COLUMN,
        ]
        assert df["userId"].map(type).eq(str).all() and df["movieId"].map(type).eq(str).all()
        assert df[REVIEW_ORDER_COLUMN].tolist() == list(range(len(df)))
        with gzip.open(tmp_path / "reviews_Beauty_5.json.gz", "rt") as f:
            first = json.loads(f.readline())
        row = df.iloc[0]
        assert (row["userId"], row["movieId"]) == (first["reviewerID"], first["asin"])
        assert row["timestamp"] == first["unixReviewTime"]  # already epoch seconds
        assert row["rating"] == pytest.approx(first["overall"])

    def test_metadata_is_parsed_as_python_literals(self, tmp_path):
        from data.amazon import read_metadata

        write_tiny_amazon_raw(str(tmp_path))
        with gzip.open(tmp_path / "meta_Beauty.json.gz", "rt") as f:
            line = f.readline()
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)  # single-quoted dict literal, not JSON

        from tests.conftest import tiny_amazon_asins

        asins = tiny_amazon_asins()
        meta = read_metadata(_amazon_cfg(tmp_path), asins)
        assert TINY_AMAZON_NO_META_ASIN not in meta
        assert TINY_AMAZON_UNREVIEWED_ASIN not in meta  # not asked for
        assert len(meta) == len(asins) - 1
        # The fixture's k == 1 title contains an apostrophe, so repr() double-quotes it.
        assert meta[asins[1]]["title"].startswith("Men's")
        assert "title" not in meta[asins[0]] and "categories" not in meta[asins[0]]

    @pytest.mark.parametrize("missing", ["reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"])
    def test_a_missing_file_names_the_file_and_directory(self, tmp_path, missing):
        from data.amazon import acquire

        write_tiny_amazon_raw(str(tmp_path))
        os.remove(tmp_path / missing)
        with pytest.raises(FileNotFoundError) as exc:
            acquire(_amazon_cfg(tmp_path))
        assert missing in str(exc.value) and str(tmp_path) in str(exc.value)

    def test_category_tokens_are_second_level_and_leaf(self):
        from data.amazon import _category_tokens

        assert _category_tokens([
            ["Beauty", "Skin Care", "Face", "Creams"], ["Beauty", "Makeup"], ["Beauty"],
        ]) == ["Skin Care", "Creams", "Makeup"]
        assert _category_tokens(None) == []
        # The 2014 metadata keeps HTML entities in its strings.
        assert _category_tokens([["Beauty", "Bath &amp; Body"]]) == ["Bath & Body"]


# ---------------------------------------------------------------- preprocessing


class TestPreprocessing:
    def test_same_day_ties_keep_file_order_through_the_loo_split(self, tiny_amazon_processed_dir):
        processed = tiny_amazon_processed_dir()
        raw = os.path.join(os.path.dirname(processed), "raw")
        expected = _sasrec_sequences(raw)
        user_map = _load(processed, "user_mapping.pkl")
        item_map = _load(processed, "item_mapping.pkl")
        splits = _load(processed, "splits.pkl")

        for reviewer, asins in expected.items():
            items = [item_map[a] for a in asins]
            u = user_map[reviewer]
            assert splits["train"][u] == items[:-2], reviewer
            assert splits["val"][u] == [items[-2]], reviewer
            assert splits["test"][u] == [items[-1]], reviewer

        # The fixture really contains same-day ties whose file order is neither asin
        # order nor remapped-id order, including tied val/test pairs -- otherwise this
        # test could pass under a sort that breaks ties some other way.
        by_day = {}
        with gzip.open(os.path.join(raw, "reviews_Beauty_5.json.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                by_day.setdefault((r["reviewerID"], r["unixReviewTime"]), []).append(r["asin"])
        tied = [v for v in by_day.values() if len(v) > 1]
        assert any(v != sorted(v) for v in tied)
        assert any([item_map[a] for a in v] != sorted(item_map[a] for a in v) for v in tied)
        last_two_tied = 0
        with gzip.open(os.path.join(raw, "reviews_Beauty_5.json.gz"), "rt") as f:
            stamps = {}
            for line in f:
                r = json.loads(line)
                stamps.setdefault(r["reviewerID"], []).append(r["unixReviewTime"])
        for reviewer, ts in stamps.items():
            ts = sorted(ts)
            last_two_tied += ts[-1] == ts[-2]
        assert last_two_tied > 0, "no user has a same-day validation/test pair"

    def test_test_examples_end_with_the_tied_val_item(self, tiny_amazon_processed_dir):
        processed = tiny_amazon_processed_dir()
        splits = _load(processed, "splits.pkl")
        test = _load(processed, "test_examples.pkl")
        for row, (u, target) in enumerate(zip(test["user_ids"], test["targets"])):
            seq = [int(i) for i in test["sequences"][row] if i != 0]
            assert int(target) == splits["test"][int(u)][0]
            assert seq[-1] == splits["val"][int(u)][0]

    def test_stats_and_expected_stats_warning(self, tmp_path, caplog):
        from data.preprocessing import run_preprocessing
        from tests.conftest import TINY_PREPROCESS

        write_tiny_amazon_raw(str(tmp_path / "raw"))
        cfg = _amazon_cfg(tmp_path / "raw", tmp_path / "processed")
        for k, v in TINY_PREPROCESS.items():
            setattr(cfg, k, v)
        with caplog.at_level(logging.INFO):
            run_preprocessing(cfg)
        assert "differ from the expected full 5-core" in caplog.text
        assert "40 users, 60 items" in caplog.text
        stats = json.load(open(tmp_path / "processed" / "dataset_stats.json"))
        assert (stats["num_users"], stats["num_items"]) == (40, 60)

    def test_matching_stats_log_no_warning(self, caplog):
        from data.amazon import EXPECTED_STATS, check_expected_stats

        with caplog.at_level(logging.INFO):
            check_expected_stats(Config(amazon_category="Beauty"), dict(EXPECTED_STATS["Beauty"]))
        assert "22363 users, 12101 items, 198502 interactions" in caplog.text
        assert "differ" not in caplog.text

    def test_empty_cold_set_is_allowed_and_recorded(self, tiny_amazon_processed_dir):
        processed = tiny_amazon_processed_dir()
        assert _load(processed, "cold_items.pkl") == set()
        assert len(_load(processed, "warm_items.pkl")) == 60
        manifest = json.load(open(os.path.join(processed, "preprocessing_manifest.json")))
        assert manifest["num_cold_items"] == 0
        assert manifest["dataset_format"] == "amazon2014"
        assert manifest["amazon_category"] == "Beauty"

    def test_movielens_keeps_the_empty_cold_set_error(self, tmp_path):
        from data.preprocessing import identify_cold_items

        cfg = Config()
        cfg.processed_data_dir = str(tmp_path)
        # Every item has a training interaction, so "unseen_in_training" is empty.
        seqs = {0: [1, 2], 1: [2, 1]}
        ts = {0: [1.0, 2.0], 1: [1.0, 2.0]}
        train = {0: [1], 1: [2]}
        with pytest.raises(RuntimeError, match="EMPTY"):
            identify_cold_items(train, {}, {}, ts, seqs, 2.0, cfg)
        cfg.dataset_format = "amazon2014"
        cold, warm, filtered = identify_cold_items(train, {}, {}, ts, seqs, 2.0, cfg)
        assert cold == set() and warm == {1, 2} and filtered == train

    def test_processed_dir_of_the_other_dataset_is_refused(self, tiny_amazon_processed_dir):
        from data.preprocessing import check_processed_dir

        processed = tiny_amazon_processed_dir()
        ml = Config()
        ml.processed_data_dir = processed
        with pytest.raises(ValueError, match="dataset_format='amazon2014'"):
            check_processed_dir(ml)
        amazon = copy.deepcopy(ml)
        amazon.dataset_format = "amazon2014"
        check_processed_dir(amazon)
        amazon.amazon_category = "Sports_and_Outdoors"
        with pytest.raises(ValueError, match="category"):
            check_processed_dir(amazon)


# ---------------------------------------------------------------- item features


class TestFeatures:
    def test_matrix_shape_blocks_and_zero_rows(self, tiny_amazon_processed_dir):
        from tests.conftest import tiny_amazon_asins

        processed = tiny_amazon_processed_dir()
        mat = np.load(os.path.join(processed, "item_metadata.npy"))
        item_map = _load(processed, "item_mapping.pkl")
        num_items = len(item_map)
        cfg = Config()
        fixed = cfg.amazon_top_brands + cfg.amazon_price_bins + cfg.amazon_top_categories
        title_dim = mat.shape[1] - fixed
        assert mat.shape[0] == num_items + 2
        assert 0 < title_dim <= cfg.amazon_title_dim
        assert not mat[0].any() and not mat[num_items + 1].any()  # pad and MASK
        assert not mat[item_map[TINY_AMAZON_NO_META_ASIN]].any()  # no metadata line
        assert not mat[item_map[tiny_amazon_asins()[0]]].any()   # every field missing
        manifest = json.load(open(os.path.join(processed, "preprocessing_manifest.json")))
        assert manifest["feature_dim"] == mat.shape[1]

        brand = mat[:, title_dim:title_dim + cfg.amazon_top_brands]
        price = mat[:, title_dim + cfg.amazon_top_brands:title_dim + cfg.amazon_top_brands
                    + cfg.amazon_price_bins]
        cats = mat[:, -cfg.amazon_top_categories:]
        assert set(np.unique(brand)) <= {0.0, 1.0} and brand.sum(1).max() == 1
        assert set(np.unique(price)) <= {0.0, 1.0} and price.sum(1).max() == 1
        assert set(np.unique(cats)) <= {0.0, 1.0}
        # Multi-hot: the fixture's items carry two paths -> up to three tokens.
        assert cats.sum(1).max() >= 2
        norms = np.linalg.norm(mat[:, :title_dim], axis=1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)

    def test_unreviewed_meta_never_enters_the_vocabularies(self, tiny_amazon_processed_dir):
        processed = tiny_amazon_processed_dir()
        assert TINY_AMAZON_UNREVIEWED_ASIN not in _load(processed, "item_mapping.pkl")

    def test_metadata_off_drops_the_matrix(self, tiny_amazon_processed_dir):
        from data.sequence_generator import load_metadata_matrix

        cfg = Config(dataset_format="amazon2014", use_metadata=False)
        assert load_metadata_matrix(tiny_amazon_processed_dir(), cfg) is None


# ---------------------------------------------------------------- preset and CLI


class TestPreset:
    def test_preset_values(self):
        cfg = Config()
        cfg.apply_dataset_preset("amazon_beauty_2014")
        assert cfg.dataset_format == "amazon2014" and cfg.amazon_category == "Beauty"
        assert cfg.raw_data_dir == "data/raw_beauty"
        assert cfg.processed_data_dir == "data/processed_beauty"
        assert cfg.max_seq_length == 20
        assert cfg.sequence_length_grid == [5, 10, 15, 20]
        assert cfg.pretrain_epochs == 300
        assert cfg.epoch_snapshot_interval == 0
        for name in ("output_dir", "checkpoint_dir", "log_dir", "plot_dir", "ablation_dir"):
            value = getattr(cfg, name)
            assert value.startswith("outputs/amazon_beauty_2014"), name
            assert value != getattr(Config(), name)
        # Everything else stays at the defaults.
        unchanged = {k for k in asdict(Config()) if k not in DATASET_PRESETS["amazon_beauty_2014"]}
        base = asdict(Config())
        assert {k: base[k] for k in unchanged} == {k: asdict(cfg)[k] for k in unchanged}
        assert cfg.batch_size == 256 and cfg.multi_step_k == 3
        cfg.validate()

    def test_reference_arm_coincides_with_the_longest_h2_arm(self):
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        cfg = Config()
        cfg.apply_dataset_preset("amazon_beauty_2014")
        h = {c._study_label: c for c in resolve_study_configs("H", cfg)}
        assert [k for k in h if k.startswith("H2_")] == [
            "H2_seq5", "H2_seq10", "H2_seq15", "H2_seq20",
        ]
        assert _training_fingerprint(h["H2_seq20"]) == _training_fingerprint(h["H3_cold_start"])

    def test_movielens_preset_is_the_defaults(self):
        cfg = Config()
        cfg.apply_dataset_preset("movielens_20m")
        assert asdict(cfg) == asdict(Config())

    def test_unknown_preset_and_format_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown dataset preset"):
            Config().apply_dataset_preset("netflix")
        with pytest.raises(ValueError, match="dataset_format"):
            Config(dataset_format="amazon2023").validate()
        with pytest.raises(ValueError, match="epoch_snapshot_interval"):
            Config(epoch_snapshot_interval=-1).validate()

    def _cli(self, argv):
        from main import apply_cli_overrides, parse_args

        args = parse_args(argv)
        cfg = Config()
        apply_cli_overrides(cfg, args)
        return cfg

    def test_cli_flag_applies_the_preset(self):
        cfg = self._cli(["--stage", "ablation", "--dataset_preset", "amazon_beauty_2014"])
        expected = Config()
        expected.apply_dataset_preset("amazon_beauty_2014")
        assert asdict(cfg) == asdict(expected)

    def test_explicit_flags_override_the_preset(self):
        cfg = self._cli([
            "--stage", "ablation", "--dataset_preset", "amazon_beauty_2014",
            "--epochs", "7", "--max_seq_length", "10", "--sequence_length_grid", "5", "10",
            "--raw_data_dir", "/tmp/r", "--processed_data_dir", "/tmp/p",
            "--user_sample_fraction", "0.5",
        ])
        assert cfg.dataset_format == "amazon2014"
        assert (cfg.pretrain_epochs, cfg.max_seq_length) == (7, 10)
        assert cfg.sequence_length_grid == [5, 10]
        assert (cfg.raw_data_dir, cfg.processed_data_dir) == ("/tmp/r", "/tmp/p")

    def test_user_sample_dirs_are_namespaced_by_dataset(self):
        beauty = self._cli([
            "--stage", "preprocess", "--dataset_preset", "amazon_beauty_2014",
            "--user_sample_fraction", "0.1",
        ])
        assert beauty.processed_data_dir == "data/processed_beauty_users10pct"
        ml = self._cli(["--stage", "preprocess", "--user_sample_fraction", "0.1"])
        assert ml.processed_data_dir == "data/processed_users10pct"
        ml_seed = self._cli([
            "--stage", "preprocess", "--user_sample_fraction", "0.1", "--user_sample_seed", "7",
        ])
        assert ml_seed.processed_data_dir == "data/processed_users10pct_seed7"

    def test_preset_overrides_are_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="main"):
            self._cli(["--stage", "ablation", "--dataset_preset", "amazon_beauty_2014"])
        assert "pretrain_epochs [preset amazon_beauty_2014] 20 -> 300" in caplog.text


# ---------------------------------------------------------------- fingerprints


def _legacy_fingerprint(cfg):
    """The fingerprint payload as computed before the Amazon fields existed."""
    from training.ablation_runner import (
        TRAINING_SEMANTICS_VERSION, _GRADNORM_ONLY_FIELDS, _DEFAULT_OMITTED_FIELDS,
        _PATH_FIELDS, _REMOVED_FIELD_DEFAULTS, _RUN_SELECTION_FIELDS, _RUNTIME_ONLY_FIELDS,
        POST_TRAINING_ONLY_FIELDS,
    )

    # The config then still had the LoRA / federated fields, at their defaults.
    full = {**copy.deepcopy(_REMOVED_FIELD_DEFAULTS), **asdict(cfg)}
    d = {k: v for k, v in full.items() if k not in AMAZON_FIELDS}
    fields = {
        k: v for k, v in d.items()
        if k not in POST_TRAINING_ONLY_FIELDS and k not in _PATH_FIELDS
        and k not in _RUN_SELECTION_FIELDS and k not in _RUNTIME_ONLY_FIELDS
        and not (k in _DEFAULT_OMITTED_FIELDS and v == _DEFAULT_OMITTED_FIELDS[k]
                 and type(v) is type(_DEFAULT_OMITTED_FIELDS[k]))
        and not (k in _GRADNORM_ONLY_FIELDS and not d.get("use_gradnorm", True))
    }
    fields["_training_semantics"] = TRAINING_SEMANTICS_VERSION
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:12]


class TestFingerprint:
    def test_movielens_default_fingerprint_is_unchanged(self):
        from training.ablation_runner import _training_fingerprint

        assert _training_fingerprint(Config()) == ML_DEFAULT_FINGERPRINT
        cfg = Config()
        cfg.apply_model_config("t-base")
        assert _training_fingerprint(cfg) == ML_DEFAULT_FINGERPRINT

    def test_every_movielens_arm_matches_the_pre_amazon_payload(self):
        from training.ablation_runner import (
            STUDY_CONFIGS, _training_fingerprint, resolve_study_configs,
        )

        for model in ("t-small", "t-base"):
            base = Config()
            base.apply_model_config(model)
            for study in STUDY_CONFIGS:
                for arm in resolve_study_configs(study, base):
                    assert _training_fingerprint(arm) == _legacy_fingerprint(arm), (
                        study, arm._study_label,
                    )

    def test_beauty_fingerprints_differ_from_movielens(self):
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        beauty = Config()
        beauty.apply_dataset_preset("amazon_beauty_2014")
        assert _training_fingerprint(beauty) != ML_DEFAULT_FINGERPRINT
        # Even with every other field equal to MovieLens', the format alone separates them.
        same_fields = Config(dataset_format="amazon2014")
        assert _training_fingerprint(same_fields) != ML_DEFAULT_FINGERPRINT
        ml_fps = {_training_fingerprint(c) for c in resolve_study_configs("A", Config())}
        b_fps = {_training_fingerprint(c) for c in resolve_study_configs("A", beauty)}
        assert not ml_fps & b_fps
        bigger_titles = copy.deepcopy(beauty)
        bigger_titles.amazon_title_dim = 64
        assert _training_fingerprint(bigger_titles) != _training_fingerprint(beauty)

    def test_snapshot_interval_is_not_training_relevant(self):
        from training.ablation_runner import _training_fingerprint

        assert _training_fingerprint(Config(epoch_snapshot_interval=0)) == ML_DEFAULT_FINGERPRINT


# ---------------------------------------------------------------- snapshot retention


class TestSnapshotRetention:
    @pytest.mark.parametrize("interval,expected", [(1, [0, 1, 2]), (2, [1]), (0, [])])
    def test_interval_selects_the_kept_epochs(
        self, tiny_amazon_processed_dir, tmp_path, interval, expected,
    ):
        from training.pretrain import run_pretrain

        cfg = tiny_amazon_config(
            tiny_amazon_processed_dir(), str(tmp_path), epoch_snapshot_interval=interval,
            pretrain_epochs=3, max_steps_per_epoch=2,
        )
        result = run_pretrain(cfg, "t-small")
        ckpt_dir = cfg.model_checkpoint_dir()
        kept = sorted(
            int(n[6:9]) for n in os.listdir(ckpt_dir) if n.startswith("epoch_")
        )
        assert kept == expected
        assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "last.pt"))
        assert 0 <= result["best_epoch"] <= 2

    def test_resume_without_snapshots_is_bit_identical(
        self, tiny_amazon_processed_dir, tmp_path, monkeypatch,
    ):
        from tests.test_resume import _final_hash_from_last, _traced_run

        processed = tiny_amazon_processed_dir()
        pinned = dict(pretrain_epochs=3, max_steps_per_epoch=2)
        every = tiny_amazon_config(processed, str(tmp_path / "every"),
                                   epoch_snapshot_interval=1, **pinned)
        _traced_run(every, monkeypatch)

        none = tiny_amazon_config(processed, str(tmp_path / "none"),
                                  epoch_snapshot_interval=0, **pinned)
        result, _ = _traced_run(none, monkeypatch, fail_at_call=4)  # dies in epoch 1
        assert result is None
        result, resumed = _traced_run(copy.deepcopy(none), monkeypatch)
        assert result is not None
        assert resumed["calls"] == 4  # epochs 1 and 2 only: it resumed from last.pt
        assert _final_hash_from_last(none) == _final_hash_from_last(every)
        assert not [n for n in os.listdir(none.model_checkpoint_dir()) if n.startswith("epoch_")]


# ---------------------------------------------------------------- end to end


class TestEndToEnd:
    def test_pretrain_evaluate_and_system_stages(self, tiny_amazon_processed_dir, tmp_path):
        """pretrain -> test evaluation -> cold start (empty cold set) -> long tail."""
        from argparse import Namespace

        from main import run_cold_start_stage, run_evaluate_stage, run_longtail_stage
        from training.pretrain import run_pretrain

        cfg = tiny_amazon_config(tiny_amazon_processed_dir(), str(tmp_path),
                                 pretrain_epochs=1, max_steps_per_epoch=2)
        run_pretrain(cfg, "t-small")
        args = Namespace(model="t-small")
        run_evaluate_stage(copy.deepcopy(cfg), args)
        metrics = json.load(open(os.path.join(cfg.log_dir, "test_metrics_t_small.json")))
        assert 0.0 <= metrics["ndcg"]["mean"] <= 1.0
        assert "popularity_reference" in metrics

        run_cold_start_stage(copy.deepcopy(cfg), args)
        cold = json.load(open(os.path.join(cfg.log_dir, "cold_start_t_small.json")))
        assert cold["cold_set"]["num_cold_items"] == 0
        assert cold["cold_set"]["num_cold_test_rows"] == 0
        assert cold["cold_meta_on"] == {} and cold["cold_popularity"] == {}
        assert np.isfinite(cold["warm_meta_on"]["ndcg"]["mean"])

        run_longtail_stage(copy.deepcopy(cfg), args)
        assert os.path.exists(os.path.join(cfg.log_dir, "longtail_t_small.json"))

    def test_every_ablation_study_runs_on_the_fixture(self, tiny_amazon_processed_dir, tmp_path):
        """Every study, including H3 (empty cold set) and H5, few steps."""
        from training.ablation_runner import STUDY_CONFIGS, run_ablation

        cfg = tiny_amazon_config(tiny_amazon_processed_dir(), str(tmp_path),
                                 pretrain_epochs=1, max_steps_per_epoch=2)
        results = run_ablation(cfg, "t-small", "all")
        assert set(results) == set(STUDY_CONFIGS)
        for study, arms in results.items():
            for arm in arms:
                assert arm.get("status") == "ok", (study, arm["label"], arm.get("error"))
                assert arm["resolved_config"]["dataset_format"] == "amazon2014"
                assert np.isfinite(arm["metrics"]["ndcg"]["mean"])
        by_label = {a["label"]: a for a in results["H"]}
        assert by_label["H3_cold_start"]["cold_start"]["cold_set"]["num_cold_items"] == 0
        assert "latency" in by_label["H5_latency"]
        study_json = os.path.join(cfg.ablation_dir, "seed_42")
        assert os.path.isdir(study_json)
        # The Beauty preset keeps no per-epoch snapshot in the backbone store.
        store = os.path.join(cfg.checkpoint_dir, "ablations", "by_fingerprint")
        for dirpath, _, files in os.walk(store):
            assert not [f for f in files if f.startswith("epoch_")], dirpath
