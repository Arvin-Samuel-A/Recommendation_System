"""specs/ablation-run Phase 10 (FR-14..FR-17): splitting a study's arms across processes."""
import copy
import json
import os

import pytest

from configs.config import Config
from tests.conftest import tiny_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_FIELDS = ("num_items", "num_users", "vocab_size", "feature_dim")


def _cfg(tiny_processed_dir, out_dir, **overrides):
    pinned = dict(pretrain_epochs=1, max_steps_per_epoch=2, ablation_eval_examples=32)
    pinned.update(overrides)
    return tiny_config(tiny_processed_dir(), str(out_dir), **pinned)


class TestResolveDatasetFields:
    """T10.1: one helper for the dataset-derived fields (D20)."""

    def test_stats_dir_overrides_the_configured_path(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.ablation_runner as runner
        import training.pretrain as pretrain

        base = _cfg(tiny_processed_dir, tmp_path)
        arm = runner.resolve_study_configs("B", base)[0]

        # What _run_single_config resolves for the real dir (stop right after it).
        seen = {}

        def stop(cfg, model_name):
            seen.update({f: getattr(cfg, f) for f in DATASET_FIELDS})
            raise RuntimeError("stop after resolving")

        monkeypatch.setattr(pretrain, "run_pretrain", stop)
        with pytest.raises(RuntimeError, match="stop after resolving"):
            runner._run_single_config(copy.deepcopy(arm), "t-small", "B", base, True)

        remote = copy.deepcopy(arm)
        remote.processed_data_dir = "data/processed_users50pct"  # a path that does not exist here
        runner.resolve_dataset_fields(remote, stats_dir=tiny_processed_dir())
        assert {f: getattr(remote, f) for f in DATASET_FIELDS} == seen
        assert seen["num_items"] > 0 and seen["feature_dim"] > 0
        assert remote.processed_data_dir == "data/processed_users50pct"


class TestArmSelection:
    """T10.2 / AC-32, AC-33."""

    def test_parse_arms(self):
        from training.ablation_runner import parse_arms

        assert parse_arms(None) is None
        assert parse_arms("b, a,b ,a") == ["b", "a"]
        for empty in ("", " , ", ","):
            with pytest.raises(ValueError, match="names no arm"):
                parse_arms(empty)

    def test_only_the_listed_arms_run_in_order(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.ablation_runner as runner

        calls = []

        def fake(cfg, model_name, study, base, train, **kwargs):
            calls.append(cfg._study_label)
            return {"label": cfg._study_label, "study": study, "status": "ok"}

        monkeypatch.setattr(runner, "_run_single_config", fake)
        cfg = _cfg(tiny_processed_dir, tmp_path)
        runner.run_ablation(cfg, "t-small", "D", arms=["arv_gradnorm", "none_no_gradnorm", "arv_gradnorm"])
        assert calls == ["arv_gradnorm", "none_no_gradnorm"]

        calls.clear()
        runner.run_ablation(_cfg(tiny_processed_dir, tmp_path / "b"), "t-small", "B", arms=["input_only"])
        assert calls == ["input_only"]
        arms_dir = os.path.join(str(tmp_path / "b"), "ablations", "seed_42", "arms", "B_t_small")
        assert sorted(f for f in os.listdir(arms_dir) if f.endswith(".json")) == ["input_only.json"]

    def test_grid_labels_follow_the_config(self, tiny_processed_dir, tmp_path):
        from training.ablation_runner import select_study_configs

        cfg = _cfg(tiny_processed_dir, tmp_path)
        picked = select_study_configs("H", cfg, ["H2_seq16", "H1_no_metadata"])
        assert [c._study_label for c in picked] == ["H2_seq16", "H1_no_metadata"]
        assert picked[0].max_seq_length == 16

    @staticmethod
    def _spies(monkeypatch):
        import data.preprocessing as prep
        import data.sequence_generator as seqgen
        import training.pretrain as pretrain

        counts = {"check_processed_dir": 0, "run_pretrain": 0, "load_tensorised": 0}

        def spy(name, real):
            def wrapper(*a, **k):
                counts[name] += 1
                return real(*a, **k)
            return wrapper

        monkeypatch.setattr(prep, "check_processed_dir", spy("check_processed_dir", prep.check_processed_dir))
        monkeypatch.setattr(pretrain, "run_pretrain", spy("run_pretrain", pretrain.run_pretrain))
        monkeypatch.setattr(seqgen, "load_tensorised", spy("load_tensorised", seqgen.load_tensorised))
        return counts

    @pytest.mark.parametrize("study,arms,message", [
        ("B", ["no_such_arm"], "valid labels: \\['objective_conditioned', 'input_only'\\]"),
        ("B", ["mmoe"], "valid labels"),
        ("all", ["x"], "exactly one --study"),
        ("B", ",", "names no arm"),
    ])
    def test_bad_selection_fails_before_any_work(
        self, tiny_processed_dir, tmp_path, monkeypatch, study, arms, message,
    ):
        import training.ablation_runner as runner

        counts = self._spies(monkeypatch)
        with pytest.raises(ValueError, match=message):
            runner.run_ablation(_cfg(tiny_processed_dir, tmp_path), "t-small", study, arms=arms)
        assert counts == {"check_processed_dir": 0, "run_pretrain": 0, "load_tensorised": 0}

    @pytest.mark.parametrize("argv,message", [
        (["--stage", "ablation", "--study", "B", "--arms", "mmoe"], "valid labels"),
        (["--stage", "ablation", "--study", "all", "--arms", "x"], "exactly one --study"),
        (["--stage", "pretrain", "--arms", "input_only"], "only valid with --stage ablation"),
        (["--stage", "ablation", "--study", "B", "--arms", ""], "names no arm"),
    ])
    def test_main_rejects_bad_selection_without_exit_75(self, monkeypatch, argv, message):
        import main as cli

        counts = self._spies(monkeypatch)
        snapshots = []
        monkeypatch.setattr(cli, "save_config_snapshot", lambda *a, **k: snapshots.append(1))
        with pytest.raises(ValueError, match=message):
            cli.main(argv + ["--model", "t-small"])
        assert counts == {"check_processed_dir": 0, "run_pretrain": 0, "load_tensorised": 0}
        assert snapshots == []


class TestStudyOutputsAndFailedArms:
    """T10.3 / FR-15, D16."""

    def test_failed_arm_is_persisted_and_retried(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.ablation_runner as runner

        calls = []
        failures = {"left": 1}

        def flaky(cfg, model_name, study, base, train):
            calls.append(cfg._study_label)
            if cfg._study_label == "input_only" and failures["left"]:
                failures["left"] -= 1
                raise KeyError("boom")
            return {"label": cfg._study_label, "study": study, "status": "ok"}

        monkeypatch.setattr(runner, "_run_single_config", flaky)
        cfg = _cfg(tiny_processed_dir, tmp_path)
        runner.run_ablation(cfg, "t-small", "B")
        path = runner._arm_path(cfg, "B", "t-small", "input_only")
        record = json.load(open(path))
        assert record["status"] == "failed" and "KeyError" in record["error"]
        assert record["arm_key"] and record["code_revision"]["source_hash"]
        ok = json.load(open(runner._arm_path(cfg, "B", "t-small", "objective_conditioned")))
        assert ok["code_revision"] == record["code_revision"]

        calls.clear()
        results = runner.run_ablation(_cfg(tiny_processed_dir, tmp_path), "t-small", "B")["B"]
        assert calls == ["input_only"], "the failed arm is retried, the ok arm reused"
        assert json.load(open(path))["status"] == "ok"
        assert results[0]["reused_from_previous_run"] is True

    def test_full_run_writes_study_outputs_under_the_study_lock(
        self, tiny_processed_dir, tmp_path, monkeypatch,
    ):
        import training.ablation_runner as runner

        monkeypatch.setattr(
            runner, "_run_single_config",
            lambda cfg, m, study, base, train: {
                "label": cfg._study_label, "study": study, "status": "ok",
                "metrics": {"ndcg": {"mean": 0.1}},
            },
        )
        cfg = _cfg(tiny_processed_dir, tmp_path)
        lock_path = runner._study_lock_path(cfg, "B", "t-small")
        held = []

        def probe(name, real):
            def wrapper(*args, **kwargs):
                try:
                    with runner.file_lock(lock_path, timeout=0, poll=0.01):
                        held.append((name, False))
                except TimeoutError:
                    held.append((name, True))
                return real(*args, **kwargs)
            return wrapper

        monkeypatch.setattr(runner, "_save_study_results", probe("json", runner._save_study_results))
        monkeypatch.setattr(runner, "_plot_study", probe("plot", runner._plot_study))
        runner.run_ablation(cfg, "t-small", "B")
        assert held == [("json", True), ("plot", True)]
        assert os.path.exists(os.path.join(cfg.ablation_dir, "seed_42", "B_t_small.json"))


# ---------------------------------------------------------------- T10.4 (AC-34..AC-36)


def _patch_main_config(monkeypatch, cfg):
    """Make `main.main` build `cfg` (a tiny, temp-dir config) instead of the defaults."""
    from dataclasses import asdict

    import main as cli

    values = asdict(cfg)

    class _TinyConfig(Config):
        def __init__(self):
            super().__init__()
            for key, value in values.items():
                setattr(self, key, copy.deepcopy(value))

    monkeypatch.setattr(cli, "Config", _TinyConfig)
    return cli


def _collect_via_main(monkeypatch, cfg, *extra):
    cli = _patch_main_config(monkeypatch, cfg)
    try:
        cli.main(["--stage", "ablation_collect", "--study", "B", "--model", "t-small", *extra])
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.fixture(scope="module")
def completed_study_b(tiny_processed_dir, tmp_path_factory):
    """One real full run of study B (tiny, 1 epoch x 2 steps) -- the shared source of arm
    files for the collection tests, which work on copies."""
    from training.ablation_runner import run_ablation

    out = tmp_path_factory.mktemp("full_b")
    cfg = _cfg(tiny_processed_dir, out)
    run_ablation(cfg, "t-small", "B")
    return cfg


def _copy_results(completed, tmp_path):
    """A config whose ablation results dir is a private copy of the completed run's, with
    the study-level outputs removed."""
    import shutil

    cfg = copy.deepcopy(completed)
    cfg.ablation_dir = str(tmp_path / "ablations")
    cfg.plot_dir = str(tmp_path / "plots")
    cfg.log_dir = str(tmp_path / "logs")
    shutil.copytree(completed.ablation_dir, cfg.ablation_dir)
    os.remove(os.path.join(cfg.ablation_dir, "seed_42", "B_t_small.json"))
    return cfg


def _study_json_path(cfg):
    return os.path.join(cfg.ablation_dir, "seed_42", "B_t_small.json")


def _plot_path(cfg):
    return os.path.join(cfg.plot_dir, "ablation_B_t_small.png")


def _arm_file(cfg, label):
    return os.path.join(cfg.ablation_dir, "seed_42", "arms", "B_t_small", f"{label}.json")


def _edit_arm(cfg, label, fn):
    path = _arm_file(cfg, label)
    record = json.load(open(path))
    fn(record)
    json.dump(record, open(path, "w"))


class TestCollect:
    """T10.4: assessment, `ablation_collect`, auto-collect (AC-34, AC-35 sequential, AC-36)."""

    def test_collect_reproduces_the_full_run(self, completed_study_b, tmp_path, monkeypatch):
        full = json.load(open(_study_json_path(completed_study_b)))
        cfg = _copy_results(completed_study_b, tmp_path)
        assert _collect_via_main(monkeypatch, cfg) == 0
        collected = json.load(open(_study_json_path(cfg)))
        strip = lambda rs: [{k: v for k, v in r.items() if k != "reused_from_previous_run"} for r in rs]  # noqa: E731
        assert strip(collected) == strip(full)
        assert [r["label"] for r in collected] == ["objective_conditioned", "input_only"]
        assert all(r["code_revision"]["source_hash"] for r in collected)
        assert os.path.exists(_plot_path(cfg))

    @pytest.mark.parametrize("case", ["missing", "failed", "stale"])
    def test_incomplete_study_is_refused_and_nothing_written(
        self, completed_study_b, tmp_path, monkeypatch, caplog, case,
    ):
        cfg = _copy_results(completed_study_b, tmp_path)
        if case == "missing":
            os.remove(_arm_file(cfg, "input_only"))
        elif case == "failed":
            _edit_arm(cfg, "input_only", lambda r: r.update(status="failed", error="KeyError: x"))
        else:
            _edit_arm(cfg, "input_only", lambda r: r["code_revision"].update(source_hash="0" * 16))
        # A failed arm exits ARM_FAILED_EXIT_CODE (70), like `--stage ablation` does.
        assert _collect_via_main(monkeypatch, cfg) == (70 if case == "failed" else 1)
        assert f"{case} ['input_only']" in caplog.text or f"{case} {['input_only']}" in caplog.text
        assert not os.path.exists(_study_json_path(cfg))
        assert not os.path.exists(_plot_path(cfg))

    @pytest.mark.parametrize("case", ["missing", "failed"])
    def test_allow_incomplete_assembles_with_placeholders(
        self, completed_study_b, tmp_path, monkeypatch, case,
    ):
        cfg = _copy_results(completed_study_b, tmp_path)
        if case == "missing":
            os.remove(_arm_file(cfg, "objective_conditioned"))
        else:
            _edit_arm(cfg, "objective_conditioned", lambda r: r.update(status="failed", error="KeyError: x"))
        # Assembled either way; a failed arm still exits 70 (it is an error record, not an
        # absence the caller chose to tolerate).
        expected = 70 if case == "failed" else 0
        assert _collect_via_main(monkeypatch, cfg, "--allow_incomplete") == expected
        study = json.load(open(_study_json_path(cfg)))
        assert [r["label"] for r in study] == ["objective_conditioned", "input_only"]
        assert study[0]["status"] == case
        assert study[1]["status"] == "ok"
        assert os.path.exists(_plot_path(cfg))  # drawn from the ok arm

    @pytest.mark.parametrize("field,edit", [
        ("runtime.device_name", lambda r: r["runtime"].update(device_name="NVIDIA L4")),
        ("resolved_config.batch_size", lambda r: r["resolved_config"].update(batch_size=999)),
    ])
    def test_inconsistent_study_is_refused_even_with_allow_incomplete(
        self, completed_study_b, tmp_path, monkeypatch, caplog, field, edit,
    ):
        cfg = _copy_results(completed_study_b, tmp_path)
        _edit_arm(cfg, "input_only", edit)
        for extra in ((), ("--allow_incomplete",)):
            assert _collect_via_main(monkeypatch, cfg, *extra) == 1
            assert f"{field} differs" in caplog.text
            assert not os.path.exists(_study_json_path(cfg))

    def test_overridden_fields_may_differ(self, completed_study_b, tmp_path):
        from training.ablation_runner import assess_study

        cfg = _copy_results(completed_study_b, tmp_path)
        assessment = assess_study(cfg, "t-small", "B")
        assert assessment.complete
        ok = assessment.ok
        assert ok["objective_conditioned"]["resolved_config"]["router_conditioning"] != \
            ok["input_only"]["resolved_config"]["router_conditioning"]

    def test_collect_all_writes_a_summary_marking_incomplete_studies(
        self, completed_study_b, tmp_path, monkeypatch,
    ):
        cfg = _copy_results(completed_study_b, tmp_path)
        cli = _patch_main_config(monkeypatch, cfg)
        with pytest.raises(SystemExit) as exc:
            cli.main(["--stage", "ablation_collect", "--study", "all", "--model", "t-small"])
        assert exc.value.code == 1
        summary = json.load(open(os.path.join(cfg.ablation_dir, "seed_42", "summary_t_small.json")))
        assert summary["B"]["num_ok"] == 2
        assert summary["A"]["complete"] is False and summary["A"]["missing"]

    def test_sequential_partial_runs(self, tiny_processed_dir, tmp_path, caplog):
        import logging

        from training.ablation_runner import run_ablation

        cfg = _cfg(tiny_processed_dir, tmp_path)
        with caplog.at_level(logging.INFO, logger="training.ablation_runner"):
            run_ablation(copy.deepcopy(cfg), "t-small", "B", arms=["objective_conditioned"])
        assert not os.path.exists(_study_json_path(cfg))
        assert not os.path.exists(_plot_path(cfg))
        assert "missing ['input_only']" in caplog.text
        assert "deferred" in caplog.text

        run_ablation(copy.deepcopy(cfg), "t-small", "B", arms=["input_only"])
        study = json.load(open(_study_json_path(cfg)))
        assert [r["label"] for r in study] == ["objective_conditioned", "input_only"]
        assert all(r["status"] == "ok" for r in study)
        assert os.path.exists(_plot_path(cfg))

    def test_collect_stage_resolves_every_name(self):
        """The REV-135 guard (tests/test_cli.py) for the new stage body."""
        import ast
        import builtins
        import inspect
        import textwrap

        import main as main_module

        fn = ast.parse(textwrap.dedent(inspect.getsource(main_module.run_ablation_collect_stage))).body[0]
        bound = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
        unresolved = {
            node.id for node in ast.walk(fn)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id not in bound and not hasattr(builtins, node.id)
            and not hasattr(main_module, node.id)
        }
        assert not unresolved, unresolved


class TestOutputIsolation:
    """T10.5 / AC-42: arm-scoped side outputs under --arms; atomic run_config.json."""

    @staticmethod
    def _mock_extras(monkeypatch):
        import training.federated as federated
        import training.local_train as local_train

        def personalization(cfg, model_name):
            mn = model_name.replace("-", "_")
            os.makedirs(cfg.log_dir, exist_ok=True)
            os.makedirs(cfg.plot_dir, exist_ok=True)
            json.dump({"gain": 0.0}, open(os.path.join(cfg.log_dir, f"personalization_gain_{mn}.json"), "w"))
            for name in (f"global_vs_personalized_{mn}.png", f"personalization_gain_histogram_{mn}.png"):
                open(os.path.join(cfg.plot_dir, name), "w").close()
            return {"num_users_evaluated": 1, "with_lora": {}, "without_lora": {},
                    "gain": {"mean": 0.0}, "per_user_scores": []}

        def federated_run(cfg, model_name):
            mn = model_name.replace("-", "_")
            os.makedirs(cfg.log_dir, exist_ok=True)
            json.dump([], open(os.path.join(cfg.log_dir, f"federated_rounds_{mn}.json"), "w"))
            return {"round_logs": [{"total_round_kb": 1.0, "upload_kb": 0.5}], "final_val_ndcg": 0.1}

        monkeypatch.setattr(local_train, "run_local_personalization", personalization)
        monkeypatch.setattr(federated, "run_federated", federated_run)

    def test_side_outputs_are_scoped_per_arm(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.ablation_runner as runner

        self._mock_extras(monkeypatch)
        cfg = _cfg(tiny_processed_dir, tmp_path)
        runner.run_ablation(copy.deepcopy(cfg), "t-small", "E", arms=["global_lora"])
        runner.run_ablation(copy.deepcopy(cfg), "t-small", "A", arms=["arv_mmoe_gradnorm_lora"])

        e_dir = os.path.join(cfg.log_dir, "arms", "E_global_lora")
        a_dir = os.path.join(cfg.log_dir, "arms", "A_arv_mmoe_gradnorm_lora")
        assert os.path.exists(os.path.join(e_dir, "personalization_gain_t_small.json"))
        assert os.path.exists(os.path.join(a_dir, "personalization_gain_t_small.json"))
        assert os.path.exists(os.path.join(a_dir, "federated_rounds_t_small.json"))
        assert os.path.exists(os.path.join(cfg.plot_dir, "arms", "E_global_lora",
                                           "global_vs_personalized_t_small.png"))
        assert not [f for f in os.listdir(cfg.log_dir) if f.startswith(("personalization_gain_", "federated_rounds_"))]

        # Without --arms the paths are as before.
        base = copy.deepcopy(cfg)
        arm = runner.resolve_study_configs("E", base)[1]
        assert arm._study_label == "global_lora"
        result = runner._run_single_config(arm, "t-small", "E", base, True)
        assert result["status"] == "ok" and result["shared_backbone_with"]
        assert os.path.exists(os.path.join(cfg.log_dir, "personalization_gain_t_small.json"))
        # The arm's recorded config keeps the study's dirs either way.
        persisted = json.load(open(runner._arm_path(cfg, "E", "t-small", "global_lora")))
        assert persisted["resolved_config"]["log_dir"] == cfg.log_dir

    def test_run_config_is_atomic_and_records_the_selection(self, tiny_processed_dir, tmp_path, monkeypatch):
        import utils

        cfg = _cfg(tiny_processed_dir, tmp_path)
        cli = _patch_main_config(monkeypatch, cfg)
        writes = []
        real = utils.atomic_write_json

        def spy(path, payload, *a, **k):
            writes.append(path)
            return real(path, payload, *a, **k)

        monkeypatch.setattr(utils, "atomic_write_json", spy)
        monkeypatch.setitem(cli.STAGE_MAP, "ablation", lambda config, args: None)
        cli.main(["--stage", "ablation", "--study", "B", "--arms", "input_only", "--model", "t-small"])
        path = os.path.join(cfg.log_dir, "run_config.json")
        assert path in writes
        manifest = json.load(open(path))
        assert manifest["selection"] == {"stage": "ablation", "study": "B", "arms": "input_only"}


# ---------------------------------------------------------------- real processes (T10.6/T10.7)

DRIVER = r'''
"""Test driver: one ablation process on the tiny fixture (specs/ablation-run D21)."""
import json, logging, os, sys, time
spec = json.load(open(sys.argv[1]))
sys.path.insert(0, spec["root"])
os.chdir(spec["root"])
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(process)d %(name)s %(levelname)s %(message)s")
from tests.conftest import tiny_config
import training.ablation_runner as runner
import training.pretrain as pretrain

patches = spec.get("patches", [])
if "fast_lock_poll" in patches:
    runner.BACKBONE_LOCK_POLL_SECONDS = 0.2
if "hold_after_epoch0" in patches:
    real_save = pretrain.save_checkpoint
    def held_save(path, epoch, *a, **k):
        real_save(path, epoch, *a, **k)
        if os.path.basename(path) == "last.pt" and epoch == 0:
            open(spec["held_marker"], "w").write(str(os.getpid()))
            deadline = time.time() + 120
            while os.path.exists(spec["hold_sentinel"]) and time.time() < deadline:
                time.sleep(0.05)
    pretrain.save_checkpoint = held_save
if "trace_writes" in patches:
    real_write = runner._write_study_outputs_unlocked
    def traced(*a, **k):
        with open(spec["trace"], "a") as f:
            f.write(f"start {os.getpid()} {time.monotonic()}\n")
        time.sleep(0.5)  # widen the window an unlocked writer would overlap in
        out = real_write(*a, **k)
        with open(spec["trace"], "a") as f:
            f.write(f"end {os.getpid()} {time.monotonic()}\n")
        return out
    runner._write_study_outputs_unlocked = traced

cfg = tiny_config(spec["processed"], spec["out"], **spec.get("overrides", {}))
if spec["action"] == "ablation":
    results = runner.run_ablation(cfg, "t-small", spec["study"], arms=spec.get("arms"))
    json.dump(results, open(spec["result"], "w"), default=str)
else:
    assessment = runner.collect_study(cfg, "t-small", spec["study"])
    sys.exit(0 if assessment.written else 1)
'''

WAIT_SECONDS = 120


def _wait_for(predicate, what, procs=(), timeout=WAIT_SECONDS):
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        for proc in procs:
            assert proc.poll() is None or predicate(), f"a process exited while waiting for {what}"
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(0.1)


class _Procs:
    """Starts driver processes; every one is killed (and reaped) on exit."""

    def __init__(self, tmp_path, processed, out):
        self.tmp, self.processed, self.out = tmp_path, processed, str(out)
        self.driver = tmp_path / "driver.py"
        self.driver.write_text(DRIVER)
        self.procs = []

    def start(self, name, **spec):
        import subprocess
        import sys

        spec = {"root": ROOT, "processed": self.processed, "out": self.out,
                "result": str(self.tmp / f"{name}.result.json"), **spec}
        spec_path = self.tmp / f"{name}.spec.json"
        spec_path.write_text(json.dumps(spec))
        log = open(self.tmp / f"{name}.log", "w")
        # The venv's interpreter directly (what `uv run python` resolves to): a `uv run`
        # wrapper would survive a SIGKILL aimed at the Python process and vice versa.
        proc = subprocess.Popen(
            [sys.executable, str(self.driver), str(spec_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        proc.log_path = self.tmp / f"{name}.log"
        self.procs.append(proc)
        return proc

    def log(self, proc):
        return proc.log_path.read_text()

    def wait(self, proc, timeout=WAIT_SECONDS):
        return proc.wait(timeout=timeout)

    def close(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture
def procs(tiny_processed_dir, tmp_path):
    import subprocess

    runner = _Procs(tmp_path, tiny_processed_dir(), tmp_path / "out")
    try:
        yield runner
    finally:
        runner.close()
        leftovers = subprocess.run(
            ["pgrep", "-f", str(runner.driver)], capture_output=True, text=True,
        ).stdout.split()
        assert not leftovers, f"driver processes survived: {leftovers}"


TINY_RUN = {"pretrain_epochs": 1, "max_steps_per_epoch": 2, "ablation_eval_examples": 32}


class TestConcurrentRuns:
    """T10.6 / AC-35, AC-37, AC-40."""

    def test_concurrent_partial_runs_assemble_one_study(self, procs):
        p1 = procs.start("p1", action="ablation", study="B", arms=["objective_conditioned"],
                         overrides=TINY_RUN)
        p2 = procs.start("p2", action="ablation", study="B", arms=["input_only"],
                         overrides=TINY_RUN)
        assert procs.wait(p1) == 0, procs.log(p1)[-3000:]
        assert procs.wait(p2) == 0, procs.log(p2)[-3000:]
        cfg = tiny_config(procs.processed, procs.out)
        study = json.load(open(_study_json_path(cfg)))
        assert [r["label"] for r in study] == ["objective_conditioned", "input_only"]
        assert all(r["status"] == "ok" for r in study)
        assert os.path.exists(_plot_path(cfg))
        # Exactly one of them found the study complete and wrote it.
        written = [p for p in (p1, p2) if "study JSON and plots written" in procs.log(p)]
        assert len(written) >= 1

    def test_concurrent_collectors_serialise_and_agree(self, procs, completed_study_b, tmp_path):
        import shutil

        shutil.copytree(completed_study_b.ablation_dir, os.path.join(procs.out, "ablations"))
        os.remove(os.path.join(procs.out, "ablations", "seed_42", "B_t_small.json"))
        trace = str(tmp_path / "trace.txt")
        collectors = [
            procs.start(f"c{i}", action="collect", study="B", overrides=TINY_RUN,
                        patches=["trace_writes"], trace=trace)
            for i in range(2)
        ]
        for proc in collectors:
            assert procs.wait(proc) == 0, procs.log(proc)[-3000:]
        events = [line.split() for line in open(trace)]
        assert [e[0] for e in events] == ["start", "end", "start", "end"], events
        assert events[0][1] == events[1][1] and events[2][1] == events[3][1]
        assert float(events[1][2]) <= float(events[2][2]), "the two writes overlapped"

        cfg = tiny_config(procs.processed, procs.out)
        concurrent = json.load(open(_study_json_path(cfg)))
        from training.ablation_runner import collect_study

        single = copy.deepcopy(cfg)
        single.ablation_dir = str(tmp_path / "single")
        single.plot_dir = str(tmp_path / "single_plots")
        shutil.copytree(os.path.join(procs.out, "ablations"), single.ablation_dir)
        os.remove(_study_json_path(single))
        assert collect_study(single, "t-small", "B").written
        assert json.load(open(_study_json_path(single))) == concurrent

    def test_oom_under_arms_exits_75_then_reuses_and_resumes(
        self, tiny_processed_dir, tmp_path, monkeypatch, caplog,
    ):
        """AC-40 (in-process): OOM in the second arm's epoch 1 -> exit 75; the re-run reuses
        the first arm and resumes the second at epoch 1."""
        import logging

        import torch

        import training.pretrain as pretrain

        cfg = _cfg(tiny_processed_dir, tmp_path, pretrain_epochs=2)
        argv = ["--stage", "ablation", "--study", "B", "--arms",
                "objective_conditioned,input_only", "--model", "t-small"]
        steps = {"n": 0}
        real_step = pretrain.train_one_step

        def oom_in_second_arm(*args, **kwargs):
            steps["n"] += 1
            # arm 1: steps 1-4; arm 2: steps 5-8 -> step 7 is its epoch 1, first step.
            if steps["n"] == 7:
                raise torch.OutOfMemoryError("CUDA out of memory (injected)")
            return real_step(*args, **kwargs)

        monkeypatch.setattr(pretrain, "train_one_step", oom_in_second_arm)
        cli = _patch_main_config(monkeypatch, cfg)
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 75
        arms_dir = os.path.join(cfg.ablation_dir, "seed_42", "arms", "B_t_small")
        assert sorted(os.listdir(arms_dir)) == ["objective_conditioned.json"]

        monkeypatch.setattr(pretrain, "train_one_step", real_step)
        counts = _arm_persistence_spies(monkeypatch)
        steps["n"] = 0

        def counting(*args, **kwargs):
            steps["n"] += 1
            return real_step(*args, **kwargs)

        monkeypatch.setattr(pretrain, "train_one_step", counting)
        with caplog.at_level(logging.INFO):
            cli.main(argv)
        assert counts == {"run_pretrain": 1, "evaluate_model": 1}
        assert steps["n"] == 2, "only the second arm's remaining epoch is trained"
        assert "at epoch 1 (global step 2)" in caplog.text
        study = json.load(open(_study_json_path(cfg)))
        assert [r["status"] for r in study] == ["ok", "ok"]
        assert study[0].get("reused_from_previous_run") is None  # assembled from arm files


def _arm_persistence_spies(monkeypatch):
    from tests.test_ablation_robustness import TestArmPersistence

    return TestArmPersistence._spies(monkeypatch)


class TestSharedFingerprintAcrossProcesses:
    """T10.7 / AC-38, AC-39: B:objective_conditioned and C:arv_mmoe share a backbone."""

    HOLD_RUN = {"pretrain_epochs": 2, "max_steps_per_epoch": 2, "ablation_eval_examples": 32}

    def _shared_fingerprint(self, procs):
        from training.ablation_runner import (
            _training_fingerprint, resolve_dataset_fields, select_study_configs,
        )

        cfg = tiny_config(procs.processed, procs.out, **self.HOLD_RUN)
        fps = []
        for study, label in (("B", "objective_conditioned"), ("C", "arv_mmoe")):
            arm = select_study_configs(study, cfg, [label])[0]
            resolve_dataset_fields(arm)
            fps.append(_training_fingerprint(arm))
        assert fps[0] == fps[1], "the two arms must share a training fingerprint"
        return fps[0]

    def _start_holder(self, procs, tmp_path):
        held, sentinel = tmp_path / "held", tmp_path / "hold"
        sentinel.write_text("hold")
        p1 = procs.start("p1", action="ablation", study="B", arms=["objective_conditioned"],
                         overrides=self.HOLD_RUN, patches=["hold_after_epoch0"],
                         held_marker=str(held), hold_sentinel=str(sentinel))
        _wait_for(held.exists, "P1 to finish epoch 0 inside the lock", [p1])
        return p1, sentinel

    def _start_waiter(self, procs, p1):
        p2 = procs.start("p2", action="ablation", study="C", arms=["arv_mmoe"],
                         overrides=self.HOLD_RUN, patches=["fast_lock_poll"])
        _wait_for(
            lambda: f"(pid host time: {p1.pid} " in procs.log(p2),
            "P2 to log that it waits on P1", [p2],
        )
        return p2

    def test_waiter_reuses_the_holders_backbone(self, procs, tmp_path):
        import glob

        from training.ablation_runner import backbone_store_dir

        fp = self._shared_fingerprint(procs)
        p1, sentinel = self._start_holder(procs, tmp_path)
        lock = os.path.join(backbone_store_dir(os.path.join(procs.out, "checkpoints"), fp), ".train.lock")
        assert open(lock).read().split()[0] == str(p1.pid)
        p2 = self._start_waiter(procs, p1)
        assert p2.poll() is None
        sentinel.unlink()
        assert procs.wait(p1) == 0, procs.log(p1)[-3000:]
        assert procs.wait(p2) == 0, procs.log(p2)[-3000:]

        result = json.load(open(tmp_path / "p2.result.json"))["C"][0]
        assert result["status"] == "ok"
        assert result["shared_backbone_with"] == "B_objective_conditioned"
        assert "training_log_dir" not in result
        assert "Resuming" not in procs.log(p2) and "Epoch 0:" not in procs.log(p2)
        logs = glob.glob(os.path.join(procs.out, "checkpoints", "ablations", "by_fingerprint", "*", "logs", "pretrain_t_small.log"))
        assert len(logs) == 1

    def test_waiter_takes_over_a_killed_holder_by_resuming(self, procs, tmp_path):
        import signal

        self._shared_fingerprint(procs)
        p1, _sentinel = self._start_holder(procs, tmp_path)
        p2 = self._start_waiter(procs, p1)
        p1.send_signal(signal.SIGKILL)
        p1.wait(timeout=WAIT_SECONDS)
        assert procs.wait(p2) == 0, procs.log(p2)[-3000:]
        log = procs.log(p2)
        assert "Resuming t-small from" in log and "at epoch 1 (global step 2)" in log
        result = json.load(open(tmp_path / "p2.result.json"))["C"][0]
        assert result["status"] == "ok"
        assert result["training_log_dir"]
        from training.ablation_runner import find_backbone

        assert find_backbone(os.path.join(procs.out, "checkpoints"), result["training_fingerprint"])
