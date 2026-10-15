"""specs/ablation-run: robustness of a study run (FR-7 checks, FR-9..FR-11, FR-13)."""
import copy
import os

import pytest

from configs.config import Config
from training.ablation_runner import _training_fingerprint


class TestFingerprint:
    """AC-29: runtime-only fields leave the fingerprint; training-relevant ones change it."""

    @pytest.mark.parametrize("field,value", [
        ("num_workers", 4),
        ("pin_memory", False),
        ("resume_training", False),
    ])
    def test_runtime_only_fields_do_not_change_it(self, field, value):
        base = Config()
        other = copy.deepcopy(base)
        assert getattr(other, field) != value
        setattr(other, field, value)
        assert _training_fingerprint(other) == _training_fingerprint(base)

    @pytest.mark.parametrize("field,value", [
        ("persistent_workers", True),
        ("min_lr_ratio", 0.0),
        ("user_sample_fraction", 0.1),
        ("user_sample_seed", 7),
        ("attention_backend", "sdpa"),
    ])
    def test_training_relevant_fields_change_it(self, field, value):
        base = Config()
        other = copy.deepcopy(base)
        assert getattr(other, field) != value
        setattr(other, field, value)
        assert _training_fingerprint(other) != _training_fingerprint(base)


class TestProcessedDirCheck:
    """specs/ablation-run T2.5 / AC-20: a mismatched user sample fails before training."""

    @staticmethod
    def _spy_steps(monkeypatch):
        import training.pretrain as pretrain

        calls = {"n": 0}
        real = pretrain.train_one_step

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(pretrain, "train_one_step", counting)
        return calls

    @pytest.mark.parametrize("fraction,seed", [(1.0, 42), (0.5, 7)])
    def test_pretrain_and_ablation_refuse_a_mismatched_sample(
        self, tiny_processed_dir, tmp_path, monkeypatch, fraction, seed,
    ):
        from tests.conftest import tiny_config
        from training.ablation_runner import run_ablation
        from training.pretrain import run_pretrain

        processed = tiny_processed_dir(user_sample_fraction=0.5)
        cfg = tiny_config(
            processed, str(tmp_path), user_sample_fraction=fraction, user_sample_seed=seed,
        )
        calls = self._spy_steps(monkeypatch)
        with pytest.raises(ValueError, match="--stage preprocess"):
            run_pretrain(copy.deepcopy(cfg), "t-small")
        with pytest.raises(ValueError, match="--stage preprocess"):
            run_ablation(copy.deepcopy(cfg), "t-small", "B")
        assert calls["n"] == 0

    def test_matching_sample_is_accepted(self, tiny_processed_dir, tmp_path):
        from data.preprocessing import check_processed_dir
        from tests.conftest import tiny_config

        cfg = tiny_config(
            tiny_processed_dir(user_sample_fraction=0.5), str(tmp_path),
            user_sample_fraction=0.5,
        )
        check_processed_dir(cfg)

    def test_missing_manifest_warns_and_proceeds(self, tiny_processed_dir, tmp_path, caplog):
        import logging
        import shutil

        from tests.conftest import tiny_config
        from training.pretrain import run_pretrain

        processed = str(tmp_path / "processed")
        shutil.copytree(tiny_processed_dir(), processed)
        os.remove(os.path.join(processed, "preprocessing_manifest.json"))
        cfg = tiny_config(
            processed, str(tmp_path / "out"), pretrain_epochs=1, max_steps_per_epoch=1,
        )
        with caplog.at_level(logging.WARNING):
            result = run_pretrain(cfg, "t-small")
        assert "preprocessing_manifest.json not found" in caplog.text
        assert result["best_epoch"] == 0


class TestRuntimeRecord:
    """specs/ablation-run T4.3 / AC-30: each arm records the backend and device it ran on."""

    def test_arm_result_carries_runtime(self, tiny_processed_dir, tmp_path):
        from tests.conftest import tiny_config
        from training.ablation_runner import _run_single_config, resolve_study_configs

        base = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=1, max_steps_per_epoch=2,
        )
        arm = resolve_study_configs("B", base)[0]
        result = _run_single_config(arm, "t-small", "B", base, True)
        assert result["status"] == "ok"
        runtime = result["runtime"]
        assert runtime["attention_backend"] == "sdpa"
        assert runtime["device"] == "cpu"
        assert runtime["device_name"]
        assert runtime["dtype"] == "bf16" and runtime["mixed_precision"] is False


class TestOOMHandling:
    """specs/ablation-run T7.1 / AC-24."""

    @pytest.mark.parametrize("exc,expected", [
        (lambda: __import__("torch").OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"), True),
        (lambda: RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"), True),
        (lambda: RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"), True),
        (lambda: RuntimeError("expected scalar type Float but found BFloat16"), False),
        (lambda: MemoryError(), False),
        (lambda: ValueError("out of memory"), False),
    ])
    def test_predicate(self, exc, expected):
        from utils import is_oom_error

        assert is_oom_error(exc()) is expected

    def test_exit_code_constant(self):
        from utils import OOM_EXIT_CODE

        assert OOM_EXIT_CODE == 75

    def _main_with_stage(self, monkeypatch, raiser):
        import main as cli

        monkeypatch.setitem(cli.STAGE_MAP, "pretrain", raiser)
        monkeypatch.setattr(cli, "save_config_snapshot", lambda config: None)
        return cli.main(["--stage", "pretrain", "--model", "t-small"])

    def test_main_exits_75_on_oom(self, monkeypatch):
        import torch

        def oom(config, args):
            raise torch.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(SystemExit) as exc:
            self._main_with_stage(monkeypatch, oom)
        assert exc.value.code == 75

    def test_main_reraises_other_errors(self, monkeypatch):
        def bad(config, args):
            raise ValueError("not an OOM")

        with pytest.raises(ValueError, match="not an OOM"):
            self._main_with_stage(monkeypatch, bad)

    def _run_study_d(self, tiny_processed_dir, tmp_path, monkeypatch, failure):
        import training.ablation_runner as runner
        from tests.conftest import tiny_config

        calls = []

        def fake(cfg, model_name, study, base, train):
            calls.append(cfg._study_label)
            if len(calls) == 2:
                raise failure
            return {"label": cfg._study_label, "study": study, "status": "ok"}

        monkeypatch.setattr(runner, "_run_single_config", fake)
        cfg = tiny_config(tiny_processed_dir(), str(tmp_path))
        return runner.run_ablation(cfg, "t-small", "D"), calls

    def test_run_ablation_propagates_an_arm_oom(self, tiny_processed_dir, tmp_path, monkeypatch):
        import torch

        with pytest.raises(torch.OutOfMemoryError):
            self._run_study_d(
                tiny_processed_dir, tmp_path, monkeypatch, torch.OutOfMemoryError("CUDA out of memory"),
            )

    def test_run_ablation_records_other_arm_failures(self, tiny_processed_dir, tmp_path, monkeypatch):
        results, calls = self._run_study_d(
            tiny_processed_dir, tmp_path, monkeypatch, KeyError("missing"),
        )
        assert len(calls) == 4, "arm 3 (and 4) must still run after a non-OOM failure"
        assert results["D"][1]["status"] == "failed"
        assert "KeyError" in results["D"][1]["error"]
        assert results["D"][2]["status"] == "ok"


class TestBackboneLock:
    """specs/ablation-run T7.3 / AC-26."""

    HOLDER = (
        "import os, sys, time\n"
        "from training.ablation_runner import backbone_lock\n"
        "root, fp, ready, release = sys.argv[1:5]\n"
        "with backbone_lock(root, fp):\n"
        "    open(ready, 'w').write(str(os.getpid()))\n"
        "    deadline = time.time() + 120\n"
        "    while not os.path.exists(release) and time.time() < deadline:\n"
        "        time.sleep(0.05)\n"
    )

    def _start_holder(self, root, fingerprint, tmp_path):
        import subprocess
        import sys
        import time

        ready, release = tmp_path / "ready", tmp_path / "release"
        proc = subprocess.Popen(
            [sys.executable, "-c", self.HOLDER, root, fingerprint, str(ready), str(release)],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        deadline = time.monotonic() + 60
        while not ready.exists() or not ready.read_text():
            assert proc.poll() is None, "lock holder exited early"
            assert time.monotonic() < deadline, "lock holder never acquired the lock"
            time.sleep(0.05)
        return proc, release

    def test_waiter_times_out_then_reuses_the_holders_backbone(
        self, tiny_processed_dir, tmp_path, monkeypatch,
    ):
        import threading

        import training.ablation_runner as runner
        import training.pretrain as pretrain
        from tests.conftest import tiny_config

        processed = tiny_processed_dir()
        # Train one real backbone (root A) so there is a checkpoint to share.
        trained_base = tiny_config(
            processed, str(tmp_path / "a"), pretrain_epochs=1, max_steps_per_epoch=2,
        )
        arm_a = runner.resolve_study_configs("B", trained_base)[0]
        first = runner._run_single_config(arm_a, "t-small", "B", trained_base, True)
        fingerprint = first["training_fingerprint"]

        base = tiny_config(
            processed, str(tmp_path / "b"), pretrain_epochs=1, max_steps_per_epoch=2,
        )
        root = base.checkpoint_dir
        proc, release = self._start_holder(root, fingerprint, tmp_path)
        try:
            lock_file = os.path.join(runner.backbone_store_dir(root, fingerprint), ".train.lock")
            assert open(lock_file).read().split()[0] == str(proc.pid)

            with pytest.raises(TimeoutError):
                with runner.backbone_lock(root, fingerprint, timeout=0.5, poll=0.05):
                    pass

            def must_not_train(*args, **kwargs):
                raise AssertionError("run_pretrain was called although a backbone was registered")

            monkeypatch.setattr(pretrain, "run_pretrain", must_not_train)
            monkeypatch.setattr(runner, "BACKBONE_LOCK_POLL_SECONDS", 0.05)
            arm_b = runner.resolve_study_configs("B", base)[0]
            outcome = {}

            def waiter():
                try:
                    outcome["result"] = runner._run_single_config(arm_b, "t-small", "B", base, True)
                except BaseException as exc:  # noqa: BLE001 - surfaced below
                    outcome["error"] = exc

            thread = threading.Thread(target=waiter)
            thread.start()
            thread.join(timeout=1.0)
            assert thread.is_alive(), f"the waiter did not block on the held lock: {outcome}"

            # The holder finishes: registers its backbone, then releases the lock.
            trained_cfg = runner.resolve_study_configs("B", trained_base)[0]
            trained_cfg.checkpoint_dir = runner.backbone_store_dir(trained_base.checkpoint_dir, fingerprint)
            runner.register_backbone(root, fingerprint, trained_cfg, "holder_process")
            release.write_text("go")
            thread.join(timeout=120)
            assert not thread.is_alive()
        finally:
            release.write_text("go")
            proc.wait(timeout=60)

        assert "error" not in outcome, outcome.get("error")
        result = outcome["result"]
        assert result["status"] == "ok"
        assert result["shared_backbone_with"] == "holder_process"
        assert result["training_fingerprint"] == fingerprint

    def test_lock_dies_with_its_process(self, tmp_path):
        """A killed holder never leaves a stale lock (flock is released by the kernel)."""
        import signal

        import training.ablation_runner as runner

        root = str(tmp_path / "ckpt")
        proc, _ = self._start_holder(root, "deadbeef0000", tmp_path)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=60)
        with runner.backbone_lock(root, "deadbeef0000", timeout=2.0, poll=0.05):
            pass


class TestArmPersistence:
    """specs/ablation-run T7.4 / AC-27."""

    @staticmethod
    def _spies(monkeypatch):
        import evaluation.metrics as metrics
        import training.pretrain as pretrain

        counts = {"run_pretrain": 0, "evaluate_model": 0}
        real_train, real_eval = pretrain.run_pretrain, metrics.evaluate_model

        def train_spy(*a, **k):
            counts["run_pretrain"] += 1
            return real_train(*a, **k)

        def eval_spy(*a, **k):
            counts["evaluate_model"] += 1
            return real_eval(*a, **k)

        monkeypatch.setattr(pretrain, "run_pretrain", train_spy)
        monkeypatch.setattr(metrics, "evaluate_model", eval_spy)
        return counts

    @staticmethod
    def _study_json(cfg):
        import json

        with open(os.path.join(cfg.ablation_dir, "seed_42", "B_t_small.json")) as f:
            return json.load(f)

    @staticmethod
    def _strip(results):
        return [{k: v for k, v in r.items() if k != "reused_from_previous_run"} for r in results]

    def test_rerun_reuses_completed_arms(self, tiny_processed_dir, tmp_path, monkeypatch):
        from tests.conftest import tiny_config
        from training.ablation_runner import run_ablation

        def make(**overrides):
            pinned = dict(pretrain_epochs=1, max_steps_per_epoch=2, ablation_eval_examples=32)
            pinned.update(overrides)
            return tiny_config(tiny_processed_dir(), str(tmp_path), **pinned)

        counts = self._spies(monkeypatch)
        cfg = make()
        run_ablation(cfg, "t-small", "B")
        assert counts == {"run_pretrain": 2, "evaluate_model": 2}
        first = self._study_json(cfg)
        arms_dir = os.path.join(cfg.ablation_dir, "seed_42", "arms", "B_t_small")
        assert sorted(os.listdir(arms_dir)) == ["input_only.json", "objective_conditioned.json"]
        assert all(r["status"] == "ok" and r["arm_key"] for r in first)

        # Re-run of the same study: nothing is trained or evaluated again.
        counts.update(run_pretrain=0, evaluate_model=0)
        run_ablation(make(), "t-small", "B")
        assert counts == {"run_pretrain": 0, "evaluate_model": 0}
        second = self._study_json(cfg)
        assert all(r["reused_from_previous_run"] is True for r in second)
        assert self._strip(second) == first

        # A post-training field changes the arm key -> evaluation re-runs, training is
        # served by the backbone store.
        counts.update(run_pretrain=0, evaluate_model=0)
        run_ablation(make(ablation_eval_examples=16), "t-small", "B")
        assert counts == {"run_pretrain": 0, "evaluate_model": 2}
        third = self._study_json(cfg)
        assert all(r["num_eval_examples"] == 16 for r in third)
        assert not any(r.get("reused_from_previous_run") for r in third)

        # --no_resume re-runs every arm's evaluation (store reuse is unaffected).
        counts.update(run_pretrain=0, evaluate_model=0)
        run_ablation(make(ablation_eval_examples=16, resume_training=False), "t-small", "B")
        assert counts == {"run_pretrain": 0, "evaluate_model": 2}

    def test_source_revision_change_reruns_the_arm(self, tiny_processed_dir, tmp_path, monkeypatch):
        import utils
        from tests.conftest import tiny_config
        from training.ablation_runner import _arm_key

        cfg = tiny_config(tiny_processed_dir(), str(tmp_path))
        before = _arm_key(cfg, "B", "input_only", "t-small")
        real = utils.get_source_revision
        monkeypatch.setattr(
            utils, "get_source_revision", lambda *a, **k: {**real(), "source_hash": "ffff"},
        )
        assert _arm_key(cfg, "B", "input_only", "t-small") != before

    def test_results_are_seed_namespaced(self, tmp_path):
        from training.ablation_runner import _results_root

        cfg = Config()
        cfg.ablation_dir = str(tmp_path)
        cfg.seed = 123
        assert _results_root(cfg) == os.path.join(str(tmp_path), "seed_123")


class TestPerArmTrainingLogs:
    """Review REV-002: each trained arm logs into its own fingerprint's store directory."""

    def test_arms_write_separate_training_logs(self, tiny_processed_dir, tmp_path):
        import json

        from tests.conftest import tiny_config
        from training.ablation_runner import run_ablation, training_log_dir

        cfg = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=2, max_steps_per_epoch=2,
        )
        results = run_ablation(cfg, "t-small", "B")["B"]
        fingerprints = [r["training_fingerprint"] for r in results]
        assert len(set(fingerprints)) == 2

        for r in results:
            log_dir = training_log_dir(cfg.checkpoint_dir, r["training_fingerprint"])
            assert r["training_log_dir"] == log_dir
            lines = [l for l in open(os.path.join(log_dir, "pretrain_t_small.log")) if "epoch=" in l]
            assert [l.split("epoch=")[1].split(" ")[0] for l in lines] == ["0", "1"], r["label"]
            steps = [json.loads(l)["step"] for l in open(os.path.join(log_dir, "gradnorm_t_small.jsonl"))]
            assert steps == [0], r["label"]
        # Nothing is written to the shared per-model files any more.
        assert not os.path.exists(os.path.join(cfg.log_dir, "pretrain_t_small.log"))
        assert not os.path.exists(os.path.join(cfg.log_dir, "gradnorm_t_small.jsonl"))
        # The arm's own config keeps the study's log dir for its post-training reports.
        stored = json.load(open(os.path.join(cfg.ablation_dir, "seed_42", "B_t_small.json")))
        assert all(r["resolved_config"]["log_dir"] == cfg.log_dir for r in stored)


class TestSingleArmRerun:
    """Review REV-004: changing ONE arm's config re-runs only that arm."""

    def test_only_the_changed_arm_reruns(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.ablation_runner as runner
        from tests.conftest import tiny_config

        cfg = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=1, max_steps_per_epoch=2,
            ablation_eval_examples=32,
        )
        runner.run_ablation(cfg, "t-small", "B")

        real_resolve = runner.resolve_study_configs

        def change_input_only(study, base):
            arms = real_resolve(study, base)
            for arm in arms:
                if arm._study_label == "input_only":
                    arm.ablation_eval_examples = 16  # post-training field: no retraining
            return arms

        monkeypatch.setattr(runner, "resolve_study_configs", change_input_only)
        counts = TestArmPersistence._spies(monkeypatch)
        results = {
            r["label"]: r for r in runner.run_ablation(
                tiny_config(tiny_processed_dir(), str(tmp_path), pretrain_epochs=1,
                            max_steps_per_epoch=2, ablation_eval_examples=32),
                "t-small", "B",
            )["B"]
        }
        assert counts == {"run_pretrain": 0, "evaluate_model": 1}
        assert results["objective_conditioned"].get("reused_from_previous_run") is True
        assert not results["input_only"].get("reused_from_previous_run")
        assert results["input_only"]["num_eval_examples"] == 16
        assert results["objective_conditioned"]["num_eval_examples"] == 32
