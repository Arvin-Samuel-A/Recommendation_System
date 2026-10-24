"""specs/ablation-run FR-8: atomic checkpoints, `last.pt`, and exact epoch-granular resume."""
import glob
import os

import pytest
import torch

from tests.conftest import tiny_config

RESUME_KEYS = {
    "format", "epoch", "global_step", "total_steps", "training_fingerprint",
    "best_val_ndcg", "best_epoch", "epoch_mean_losses", "epoch_loss_windows", "scaler",
    "rng", "loader_generators",
}
RNG_KEYS = {"python", "torch_mask", "torch_cpu", "torch_cuda", "torch_mps"}


def _resume_config(processed, out_dir, **overrides):
    pinned = dict(
        pretrain_epochs=3, max_steps_per_epoch=4, batch_size=16, num_workers=0,
        use_contrastive=True, use_gradnorm=True, device="cpu", seed=42,
        # GradNorm lambdas are logged every `100` steps; the whole run is 12 steps, so
        # step 0 is the only record -- enough to see duplication.
    )
    # Same raw dir for every run: it is not a path field of the fingerprint, and these
    # tests compare runs whose output dirs differ.
    pinned["raw_data_dir"] = os.path.join(os.path.dirname(processed), "raw")
    pinned.update(overrides)
    return tiny_config(processed, out_dir, **pinned)


class TestLastCheckpoint:
    """T6.1 / AC-23."""

    @pytest.fixture
    def finished_two_epochs(self, tiny_processed_dir, tmp_path):
        from training.pretrain import run_pretrain

        cfg = _resume_config(tiny_processed_dir(), str(tmp_path), pretrain_epochs=2)
        run_pretrain(cfg, "t-small")
        return cfg

    def test_last_pt_holds_the_full_resume_state(self, finished_two_epochs):
        cfg = finished_two_epochs
        path = os.path.join(cfg.model_checkpoint_dir(), "last.pt")
        assert os.path.exists(path)
        ckpt = torch.load(path, weights_only=False)
        state = ckpt["resume_state"]
        assert RESUME_KEYS <= set(state), RESUME_KEYS - set(state)
        assert RNG_KEYS <= set(state["rng"])
        assert set(state["loader_generators"]) == {"train", "diag"}
        assert state["epoch"] == 1
        assert state["global_step"] == 8 and state["total_steps"] == 8
        for key in ("model_state_dict", "moe_state_dict", "optimizer_state_dict",
                    "scheduler_state_dict", "gradnorm_state_dict", "loss_fns_state_dict",
                    "source_revision", "config"):
            assert key in ckpt, key
        assert not glob.glob(os.path.join(cfg.model_checkpoint_dir(), "*.tmp-*"))

    def test_a_crash_mid_write_leaves_the_previous_checkpoints_intact(
        self, finished_two_epochs, monkeypatch,
    ):
        import utils
        from training import pretrain

        cfg = finished_two_epochs
        ckpt_dir = cfg.model_checkpoint_dir()
        before = {
            name: torch.load(os.path.join(ckpt_dir, name), weights_only=False)
            for name in ("last.pt", "best.pt")
        }
        sizes = {name: os.path.getsize(os.path.join(ckpt_dir, name)) for name in before}

        real_save = torch.save

        def partial_then_crash(obj, f, *args, **kwargs):
            f.write(b"PARTIAL-CHECKPOINT-BYTES")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(torch, "save", partial_then_crash)
        for name in ("last.pt", "best.pt"):
            with pytest.raises(RuntimeError, match="simulated crash"):
                utils.atomic_torch_save({"garbage": 1}, os.path.join(ckpt_dir, name))
        with pytest.raises(RuntimeError, match="simulated crash"):
            # Through the real writer as well, not only the helper.
            model, moe = pretrain.build_model(cfg)
            opt = torch.optim.AdamW(model.parameters())
            pretrain.save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"), 9, model, moe, opt,
                pretrain.build_scheduler(opt, cfg, 10), None, {}, cfg,
            )
        monkeypatch.setattr(torch, "save", real_save)

        for name, old in before.items():
            path = os.path.join(ckpt_dir, name)
            assert os.path.getsize(path) == sizes[name], f"{name} was modified"
            reloaded = torch.load(path, weights_only=False)
            assert reloaded["epoch"] == old["epoch"]
            for key, tensor in old["model_state_dict"].items():
                assert torch.equal(reloaded["model_state_dict"][key], tensor), key
        assert not glob.glob(os.path.join(ckpt_dir, "*.tmp-*")), "a temp file was left behind"


# ---------------------------------------------------------------- T6.2 (AC-21, AC-22)


class _Injected(RuntimeError):
    pass


def _traced_run(cfg, monkeypatch, fail_at_call=None):
    """`run_pretrain` through a spy on `train_one_step`.

    Returns (result_or_None, trace). `trace` holds per-call LRs (after the step), the
    scheduler position seen at entry, and the trained objects for hashing.
    """
    import hashlib

    import training.pretrain as pretrain
    from utils import set_seed

    trace = {"lrs": [], "entry_last_epoch": [], "objects": None, "calls": 0}
    real_step = pretrain.train_one_step

    def spy(model, moe, batch, config, loss_fns, gradnorm, optimizer, scheduler, *a, **kw):
        trace["calls"] += 1
        trace["objects"] = (model, moe, loss_fns, gradnorm)
        if fail_at_call is not None and trace["calls"] == fail_at_call:
            raise _Injected(f"injected failure at step call {fail_at_call}")
        trace["entry_last_epoch"].append(scheduler.last_epoch)
        out = real_step(model, moe, batch, config, loss_fns, gradnorm, optimizer, scheduler, *a, **kw)
        trace["lrs"].append(float(optimizer.param_groups[0]["lr"]))
        return out

    monkeypatch.setattr(pretrain, "train_one_step", spy)
    set_seed(cfg.seed)
    try:
        result = pretrain.run_pretrain(cfg, "t-small")
    except _Injected:
        result = None
    finally:
        monkeypatch.setattr(pretrain, "train_one_step", real_step)

    def param_hash(objects):
        model, moe, loss_fns, gradnorm = objects
        digest = hashlib.sha256()
        for state in [model.state_dict(), moe.state_dict()] + [
            fn.state_dict() for fn in loss_fns.values()
        ]:
            for name, t in state.items():
                digest.update(name.encode())
                digest.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        if gradnorm is not None:
            digest.update(gradnorm.log_lambdas.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    trace["hash"] = param_hash(trace["objects"]) if trace["objects"] else None
    return result, trace


def _final_hash_from_last(cfg):
    """Hash of the final weights as stored in last.pt (used when a run trains 0 steps)."""
    import hashlib

    ckpt = torch.load(os.path.join(cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
    digest = hashlib.sha256()
    for key in ("model_state_dict", "moe_state_dict"):
        for name, t in ckpt[key].items():
            digest.update(name.encode())
            digest.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def uninterrupted(tiny_processed_dir, tmp_path_factory):
    """The reference: one uninterrupted 3-epoch run (12 steps)."""
    mp = pytest.MonkeyPatch()
    try:
        out = str(tmp_path_factory.mktemp("uninterrupted"))
        cfg = _resume_config(tiny_processed_dir(), out)
        result, trace = _traced_run(cfg, mp)
    finally:
        mp.undo()
    assert trace["calls"] == 12
    return cfg, result, trace


class TestResume:
    """T6.2 / AC-21, AC-22."""

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_interrupted_run_resumes_bit_identically(
        self, uninterrupted, tiny_processed_dir, tmp_path, monkeypatch, num_workers,
    ):
        import json

        ref_cfg, ref_result, ref_trace = uninterrupted
        cfg = _resume_config(tiny_processed_dir(), str(tmp_path), num_workers=num_workers)

        # Fail at the FIRST step of epoch 2 (call 9 of 12).
        result, first = _traced_run(cfg, monkeypatch, fail_at_call=9)
        assert result is None and len(first["lrs"]) == 8
        last = torch.load(os.path.join(cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
        assert last["resume_state"]["epoch"] == 1

        result, second = _traced_run(cfg, monkeypatch)
        assert second["calls"] == 4, "the resumed run must start at epoch 2, not repeat 0/1"
        assert second["entry_last_epoch"][0] == 8  # scheduler.last_epoch == global_step

        assert first["lrs"] + second["lrs"] == ref_trace["lrs"]
        assert second["hash"] == ref_trace["hash"]
        assert result["best_val_ndcg"] == ref_result["best_val_ndcg"]
        assert result["best_epoch"] == ref_result["best_epoch"]
        assert result["epoch_mean_losses"] == ref_result["epoch_mean_losses"]

        log_path = os.path.join(cfg.log_dir, "pretrain_t_small.log")
        lines = [l for l in open(log_path) if "epoch=" in l]
        assert [l.split("epoch=")[1].split(" ")[0] for l in lines] == ["0", "1", "2"]
        jsonl = os.path.join(cfg.log_dir, "gradnorm_t_small.jsonl")
        steps = [json.loads(l)["step"] for l in open(jsonl)]
        assert len(steps) == len(set(steps)) and steps, steps

    def test_resume_logs_where_it_starts(
        self, tiny_processed_dir, tmp_path, monkeypatch, caplog,
    ):
        import logging

        cfg = _resume_config(tiny_processed_dir(), str(tmp_path))
        _traced_run(cfg, monkeypatch, fail_at_call=5)  # dies in epoch 1
        with caplog.at_level(logging.INFO, logger="training.pretrain"):
            _traced_run(cfg, monkeypatch)
        assert "Resuming t-small from" in caplog.text
        assert "at epoch 1 (global step 4)" in caplog.text

    @pytest.mark.parametrize("change", ["learning_rate", "pretrain_epochs", "source_revision"])
    def test_resume_is_refused_for_another_run(
        self, uninterrupted, tiny_processed_dir, tmp_path, monkeypatch, change,
    ):
        import shutil

        import training.pretrain as pretrain

        ref_cfg = uninterrupted[0]
        cfg = _resume_config(tiny_processed_dir(), str(tmp_path))
        shutil.copytree(ref_cfg.checkpoint_dir, cfg.checkpoint_dir)
        last = os.path.join(cfg.model_checkpoint_dir(), "last.pt")
        if change == "learning_rate":
            cfg.learning_rate = 5e-4
        elif change == "pretrain_epochs":
            cfg.pretrain_epochs = 4
        else:
            real = pretrain.get_source_revision
            monkeypatch.setattr(
                pretrain, "get_source_revision",
                lambda *a, **k: {**real(), "source_hash": "0000000000000000"},
            )
        with pytest.raises(RuntimeError) as exc:
            _traced_run(cfg, monkeypatch)
        assert last in str(exc.value)
        assert "--no_resume" in str(exc.value)
        expected = {
            "learning_rate": "training_fingerprint differs",
            "pretrain_epochs": "training_fingerprint differs",
            "source_revision": "source_revision.source_hash differs",
        }[change]
        assert expected in str(exc.value)

    def test_unchanged_copy_is_accepted(
        self, uninterrupted, tiny_processed_dir, tmp_path, monkeypatch,
    ):
        """Non-vacuity for the refusals above: the same run in another dir resumes."""
        import shutil

        cfg = _resume_config(tiny_processed_dir(), str(tmp_path))
        shutil.copytree(uninterrupted[0].checkpoint_dir, cfg.checkpoint_dir)
        _, trace = _traced_run(cfg, monkeypatch)
        assert trace["calls"] == 0

    def test_no_resume_trains_from_scratch(
        self, uninterrupted, tiny_processed_dir, tmp_path, monkeypatch,
    ):
        import shutil

        ref_cfg, _, ref_trace = uninterrupted
        cfg = _resume_config(tiny_processed_dir(), str(tmp_path), resume_training=False)
        shutil.copytree(ref_cfg.checkpoint_dir, cfg.checkpoint_dir)
        _, trace = _traced_run(cfg, monkeypatch)
        assert trace["calls"] == 12
        assert trace["hash"] == ref_trace["hash"]

    def test_completed_run_trains_zero_steps(
        self, uninterrupted, tiny_processed_dir, tmp_path, monkeypatch,
    ):
        import shutil

        ref_cfg, ref_result, ref_trace = uninterrupted
        cfg = _resume_config(tiny_processed_dir(), str(tmp_path))
        shutil.copytree(ref_cfg.checkpoint_dir, cfg.checkpoint_dir)
        result, trace = _traced_run(cfg, monkeypatch)
        assert trace["calls"] == 0
        assert result["best_val_ndcg"] == ref_result["best_val_ndcg"]
        assert result["best_epoch"] == ref_result["best_epoch"]
        assert result["epoch_mean_losses"] == ref_result["epoch_mean_losses"]

    def test_total_steps_mismatch_is_refused(self, uninterrupted):
        """The data under the same config can change size (e.g. a re-preprocessed dir)."""
        from training.pretrain import _load_resume_checkpoint

        cfg = uninterrupted[0]
        last = os.path.join(cfg.model_checkpoint_dir(), "last.pt")
        _load_resume_checkpoint(last, cfg, 12)  # accepted as-is
        with pytest.raises(RuntimeError, match="total_steps differs") as exc:
            _load_resume_checkpoint(last, cfg, 13)
        assert last in str(exc.value)


class TestResumeLogCleanup:
    """Review REV-003: a crash AFTER an epoch's log line and GradNorm records were written
    but BEFORE its `last.pt` redoes that epoch on resume; the redone epoch must replace,
    not duplicate, those lines."""

    def test_crash_between_epoch_log_and_last_pt(self, tiny_processed_dir, tmp_path, monkeypatch):
        import json

        import training.pretrain as pretrain

        # Log lambdas every 2 steps so the redone epoch's GradNorm records exist (3 epochs x 4
        # steps -> steps 0, 2, ..., 10).
        monkeypatch.setattr(pretrain, "GRADNORM_LOG_INTERVAL_STEPS", 2)
        cfg = _resume_config(tiny_processed_dir(), str(tmp_path))
        real_save = pretrain.save_checkpoint

        def crash_on_final_last_pt(path, epoch, *args, **kwargs):
            if os.path.basename(path) == "last.pt" and epoch == 2:
                raise _Injected("crash after log_epoch, before last.pt")
            return real_save(path, epoch, *args, **kwargs)

        monkeypatch.setattr(pretrain, "save_checkpoint", crash_on_final_last_pt)
        result, first = _traced_run(cfg, monkeypatch)
        assert result is None and first["calls"] == 12

        log_path = os.path.join(cfg.log_dir, "pretrain_t_small.log")
        jsonl = os.path.join(cfg.log_dir, "gradnorm_t_small.jsonl")

        def epochs():
            return [l.split("epoch=")[1].split(" ")[0] for l in open(log_path) if "epoch=" in l]

        def steps():
            return [json.loads(l)["step"] for l in open(jsonl)]

        # Non-vacuity: the crashed epoch DID write its log line and its GradNorm records.
        assert epochs() == ["0", "1", "2"]
        assert steps() == [0, 2, 4, 6, 8, 10]

        monkeypatch.setattr(pretrain, "save_checkpoint", real_save)
        result, second = _traced_run(cfg, monkeypatch)
        assert second["calls"] == 4, "epoch 2 must be redone exactly once"
        assert result is not None
        assert epochs() == ["0", "1", "2"], "the redone epoch duplicated its log line"
        assert steps() == [0, 2, 4, 6, 8, 10], "the redone epoch duplicated GradNorm records"


class TestCudaRngDeviceCount:
    """specs/ablation-run AC-41: resume tolerates a different number of visible GPUs."""

    @staticmethod
    def _state():
        import random

        import numpy as np

        from training.pretrain import _capture_rng_state

        state = _capture_rng_state(
            random.Random(0), torch.Generator().manual_seed(0), torch.device("cpu"),
        )
        state["torch_cuda"] = [torch.full((8,), i, dtype=torch.uint8) for i in range(4)]
        state["torch_cuda_index"] = 2
        return state

    def _restore(self, monkeypatch, visible):
        import random

        import numpy as np

        from training.pretrain import _restore_rng_state

        calls = {"all": [], "one": []}
        monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: calls["all"].append(states))
        monkeypatch.setattr(torch.cuda, "set_rng_state", lambda st, *a, **k: calls["one"].append(st))
        _restore_rng_state(
            self._state(), random.Random(1), torch.Generator(), torch.device("cuda"),
        )
        return calls

    def test_fewer_visible_devices_restores_the_saved_device_and_warns(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="training.pretrain"):
            calls = self._restore(monkeypatch, visible=1)
        assert calls["all"] == []
        assert len(calls["one"]) == 1 and torch.equal(calls["one"][0], torch.full((8,), 2, dtype=torch.uint8))
        assert "saved 4 CUDA RNG states but 1 device(s) are visible" in caplog.text

    def test_equal_counts_restore_every_device_as_before(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="training.pretrain"):
            calls = self._restore(monkeypatch, visible=4)
        assert len(calls["all"]) == 1 and len(calls["all"][0]) == 4
        assert calls["one"] == []
        assert "CUDA RNG states" not in caplog.text
