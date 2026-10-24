"""specs/ablation-run: training semantics are frozen while the hot path changes.

`test_golden_tiny_run` (T3.2, AC-11) pins a tiny CPU run's final parameters, per-step loss
trace and per-step LR trace. It was captured BEFORE any FR-4 (utilisation) or FR-8
(resume) change and must stay green through them: those changes may make training faster,
never different (steering/code-quality.md, "Training Hot Path and Reproducibility").
"""
import hashlib
import os
import platform

import pytest
import torch

from tests.conftest import tiny_config

#: The environment the constants were captured on. Bit-identity is only meaningful on the
#: same torch build and CPU architecture; elsewhere the test skips rather than fails.
GOLDEN_TORCH = "2.14.0"
GOLDEN_MACHINE = "arm64"

#: Captured at T3.2, RE-captured once before any FR-4/FR-8 change after T4.2's probe was
#: made RNG-neutral (see specs/ablation-run/tasks.md, T3.2 and T5.1 BUILD notes), and
#: RE-captured on 2026-10-25 for TRAINING_SEMANTICS_VERSION 3 (leave-one-out data,
#: chunked whole-sequence training with full-catalogue next-item / multi-step losses), and
#: again for TRAINING_SEMANTICS_VERSION 4 (masked input only for the masked-item loss,
#: next-item-primary GradNorm with beta = 3, multi_step_k = 9).
GOLDEN_PARAM_SHA256 = "0d4d043284eae748fc8a4f66cf6adb61314e741be76d9f790449dca1032bdf9a"
GOLDEN_LOSSES = [
    {"next_item": 5.0, "masked_item": 5.03125, "multi_step": 5.03125, "contrastive": 2.418726, "total": 17.48123},
    {"next_item": 5.0, "masked_item": 5.0, "multi_step": 5.03125, "total": 15.0979},
    {"next_item": 5.03125, "masked_item": 5.03125, "multi_step": 5.0, "total": 15.06236},
    {"next_item": 5.0, "masked_item": 5.0625, "multi_step": 5.0, "total": 15.06182},
    {"next_item": 5.0, "masked_item": 5.0, "multi_step": 5.0, "contrastive": 2.80951, "total": 17.72784},
    {"next_item": 5.0, "masked_item": 5.03125, "multi_step": 5.0, "total": 15.16094},
    {"next_item": 5.0, "masked_item": 5.03125, "multi_step": 5.0, "contrastive": 2.153356, "total": 17.11738},
    {"next_item": 5.03125, "masked_item": 5.03125, "multi_step": 5.0, "total": 15.19997},
    {"next_item": 4.96875, "masked_item": 5.0625, "multi_step": 5.0, "total": 15.02872},
    {"next_item": 4.96875, "masked_item": 5.0, "multi_step": 5.0, "total": 14.96875},
    {"next_item": 5.0, "masked_item": 5.0, "multi_step": 5.0, "contrastive": 2.396716, "total": 17.29729},
    {"next_item": 4.9375, "masked_item": 5.0625, "multi_step": 5.0, "total": 15.16834},
]
GOLDEN_LRS = [
    0.001, 0.000981771838126524, 0.0009285640897740315, 0.0008446873302753783,
    0.0007369367558508489, 0.0006140416772229784, 0.0004859583227770218, 0.00036306324414915115,
    0.00025531266972462173, 0.00017143591022596844, 0.00011822816187347623, 0.0001,
]


def golden_config(processed, out_dir, **overrides):
    """The fully pinned T3.2 configuration."""
    pinned = dict(
        pretrain_epochs=2,
        max_steps_per_epoch=6,
        batch_size=16,
        num_workers=0,
        use_contrastive=True,
        use_gradnorm=True,
        min_lr_ratio=0.1,
        dtype="bf16",
        mixed_precision=False,
        device="cpu",
        seed=42,
    )
    pinned.update(overrides)
    return tiny_config(processed, out_dir, **pinned)


def _sig(value: float) -> float:
    return float(f"{float(value):.7g}")


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def run_and_capture(cfg, monkeypatch):
    """Run `run_pretrain` through a thin spy on `train_one_step`.

    Returns (param_sha256, loss_trace, lr_trace, result). The spy records the step's loss
    values (rounded to 7 significant digits) and the LR after the scheduler step, and keeps
    references to the trained objects so their final state can be hashed.
    """
    import training.pretrain as pretrain
    from utils import set_seed

    captured = {"losses": [], "lrs": [], "objects": None}
    real_step = pretrain.train_one_step

    def spy(model, moe, batch, config, loss_fns, gradnorm, optimizer, scheduler, *a, **kw):
        loss_values, snapshots = real_step(
            model, moe, batch, config, loss_fns, gradnorm, optimizer, scheduler, *a, **kw,
        )
        step = {
            name: _sig(value) for name, value in loss_values.items()
            if not name.startswith("_")
        }
        captured["losses"].append(step)
        captured["lrs"].append(float(optimizer.param_groups[0]["lr"]))
        captured["objects"] = (model, moe, loss_fns, gradnorm)
        return loss_values, snapshots

    monkeypatch.setattr(pretrain, "train_one_step", spy)
    set_seed(cfg.seed)
    result = pretrain.run_pretrain(cfg, "t-small")

    model, moe, loss_fns, gradnorm = captured["objects"]
    digest = hashlib.sha256()
    for state in [model.state_dict(), moe.state_dict()] + [
        fn.state_dict() for fn in loss_fns.values()
    ]:
        for name, tensor in state.items():
            digest.update(name.encode())
            digest.update(_tensor_bytes(tensor))
    if gradnorm is not None:
        digest.update(_tensor_bytes(gradnorm.log_lambdas))
    return digest.hexdigest(), captured["losses"], captured["lrs"], result


def _require_golden_platform():
    if torch.__version__.split("+")[0] != GOLDEN_TORCH or platform.machine() != GOLDEN_MACHINE:
        pytest.skip(
            f"golden constants were captured on torch {GOLDEN_TORCH} / {GOLDEN_MACHINE}; "
            f"this is torch {torch.__version__} / {platform.machine()}"
        )


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("repeat", [0, 1])
def test_golden_tiny_run(repeat, tiny_processed_dir, tmp_path, monkeypatch, single_thread):
    """AC-11. Parametrised twice so one session shows the run is deterministic."""
    _require_golden_platform()
    cfg = golden_config(tiny_processed_dir(), str(tmp_path))
    param_hash, losses, lrs, _ = run_and_capture(cfg, monkeypatch)

    if os.environ.get("GOLDEN_CAPTURE"):
        print(f"\nGOLDEN_PARAM_SHA256 = {param_hash!r}")
        print(f"GOLDEN_LOSSES = {losses!r}")
        print(f"GOLDEN_LRS = {lrs!r}")

    assert len(losses) == 12
    assert lrs == GOLDEN_LRS
    assert losses == GOLDEN_LOSSES
    assert param_hash == GOLDEN_PARAM_SHA256


# ---------------------------------------------------------------- T5.1 (AC-10, AC-12)


def _batches(loader, epochs=2):
    out = []
    for _ in range(epochs):
        out.extend(list(loader))
    return out


def test_workers_yield_identical_batches(tiny_processed_dir, tmp_path):
    """AC-10: worker processes + pinned memory change nothing about the batch sequence."""
    import os as _os

    from data.sequence_generator import build_dataloader, load_metadata_matrix, load_train_dataset
    import json

    processed = tiny_processed_dir()
    cfg = golden_config(processed, str(tmp_path))
    with open(_os.path.join(processed, "dataset_stats.json")) as f:
        cfg.num_items = json.load(f)["num_items"]
    meta = load_metadata_matrix(processed, cfg)
    dataset = load_train_dataset(processed, cfg.max_seq_length, meta)

    cfg.num_workers, cfg.pin_memory = 0, False
    reference = _batches(build_dataloader(dataset, cfg, shuffle=True))
    cfg.num_workers, cfg.pin_memory = 2, True
    with_workers = _batches(build_dataloader(dataset, cfg, shuffle=True))

    assert len(reference) == len(with_workers) > 2
    for i, (a, b) in enumerate(zip(reference, with_workers)):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), f"batch {i} key {key} differs"


def test_golden_hash_with_two_workers(tiny_processed_dir, tmp_path, monkeypatch, single_thread):
    """T5.1: worker loading reproduces the golden parameters bit for bit."""
    _require_golden_platform()
    cfg = golden_config(tiny_processed_dir(), str(tmp_path), num_workers=2, pin_memory=True)
    param_hash, losses, lrs, _ = run_and_capture(cfg, monkeypatch)
    assert lrs == GOLDEN_LRS
    assert losses == GOLDEN_LOSSES
    assert param_hash == GOLDEN_PARAM_SHA256
