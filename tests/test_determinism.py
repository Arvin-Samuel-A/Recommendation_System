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

#: Captured at T3.2 and RE-captured once before any FR-4/FR-8 change, after T4.2's probe
#: was made RNG-neutral (see specs/ablation-run/tasks.md, T3.2 and T5.1 BUILD notes).
GOLDEN_PARAM_SHA256 = "a0cef99fdcd7e65aec37fd730663599cd726fd508e511c07971c2bcfbda18b5b"
GOLDEN_LOSSES = [
    {"next_item": 2.421875, "masked_item": 5.0625, "multi_step": 5.0, "contrastive": 2.883421, "total": 15.3678},
    {"next_item": 2.390625, "masked_item": 5.03125, "multi_step": 5.0, "total": 12.48367},
    {"next_item": 2.40625, "masked_item": 5.0, "multi_step": 5.0, "total": 12.4169},
    {"next_item": 2.40625, "masked_item": 4.96875, "multi_step": 5.0, "total": 12.39856},
    {"next_item": 2.375, "masked_item": 5.0, "multi_step": 5.0, "contrastive": 2.637355, "total": 14.961},
    {"next_item": 2.390625, "masked_item": 4.96875, "multi_step": 5.03125, "total": 12.54682},
    {"next_item": 2.359375, "masked_item": 5.03125, "multi_step": 5.03125, "contrastive": 2.973651, "total": 15.33676},
    {"next_item": 2.375, "masked_item": 5.03125, "multi_step": 5.0, "total": 12.58311},
    {"next_item": 2.4375, "masked_item": 4.96875, "multi_step": 5.0, "total": 12.45845},
    {"next_item": 2.421875, "masked_item": 5.0, "multi_step": 5.0, "total": 12.48164},
    {"next_item": 2.40625, "masked_item": 5.03125, "multi_step": 5.0, "contrastive": 3.261196, "total": 15.62109},
    {"next_item": 2.390625, "masked_item": 5.03125, "multi_step": 5.0, "total": 12.66273},
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

    from data.sequence_generator import build_dataloader, load_metadata_matrix, load_tensorised
    import json

    processed = tiny_processed_dir()
    cfg = golden_config(processed, str(tmp_path))
    with open(_os.path.join(processed, "dataset_stats.json")) as f:
        cfg.num_items = json.load(f)["num_items"]
    meta = load_metadata_matrix(processed, cfg)
    dataset = load_tensorised(
        _os.path.join(processed, "train_tensorised.pt"), meta, max_seq_length=cfg.max_seq_length,
    )

    cfg.num_workers, cfg.pin_memory = 0, False
    reference = _batches(build_dataloader(dataset, cfg, shuffle=True))
    cfg.num_workers, cfg.pin_memory = 2, True
    with_workers = _batches(build_dataloader(dataset, cfg, shuffle=True))

    assert len(reference) == len(with_workers) > 2
    for i, (a, b) in enumerate(zip(reference, with_workers)):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), f"batch {i} key {key} differs"


def _cpu_batch(processed, n=16):
    import json
    import os as _os

    from data.sequence_generator import load_tensorised

    data = load_tensorised(_os.path.join(processed, "train_tensorised.pt"), max_seq_length=8)
    with open(_os.path.join(processed, "dataset_stats.json")) as f:
        num_items = json.load(f)["num_items"]
    batch = {
        "targets": data.targets[:n].clone(),
        "user_ids": data.user_ids[:n].clone(),
        "sequences": data.sequences[:n].clone(),
    }
    return batch, num_items


def test_cpu_side_negatives_identical(tiny_processed_dir, tmp_path):
    """AC-12: same RandomState state -> same negatives, wherever the batch lives."""
    import os as _os
    import pickle

    import numpy as np

    from data.sequence_generator import GlobalNegativePool
    from training.pretrain import attach_training_negatives

    processed = tiny_processed_dir()
    cfg = golden_config(processed, str(tmp_path))
    batch, cfg.num_items = _cpu_batch(processed)
    pool = GlobalNegativePool.load(
        _os.path.join(processed, "global_negative_pool.npy"), set(),
    ).pool
    with open(_os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        hist = {int(u): np.asarray(v, dtype=np.int64) for u, v in pickle.load(f).items()}

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    def draw(b, device):
        rng = np.random.RandomState(1234)
        return attach_training_negatives(dict(b), pool, hist, cfg, rng, device)["neg_ids"]

    cpu_neg = draw(batch, torch.device("cpu"))
    # A ".to('cpu')" copy stands in for a device batch on a CPU-only host.
    copied = {k: v.to("cpu", copy=True) for k, v in batch.items()}
    assert torch.equal(draw(copied, torch.device("cpu")), cpu_neg)
    for device in devices[1:]:
        on_device = {k: v.to(device) for k, v in batch.items()}
        assert torch.equal(draw(on_device, device).cpu(), cpu_neg), device


def test_step_loop_samples_negatives_before_the_device_copy(
    tiny_processed_dir, tmp_path, monkeypatch,
):
    """AC-12: `run_pretrain` hands the negative sampler the loader's CPU batch, so no
    `targets`/`user_ids` device->host copy happens in the step loop. Runs on MPS when
    available, where the order is observable."""
    import training.pretrain as pretrain

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = golden_config(
        tiny_processed_dir(), str(tmp_path), device=device,
        pretrain_epochs=1, max_steps_per_epoch=2,
    )
    seen = []
    real = pretrain.attach_training_negatives

    def spy(batch, *args, **kwargs):
        seen.append((batch["targets"].device.type, batch["user_ids"].device.type))
        return real(batch, *args, **kwargs)

    monkeypatch.setattr(pretrain, "attach_training_negatives", spy)

    pretrain.run_pretrain(cfg, "t-small")
    assert seen and all(pair == ("cpu", "cpu") for pair in seen), seen
    assert len(seen) == 2


def test_golden_hash_with_two_workers(tiny_processed_dir, tmp_path, monkeypatch, single_thread):
    """T5.1: worker loading reproduces the golden parameters bit for bit."""
    _require_golden_platform()
    cfg = golden_config(tiny_processed_dir(), str(tmp_path), num_workers=2, pin_memory=True)
    param_hash, losses, lrs, _ = run_and_capture(cfg, monkeypatch)
    assert lrs == GOLDEN_LRS
    assert losses == GOLDEN_LOSSES
    assert param_hash == GOLDEN_PARAM_SHA256
