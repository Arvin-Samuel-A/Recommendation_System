"""Every ablation arm reseeds before it builds its model, so its result does not depend on
which arms ran earlier in the same process (measured on the 10% sample before the fix:
H2_seq4 scored 0.4154 after the earlier H arms and 0.4018 alone)."""
import copy
import random

import numpy as np
import pytest
import torch

from tests.conftest import tiny_config
from utils import set_seed

ARM = "input_only"  # study B's second arm: in a full run it follows objective_conditioned


def _cfg(tiny_processed_dir, out_dir):
    return tiny_config(
        tiny_processed_dir(), str(out_dir),
        pretrain_epochs=1, max_steps_per_epoch=2, ablation_eval_examples=32,
    )


def _disturb_global_rng():
    """Stand-in for whatever earlier arms in the process drew from the global generators."""
    random.random()
    np.random.rand(17)
    torch.rand(1013)


def _arm_metrics(results, label=ARM):
    (record,) = [r for r in results["B"] if r["label"] == label]
    assert record.get("status") == "ok", record
    return record["metrics"]


def test_arm_metrics_do_not_depend_on_execution_order(tiny_processed_dir, tmp_path):
    from training.ablation_runner import run_ablation

    set_seed(42)
    alone = run_ablation(_cfg(tiny_processed_dir, tmp_path / "alone"), "t-small", "B", arms=[ARM])

    set_seed(42)
    after_others = run_ablation(_cfg(tiny_processed_dir, tmp_path / "full"), "t-small", "B")

    set_seed(42)
    _disturb_global_rng()
    split = run_ablation(_cfg(tiny_processed_dir, tmp_path / "split"), "t-small", "B", arms=[ARM])

    assert _arm_metrics(after_others) == _arm_metrics(alone)
    assert _arm_metrics(split) == _arm_metrics(alone)


def test_arm_reusing_a_stored_backbone_is_reseeded_before_building(
    tiny_processed_dir, tmp_path, monkeypatch,
):
    """An arm that loads a stored backbone still draws random numbers after training
    (LoRA init, federated client choice, eval negatives), so it reseeds too -- and before
    `build_model`, whose weight initialisation uses the global generators."""
    import training.pretrain as pretrain
    from training.ablation_runner import run_ablation

    cfg = _cfg(tiny_processed_dir, tmp_path)
    run_ablation(copy.deepcopy(cfg), "t-small", "B")  # trains and stores both backbones

    set_seed(cfg.seed)
    expected = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())

    seen = []
    real_build = pretrain.build_model

    def spy(c):
        seen.append((torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()))
        return real_build(c)

    # Same backbone store, fresh results dir: the arms re-run instead of reusing their
    # persisted results, and each loads its stored backbone rather than training.
    rerun = copy.deepcopy(cfg)
    rerun.ablation_dir = str(tmp_path / "ablations_rerun")
    monkeypatch.setattr(pretrain, "build_model", spy)
    _disturb_global_rng()
    results = run_ablation(rerun, "t-small", "B")

    assert all(r.get("shared_backbone_with") for r in results["B"])
    assert len(seen) == 2
    for torch_state, np_state, py_state in seen:
        assert torch.equal(torch_state, expected[0])
        assert np.array_equal(np_state, expected[1])
        assert py_state == expected[2]
