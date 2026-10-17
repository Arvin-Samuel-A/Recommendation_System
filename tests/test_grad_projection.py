"""Next-item-protected gradient projection (training/grad_projection.py, grad_projection)."""
import copy
import glob

import pytest
import torch

from configs.config import Config
from training.grad_projection import project_conflicting_gradients, static_objective_weights

W_ALL = {"next_item": 1.0, "masked_item": 1.0, "multi_step": 1.0, "contrastive": 1.0}


def _linear_losses(directions):
    """One parameter vector `w` and, per objective, the loss `d . w`, whose gradient is
    exactly `d`."""
    w = torch.nn.Parameter(torch.zeros(len(next(iter(directions.values())))))
    losses = {n: (torch.tensor(d) * w).sum() for n, d in directions.items()}
    return w, losses


def _vec(*xs):
    return torch.tensor(xs, dtype=torch.float32)


class TestProjectionMath:
    def test_conflicting_aux_loses_its_component_along_next_item(self):
        w, losses = _linear_losses({"next_item": [1.0, 0.0], "masked_item": [-1.0, 1.0]})
        stats = project_conflicting_gradients(losses, W_ALL, [w])
        # aux (-1, 1) projected off (1, 0) -> (0, 1); total = (1, 0) + (0, 1).
        assert torch.allclose(w.grad, _vec(1.0, 1.0))
        assert stats["proj_frac_masked_item"].item() == 1.0
        assert stats["proj_cos_masked_item"].item() == pytest.approx(-2 ** -0.5, rel=1e-5)

    def test_agreeing_aux_is_kept(self):
        w, losses = _linear_losses({"next_item": [1.0, 0.0], "masked_item": [1.0, 1.0]})
        stats = project_conflicting_gradients(losses, W_ALL, [w])
        assert torch.allclose(w.grad, _vec(2.0, 1.0))
        assert stats["proj_frac_masked_item"].item() == 0.0

    def test_next_item_component_is_never_reduced(self):
        """After projection no auxiliary pulls against next-item."""
        torch.manual_seed(0)
        dirs = {n: torch.randn(16).tolist() for n in W_ALL}
        w, losses = _linear_losses(dirs)
        project_conflicting_gradients(losses, W_ALL, [w])
        g_p = torch.tensor(dirs["next_item"])
        for name in ("masked_item", "multi_step", "contrastive"):
            a = torch.tensor(dirs[name])
            dot = (a @ g_p).item()
            projected = a - min(dot, 0.0) / (g_p @ g_p) * g_p
            assert (projected @ g_p).item() >= -1e-5
        expected = g_p + sum(
            torch.tensor(dirs[n]) - min((torch.tensor(dirs[n]) @ g_p).item(), 0.0)
            / (g_p @ g_p) * g_p
            for n in ("masked_item", "multi_step", "contrastive")
        )
        assert torch.allclose(w.grad, expected, atol=1e-5)

    def test_weights_scale_each_objective(self):
        w, losses = _linear_losses({"next_item": [1.0, 0.0], "masked_item": [1.0, 1.0]})
        project_conflicting_gradients(losses, {"next_item": 1.0, "masked_item": 0.5}, [w])
        assert torch.allclose(w.grad, _vec(1.5, 0.5))

    def test_no_conflict_equals_plain_backward(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(4, 3)
        x = torch.randn(8, 4)
        out = model(x)
        losses = {"next_item": out.pow(2).mean(), "masked_item": out.pow(2).sum() * 0.1}
        project_conflicting_gradients(losses, W_ALL, list(model.parameters()))
        projected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        out = model(x)
        (out.pow(2).mean() + out.pow(2).sum() * 0.1).backward()
        for a, p in zip(projected, model.parameters()):
            assert torch.allclose(a, p.grad, atol=1e-6)

    def test_parameter_reached_by_one_objective_keeps_its_gradient(self):
        shared = torch.nn.Parameter(torch.zeros(2))
        head = torch.nn.Parameter(torch.zeros(2))
        losses = {
            "next_item": (_vec(1.0, 0.0) * shared).sum(),
            "masked_item": (_vec(-1.0, 1.0) * shared).sum() + (_vec(3.0, 4.0) * head).sum(),
        }
        project_conflicting_gradients(losses, W_ALL, [shared, head])
        assert torch.allclose(shared.grad, _vec(1.0, 1.0))
        assert torch.allclose(head.grad, _vec(3.0, 4.0))

    def test_primary_must_be_active(self):
        w, losses = _linear_losses({"masked_item": [1.0, 0.0]})
        with pytest.raises(ValueError):
            project_conflicting_gradients(losses, W_ALL, [w])

    def test_static_weights_follow_config(self):
        cfg = Config()
        cfg.masked_lm_weight = 0.2
        assert static_objective_weights(cfg)["masked_item"] == 0.2
        assert static_objective_weights(cfg)["next_item"] == 1.0


class TestConfigAndArm:
    def test_validation(self):
        cfg = Config()
        cfg.use_gradnorm = False
        cfg.grad_projection = "next_item"
        cfg.validate()
        for bad in (dict(grad_projection="nope"), dict(use_gradnorm=True)):
            other = copy.deepcopy(cfg)
            for k, v in bad.items():
                setattr(other, k, v)
            with pytest.raises(ValueError):
                other.validate()

    def test_projection_changes_only_its_own_fingerprint(self):
        """No study arm uses projection yet (docs/gradient-projection.md): defaults keep
        every fingerprint, and switching it on for arv_mmoe gives a new one."""
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        arms = {c._study_label: c for c in resolve_study_configs("A", Config())}
        assert all(not c.grad_projection for c in arms.values())
        proj = copy.deepcopy(arms["arv_mmoe"])
        proj.grad_projection = "next_item"
        proj.validate()
        assert _training_fingerprint(proj) != _training_fingerprint(arms["arv_mmoe"])


class TestPretrain:
    def test_projection_trains_and_logs_conflicts(self, tiny_processed_dir, tmp_path):
        from tests.conftest import tiny_config
        from training.pretrain import run_pretrain

        cfg = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=2, max_steps_per_epoch=3,
            use_gradnorm=False, grad_projection="next_item", contrastive_interval=1,
        )
        cfg.validate()
        result = run_pretrain(cfg, "t-small")
        assert result["best_val_ndcg"] >= 0.0
        logs = "".join(open(p).read() for p in glob.glob(f"{cfg.log_dir}/pretrain_*.log"))
        for name in ("masked_item", "multi_step", "contrastive"):
            assert f"proj_cos_{name}=" in logs and f"proj_frac_{name}=" in logs
