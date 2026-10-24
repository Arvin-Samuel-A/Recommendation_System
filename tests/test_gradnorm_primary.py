"""Primary-objective GradNorm (gradnorm_primary_objective / gradnorm_aux_budget).

With a primary objective, its lambda stays exactly 1 and GradNorm balances only the
auxiliary objectives, whose lambdas sum to the budget. The defaults ("" / None) must
reproduce the previous controller and every training fingerprint exactly.
"""
import copy
import math
import os

import pytest
import torch

from configs.config import Config, OBJECTIVE_ORDER
from training.gradnorm import GradNormController

AUX = [n for n in OBJECTIVE_ORDER if n != "next_item"]


def _drive(gn, steps=40, drop=(), scales=None, seed=0):
    """Run `steps` GradNorm updates on a toy linear problem; objective i's loss is
    scaled by `scales[name]` (default i + 1). Returns the lambdas after every step."""
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(16, 8))
    scales = scales or {n: float(i + 1) for i, n in enumerate(OBJECTIVE_ORDER)}
    history = []
    for _ in range(steps):
        h = torch.randn(32, 16) @ W
        losses = {n: h.pow(2).mean() * scales[n] for n in OBJECTIVE_ORDER if n not in drop}
        gn.update(losses, [W])
        history.append(gn.lambdas.detach().clone())
        with torch.no_grad():
            W -= 0.01 * torch.randn_like(W)
    return history


def _lam(gn, name):
    return gn.lambdas[OBJECTIVE_ORDER.index(name)].item()


class TestController:
    @pytest.mark.parametrize("scope", ["active", "enabled"])
    @pytest.mark.parametrize("budget", [None, 0.3, 1.0, 3.0])
    def test_primary_fixed_and_aux_sum_to_budget(self, scope, budget):
        gn = GradNormController(1.5, 1e-2, renorm_scope=scope,
                                primary_objective="next_item", aux_budget=budget)
        target = len(AUX) if budget is None else budget
        for lam in _drive(gn):
            assert lam[0].item() == 1.0
            assert lam[1:].sum().item() == pytest.approx(target, rel=1e-5)

    def test_aux_start_on_the_budget(self):
        gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                primary_objective="next_item", aux_budget=0.3)
        assert _lam(gn, "next_item") == 1.0
        for name in AUX:
            assert _lam(gn, name) == pytest.approx(0.1)

    def test_active_scope_shares_the_budget_with_an_absent_aux(self):
        gn = GradNormController(1.5, 1e-2, renorm_scope="active",
                                primary_objective="next_item", aux_budget=3.0)
        last = _drive(gn, steps=5, drop=("contrastive",))[-1]
        assert last[0].item() == 1.0
        assert (last[1] + last[2]).item() == pytest.approx(2.0, rel=1e-5)

    def test_enabled_scope_with_a_disabled_aux(self):
        gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                enabled_objectives=["next_item", "masked_item", "multi_step"],
                                primary_objective="next_item", aux_budget=1.0)
        last = _drive(gn, drop=("contrastive",))[-1]
        assert last[0].item() == 1.0
        assert (last[1] + last[2]).item() == pytest.approx(1.0, rel=1e-5)

    def test_only_primary_active_is_a_no_op(self):
        gn = GradNormController(1.5, 1e-2, primary_objective="next_item", aux_budget=1.0)
        before = gn.log_lambdas.detach().clone()
        W = torch.nn.Parameter(torch.randn(4, 4))
        assert gn.update({"next_item": (W ** 2).sum()}, [W]) is None
        assert torch.equal(gn.log_lambdas.detach(), before)

    def test_primary_gradient_does_not_move_the_aux(self):
        """The primary's loss is invisible to GradNorm: scaling it changes nothing."""
        runs = []
        for primary_scale in (1.0, 50.0):
            gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                    primary_objective="next_item", aux_budget=1.0)
            scales = {"next_item": primary_scale, "masked_item": 2.0,
                      "multi_step": 3.0, "contrastive": 4.0}
            runs.append(_drive(gn, scales=scales)[-1])
        assert torch.equal(runs[0], runs[1])

    def test_aux_are_still_balanced(self):
        """An auxiliary with a 5x gradient ends with the smallest lambda."""
        gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                primary_objective="next_item", aux_budget=1.0)
        scales = {"next_item": 1.0, "masked_item": 1.0, "multi_step": 1.0, "contrastive": 5.0}
        _drive(gn, steps=200, scales=scales)
        assert _lam(gn, "contrastive") < min(_lam(gn, "masked_item"), _lam(gn, "multi_step"))

    def test_weighted_sum_uses_primary_at_one(self):
        gn = GradNormController(1.5, 1e-2, primary_objective="next_item", aux_budget=0.3)
        losses = {n: torch.tensor(1.0) for n in OBJECTIVE_ORDER}
        assert gn.weighted_sum(losses).item() == pytest.approx(1.3)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            GradNormController(1.5, 1e-2, primary_objective="nope")
        with pytest.raises(ValueError):
            GradNormController(1.5, 1e-2, aux_budget=1.0)
        with pytest.raises(ValueError):
            GradNormController(1.5, 1e-2, primary_objective="next_item", aux_budget=0.0)

    def test_state_dict_round_trip(self):
        gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                primary_objective="next_item", aux_budget=1.0)
        _drive(gn, steps=5)
        other = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                   primary_objective="next_item", aux_budget=1.0)
        other.load_state_dict(gn.state_dict())
        assert torch.equal(other.log_lambdas.detach(), gn.log_lambdas.detach())


class TestConfig:
    def test_defaults_are_next_item_primary_with_budget_3(self):
        """Owner decision 2026-10-25."""
        cfg = Config()
        assert cfg.gradnorm_primary_objective == "next_item" and cfg.gradnorm_aux_budget == 3.0
        cfg.validate()

    def test_standard_gradnorm_still_selectable(self):
        cfg = Config()
        cfg.gradnorm_primary_objective, cfg.gradnorm_aux_budget = "", None
        cfg.validate()

    @pytest.mark.parametrize("primary,budget", [("nope", None), ("", 1.0), ("next_item", -1.0)])
    def test_invalid_rejected(self, primary, budget):
        cfg = Config()
        cfg.gradnorm_primary_objective = primary
        cfg.gradnorm_aux_budget = budget
        with pytest.raises(ValueError):
            cfg.validate()


class TestFingerprint:
    @staticmethod
    def _primary(cfg, budget=1.0):
        cfg = copy.deepcopy(cfg)
        cfg.gradnorm_primary_objective = "next_item"
        cfg.gradnorm_aux_budget = budget
        return cfg

    @staticmethod
    def _standard(cfg):
        cfg = copy.deepcopy(cfg)
        cfg.gradnorm_primary_objective, cfg.gradnorm_aux_budget = "", None
        return cfg

    def test_pre_feature_values_omitted(self):
        from training.ablation_runner import _training_fields
        from dataclasses import asdict

        fields = _training_fields(asdict(self._standard(Config())))
        assert "gradnorm_primary_objective" not in fields
        assert "gradnorm_aux_budget" not in fields

    def test_gradnorm_arm_changes(self):
        from training.ablation_runner import _training_fingerprint

        base = self._standard(Config())
        assert base.use_gradnorm
        fps = {_training_fingerprint(base), _training_fingerprint(self._primary(base, None)),
               _training_fingerprint(self._primary(base, 1.0)),
               _training_fingerprint(self._primary(base, 3.0))}
        assert len(fps) == 4

    def test_equal_weight_arms_unchanged(self):
        """Passing the flags on every command must not retrain a no-GradNorm arm."""
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        base, primary = Config(), self._primary(Config())
        for study in ("A", "D"):
            plain = {c._study_label: c for c in resolve_study_configs(study, base)}
            withp = {c._study_label: c for c in resolve_study_configs(study, primary)}
            assert plain.keys() == withp.keys()
            for label, cfg in plain.items():
                same = _training_fingerprint(cfg) == _training_fingerprint(withp[label])
                assert same == (not cfg.use_gradnorm), label


class TestCLIAndPackager:
    def test_flags_reach_the_config(self):
        import main as cli

        cfg = Config()
        cli.apply_cli_overrides(cfg, cli.parse_args([
            "--stage", "pretrain", "--gradnorm_primary_objective", "next_item",
            "--gradnorm_aux_budget", "0.3",
        ]))
        assert cfg.gradnorm_primary_objective == "next_item" and cfg.gradnorm_aux_budget == 0.3

    def test_absent_flags_keep_defaults(self):
        import main as cli

        cfg = Config()
        cli.apply_cli_overrides(cfg, cli.parse_args(["--stage", "pretrain"]))
        assert cfg.gradnorm_primary_objective == "next_item" and cfg.gradnorm_aux_budget == 3.0

    def test_packager_passes_flags_through_only_when_given(self):
        import scripts.package_ablation_run as pkg

        args = pkg.parse_args([])
        assert "gradnorm_primary" not in pkg.study_command(args, "sdpa", "python3", study="A")
        args = pkg.parse_args(["--gradnorm-primary-objective", "next_item",
                               "--gradnorm-aux-budget", "1"])
        cmd = pkg.study_command(args, "sdpa", "python3", study="A")
        assert "--gradnorm_primary_objective next_item" in cmd
        assert "--gradnorm_aux_budget 1.0" in cmd
        base = pkg.base_config_from_argv(pkg.study_argv(args, "sdpa", "A"))
        assert base.gradnorm_primary_objective == "next_item" and base.gradnorm_aux_budget == 1.0


class TestPretrain:
    def test_primary_lambda_stays_one_in_training(self, tiny_processed_dir, tmp_path):
        from tests.conftest import tiny_config
        from training.pretrain import run_pretrain

        cfg = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=2, max_steps_per_epoch=3,
            gradnorm_primary_objective="next_item", gradnorm_aux_budget=0.5,
            gradnorm_renorm_scope="enabled", contrastive_interval=1,
        )
        assert cfg.use_gradnorm
        run_pretrain(cfg, "t-small")
        ckpt = torch.load(os.path.join(cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
        lam = ckpt["gradnorm_state_dict"]["log_lambdas"].exp()
        assert lam[0].item() == 1.0
        assert lam[1:].sum().item() == pytest.approx(0.5, rel=1e-5)
        assert not math.isclose(lam[1].item(), lam[2].item()), "the auxiliaries must move"
