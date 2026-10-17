"""Logit temperature / learnable logit scale, and the GradNorm renormalisation scope.

The defaults (logit_temperature=1.0, learnable_logit_scale=False,
shared_logit_temperature=False, gradnorm_renorm_scope="active") must reproduce the
previous training exactly, including every training fingerprint.
"""
import copy
import hashlib
import json
import math
import os
from dataclasses import asdict

import pytest
import torch

from configs.config import Config, OBJECTIVE_ORDER
from training.gradnorm import GradNormController, enabled_objectives
from training.losses import (
    InfoNCELoss,
    LogitScale,
    MaskedItemLoss,
    SampledSoftmaxLoss,
    clamp_logit_scales_,
    learnable_logit_scale_values,
)

NEW_FIELDS = (
    "logit_temperature", "learnable_logit_scale", "logit_scale_max",
    "shared_logit_temperature", "gradnorm_renorm_scope",
    "gradnorm_primary_objective", "gradnorm_aux_budget", "grad_projection",
)


def _ce_floor(num_candidates: int, scale: float) -> float:
    """The lowest CE over `num_candidates` cosine logits times `scale`: target at +1,
    every other candidate at -1."""
    return math.log(1 + (num_candidates - 1) * math.exp(-2 * scale))


def _best_case_next_item(B=3, D=8, num_neg=100):
    """Embeddings at the cosine bounds: the positive is the user vector, every negative
    its opposite."""
    torch.manual_seed(0)
    user = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    table = torch.cat([torch.zeros(1, D), user, -user])  # 0 pad, 1..B pos, B+1.. neg
    pos_ids = torch.arange(1, B + 1)
    neg_ids = (torch.arange(B) + B + 1).unsqueeze(1).expand(B, num_neg)
    return user, pos_ids, neg_ids, table


class TestCEFloor:
    def test_next_item_floor_at_tau_1_and_below_it_at_lower_tau(self):
        user, pos, neg, table = _best_case_next_item()
        at_1 = SampledSoftmaxLoss(normalize_embeddings=True)(user, pos, neg, table)
        assert at_1.item() == pytest.approx(_ce_floor(101, 1.0), abs=1e-4)
        assert _ce_floor(101, 1.0) == pytest.approx(2.68, abs=0.005)
        assert math.log(101) == pytest.approx(4.62, abs=0.005)

        tau = 0.07
        low = SampledSoftmaxLoss(normalize_embeddings=True, logit_scale=LogitScale(tau))
        at_low = low(user, pos, neg, table)
        assert at_low.item() == pytest.approx(_ce_floor(101, 1 / tau), abs=1e-5)
        assert at_low.item() < 1e-6 < 2.68

    def test_full_softmax_floor(self):
        """Masked-item over C items: floor log(1 + (C-1)e^-2) at tau=1 (~7.5 at 13.5k)."""
        C, D = 13_500, 4
        assert _ce_floor(C, 1.0) == pytest.approx(7.51, abs=0.01)
        e = torch.nn.functional.normalize(torch.tensor([[1.0, 0, 0, 0]]), dim=-1)
        table = torch.cat([e, -e.expand(C - 1, D)])
        hidden = e.unsqueeze(0)  # (1, 1, D)
        mask = torch.ones(1, 1, dtype=torch.bool)
        true_ids = torch.zeros(1, 1, dtype=torch.long)
        for tau in (1.0, 0.07):
            fn = MaskedItemLoss(normalize_embeddings=True, logit_scale=LogitScale(tau))
            got = fn(hidden, mask, true_ids, table).item()
            assert got == pytest.approx(_ce_floor(C, 1 / tau), rel=1e-4, abs=1e-6)

    def test_default_is_bit_identical_to_no_scale(self):
        torch.manual_seed(1)
        user, table = torch.randn(4, 8), torch.randn(30, 8)
        pos, neg = torch.randint(1, 30, (4,)), torch.randint(1, 30, (4, 10))
        default = SampledSoftmaxLoss(normalize_embeddings=True)(user, pos, neg, table)
        explicit = SampledSoftmaxLoss(
            normalize_embeddings=True, logit_scale=LogitScale(1.0),
        )(user, pos, neg, table)
        assert torch.equal(default, explicit)
        x = torch.randn(5, 7)
        assert LogitScale()(x) is x, "tau=1 must add no op at all"
        assert list(LogitScale().parameters()) == []


class TestLearnableScale:
    def test_trains_upward_on_a_separable_problem(self):
        user, pos, neg, table = _best_case_next_item()
        fn = SampledSoftmaxLoss(
            normalize_embeddings=True, logit_scale=LogitScale(1.0, learnable=True),
        )
        opt = torch.optim.Adam(fn.parameters(), lr=0.05)
        start = fn(user, pos, neg, table).item()
        for _ in range(50):
            opt.zero_grad()
            fn(user, pos, neg, table).backward()
            opt.step()
            clamp_logit_scales_({"next_item": fn})
        assert fn.logit_scale.scale().item() > 2.0
        assert fn(user, pos, neg, table).item() < start

    def test_clamps_at_the_bound_and_can_come_back(self):
        user, pos, neg, table = _best_case_next_item()
        fn = SampledSoftmaxLoss(
            normalize_embeddings=True,
            logit_scale=LogitScale(1.0, learnable=True, max_scale=5.0),
        )
        opt = torch.optim.Adam(fn.parameters(), lr=0.2)
        for _ in range(200):
            opt.zero_grad()
            fn(user, pos, neg, table).backward()
            opt.step()
            clamp_logit_scales_({"next_item": fn})
            assert fn.logit_scale.log_scale.item() <= math.log(5.0) + 1e-6
        assert fn.logit_scale.scale().item() == pytest.approx(5.0, rel=1e-5)
        # The forward clamps too, even if the stored value was pushed past the bound.
        with torch.no_grad():
            fn.logit_scale.log_scale.fill_(10.0)
        assert fn.logit_scale.scale().item() == pytest.approx(5.0, rel=1e-5)
        # The stored value sits AT the bound after clamp_, where the gradient is live.
        clamp_logit_scales_({"next_item": fn})
        fn.logit_scale.log_scale.grad = None
        (fn.logit_scale.scale() * 1.0).backward()
        assert fn.logit_scale.log_scale.grad is not None
        assert fn.logit_scale.log_scale.grad.item() != 0.0

    def test_init_from_temperature_and_stays_fp32(self):
        s = LogitScale(0.07, learnable=True)
        assert s.scale().item() == pytest.approx(1 / 0.07, rel=1e-6)
        assert s.log_scale.dtype == torch.float32
        logits = torch.randn(3, 4, dtype=torch.bfloat16)
        assert s(logits).dtype == torch.bfloat16
        with pytest.raises(ValueError):
            LogitScale(0.001, learnable=True, max_scale=100.0)

    def test_build_loss_fns(self):
        from training.pretrain import _build_loss_fns

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.learnable_logit_scale = True
        cfg.logit_temperature = 0.1
        fns = _build_loss_fns(cfg, torch.device("cpu"))
        values = learnable_logit_scale_values(fns)
        assert set(values) == {"next_item", "masked_item", "multi_step"}
        assert all(v == pytest.approx(10.0, rel=1e-5) for v in values.values())
        assert fns["multi_step"].logit_scale.log_scale.dtype == torch.float32
        assert fns["contrastive"].temperature == cfg.contrastive_temperature
        assert not hasattr(fns["contrastive"], "logit_scale")


class TestSharedTemperature:
    def _z(self):
        torch.manual_seed(2)
        return torch.randn(6, 8), torch.randn(6, 8)

    def test_contrastive_uses_logit_temperature(self):
        from training.pretrain import _build_loss_fns

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.logit_temperature = 0.2
        cfg.shared_logit_temperature = True
        fn = _build_loss_fns(cfg, torch.device("cpu"))["contrastive"]
        a, b = self._z()
        assert fn(a, b).item() == pytest.approx(InfoNCELoss(0.2)(a, b).item(), abs=1e-6)

        cfg.learnable_logit_scale = True
        fns = _build_loss_fns(cfg, torch.device("cpu"))
        assert set(learnable_logit_scale_values(fns)) == set(OBJECTIVE_ORDER)
        assert fns["contrastive"](a, b).item() == pytest.approx(
            InfoNCELoss(0.2)(a, b).item(), abs=1e-5,
        )

    def test_default_contrastive_unchanged(self):
        from training.pretrain import _build_loss_fns

        cfg = Config()
        cfg.apply_model_config("t-small")
        fn = _build_loss_fns(cfg, torch.device("cpu"))["contrastive"]
        a, b = self._z()
        assert torch.equal(fn(a, b), InfoNCELoss(0.07)(a, b))


class TestEvalRankInvariance:
    @pytest.mark.parametrize("scale", [0.5, 1 / 0.07, 100.0])
    def test_metrics_invariant_to_positive_scale(self, scale):
        from evaluation.metrics import compute_metrics_batch

        torch.manual_seed(3)
        scores = torch.randn(64, 101).clamp(-1, 1)
        scores[:8, 1] = scores[:8, 0]  # ties with the target
        base = compute_metrics_batch(scores, target_idx=0, k=10)
        scaled = compute_metrics_batch(scores * scale, target_idx=0, k=10)
        for name in base:
            assert torch.allclose(base[name], scaled[name], atol=1e-6), name


def _gradnorm_run(scope, steps=400, interval=4, seed=0):
    """Four objectives on one anchor, contrastive active every `interval` steps with a
    5x larger gradient. Returns the controller and the lambda_contrastive trajectory."""
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(8, 8))
    x = torch.randn(32, 8)
    coef = {"next_item": 1.0, "masked_item": 1.5, "multi_step": 0.7, "contrastive": 5.0}
    gn = GradNormController(1.5, 1e-2, renorm_scope=scope)
    traj = []
    for step in range(steps):
        losses = {
            name: coef[name] * ((x @ W).pow(2).mean() + 1.0)
            for name in OBJECTIVE_ORDER
            if not (name == "contrastive" and step % interval)
        }
        gn.update(losses, [W])
        lam = gn.lambdas.detach()
        if scope == "enabled":
            assert lam.sum().item() == pytest.approx(len(OBJECTIVE_ORDER), abs=1e-4)
        else:
            idx = [OBJECTIVE_ORDER.index(n) for n in losses]
            assert lam[idx].sum().item() == pytest.approx(len(idx), abs=1e-4)
        traj.append(lam[OBJECTIVE_ORDER.index("contrastive")].item())
    return gn, coef, traj


class TestGradNormRenormScope:
    def test_active_scope_pins_the_intermittent_lambda(self):
        _, _, traj = _gradnorm_run("active")
        assert min(traj) > 0.85, "baseline: active scope keeps lambda_contrastive near 1"

    def test_enabled_scope_lets_it_reach_the_gradnorm_balance(self):
        gn, coef, traj = _gradnorm_run("enabled")
        lam = gn.lambdas.detach()
        assert traj[-1] < 0.4, "lambda_contrastive must be free to fall"
        # GradNorm's fixed point: lambda_i * ||grad L_i|| equal across objectives.
        weighted = [lam[OBJECTIVE_ORDER.index(n)].item() * coef[n] for n in OBJECTIVE_ORDER]
        assert max(weighted) / min(weighted) < 1.05

    def test_disabled_objectives_are_never_rescaled(self):
        gn = GradNormController(1.5, 1e-2, renorm_scope="enabled",
                                enabled_objectives=["next_item", "masked_item"])
        W = torch.nn.Parameter(torch.randn(4, 4))
        for _ in range(20):
            gn.update({"next_item": W.pow(2).sum(), "masked_item": 3 * W.pow(2).sum()}, [W])
        lam = gn.lambdas.detach()
        assert lam[:2].sum().item() == pytest.approx(2.0, abs=1e-4)
        assert torch.equal(lam[2:], torch.ones(2))

    def test_enabled_objectives_from_config(self):
        cfg = Config()
        assert enabled_objectives(cfg) == OBJECTIVE_ORDER
        cfg.use_contrastive = False
        cfg.use_multi_step = False
        assert enabled_objectives(cfg) == ["next_item", "masked_item"]

    def test_unknown_scope_rejected(self):
        with pytest.raises(ValueError):
            GradNormController(1.5, 1e-2, renorm_scope="all")
        cfg = Config()
        cfg.gradnorm_renorm_scope = "all"
        with pytest.raises(ValueError):
            cfg.validate()


class TestFingerprint:
    @staticmethod
    def _legacy_fingerprint(cfg):
        """The fingerprint payload exactly as computed before the new fields existed."""
        from training.ablation_runner import (
            TRAINING_SEMANTICS_VERSION, POST_TRAINING_ONLY_FIELDS, _PATH_FIELDS,
            _RUN_SELECTION_FIELDS, _RUNTIME_ONLY_FIELDS,
        )

        fields = {
            k: v for k, v in asdict(cfg).items()
            if k not in POST_TRAINING_ONLY_FIELDS and k not in _PATH_FIELDS
            and k not in _RUN_SELECTION_FIELDS and k not in _RUNTIME_ONLY_FIELDS
            and k not in NEW_FIELDS
        }
        fields["_training_semantics"] = TRAINING_SEMANTICS_VERSION
        payload = json.dumps(fields, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def test_defaults_leave_fingerprints_unchanged(self):
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        base = Config()
        assert _training_fingerprint(base) == self._legacy_fingerprint(base)
        for study in ("A", "D", "E"):
            for arm in resolve_study_configs(study, base):
                assert _training_fingerprint(arm) == self._legacy_fingerprint(arm)

    @pytest.mark.parametrize("field,value", [
        ("logit_temperature", 0.07),
        ("learnable_logit_scale", True),
        ("logit_scale_max", 50.0),
        ("shared_logit_temperature", True),
        ("gradnorm_renorm_scope", "enabled"),
        ("contrastive_interval", 1),
    ])
    def test_non_defaults_change_it(self, field, value):
        from training.ablation_runner import _training_fingerprint

        base = Config()
        other = copy.deepcopy(base)
        setattr(other, field, value)
        assert _training_fingerprint(other) != _training_fingerprint(base)


class TestCLI:
    def test_flags_reach_the_config(self):
        import main as cli

        args = cli.parse_args([
            "--stage", "pretrain", "--logit_temperature", "0.07", "--learnable_logit_scale",
            "--logit_scale_max", "50", "--shared_logit_temperature",
            "--gradnorm_renorm_scope", "enabled", "--contrastive_interval", "2",
        ])
        cfg = Config()
        cli.apply_cli_overrides(cfg, args)
        assert cfg.logit_temperature == 0.07 and cfg.learnable_logit_scale is True
        assert cfg.logit_scale_max == 50.0 and cfg.shared_logit_temperature is True
        assert cfg.gradnorm_renorm_scope == "enabled" and cfg.contrastive_interval == 2

    def test_absent_flags_keep_defaults(self):
        import main as cli

        cfg = Config()
        cli.apply_cli_overrides(cfg, cli.parse_args(["--stage", "pretrain"]))
        for field in NEW_FIELDS + ("contrastive_interval",):
            assert getattr(cfg, field) == getattr(Config(), field), field

    def test_packager_passes_flags_through_only_when_given(self):
        import scripts.package_ablation_run as pkg

        args = pkg.parse_args([]) if hasattr(pkg, "parse_args") else None
        if args is None:
            pytest.skip("packager has no parse_args")
        plain = pkg.study_command(args, "sdpa", "python3", study="A")
        assert "logit" not in plain and "contrastive_interval" not in plain
        args.logit_temperature = 0.07
        args.learnable_logit_scale = True
        args.gradnorm_renorm_scope = "enabled"
        args.contrastive_interval = 1
        cmd = pkg.study_command(args, "sdpa", "python3", study="A")
        assert "--logit_temperature 0.07" in cmd and "--learnable_logit_scale" in cmd
        assert "--gradnorm_renorm_scope enabled" in cmd and "--contrastive_interval 1" in cmd
        # The rendered argv resolves to a config with those values (planner == run).
        base = pkg.base_config_from_argv(pkg.study_argv(args, "sdpa", "A"))
        assert base.logit_temperature == 0.07 and base.learnable_logit_scale is True
        assert base.gradnorm_renorm_scope == "enabled" and base.contrastive_interval == 1

    def test_packager_passes_epochs_through(self):
        import scripts.package_ablation_run as pkg

        args = pkg.parse_args([])
        assert "--epochs" not in pkg.study_command(args, "sdpa", "python3", study="A")
        args = pkg.parse_args(["--epochs", "10"])
        cmd = pkg.study_command(args, "sdpa", "python3", study="A", stage="ablation_collect")
        assert "--epochs 10" in cmd
        base = pkg.base_config_from_argv(pkg.study_argv(args, "sdpa", "A"))
        assert base.pretrain_epochs == 10

    def test_packager_sequence_length_grid_narrows_study_h(self):
        import scripts.package_ablation_run as pkg
        from training.ablation_runner import resolve_study_configs

        args = pkg.parse_args([])
        assert "sequence_length_grid" not in pkg.study_command(args, "sdpa", "python3", study="H")
        args.sequence_length_grid = [25, 50, 100]
        cmd = pkg.study_command(args, "sdpa", "python3", study="H", stage="ablation_collect")
        assert "--sequence_length_grid 25 50 100" in cmd
        base = pkg.base_config_from_argv(pkg.study_argv(args, "sdpa", "H"))
        labels = [c._study_label for c in resolve_study_configs("H", base)]
        assert "H2_seq200" not in labels and "H2_seq100" in labels


class TestCheckpointRoundTrip:
    def _run(self, tiny_processed_dir, tmp_path, **overrides):
        from tests.conftest import tiny_config
        from training.pretrain import run_pretrain

        cfg = tiny_config(
            tiny_processed_dir(), str(tmp_path), pretrain_epochs=2, max_steps_per_epoch=3,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg, run_pretrain(cfg, "t-small")

    def test_learnable_scale_saved_restored_and_logged(self, tiny_processed_dir, tmp_path):
        from training.pretrain import _build_loss_fns, build_model, load_checkpoint

        cfg, result = self._run(
            tiny_processed_dir, tmp_path, learnable_logit_scale=True,
            logit_temperature=0.1, shared_logit_temperature=True,
            gradnorm_renorm_scope="enabled",
        )
        final = result["final_logit_scales"]
        assert set(final) == set(OBJECTIVE_ORDER)
        assert any(abs(v - 10.0) > 1e-6 for v in final.values()), "the scales must train"

        ckpt_path = os.path.join(cfg.model_checkpoint_dir(), "last.pt")
        ckpt = torch.load(ckpt_path, weights_only=False)
        for name in OBJECTIVE_ORDER:
            assert "logit_scale.log_scale" in ckpt["loss_fns_state_dict"][name], name

        model, moe = build_model(cfg)
        fresh = _build_loss_fns(cfg, torch.device("cpu"))
        load_checkpoint(ckpt_path, model, moe, loss_fns=fresh)
        restored = learnable_logit_scale_values(fresh)
        for name, value in final.items():
            assert restored[name] == pytest.approx(value, rel=1e-6), name

        # Excluded from weight decay: its own optimizer group.
        groups = ckpt["optimizer_state_dict"]["param_groups"]
        assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0
        assert len(groups[1]["params"]) == len(OBJECTIVE_ORDER)

        log_files = [f for f in os.listdir(cfg.log_dir) if f.startswith("pretrain")]
        text = "".join(open(os.path.join(cfg.log_dir, f)).read() for f in log_files)
        assert "logit_scale_next_item=" in text and "logit_scale_contrastive=" in text

    def test_defaults_add_nothing_to_checkpoints(self, tiny_processed_dir, tmp_path):
        cfg, result = self._run(tiny_processed_dir, tmp_path)
        assert "final_logit_scales" not in result
        ckpt = torch.load(os.path.join(cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
        assert set(ckpt["loss_fns_state_dict"]) == {"multi_step"}
        assert not any("logit_scale" in k for k in ckpt["loss_fns_state_dict"]["multi_step"])
        assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 1


def test_lora_local_loss_uses_the_logit_temperature(monkeypatch):
    import numpy as np

    import training.local_train as lt
    from models.lora import inject_lora
    from models.moe import MoELayer
    from models.transformer import TransformerModel

    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items, cfg.vocab_size, cfg.use_metadata = 20, 22, False
    cfg.local_epochs = 0
    cfg.logit_temperature = 0.25
    captured = {}
    real = lt.SampledSoftmaxLoss

    def spy(*args, **kwargs):
        obj = real(*args, **kwargs)
        captured["scale"] = obj.logit_scale.scale()
        return obj

    monkeypatch.setattr(lt, "SampledSoftmaxLoss", spy)
    model = TransformerModel(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    inject_lora(model, cfg.lora_rank)
    lt.train_user_lora(model, MoELayer(cfg), [], [], np.arange(1, 21), set(), cfg,
                       torch.device("cpu"))
    assert captured["scale"] == pytest.approx(4.0)
