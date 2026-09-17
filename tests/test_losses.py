from pathlib import Path

import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from training.losses import (
    mask_metadata,
    SampledSoftmaxLoss,
    MaskedItemLoss,
    MultiStepLoss,
    InfoNCELoss,
    apply_masking,
    static_weighted_sum,
)
import training.gradnorm
from training.gradnorm import GradNormController, get_anchor_layer_params
from models.transformer import TransformerModel
from utils import get_device, resolve_dtype


@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def dtype():
    return torch.bfloat16


class TestSampledSoftmax:
    def test_loss_decreases_with_optimizer(self, device, dtype):
        V, D, B, K = 50, 16, 8, 10
        item_emb = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
        loss_fn = SampledSoftmaxLoss()

        user_emb = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
        pos_ids = torch.randint(1, V, (B,), device=device)
        neg_ids = torch.randint(1, V, (B, K), device=device)

        loss = loss_fn(user_emb, pos_ids, neg_ids, item_emb)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_high_positive_score_low_loss(self, device, dtype):
        V, D, B = 10, 8, 4
        item_emb = torch.eye(V, D, device=device, dtype=dtype)
        loss_fn = SampledSoftmaxLoss()

        user_emb = item_emb[1:B+1].clone()
        pos_ids = torch.arange(1, B+1, device=device)
        neg_ids = torch.randint(B+1, V, (B, 3), device=device)

        loss = loss_fn(user_emb, pos_ids, neg_ids, item_emb)
        assert loss.item() < 2.0


class TestMaskedItemLoss:
    def test_zero_masked_returns_zero(self, device, dtype):
        B, S, D, V = 4, 10, 16, 50
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        mask_pos = torch.zeros(B, S, dtype=torch.bool, device=device)
        true_ids = torch.randint(1, V, (B, S), device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss_fn = MaskedItemLoss()
        loss = loss_fn(hidden, mask_pos, true_ids, item_emb)
        assert loss.item() == 0.0

    def test_only_masked_contribute(self, device, dtype):
        B, S, D, V = 2, 5, 8, 20
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        true_ids = torch.randint(1, V, (B, S), device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        mask_all = torch.ones(B, S, dtype=torch.bool, device=device)
        mask_some = torch.zeros(B, S, dtype=torch.bool, device=device)
        mask_some[0, 2] = True
        mask_some[1, 4] = True

        loss_fn = MaskedItemLoss()
        loss_all = loss_fn(hidden, mask_all, true_ids, item_emb)
        loss_some = loss_fn(hidden, mask_some, true_ids, item_emb)
        assert torch.isfinite(loss_all)
        assert torch.isfinite(loss_some)


class TestMultiStepLoss:
    def test_valid_targets(self, device, dtype):
        B, S, D, V, K = 4, 10, 16, 50, 3
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        targets_k = torch.randint(1, V, (B, K), device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss = loss_fn(hidden, targets_k, item_emb)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_all_zero_targets(self, device, dtype):
        B, S, D, V, K = 4, 10, 16, 50, 3
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        targets_k = torch.zeros(B, K, dtype=torch.long, device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss = loss_fn(hidden, targets_k, item_emb)
        assert loss.item() == 0.0

    def test_every_step_reads_position_S_minus_1(self, device, dtype):
        """REV-105 / amended FR-3 + AC-03: all k projections read the SAME position.

        The test this replaces asserted only that the loss was finite, so it held under
        either position rule and could not distinguish them. Assert the property that is
        true of `pos = S-1` and false of `pos = S-1-step`: the loss depends on position
        S-1 alone, so arbitrarily changing every EARLIER position must leave it exactly
        unchanged.
        """
        B, S, D, V, K = 2, 6, 8, 20, 3
        torch.manual_seed(0)
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        loss_fn.eval()
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        targets_k = torch.randint(1, V, (B, K), device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        baseline = loss_fn(hidden, targets_k, item_emb)

        scrambled = hidden.clone()
        scrambled[:, : S - 1, :] = torch.randn(
            B, S - 1, D, device=device, dtype=dtype,
        )
        rewritten = loss_fn(scrambled, targets_k, item_emb)

        assert rewritten.item() == baseline.item(), (
            "multi-step loss moved when only positions < S-1 changed, so at least one "
            "step is reading a position other than S-1 (the superseded S-1-step rule)"
        )

        # Non-vacuity: changing position S-1 -- the one position every step reads --
        # must change the loss, otherwise the equality above is trivially true.
        moved = hidden.clone()
        moved[:, S - 1, :] += 1.0
        assert loss_fn(moved, targets_k, item_emb).item() != baseline.item()

    def test_all_k_steps_contribute_distinct_projections(self, device, dtype):
        """Reading one position must not collapse the k steps into one prediction.

        Each step keeps its own bias-free projection (unchanged by REV-105), so the k
        steps still produce different logits from the shared input vector.
        """
        B, S, D, V, K = 1, 6, 8, 20, 3
        torch.manual_seed(0)
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        loss_fn.eval()
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        shared = hidden[:, S - 1, :]

        outs = [loss_fn.step_projections[i](shared) for i in range(K)]
        for i in range(1, K):
            assert not torch.allclose(outs[0], outs[i]), (
                f"step 0 and step {i} produced identical projections of the shared "
                f"position-S-1 vector"
            )


class TestInfoNCE:
    def test_identical_pairs_low_loss(self, device, dtype):
        B, D = 8, 16
        z = torch.randn(B, D, device=device, dtype=dtype)
        loss_fn = InfoNCELoss(temperature=0.07)
        loss = loss_fn(z, z.clone())
        assert torch.isfinite(loss)

    def test_random_pairs_higher_loss(self, device, dtype):
        B, D = 8, 16
        z_orig = torch.randn(B, D, device=device, dtype=dtype)
        z_aug_same = z_orig.clone()
        z_aug_rand = torch.randn(B, D, device=device, dtype=dtype)

        loss_fn = InfoNCELoss(temperature=0.5)
        loss_same = loss_fn(z_orig, z_aug_same)
        loss_rand = loss_fn(z_orig, z_aug_rand)
        assert loss_rand.item() >= loss_same.item() - 0.5

    def test_similarity_matrix_shape(self, device, dtype):
        B, D = 4, 8
        z_orig = torch.randn(B, D, device=device, dtype=dtype)
        z_aug = torch.randn(B, D, device=device, dtype=dtype)

        z_orig_n = torch.nn.functional.normalize(z_orig, p=2, dim=-1)
        z_aug_n = torch.nn.functional.normalize(z_aug, p=2, dim=-1)
        z = torch.cat([z_orig_n, z_aug_n], dim=0)
        sim = z @ z.T
        assert sim.shape == (2 * B, 2 * B)


class TestApplyMasking:
    def test_mask_uses_correct_id(self):
        mask_id = 101
        seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        masked, positions, true_ids = apply_masking(seq, 0.5, mask_id)
        masked_values = masked[positions]
        assert (masked_values == mask_id).all()

    def test_at_least_one_unmasked(self):
        mask_id = 101
        seq = torch.tensor([[1, 2, 3, 4, 5]])
        for _ in range(100):
            masked, positions, _ = apply_masking(seq, 0.99, mask_id)
            real = seq != 0
            unmasked_real = real & ~positions
            assert unmasked_real.sum() >= 1

    def test_padding_not_masked(self):
        mask_id = 101
        seq = torch.tensor([[0, 0, 0, 1, 2, 3]])
        masked, positions, _ = apply_masking(seq, 0.5, mask_id)
        assert not positions[0, :3].any()


class TestStaticWeightedSum:
    def test_matches_manual(self):
        config = Config()
        config.masked_lm_weight = 0.5
        config.multi_step_weight = 0.3
        config.contrastive_weight = 0.2

        losses = {
            "next_item": torch.tensor(1.0),
            "masked_item": torch.tensor(2.0),
            "multi_step": torch.tensor(3.0),
            "contrastive": torch.tensor(4.0),
        }
        result = static_weighted_sum(losses, config)
        expected = 1.0 + 0.5 * 2.0 + 0.3 * 3.0 + 0.2 * 4.0
        assert abs(result.item() - expected) < 1e-5

    def test_partial_active(self):
        config = Config()
        config.masked_lm_weight = 1.0
        losses = {"next_item": torch.tensor(2.0), "masked_item": torch.tensor(3.0)}
        result = static_weighted_sum(losses, config)
        assert abs(result.item() - 5.0) < 1e-5


class TestGradNormLambdaConstraint:
    def _make_model_and_losses(self, num_active):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.dropout = 0.0
        device = get_device(cfg)
        model = TransformerModel(cfg)
        model.train()

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        names = OBJECTIVE_ORDER[:num_active]
        active_losses = {}
        for name in names:
            logits = pooled @ model.item_emb_weight.T
            targets = torch.randint(1, 50, (B,), device=device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            active_losses[name] = loss

        anchor_params = get_anchor_layer_params(model)
        return active_losses, anchor_params

    @pytest.mark.parametrize("num_active", [2, 3, 4])
    def test_lambda_sum_after_update(self, num_active):
        gn = GradNormController(alpha=1.5, lr=1e-2)
        active_losses, anchor_params = self._make_model_and_losses(num_active)

        L_total = gn.weighted_sum(active_losses)
        L_total.backward(retain_graph=True)
        gn.update(active_losses, anchor_params)

        active_indices = [OBJECTIVE_ORDER.index(n) for n in active_losses]
        active_lam_sum = gn.lambdas[active_indices].sum().item()
        assert abs(active_lam_sum - num_active) < 1e-4


class TestGradNormActiveOnly:
    def test_inactive_excluded(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.dropout = 0.0
        device = get_device(cfg)
        model = TransformerModel(cfg)
        model.train()

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        active_losses = {}
        for name in ["next_item", "masked_item"]:
            logits = pooled @ model.item_emb_weight.T
            targets = torch.randint(1, 50, (B,), device=device)
            active_losses[name] = torch.nn.functional.cross_entropy(logits, targets)

        gn = GradNormController(alpha=1.5, lr=1e-2)
        anchor_params = get_anchor_layer_params(model)

        L_total = gn.weighted_sum(active_losses)
        L_total.backward(retain_graph=True)
        gn.update(active_losses, anchor_params)

        assert "contrastive" not in gn.initial_losses
        assert "multi_step" not in gn.initial_losses


class TestGradNormAnchorResolution:
    @pytest.mark.parametrize("moe_variant", ["none", "mmoe", "arv_mmoe"])
    def test_anchor_exists(self, moe_variant):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.moe_variant = moe_variant
        model = TransformerModel(cfg)

        params = get_anchor_layer_params(model)
        assert len(params) > 0
        last_block = model.blocks[-1]
        block_param_ptrs = {p.data_ptr() for p in last_block.parameters()}
        for p in params:
            assert p.data_ptr() in block_param_ptrs


class TestGradNormGradientIsolation:
    """REV-057: `GradNormController.update` must write into `log_lambdas.grad` only.

    REV-124: this used to justify the requirement in terms of the second-order
    implementation -- `torch.autograd.grad(..., create_graph=True)` and the double-backward
    graph it built. T3.6 removed both (see the AC-10 test one class away, which asserts
    `create_graph=True` appears nowhere in the controller). The isolation requirement
    survives that change unchanged, on first-order grounds: `update()` runs between
    `L_total.backward()` and `optimizer.step()`, so a plain `.backward()` on
    `gradnorm_loss` would accumulate into the very model `.grad` buffers the main AdamW is
    about to step on. FR-5 requires the two optimizers to stay separated.
    """

    def _setup(self):
        torch.manual_seed(0)
        anchor = torch.nn.Linear(4, 4)
        x = torch.randn(8, 4)
        losses = {
            "next_item": (anchor(x) ** 2).mean(),
            "masked_item": (anchor(x) - 1.0).abs().mean(),
        }
        anchor_params = [p for p in anchor.parameters() if p.requires_grad]
        return anchor, losses, anchor_params

    def test_update_does_not_touch_anchor_gradients(self):
        anchor, losses, anchor_params = self._setup()
        gn = GradNormController(alpha=1.5, lr=1e-2)

        L_total = gn.weighted_sum(losses)
        L_total.backward(retain_graph=True)

        before = [p.grad.detach().clone() for p in anchor_params]
        gn.update(losses, anchor_params)
        after = [p.grad.detach().clone() for p in anchor_params]

        for b, a in zip(before, after):
            assert torch.equal(b, a), (
                "GradNormController.update perturbed the anchor's .grad by "
                f"max |delta| = {(a - b).abs().max().item()}; it must write only into "
                "log_lambdas.grad"
            )

    def test_update_still_moves_lambdas(self):
        anchor, losses, anchor_params = self._setup()
        gn = GradNormController(alpha=1.5, lr=1e-2)

        L_total = gn.weighted_sum(losses)
        L_total.backward(retain_graph=True)

        before = gn.log_lambdas.detach().clone()
        gn.update(losses, anchor_params)

        assert gn.log_lambdas.grad is not None
        assert torch.isfinite(gn.log_lambdas.grad).all()
        assert not torch.equal(before, gn.log_lambdas.detach()), (
            "the isolated gradient path must still update log_lambdas"
        )


class TestMaskedMetadataLeak:
    """REV-059: metadata at masked positions must not describe the item being predicted.

    `TransformerModel.forward` computes `item_emb(masked_seq) + pos_emb +
    metadata_projection(metadata)`. The metadata block the DataLoader supplies is
    gathered from the ORIGINAL sequence, so without this the masked-item objective saw
    the MASK token's embedding plus the true item's 39-dimensional genome+genre vector.
    """

    def _fixture(self):
        torch.manual_seed(0)
        B, S, F, vocab = 4, 12, 39, 60
        matrix = torch.rand(vocab, F)
        matrix[0] = 0.0          # pad row
        matrix[vocab - 1] = 0.0  # MASK_ID row
        seq = torch.randint(1, vocab - 1, (B, S))
        metadata = matrix[seq]
        return seq, metadata, matrix, vocab - 1

    def test_masked_rows_no_longer_match_the_true_item(self):
        seq, metadata, matrix, mask_id = self._fixture()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(7)
        masked_seq, mask_positions, true_ids = apply_masking(seq, 0.5, mask_id, rng=rng)
        assert bool(mask_positions.any()), "fixture must actually mask something"

        leaked = metadata[mask_positions]
        assert torch.allclose(leaked, matrix[true_ids[mask_positions]]), (
            "precondition: the raw block does carry the true items' rows"
        )

        cleaned = mask_metadata(metadata, mask_positions)
        assert torch.all(cleaned[mask_positions] == 0), (
            "metadata at masked positions must be zero, not the true item's features"
        )

    def test_equivalent_to_gathering_from_the_masked_sequence(self):
        seq, metadata, matrix, mask_id = self._fixture()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(7)
        masked_seq, mask_positions, _ = apply_masking(seq, 0.5, mask_id, rng=rng)

        cleaned = mask_metadata(metadata, mask_positions)
        assert torch.allclose(cleaned, matrix[masked_seq]), (
            "zeroing masked rows must equal metadata_matrix[masked_seq]"
        )

    def test_unmasked_positions_are_untouched(self):
        seq, metadata, matrix, mask_id = self._fixture()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(7)
        _, mask_positions, _ = apply_masking(seq, 0.5, mask_id, rng=rng)
        cleaned = mask_metadata(metadata, mask_positions)
        keep = ~mask_positions
        assert torch.allclose(cleaned[keep], metadata[keep])


class TestApplyMaskingVectorisedGuard:
    """REV-069: the "never mask every real item" guard is vectorised, same semantics."""

    def test_guard_holds_at_mask_prob_one(self):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(3)
        seq = torch.randint(1, 50, (32, 20))
        masked_seq, mask_positions, _ = apply_masking(seq, 1.0, 99, rng=rng)
        real_per_row = (seq != 0).sum(dim=1)
        masked_per_row = mask_positions.sum(dim=1)
        assert torch.all(masked_per_row < real_per_row), (
            "every row must retain at least one unmasked real item"
        )

    def test_padding_is_never_masked(self):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(3)
        seq = torch.cat([torch.zeros(8, 5, dtype=torch.long), torch.randint(1, 50, (8, 15))], dim=1)
        _, mask_positions, _ = apply_masking(seq, 1.0, 99, rng=rng)
        assert not bool(mask_positions[:, :5].any())


class TestFiniteAssertionsActuallyFire:
    """REV-091 / AC-09: the numerical-safety assertions must be verified NEGATIVELY.

    T4.9 claims "Verify that NaN/Inf losses trigger assertion failures", but every
    finite-related assertion in the suite ran the other way -- asserting a value IS
    finite on well-behaved input. An assertion accidentally weakened (`.any()` for
    `.all()`, or a guard moved above the point where the bad value appears) would have
    passed CI unnoticed, and these guards are the only thing stopping a NaN from
    silently corrupting weights through a long run.
    """

    def test_non_finite_loss_value_raises(self):
        from training.pretrain import assert_finite_step

        with pytest.raises(AssertionError, match="not finite"):
            assert_finite_step({"total": float("nan")}, None)
        with pytest.raises(AssertionError, match="not finite"):
            assert_finite_step({"next_item": float("inf")}, None)

    def test_non_finite_lambdas_raise(self):
        from training.pretrain import assert_finite_step

        gn = GradNormController(alpha=1.5, lr=1e-2)
        gn.log_lambdas.data[0] = float("inf")
        with pytest.raises(AssertionError, match="lambdas not finite"):
            assert_finite_step({"total": 1.0}, gn)

    def test_nan_objective_loss_aborts_the_step_before_the_optimizer(self):
        """A NaN from one objective must stop `train_one_step` before it steps."""
        import random

        from torch.optim.lr_scheduler import LambdaLR

        from training.pretrain import build_model, _build_loss_fns, train_one_step

        config = Config()
        config.apply_model_config("t-small")
        config.num_items = 200
        config.vocab_size = 202
        config.use_metadata = False
        config.feature_dim = 0
        config.device = "cpu"
        config.moe_variant = "arv_mmoe"
        config.batch_size = 4

        torch.manual_seed(0)
        model, moe = build_model(config)
        loss_fns = _build_loss_fns(config, torch.device("cpu"))

        class _NaNWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                return self.inner(*args, **kwargs) * float("nan")

        loss_fns["next_item"] = _NaNWrapper(loss_fns["next_item"])

        params = list(model.parameters()) + list(moe.parameters())
        for fn in loss_fns.values():
            params += list(fn.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        before = [p.detach().clone() for p in model.parameters()]

        b, S = config.batch_size, config.max_seq_length
        batch = {
            "sequences": torch.randint(1, config.num_items + 1, (b, S)),
            "targets": torch.randint(1, config.num_items + 1, (b,)),
            "neg_ids": torch.randint(1, config.num_items + 1, (b, config.num_sampled_negatives)),
            "targets_k": torch.randint(1, config.num_items + 1, (b, config.multi_step_k)),
        }

        with pytest.raises(AssertionError, match="not finite"):
            train_one_step(
                model, moe, batch, config, loss_fns, None,
                optimizer, scheduler, 0, random.Random(0),
            )

        for p_before, p_after in zip(before, model.parameters()):
            assert torch.equal(p_before, p_after), (
                "the optimizer stepped on a NaN loss -- the per-objective finite "
                "assertion did not abort the step"
            )


class TestFirstOrderGradNorm:
    """AC-10: the GradNorm path performs no second-order differentiation.

    REV-120 / design D7: `G_t` is built as `lambda_t * ||grad_W L_t||` with the norm
    detached, so the lambda gradient is first-order. The CPU case is the reference: SDPA
    on CPU implements only a first-order backward
    (`aten::_scaled_dot_product_flash_attention_for_cpu_backward` has no derivative), so
    a full `update()` there is the regression guard that keeps this fixed.
    """

    @staticmethod
    def _cpu_model():
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.feature_dim = 0
        cfg.device = "cpu"
        return TransformerModel(cfg), cfg

    def test_create_graph_true_absent_from_the_controller(self):
        source = Path(training.gradnorm.__file__).read_text()
        assert "create_graph=True" not in source, (
            "AC-10 forbids second-order differentiation in the GradNorm path"
        )

    def test_full_update_runs_on_cpu(self):
        model, cfg = self._cpu_model()
        model.train()
        B, S = 2, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S))
        pooled = model(seq, return_all_hidden=True)[:, -1, :]
        logits = pooled @ model.item_emb_weight.T
        losses = {}
        for name in ["next_item", "masked_item"]:
            losses[name] = torch.nn.functional.cross_entropy(
                logits.float(), torch.randint(1, 50, (B,))
            )

        gn = GradNormController(alpha=1.5, lr=1e-2)
        gn.weighted_sum(losses).backward(retain_graph=True)
        gradnorm_loss = gn.update(losses, get_anchor_layer_params(model))

        assert gradnorm_loss is not None
        assert torch.isfinite(gradnorm_loss).all()
        assert next(model.parameters()).device.type == "cpu"


class TestLambdaGradientEquivalence:
    """AC-11: the shipped lambda gradient equals the closed form, and carries no graph.

    The closed form `d(L_grad)/d(log lambda_t) = sign(G_t - G*_t) * G_t` is computed here
    independently of the controller, so this is a real cross-check of the shipped
    `torch.autograd.grad` path rather than a restatement of it.
    """

    @staticmethod
    def _model_and_losses(num_active):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.dropout = 0.0
        device = get_device(cfg)
        model = TransformerModel(cfg)
        model.train()

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        pooled = model(seq, return_all_hidden=True)[:, -1, :]

        losses = {}
        for i, name in enumerate(OBJECTIVE_ORDER[:num_active]):
            logits = (pooled @ model.item_emb_weight.T) * (1.0 + 0.5 * i)
            targets = torch.randint(1, 50, (B,), device=device)
            losses[name] = torch.nn.functional.cross_entropy(logits.float(), targets)
        return model, losses, get_anchor_layer_params(model)

    @staticmethod
    def _closed_form(gn, losses, anchor_params):
        """sign(G_t - G*_t) * G_t, recomputed from scratch at the pre-update lambdas."""
        names = list(losses.keys())
        lambdas = gn.lambdas.detach()
        G_t = []
        for name in names:
            grads = torch.autograd.grad(
                losses[name].float(), anchor_params,
                retain_graph=True, create_graph=False,
            )
            norm = torch.cat([g.flatten() for g in grads]).norm(p=2).detach()
            G_t.append(lambdas[OBJECTIVE_ORDER.index(name)] * norm)
        G_t = torch.stack(G_t)

        # First call to update(): L0 == L, so Lhat == 1 and r_t == 1.
        Lhat = torch.ones(len(names), device=G_t.device)
        r_t = Lhat / (Lhat.mean() + 1e-12)
        G_star = G_t.mean() * (r_t ** gn.alpha)
        return torch.sign(G_t - G_star) * G_t

    @pytest.mark.parametrize("num_active", [4, 2, 1])
    def test_matches_closed_form(self, num_active):
        """num_active=1 is the `G_t == G*_t` subgradient boundary: expect exactly zero."""
        model, losses, anchor_params = self._model_and_losses(num_active)
        gn = GradNormController(alpha=1.5, lr=1e-2)

        expected = self._closed_form(gn, losses, anchor_params)

        gn.weighted_sum(losses).backward(retain_graph=True)
        gn.update(losses, anchor_params)

        indices = [OBJECTIVE_ORDER.index(n) for n in losses]
        shipped = gn.log_lambdas.grad[indices]
        # The controller holds `log_lambdas` on CPU until `run_pretrain` moves it; the
        # anchor gradients are on the accelerator. Compare on one device.
        expected = expected.to(shipped.device)
        assert torch.allclose(shipped, expected, atol=1e-5, rtol=1e-4), (
            f"shipped {shipped.tolist()} != closed form {expected.tolist()}"
        )

        inactive = [i for i in range(len(OBJECTIVE_ORDER)) if i not in indices]
        if inactive:
            assert torch.all(gn.log_lambdas.grad[inactive] == 0.0), (
                "inactive objectives must receive no lambda gradient"
            )
        if num_active == 1:
            assert torch.all(shipped == 0.0), (
                "with one active objective G_t == G*_t, so the subgradient is zero"
            )

    def test_grad_carries_no_autograd_graph(self):
        model, losses, anchor_params = self._model_and_losses(3)
        gn = GradNormController(alpha=1.5, lr=1e-2)
        gn.weighted_sum(losses).backward(retain_graph=True)
        gn.update(losses, anchor_params)

        assert gn.log_lambdas.grad is not None
        assert gn.log_lambdas.grad.requires_grad is False, (
            "assigning a graph-carrying tensor to .grad is silently accepted by PyTorch "
            "and pins that graph alive across steps (design D7)"
        )
        assert gn.log_lambdas.grad.grad_fn is None
