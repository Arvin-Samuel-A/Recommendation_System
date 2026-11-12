from pathlib import Path

import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from training.losses import (
    mask_metadata,
    NextItemLoss,
    MaskedItemLoss,
    MultiStepLoss,
    LogitScale,
    full_softmax_cross_entropy,
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


class TestFullSoftmaxCrossEntropy:
    def test_chunked_matches_unchunked_value_and_gradients(self, monkeypatch):
        import training.losses as losses

        torch.manual_seed(0)
        h = torch.randn(300, 16, requires_grad=True)
        w = torch.randn(40, 16, requires_grad=True)
        t = torch.randint(0, 40, (300,))
        scale = LogitScale(0.5, learnable=True)

        ref = torch.nn.functional.cross_entropy(scale(h @ w.T), t)
        ref_grads = torch.autograd.grad(ref, [h, w, scale.log_scale])

        # 40 classes x 32 rows per chunk -> 10 chunks, the last one short.
        monkeypatch.setattr(losses, "FULL_SOFTMAX_CHUNK_ELEMENTS", 40 * 32)
        monkeypatch.setattr(losses, "FULL_SOFTMAX_UNCHECKPOINTED_ELEMENTS", 0)
        chunked = full_softmax_cross_entropy(h, t, w, scale)
        assert torch.allclose(chunked, ref, atol=1e-6)
        # GradNorm differentiates the same loss more than once (retain_graph).
        first = torch.autograd.grad(chunked, [h, w, scale.log_scale], retain_graph=True)
        second = torch.autograd.grad(chunked, [h, w, scale.log_scale])
        for got, again, want in zip(first, second, ref_grads):
            assert torch.allclose(got, want, atol=1e-6)
            assert torch.equal(got, again)


    def test_small_calls_are_not_checkpointed_large_ones_are(self, monkeypatch):
        """A call above the threshold must not keep its logits for backward."""
        import training.losses as losses

        calls = []
        real = losses.checkpoint
        monkeypatch.setattr(losses, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
        h, w = torch.randn(10, 8), torch.randn(30, 8)
        t = torch.randint(0, 30, (10,))
        full_softmax_cross_entropy(h, t, w, LogitScale())
        assert not calls
        monkeypatch.setattr(losses, "FULL_SOFTMAX_UNCHECKPOINTED_ELEMENTS", 100)
        full_softmax_cross_entropy(h, t, w, LogitScale())
        assert len(calls) == 1


class TestNextItemLoss:
    def test_every_real_position_is_a_target_and_padding_is_ignored(self):
        torch.manual_seed(0)
        B, S, D, V = 3, 5, 8, 30
        hidden = torch.randn(B, S, D)
        item_emb = torch.randn(V, D)
        targets = torch.randint(1, V, (B, S))
        targets[0, :2] = 0      # left padding
        loss_fn = NextItemLoss(normalize_embeddings=True)

        valid = targets != 0
        h = torch.nn.functional.normalize(hidden[valid], dim=-1)
        w = torch.nn.functional.normalize(item_emb, dim=-1)
        expected = torch.nn.functional.cross_entropy(h @ w.T, targets[valid])
        assert torch.allclose(loss_fn(hidden, targets, item_emb), expected, atol=1e-6)

        # Padded positions do not matter, whatever their hidden state.
        moved = hidden.clone()
        moved[0, :2] += 5.0
        assert torch.allclose(loss_fn(moved, targets, item_emb), expected, atol=1e-6)

    def test_no_targets_returns_detached_zero(self):
        loss = NextItemLoss()(torch.randn(2, 4, 8), torch.zeros(2, 4, dtype=torch.long),
                              torch.randn(10, 8))
        assert loss.item() == 0.0 and not loss.requires_grad


class TestMultiStepLoss:
    def test_valid_targets(self, device, dtype):
        B, S, D, V, K = 4, 10, 16, 50, 3
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        targets = torch.randint(1, V, (B, S), device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss = loss_fn(hidden, targets, item_emb)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_all_zero_targets(self, device, dtype):
        B, S, D, V, K = 4, 10, 16, 50, 3
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        targets = torch.zeros(B, S, dtype=torch.long, device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss = loss_fn(hidden, targets, item_emb)
        assert loss.item() == 0.0

    def test_step_s_predicts_horizon_s_plus_2_from_every_position(self):
        """Position t, step s targets the item s + 2 steps ahead: `targets[:, t + s + 1]`."""
        torch.manual_seed(0)
        B, S, D, V, K = 2, 6, 8, 20, 3
        loss_fn = MultiStepLoss(D, K)
        hidden = torch.randn(B, S, D)
        targets = torch.randint(1, V, (B, S))
        targets[1, :2] = 0      # left padding on row 1
        item_emb = torch.randn(V, D)

        per_step = []
        for step in range(K):
            shift = step + 1
            valid = (targets[:, :S - shift] != 0) & (targets[:, shift:] != 0)
            logits = loss_fn.step_projections[step](hidden[:, :S - shift][valid]) @ item_emb.T
            per_step.append(torch.nn.functional.cross_entropy(logits, targets[:, shift:][valid]))
        expected = torch.stack(per_step).mean()
        assert torch.allclose(loss_fn(hidden, targets, item_emb), expected, atol=1e-6)

    def test_never_reads_a_position_after_its_horizon_targets(self):
        """The last position has no horizon target inside the chunk, so it is unused."""
        torch.manual_seed(0)
        B, S, D, V, K = 2, 6, 8, 20, 3
        loss_fn = MultiStepLoss(D, K)
        hidden = torch.randn(B, S, D)
        targets = torch.randint(1, V, (B, S))
        item_emb = torch.randn(V, D)
        baseline = loss_fn(hidden, targets, item_emb)
        moved = hidden.clone()
        moved[:, S - 1] += 3.0
        assert loss_fn(moved, targets, item_emb).item() == baseline.item()
        moved[:, 0] += 3.0
        assert loss_fn(moved, targets, item_emb).item() != baseline.item()


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
            "targets": torch.randint(1, config.num_items + 1, (b, S)),
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


class TestVectorisedRenormalisation:
    """specs/ablation-run T5.2 / AC-13: the branch-free renorm equals the old loop."""

    @staticmethod
    def _old_loop(log_lambdas, active_indices):
        """Verbatim from HEAD 0a91094 (`GradNormController.update`)."""
        lambdas = lambda: log_lambdas.exp()  # noqa: E731
        num_active = len(active_indices)
        with torch.no_grad():
            active_lams = lambdas()[active_indices]
            renorm = active_lams.sum()
            if renorm > 0:
                scale = num_active / renorm
                for idx in active_indices:
                    log_lambdas[idx] = (lambdas()[idx] * scale).log()

    def test_matches_the_per_index_loop_bit_for_bit(self):
        import random

        rnd = random.Random(0)
        gen = torch.Generator().manual_seed(0)
        for trial in range(100):
            gn = GradNormController(alpha=1.5, lr=1e-2)
            start = torch.randn(4, generator=gen) * (3.0 if trial % 2 else 0.5)
            k = rnd.randint(1, 4)
            active = sorted(rnd.sample(range(4), k))
            expected = start.clone()
            self._old_loop(expected, active)
            with torch.no_grad():
                gn.log_lambdas.copy_(start)
            gn._renormalize(active)
            assert torch.equal(gn.log_lambdas.detach(), expected), (trial, active)


class TestMaskedSequenceOnlyFeedsMaskedItem:
    """Owner decision 2026-10-25: masking corrupts only the masked-item objective's input."""

    def _step(self, use_masked_lm):
        import random

        from torch.optim.lr_scheduler import LambdaLR

        from training.pretrain import _build_loss_fns, build_model, train_one_step

        config = Config()
        config.apply_model_config("t-small")
        config.num_items, config.vocab_size = 200, 202
        config.use_metadata, config.feature_dim = False, 0
        config.device, config.dropout = "cpu", 0.0
        config.batch_size, config.max_seq_length = 4, 12
        config.use_masked_lm, config.mask_prob = use_masked_lm, 0.5
        config.use_gradnorm = False
        torch.manual_seed(0)
        model, moe = build_model(config)
        loss_fns = _build_loss_fns(config, torch.device("cpu"))
        params = list(model.parameters()) + list(moe.parameters())
        for fn in loss_fns.values():
            params += list(fn.parameters())
        optimizer = torch.optim.SGD(params, lr=0.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        gen = torch.Generator().manual_seed(1)
        batch = {
            "sequences": torch.randint(1, 201, (4, 12), generator=gen),
            "targets": torch.randint(1, 201, (4, 12), generator=gen),
        }
        values, _ = train_one_step(
            model, moe, batch, config, loss_fns, None, optimizer, scheduler, 0,
            random.Random(0), mask_rng=torch.Generator().manual_seed(2),
        )
        return values

    def test_clean_objectives_are_unchanged_by_masking(self):
        with_mim, without_mim = self._step(True), self._step(False)
        assert "masked_item" in with_mim and "masked_item" not in without_mim
        for name in ("next_item", "multi_step", "contrastive"):
            assert torch.equal(with_mim[name], without_mim[name]), name


class TestContrastivePoolsTheSequence:
    """Contrastive reads the mean over real positions, so its gradient reaches every real
    position equally and never concentrates on the last (scored) one."""

    def test_mean_pool_ignores_padding(self):
        from training.losses import mean_pool_real

        hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        seq = torch.tensor([[0, 5, 6], [7, 8, 9]])
        out = mean_pool_real(hidden, seq)
        assert torch.allclose(out[0], hidden[0, 1:].mean(0))
        assert torch.allclose(out[1], hidden[1].mean(0))

    def test_gradient_is_spread_over_real_positions(self):
        from training.losses import mean_pool_real

        torch.manual_seed(0)
        hidden = torch.randn(4, 6, 8, requires_grad=True)
        seq = torch.randint(1, 50, (4, 6))
        seq[0, :2] = 0
        z = torch.nn.functional.normalize(mean_pool_real(hidden, seq), dim=-1)
        z_aug = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        InfoNCELoss(0.07)(z, z_aug).backward()
        g = hidden.grad.norm(dim=-1)
        assert torch.all(g[0, :2] == 0)
        for row in range(4):
            real = g[row][seq[row] != 0]
            assert torch.allclose(real, real[0].expand_as(real)), row

    def test_train_step_feeds_contrastive_the_pooled_views(self, monkeypatch):
        """The last position alone must not decide the contrastive loss."""
        import random

        from torch.optim.lr_scheduler import LambdaLR

        import training.pretrain as P

        seen = []
        real = P.mean_pool_real
        monkeypatch.setattr(P, "mean_pool_real", lambda h, s: seen.append(h.shape) or real(h, s))
        config = Config()
        config.apply_model_config("t-small")
        config.num_items, config.vocab_size = 200, 202
        config.use_metadata, config.feature_dim = False, 0
        config.device, config.batch_size, config.max_seq_length = "cpu", 4, 12
        config.use_gradnorm, config.contrastive_interval = False, 1
        torch.manual_seed(0)
        model, moe = P.build_model(config)
        loss_fns = P._build_loss_fns(config, torch.device("cpu"))
        params = P.collect_trainable_parameters(model, moe, loss_fns)
        optimizer = torch.optim.SGD(params, lr=0.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        batch = {"sequences": torch.randint(1, 201, (4, 12)), "targets": torch.randint(1, 201, (4, 12))}
        values, _ = P.train_one_step(model, moe, batch, config, loss_fns, None, optimizer,
                                     scheduler, 0, random.Random(0))
        assert "contrastive" in values
        # Routing input and representation, for the original and the augmented view.
        assert len(seen) == 4
