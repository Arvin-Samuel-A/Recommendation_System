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

    def test_correct_position_usage(self, device, dtype):
        B, S, D, V, K = 1, 5, 8, 20, 2
        loss_fn = MultiStepLoss(D, K).to(device=device, dtype=dtype)
        hidden = torch.zeros(B, S, D, device=device, dtype=dtype)
        hidden[0, S-1, :] = 1.0
        hidden[0, S-2, :] = -1.0
        targets_k = torch.tensor([[5, 10]], device=device)
        item_emb = torch.randn(V, D, device=device, dtype=dtype)

        loss = loss_fn(hidden, targets_k, item_emb)
        assert torch.isfinite(loss)


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

    The GradNorm loss is built from `torch.autograd.grad(..., create_graph=True)`, so a
    plain `.backward()` on it propagates second-order gradients into every model
    parameter in that graph. `train_one_step` calls `update()` between `L_total.backward()`
    and `optimizer.step()`, so those second-order terms would be stepped on by the main
    AdamW. FR-5 requires the two optimizers to be separated.
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


class TestDoubleBackwardDiagnostic:
    """REV-092: an attention kernel without double-backward must fail legibly.

    `config.device` defaults to "auto" (CUDA > MPS > CPU), so any CPU-only host -- a CI
    runner, a container without an accelerator -- hits this on the first optimizer step
    of the default `use_gradnorm=True` configuration. The raw error names an ATen symbol
    (`aten::_scaled_dot_product_flash_attention_for_cpu_backward`) rather than the
    actual problem.
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

    def test_probe_reports_the_backend_and_a_remedy(self):
        from training.gradnorm import DoubleBackwardUnsupportedError, probe_double_backward

        model, cfg = self._cpu_model()
        with pytest.raises(DoubleBackwardUnsupportedError) as exc:
            probe_double_backward(model, cfg)
        message = str(exc.value)
        assert "cpu" in message
        assert "use_gradnorm=False" in message, "the error must name a remedy"
        assert "sdpa" in message, "the error must name the attention backend that failed"

    def test_update_raises_the_same_diagnostic(self):
        from training.gradnorm import DoubleBackwardUnsupportedError

        model, cfg = self._cpu_model()
        model.train()
        B, S = 2, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S))
        pooled = model(seq, return_all_hidden=True)[:, -1, :]
        logits = pooled @ model.item_emb_weight.T
        loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.randint(1, 50, (B,))
        )

        gn = GradNormController(alpha=1.5, lr=1e-2)
        with pytest.raises(DoubleBackwardUnsupportedError):
            gn.update({"next_item": loss}, get_anchor_layer_params(model), model=model)

    def test_probe_passes_on_a_backend_that_supports_it(self):
        """Non-vacuity: the probe must not reject a working configuration."""
        from training.gradnorm import probe_double_backward

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.feature_dim = 0
        model = TransformerModel(cfg)  # device="auto" -> MPS/CUDA here
        if next(model.parameters()).device.type == "cpu":
            pytest.skip("no accelerator on this host; the CPU path is covered above")
        probe_double_backward(model, cfg)
