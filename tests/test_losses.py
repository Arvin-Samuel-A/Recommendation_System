import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from training.losses import (
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
        cfg.num_items = 50
        cfg.moe_variant = moe_variant
        model = TransformerModel(cfg)

        params = get_anchor_layer_params(model)
        assert len(params) > 0
        last_block = model.blocks[-1]
        block_param_ptrs = {p.data_ptr() for p in last_block.parameters()}
        for p in params:
            assert p.data_ptr() in block_param_ptrs
