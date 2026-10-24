import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from models.transformer import TransformerModel
from models.moe import MoELayer
from training.losses import (
    NextItemLoss,
    MaskedItemLoss,
    MultiStepLoss,
    InfoNCELoss,
    apply_masking,
)
from training.gradnorm import GradNormController, get_anchor_layer_params
from data.augmentation import augment_sequence
from utils import get_device, resolve_dtype


class TestOneBatchEndToEnd:
    def test_full_pipeline(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        # REV-062 (resolved, T5.2): this is the BUILD_ORDER Phase 16 gate that helps
        # authorise full MovieLens training, and `steering/code-quality.md` is explicit --
        # "MPS smoke test must fail fast if SDPA cannot execute -- no silent fallback."
        # The arming stays: with the default `require_attention_backend=False` the gate
        # would pass on whatever kernel happened to be substituted, certifying something
        # other than what the manifest names. SDPA is now the sanctioned MPS/CPU kernel
        # (FR-4, design D2b), so the armed gate passes because the preferred backend
        # genuinely executes -- not because the requirement was relaxed.
        cfg.require_attention_backend = True
        cfg.num_items = 100
        cfg.feature_dim = 40
        cfg.moe_variant = "arv_mmoe"
        cfg.dropout = 0.0
        cfg.use_gradnorm = True
        cfg.multi_step_k = 3

        device = get_device(cfg)
        dtype = resolve_dtype(cfg)
        D = cfg.hidden_dim
        V = cfg.num_items + 2
        mask_id = cfg.num_items + 1

        model = TransformerModel(cfg)
        model.train()

        moe = MoELayer(cfg).to(device=device, dtype=dtype)
        moe.train()

        next_loss_fn = NextItemLoss()
        mim_loss_fn = MaskedItemLoss()
        ms_loss_fn = MultiStepLoss(D, cfg.multi_step_k).to(device=device, dtype=dtype)
        cl_loss_fn = InfoNCELoss(temperature=cfg.contrastive_temperature)

        gn = GradNormController(alpha=cfg.gradnorm_alpha, lr=cfg.gradnorm_lr)

        B, S = 8, cfg.max_seq_length
        seq = torch.randint(1, cfg.num_items + 1, (B, S), device=device)
        seq[:, :3] = 0
        targets = torch.randint(1, cfg.num_items + 1, (B, S), device=device)
        targets[:, :3] = 0
        metadata = torch.randn(B, S, cfg.feature_dim, device=device, dtype=dtype)

        masked_seq, mask_positions, true_ids = apply_masking(seq, cfg.mask_prob, mask_id)

        all_hidden = model(masked_seq, metadata=metadata, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        active_losses = {}

        moe.clear_cache()
        Z_next, r_next = moe(all_hidden, all_hidden, 0)
        active_losses["next_item"] = next_loss_fn(Z_next, targets, model.item_emb_weight)

        Z_mim, r_mim = moe(all_hidden, pooled, 1)
        active_losses["masked_item"] = mim_loss_fn(Z_mim, mask_positions, true_ids, model.item_emb_weight)

        Z_ms, r_ms = moe(all_hidden, all_hidden, 2)
        active_losses["multi_step"] = ms_loss_fn(Z_ms, targets, model.item_emb_weight)

        Z_cl, r_cl = moe(all_hidden, pooled, 3)
        z_orig = torch.nn.functional.normalize(Z_cl[:, -1, :].float(), p=2, dim=-1)

        import random
        aug_seqs = []
        rng = random.Random(42)
        for i in range(B):
            real = seq[i][seq[i] != 0].tolist()
            aug = augment_sequence(real, mask_id, rng, cfg.crop_ratio, cfg.mask_ratio)
            padded = [0] * (S - len(aug)) + aug
            padded = padded[-S:]
            aug_seqs.append(padded)
        aug_seq_t = torch.tensor(aug_seqs, device=device)

        aug_hidden = model(aug_seq_t, return_all_hidden=True)
        aug_pooled = aug_hidden[:, -1, :]
        moe.clear_cache()
        Z_aug, _ = moe(aug_hidden, aug_pooled, 3)
        z_aug = torch.nn.functional.normalize(Z_aug[:, -1, :].float(), p=2, dim=-1)

        active_losses["contrastive"] = cl_loss_fn(z_orig.to(dtype), z_aug.to(dtype))

        for name, loss in active_losses.items():
            assert torch.isfinite(loss), f"{name} loss is not finite: {loss.item()}"

        anchor_params = get_anchor_layer_params(model)
        L_total = gn.weighted_sum(active_losses)
        assert torch.isfinite(L_total), f"Total loss not finite: {L_total.item()}"

        L_total.backward(retain_graph=True)
        gn_loss = gn.update(active_losses, anchor_params)

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf"
        assert has_grad

        active_indices = [OBJECTIVE_ORDER.index(n) for n in active_losses]
        lam_sum = gn.lambdas[active_indices].sum().item()
        assert abs(lam_sum - len(active_losses)) < 1e-3

        assert r_next is not None
        assert r_mim is not None
        assert not torch.allclose(r_next[:, -1], r_mim, atol=1e-5)

    def test_none_variant_pipeline(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 100
        cfg.feature_dim = 0
        cfg.use_metadata = False
        cfg.moe_variant = "none"
        cfg.dropout = 0.0

        device = get_device(cfg)
        dtype = resolve_dtype(cfg)
        D = cfg.hidden_dim
        mask_id = cfg.num_items + 1

        model = TransformerModel(cfg)
        model.train()
        moe = MoELayer(cfg)

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, cfg.num_items + 1, (B, S), device=device)
        targets = torch.randint(1, cfg.num_items + 1, (B, S), device=device)

        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        Z, r = moe(all_hidden, pooled, 0)
        assert r is None
        assert torch.equal(Z, all_hidden)

        loss = NextItemLoss()(Z, targets, model.item_emb_weight)
        assert torch.isfinite(loss)

        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
