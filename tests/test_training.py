import os
import torch
import pytest

from configs.config import Config
from models.transformer import TransformerModel
from models.moe import MoELayer
from training.pretrain import (
    build_scheduler,
    build_model,
    save_checkpoint,
    load_checkpoint,
)
from training.gradnorm import GradNormController
from utils import get_device, resolve_dtype


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 50
    cfg.feature_dim = 0
    cfg.use_metadata = False
    cfg.dropout = 0.0
    return cfg


class TestLRScheduler:
    def test_warmup_increases(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-8

    def test_cosine_reaches_near_zero(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)

        for _ in range(100):
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < 1e-5

    def test_per_batch_stepping(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=50)

        lr_at_0 = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        lr_at_1 = optimizer.param_groups[0]["lr"]
        assert lr_at_1 != lr_at_0


class TestCheckpointRoundtrip:
    def test_save_load_identical_output(self, base_config, tmp_path):
        device = get_device(base_config)
        model, moe = build_model(base_config)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(moe.parameters()), lr=1e-3,
        )
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)
        gn = GradNormController(alpha=1.5, lr=1e-2)
        gn.initial_losses = {"next_item": 3.5, "masked_item": 2.1}

        B, S = 4, base_config.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)

        model.eval()
        with torch.no_grad():
            out_before = model(seq, return_all_hidden=False).clone()

        path = str(tmp_path / "test_ckpt.pt")
        val_metrics = {"ndcg": {"mean": 0.5}, "mrr": {"mean": 0.3}}
        save_checkpoint(path, 5, model, moe, optimizer, scheduler, gn, val_metrics, base_config)

        model2, moe2 = build_model(base_config)
        optimizer2 = torch.optim.AdamW(
            list(model2.parameters()) + list(moe2.parameters()), lr=1e-3,
        )
        scheduler2 = build_scheduler(optimizer2, base_config, total_steps=100)
        gn2 = GradNormController(alpha=1.5, lr=1e-2)

        ckpt = load_checkpoint(path, model2, moe2, optimizer2, scheduler2, gn2)

        model2.eval()
        with torch.no_grad():
            out_after = model2(seq, return_all_hidden=False)

        assert torch.allclose(out_before, out_after, atol=1e-6)
        assert ckpt["epoch"] == 5
        assert abs(ckpt["val_ndcg"] - 0.5) < 1e-6
        assert gn2.initial_losses["next_item"] == 3.5
