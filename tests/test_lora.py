import torch
import pytest

from configs.config import Config
from models.transformer import TransformerModel
from models.lora import (
    LoRALinear,
    inject_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    count_lora_parameters,
)
from utils import get_device


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 50
    cfg.dropout = 0.0
    return cfg


class TestLoRAInjection:
    def test_q_v_wrapped(self, base_config):
        model = TransformerModel(base_config)
        inject_lora(model, rank=8)

        for block in model.blocks:
            assert isinstance(block.attn.q_proj, LoRALinear)
            assert isinstance(block.attn.v_proj, LoRALinear)
            assert not isinstance(block.attn.k_proj, LoRALinear)
            assert not isinstance(block.attn.out_proj, LoRALinear)

    def test_forward_changes_after_training(self, base_config):
        device = get_device(base_config)
        model = TransformerModel(base_config)
        model.eval()

        B, S = 2, base_config.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)

        with torch.no_grad():
            out_before = model(seq, return_all_hidden=False).clone()

        inject_lora(model, rank=8)
        model.train()

        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(lora_params, lr=1e-2)

        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]
        targets = torch.randint(1, 50, (B,), device=device)
        loss = torch.nn.functional.cross_entropy(pooled @ model.item_emb_weight.T, targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_after = model(seq, return_all_hidden=False)

        assert not torch.allclose(out_before, out_after, atol=1e-6)


class TestLoRAStateDict:
    def test_no_moe_keys(self, base_config):
        base_config.moe_variant = "arv_mmoe"
        model = TransformerModel(base_config)
        inject_lora(model, rank=8)

        sd = get_lora_state_dict(model)
        for key in sd:
            assert not key.startswith("moe.")
            assert "lora_" in key

    def test_only_lora_keys(self, base_config):
        model = TransformerModel(base_config)
        inject_lora(model, rank=8)

        sd = get_lora_state_dict(model)
        assert len(sd) > 0
        for key in sd:
            assert "lora_A" in key or "lora_B" in key


class TestParameterCount:
    def test_count_matches_expected(self, base_config):
        model = TransformerModel(base_config)
        rank = 8
        inject_lora(model, rank=rank)

        D = base_config.hidden_dim
        num_layers = base_config.num_layers
        expected = 2 * num_layers * (D * rank + rank * D)

        actual = count_lora_parameters(model)
        assert actual == expected


class TestFreezeEnforcement:
    def test_zero_trainable_backbone(self, base_config):
        model = TransformerModel(base_config)
        inject_lora(model, rank=8)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in trainable_names:
            assert "lora_A" in name or "lora_B" in name

    def test_only_lora_requires_grad(self, base_config):
        model = TransformerModel(base_config)
        inject_lora(model, rank=8)

        backbone_trainable = 0
        lora_trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_" in name:
                    lora_trainable += 1
                else:
                    backbone_trainable += 1

        assert backbone_trainable == 0
        assert lora_trainable > 0
