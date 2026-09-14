import torch
import pytest

from configs.config import Config, MODEL_CONFIGS
from models.transformer import TransformerModel, AttentionBackend
from utils import get_device, resolve_dtype


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.num_items = 100
    cfg.feature_dim = 40
    cfg.max_seq_length = 10
    cfg.dropout = 0.0
    return cfg


class TestAttentionBackend:
    def test_self_test_passes(self, base_config):
        backend = AttentionBackend(base_config)
        result = backend.self_test()
        assert result["status"] == "passed"

    def test_causal_masking(self, base_config):
        device = get_device(base_config)
        dtype = resolve_dtype(base_config)
        B, H, S, D = 1, 1, 4, 8
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.zeros(B, H, S, D, device=device, dtype=dtype)
        v[0, 0, 3, :] = 1.0

        backend = AttentionBackend(base_config)
        out = backend.forward(q, k, v, is_causal=True)
        assert out[0, 0, 0, :].abs().max().item() < 1e-3

    def test_padding_masking(self, base_config):
        base_config.apply_model_config("t-small")
        base_config.num_items = 100
        base_config.feature_dim = 40
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        seq_padded = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8]], device=device)
        seq_no_pad = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=device)

        with torch.no_grad():
            out1 = model(seq_padded, return_all_hidden=True)
            out2 = model(seq_no_pad, return_all_hidden=True)
        assert out1.shape == out2.shape


class TestModelShapes:
    @pytest.mark.parametrize("model_name", ["t-small", "t-base", "t-large"])
    def test_output_shapes(self, base_config, model_name):
        base_config.apply_model_config(model_name)
        base_config.num_items = 100
        base_config.feature_dim = 40
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        B, S = 4, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)

        with torch.no_grad():
            pooled = model(seq, return_all_hidden=False)
            assert pooled.shape == (B, base_config.hidden_dim)

            all_hidden = model(seq, return_all_hidden=True)
            assert all_hidden.shape == (B, S, base_config.hidden_dim)


class TestTiedWeights:
    def test_item_embedding_is_output_weight(self, base_config):
        base_config.apply_model_config("t-small")
        model = TransformerModel(base_config)
        assert model.item_emb_weight is model.item_embedding.weight


class TestMetadataIntegration:
    def test_metadata_changes_output(self, base_config):
        base_config.apply_model_config("t-small")
        base_config.use_metadata = True
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        B, S = 2, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)
        meta = torch.randn(B, S, base_config.feature_dim, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            out_with = model(seq, metadata=meta, return_all_hidden=False)
            out_without = model(seq, metadata=None, return_all_hidden=False)
        assert not torch.allclose(out_with, out_without, atol=1e-3)

    def test_no_metadata_ignores_tensor(self, base_config):
        base_config.apply_model_config("t-small")
        base_config.use_metadata = False
        base_config.feature_dim = 0
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        B, S = 2, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)

        with torch.no_grad():
            out1 = model(seq, return_all_hidden=False)
            out2 = model(seq, return_all_hidden=False)
        assert torch.allclose(out1, out2)


class TestBF16:
    def test_parameters_are_bf16(self, base_config):
        base_config.apply_model_config("t-small")
        model = TransformerModel(base_config)
        for name, param in model.named_parameters():
            assert param.dtype == torch.bfloat16, f"Parameter {name} is {param.dtype}"

    def test_output_is_bf16(self, base_config):
        base_config.apply_model_config("t-small")
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        seq = torch.randint(1, 50, (2, base_config.max_seq_length), device=device)

        with torch.no_grad():
            out = model(seq, return_all_hidden=False)
        assert out.dtype == torch.bfloat16


class TestEmbeddingNormalization:
    def test_scores_bounded(self, base_config):
        base_config.apply_model_config("t-small")
        base_config.normalize_embeddings = True
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        seq = torch.randint(1, 50, (4, base_config.max_seq_length), device=device)

        with torch.no_grad():
            pooled = model(seq, return_all_hidden=False)
            item_embs = torch.nn.functional.normalize(model.item_emb_weight, p=2, dim=-1)
            scores = pooled @ item_embs.T

        assert scores.max().item() <= 1.0 + 1e-3
        assert scores.min().item() >= -1.0 - 1e-3


class TestSmoke:
    def test_one_batch_forward_backward(self, base_config):
        base_config.apply_model_config("t-small")
        model = TransformerModel(base_config)
        model.train()

        device = get_device(base_config)
        B, S = 4, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)
        targets = torch.randint(1, base_config.num_items + 1, (B,), device=device)

        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]
        logits = pooled @ model.item_emb_weight.T
        loss = torch.nn.functional.cross_entropy(logits, targets)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                assert torch.isfinite(p.grad).all(), "Gradient contains NaN/Inf"
        assert has_grad, "No gradients computed"
