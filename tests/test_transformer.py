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

    def test_padding_mask_zeros_padded_key_contributions(self, base_config):
        """REV-080 / T1.2: padded key positions contribute exactly nothing.

        The two tests this replaces compared a tensor against ITSELF -- both sequences
        were `[[0, 0] + suffix]`, byte-identical, since padding is always id 0 -- so they
        held for any deterministic model whether or not padding was masked. Assert the
        property at the layer that builds the mask: put a distinctive value at ONLY the
        padded positions and require the output at every real query position to be zero.
        """
        from models.transformer import CausalSelfAttention

        device, dtype = get_device(base_config), resolve_dtype(base_config)
        B, S, D, H = 1, 6, 8, 2
        backend = AttentionBackend(base_config)
        attn = CausalSelfAttention(D, H, 0.0, backend).to(device=device, dtype=dtype)
        attn.eval()

        # Identity value/output projections so `v` is exactly the input and the layer's
        # output is exactly the attention mixture -- nothing else can hide a leak.
        with torch.no_grad():
            for proj in (attn.v_proj, attn.out_proj):
                proj.weight.copy_(torch.eye(D, device=device, dtype=dtype))
                proj.bias.zero_()

        padding_mask = torch.tensor(
            [[False, False, True, True, True, True]], device=device
        )
        x = torch.zeros(B, S, D, device=device, dtype=dtype)
        x[0, :2, :] = 100.0  # the distinctive value sits ONLY at padded positions

        with torch.no_grad():
            masked = attn(x, padding_mask=padding_mask)

        real = masked[0, 2:, :]
        assert torch.isfinite(real).all(), "real query positions must stay finite"
        assert real.abs().max().item() == 0.0, (
            "a padded key leaked into a real query position -- the padding mask at "
            "CausalSelfAttention.forward is not excluding pad keys"
        )

        # Non-vacuity: without the padding mask the same value DOES reach real queries,
        # so the assertion above is testing the mask and not the arithmetic.
        with torch.no_grad():
            unmasked = attn(x, padding_mask=None)
        assert unmasked[0, 2:, :].abs().max().item() > 0.0, (
            "without the padding mask the padded value must reach real queries; if it "
            "does not, the test above proves nothing"
        )

    def test_padding_mask_marks_every_pad_position(self, base_config):
        """The model-level half: `seq != 0` is what feeds the layer's mask, and a real
        item in the prefix must change the output (otherwise masking would be inert)."""
        base_config.apply_model_config("t-small")
        base_config.num_items = 100
        base_config.feature_dim = 40
        base_config.max_seq_length = 10
        model = TransformerModel(base_config)
        model.eval()
        device = get_device(base_config)

        items = [3, 9, 14, 27]
        seq_pad = torch.tensor([[0] * 6 + items], device=device)
        assert not bool((seq_pad != 0)[0, :6].any()), "pad positions must be False"

        seq_junk = torch.tensor([[7] * 6 + items], device=device)
        with torch.no_grad():
            a = model(seq_pad, return_all_hidden=True)[:, -1, :]
            c = model(seq_junk, return_all_hidden=True)[:, -1, :]
        assert not torch.equal(a.float(), c.float()), (
            "if real items in the prefix produced the same output as padding, the "
            "attention mask would be doing nothing"
        )


class TestAttentionBackendGate:
    """REV-062: `require_attention_backend=True` must turn an unavailable preferred
    backend into a hard failure, as `steering/code-quality.md` requires ("no silent
    fallback"). The gate exists and works; nothing covered it."""

    def test_require_backend_raises_when_preferred_is_unavailable(self, base_config, monkeypatch):
        from models.transformer import AttentionBackendError

        base_config.attention_backend = "auto"
        base_config.require_attention_backend = True

        def always_fails(self, backend):
            return False, "probe forced to fail for this test"

        monkeypatch.setattr(AttentionBackend, "_probe", always_fails)
        with pytest.raises(AttentionBackendError, match="forbids falling back"):
            AttentionBackend(base_config)

    def test_fallback_is_recorded_when_not_required(self, base_config, monkeypatch):
        """Without the flag the fallback is permitted -- but never silent: the resolved
        backend and the reason must both be reportable into the run manifest."""
        preferred_calls = {"n": 0}
        real_probe = AttentionBackend._probe

        def fail_preferred(self, backend):
            if backend != "sdpa":
                preferred_calls["n"] += 1
                return False, "probe forced to fail for this test"
            return real_probe(self, backend)

        base_config.attention_backend = "auto"
        base_config.require_attention_backend = False
        monkeypatch.setattr(AttentionBackend, "_probe", fail_preferred)

        backend = AttentionBackend(base_config)
        result = backend.self_test()
        assert result["backend"] == "sdpa"
        assert result["requested_backend"] == "auto"
        assert result["fallback_reason"], "the fallback reason must be recorded"

    def test_explicit_unavailable_backend_raises(self, base_config, monkeypatch):
        from models.transformer import AttentionBackendError

        base_config.attention_backend = "flash_attention_4"
        base_config.require_attention_backend = False

        def always_fails(self, backend):
            return False, "probe forced to fail for this test"

        monkeypatch.setattr(AttentionBackend, "_probe", always_fails)
        with pytest.raises(AttentionBackendError, match="Refusing to fall back"):
            AttentionBackend(base_config)


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
        """REV-102 / AC-06: with `use_metadata=False` the output must be identical
        regardless of the metadata tensor.

        This previously never supplied a metadata tensor -- it called the model twice
        with no metadata and asserted the two agreed -- so it covered neither the spec's
        behaviour nor the implementation's.
        """
        base_config.apply_model_config("t-small")
        base_config.use_metadata = False
        base_config.feature_dim = 0
        model = TransformerModel(base_config)
        model.eval()

        device = get_device(base_config)
        dtype = resolve_dtype(base_config)
        B, S = 2, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)
        meta = torch.randn(B, S, 40, device=device, dtype=dtype)

        with torch.no_grad():
            out_none = model(seq, metadata=None, return_all_hidden=False)
            out_meta = model(seq, metadata=meta, return_all_hidden=False)
        assert torch.equal(out_none, out_meta), (
            "supplying metadata to a use_metadata=False model must not affect "
            "computation (requirements.md Edge Cases, AC-06)"
        )

    def test_no_metadata_warns_once_when_tensor_supplied(self, base_config, caplog):
        """The tensor is ignored, but not silently -- one warning per process."""
        import logging

        base_config.apply_model_config("t-small")
        base_config.use_metadata = False
        base_config.feature_dim = 0
        model = TransformerModel(base_config)
        model.eval()
        TransformerModel._warned_metadata_ignored = False

        device = get_device(base_config)
        dtype = resolve_dtype(base_config)
        B, S = 1, base_config.max_seq_length
        seq = torch.randint(1, 50, (B, S), device=device)
        meta = torch.randn(B, S, 40, device=device, dtype=dtype)

        with caplog.at_level(logging.WARNING, logger="models.transformer"):
            with torch.no_grad():
                model(seq, metadata=meta)
                model(seq, metadata=meta)
        warnings = [r for r in caplog.records if "ignored" in r.message]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


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


class TestFlashAttentionVarlen:
    """REV-081: the FA4 path raised on any padded batch, so CUDA could not train.

    `CausalSelfAttention.forward` passes `padding_mask` whenever the batch contains a
    pad token, and every training batch is left-padded, so the first real batch on the
    H100 aborted the run (FR-4, AC-03). Padded batches now go through the variable-length
    API.

    The CUDA kernel itself cannot run on this host, so it is replaced with an exact SDPA
    reference. What that leaves under test is everything this repository owns: the mask
    convention, the cumulative sequence lengths, the ordering assumption that makes
    left-padding safe to strip, and the repadding.
    """

    @staticmethod
    def _install_reference_flash_attn(monkeypatch):
        import sys
        import types

        import torch
        import torch.nn.functional as F

        def unpad_input(hidden, attention_mask):
            batch, seqlen = attention_mask.shape
            seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            flat = hidden.reshape(batch * seqlen, *hidden.shape[2:])
            cu = torch.cat([
                torch.zeros(1, dtype=torch.int32),
                seqlens.cumsum(0, dtype=torch.int32),
            ])
            return flat[indices], indices, cu, int(seqlens.max()), None

        def pad_input(hidden, indices, batch, seqlen):
            out = torch.zeros(batch * seqlen, *hidden.shape[1:], dtype=hidden.dtype)
            out[indices] = hidden
            return out.reshape(batch, seqlen, *hidden.shape[1:])

        def flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            causal=False,
        ):
            outs = []
            for i in range(len(cu_seqlens_q) - 1):
                start, end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                qi, ki, vi = (
                    t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v)
                )
                out = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
                outs.append(out.squeeze(0).transpose(0, 1))
            return torch.cat(outs, dim=0)

        def flash_attn_func(q, k, v, causal=False):
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=causal,
            )
            return out.transpose(1, 2)

        module = types.ModuleType("flash_attn")
        module.flash_attn_func = flash_attn_func
        module.flash_attn_varlen_func = flash_attn_varlen_func
        bert_padding = types.ModuleType("flash_attn.bert_padding")
        bert_padding.unpad_input = unpad_input
        bert_padding.pad_input = pad_input
        module.bert_padding = bert_padding

        monkeypatch.setitem(sys.modules, "flash_attn", module)
        monkeypatch.setitem(sys.modules, "flash_attn.bert_padding", bert_padding)

    @staticmethod
    def _inputs():
        import torch

        torch.manual_seed(0)
        b, h, s, d = 3, 2, 8, 4
        q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
        # Left-padded rows of differing real lengths, as the data pipeline produces.
        padding_mask = torch.zeros(b, s, dtype=torch.bool)
        real_lens = [8, 5, 1]
        for i, length in enumerate(real_lens):
            padding_mask[i, s - length:] = True
        return q, k, v, padding_mask, real_lens, s

    def test_padded_batch_no_longer_raises(self, monkeypatch):
        from models.transformer import AttentionBackend

        self._install_reference_flash_attn(monkeypatch)
        q, k, v, padding_mask, _, _ = self._inputs()

        out = AttentionBackend._flash_attention_4(q, k, v, padding_mask, is_causal=True)
        assert out.shape == q.shape

    def test_varlen_matches_per_row_dense_causal_attention(self, monkeypatch):
        import torch
        import torch.nn.functional as F

        from models.transformer import AttentionBackend

        self._install_reference_flash_attn(monkeypatch)
        q, k, v, padding_mask, real_lens, s = self._inputs()

        got = AttentionBackend._flash_attention_4_varlen(
            q, k, v, padding_mask, is_causal=True,
        )

        for i, length in enumerate(real_lens):
            rows = slice(s - length, s)
            expected = F.scaled_dot_product_attention(
                q[i:i + 1, :, rows, :], k[i:i + 1, :, rows, :],
                v[i:i + 1, :, rows, :], is_causal=True,
            )[0]
            assert torch.allclose(got[i, :, rows, :], expected, atol=1e-5), (
                f"row {i} (real length {length}) does not match dense attention"
            )
            padded = got[i, :, :s - length, :]
            if padded.numel():
                assert padded.abs().max() == 0, f"row {i} wrote into its padding"

    def test_unpadded_batch_still_uses_the_dense_kernel(self, monkeypatch):
        import torch
        import torch.nn.functional as F

        from models.transformer import AttentionBackend

        self._install_reference_flash_attn(monkeypatch)
        q, k, v, _, _, s = self._inputs()
        full = torch.ones(q.shape[0], s, dtype=torch.bool)

        got = AttentionBackend._flash_attention_4(q, k, v, full, is_causal=True)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_missing_varlen_api_raises_a_clear_error(self, monkeypatch):
        import sys
        import types

        import torch

        from models.transformer import AttentionBackend, AttentionBackendError

        # A flash-attn build without the variable-length API must say so, not fail
        # obscurely deep in the call.
        module = types.ModuleType("flash_attn")
        module.flash_attn_func = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "flash_attn", module)
        monkeypatch.setitem(sys.modules, "flash_attn.bert_padding", None)

        q, k, v, padding_mask, _, _ = self._inputs()
        with pytest.raises(AttentionBackendError, match="variable-length"):
            AttentionBackend._flash_attention_4_varlen(
                q, k, v, padding_mask, is_causal=True,
            )
