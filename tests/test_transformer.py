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
    fallback"). The gate exists and works; nothing covered it.

    The sanctioned preferred kernel is SDPA on MPS/CPU/non-Hopper CUDA and FlashAttention-3
    on Hopper CUDA (specs/ablation-run FR-2). These tests force `_probe` to fail rather than naming a kernel,
    so they exercise the gate itself and are unaffected by which backend is preferred.
    """

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
        backend and the reason must both be reportable into the run manifest.

        T1.3: the preferred kernel is pinned to FA3 here rather than left to the host.
        MPS and CPU now both PREFER sdpa, so relying on the host's preference would make
        the fallback path unreachable and the test vacuous -- it would assert that no
        fallback happened while claiming to test that one was recorded.
        """
        real_probe = AttentionBackend._probe

        def fail_preferred(self, backend):
            if backend != "sdpa":
                return False, "probe forced to fail for this test"
            return real_probe(self, backend)

        base_config.attention_backend = "auto"
        base_config.require_attention_backend = False
        monkeypatch.setattr(AttentionBackend, "_preferred_backend", lambda self: "flash_attention_3")
        monkeypatch.setattr(AttentionBackend, "_probe", fail_preferred)

        backend = AttentionBackend(base_config)
        result = backend.self_test()
        assert result["backend"] == "sdpa"
        assert result["requested_backend"] == "auto"
        assert result["fallback_reason"], "the fallback reason must be recorded"
        assert "flash_attention_3" in result["fallback_reason"], (
            "the reason must name the kernel that was substituted for"
        )

    def test_explicit_unavailable_backend_raises(self, base_config, monkeypatch):
        from models.transformer import AttentionBackendError

        base_config.attention_backend = "flash_attention_3"
        base_config.require_attention_backend = False

        def always_fails(self, backend):
            return False, "probe forced to fail for this test"

        monkeypatch.setattr(AttentionBackend, "_probe", always_fails)
        with pytest.raises(AttentionBackendError, match="Refusing to fall back"):
            AttentionBackend(base_config)


class TestNarrowedBackendSurface:
    """T1.4 / AC-09 / AC-02: FlexAttention is gone from the backend surface.

    It has no backward pass on MPS in the pinned PyTorch build, so with the MPS gate
    armed it could never satisfy the training gates on the declared dev host. Rather than
    keep a selectable backend that provably cannot train, `flex_attention` is rejected by
    config validation and no code path remains (FR-4 rationale, design D2b).
    """

    def test_flex_attention_is_rejected_by_config_validation(self):
        """AC-09: rejected, with a message naming ONLY the supported choices."""
        cfg = Config()
        cfg.attention_backend = "flex_attention"
        with pytest.raises(ValueError) as exc:
            cfg.validate()

        message = str(exc.value)
        assert "flex" not in message.split("'flex_attention'")[-1], (
            "the error message still offers flex_attention as a choice"
        )
        for choice in ("auto", "flash_attention_3", "sdpa"):
            assert choice in message, f"the message does not name '{choice}'"
        assert "flash_attention_4" not in message

    def test_flex_attention_is_not_a_resolvable_backend(self, base_config):
        """AC-09: not in the enum, and not reachable through the dispatcher."""
        from models.transformer import VALID_BACKENDS

        assert "flex_attention" not in VALID_BACKENDS
        assert not hasattr(AttentionBackend, "_flex_attention"), (
            "the FlexAttention kernel is still present on the class"
        )

        base_config.attention_backend = "flex_attention"
        with pytest.raises(ValueError, match="Unknown attention_backend"):
            AttentionBackend(base_config)

    def test_cpu_prefers_sdpa(self, base_config):
        base_config.device = "cpu"
        backend = AttentionBackend(base_config)
        assert backend.backend_name == "sdpa"
        assert backend.fallback_reason is None

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="requires an MPS device"
    )
    def test_mps_auto_resolves_to_sdpa_with_the_gate_armed(self, base_config):
        """AC-02: the armed gate passes on MPS, and nothing was substituted.

        This is the gate that used to fail: `auto` preferred FlexAttention, whose probe
        raised `NotImplementedError: FlexAttention does not support backward on MPS`, and
        `require_attention_backend=True` correctly refused to fall back.
        """
        base_config.device = "mps"
        base_config.attention_backend = "auto"
        base_config.require_attention_backend = True

        backend = AttentionBackend(base_config)
        assert backend.backend_name == "sdpa"
        assert backend.fallback_reason is None, (
            "a fallback reason was recorded, so SDPA was reached by substitution rather "
            "than by being the preferred kernel"
        )

        result = backend.self_test()
        assert result["status"] == "passed"
        assert result["backend"] == "sdpa"
        assert result["fallback_reason"] is None
        assert result["device"].startswith("mps")


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


def _reference_varlen(calls=None):
    """A pure-torch stand-in for FA3's `flash_attn_varlen_func`: per-sequence SDPA."""
    import torch.nn.functional as F

    def flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False,
        **_ignored,
    ):
        if calls is not None:
            calls.append({"causal": causal, "cu_seqlens": cu_seqlens_q.clone(),
                          "max_seqlen": max_seqlen_q})
        outs = []
        for i in range(len(cu_seqlens_q) - 1):
            start, end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            qi, ki, vi = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            out = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
            outs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outs, dim=0)

    return flash_attn_varlen_func


def install_reference_fa3(monkeypatch, returns_lse=False, calls=None):
    """Install a reference `flash_attn_interface` into `sys.modules`.

    `returns_lse=True` mimics FA3 builds whose functions return `(out, softmax_lse)`.
    """
    import sys
    import types

    import torch.nn.functional as F

    varlen = _reference_varlen(calls)

    def flash_attn_func(q, k, v, causal=False, **_ignored):
        if calls is not None:
            calls.append({"causal": causal, "dense": True})
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal,
        ).transpose(1, 2)
        return (out, torch.zeros(1)) if returns_lse else out

    def flash_attn_varlen_func(*args, **kwargs):
        out = varlen(*args, **kwargs)
        return (out, torch.zeros(1)) if returns_lse else out

    module = types.ModuleType("flash_attn_interface")
    module.flash_attn_func = flash_attn_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn_interface", module)
    return module


def _left_padded_qkv(lengths, S=8, H=2, D=4, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    B = len(lengths)
    q, k, v = (torch.randn(B, H, S, D, dtype=dtype, requires_grad=True) for _ in range(3))
    padding_mask = torch.zeros(B, S, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, S - length:] = True
    return q, k, v, padding_mask


def _masked_sdpa(q, k, v, padding_mask):
    """Causal attention with a key-padding mask -- the SDPA semantics FA3 must match."""
    S = q.shape[2]
    blocked = ~padding_mask[:, None, None, :] | torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1,
    )
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)


def _fa3_backend(dtype=torch.bfloat16):
    """An `AttentionBackend` on CPU pinned to `flash_attention_3` (built via SDPA)."""
    cfg = Config()
    cfg.device = "cpu"
    cfg.attention_backend = "sdpa"
    backend = AttentionBackend(cfg)
    backend.backend_name = "flash_attention_3"
    backend.dtype = dtype
    return backend


class TestVarlenCore:
    """specs/ablation-run T4.1 / AC-05: our own unpad -> varlen -> repad, on CPU."""

    def test_roundtrip_keeps_real_positions_and_zeros_pads(self):
        from models.transformer import repad, unpad, varlen_meta

        S = 8
        _, _, _, mask = _left_padded_qkv([S, 5, 1], S=S)
        x = torch.randn(3, S, 2, 4)
        meta = varlen_meta(mask)
        back = repad(unpad(x, meta), meta, 3, S)
        assert torch.equal(back[mask], x[mask])
        assert torch.count_nonzero(back[~mask]) == 0

    def test_metadata_for_lengths_s_5_1(self):
        from models.transformer import varlen_meta

        S = 8
        _, _, _, mask = _left_padded_qkv([S, 5, 1], S=S)
        meta = varlen_meta(mask)
        assert meta.cu_seqlens.tolist() == [0, S, S + 5, S + 6]
        assert meta.cu_seqlens.dtype == torch.int32
        assert meta.max_seqlen == S
        assert meta.all_real is False
        assert meta.indices.numel() == S + 6

    def test_unpadded_mask_is_all_real(self):
        from models.transformer import varlen_meta

        meta = varlen_meta(torch.ones(3, 8, dtype=torch.bool))
        assert meta.all_real is True
        assert meta.max_seqlen == 8

    def test_output_and_gradients_match_masked_sdpa(self):
        from models.transformer import varlen_attention

        lengths = [8, 5, 1]
        q, k, v, mask = _left_padded_qkv(lengths)
        calls = []
        out = varlen_attention(q, k, v, mask, True, _reference_varlen(calls))
        ref_q, ref_k, ref_v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        ref = _masked_sdpa(ref_q, ref_k, ref_v, mask)

        real = mask[:, None, :, None].expand_as(out)
        assert torch.allclose(out[real], ref[real], atol=1e-5)
        assert torch.count_nonzero(out[~real]) == 0

        # Gradients through real outputs only: pads are never read by a loss.
        weight = torch.randn_like(out) * real
        (out * weight).sum().backward()
        (ref * weight).sum().backward()
        for got, exp in ((q.grad, ref_q.grad), (k.grad, ref_k.grad), (v.grad, ref_v.grad)):
            assert torch.allclose(got[real], exp[real], atol=1e-5)

        assert calls and all(c["causal"] is True for c in calls)

    def test_metadata_is_computed_once_per_model_forward(self, monkeypatch):
        import models.transformer as tm

        install_reference_fa3(monkeypatch)
        cfg = Config()
        cfg.device = "cpu"
        cfg.apply_model_config("t-small")
        assert cfg.num_layers == 3
        cfg.num_items, cfg.feature_dim, cfg.use_metadata = 50, 0, False
        cfg.max_seq_length = 8
        cfg.attention_backend = "sdpa"
        model = TransformerModel(cfg)
        model.backend.backend_name = "flash_attention_3"
        model.eval()

        counter = {"n": 0}
        real_meta = tm.varlen_meta

        def counting(mask):
            counter["n"] += 1
            return real_meta(mask)

        monkeypatch.setattr(tm, "varlen_meta", counting)
        seq = torch.tensor([[0, 0, 0, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            model(seq)
        assert counter["n"] == 1, f"varlen_meta ran {counter['n']} times for 3 layers"
        with torch.no_grad():
            model(seq)
        assert counter["n"] == 2, "a new forward (new mask object) must recompute"


class TestFlashAttention3Backend:
    """specs/ablation-run T4.2 / AC-03, AC-04, AC-06 on a host without FA3."""

    @pytest.mark.parametrize("device_type,capability,expected", [
        ("cuda", (9, 0), "flash_attention_3"),
        ("cuda", (8, 0), "sdpa"),
        ("cuda", (8, 9), "sdpa"),
        ("cuda", (7, 5), "sdpa"),
        ("mps", None, "sdpa"),
        ("cpu", None, "sdpa"),
    ])
    def test_preferred_backend(self, device_type, capability, expected):
        from models.transformer import preferred_backend

        assert preferred_backend(device_type, capability) == expected

    def test_valid_backends_are_exactly_three(self):
        from models.transformer import VALID_BACKENDS

        assert set(VALID_BACKENDS) == {"auto", "flash_attention_3", "sdpa"}
        assert not any(name.startswith("_flash_attention_4") for name in dir(AttentionBackend))

    @pytest.mark.parametrize("bad", ["flash_attention_4", "flex_attention"])
    def test_config_rejects_removed_backends_naming_the_three(self, bad):
        cfg = Config()
        cfg.attention_backend = bad
        with pytest.raises(ValueError) as exc:
            cfg.validate()
        message = str(exc.value).split("Choose from")[-1]
        for choice in ("auto", "flash_attention_3", "sdpa"):
            assert choice in message
        assert "flash_attention_4" not in message and "flex" not in message

    def _mock_hopper(self, monkeypatch):
        import sys

        from models.transformer import preferred_backend

        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)  # FA3 unavailable
        monkeypatch.setattr(
            AttentionBackend, "_preferred_backend",
            lambda self: preferred_backend("cuda", (9, 0)),
        )

    def test_auto_on_hopper_without_fa3_falls_back_with_a_reason(self, monkeypatch):
        self._mock_hopper(monkeypatch)
        cfg = Config()
        cfg.device = "cpu"
        cfg.attention_backend = "auto"
        backend = AttentionBackend(cfg)
        assert backend.backend_name == "sdpa"
        assert "flash_attention_3" in backend.fallback_reason

    def test_auto_on_hopper_without_fa3_raises_when_required(self, monkeypatch):
        from models.transformer import AttentionBackendError

        self._mock_hopper(monkeypatch)
        cfg = Config()
        cfg.device = "cpu"
        cfg.attention_backend = "auto"
        cfg.require_attention_backend = True
        with pytest.raises(AttentionBackendError, match="flash_attention_3"):
            AttentionBackend(cfg)

    def test_explicit_fa3_without_the_package_raises(self, monkeypatch):
        import sys

        from models.transformer import AttentionBackendError

        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
        cfg = Config()
        cfg.device = "cpu"
        cfg.attention_backend = "flash_attention_3"
        with pytest.raises(AttentionBackendError, match="flash_attn_interface"):
            AttentionBackend(cfg)

    def test_import_and_sdpa_build_need_no_fa3(self):
        """AC-04, in a fresh interpreter so the module import itself is exercised."""
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['flash_attn_interface'] = None\n"
            "import models.transformer as tm\n"
            "from configs.config import Config\n"
            "c = Config(); c.device = 'cpu'; c.attention_backend = 'sdpa'\n"
            "c.num_items = 20; c.use_metadata = False; c.apply_model_config('t-small')\n"
            "m = tm.TransformerModel(c)\n"
            "assert m.backend.backend_name == 'sdpa'\n"
            "assert 'flash_attn_interface' not in [k for k, v in sys.modules.items() if v]\n"
            "print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().endswith("ok")

    @pytest.mark.parametrize("returns_lse", [False, True])
    def test_padded_batch_matches_masked_sdpa(self, monkeypatch, returns_lse):
        install_reference_fa3(monkeypatch, returns_lse=returns_lse)
        backend = _fa3_backend()
        q, k, v, mask = _left_padded_qkv([8, 5, 1], dtype=torch.bfloat16)
        out = backend.forward(q, k, v, padding_mask=mask, is_causal=True)
        ref = _masked_sdpa(q.float(), k.float(), v.float(), mask)
        real = mask[:, None, :, None].expand_as(out)
        assert torch.allclose(out.float()[real], ref[real], atol=2e-2)
        assert torch.count_nonzero(out[~real]) == 0

    @pytest.mark.parametrize("returns_lse", [False, True])
    def test_unpadded_batch_takes_the_dense_call(self, monkeypatch, returns_lse):
        calls = []
        install_reference_fa3(monkeypatch, returns_lse=returns_lse, calls=calls)
        backend = _fa3_backend()
        q, k, v, mask = _left_padded_qkv([8, 8], dtype=torch.bfloat16)
        assert mask.all()
        out = backend.forward(q, k, v, padding_mask=mask, is_causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.allclose(out.float(), ref.float(), atol=2e-2)
        assert [c.get("dense") for c in calls] == [True]

    def test_length_one_and_full_rows_take_the_varlen_call(self, monkeypatch):
        calls = []
        install_reference_fa3(monkeypatch, calls=calls)
        backend = _fa3_backend()
        q, k, v, mask = _left_padded_qkv([1, 8], dtype=torch.bfloat16)
        backend.forward(q, k, v, padding_mask=mask, is_causal=True)
        assert len(calls) == 1 and "dense" not in calls[0]
        assert calls[0]["cu_seqlens"].tolist() == [0, 1, 9]

    def test_fp32_input_raises_naming_the_dtype(self, monkeypatch):
        from models.transformer import AttentionBackendError

        install_reference_fa3(monkeypatch)
        backend = _fa3_backend(dtype=torch.float32)
        q, k, v, mask = _left_padded_qkv([8, 5], dtype=torch.float32)
        with pytest.raises(AttentionBackendError, match="float32"):
            backend.forward(q, k, v, padding_mask=mask, is_causal=True)

    def test_probe_uses_the_model_head_dimension(self, monkeypatch):
        cfg = Config()
        cfg.device = "cpu"
        cfg.apply_model_config("t-base")
        backend = AttentionBackend(cfg)
        seen = []
        original = backend._dispatch

        def spy(name, q, k, v, padding_mask, is_causal):
            seen.append(q.shape[-1])
            return original(name, q, k, v, padding_mask, is_causal)

        backend._dispatch = spy
        ok, reason = backend._probe("sdpa")
        assert ok, reason
        assert seen and set(seen) == {80}


def _hopper_with_fa3() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _hopper_with_fa3(),
    reason="real FlashAttention-3 kernel test: needs CUDA capability 9.x and flash_attn_interface",
)
class TestRealFlashAttention3Kernel:
    """AC-07 (target machine): the real kernel matches SDPA on a left-padded batch."""

    def test_left_padded_parity_against_sdpa(self):
        cfg = Config()
        cfg.device = "cuda"
        cfg.attention_backend = "flash_attention_3"
        backend = AttentionBackend(cfg)
        assert backend.backend_name == "flash_attention_3"
        torch.manual_seed(0)
        B, H, S, D = 3, 4, 50, 80
        q, k, v = (
            torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        mask = torch.zeros(B, S, dtype=torch.bool, device="cuda")
        for i, length in enumerate([S, 25, 1]):
            mask[i, S - length:] = True
        out = backend.forward(q, k, v, padding_mask=mask, is_causal=True)
        blocked = ~mask[:, None, None, :] | torch.triu(
            torch.ones(S, S, dtype=torch.bool, device="cuda"), diagonal=1,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=~blocked,
        )
        real = mask[:, None, :, None].expand_as(out)
        assert torch.allclose(out.float()[real], ref[real], atol=2e-2, rtol=2e-2)


class TestCausalIntentReachesEveryBackend:
    """REV-122: causality must survive the trip from the layer to whichever kernel runs.

    `CausalSelfAttention.forward` used to hardcode `is_causal=False` on the padded branch
    and encode causality in an additive mask instead. SDPA reads that mask; the
    FlashAttention branch discards it and takes causality only from the flag, so on CUDA
    every left-padded batch -- i.e. every training batch -- attended BIDIRECTIONALLY.

    These tests go through `CausalSelfAttention.forward`, which is the whole point.
    The FlashAttention-3 kernel is replaced by a pure-torch reference
    (`install_reference_fa3`); FA3 takes bf16 only, so that branch runs in bf16.
    """

    @staticmethod
    def _backend(backend_name, dtype):
        """An `AttentionBackend` pinned to `backend_name`, on cpu.

        Construction probes `sdpa` (which runs here), then the name is overridden so
        `_dispatch` routes to the branch under test.
        """
        cfg = Config()
        cfg.device = "cpu"
        cfg.attention_backend = "sdpa"
        backend = AttentionBackend(cfg)
        backend.backend_name = backend_name
        backend.dtype = dtype
        return backend

    @staticmethod
    def _attention(backend, dtype, D=8, H=2):
        from models.transformer import CausalSelfAttention

        torch.manual_seed(0)
        attn = CausalSelfAttention(D, H, 0.0, backend).to(dtype=dtype)
        attn.eval()
        return attn

    @staticmethod
    def _left_padded_input(dtype, B=1, S=8, D=8, n_pad=3):
        torch.manual_seed(1)
        x = torch.randn(B, S, D, dtype=torch.float32).to(dtype)
        x[:, :n_pad, :] = 0.0
        padding_mask = torch.ones(B, S, dtype=torch.bool)
        padding_mask[:, :n_pad] = False
        return x, padding_mask, n_pad

    @pytest.mark.parametrize("backend_name,dtype", [
        ("sdpa", torch.float32), ("flash_attention_3", torch.bfloat16),
    ])
    def test_future_token_cannot_reach_earlier_positions(self, backend_name, dtype, monkeypatch):
        """Perturbing position S-1 must leave every earlier position bit-identical."""
        if backend_name == "flash_attention_3":
            install_reference_fa3(monkeypatch)

        backend = self._backend(backend_name, dtype)
        attn = self._attention(backend, dtype)
        x, padding_mask, n_pad = self._left_padded_input(dtype)
        S = x.shape[1]

        with torch.no_grad():
            base = attn(x, padding_mask)
            perturbed_in = x.clone()
            perturbed_in[:, S - 1, :] += 1.0
            perturbed = attn(perturbed_in, padding_mask)

        delta = (perturbed.float() - base.float()).abs().amax(dim=-1)[0]
        for pos in range(n_pad, S - 1):
            assert delta[pos].item() == 0.0, (
                f"[{backend_name}] position {pos} moved by {delta[pos].item():.6f} when "
                f"only position {S - 1} changed -- attention is not causal"
            )
        # Non-vacuity: the perturbed position itself must actually move.
        assert delta[S - 1].item() > 0.0

    def test_flash_attention_kernel_receives_causal_true_on_padded_batches(self, monkeypatch):
        """The flag itself, not only its effect: FA3 takes causality ONLY from here."""
        calls = []
        install_reference_fa3(monkeypatch, calls=calls)

        backend = self._backend("flash_attention_3", torch.bfloat16)
        attn = self._attention(backend, torch.bfloat16)
        x, padding_mask, _ = self._left_padded_input(torch.bfloat16)

        with torch.no_grad():
            attn(x, padding_mask)          # padded -> varlen
            attn(x, None)                  # unpadded -> dense

        assert len(calls) == 2, "the stubbed FlashAttention-3 kernel was not reached twice"
        assert [c.get("dense", False) for c in calls] == [False, True]
        assert all(c["causal"] is True for c in calls), (
            f"FlashAttention-3 received {calls}; every call from CausalSelfAttention "
            f"must be causal. `causal=False` here is silent label leakage on CUDA."
        )

    def test_probe_covers_the_padded_branch(self, base_config):
        """REV-122: the startup gate must exercise the branch training actually takes."""
        backend = AttentionBackend(base_config)
        seen = []
        original = backend._dispatch

        def spy(backend_name, q, k, v, padding_mask, is_causal):
            seen.append(padding_mask)
            return original(backend_name, q, k, v, padding_mask, is_causal)

        backend._dispatch = spy
        ok, reason = backend._probe(backend.backend_name)

        assert ok, reason
        assert any(m is None for m in seen), "probe never exercised the unpadded branch"
        padded = [m for m in seen if m is not None]
        assert padded, "probe never exercised the PADDED branch"
        assert not padded[0].all(), "probe's 'padded' mask has no padded positions"


class TestPositionGuard:
    """specs/ablation-run T2.4 / AC-18: a model never sees more positions than it has."""

    def test_wider_input_raises_naming_both_widths(self, base_config):
        model = TransformerModel(base_config)
        S = base_config.max_seq_length
        seq = torch.ones(2, S + 1, dtype=torch.long, device=get_device(base_config))
        with pytest.raises(ValueError) as exc:
            model(seq)
        assert str(S + 1) in str(exc.value)
        assert f"max_seq_length={S}" in str(exc.value)

    def test_model_width_input_still_runs(self, base_config):
        model = TransformerModel(base_config)
        S = base_config.max_seq_length
        seq = torch.ones(2, S, dtype=torch.long, device=get_device(base_config))
        assert model(seq).shape == (2, base_config.hidden_dim)
