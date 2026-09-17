# Transformer Backbone — Tasks

Ordered tasks. Depends on data-pipeline T1.1 (Config) and T1.2 (utils.py).

## Phase 1: Attention Backend

- [ ] **T1.1: Finalise attention backend wrapper in `models/transformer.py`** — `AttentionBackend` resolves backend from config/device: SDPA for MPS/CPU, FlashAttention-4 for CUDA, with `auto` resolution. `self_test()` runs forward *and* backward through the selected kernel and verifies output shape/finiteness. `forward(q, k, v, causal_mask, padding_mask)` delegates to the selected backend in its expected argument format. Fail fast (raise, do not substitute) if the selected backend cannot execute. Record resolved backend in returned metadata. (FR-4, AC-02, AC-03)
  - BUILD note (REV-081): the SDPA half is implemented and tested. The CUDA half now
    routes padded batches through `flash_attn_varlen_func` instead of raising on every
    left-padded batch, and the unpad / cumulative-seqlen / repad plumbing is tested
    against dense causal attention with the kernel stubbed
    (`tests/test_transformer.py::TestFlashAttentionVarlen`). The real FlashAttention
    kernel has NOT been run -- this host has no CUDA device and no flash-attn build --
    so AC-03 cannot be signed off here. Leave unchecked until it is verified on the H100.
  - PLAN note (REV-062, resolved): the MPS half of this task is no longer blocked. SDPA
    is the sanctioned MPS kernel; the FlexAttention branch is removed by T1.3.

- [x] **T1.3: Remove FlexAttention from the backend surface** — Delete `AttentionBackend._flex_attention` and its `_dispatch` branch in `models/transformer.py`; drop `"flex_attention"` from `VALID_BACKENDS` and from the `attention_backend` validation list and error message in `configs/config.py`; change `_preferred_backend()` so `mps` returns `"sdpa"`. Update the class docstring so it no longer describes FlexAttention as a resolvable backend. Do not weaken `require_attention_backend` or the fail-fast behaviour — only the sanctioned kernel changes. (FR-4, AC-02, AC-09, D2b)
  - BUILD note: done and verified. `_flex_attention` and its `_dispatch` branch are
    deleted, `VALID_BACKENDS` is `("auto", "flash_attention_4", "sdpa")`, and
    `_preferred_backend()` returns `"sdpa"` for every non-CUDA device. Measured on this
    host: `Config(attention_backend="flex_attention").validate()` raises
    `ValueError: Unknown attention_backend 'flex_attention'. Choose from 'auto',
    'flash_attention_4', 'sdpa'`; `attention_backend="auto"` with
    `require_attention_backend=True` on MPS resolves to `sdpa` with `fallback_reason is
    None`. `require_attention_backend` and the fail-fast paths are untouched.

- [x] **T1.4: Test the narrowed backend surface** — Assert `Config(attention_backend="flex_attention")` raises `ValueError` listing only `auto`/`flash_attention_4`/`sdpa`. Assert that on MPS (skip otherwise) `attention_backend="auto"` with `require_attention_backend=True` constructs successfully, resolves to `"sdpa"`, and reports no `fallback_reason`. Keep the existing `TestAttentionBackendGate` tests (they monkeypatch `_probe` and do not depend on FlexAttention) and update their comments to name SDPA. (AC-02, AC-09)

- [x] **T1.2: Test attention backends** — Unit test: self-test passes on available hardware. Test: causal mask prevents future attention (inject signal at future position, verify it doesn't appear in output). Test: padding mask zeros out padded key contributions. (AC-02, AC-03, AC-04)

## Phase 2: Embedding Layer

- [x] **T2.1: Implement embedding layer** — `nn.Embedding(num_items + 2, hidden_dim)` for items (row 0 = pad, row num_items+1 = mask). Learnable `nn.Embedding(max_seq_length, hidden_dim)` for positions. Ensure tied output weight: the item embedding weight tensor will be passed to loss functions as `item_emb_weight`. (FR-1, AC-05)

- [x] **T2.2: Implement metadata projection** — `nn.Linear(feature_dim, hidden_dim, bias=False)` when `use_metadata=True`. Compute: `x = item_emb + pos_emb + metadata_proj(metadata)`. When `use_metadata=False`: `x = item_emb + pos_emb` only, no projection in computation graph. Validate metadata feature_dim matches expected dimension. (FR-7, AC-06)

## Phase 3: Transformer Block

- [x] **T3.1: Implement Transformer block** — Pre-LN architecture: LayerNorm → Self-Attention (via backend) → Residual + Dropout → LayerNorm → FFN (Linear D→4D, GELU, Dropout, Linear 4D→D) → Residual + Dropout. All in BF16. (FR-2, FR-9)

- [x] **T3.2: Implement full Transformer model** — Stack N blocks per model size variant. Implement `forward(seq, metadata=None, return_all_hidden=False)`. When `return_all_hidden=True`: return `(B, S, D)` full sequence. When `False`: return pooled `(B, D)` from position S-1 (left-padding convention). Apply embedding normalization when `config.normalize_embeddings=True`. (FR-6, FR-8, AC-01, AC-07)

- [x] **T3.3: Implement model size factory** — Map model name to (layers, hidden_dim, heads): T-Small=(3,256,4), T-Base=(4,320,4), T-Large=(5,384,6). Validate `hidden_dim % heads == 0`. (FR-3)

## Phase 4: Testing

- [x] **T4.1: Shape tests** — Verify T-Small, T-Base, T-Large produce correct output shapes for both `return_all_hidden` modes. Batch size 4, seq length from config. (AC-01)

- [x] **T4.2: Tied weight identity test** — Verify `model.item_embedding.weight is` the same object used for scoring. (AC-05)

- [x] **T4.3: Metadata integration test** — Forward pass with `use_metadata=True` and `False` on same input with metadata tensor. Verify: with `True`, output differs from `False`; with `False`, supplying metadata tensor doesn't change output. (AC-06)

- [x] **T4.4: BF16 verification** — Verify all model parameters are bfloat16. Verify output tensors are bfloat16. (AC-08)

- [x] **T4.5: Embedding normalization test** — Verify dot-product scores bounded to [-1, 1] when `normalize_embeddings=True`. (AC-07)

- [x] **T4.6: One-batch smoke test** — Run one forward + backward pass with real input shapes. Verify loss is finite, gradients exist, no NaN/Inf. (Smoke test gate per steering/testing.md Phase 9)

## Phase 5: Documentation Sync (REV-062)

- [x] **T5.1: Sync `README.md` to the SDPA decision** — Update the architecture ASCII block (`FlexAttention (MPS) or FA4 (H100)` → `SDPA (MPS) or FA4 (H100)`), the Hardware Requirements table (Smoke testing / Development rows → `SDPA`), and the Fail-fast policy paragraph (`fails immediately if FlexAttention cannot execute` → `if SDPA cannot execute`). No other README content changes. (FR-4, D2b)

- [x] **T5.2: Update stale FlexAttention comments in the smoke gates** — `tests/test_smoke.py::TestOneBatchEndToEnd::test_full_pipeline` and `tests/test_training.py::TestFewStepSmokeViaRunPretrain` both cite the old steering line ("MPS smoke test must fail fast if FlexAttention cannot execute"). Reword to quote the current `steering/code-quality.md` line naming SDPA. `require_attention_backend = True` stays set in both — the arming is correct and is not what changed. (FR-4, AC-02)
