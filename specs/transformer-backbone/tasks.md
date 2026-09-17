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

- [x] **T1.5: Route causal intent to every backend, not through an SDPA-dialect mask (REV-122)** — `CausalSelfAttention.forward` built one additive `attn_float` mask carrying causal+padding and passed `is_causal=False` alongside it. `_dispatch` forwards `attn_mask` to SDPA (which reads it) but hands FlashAttention-4 only `(padding_mask, is_causal)` — so the FA4 branch discarded causality and ran **bidirectional** on every left-padded batch, i.e. every training batch on CUDA. Apply correction (b): `_dispatch` takes the unambiguous `(padding_mask, is_causal)` pair and each backend formats its own mask (design D2 point 3); `CausalSelfAttention.forward` passes `is_causal=True` on both branches and builds no mask. Add a regression test that goes through `CausalSelfAttention.forward` (NOT the private `_flash_attention_4*` helpers) with the stubbed kernel and asserts positions `< S-1` are bit-identical when position `S-1` changes. Extend `_probe` to a padded input so the H100 gate covers the branch training takes. (FR-4, FR-5, AC-03, AC-04)
  - BUILD note: done and verified. `AttentionBackend.forward`/`_dispatch` no longer accept
    an `attn_mask` parameter at all — an additive mask is SDPA dialect, and accepting one
    is what let causality vanish. New `_sdpa_additive_mask(q, padding_mask, is_causal)`
    folds the causal half into the mask for SDPA only (SDPA forbids `attn_mask` with
    `is_causal=True`) and returns `None` when unpadded, so the native `is_causal` fast
    path is kept. `_probe` now runs forward+backward over BOTH an unpadded and a
    left-padded input and labels which case failed.
    Tests added (`tests/test_transformer.py::TestCausalIntentReachesEveryBackend`):
      * `test_future_token_cannot_reach_earlier_positions[sdpa|flash_attention_4]` — drives
        `CausalSelfAttention.forward`, perturbs only position S-1, asserts every earlier
        real position moves by **exactly 0.0**, plus a non-vacuity assertion that position
        S-1 itself does move.
      * `test_flash_attention_kernel_receives_causal_true_on_padded_batches` — spies on the
        stubbed kernel and asserts the `causal=` flag it actually received is True on both
        the varlen (padded) and dense (unpadded) calls.
      * `test_probe_covers_the_padded_branch` — asserts `_probe` dispatches with a padding
        mask that has real padded positions, so the AC-03 self-test cannot pass while
        training runs non-causally.
    Non-vacuity verified, not assumed: the pre-REV-122 contract was reinstated in a scratch
    copy and the suite re-run — `test_future_token_cannot_reach_earlier_positions[flash_
    attention_4]` and `test_flash_attention_kernel_receives_causal_true_on_padded_batches`
    both FAIL on it (the spy records `causal=[False, True]`, False on the padded call), and
    both pass on the fix. The four pre-existing `TestFlashAttentionVarlen` tests pass
    unchanged on both, which is exactly why they could not catch this.
    `tests/test_transformer.py` 30 passed.
    **T1.1 stays unchecked.** This fix does not give it AC-03: the real FlashAttention
    kernel still has not run (no CUDA device, no flash-attn build on this host), so the
    CUDA half remains unverified on real hardware.

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
