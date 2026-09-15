# Transformer Backbone — Tasks

Ordered tasks. Depends on data-pipeline T1.1 (Config) and T1.2 (utils.py).

## Phase 1: Attention Backend

- [ ] **T1.1: Implement attention backend wrapper in `models/transformer.py`** — Create `AttentionBackend` class that resolves backend from config/device: FlexAttention for MPS, FlashAttention-4 for CUDA, with `auto` resolution. Implement `self_test()` method that runs a small forward pass and verifies output. Implement `forward(q, k, v, causal_mask, padding_mask)` that delegates to the selected backend with its expected argument format. Fail fast (raise, do not fallback) if selected backend cannot initialize. Record resolved backend in returned metadata. (FR-4, AC-02, AC-03)
  - BUILD note (REV-081): the MPS/FlexAttention and SDPA halves are implemented and
    tested. The CUDA half now routes padded batches through
    `flash_attn_varlen_func` instead of raising on every left-padded batch, and the
    unpad / cumulative-seqlen / repad plumbing is tested against dense causal
    attention with the kernel stubbed (`tests/test_transformer.py::
    TestFlashAttentionVarlen`). The real FlashAttention kernel has NOT been run --
    this host has no CUDA device and no flash-attn build -- so AC-03 cannot be signed
    off here. Left unchecked until it is verified on the H100.

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
