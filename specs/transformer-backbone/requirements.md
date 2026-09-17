# Transformer Backbone — Requirements

## Overview

Implement the Transformer sequential recommendation backbone with three model size variants, optimized attention backends per hardware, metadata integration, embedding normalization, and correct causal/padding masking — all in BF16.

## Problem Statement

The backbone must produce both pooled representations (for next-item prediction and routing) and full sequence hidden states (for MIM and multi-step objectives). It must support hardware-specific attention backends (SDPA on MPS/CPU, FlashAttention-4 on CUDA) without silent fallback, and integrate optional item metadata without changing the evaluation protocol.

## Goals

- G1: Pre-LN Transformer with three size variants (T-Small, T-Base, T-Large)
- G2: Hardware-aware attention backends (SDPA on MPS/CPU, FA4 on CUDA) with startup self-tests and no silent fallback
- G3: Correct causal mask + padding validity for left-padded sequences
- G4: Dual output mode: pooled final representation or full sequence hidden states
- G5: Optional metadata integration via linear projection
- G6: L2 embedding normalization for dot-product score stability
- G7: All computation in BF16

## Non-Goals

- Implementing a general-purpose Transformer library
- Supporting attention backends beyond SDPA and FlashAttention-4
- FlexAttention support of any kind (see FR-4 rationale)
- Sinusoidal positional embeddings

## Functional Requirements

### FR-1: Embedding Layer
- Item embedding: `nn.Embedding(num_items + 2, hidden_dim)` — row 0 for padding, row num_items+1 for mask token
- Positional embedding: learnable `nn.Embedding(max_seq_length, hidden_dim)`
- Tied output weights: item embedding weight is reused as the output projection — never create a separate output matrix

### FR-2: Transformer Layer (Pre-LN)
- Architecture: Pre-LN → Self-Attention → Residual + Dropout → Pre-LN → FFN (4× expansion) → Residual + Dropout
- FFN expansion factor: `config.ffn_expansion = 4`
- Dropout rate: `config.dropout`

### FR-3: Model Size Variants
| Variant | Layers | Hidden | Heads |
|---|---:|---:|---:|
| T-Small | 3 | 256 | 4 |
| T-Base | 4 | 320 | 4 |
| T-Large | 5 | 384 | 6 |
- Validate `hidden_dim % heads == 0`

### FR-4: Attention Backends
- Sanctioned backends are exactly two: `sdpa` and `flash_attention_4`. `flex_attention` must not be accepted as a configuration value.
- MPS/CPU → SDPA (`torch.nn.functional.scaled_dot_product_attention`) — this is the sanctioned local/dev training kernel, not a degraded fallback
- CUDA/H100 → FlashAttention-4 (compatible package)
- `attention_backend="auto"` resolves by detected hardware
- Dedicated attention wrapper that changes only the backend, not the mathematical operation
- Startup self-test before any training, exercising **forward and backward** — fail fast on failure
- An explicitly requested backend that cannot execute raises; `require_attention_backend=True` also forbids substituting for an unavailable `"auto"` preference
- Record resolved backend in run manifest

**Rationale (REV-062, owner decision):** FlexAttention has no backward implementation on
MPS in the pinned PyTorch build, so it cannot train on the project's declared dev host.
Rather than pin a speculative PyTorch build, SDPA is sanctioned as the MPS/CPU kernel.
SDPA computes the same attention function, supports causal and additive key masks, and
trains on MPS. The H100 path is unchanged.

### FR-5: Masking
- Causal mask: future positions cannot contribute
- Padding validity: padded keys (ID=0) cannot contribute
- Mask format must match the selected backend's requirements (not all backends accept `nn.MultiheadAttention` arguments)

### FR-6: Hidden State Output
- `forward(..., return_all_hidden=True)` → `(B, S, D)` full hidden sequence
- `forward(..., return_all_hidden=False)` → pooled final valid representation
- `return_all_hidden=True` required whenever `multi_step_k > 0` or `use_masked_lm=True` or `moe_variant != "none"`

### FR-7: Metadata Integration (Step 13)
- Optional, enabled by default (`config.use_metadata = True`)
- Shared linear projection: `nn.Linear(feature_dim, hidden_dim, bias=False)`
- Input: `x = item_embeddings + positional_embeddings + metadata_hidden`
- Zero-metadata cold items contribute zero metadata vector (no learned missing-item signal)
- When `use_metadata=False`: omit projection, use only item + positional embeddings
- Forward interface: `forward(seq, metadata=None, return_all_hidden=False)`

### FR-8: Embedding Normalization (Step 12)
- When `config.normalize_embeddings = True`: L2-normalize both user repr and item embedding rows before any dot-product scoring
- Applied during both training and evaluation
- Bounds dot-product scores to [-1, 1], stabilizes sampled-softmax training

### FR-9: BF16
- All model computation in `torch.bfloat16`
- No FP16 autocast, no GradScaler

## Edge Cases

- Sequence entirely padding (all zeros): valid hidden state at S-1 still computed but should not be used as a meaningful representation
- Metadata tensor dimension mismatch: validate `feature_dim` matches expected dimension
- `use_metadata=False` with metadata tensor supplied: must not affect computation

## Constraints

- No sinusoidal positional embeddings
- No `.cuda()` calls — use `get_device(config)`
- Vocab size = `num_items + 2`

## Dependencies

- `configs/config.py` for all model parameters
- `utils.py` for `get_device()`, `resolve_dtype()`
- SDPA: PyTorch native (`torch.nn.functional.scaled_dot_product_attention`) — MPS/CPU
- FlashAttention-4: external package (CUDA only)

## Acceptance Criteria

- **AC-01:** T-Small/T-Base/T-Large produce correct output shapes for both `return_all_hidden` modes
- **AC-02:** SDPA self-test (forward + backward) passes on MPS before MPS training, and `attention_backend="auto"` on MPS resolves to `sdpa` with no fallback reason recorded
- **AC-03:** FlashAttention-4 self-test passes on H100 before CUDA training
- **AC-04:** Causal mask prevents future attention; padding positions contribute nothing
- **AC-05:** Item embedding weight is the same object used as output projection (tied weights verified by identity)
- **AC-06:** With `use_metadata=True`, metadata projection adds to input; with `False`, output is identical regardless of metadata tensor
- **AC-07:** L2 normalization bounds dot-product scores to [-1, 1] when `normalize_embeddings=True`
- **AC-08:** All parameters and computation are BF16; no silent dtype changes
- **AC-09:** `attention_backend="flex_attention"` is rejected by config validation with a message naming only the supported choices, and no FlexAttention code path remains reachable
