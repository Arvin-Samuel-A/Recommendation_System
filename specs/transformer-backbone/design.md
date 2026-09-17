# Transformer Backbone — Design

## Relevant Existing Architecture

Greenfield project. Design follows Prompt_v6.2 Step 8.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `models/transformer.py` | Create | Full Transformer backbone with attention wrapper, metadata integration, embedding normalization |
| `configs/config.py` | Extend | Model size variants, attention backend, dropout, ffn_expansion |

## Proposed Architecture

### Layer Stack (per Transformer block)

```
Input (B, S, D)
  → LayerNorm (Pre-LN)
  → CausalSelfAttention (hardware-aware backend)
  → Residual + Dropout
  → LayerNorm (Pre-LN)
  → FFN: Linear(D, 4D) → GELU → Dropout → Linear(4D, D)
  → Residual + Dropout
Output (B, S, D)
```

### Embedding Computation

```
seq (B, S) → item_embedding (V, D) → item_emb (B, S, D)
positions (S,) → pos_embedding (S, D) → pos_emb (B, S, D)
[metadata (B, S, F) → metadata_projection (F, D) → meta_emb (B, S, D)]

x = item_emb + pos_emb [+ meta_emb]
x = Dropout(x)
```

### Attention Wrapper Design

```python
class AttentionBackend:
    """
    Unified interface. __init__ resolves backend from config.
    forward(q, k, v, causal_mask, padding_mask) handles backend-specific format.
    self_test() validates the backend before training.
    """
```

The wrapper translates the common `(q, k, v, masks)` interface into whatever the backend requires:
- SDPA (MPS/CPU): `F.scaled_dot_product_attention` with `is_causal=True` when there is no padding, and an additive/boolean `attn_mask` combining causal + key padding when there is
- FlashAttention-4 (CUDA): uses the FA4 package API with its expected argument format (varlen path for left-padded batches)
- Neither changes the mathematical operation — only the implementation

FlexAttention is deliberately absent. It was the original MPS choice, but it has no
backward pass on MPS in the pinned PyTorch build, so it cannot train on the dev host;
see requirements FR-4 rationale and D2b.

### Output Modes

```python
def forward(self, seq, metadata=None, return_all_hidden=False):
    # ...transformer layers...
    if return_all_hidden:
        return all_hidden_states  # (B, S, D) — full sequence
    else:
        return pooled             # (B, D) — last valid position
```

For pooled output: under left-padding convention, the last valid position is always `S-1`.

### Tied Output Projection

```python
self.item_embedding = nn.Embedding(vocab_size, hidden_dim)
# No separate output projection — scoring is:
# logits = hidden @ self.item_embedding.weight.T
```

The embedding weight tensor is passed to loss functions as `item_emb_weight`.

## Key Design Decisions

### D1: Pre-LN vs Post-LN
Pre-LN chosen (LayerNorm before attention/FFN, not after). This is specified in the prompt and is known to improve training stability, especially important for BF16 training.

### D2: Attention Backend Abstraction
A thin wrapper rather than a deep abstraction. The wrapper's job is:
1. Select the correct backend based on device detection
2. Run a self-test at startup — forward *and* backward, since training needs both
3. Format masks for the selected backend
4. Fail fast (never silently substitute a different kernel than the one recorded)

### D2b: SDPA as the sanctioned MPS backend (REV-062, owner decision)
**Decision:** MPS/CPU resolve to `sdpa`; `flex_attention` is removed from the backend
enum and its kernel deleted. CUDA still resolves to `flash_attention_4`.

**Why:** `AttentionBackend._probe` runs a backward pass because training requires one,
and FlexAttention raises `NotImplementedError: FlexAttention does not support backward
on MPS` in the pinned PyTorch build. With the MPS gate armed, the two smoke tests that
authorise full training could not pass on the declared dev host.

**Alternatives considered:**
- *Pin a PyTorch build where FlexAttention trains on MPS* — rejected. No such build is
  known to exist today; it would gate the entire project on an upstream fix, and the dev
  host is used only for smoke/T-Small work where SDPA's performance is sufficient.
- *Leave `require_attention_backend=False` and let the fallback carry MPS* — rejected.
  That is the silent-fallback pattern `steering/code-quality.md` forbids, and it makes
  the gate certify a kernel the architecture does not name.
- *Keep `flex_attention` in the enum as an opt-in* — rejected. A selectable backend that
  provably cannot train is a trap; config validation should reject it outright (AC-09).

**Consequence:** the "no silent fallback" discipline is unchanged and stays armed — only
the sanctioned MPS kernel changes. `require_attention_backend=True` remains set in the
MPS smoke gates, and it now passes because the preferred backend genuinely executes.

### D3: Metadata as Additive Projection
Metadata is projected to hidden_dim and **added** to the input embeddings. This is simpler than concatenation (which would change D) and allows the model to learn to ignore metadata when it's zero (cold items with no metadata). The projection has no bias to preserve the zero-vector semantics.

### D4: Embedding Normalization Location
L2 normalization is applied to the final representations before scoring, not to the embedding table directly. This preserves the embedding table's gradient properties while bounding the dot-product output.

## Affected Components

- `models/moe.py` receives `all_hidden_states` and `pooled_hidden` from this module
- `training/losses.py` receives `item_embedding.weight` for tied projections
- `training/pretrain.py` calls `forward(return_all_hidden=True)` when MoE or MIM/multi-step is active
- `evaluation/flops.py` benchmarks the full forward pass including metadata

## Testing Strategy

- Shape tests: verify output dimensions for all three model sizes and both output modes
- Attention self-test: run before any training, forward + backward, fail fast
- Backend enum test: `attention_backend="flex_attention"` is rejected by `Config.validate()` (AC-09)
- Causal masking test: verify position i cannot attend to position j > i (use gradient flow)
- Padding masking test: verify padded positions contribute nothing (compare output with/without padding)
- Tied weight identity test: `model.output_projection_weight is model.item_embedding.weight`
- Metadata test: verify `use_metadata=False` produces identical output regardless of metadata input
- BF16 test: verify all parameters and outputs are bfloat16

### D-dropout: attention dropout is applied to the output, not the weights (REV-082)

`CausalSelfAttention` applies `attn_dropout` to the attention **output**, after the
head-merge and before `out_proj`, rather than to the attention weight matrix. SDPA's
`dropout_p` argument is not usable uniformly across the two sanctioned backends, so
applying it at the output keeps the regularisation backend-agnostic and keeps FR-4's
"changes only the backend, not the mathematical operation" property intact.

FR-2 specifies only the dropout *rate* (`config.dropout`) and does not pin the placement,
so this is conforming. Recorded because it deviates from textbook attention-weight dropout
and the paper's method section should say so.
