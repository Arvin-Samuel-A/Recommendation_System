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
- FlexAttention (MPS): uses PyTorch's `torch.nn.attention.flex_attention` API with score_mod or block_mask for causal + padding
- FlashAttention-4 (CUDA): uses the FA4 package API with its expected argument format
- Neither changes the mathematical operation — only the implementation

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
2. Run a self-test at startup
3. Format masks for the selected backend
4. Fail fast (never silently fall back to vanilla attention)

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
- Attention self-test: run before any training, fail fast
- Causal masking test: verify position i cannot attend to position j > i (use gradient flow)
- Padding masking test: verify padded positions contribute nothing (compare output with/without padding)
- Tied weight identity test: `model.output_projection_weight is model.item_embedding.weight`
- Metadata test: verify `use_metadata=False` produces identical output regardless of metadata input
- BF16 test: verify all parameters and outputs are bfloat16
