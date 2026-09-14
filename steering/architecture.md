# Architecture

## System Overview

ARV-MMoE is a communication-efficient multi-objective federated sequential recommendation system. It trains a Transformer-based sequential recommender with four objectives, routes per-objective representations through a shared Mixture-of-Experts layer with adaptive routing, balances objective gradients dynamically via GradNorm, and personalizes via LoRA adapters in a federated simulation.

## Technology Stack

- **Framework:** PyTorch 2.x+ (manual implementation — no external recommendation libraries)
- **Python:** 3.10+
- **Environment manager:** `uv venv` only (no conda, no `python -m venv`)
- **Dtype:** BF16 everywhere (`torch.bfloat16`) — no FP16 autocast, no GradScaler
- **Dataset:** MovieLens-20M via Kaggle API
- **Dev hardware:** Apple Silicon Mac (MPS, FlexAttention)
- **Prod hardware:** NVIDIA H100 (CUDA, FlashAttention-4)

## Component Map

```
data/              → Preprocessing, sequence generation, augmentation, tensorisation
models/            → Transformer backbone, ARV-MMoE/MMoE, LoRA, ANN retrieval
training/          → Pretraining loop, local personalization, federated simulation,
                     loss functions, GradNorm, ablation runner
evaluation/        → Metrics, cold-start, long-tail, latency, routing/gradient diagnostics
configs/config.py  → Single source of truth for all hyperparameters
utils.py           → Device helper, seed utilities
main.py            → CLI entrypoint dispatching all stages
```

## Data Flow

```
Raw CSVs (Kaggle)
  → Preprocessing (filtering, ID remapping, metadata extraction, cold-start holdout)
  → Sequence generation (chronological split, sliding window, left-padding, tensorisation)
  → Global negative pool (warm items only, deterministic, shared)
  → Fixed evaluation candidate sets (generated once, shared across all models)

Training:
  User sequence → Item embedding + Positional embedding [+ Metadata projection]
    → Transformer backbone (Pre-LN, causal self-attention)
    → [MoE layer: shared experts computed once, per-objective routing]
    → Per-objective heads (next-item, MIM, multi-step, contrastive)
    → GradNorm-balanced total loss → Optimizer step

Serving:
  User sequence → Transformer → ARV routing with e_rec ≡ e_next
    → Expert mixture Z_rec → [ANN candidate retrieval] → Scoring → Top-N
```

## Component Boundaries

| Boundary | Rule |
|---|---|
| Config ↔ Code | Config is the single source of truth; no hardcoded hyperparameters elsewhere |
| Training ↔ Evaluation | Test data untouched until model configuration and best checkpoint are frozen |
| Global ↔ Local (federated) | Only LoRA adapter state crosses the wire; backbone, MoE, router, experts, objective embeddings are frozen global state |
| MoE variants | `moe_variant="none"` is a fully working direct-hidden-state path; `"mmoe"` uses independent per-objective gates; `"arv_mmoe"` uses the shared objective-conditioned router |
| Preprocessing ↔ Model | ID scheme (pad=0, items=1..N, mask=N+1) is enforced at preprocessing and respected everywhere |

## Key Architectural Decisions

1. **Shared experts, differentiated routing.** The K expert FFN outputs are computed once per forward pass and reused by all four objectives — only the mixture weights r_{u,t} differ per objective.
2. **Pooled-then-broadcast routing.** Routing distribution is computed once from pooled H_u, then broadcast across all sequence positions for position-wise objectives (MIM, multi-step).
3. **e_rec ≡ e_next at inference.** No separate inference-only embedding; the next-item objective embedding serves double duty.
4. **LoRA-only federated communication.** Router, experts, objective embeddings, and GradNorm state remain global/frozen during local training.
5. **GradNorm on shared Transformer anchor.** The anchor is the last shared Transformer block (before optional MoE), ensuring it exists in every model variant.
