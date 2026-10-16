# Architecture

## System Overview

ARV-MMoE is a communication-efficient multi-objective federated sequential recommendation system. It trains a Transformer-based sequential recommender with four objectives, routes per-objective representations through a shared Mixture-of-Experts layer with adaptive routing, balances objective gradients dynamically via GradNorm, and personalizes via LoRA adapters in a federated simulation.

## Technology Stack

- **Framework:** PyTorch 2.x+ (manual implementation — no external recommendation libraries)
- **Python:** 3.10+
- **Environment manager:** `uv venv` only (no conda, no `python -m venv`)
- **Dtype:** BF16 by default (`torch.bfloat16`). Optional mixed precision (`mixed_precision=True`): fp32 master weights + autocast in bf16 or fp16; fp16 requires it and adds a GradScaler
  - **Experiment runs use mixed bf16** (`--dtype bf16 --mixed_precision`) — owner decision 2026-10-15. Pure bf16 weights round away updates below ~0.4% of the weight and plateau (measured: next-item loss 4.56 flat vs 4.436 still falling under mixed precision, T4, t-small, 360 steps). The `Config` default stays pure bf16; the run commands carry the flags.
- **Dataset:** MovieLens-20M via Kaggle API; experiment runs may use a seeded user sample (`user_sample_fraction`, processed into its own `data/processed_users{pct}pct` directory)
- **Dev hardware:** Apple Silicon Mac (MPS, SDPA). Never runs CUDA code; FlashAttention is never imported there.
- **Target hardware** (owner decision 2026-10-15, supersedes "H100 → FlashAttention-4"):
  - Shared H100 PCIe (Hopper, sm_90): FlashAttention-3 (`flash_attn_interface`, built from the flash-attention repo's `hopper/` directory; installed per machine, never in `requirements.txt`).
  - Any non-Hopper CUDA GPU (e.g. the AWS VM's L4): SDPA.
  - Arms that are compared with each other (one ablation study) run on one machine, so they share hardware, precision and attention backend. The backend is passed explicitly in run commands and is part of the training fingerprint.

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

Stored vs. model sequence width:
  Splits are stored at max(sequence_length_grid ∪ {max_seq_length}) (recorded as
  `stored_seq_length` in the preprocessing manifest). Every reader narrows to the consuming
  config's max_seq_length (most recent positions kept); asking for more than is stored, or
  feeding a model more positions than it was built for, raises.

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

## Run Execution (long runs on shared GPUs)

Approved pattern introduced by `specs/ablation-run` (owner decisions 2026-10-15); the backbone store exists today, the rest lands with that spec.

- **Backbone store.** Ablation arms are trained into `checkpoints/ablations/by_fingerprint/<training fingerprint>/` and reused by every arm, study and process on that machine whose training-relevant config matches. Training a fingerprint holds an exclusive OS file lock on its store directory.
- **Resume.** `run_pretrain` writes `last.pt` (full state incl. RNGs and loader generators) atomically at every epoch end and resumes from it by default; it refuses to resume across a different training fingerprint or `source_revision`. Resume is epoch-granular.
- **OOM.** An out-of-memory error ends the process with exit status 75; `scripts/run_with_retry.py` waits and re-runs the same command, which resumes. Completed ablation arms are persisted and reused on re-run.
- **Code transport.** Target machines without git receive code as a tarball of the tracked files; their `source_revision` is identified by `source_hash`.
- **Parallel arms (owner decision 2026-10-16, `specs/ablation-run` FR-14–FR-18).** The arms of one study may run as separate processes on separate GPUs **of one machine** (`--study X --arms ...`, `CUDA_VISIBLE_DEVICES=k`), one process per GPU, batch size unchanged. A study is never split across machines. Arms sharing an untrained fingerprint are placed on the same GPU; a process that still meets a fingerprint being trained elsewhere waits on the store lock and reuses it (or resumes it if the trainer died).
- **Study outputs are assembled, not accumulated.** Each arm's result is its own atomically written file; the study JSON and plots are (re)built from those files under a per-study lock, only when every arm is present, consistent (same code `source_hash`, backend, device type, non-overridden config) and ok — automatically by the process that completes the study, or explicitly by `--stage ablation_collect`.
