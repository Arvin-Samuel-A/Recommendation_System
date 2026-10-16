# ARV-MMoE: Communication-Efficient Multi-Objective Federated Sequential Recommendation with Adaptive Expert Routing

> VIT Chennai — Project Report, August 2026

A federated sequential recommendation system that combines a Transformer backbone with Adaptive Routing Vector Mixture-of-Experts (ARV-MMoE), GradNorm dynamic loss balancing, and LoRA-based personalization. Trained and evaluated on MovieLens-20M.

---

## Table of Contents

- [Research Questions](#research-questions)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Individual Stages](#individual-stages)
  - [Ablation Studies](#ablation-studies)
  - [CLI Reference](#cli-reference)
- [Model Variants](#model-variants)
- [Training Objectives](#training-objectives)
- [MoE Routing Variants](#moe-routing-variants)
- [Federated Personalization](#federated-personalization)
- [Evaluation Protocol](#evaluation-protocol)
- [Ablation Studies A–H](#ablation-studies-ah)
- [Outputs & Deliverables](#outputs--deliverables)
- [Reproducibility](#reproducibility)
- [Hardware Requirements](#hardware-requirements)
- [Build Order](#build-order)
- [Specifications & Steering](#specifications--steering)
- [Key Design Decisions](#key-design-decisions)
- [Strict Rules](#strict-rules)

---

## Research Questions

| RQ | Question | Addressed By |
|---|---|---|
| **RQ1** | Does multi-objective training improve sequential recommendation over single-objective? | Ablation A (configs 1 vs 2) |
| **RQ2** | Does ARV-MMoE's objective-conditioned routing outperform conventional MMoE and no-routing baselines? | Ablations A, B, C |
| **RQ3** | Do ARV-MMoE and GradNorm reduce gradient interference, and are their effects complementary? | Ablation D + gradient diagnostics |
| **RQ4** | Can federated LoRA personalization achieve competitive accuracy with communication-efficient payloads? | Ablation A (config 6), Ablation E, communication accounting |
| **RQ5** | What is the inference latency overhead of the routing layer? | Latency benchmarking, efficiency report |

---

## Architecture

```
User interaction sequence
  │
  ▼
┌─────────────────────────────────────┐
│  Item Embedding + Positional Emb    │
│  [+ Metadata Projection]            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Transformer Backbone (Pre-LN)      │
│  Causal Self-Attention (BF16)       │
│  SDPA (MPS/CPU/L4) or FA3 (Hopper)  │
│  3–5 layers depending on variant    │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   all_hidden    pooled_H_u
        │             │
        ▼             ▼
┌─────────────────────────────────────┐
│  ARV-MMoE Layer                     │
│  K shared experts (computed once)   │
│  Router: softmax(W_r[H_u; e_t])    │
│  Per-objective mixture Z_{u,t}      │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┬───────────┐
    ▼          ▼          ▼           ▼
 Next-Item   MIM     Multi-Step  Contrastive
  Loss       Loss      Loss       Loss
    │          │          │           │
    └──────────┴──────────┴───────────┘
               │
               ▼
        GradNorm Balancing
               │
               ▼
          L_total → Backprop
```

At inference, the system uses the next-item objective embedding (`e_rec ≡ e_next`) for routing, producing a single expert mixture for scoring.

---

## Project Structure

```
project_root/
│
├── data/
│   ├── raw/                        ← 6 MovieLens-20M CSV files (auto-downloaded)
│   ├── processed/                  ← Tensorised datasets, mappings, pools
│   ├── preprocessing.py            ← Kaggle acquisition, ratings pipeline, cold-start, metadata
│   ├── sequence_generator.py       ← Chronological splitting, tensorisation, negative pool
│   └── augmentation.py             ← Random crop, mask, subsequence sampling
│
├── models/
│   ├── transformer.py              ← Backbone with attention backends, metadata integration
│   ├── moe.py                      ← ARV-MMoE, conventional MMoE, shared experts
│   ├── lora.py                     ← LoRA injection, state dict utilities
│   └── ann_retrieval.py            ← Optional ANN / brute-force retrieval
│
├── training/
│   ├── pretrain.py                 ← Central pretraining loop, LR scheduling, checkpointing
│   ├── local_train.py              ← 1000-user local personalization
│   ├── federated.py                ← Federated simulation, weighted FedAvg
│   ├── losses.py                   ← All 4 loss functions
│   ├── gradnorm.py                 ← GradNorm dynamic loss balancing
│   └── ablation_runner.py          ← Config-driven ablation studies A–H
│
├── evaluation/
│   ├── metrics.py                  ← NDCG@10, MRR@10, Recall@10, Precision@10
│   ├── flops.py                    ← Latency benchmarking, parameter counting
│   ├── cold_start.py               ← Cold-start evaluation (metadata ON/OFF)
│   ├── longtail.py                 ← Head/mid/tail popularity bucket evaluation
│   ├── routing_diagnostics.py      ← Entropy, utilization, collapse, heatmaps
│   └── gradient_diagnostics.py     ← Gradient cosine similarity, gradient norms
│
├── configs/
│   └── config.py                   ← Single Config dataclass (all hyperparameters)
│
├── outputs/
│   ├── checkpoints/{model}/        ← best.pt + epoch checkpoints per model
│   ├── logs/                       ← JSON/JSONL/log training and evaluation logs
│   ├── plots/                      ← All generated PNG plots (150 DPI)
│   └── ablations/                  ← Per-study JSON results (A–H)
│
├── steering/                       ← Project-wide architectural guidance
├── specs/                          ← Per-feature requirements, design, and task specs
├── utils.py                        ← Device resolution, dtype enforcement, seed setup
├── requirements.txt
├── main.py                         ← CLI entrypoint
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (for development/smoke testing); NVIDIA Hopper (H100) or other CUDA GPUs for experiments
- Kaggle account with API credentials configured (`~/.kaggle/kaggle.json`)

### Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Only `uv venv` is supported. Do not use conda, `python -m venv`, or virtualenv.

### Kaggle Credentials

Place your Kaggle API token at `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

```bash
chmod 600 ~/.kaggle/kaggle.json
```

The dataset is downloaded automatically on first run. Subsequent runs reuse the cached copy.

### FlashAttention-3 (Hopper GPUs only)

**Owner decision 2026-10-15:** Hopper GPUs (H100, compute capability 9.x) use FlashAttention-3 (`flash_attn_interface`, built from the `hopper/` directory of the flash-attention repository). Every other device — Apple Silicon (MPS), CPU and non-Hopper CUDA GPUs such as the AWS L4 — uses PyTorch SDPA. The former `flash_attention_4` backend has been removed: despite its name it called the FlashAttention-2 `flash_attn` API.

FA3 is optional and machine-local: it is never in `requirements.txt` and nothing on the Mac imports it. Building it on the H100 (torch 2.14.1+cu130, CUDA 13, Python 3.12, no git) is scripted by the `ablation-run` spec (`specs/ablation-run/`): the flash-attention source travels as a tarball made on the Mac (`scripts/fetch_fa3_source.sh`), is built on the H100 by `scripts/install_fa3.sh`, and is verified with `scripts/verify_attention_backend.py --backend flash_attention_3` (self-test plus output/gradient parity against SDPA; exit 0 pass, 1 parity failure, 2 unavailable). `install_fa3.sh` exits 0 when FA3 is built (or already installed) and verified, 1 on a parity failure, 2 when a preflight check fails or FA3 is still unavailable, and 3 when extraction or the `hopper/` build fails (log: `logs/fa3_build.log`). See [Ablation run (10% users)](#ablation-run-10-users) for the commands.

---

## Usage

### Full Pipeline

Run the complete pipeline (preprocessing through final plots) for a single model:

```bash
python main.py --stage all --model t-base
```

This executes: preprocess → pretrain → personalize → federated → evaluate → cold_start → longtail → latency → routing_diagnostics → gradient_diagnostics → plots.

The ablation suite is **not** included in `--stage all` — it must be run explicitly.

### Individual Stages

```bash
# Data preprocessing (downloads dataset on first run)
python main.py --stage preprocess

# Train a specific model
python main.py --stage pretrain --model t-base

# Train with overrides
python main.py --stage pretrain --model t-base --epochs 5 --batch_size 128

# Train with specific MoE variant
python main.py --stage pretrain --model t-base --moe_variant arv_mmoe --use_gradnorm

# Local personalization (1000-user sample)
python main.py --stage personalize --model t-base

# Federated simulation
python main.py --stage federated --model t-base

# Evaluation (skips gracefully if no checkpoint exists)
python main.py --stage evaluate --model t-base

# Cold-start, long-tail, and latency evaluations
python main.py --stage cold_start --model t-base
python main.py --stage longtail --model t-base
python main.py --stage latency --model t-base

# Routing and gradient diagnostics
python main.py --stage routing_diagnostics --model t-base
python main.py --stage gradient_diagnostics --model t-base

# Generate all plots from completed results
python main.py --stage plots
```

### Ablation Studies

```bash
# Run a specific ablation study
python main.py --stage ablation --study A --model t-base

# Run all ablation studies sequentially
python main.py --stage ablation --study all --model t-base
```

Each ablation is config-driven — no source code edits required.

### CLI Reference

| Flag | Values | Description |
|---|---|---|
| `--stage` | `preprocess`, `pretrain`, `personalize`, `federated`, `evaluate`, `cold_start`, `longtail`, `latency`, `routing_diagnostics`, `gradient_diagnostics`, `ablation`, `ablation_collect`, `plots`, `all` | Pipeline stage to run (`ablation_collect`: assemble study JSON + plots from persisted arms, train nothing; exit 1 if a study is incomplete) |
| `--model` | `t-small`, `t-base`, `t-large` | Model size variant |
| `--epochs` | integer | Override `pretrain_epochs` (pretrain) or `federated_rounds` (federated) |
| `--batch_size` | integer | Override `batch_size` for all stages |
| `--moe_variant` | `none`, `mmoe`, `arv_mmoe` | Override MoE routing variant |
| `--num_experts` | integer | Override number of experts (K) |
| `--use_gradnorm` | flag | Force GradNorm on |
| `--no_gradnorm` | flag | Force GradNorm off |
| `--study` | `A`–`H`, `all` | Ablation study to run (with `--stage ablation`) |
| `--no_train` | flag | Ablations only: evaluate existing backbones, skip arms without one |
| `--seed` | integer | Training seed |
| `--dtype` | `bf16`, `fp16` | Compute dtype (`fp16` requires `--mixed_precision`) |
| `--mixed_precision` | flag | fp32 master weights + autocast in `--dtype`; experiment runs use `--dtype bf16 --mixed_precision` |
| `--max_steps` | integer | Cap optimizer steps per epoch (0 = uncapped) |
| `--eval_users`, `--clients_per_round`, `--local_epochs` | integer | Personalization / federated overrides |
| `--user_sample_fraction` | float in (0, 1] | Keep a seeded random fraction of users; below 1 the processed dir becomes `data/processed_users{pct}pct` |
| `--user_sample_seed` | integer | Seed of the user sample (independent of `--seed`; non-default adds `_seed{S}` to the dir) |
| `--sequence_length_grid` | integers | Ablation H2 lengths; preprocessing stores every split at `max(grid + [max_seq_length])` |
| `--attention_backend` | `auto`, `flash_attention_3`, `sdpa` | Attention kernel; an explicit backend that cannot run fails fast |
| `--num_workers` | integer | DataLoader workers (same batches as 0 workers) |
| `--min_lr_ratio` | float in [0, 1] | Cosine LR floor as a fraction of the peak LR (default 0.1) |
| `--no_resume` | flag | Ignore `last.pt` and persisted ablation arms; start from scratch |
| `--arms` | comma-separated labels | `--stage ablation` only: run just these arms of ONE `--study` (e.g. `--study H --arms H2_seq200`); an unknown label fails before any work |
| `--allow_incomplete` | flag | `ablation_collect` only: assemble a study with missing/failed/stale arms as placeholders (never an inconsistent one) |

All CLI overrides are logged with original and new values. A stage that runs out of GPU
memory exits with status **75** so `scripts/run_with_retry.py` can re-run it.

### Ablation run (10% users)

The `specs/ablation-run` runbook: studies A–H at t-base, seed 42, on a seeded 10% user sample,
mixed bf16, batch size 256 — studies A, D, E, H on the shared H100 (FlashAttention-3) and
B, C, F, G on the AWS VM (SDPA). The exact per-machine copy/extract/launch commands are
generated, not written by hand: see `launch/LAUNCH.md` produced by
`scripts/package_ablation_run.py`.

```bash
# 1. Mac: build the sample (stored width 200 covers every H2 arm)
uv run python main.py --stage preprocess --user_sample_fraction 0.1      # -> data/processed_users10pct
# 2. Mac: FlashAttention-3 source for the H100 (needs network), then package everything
bash scripts/fetch_fa3_source.sh <tag> dist
# (commit first: the packager refuses while launch-time files or covered sources are untracked
#  or have uncommitted changes; --allow-dirty overrides)
uv run python scripts/package_ablation_run.py   # dist/code.tar.gz, dist/processed_users10pct.tar.gz, launch/
# 3. H100: build + verify FA3, detached (skips the build if it already imports)
setsid nohup bash scripts/install_fa3.sh flash-attention-src.tar.gz > logs/fa3_install.log 2>&1 < /dev/null &
#    exit status: 0 ok, 1 parity failure, 2 preflight/unavailable, 3 build failed
python3 scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base   # exit 0 = parity OK
# 4. each machine: launch detached (from the extracted directory)
mkdir -p logs && setsid nohup bash launch/run_h100.sh > logs/run_h100.log 2>&1 < /dev/null &
# 5. monitor
tail -f logs/run_h100.log logs/ablation_A_t_base_seed42.log
```

- **Resume / retry.** Each study runs through `scripts/run_with_retry.py`: an out-of-memory
  exit (75) waits and re-runs the same command. `run_pretrain` resumes from `last.pt` (the
  last finished epoch), completed arms are reused from `outputs/ablations/seed_42/arms/`, and
  finished backbones from the shared store (`outputs/checkpoints/ablations/by_fingerprint/`,
  one locked trainer per fingerprint). A host-RAM kill (SIGKILL) is not retried; re-launching
  the script resumes the same way. `--no_resume` starts over.
- **Collect.** Copy back `outputs/ablations/seed_42/` (study JSONs, `arms/`, summary),
  `outputs/plots/`, `logs/` and each backbone's training logs
  (`outputs/checkpoints/ablations/by_fingerprint/<fp>/logs/`) from both machines; every arm records the backend and device it
  ran on under `runtime`.

#### Rescoring a finished run

`--stage ablation_rescore` (`specs/eval-integrity` FR-8) re-evaluates a finished run's saved
epoch checkpoints with the current evaluation protocol: fp32 scores, tie-averaged metrics,
the full test split and a popularity floor. It trains nothing and modifies no checkpoint and
no original result.

```bash
# 1. Mac: package the new code with the rescore scripts (same flags as the run's package)
uv run python scripts/package_ablation_run.py --rescore-scripts --skip-data   # dist/code.tar.gz
# 2. each machine, in the run's own directory: extract the new code OVER the old code.
#    The tarball holds only tracked code + launch/, never outputs/ or data/, so the
#    checkpoints, results and processed data stay untouched.
tar -xzf code.tar.gz
# 3. rescore detached (resumable: a re-run continues at the first unscored epoch)
mkdir -p logs && setsid nohup bash launch/rescore_aws.sh > logs/rescore_aws.log 2>&1 < /dev/null &
tail -f logs/rescore_aws.log logs/rescore_B_t_base_seed42.log
```

- **What it runs.** `launch/rescore_<machine>.sh` runs `--stage ablation_rescore --study X`
  for each of that machine's studies, with exactly the base flags of its run commands
  (sample fraction and seed, `--dtype bf16 --mixed_precision`, the backend), through
  `scripts/run_with_retry.py`. A final `--study all` writes the rescored summary; it exits 1
  because the other machine's studies are absent there, which is expected.
- **Outputs.** `outputs/ablations/rescored/seed_42/` (rescored arm records under `arms/`,
  study JSONs and `summary_t_base.json`), one `rescore.json` per backbone in
  `outputs/checkpoints/ablations/by_fingerprint/<fp>/`, and plots in `outputs/plots/rescored/`.
  The original `outputs/ablations/seed_42/` stays next to them. Each rescored arm keeps the
  original numbers under `rescore.original_*`, and its weight-dependent extras (routing,
  gradient, cold-start, long-tail, personalisation, communication) under
  `rescore.pre_rescore_extras`.
- **Locally.** Copy a machine's `outputs/` and the processed data into the checkout and run the
  same command with `uv run python main.py --stage ablation_rescore --study B ...` (the run's
  flags). The device is chosen automatically (CUDA, then MPS, then CPU). Use MPS: about 8 min
  per backbone, against about 3 h on CPU. `--study` limits the work.
- **Caveat.** Rescoring gives honest numbers for these weights, but the weights were trained
  under the stale-autocast-cache bug (eval-integrity FR-3). The A–H comparison requires a
  re-run with the fixed training code.
- **The re-run retrains every arm.** The training fingerprint now carries a
  training-semantics version (FR-4), so the next `--stage ablation` on these machines never
  reuses a pre-fix backbone; it trains new ones beside the old store entries.

### Splitting studies across GPUs

On a machine with several GPUs the packager plans that machine's arms over them
(`specs/ablation-run` FR-14–FR-18). The generated `launch/LAUNCH.md` remains the per-run source
of truth for the exact commands; this is the overview.

```bash
uv run python scripts/package_ablation_run.py --aws-gpus 4 \
    --h100-studies A,D,E --aws-studies B,C,F,G,H \
    --estimate-json estimate.json      # optional: scripts/estimate_training_time.py --out
```

| Packager option | Meaning |
|---|---|
| `--gpus N` | GPUs on every machine (default 1 = the single sequential loop, unchanged output). It also applies to the usually single-GPU H100 (the packager warns); prefer `--aws-gpus N` |
| `--h100-gpus N` / `--aws-gpus N` | Per-machine GPU count |
| `--h100-gpu-ids` / `--aws-gpu-ids` | Explicit `CUDA_VISIBLE_DEVICES` ids, e.g. `0,1,2,3` |
| `--estimate-json PATH` | Training hours per arm from `estimate_training_time.py --out`; otherwise the L4 fallback (1.81 h for ≤ 50 positions, 4.5 h × S/100 above, scaled by user fraction and epochs) |
| `--allow-dirty` | Package an uncommitted tree (otherwise refused) |

- **Plan.** Arms that share a training fingerprint are one group and always land on the same
  GPU; groups are assigned longest-first to the least-loaded GPU. `launch/plan_<machine>.json`
  lists each group, its fingerprint and hours, and each GPU's arms and hours.
- **Launch.** The machine's usual one-liner runs `launch/run_<machine>.sh`, which starts one
  `launch/run_<machine>_gpu<id>.sh` per GPU in the background (each exports
  `CUDA_VISIBLE_DEVICES` and runs `--study X --arms ...` commands through the retry wrapper),
  waits for all of them, then runs `--stage ablation_collect` for every study. Per-GPU logs:
  `logs/run_<machine>_gpu<id>.log` and `logs/ablation_<model>_seed<seed>_gpu<id>_NN_<study>.log`.
- **Study outputs.** Each arm is persisted on its own; a study's JSON and plots are written,
  under a per-study lock, by whichever process completes the study (others log `... arms
  complete; missing [...] -- study JSON and plots deferred ...`), and again by the final
  collect. Re-run `python main.py --stage ablation_collect --study X ...` (same flags as the
  run) at any time; it exits 1 and lists missing, failed, stale or inconsistent arms.
- **Waiting.** A process that needs a backbone another process is training logs
  `Backbone <fp> is being trained by another process (pid host time: ...) -- waiting for its
  lock` once and then reuses it; if the holder dies, the waiter resumes the holder's
  `last.pt`. The planner avoids this by keeping groups on one GPU.

---

## Model Variants

| Variant | Layers | Hidden Dim | Heads | Approx. Params |
|---|---:|---:|---:|---|
| **T-Small** | 3 | 256 | 4 | ~2M |
| **T-Base** | 4 | 320 | 4 | ~4M |
| **T-Large** | 5 | 384 | 6 | ~6M |

All variants use:
- Pre-LN Transformer architecture
- Learnable positional embeddings
- BF16 computation throughout
- Tied item embedding / output projection weights

---

## Training Objectives

The system trains with four objectives simultaneously:

| Objective | Description | Loss Function |
|---|---|---|
| **Next-Item Prediction** | Predict the next item in the sequence | Sampled softmax with tied weights |
| **Masked Item Modeling (MIM)** | BERT4Rec-style: predict randomly masked items | Cross-entropy at masked positions |
| **Multi-Step Prediction** | Predict next k items from causal positions | Per-step projected cross-entropy |
| **Contrastive Learning** | InfoNCE between original and augmented views | In-batch negatives with temperature |

**Loss Balancing:**
- **GradNorm** (default): Dynamically learns per-objective weights λ_t, anchored on the last shared Transformer block. Renormalizes so active lambdas sum to the number of active objectives.
- **Static weights** (ablation only): Fixed weights from config (`masked_lm_weight`, `multi_step_weight`, `contrastive_weight`).

The contrastive loss is computed every `contrastive_interval` batches (default 4) to limit training time overhead.

---

## MoE Routing Variants

| Variant | Config | Description |
|---|---|---|
| **No routing** | `moe_variant="none"` | Loss heads consume raw Transformer hidden states directly |
| **Conventional MMoE** | `moe_variant="mmoe"` | Independent per-objective gates; no shared routing parameters |
| **ARV-MMoE** (proposed) | `moe_variant="arv_mmoe"` | Single shared router conditioned on `[H_u; e_t]` — user-adaptive, objective-conditioned |

ARV-MMoE computes K shared expert outputs once per forward pass. Each objective gets a different expert mixture based on its objective embedding, while adapting to the user's representation. The same architecture serves both training and inference — at inference time, the next-item embedding (`e_rec ≡ e_next`) is used for routing.

---

## Federated Personalization

The federated setting personalizes a globally pretrained model using LoRA adapters:

1. **Global pretraining**: Train the full model centrally (Transformer + MoE + GradNorm)
2. **LoRA injection**: Replace Q and V attention projections with low-rank adapters
3. **Local training**: Each client trains only LoRA parameters (backbone, MoE, router, experts, and objective embeddings are frozen)
4. **Aggregation**: Weighted FedAvg on LoRA state dicts (weighted by client sample count)
5. **Communication**: Only LoRA adapter weights cross the wire — backbone and MoE parameters are never transmitted

**Parameter isolation guarantees:**
- Backbone checksum verified before and after every federated round
- LoRA state dict explicitly filtered to exclude MoE/router/expert/objective-embedding keys
- Zero trainable backbone parameters asserted before each local training run

---

## Evaluation Protocol

### Metrics

| Metric | Formula (single target, rank ≤ k) |
|---|---|
| NDCG@10 | `1 / log2(rank + 1)` |
| MRR@10 | `1 / rank` |
| Recall@10 | `1` (= Hit Rate for single-target protocol) |
| Precision@10 | `1 / 10` |

### Fixed Candidate Protocol

Every evaluation example uses a pre-stored candidate set containing the ground-truth target plus filtered negatives. Candidate sets are generated once and shared identically across all model evaluations. Negatives exclude the target, padding, mask, and history-filtered items.

### Specialized Evaluations

| Evaluation | Description |
|---|---|
| **Cold-start** | Items first seen after the training cutoff; evaluated with metadata ON and OFF |
| **Long-tail** | Items split into head (top 20%), mid (middle 20%), tail (bottom 60%) by training popularity |
| **Latency** | Single-sample inference timing with mandatory device synchronization |
| **Routing diagnostics** | Entropy, expert utilization, objective differentiation, collapse tests |
| **Gradient diagnostics** | Per-objective gradient cosine similarity and gradient norm analysis |

---

## Ablation Studies A–H

| Study | Question | Configurations |
|---|---|---|
| **A** | Architecture progression | 6 configs: single-obj → +multi-obj → +MMoE → +ARV → +GradNorm → +federated |
| **B** | Routing formulation | `objective_conditioned` vs `input_only` router conditioning |
| **C** | MMoE vs ARV-MMoE | Conventional per-objective gates vs shared conditioned router |
| **D** | ARV vs GradNorm | 2×2 grid: {no routing, ARV} × {no GradNorm, GradNorm} |
| **E** | Personalization | Global only, +LoRA, +LoRA+ARV |
| **F** | Objective ablation | Full system vs remove each auxiliary objective |
| **G** | Expert count | K ∈ {2, 4, 6, 8} |
| **H** | System-level | H1: metadata, H2: seq length, H3: cold-start, H4: long-tail, H5: latency, H6: communication |

All ablations are config-driven and executed via:

```bash
python main.py --stage ablation --study {A..H|all} --model {model}
```

Results are saved to `outputs/ablations/{study}_{model}.json` with full resolved configurations, metrics (mean ± std), and pointers to generated plots.

---

## Outputs & Deliverables

### Checkpoints

| Path | Content |
|---|---|
| `outputs/checkpoints/{model}/best.pt` | Best checkpoint (selected by val NDCG@10 only) |
| `outputs/checkpoints/{model}/epoch_{N:03d}.pt` | Per-epoch checkpoints |

### Logs

| File | Content |
|---|---|
| `outputs/logs/pretrain_{model}.log` | Per-epoch training loss, val metrics, LR, GradNorm lambdas |
| `outputs/logs/personalization_gain_{model}.json` | Per-user LoRA personalization gains |
| `outputs/logs/federated_rounds_{model}.json` | Per-round val NDCG, communication cost |
| `outputs/logs/cold_start_{model}.json` | Cold/warm × metadata ON/OFF metrics |
| `outputs/logs/longtail_{model}.json` | Head/mid/tail metrics |
| `outputs/logs/efficiency_{model}.json` | Parameter counts, latency stats, model sizes |
| `outputs/logs/routing_diagnostics_{model}.json` | Entropy, utilization, collapse tests |
| `outputs/logs/gradient_diagnostics_{model}.json` | Cosine similarity matrix, gradient norms |
| `outputs/logs/gradnorm_{model}.jsonl` | Per-step GradNorm lambda trajectory |
| `outputs/logs/moe_parameter_report_{model}.json` | Router/expert/gate parameter breakdown |
| `outputs/logs/run_config.json` | Resolved configuration snapshot |

### Plots

| File | Content |
|---|---|
| `accuracy_vs_model_size.png` | NDCG@10 vs model size (BF16 MB) |
| `accuracy_vs_communication.png` | NDCG@10 vs communication cost/round |
| `accuracy_vs_flops.png` | NDCG@10 vs FLOPs |
| `personalization_gain_histogram_{model}.png` | Distribution of per-user NDCG gains |
| `global_vs_personalized_{model}.png` | Scatter: global vs personalized NDCG per user |
| `federated_convergence_{model}.png` | Val NDCG@10 vs FL round |
| `cold_vs_warm_{model}.png` | Cold/warm × metadata ON/OFF comparison |
| `longtail_{model}.png` | Head/mid/tail metric comparison |
| `architecture_progression.png` | Ablation A: 6-config metric progression |
| `mmoe_vs_arvmmoe_comparison.png` | Ablation C: grouped bar comparison |
| `expert_count_ablation.png` | Ablation G: NDCG + latency vs K |
| `gradnorm_lambda_trajectory_{model}.png` | λ_t per objective over training |
| `routing_entropy_{model}.png` | Mean entropy per objective |
| `expert_utilization_{model}.png` | Per-expert utilization with 1/K reference |
| `objective_expert_heatmap_{model}.png` | Objective × expert mean routing weights |
| `gradient_cosine_similarity_{model}.png` | Pairwise objective gradient cosine sim |
| `gradient_norms_{model}.png` | Per-objective gradient norms |

### Ablation Results

| File | Content |
|---|---|
| `outputs/ablations/seed_{seed}/{study}_{model}.json` | Per-study configurations, metrics, runtime (backend/device) |
| `outputs/ablations/seed_{seed}/arms/{study}_{model}/{label}.json` | Each completed arm, persisted as it finishes (reused on re-run) |
| `outputs/ablations/seed_{seed}/summary_{model}.json` | Aggregated headline numbers across all studies |

### Data Artifacts

| File | Content |
|---|---|
| `data/processed/dataset_stats.json` | num_users, num_items, vocab_size, feature_dim |
| `data/processed/item_metadata.npy` | Dense (num_items+1, feature_dim) metadata matrix |
| `data/processed/cold_items.pkl` / `warm_items.pkl` | Cold/warm item ID sets |
| `data/processed/popularity_buckets.pkl` | Head/mid/tail item assignments |
| `data/processed/global_negative_pool.npy` | Deterministic warm-item negative pool |

---

## Reproducibility

### Seeds

- **Development/smoke testing:** single seed `42`
- **Final paper results:** independent seeds `[42, 123, 456]` — mean ± std reported across seeds

### Determinism

Every run records:
- `torch.manual_seed`, `np.random.seed`, `random.seed` values
- `torch.backends.cudnn.deterministic = True`
- PyTorch version, device, dtype, resolved attention backend
- Kaggle dataset identifier and dataset fingerprint
- Global negative pool fingerprint
- Fixed evaluation candidate set fingerprint

### Resolved Configuration

The complete resolved configuration (including any CLI overrides) is saved to `outputs/logs/run_config.json` at the start of every run.

### Checklist

- [ ] All random state sources seeded
- [ ] Final runs use independent seeds from `config.final_seeds`
- [ ] Dataset fingerprint matches across runs
- [ ] Negative pool and candidate sets are reused (not regenerated)
- [ ] `train_user_history is not eval_user_history` asserted
- [ ] Test data untouched until checkpoint selection is finalized
- [ ] GradNorm lambda sum constraint verified per step
- [ ] LoRA state dict contains no MoE/router/expert/embedding keys
- [ ] Frozen backbone checksum unchanged through all federated rounds
- [ ] BF16 used throughout; no silent dtype substitution
- [ ] Attention backend self-test passes before training

---

## Hardware Requirements

| Stage | Hardware | Attention Backend | Notes |
|---|---|---|---|
| Smoke testing | Apple Silicon Mac (MPS) | SDPA | One-batch and few-step T-Small tests on real data |
| Development | Apple Silicon Mac (MPS) | SDPA | Full T-Small training feasible |
| Experiments | NVIDIA H100 (Hopper, CUDA) | FlashAttention-3 | Ablations A, D, E, H of the 10% run; final runs |
| Experiments | Other CUDA GPUs (e.g. AWS L4) | SDPA | Ablations B, C, F, G of the 10% run |

Arms that are compared with each other (one ablation study) always run on one machine, so they share hardware, precision and attention backend.

**Dtype:** BF16 by default (`dtype="bf16"`, pure bf16 weights). `--mixed_precision` switches to fp32 master weights with autocast in `--dtype` (bf16 or fp16); fp16 is only allowed with mixed precision and adds a GradScaler. Use `--dtype fp16 --mixed_precision` on GPUs without native bf16 (e.g. T4). **Experiment runs use `--dtype bf16 --mixed_precision`** (owner decision 2026-10-15): with pure bf16 weights small updates are rounded away and training plateaus.

**Fail-fast policy:** The MPS smoke test fails immediately if SDPA cannot execute. A run that requests `--attention_backend flash_attention_3` fails immediately if FlashAttention-3 cannot initialize; experiment commands always name their backend explicitly. Silent fallback to another kernel is forbidden.

---

## Build Order

The full 22-phase implementation sequence with test gates is in [`specs/BUILD_ORDER.md`](specs/BUILD_ORDER.md). Each step maps to specific spec tasks. Key blocking gates:

| Phase | Gate | Blocks |
|---|---|---|
| 0 | Environment validates | Everything |
| 3 | Sequence/pool/candidate tests pass | All model training |
| 5 | One-batch forward+backward smoke test | Multi-epoch training |
| 11 | **One-batch + few-step T-Small smoke tests (real data)** | **Full MovieLens-20M training** |
| 13 | Hopper FlashAttention-3 self-test + parity (if CUDA) | H100 training |

**Dependency chain:** data-pipeline → transformer-backbone → moe-routing → loss-functions → training-pipeline → personalization-federated → evaluation-diagnostics → ablations-cli

---

## Specifications & Steering

This project uses a spec-driven development workflow. Implementation contracts live under `steering/` and `specs/`:

### Steering (project-wide guidance)

| Document | Content |
|---|---|
| [`steering/architecture.md`](steering/architecture.md) | System overview, component map, data flow, boundaries |
| [`steering/code-quality.md`](steering/code-quality.md) | BF16 discipline, device handling, numerical safety, anti-patterns |
| [`steering/testing.md`](steering/testing.md) | Test gate, 18-phase test progression, runtime assertions |
| [`steering/conventions.md`](steering/conventions.md) | ID scheme, objective order, left-padding, file organization |

### Feature Specs (per-feature contracts)

| Spec | Requirements | Design | Tasks |
|---|---|---|---|
| [`data-pipeline`](specs/data-pipeline/) | Acquisition, preprocessing, sequences, negative pool | Data flow, split-before-truncate, ID scheme | 23 tasks |
| [`transformer-backbone`](specs/transformer-backbone/) | Attention backends, model sizes, metadata, normalization | Pre-LN architecture, backend wrapper, tied weights | 13 tasks |
| [`moe-routing`](specs/moe-routing/) | ARV-MMoE, conventional MMoE, passthrough, parameter accounting | Expert caching, broadcast routing, serving path | 17 tasks |
| [`loss-functions`](specs/loss-functions/) | 4 losses, GradNorm, augmentation | Causal positions, interval gating, anchor selection | 19 tasks |
| [`training-pipeline`](specs/training-pipeline/) | Pretraining loop, LR scheduling, checkpointing | Per-batch scheduling, retain_graph, routing buffers | 18 tasks |
| [`personalization-federated`](specs/personalization-federated/) | LoRA, personalization, federated simulation | Fresh LoRA per user, checksum enforcement, weighted FedAvg | 21 tasks |
| [`evaluation-diagnostics`](specs/evaluation-diagnostics/) | Metrics, cold-start, long-tail, latency, diagnostics | Fixed candidates, device sync, diagnostic batches | 25 tasks |
| [`ablations-cli`](specs/ablations-cli/) | CLI, ablation runner A–H, plots | Config-driven studies, multi-seed aggregation | 21 tasks |
| [`ablation-run`](specs/ablation-run/) | 10%-user ablation run: mixed bf16, FA3 on Hopper, LR floor, user sampling, wide storage, resume, OOM retry, packaging | FA3 unpad/repad, bit-identical throughput levers, epoch-granular resume, store lock | 28 tasks |

**Total: 157 implementation tasks** across 7 feature specs, ordered by dependency chain.

---

## Key Design Decisions

1. **Shared experts, differentiated routing.** K expert FFN outputs are computed once per forward pass and reused by all four objectives — only the mixture weights differ per objective. This avoids redundant computation while enabling objective-specific representations.

2. **Pooled-then-broadcast routing.** The routing distribution is computed once from the user's pooled representation `H_u`, then broadcast across all sequence positions. This keeps routing cheap and user-adaptive without introducing a position-dependent routing matrix.

3. **`e_rec ≡ e_next` at inference.** No separate inference-only embedding exists. The next-item objective embedding is already trained through the next-item loss and serves as the routing key at inference time.

4. **LoRA-only federated communication.** Router, experts, objective embeddings, and GradNorm state are global parameters that never enter the federated payload. This is central to the communication-efficiency claim (RQ4).

5. **GradNorm on shared Transformer anchor.** The anchor is the last shared Transformer block (before MoE), ensuring it exists in every model variant including `moe_variant="none"`. A separate optimizer with its own learning rate controls the loss weights.

6. **Split-before-truncate.** Chronological train/val/test boundaries are determined on complete user histories. Per-example truncation to `max_seq_length` happens afterward — preventing information leakage through shifted split points.

7. **Fixed evaluation candidate sets.** Candidate sets are generated once and shared across all model evaluations. This guarantees identical comparison conditions — metric differences reflect model quality, not candidate sampling variance.

---

## Strict Rules

1. All model components implemented manually in PyTorch — no external recommendation libraries.
2. BF16 by default; FP16 only via `--mixed_precision` (fp32 master weights + GradScaler). No silent dtype substitution.
3. `configs/config.py` is the single source of truth — no hardcoded hyperparameters elsewhere.
4. Test data is never used for model selection, threshold tuning, or diagnostic decisions.
5. Attention backends must pass a startup self-test and never silently fall back.
6. Cold items must be genuinely unseen during training — verified by set-disjointness assertions.
7. LoRA state dicts must never contain MoE/router/expert/objective-embedding keys.
8. Frozen backbone checksum must be unchanged through every federated round.
9. Full MovieLens-20M training is forbidden until all unit, integration, and smoke tests pass.
10. No ablation may require commenting/uncommenting code or manually editing source files.

---

## License

This project is part of academic coursework at VIT Chennai. See the project report for citation and usage terms.
