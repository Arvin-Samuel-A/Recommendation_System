# Build Order

The canonical implementation sequence. Each step maps to specific spec tasks and lists its test gate. The BUILD agent must follow this order — no step may begin until its prerequisites and test gates are satisfied.

**Legend:**
- `→` references a spec task (e.g., `data-pipeline T1.1`)
- `🧪 GATE` marks a test that must pass before proceeding to the next phase
- `⛔ BLOCKED` marks what is forbidden until the gate passes

---

## Phase 0 — Foundation

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 0.1 | Create `configs/config.py` | `data-pipeline T1.1` | Config dataclass with all hyperparameters |
| 0.2 | Create `utils.py` | `data-pipeline T1.2` | `get_device()`, `resolve_dtype()`, seed setup |
| 0.3 | Create `requirements.txt` | `data-pipeline T1.3` | Dependency manifest |

```
🧪 GATE: Environment validates — uv venv creates, dependencies install, imports succeed
```

---

## Phase 1 — Dataset Acquisition & Validation

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 1.1 | Kaggle acquisition + file validation | `data-pipeline T2.1` | 6 CSVs under `data/raw/`, fingerprint |
| 1.2 | Test acquisition | `data-pipeline T2.2` | Validated file presence, columns, row counts |

```
🧪 GATE: All 6 CSV files present, validated, and fingerprinted
```

---

## Phase 2 — Data Preprocessing

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 2.1 | Ratings pipeline (filter, remap, sort) | `data-pipeline T3.1` | `user_sequences.pkl`, mappings, `dataset_stats.json` |
| 2.2 | Test ratings pipeline | `data-pipeline T3.2` | ID scheme verified (pad=0, items=1..N, mask=N+1) |
| 2.3 | Chronological splitting | `data-pipeline T4.1` | Train/val/test boundaries on complete histories |
| 2.4 | Cold-start holdout | `data-pipeline T4.2` | `cold_items.pkl`, `warm_items.pkl` |
| 2.5 | Test cold-start | `data-pipeline T4.3` | Cold items verified post-cutoff, disjoint from training |
| 2.6 | Metadata extraction | `data-pipeline T5.1` | `item_metadata.npy` |
| 2.7 | Test metadata | `data-pipeline T5.2` | Matrix shape, alignment, zero-padding verified |

```
🧪 GATE: Chronological split, cold-start, and metadata unit tests pass
```

---

## Phase 3 — Sequences, Augmentation, & Negative Pool

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 3.1 | Sequence augmentation strategies | `data-pipeline T6.1` | `data/augmentation.py` |
| 3.2 | Test augmentation | `data-pipeline T6.2` | Crop/mask/subsequence verified |
| 3.3 | Sequence generation + left-padding | `data-pipeline T7.1` | Sliding-window examples, separate history dicts |
| 3.4 | Pre-tensorisation (`RecDataset`) | `data-pipeline T7.2` | Cached tensors, index-only `__getitem__` |
| 3.5 | DataLoader construction | `data-pipeline T7.3` | Configured DataLoader |
| 3.6 | Test sequences + tensorisation | `data-pipeline T7.4` | Split-before-truncate, padding, history separation verified |
| 3.7 | Global negative pool | `data-pipeline T8.1` | `global_negative_pool.npy` |
| 3.8 | Per-example negative filtering | `data-pipeline T8.2` | Filtering logic for training |
| 3.9 | Fixed evaluation candidate sets | `data-pipeline T8.3` | Pre-stored candidate sets shared across models |
| 3.10 | Test negative pool + candidates | `data-pipeline T8.4` | Warm-only, deterministic, cross-model consistent |

```
🧪 GATE: Sequence, left-padding, multi-step indexing, negative pool, and candidate set tests pass
⛔ BLOCKED: No model training until this gate passes
```

---

## Phase 4 — Transformer Backbone

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 4.1 | Attention backend wrapper + self-test | `transformer-backbone T1.1, T1.3, T1.4` | `AttentionBackend` with SDPA/FA3 (FA4 renamed to FA3 by `ablation-run T4.2`) |
| 4.2 | Test attention backends | `transformer-backbone T1.2` | Causal + padding mask verified |
| 4.3 | Embedding layer (item + position + tied output) | `transformer-backbone T2.1` | Embedding modules |
| 4.4 | Metadata projection | `transformer-backbone T2.2` | Optional linear projection |
| 4.5 | Transformer block (Pre-LN) | `transformer-backbone T3.1` | Single block |
| 4.6 | Full Transformer model + size factory | `transformer-backbone T3.2, T3.3` | T-Small/Base/Large forward |
| 4.7 | Shape + tied weight + BF16 tests | `transformer-backbone T4.1–T4.5` | All model shapes and invariants verified |

```
🧪 GATE: Transformer + MPS SDPA self-test passes (forward + backward)
```

---

## Phase 5 — Transformer Smoke Test

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 5.1 | One-batch forward + backward smoke test | `transformer-backbone T4.6` | Loss finite, gradients exist, no NaN/Inf |

```
🧪 GATE: Transformer baseline one-batch smoke test passes
⛔ BLOCKED: No multi-epoch training until this gate passes
```

---

## Phase 6 — Loss Functions

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 6.1 | Sampled softmax loss | `loss-functions T1.1` | `SampledSoftmaxLoss` |
| 6.2 | Masked item modeling loss | `loss-functions T1.2` | `MaskedItemLoss` |
| 6.3 | Multi-step prediction loss | `loss-functions T1.3` | `MultiStepLoss` |
| 6.4 | InfoNCE contrastive loss | `loss-functions T1.4` | `InfoNCELoss` |
| 6.5 | Input masking function | `loss-functions T2.1` | `apply_masking()` |
| 6.6 | Test all loss functions | `loss-functions T4.1–T4.4` | Each loss verified on toy data |

```
🧪 GATE: All loss function unit tests pass
```

---

## Phase 7 — ARV-MMoE & Conventional MMoE

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 7.1 | Expert + SharedExperts | `moe-routing T1.1, T1.2` | K position-wise FFN experts |
| 7.2 | ARVRouter + objective embeddings | `moe-routing T1.3, T1.4` | Shared conditioned router |
| 7.3 | ConventionalMMoEGates | `moe-routing T1.5` | Independent per-objective gates |
| 7.4 | `route_objective` + `MoELayer` | `moe-routing T2.1, T2.2` | Unified MoE interface |
| 7.5 | `count_router_parameters` | `moe-routing T2.3` | Parameter accounting |
| 7.6 | Test MoE: shapes, differentiation, caching, passthrough, accounting | `moe-routing T3.1–T3.9` | All MoE invariants verified |

```
🧪 GATE: ARV-MMoE / conventional MMoE shape and sharing tests pass
```

---

## Phase 8 — GradNorm

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 8.1 | `GradNormController` (lambdas, optimizer, initial losses) | `loss-functions T3.1` | Controller with weighted_sum |
| 8.2 | GradNorm update step (gradients, renormalization) | `loss-functions T3.2` | Full update cycle |
| 8.3 | Anchor resolution (`last_shared_transformer`) | `loss-functions T3.3` | Anchor that works for all variants |
| 8.4 | GradNorm state dict save/load | `loss-functions T3.4` | Checkpoint compatibility |
| 8.5 | Static weight fallback | `loss-functions T3.5` | Ablation-ready fallback |
| 8.6 | Test GradNorm: lambda sum, active-only, anchor, finite assertions | `loss-functions T4.5–T4.9` | All GradNorm invariants verified |

```
🧪 GATE: GradNorm active-objective / lambda / anchor tests pass
```

---

## Phase 9 — LoRA

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 9.1 | `LoRALinear` wrapper | `personalization-federated T1.1` | Low-rank adapter module |
| 9.2 | `inject_lora` (Q and V replacement + freeze) | `personalization-federated T1.2` | LoRA-injected model |
| 9.3 | `get_lora_state_dict` (filtered, MoE-excluded) | `personalization-federated T1.3` | Clean LoRA state dict |
| 9.4 | `load_lora_state_dict` + `count_lora_parameters` | `personalization-federated T1.4` | Load + count utilities |
| 9.5 | Test LoRA: injection, filtering, count, freeze | `personalization-federated T2.1–T2.4` | All LoRA invariants verified |

```
🧪 GATE: LoRA exact-key / global-freeze tests pass
```

---

## Phase 10 — Evaluation Metrics

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 10.1 | Metric functions (NDCG, MRR, Recall, Precision) | `evaluation-diagnostics T1.1` | `compute_metrics()` |
| 10.2 | Evaluation loop with fixed candidates | `evaluation-diagnostics T1.2` | `evaluate_model()` |
| 10.3 | ANN retrieval path (optional) | `evaluation-diagnostics T1.3` | ANN + force-inclusion |
| 10.4 | Calibration support (optional) | `evaluation-diagnostics T1.4` | Temperature scaling |
| 10.5 | Test metrics on toy inputs | `evaluation-diagnostics T1.5` | Exact values verified |

```
🧪 GATE: Evaluation metric toy tests (including MRR/Recall/Precision) pass
🧪 GATE: ANN target force-inclusion toy test passes
```

---

## Phase 11 — End-to-End Smoke Tests

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 11.1 | One-batch end-to-end smoke test on Mac | `training-pipeline T6.2` | Full pipeline: data → model → MoE → losses → GradNorm → backward → step |
| 11.2 | Few-step T-Small smoke test on Mac | `training-pipeline T6.3` | Capped run of `run_pretrain` on real MovieLens data (`--max_steps N`): validation metrics computed, checkpoint saved, log written |

```
🧪 GATE: One-batch AND few-step T-Small (real data) smoke tests pass
⛔ BLOCKED: Full MovieLens-20M training is FORBIDDEN until this gate passes
```

---

## Phase 12 — Training Pipeline

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 12.1 | LR scheduler (`build_scheduler`) | `training-pipeline T1.1` | Warmup + cosine decay |
| 12.2 | Test LR scheduler | `training-pipeline T1.2` | Per-batch stepping verified |
| 12.3 | Model construction function | `training-pipeline T2.1` | Model factory |
| 12.4 | Full training loop | `training-pipeline T2.2` | Complete train step with all objectives |
| 12.5 | Contrastive loss integration (interval gating) | `training-pipeline T2.3` | Interval-gated second forward pass |
| 12.6 | Validation step | `training-pipeline T2.4` | Per-epoch validation |
| 12.7 | Checkpoint save/load + best selection | `training-pipeline T3.1–T3.3` | Checkpointing by val NDCG@10 |
| 12.8 | Per-epoch + GradNorm logging | `training-pipeline T4.1–T4.3` | Log files, lambda trajectory |
| 12.9 | `run_pretrain` orchestrator | `training-pipeline T5.1` | Full pretraining function |
| 12.10 | BF16 + finite assertions | `training-pipeline T5.2` | Runtime safety |
| 12.11 | Test checkpoint roundtrip + negative pool reuse | `training-pipeline T6.1, T6.4` | State preservation verified |

---

## Phase 13 — Target-GPU Validation (if on CUDA)

> **Owner decision (2026-10-15):** FlashAttention-4 is replaced by FlashAttention-3 on Hopper; other
> CUDA GPUs run SDPA. See `specs/ablation-run` FR-2/FR-3 and `steering/code-quality.md`.

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 13.1 | Hopper: install FA3 + self-test + parity vs SDPA | `ablation-run T4.4, T4.5` (run on the H100) | `logs/verify_fa3.json` with `passed: true` |
| 13.2 | Non-Hopper CUDA: SDPA self-test + parity | `ablation-run T4.4` (`--backend sdpa`) | verification report |

```
🧪 GATE: scripts/verify_attention_backend.py passes for the backend the machine will train with
⛔ BLOCKED: No training on that machine until this gate passes
```

---

## Phase 14 — Full Central Pretraining

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 14.1 | Full pretraining on target hardware | `training-pipeline T5.1` | `best.pt` per model |

```
🧪 GATE: Validation NDCG@10 improving; best checkpoint selected
```

---

## Phase 15 — Local Personalization

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 15.1 | Per-user LoRA training function | `personalization-federated T3.2` | Local training |
| 15.2 | 1000-user personalization evaluation | `personalization-federated T3.1` | Per-user metrics |
| 15.3 | Personalization report + plots | `personalization-federated T3.3, T3.4` | JSON + histogram + scatter |

---

## Phase 16 — Federated Simulation

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 16.1 | Weighted FedAvg | `personalization-federated T4.1` | Aggregation function |
| 16.2 | Federated loop with checksum enforcement | `personalization-federated T4.2, T4.3` | Multi-round simulation |
| 16.3 | Communication cost logging | `personalization-federated T4.4` | Per-round KB metrics |
| 16.4 | Convergence plot | `personalization-federated T4.5` | `federated_convergence_{model}.png` |
| 16.5 | Test FedAvg + checksum + convergence | `personalization-federated T5.1–T5.4` | Federated invariants verified |

---

## Phase 17 — Evaluation Suite

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 17.1 | Full test evaluation | `evaluation-diagnostics T1.2` | Final test metrics |
| 17.2 | Cold-start evaluation + plot | `evaluation-diagnostics T2.1, T2.2` | `cold_start_{model}.json` + plot |
| 17.3 | Long-tail evaluation + plot | `evaluation-diagnostics T3.1, T3.2` | `longtail_{model}.json` + plot |
| 17.4 | Latency benchmarking + efficiency report | `evaluation-diagnostics T4.1–T4.3` | `efficiency_{model}.json` |

---

## Phase 18 — Routing & Gradient Diagnostics

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 18.1 | Routing diagnostics (entropy, utilization, collapse, heatmap) | `evaluation-diagnostics T5.1–T5.5` | `routing_diagnostics_{model}.json` + plots |
| 18.2 | Gradient diagnostics (cosine sim, norms) | `evaluation-diagnostics T6.1–T6.4` | `gradient_diagnostics_{model}.json` + plots |
| 18.3 | Test diagnostics correctness | `evaluation-diagnostics T7.1–T7.4` | Entropy/symmetry/partition verified |

---

## Phase 19 — Ablations A–H

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 19.1 | Ablation runner config resolution | `ablations-cli T2.1` | Study-specific config overrides |
| 19.2 | Ablation A — Architecture progression | `ablations-cli T2.2` | 6 configs compared |
| 19.3 | Ablation B — Routing formulation | `ablations-cli T2.3` | Conditioned vs input-only |
| 19.4 | Ablation C — MMoE vs ARV-MMoE | `ablations-cli T2.4` | Parameter-reported comparison |
| 19.5 | Ablation D — ARV vs GradNorm | `ablations-cli T2.5` | 2×2 grid + gradient diagnostics |
| 19.6 | Ablation E — Personalization | `ablations-cli T2.6` | 3-config comparison |
| 19.7 | Ablation F — Objective ablation | `ablations-cli T2.7` | Reference + 3 removals |
| 19.8 | Ablation G — Expert count | `ablations-cli T2.8` | K sweep |
| 19.9 | Ablation H — System-level | `ablations-cli T2.9` | 6 sub-studies |
| 19.10 | `--study all` aggregation | `ablations-cli T2.10` | `summary_{model}.json` |

---

## Phase 20 — CLI Integration

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 20.1 | Argument parsing + override logic | `ablations-cli T1.1, T1.2` | CLI accepts all flags |
| 20.2 | Startup routine (seeding, config snapshot) | `ablations-cli T1.3` | `run_config.json` |
| 20.3 | Stage dispatch + `--stage all` | `ablations-cli T1.4` | All stages routable |
| 20.4 | Test CLI parsing + missing checkpoint handling | `ablations-cli T4.1–T4.5` | All CLI invariants verified |

> **Note:** CLI scaffolding (argument parsing, stage dispatch) can be built early in parallel with Phase 0–4 and wired up incrementally as each stage is implemented. The full test of `--stage all` only runs after Phase 17.

---

## Phase 21 — Multi-Seed Final Runs

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 21.1 | Run full pipeline for each seed in `[42, 123, 456]` | `ablations-cli T3.2` | Independent per-seed outputs |
| 21.2 | Aggregate results (mean ± std across seeds) | `ablations-cli T3.2` | Final paper numbers |

---

## Phase 22 — Final Plots & Deliverables

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 22.1 | Core tradeoff plots | `ablations-cli T3.1` | accuracy_vs_model_size, vs_communication, vs_flops, vs_lora_rank |
| 22.2 | Verify all deliverables present | `data-pipeline T9.1` | All output files accounted for |

---

## Phase 23 — Ablation Run Readiness (10% users, t-base, one seed, mixed bf16)

Spec: `specs/ablation-run/`. Prerequisites: Phases 0–20 implemented (they are). This phase makes
the code, tooling and launch artifacts for the two-machine ablation run; the run itself (Phase 14
training + Phase 19 studies on the target machines) is started by the owner from the generated
`LAUNCH.md` once every gate below has passed. **BUILD runs targeted test files only (never the full
suite) and only capped runs on the Mac** — see `specs/ablation-run/tasks.md` "Rules for BUILD".

| # | Step | Spec Tasks | Produces |
|---|---|---|---|
| 23.1 | Tiny synthetic dataset fixture | `ablation-run T0.1` | `tests/conftest.py` |
| 23.2 | Config fields, CLI flags, fingerprint hygiene | `ablation-run T1.1–T1.3` | `min_lr_ratio`, `user_sample_*`, `resume_training`; derived processed dir |
| 23.3 | User sampling, wide storage, narrowing everywhere, width/position/manifest guards | `ablation-run T2.1–T2.5` | `data/processed_users{pct}pct`, `stored_seq_length` |
| 23.4 | Cosine LR floor, then golden characterisation test | `ablation-run T3.1–T3.2` | floor at 10% of peak; golden constants |
| 23.5 | FlashAttention-3 backend, runtime record, verify + install scripts | `ablation-run T4.1–T4.5` | `flash_attention_3`; `scripts/verify_attention_backend.py`, `install_fa3.sh` |
| 23.6 | Bit-identical utilisation levers | `ablation-run T5.1–T5.2` | workers + pinned memory, CPU-side negatives, fewer syncs |
| 23.7 | Exact epoch-granular resume | `ablation-run T6.1–T6.2` | `last.pt`, atomic checkpoints |
| 23.8 | OOM exit 75 + retry launcher, store lock, arm persistence | `ablation-run T7.1–T7.4` | `scripts/run_with_retry.py`, `ablations/seed_{seed}/` |
| 23.9 | Packaging + README runbook | `ablation-run T8.1–T8.2` | `scripts/package_ablation_run.py`, `dist/` |
| 23.10 | Mac verification | `ablation-run T9.1–T9.2` | targeted test sweep; capped real-data smoke incl. kill/resume |

```
🧪 GATE: test_determinism.py golden test green from T3.2 through T6.2 (training unchanged by FR-4/FR-8)
🧪 GATE: T9.1 targeted sweep green; T9.2 capped smoke shows resume after kill -9 and arm reuse
⛔ BLOCKED: no launch on the H100 until Phase 13.1 (FA3 verification) passes there — run_h100.sh enforces it
```

---

## Summary

| Phase | Description | Test Gate |
|---:|---|---|
| 0 | Foundation (Config, utils, deps) | Environment validates |
| 1 | Dataset acquisition | 6 CSVs present and valid |
| 2 | Preprocessing (ratings, splits, cold-start, metadata) | Split/cold-start/metadata unit tests |
| 3 | Sequences, augmentation, negative pool | Sequence/pool/candidate tests |
| 4 | Transformer backbone | SDPA self-test |
| 5 | Transformer smoke test | One-batch forward+backward |
| 6 | Loss functions | All loss unit tests |
| 7 | ARV-MMoE + conventional MMoE | MoE shape/sharing tests |
| 8 | GradNorm | Lambda/anchor/active-only tests |
| 9 | LoRA | Key-filtering/freeze tests |
| 10 | Evaluation metrics | Metric toy tests |
| 11 | **End-to-end smoke tests** | **One-batch + few-step T-Small on real data** |
| 12 | Training pipeline | Checkpoint roundtrip |
| 13 | Target-GPU validation (if CUDA) | FA3 (Hopper) / SDPA self-test + parity |
| 14 | Full central pretraining | Val NDCG improving |
| 15 | Local personalization | 1000-user evaluation complete |
| 16 | Federated simulation | Convergence + checksum verified |
| 17 | Evaluation suite | Test metrics + specialized evals |
| 18 | Routing + gradient diagnostics | Diagnostic correctness |
| 19 | Ablations A–H | All studies produce results |
| 20 | CLI integration | All stages dispatchable |
| 21 | Multi-seed final runs | 3 independent seeds aggregated |
| 22 | Final plots + deliverables | All outputs present |
| 23 | Ablation run readiness (`ablation-run`) | Golden test green throughout; targeted sweep; capped smoke with resume |

**Critical blocking gate:** Phase 11 (end-to-end smoke tests). Full MovieLens-20M training is **forbidden** until both the one-batch and few-step T-Small smoke tests pass.

> **Owner decision (2026-09-17):** step 11.2 was "one-epoch T-Small smoke test ... full epoch
> on real MovieLens data". A full epoch is 30,447 steps (~2 h measured on the dev host);
> what the gate exists to establish — that `run_pretrain` works end to end on real data,
> computes validation metrics, checkpoints and logs — is established by a few hundred
> steps. REV-066 was right to refuse this relaxation when BUILD proposed it; this one is
> the spec owner's. Note that training-pipeline **AC-04 ("training loss decreases over
> epochs on real data") is NOT covered by the reduced gate** — LR warmup makes any short
> window uninformative — and stays outstanding until full training in Phase 14.
