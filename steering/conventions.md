# Conventions

## Canonical Item ID Indexing Scheme

| ID Value | Meaning |
|---|---|
| `0` | Padding token (reserved; never a real item) |
| `1 .. num_items` | Real item IDs (contiguous integers) |
| `num_items + 1` | Mask token (BERT4Rec masking) |
| — | Embedding vocab size = `num_items + 2` |

This scheme is enforced during ID remapping and must be respected everywhere: embedding tables, negative samplers, metadata lookups, loss functions, evaluation candidate sets.

## Objective Order (Fixed, Global)

```python
OBJECTIVE_ORDER = ["next_item", "masked_item", "multi_step", "contrastive"]
```

This order must be identical across: model definition, loss computation, GradNorm controller, objective embeddings, routing diagnostics, gradient diagnostics, and all plots. Index 0 = next_item, 1 = masked_item, 2 = multi_step, 3 = contrastive.

## Left-Padding Convention

All model inputs use fixed length `S = config.max_seq_length` with left padding (pad ID = 0). The final valid hidden state is always at index `S-1`. Multi-step step k reads position `S-1-k`. Never use `seq_lens-1-step` for position selection under this convention.

**Stored width ≠ model width** (owner decision 2026-10-15, `specs/ablation-run`). Preprocessing stores every split at `max(sequence_length_grid ∪ {max_seq_length})` (manifest field `stored_seq_length`). Every consumer narrows to its own `max_seq_length` by keeping the **last** `S` positions and clamping `lengths` to `S` — through `load_tensorised(..., max_seq_length=S)` for `*_tensorised.pt` and `load_examples(path, S)` for `*_examples.pkl`. Never read either artifact without narrowing. Narrowing is exact: it equals generating the data at width `S`.

## File Organization

```
project_root/
├── data/                          # Data acquisition, preprocessing, sequences
│   ├── raw/                       # 6 raw CSV files from Kaggle
│   ├── processed/                 # Tensorised outputs, mappings, pools (all users)
│   ├── processed_users{pct}pct/   # Same artifacts for a seeded user sample (+ `_seed{S}` if non-default seed)
│   ├── preprocessing.py
│   ├── sequence_generator.py
│   └── augmentation.py
├── models/                        # All model components
│   ├── transformer.py             # Backbone + metadata integration
│   ├── moe.py                     # ARV-MMoE + conventional MMoE
│   ├── lora.py                    # LoRA injection and utilities
│   └── ann_retrieval.py           # ANN / brute-force retrieval
├── training/                      # Training loops and loss functions
│   ├── pretrain.py                # Central pretraining
│   ├── local_train.py             # Local personalization
│   ├── federated.py               # Federated simulation
│   ├── losses.py                  # ALL loss functions in one file
│   ├── gradnorm.py                # GradNorm controller
│   └── ablation_runner.py         # Ablation studies A–H
├── evaluation/                    # All evaluation and diagnostics
│   ├── metrics.py                 # NDCG, MRR, Recall, Precision, calibration
│   ├── flops.py                   # Latency benchmarking, parameter accounting
│   ├── cold_start.py              # Cold-start evaluation
│   ├── longtail.py                # Long-tail evaluation
│   ├── routing_diagnostics.py     # Entropy, utilization, collapse tests
│   └── gradient_diagnostics.py    # Gradient cosine sim, gradient norms
├── configs/config.py              # Single Config dataclass
├── utils.py                       # get_device(), resolve_dtype(), seed utils
├── requirements.txt
└── main.py                        # CLI entrypoint
```

## Output Organization

```
outputs/
├── checkpoints/{model}/           # best.pt + epoch_NNN.pt + last.pt (resume) per model
├── checkpoints/ablations/by_fingerprint/{fp}/   # shared backbone store (+ .train.lock, backbone.json)
├── logs/                          # JSON/JSONL/log files per experiment
├── plots/                         # All PNG plots (150 DPI)
└── ablations/seed_{seed}/         # Per-study JSON results (A–H), summary, arms/{study}_{model}/{label}.json (+ .study.lock)
```

Ablation results are seed-namespaced (`specs/ablation-run` FR-11) so runs with different seeds never overwrite each other.

When arms run in separate processes (`--arms`, `specs/ablation-run` FR-17), each arm's post-training side outputs go to `logs/arms/{study}_{label}/` and `plots/arms/{study}_{label}/`, so no two processes write the same file. A file written by more than one process (run manifest, study JSON) is written atomically, and study-level outputs are written under the study lock.

## Naming Conventions

- Model names: `t-small`, `t-base`, `t-large` (CLI); `t_small`, `t_base`, `t_large` (directories/files)
- Output files: `{metric_type}_{model}.{ext}` (e.g., `pretrain_t_base.log`, `cold_start_t_base.json`)
- Plot files: `{descriptive_name}_{model}.png` or `{descriptive_name}.png` for cross-model plots
- Checkpoint files: `epoch_{N:03d}.pt`, `best.pt`, `last.pt` (latest full resume state)
- Process exit status `75` means "out of memory, safe to retry" (`utils.OOM_EXIT_CODE`)

## Config-Driven Design

- Every experiment variation (ablation, MoE variant, GradNorm on/off, expert count) must be expressible as config flags.
- No ablation requires commenting/uncommenting code or manually editing source files.
- The ablation runner sets config values programmatically from `--study`.
- Every resolved configuration is saved with results.

## Tied Weights

- Sampled softmax uses tied item/output weights: `item_emb_weight` from the item embedding table is reused as the output projection.
- Never create a separate output weight matrix.

## Embedding Normalization

When `config.normalize_embeddings = True` (default), L2-normalize both user and item embeddings before any dot-product scoring, in both training and evaluation.

## Plot Standards

- Format: PNG, 150 DPI
- All plots have labeled axes, legends, and consistent palette
- Saved to `outputs/plots/`
