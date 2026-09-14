# EXECUTION PROMPT v6.2 — Communication-Efficient Multi-Objective Federated Sequential Recommendation with Adaptive Expert Routing (ARV-MMoE)

> **Version:** 6.2 — Production Implementation Prompt
> **Development / smoke-test hardware:** Apple Silicon Mac, MPS
> **Final experiment hardware:** NVIDIA H100, CUDA
> **Environment manager:** `uv venv`
> **Dataset:** MovieLens-20M, acquired automatically from Kaggle via API
> **Reference:** "Communication-Efficient Multi-Objective Federated Sequential Recommendation with Adaptive Expert Routing" project report (VIT Chennai, Aug 2026)

Follow this prompt strictly. This document is the complete implementation specification. Do not layer another specification on top of it or infer omitted requirements that are not stated here.

Do not redesign the paper architecture or experimental questions. Do not use external recommendation libraries. All recommendation-model components must be implemented manually in PyTorch.

## NON-NEGOTIABLE CORRECTNESS PRINCIPLES

1. Never use test data for model selection, temperature fitting, threshold tuning, hyperparameter selection, or diagnostic decisions.
2. Split complete chronological histories before per-example truncation.
3. Cold items must be genuinely unseen during model training.
4. Training and evaluation histories are separate objects.
5. All compared models use identical splits, targets, candidate sets, and evaluation protocol.
6. All final paper results use multiple independent seeds, e.g. `[42, 123, 456]`; a single seed is for development/smoke testing only.
7. Do not claim parameter matching unless parameter counts demonstrate it.
8. Do not turn routing or gradient diagnostics into stronger causal claims than the measurements support.
9. Do not silently fall back from the requested H100 FlashAttention-4 or MPS FlexAttention backend during the final experiment.
10. Full MovieLens-20M training is forbidden until all required unit/integration tests, one-batch smoke tests, and one-epoch T-Small smoke tests pass.

## RESEARCH AND IMPLEMENTATION SCOPE

The implementation includes masked item modeling, multi-step prediction, contrastive learning, metadata, cold-start evaluation, weighted FedAvg, persistent client state, communication accounting, optional ANN retrieval, LR scheduling, sequence augmentation, embedding normalisation, long-tail evaluation, latency benchmarking, personalization comparison, federated convergence, ARV-MMoE routing, GradNorm, routing diagnostics, gradient diagnostics, parameter accounting, and ablations A–H.

Logit temperature scaling is **not** treated as a ranking correction: dividing logits by a positive fixed temperature and applying log-softmax do not change ranking for a fixed candidate set. If calibration is retained, it is for probability/log-probability reporting and must never be claimed to improve NDCG by changing ranking.

## GLOBAL CONSTRAINTS

- Framework: PyTorch 2.x+ with MPS for smoke testing and CUDA/H100 for final runs.
- Python: 3.10+.
- Environment manager: `uv venv` only.
- Default development seed: `42`. Final paper runs use multiple independent seeds.
- Computation dtype: **BF16 everywhere** (`torch.bfloat16`) for model/data computation on both MPS and CUDA. No blanket FP16 mixed-precision policy.
- MPS attention backend: **FlexAttention**.
- H100/CUDA attention backend: **FlashAttention-4**.
- `batch_size`, `num_workers`, and `max_seq_length` are configurable in `configs/config.py`.
- Default `num_workers=0`; do not introduce worker processes merely for convention.
- Dataset tensors are pre-tensorised once and reused.
- A deterministic global shared negative pool is generated once and reused.
- Sampled softmax uses tied item/output weights.
- All four objectives use the shared objective-conditioned ARV-MMoE layer by default.
- Conventional MMoE and `moe_variant="none"` remain available only for the defined ablations/baseline comparisons.
- Per-objective loss balancing uses GradNorm by default; static weights are only for the explicitly defined ablation.
- Federated communication contains only LoRA adapter state. Router, objective embeddings, shared experts, backbone, embeddings, heads, and GradNorm state remain global/frozen during local personalization.
- Every experiment saves its resolved configuration, seed, hardware, dtype, attention backend, dataset fingerprint, metrics, and checkpoint metadata.
- All paths are relative to `project_root/`.


## STEP 0 — ENVIRONMENT SETUP (`uv venv`)

Use `uv` only. Do not use conda, `python -m venv`, or virtualenv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Required dependencies

```text
torch>=2.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
kagglehub>=0.3.0
```

For H100 runs, install the FlashAttention-4 package/version compatible with the selected PyTorch and CUDA stack. Keep its exact resolved version in the run manifest. Do not install FA4 on the Mac unless the smoke-test environment specifically needs it.

Do not add FAISS unless `use_ann=True` and the active environment supports it.


## STEP 1 — DATASET ACQUISITION FROM KAGGLE

The script itself must acquire MovieLens-20M. Do not require the user to manually download or copy the dataset.

### 1.1 Kaggle source

Use the official Kaggle API/tooling to retrieve the MovieLens-20M dataset from the configured dataset identifier. Store credentials outside the repository and never log them. The code must support an already-cached/local dataset so repeated runs do not redownload unnecessarily.

Configuration:

```python
kaggle_dataset: str = "grouplens/movielens-20m-dataset"
auto_download_dataset: bool = True
force_download_dataset: bool = False
```

### 1.2 Acquisition procedure

At startup:

1. Check whether the expected six raw files already exist under `config.raw_data_dir`.
2. If they exist and `force_download_dataset=False`, validate and reuse them.
3. Otherwise authenticate through the Kaggle API and download the configured dataset.
4. Extract the archive when necessary.
5. Locate the six expected CSV files and place/copy them under `data/raw/`.
6. Validate file presence, required columns, and approximate row counts.
7. Compute and save a dataset fingerprint/metadata manifest.

If Kaggle credentials are unavailable and the dataset is not already present, fail with a clear authentication message; never silently substitute another dataset.

### 1.3 Required files

| File | Expected rows (approx.) | Required columns |
|---|---:|---|
| `rating.csv` | ~20,000,263 | `userId,movieId,rating,timestamp` |
| `movie.csv` | ~27,279 | `movieId,title,genres` |
| `tag.csv` | ~465,565 | `userId,movieId,tag,timestamp` |
| `genome_scores.csv` | ~11,709,768 | `movieId,tagId,relevance` |
| `genome_tags.csv` | ~1,128 | `tagId,tag` |
| `link.csv` | ~27,279 | `movieId,imdbId,tmdbId` |

At startup, verify all six files and log actual row counts. Raise a clear error if a required file/column is missing.

**Note:** the `rating.csv` row in the table must be interpreted as exactly `userId,movieId,rating,timestamp`; do not create or expect a duplicate `userId` column.


## STEP 2 — PROJECT STRUCTURE

```
project_root/
│
├── data/
│   ├── raw/                        ← all 6 CSV files
│   ├── processed/                  ← preprocessing outputs
│   ├── preprocessing.py
│   ├── sequence_generator.py
│   └── augmentation.py             ← sequence augmentation strategies
│
├── models/
│   ├── transformer.py
│   ├── moe.py                      ← ARV-MMoE + conventional MMoE
│   ├── lora.py
│   └── ann_retrieval.py            ← ANN / brute-force retrieval stage
│
├── training/
│   ├── pretrain.py
│   ├── local_train.py
│   ├── federated.py
│   ├── losses.py                   ← all loss functions in one place
│   ├── gradnorm.py                 ← GradNorm controller
│   └── ablation_runner.py          ← runs ablation studies A–H
│
├── evaluation/
│   ├── metrics.py
│   ├── flops.py
│   ├── cold_start.py               ← cold-start evaluation protocol
│   ├── longtail.py                 ← long-tail evaluation
│   ├── routing_diagnostics.py      ← entropy, utilization, collapse tests
│   └── gradient_diagnostics.py     ← grad cosine sim, grad norms
│
├── configs/
│   └── config.py
│
├── outputs/
│   ├── checkpoints/
│   │   ├── t_small/
│   │   ├── t_base/
│   │   ├── t_large/
│   ├── logs/
│   ├── plots/
│   └── ablations/                  ← per-study (A–H) results
│
├── utils.py                        ← device helper, seed utils
├── requirements.txt
└── main.py
```

---

## STEP 3 — CONFIGURATION (`configs/config.py`)

`Config` is the single source of truth. Do not hardcode hyperparameters elsewhere.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Reproducibility
    seed: int = 42
    final_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Hardware / dtype
    device: str = "auto"                 # "auto" | "mps" | "cuda" | "cpu"
    dtype: str = "bf16"                  # the current specification: BF16 everywhere
    attention_backend: str = "auto"      # auto | flex | flashattention4

    # Data paths / Kaggle
    raw_data_dir: str = "data/raw/"
    processed_dir: str = "data/processed/"
    kaggle_dataset: str = "grouplens/movielens-20m-dataset"
    auto_download_dataset: bool = True
    force_download_dataset: bool = False
    ratings_file: str = "rating.csv"
    movies_file: str = "movie.csv"
    tags_file: str = "tag.csv"
    genome_scores_file: str = "genome_scores.csv"
    genome_tags_file: str = "genome_tags.csv"
    links_file: str = "link.csv"

    # Preprocessing / sequences
    min_interactions: int = 5
    implicit_threshold: float = 4.0
    max_seq_length: int = 50
    sequence_length_grid: List[int] = field(default_factory=lambda: [20, 50, 100])

    # Metadata integration
    use_metadata: bool = True

    # DataLoader / tensorisation
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Genome features
    genome_top_tags: int = 20

    # Cold-start / splits
    cold_start_ratio: float = 0.10
    train_ratio: float = 0.80
    val_ratio: float = 0.10

    # Model
    dropout: float = 0.1
    ffn_expansion: int = 4

    # Pretraining
    learning_rate: float = 1e-3
    pretrain_epochs: int = 10
    warmup_ratio: float = 0.05
    use_lr_scheduler: bool = True

    # Sampled softmax / negatives
    num_sampled_negatives: int = 256
    negative_pool_size: int = 100000
    negative_pool_seed_offset: int = 10000

    # Evaluation
    top_k: int = 10
    eval_negatives: int = 100
    eval_temperature: float = 1.0
    use_logit_calibration: bool = False
    use_ann: bool = False
    ann_candidates: int = 200

    # Masked item modeling
    use_masked_lm: bool = True
    mask_prob: float = 0.15
    masked_lm_weight: float = 0.2

    # Multi-step prediction
    multi_step_k: int = 3
    multi_step_weight: float = 0.5

    # Contrastive loss
    use_contrastive: bool = True
    contrastive_weight: float = 0.1
    contrastive_temp: float = 0.2
    contrastive_interval: int = 4
    aug_crop_ratio: float = 0.6
    aug_mask_ratio: float = 0.2

    # Embedding normalisation
    normalize_embeddings: bool = True

    # ARV-MMoE
    moe_variant: str = "arv_mmoe"
    num_experts: int = 4
    expert_ffn_expansion: int = 2
    objective_embedding_dim: int = 32
    num_objectives: int = 4
    match_mmoe_expert_count: bool = True

    # GradNorm
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025
    gradnorm_update_interval: int = 1
    gradnorm_anchor_layer: str = "last_shared_transformer"

    # Diagnostics
    routing_diag_interval_epochs: int = 1
    uniform_collapse_cv_threshold: float = 0.1
    single_expert_collapse_threshold: float = 0.8
    grad_diag_interval_steps: int = 200
    grad_diag_num_batches: int = 20

    # Ablations
    expert_count_grid: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    router_conditioning: str = "objective_conditioned"

    # LoRA / federated
    lora_rank: int = 8
    local_epochs: int = 3
    clients_per_round: int = 100
    federated_rounds: int = 20
    personalization_eval_users: int = 1000
    use_weighted_fedavg: bool = True

    # Long-tail
    longtail_head_pct: float = 0.20
    longtail_tail_pct: float = 0.60

    # Latency
    latency_warmup_runs: int = 20
    latency_timed_runs: int = 100

    # Outputs
    checkpoint_dir: str = "outputs/checkpoints/"
    log_dir: str = "outputs/logs/"
    plot_dir: str = "outputs/plots/"
    plot_format: str = "png"
    plot_dpi: int = 150
```

### Device and dtype helpers

```python
def resolve_dtype(config):
    if config.dtype != "bf16":
        raise ValueError("dtype must be 'bf16' everywhere")
    return torch.bfloat16

def get_device(config):
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    if config.device == "cpu" or config.device == "auto":
        return torch.device("cpu")
    raise RuntimeError(f"Requested device unavailable: {config.device}")
```

Do not call `.cuda()` or `.to("cuda")` directly.


## STEP 4 — DATA PREPROCESSING (`data/preprocessing.py`)

### 4.1 Load All 6 Files

```python
ratings_df  = pd.read_csv(raw_data_dir / "rating.csv")
movies_df   = pd.read_csv(raw_data_dir / "movie.csv")
tags_df     = pd.read_csv(raw_data_dir / "tag.csv")
genome_df   = pd.read_csv(raw_data_dir / "genome_scores.csv")
genome_tags = pd.read_csv(raw_data_dir / "genome_tags.csv")
links_df    = pd.read_csv(raw_data_dir / "link.csv")
```

### 4.2 Ratings Pipeline

**Canonical Item ID Indexing Scheme — enforced globally:**

| ID value | Meaning |
|---|---|
| `0` | Padding token (reserved; never a real item) |
| `1 .. num_items` | Real item IDs (contiguous integers) |
| `num_items + 1` | Mask token (BERT4Rec masking; a dedicated slot) |
| — | Embedding vocab size = `num_items + 2` |

This scheme must be applied during the ID remapping step and must be respected
everywhere: embedding tables, negative samplers, metadata lookups, and all loss
functions. Padding (0) and mask (num_items+1) are never valid targets or negatives.

Pipeline steps:
1. Validate columns `[userId, movieId, rating, timestamp]` exist.
2. Keep rows where `rating >= config.implicit_threshold`.
3. Remove users with fewer than `config.min_interactions` implicit interactions.
4. Sort per user by `timestamp` ascending.
5. Map `userId` → contiguous integers starting from **0**.
6. Map `movieId` → contiguous integers starting from **1** (NOT 0 — 0 is padding).
7. Store as `{user_id: [item_id, ...]}` in chronological order.
   All stored item IDs are in range `[1, num_items]`.
8. Derive constants and save to `dataset_stats.json`:
   ```python
   num_items   = len(item_mapping)          # number of real items
   PADDING_ID  = 0                          # constant
   MASK_ID     = num_items + 1              # constant
   VOCAB_SIZE  = num_items + 2              # embedding table size
   ```
9. Skip users with < 3 items after filtering; log total skipped count.

### 4.3 Cold-Start Item Holdout

Cold items must be genuinely unseen during model training. Do not select cold items merely by a quantile of first-appearance timestamps.

First determine the complete chronological train/validation/test split on the untruncated user histories. Define the **global training cutoff** as the latest timestamp among interactions assigned to the training split. Then:

```python
item_first_seen = ratings_df.groupby("movieId")["timestamp"].min()
cold_candidates = set(
    item_first_seen[item_first_seen > global_training_cutoff].index
)
```

Select up to `config.cold_start_ratio` of the eligible cold candidates deterministically, ordered by `(first_seen_timestamp, original_movie_id)`. If fewer candidates exist than the requested fraction, use the achievable count and report it; never manufacture cold items that appeared during training.

Remove cold-item interactions from the **training split only**. Preserve eligible cold interactions in validation/test. After remapping IDs, assert:

```python
assert cold_item_set.isdisjoint(train_interacted_items)
assert cold_item_set <= all_item_ids
assert cold_item_set | warm_item_set == all_item_ids
assert cold_item_set.isdisjoint(warm_item_set)
```

Save `cold_item_set` and `warm_item_set` to `data/processed/cold_items.pkl` and `data/processed/warm_items.pkl`. Log the requested cold ratio, eligible candidate count, selected count, global training cutoff, cold test interactions, and zero-metadata cold-item count.
### 4.4 Item Metadata Feature Extraction

Build a dense metadata matrix of shape `(num_items + 1, feature_dim)` where
`feature_dim = genome_top_tags + num_genres`. Row `0` is the all-zero padding row;
row `i` stores metadata for remapped real item ID `i` for `1 <= i <= num_items`.

**Genome tag features (from `genome_scores.csv`):**
- For each item, take the top `config.genome_top_tags = 20` tags by relevance.
- Build a dense real-item feature matrix and place it into rows `1..num_items` of
  the `(num_items + 1, genome_top_tags)` metadata array; row `0` remains zero.

**Genre features (from `movie.csv`):**
- Parse pipe-delimited `genres`. Treat `(no genres listed)` as a zero vector.
- One-hot encode genres into the real-item rows `1..num_items` of the metadata
  matrix; row `0` remains zero.

**Concatenate** horizontally → `item_metadata: np.ndarray (num_items + 1, feature_dim)`,
with row `0` reserved for padding and row `i` aligned directly with remapped item ID `i`.

**Cold-item handling:** Cold items have the same metadata extraction applied if genome
and genre data exist. If a cold item has no metadata at all, its row remains all zeros
(zero-metadata cold item). Log count of zero-metadata cold items.

Save to `data/processed/item_metadata.npy`.

### 4.5 Outputs

| File                    | Content                                              |
|-------------------------|------------------------------------------------------|
| `user_sequences.pkl`    | `{user_id: [item_ids]}` (warm items only in train)   |
| `item_mapping.pkl`      | `{original_movieId: new_item_id}`                    |
| `user_mapping.pkl`      | `{original_userId: new_user_id}`                     |
| `item_metadata.npy`     | `np.ndarray (num_items, feature_dim)`                |
| `cold_items.pkl`        | `set[int]` of cold item IDs                          |
| `warm_items.pkl`        | `set[int]` of warm item IDs                          |
| `dataset_stats.json`    | num_users, num_items, num_interactions, feature_dim, num_cold_items |

---

## STEP 5 — SEQUENCE AUGMENTATION (`data/augmentation.py`)

Implement three augmentation strategies. All operate on a 1D list of item IDs and
return an augmented copy. They are used by the contrastive loss and the masked item
modeling objective.

### 5.1 Random Crop

```python
def random_crop(seq: list, crop_ratio: float, rng: np.random.Generator) -> list:
    """
    Retain a contiguous subsequence of length >= ceil(len(seq) * crop_ratio).
    Choose start position uniformly at random.
    If seq has 1 item, return it unchanged.
    """
    min_len = max(1, math.ceil(len(seq) * crop_ratio))
    if len(seq) <= min_len:
        return seq[:]
    start = rng.integers(0, len(seq) - min_len + 1)
    return seq[start: start + min_len]
```

### 5.2 Random Item Masking

```python
def random_mask(seq: list, mask_ratio: float, mask_token: int,
                rng: np.random.Generator) -> list:
    """
    Replace each item with mask_token with probability mask_ratio.
    Always keep at least 1 unmasked item.
    mask_token = num_items + 1 (the dedicated mask slot; NOT num_items).
    """
    masked = seq[:]
    n_mask = max(1, round(len(seq) * mask_ratio))
    indices = rng.choice(len(seq), size=min(n_mask, len(seq) - 1), replace=False)
    for idx in indices:
        masked[idx] = mask_token
    return masked
```

### 5.3 Subsequence Sampling

```python
def subsequence_sample(seq: list, rng: np.random.Generator) -> list:
    """
    Sample a random contiguous prefix of seq (length 1..len(seq)-1).
    Ensures the augmented view is shorter than the original.
    """
    if len(seq) <= 1:
        return seq[:]
    end = rng.integers(1, len(seq))
    return seq[:end]
```

### 5.4 Augmentation Pipeline

```python
def augment_sequence(seq: list, config, rng: np.random.Generator,
                     mask_token: int) -> list:
    """
    Randomly apply one of the three augmentation strategies.
    Equal probability per strategy.
    """
    strategy = rng.integers(0, 3)
    if strategy == 0:
        return random_crop(seq, config.aug_crop_ratio, rng)
    elif strategy == 1:
        return random_mask(seq, config.aug_mask_ratio, mask_token, rng)
    else:
        return subsequence_sample(seq, rng)
```

The `mask_token` value is `num_items + 1` (the dedicated mask slot, never a real item).
The item embedding table must have vocab size `num_items + 2` to accommodate both
the padding token (0) and the mask token (num_items+1).

> **Rule:** Never pass `mask_token = num_items` — that would collide with the last
> real item ID (item IDs run from 1 to num_items inclusive).

---

## STEP 6 — SEQUENCE GENERATION, PRE-TENSORISATION, AND GLOBAL NEGATIVE POOL

### 6.1 Correct chronological ordering

For every user:

1. Filter to implicit interactions.
2. Sort the **complete** history chronologically.
3. Determine the user's chronological train/validation/test boundary on the complete history.
4. Apply the strict cold-item holdout to the training portion.
5. Generate sliding-window examples within each split.
6. Only for each individual example, truncate its preceding context to the most recent `config.max_seq_length` items.

Never truncate a user's complete history before determining the split.

For target position `j`:

```text
context = history[:j]
seq = context[-max_seq_length:]
target = history[j]
```

### 6.2 Left padding

All model inputs use fixed length `S=config.max_seq_length` and **left padding**. Padding ID is `0`. Real items occupy `1..num_items`. Mask ID is `num_items+1`.

For a left-padded tensor `(B,S)`, the final valid hidden state is at index `S-1`. Multi-step step `0` reads `S-1`, step `1` reads `S-2`, etc. Never use `seq_lens-1-step` under this left-padding convention.

### 6.3 Multi-step targets

For origin `j`,

```text
targets_k = [history[j+1], ..., history[j+k]]
```

Only targets inside the same split are valid. No target may cross a train/val/test boundary.

### 6.4 Separate history dictionaries

```python
train_user_history: dict[int, set[int]]
eval_user_history: dict[int, set[int]]
```

`train_user_history` contains training interactions only. `eval_user_history` contains the interactions that must be excluded when constructing evaluation candidates according to the declared evaluation protocol. They are separate objects and must be asserted unequal by identity at construction.

### 6.5 Pre-tensorised dataset

Construct tensors once. `__getitem__` must only index tensors.

```python
class RecDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, user_ids, lengths, targets_k=None, metadata=None):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.user_ids = torch.as_tensor(user_ids, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.targets_k = (torch.as_tensor(targets_k, dtype=torch.long)
                          if targets_k is not None else None)
        self.metadata = (torch.as_tensor(metadata, dtype=torch.bfloat16)
                         if metadata is not None else None)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        out = {
            "seq": self.sequences[idx],
            "target": self.targets[idx],
            "user_id": self.user_ids[idx],
            "length": self.lengths[idx],
        }
        if self.targets_k is not None:
            out["targets_k"] = self.targets_k[idx]
        if self.metadata is not None:
            out["metadata"] = self.metadata[idx]
        return out
```

Cache the tensorized arrays/files under `data/processed/`. Dataset initialization must load/cache them instead of rebuilding tensors each epoch.

### 6.6 DataLoader

```python
DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=shuffle,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    persistent_workers=(config.persistent_workers if config.num_workers > 0 else False),
)
```

Default `num_workers=0`. If `num_workers=0`, do not set worker initialization or persistent-worker options that require subprocesses.

### 6.7 Global shared negative pool

Generate one deterministic pool of **warm items only**. Cold items must never appear as training negatives. The pool is global/shared across users and models.

```python
class GlobalNegativePool:
    def __init__(self, warm_item_ids, pool_size, seed):
        self.warm_item_ids = np.asarray(sorted(warm_item_ids), dtype=np.int32)
        self.pool_size = pool_size
        self.seed = seed
        self.pool = None

    def build_once(self):
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self.warm_item_ids), size=self.pool_size)
        self.pool = self.warm_item_ids[idx]

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.pool), size=n)
        return self.pool[idx]
```

Save the pool to `data/processed/global_negative_pool.npy` and reuse it if its metadata fingerprint matches the current dataset/split/seed/config. Do not refresh it every epoch.

For each training example, filter the shared pool to remove the positive target, padding ID, mask ID, and all items forbidden by the training-history rule. If filtering removes duplicates or too many candidates, continue deterministic sampling from the same fixed global pool (wrapping through its fixed order as needed) until exactly `config.num_sampled_negatives` valid negatives are obtained; never substitute cold items or regenerate a new per-epoch pool. If the warm-item pool cannot provide enough valid negatives for an example, fail with a clear dataset/configuration error.

For evaluation, construct and save fixed candidate sets once per evaluation split using the same global pool plus the declared filtering/target-inclusion rules. Every compared model must consume the same candidate sets.


## STEP 7 — ALL LOSS FUNCTIONS (`training/losses.py`)

Place **all** loss implementations in this single file. Other modules import from here.

> **Implementation note:** Every loss below uses the same mathematical form (sampled
> softmax, masked-item cross-entropy, multi-step cross-entropy, InfoNCE) and the
> same masking, negative-sampling, and causal-position logic. The representation
> supplied to each loss is the objective-specific mixture `Z_{u,t}` produced by
> ARV-MMoE or conventional MMoE; when `moe_variant="none"`, the loss consumes
> the corresponding raw Transformer hidden representation. See Step 8.5 for the routing layer and Step 8.6 for
> how each loss's `t` maps to an objective embedding. The **loss weights**
> (`masked_lm_weight`, `multi_step_weight`, `contrastive_weight`) are no longer
> static config constants when `use_gradnorm=True` (default) — they are
> replaced by learned λₜ from the GradNorm controller in §7.5. Treat every
> `L_total = L_next + λ * L_x + ...` formula in this section as depicting the
> *terms*, not their final weighting.

### 7.1 Sampled Softmax Loss (with Tied Weights)

```python
class SampledSoftmaxLoss(nn.Module):
    def __init__(self, num_sampled: int):
        super().__init__()
        self.num_sampled = num_sampled

    def forward(self, user_emb, pos_ids, neg_ids, item_emb_weight):
        # user_emb:        (batch, hidden_dim)
        # pos_ids:         (batch,)
        # neg_ids:         (batch, K)
        # item_emb_weight: (vocab_size, hidden_dim) — tied from model
        pos_emb  = item_emb_weight[pos_ids]                           # (B, D)
        neg_emb  = item_emb_weight[neg_ids]                           # (B, K, D)
        pos_score = (user_emb * pos_emb).sum(dim=-1, keepdim=True)   # (B, 1)
        neg_score = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # (B, K)
        logits = torch.cat([pos_score, neg_score], dim=1)             # (B, K+1)
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)
        return nn.functional.cross_entropy(logits, labels)
```

### 7.2 Masked Item Modeling Loss

```python
class MaskedItemLoss(nn.Module):
    """
    BERT4Rec-style masked language model loss.
    Applied to positions where the input was replaced with mask_token.
    """
    def forward(self, hidden_states, mask_positions, true_item_ids,
                item_emb_weight):
        # hidden_states: (batch, seq_len, hidden_dim)
        # mask_positions: (batch, seq_len) bool tensor — True where masked
        # true_item_ids:  (batch, seq_len) — original item IDs before masking
        masked_hidden = hidden_states[mask_positions]    # (M, hidden_dim)
        masked_labels = true_item_ids[mask_positions]    # (M,)
        if masked_hidden.size(0) == 0:
            return torch.tensor(0.0, device=hidden_states.device)
        logits = masked_hidden @ item_emb_weight.T       # (M, vocab_size)
        return nn.functional.cross_entropy(logits, masked_labels)
```

**Integration in training loop:**

```python
L_total = L_next_item + config.masked_lm_weight * L_masked
```

For masked item modeling:
1. Before the forward pass, randomly mask `config.mask_prob = 15%` of non-padding
   items in the input sequence, replacing them with `MASK_ID = num_items + 1`.
   **Never use `num_items` as the mask token** — that is a real item ID.
2. Store the original item IDs as `true_item_ids` for the loss.
3. Pass the masked input to the transformer; collect all hidden states
   (not just the last position).
4. Compute `MaskedItemLoss` on all masked positions.

### 7.3 Multi-Step Prediction Loss

**Note:** Using `last_hidden` (position `t`) for all k prediction steps leaks
future context for steps k>1. The hidden state at position `t` has already attended
(via causal self-attention) to items near position `t`, so predicting item at `t+2`
from `h_t` is a shortcut — the model can exploit patterns from positions it shouldn't
have access to for that step.

**Fix:** For step `k` (predicting item at `t+k`), use the hidden state from position
`t−k+1`. This position has causal access only to items `[1 .. t−k+1]`, enforcing
proper information boundaries.

```python
class MultiStepLoss(nn.Module):
    """
    Predict next k items using hidden states at causal positions.
    For step k (predicting item t+k): use hidden state at position t-k+1.
    This prevents shortcut learning from future context in last_hidden.
    """
    def __init__(self, k: int, hidden_dim: int):
        super().__init__()
        self.k = k
        self.step_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(k)]
        )

    def forward(self, all_hidden_states, targets_k, item_emb_weight):
        """
        all_hidden_states: (batch, S, hidden_dim) — full left-padded hidden sequence
        targets_k:         (batch, k) — target IDs at t+1..t+k (0 = missing/pad)
        item_emb_weight:   (vocab_size, hidden_dim) — tied embedding weights
        """
        total_loss = torch.tensor(0.0, device=all_hidden_states.device,
                                  dtype=all_hidden_states.dtype)
        count = 0
        S = all_hidden_states.size(1)
        for step in range(self.k):
            step_targets = targets_k[:, step]
            valid = step_targets != 0
            if valid.sum() == 0:
                continue
            if step >= S:
                break

            # With fixed-length LEFT padding, the causal position is shared across
            # samples: step 0 -> S-1, step 1 -> S-2, ... .
            # Never derive this index from seq_lens.
            hidden_pos = S - 1 - step
            step_hidden = all_hidden_states[:, hidden_pos, :]   # (B, D)

            valid_hidden  = step_hidden[valid]
            valid_targets = step_targets[valid]
            proj   = self.step_projections[step](valid_hidden)
            logits = proj @ item_emb_weight.T
            total_loss += nn.functional.cross_entropy(logits, valid_targets)
            count += 1

        return total_loss / max(count, 1)
```

**Required calling convention:** `MultiStepLoss.forward` requires `all_hidden_states`
(full hidden sequence, not just the last token) and `targets_k`. Under the fixed-length
LEFT-padding convention, step 0 reads `S-1`, step 1 reads `S-2`, etc.; do not use
`seq_lens-1-step` for position selection. The transformer's `return_all_hidden=True`
flag must be active whenever `multi_step_k > 0`.

**Integration:**

```python
L_total = L_next_item + config.masked_lm_weight * L_masked \
          + config.multi_step_weight * L_multistep
```

Only include `L_multistep` when `config.multi_step_k > 0`.

### 7.4 Contrastive Loss (InfoNCE)

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_orig, z_aug):
        # z_orig, z_aug: (batch, hidden_dim) — L2 normalised sequence embeddings
        # In-batch negatives: each sample's augmented view is negative for all others
        batch_size = z_orig.size(0)
        z = torch.cat([z_orig, z_aug], dim=0)           # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature       # (2B, 2B)
        # Mask self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])
        return nn.functional.cross_entropy(sim, labels)
```

**Contrastive Loss Computation Frequency:**

InfoNCE requires a second full forward pass per batch, approximately doubling training
time. To keep training practical on MPS with 10-epoch runs on MovieLens-20M, compute
the contrastive loss only every `config.contrastive_interval = 4` batches. Skip the
second forward pass on all other batches.

```python
# In training loop: all floating-point model computation uses BF16.
if use_contrastive and (batch_idx % config.contrastive_interval == 0):
    aug_seq = augment_sequence(original_seq, config, rng, mask_token=MASK_ID)
    z_aug = model(aug_seq_tensor)
    z_aug = nn.functional.normalize(z_aug, p=2, dim=-1)
    L_contrastive = infonce_loss(z_orig_normalised, z_aug)
else:
    L_contrastive = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
```

Setting `contrastive_interval = 1` computes on every batch (maximum accuracy).
Setting `contrastive_interval = 0` is invalid — use `use_contrastive = False` to
disable instead.

**Integration:**

```python
L_total = L_next_item + config.masked_lm_weight * L_masked \
          + config.multi_step_weight * L_multistep \
          + config.contrastive_weight * L_contrastive
```

To compute `L_contrastive`:
1. Pass the **original** (unmasked) input sequence through the transformer; take the
   last-position embedding as `z_orig`. L2-normalise → `z_orig_normalised`.
2. Generate an augmented sequence using `augment_sequence()` (only on interval batches).
3. Pass augmented sequence through the same transformer (shared weights); take
   last-position embedding as `z_aug`. L2-normalise.
4. Apply `InfoNCELoss(z_orig_normalised, z_aug)`.

### 7.5 GradNorm — Dynamic Loss Balancing (`training/gradnorm.py`)

The GradNorm anchor must be a parameter set that exists in every model variant and is
shared by all objectives. Resolve `last_shared_transformer` to the parameters of the
final Transformer block immediately before the optional MoE layer. Do not use an
expert-only parameter as the GradNorm anchor because `moe_variant="none"` has no experts.

GradNorm is applied only to **active objectives**. The fixed objective order is:

```python
OBJECTIVE_ORDER = [
    "next_item",
    "masked_item",
    "multi_step",
    "contrastive",
]
```

For the active objective set `A`, maintain `lambda_t = exp(log_lambda_t)`. At the first valid optimization step, store `L_t(0) = detach(L_t)` for each active objective. For every update:

```text
G_t = || grad_W(lambda_t * L_t) ||_2
Lhat_t = L_t(t) / L_t(0)
r_t = Lhat_t / mean_active(Lhat)
Gbar = mean_active(G_t)
G*_t = Gbar * r_t ** alpha
L_GN = sum_active |G_t - detach(G*_t)|
```

`W` is one fixed shared Transformer anchor layer present in every model variant, before
the optional MoE routing layer. Use `last_shared_transformer` by default so GradNorm is
well-defined for both `moe_variant="none"` and the MoE variants. `log_lambdas` are
optimized with a separate optimizer. After every GradNorm update, renormalize only the active lambdas so:

```text
sum(lambda_t for t in active_objectives) == number_of_active_objectives
```

Inactive objectives have effective weight zero and are excluded entirely from GradNorm. Do **not** pass zero-valued placeholder losses to `autograd.grad`. Contrastive is inactive on batches where its configured interval does not fire.

```python
class GradNormController:
    # keep log_lambdas for the fixed objective order, but operate only on active indices
    # create a separate optimizer over log_lambdas
    # store initial losses only for active objectives
    # compute gradients only for active losses
    # renormalize only active lambda values after each update
    ...
```

Training-loop contract:

```python
active_losses = {
    name: loss
    for name, loss in objective_losses.items()
    if objective_is_active(name, config, batch=batch)
}

L_total = (
    gradnorm.weighted_sum(active_losses)
    if config.use_gradnorm
    else static_weighted_sum(active_losses, config)
)

optimizer.zero_grad(set_to_none=True)
L_total.backward(retain_graph=config.use_gradnorm)
if config.use_gradnorm and step % config.gradnorm_update_interval == 0:
    gradnorm.update(
        active_losses,
        shared_layer_params=get_anchor_layer_params(model, config),
    )
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
if config.use_lr_scheduler:
    scheduler.step()
```

Every update must assert finite losses, finite GradNorm loss, finite active lambdas, and the active-lambda sum constraint.

## STEP 8 — TRANSFORMER BACKBONE (`models/transformer.py`)

### 8.1 Architecture

**Item embedding:** `(num_items + 2, hidden_dim)` with row 0 padding and row `num_items+1` mask token.

**Positional embedding:** learnable `nn.Embedding(config.max_seq_length, hidden_dim)`.

**Transformer layer:** Pre-LN → self-attention → residual/dropout → Pre-LN → FFN (4×) → residual/dropout.

### 8.2 Optimized attention backends

The model must use a dedicated attention wrapper. The mathematical attention operation remains standard causal self-attention; only the backend changes by hardware.

- On **MPS**, use **FlexAttention**.
- On **CUDA/H100**, use **FlashAttention-4**.
- `attention_backend="auto"` resolves to the correct hardware-specific backend.
- The selected backend must pass a tiny startup self-test before training.
- The final H100 experiment must fail fast if FlashAttention-4 cannot be initialized; do not silently substitute ordinary attention.
- The MPS smoke test must fail fast if the configured FlexAttention path cannot execute; do not rewrite the experiment to accommodate MPS limitations.
- Record PyTorch version, device, dtype, and resolved attention backend.

The implementation may use the official/native PyTorch FlexAttention integration for MPS and the compatible FlashAttention-4 integration/package for H100. Keep the serving/training tensor shapes and masks compatible with the selected backend.

### 8.3 Masks

Use the causal mask and padding validity consistently. Because all sequences are fixed-length and left-padded, ensure padded keys cannot contribute and future positions cannot contribute. Implement the mask in the form required by the selected optimized backend rather than assuming every backend accepts the same `nn.MultiheadAttention` arguments.

### 8.4 Hidden states

`forward(..., return_all_hidden=True)` returns `(B,S,D)` for MIM/multi-step/MoE routing. `return_all_hidden=False` returns the pooled final valid representation.

### 8.5 Model sizes

| Variant | Layers | Hidden | Heads |
|---|---:|---:|---:|
| T-Small | 3 | 256 | 4 |
| T-Base | 4 | 320 | 4 |
| T-Large | 5 | 384 | 6 |

Validate `hidden_dim % heads == 0`.

### 8.6 BF16

All model computation uses `torch.bfloat16`. Do not wrap the model in a generic FP16 `autocast` policy. Do not add a GradScaler.


## STEP 8.5 — ADAPTIVE ROUTING VECTOR MIXTURE-OF-EXPERTS: ARV-MMoE (`models/moe.py`)

This is the central architectural contribution. It replaces the single shared
Transformer representation with **objective-conditioned expert mixing**. It is
critical to distinguish the **structure** (MMoE: shared experts, a gate that
mixes them) from the **routing formulation** (ARV: a single shared,
objective-conditioned router replacing independent per-objective gates).

### 8.5.1 Shared Experts

Build `K = config.num_experts` identical-shaped feed-forward experts, applied
**position-wise** to every position of the shared hidden sequence. The same
`K` expert outputs are computed **once** and reused by every objective — only
the mixture (routing) weights differ per objective.

```python
class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(self, h):
        # h: (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        return self.net(h)


class SharedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, expansion: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, expansion, dropout) for _ in range(num_experts)]
        )

    def forward(self, h):
        # Returns (batch, seq_len, K, hidden_dim) -- every expert's output at every position
        return torch.stack([e(h) for e in self.experts], dim=2)
```

### 8.5.2 Objective Embeddings

One learnable embedding vector per training objective, in a **fixed order**
that must be respected everywhere (dataset, losses, diagnostics, plots):

```python
OBJECTIVE_ORDER = ["next_item", "masked_item", "multi_step", "contrastive"]

self.objective_embeddings = nn.Embedding(
    num_embeddings=config.num_objectives,      # 4
    embedding_dim=config.objective_embedding_dim,
)
```

`objective_embeddings.weight[0]` is `e_next`, `[1]` is `e_MIM`, `[2]` is
`e_forecast`, `[3]` is `e_contrast`. These indices are the ONLY place objective
identity enters the routing computation — the router itself never sees a
one-hot "which objective" flag, only the embedding.

### 8.5.3 ARV-MMoE Router (Proposed)

```python
class ARVRouter(nn.Module):
    """
    A single shared router, parameterised by (W_r, b_r), conditioned on the
    user's pooled sequence representation H_u and an objective embedding e_t.
    Eq. (3)-(4) in the paper.
    """
    def __init__(self, hidden_dim: int, emb_dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(hidden_dim + emb_dim, num_experts)

    def forward(self, pooled_hidden, objective_emb):
        # pooled_hidden: (batch, hidden_dim)      -- H_u (last non-pad position)
        # objective_emb: (batch, emb_dim) or (emb_dim,) broadcastable -- e_t
        if objective_emb.dim() == 1:
            objective_emb = objective_emb.unsqueeze(0).expand(pooled_hidden.size(0), -1)
        router_input = torch.cat([pooled_hidden, objective_emb], dim=-1)
        return torch.softmax(self.router(router_input), dim=-1)   # r_{u,t}: (batch, K)
```

**Design decision (engineering completion of the paper's formulation):** the
paper defines routing over the pooled user representation `H_u`. Two of the
four objectives (MIM, multi-step) require **per-position** outputs, not just a
pooled vector. The implementation computes the routing distribution
`r_{u,t}` **once per sample from the pooled `H_u`** (the same `H_u` used for
next-item prediction), then **broadcasting that single distribution across all
positions** when mixing the position-wise expert outputs:

```python
def route_objective(pooled_hidden, all_expert_outputs, objective_emb, router):
    # pooled_hidden:      (batch, hidden_dim)
    # all_expert_outputs: (batch, seq_len, K, hidden_dim)
    # objective_emb:      (emb_dim,)
    r_ut = router(pooled_hidden, objective_emb)                      # (batch, K)
    r_ut_broadcast = r_ut.unsqueeze(1).unsqueeze(-1)                 # (batch, 1, K, 1)
    Z_ut = (r_ut_broadcast * all_expert_outputs).sum(dim=2)          # (batch, seq_len, hidden_dim)
    return Z_ut, r_ut
```

This keeps routing genuinely user-adaptive (different `H_u` for different
users yields different `r_{u,t}`, per §3.3.1 of the paper: `H_u != H_v =>
r_{u,t} != r_{v,t}` in general) while giving every objective a full
sequence-length representation to slice from:

- **Next-item head** reads `Z_{u,next}[:, last_pos, :]`.
- **MIM head** reads `Z_{u,mim}` at all masked positions.
- **Multi-step head** reads `Z_{u,forecast}` at the causal positions `t-k+1`
  (according to the causal-position rule in Step 7.3).
- **Contrastive head** reads `Z_{u,contrast}[:, last_pos, :]`, L2-normalised.

No persistent, client-specific personalization vector is introduced here —
routing is a pure function of `(H_u, e_t)`, recomputed on every forward pass.
Federated personalization remains exclusively the job of LoRA (Step 12/19).

### 8.5.4 Conventional MMoE Baseline (for Ablation C)

Each objective gets its **own independently parameterised gate** — no
parameter sharing across objectives' routing policies:

```python
class ConventionalMMoEGates(nn.Module):
    """
    Eq. (1)-(2) in the paper. One independent (W_t, b_t) gate per objective.
    """
    def __init__(self, hidden_dim: int, num_experts: int, num_objectives: int):
        super().__init__()
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_dim, num_experts) for _ in range(num_objectives)]
        )

    def forward(self, pooled_hidden, objective_idx: int):
        return torch.softmax(self.gates[objective_idx](pooled_hidden), dim=-1)
```

Used identically to `route_objective` above but calling
`gates(pooled_hidden, objective_idx)` instead of `router(pooled_hidden, e_t)`
— note the conventional gate never sees an objective embedding at all; its
task-specificity comes entirely from having separate weights per objective.

### 8.5.5 Parameter Accounting (required for Ablation C)

```python
def count_router_parameters(config) -> dict:
    d, demb, K, T = (config.hidden_dim, config.objective_embedding_dim,
                     config.num_experts, config.num_objectives)
    mmoe_gate_params = T * (K * d + K)
    arv_router_params = K * (d + demb) + K
    arv_objective_emb_params = T * demb
    arv_total = arv_router_params + arv_objective_emb_params
    return {
        "mmoe_gate_params": mmoe_gate_params,
        "arv_router_params": arv_router_params,
        "arv_objective_embedding_params": arv_objective_emb_params,
        "arv_total_router_params": arv_total,
        "difference": arv_total - mmoe_gate_params,
    }
```

Log this dict, plus total trainable parameters and expert parameters, to
`outputs/logs/moe_parameter_report_{model}.json` for every `moe_variant` run.
When `config.match_mmoe_expert_count = True` (default), `K` is identical
across the `mmoe` and `arv_mmoe` runs so any accuracy delta is attributable to
the routing formulation, not to expert capacity. If exact total-parameter
matching is not achievable, this report — not an assumption — is what Ablation
C cites.

### 8.5.6 Top-Level MoE Module

```python
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.variant = config.moe_variant       # "none" | "mmoe" | "arv_mmoe"
        if self.variant == "none":
            return   # heads will read hidden_states directly; nothing to build
        self.experts = SharedExperts(config.num_experts, config.hidden_dim,
                                      config.expert_ffn_expansion, config.dropout)
        if self.variant == "arv_mmoe":
            self.objective_embeddings = nn.Embedding(config.num_objectives,
                                                       config.objective_embedding_dim)
            self.router = ARVRouter(config.hidden_dim, config.objective_embedding_dim,
                                     config.num_experts)
        elif self.variant == "mmoe":
            self.gates = ConventionalMMoEGates(config.hidden_dim, config.num_experts,
                                                config.num_objectives)

    def forward(self, all_hidden_states, pooled_hidden, objective_idx: int):
        """
        Returns (Z_u_t, routing_weights) for the requested objective.
        objective_idx follows OBJECTIVE_ORDER (0=next,1=MIM,2=multistep,3=contrastive).
        """
        if self.variant == "none":
            return all_hidden_states, None
        expert_outputs = self.experts(all_hidden_states)          # (B, S, K, D)
        if self.variant == "arv_mmoe":
            e_t = self.objective_embeddings.weight[objective_idx]
            return route_objective(pooled_hidden, expert_outputs, e_t, self.router)
        else:  # "mmoe"
            r_ut = self.gates(pooled_hidden, objective_idx)
            r_ut_b = r_ut.unsqueeze(1).unsqueeze(-1)
            Z_ut = (r_ut_b * expert_outputs).sum(dim=2)
            return Z_ut, r_ut
```

**Ablation B — router conditioning.** `config.router_conditioning` toggles
whether the ARV router receives `[H_u ; e_t]` (`"objective_conditioned"`,
default/proposed) or `H_u` alone (`"input_only"`) — the latter is the same
architecture minus the objective embedding, isolating whether conditioning on
the objective provides a measurable benefit over a purely input-adaptive
router.

---

## STEP 8.6 — INFERENCE-TIME OBJECTIVE EMBEDDING & SERVING PATH

At training time, all four routing paths (`r_next`, `r_MIM`, `r_forecast`,
`r_contrast`) are computed on every batch that uses their respective
objective. At serving time, **no training objective is active** — the system
performs a single next-item ranking pass. Per Eq. (5)-(7) of the paper:

```python
erec = objective_embeddings.weight[OBJECTIVE_ORDER.index("next_item")]  # erec == e_next, by identity
r_rec = router(pooled_hidden, erec)
Z_rec = (r_rec.unsqueeze(1).unsqueeze(-1) * expert_outputs).sum(dim=2)
scores = Z_rec[:, -1, :] @ item_embedding.weight.T     # tied projection, as in Step 8
```

**This is not a separate, untrained embedding.** Because `e_next` already
receives gradients through the next-item loss during training, the serving
routing policy is *exactly* the policy learned for the next-item objective —
there is no cold, uninitialised inference-only parameter. Do not create a
second "erec" embedding; always index `objective_embeddings.weight[0]`
(`OBJECTIVE_ORDER[0] == "next_item"`).

**Full serving dataflow:**

```
User sequence -> embedding + Transformer -> H_u (pooled) & all_hidden_states
             -> ARV routing: r_rec = softmax(W_r[H_u; e_next] + b_r)
             -> expert mixture: Z_rec = sum_k r_rec[k] * E_k(all_hidden_states)
             -> ANN candidate generation (Step 18.3)
             -> calibrated scoring using Z_rec[:, last_pos, :] (Step 9)
             -> ranked top-N
```

When `config.moe_variant = "none"`, this step is skipped entirely and serving
uses the raw Transformer hidden representation directly.


## STEP 9 — LOGIT CALIBRATION / PROBABILITY REPORTING (`evaluation/metrics.py`)

Temperature scaling is optional and is **not** a ranking correction. For a fixed candidate set, dividing logits by any positive scalar and applying log-softmax preserves their order.

Defaults:

```python
use_logit_calibration = False
eval_temperature = 1.0
```

If temperature is learned, fit it on validation data only and freeze it before test evaluation. Never fit temperature on test data. Ranking metrics must be computed from the model ranking under the declared candidate protocol; calibration may be reported separately as probability/log-probability information.


## STEP 10 — LoRA IMPLEMENTATION (`models/lora.py`)

### 10.1 LoRALinear

```python
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int):
        super().__init__()
        d_in  = original_linear.in_features
        d_out = original_linear.out_features
        self.weight = original_linear.weight   # frozen
        self.bias   = original_linear.bias
        self.A = nn.Parameter(torch.randn(d_in,  rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, d_out))

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) \
               + (x @ self.A @ self.B)
```

### 10.2 Injection and Utilities

```python
def inject_lora(model, rank: int)           # replace Q and V in all attn layers
def get_lora_state_dict(model) -> dict      # LoRA params only
def load_lora_state_dict(model, sd: dict)   # restore LoRA params
def count_lora_parameters(model) -> int     # trainable count
```

After injection: freeze all params, then unfreeze only LoRA A and B matrices.

---

## STEP 11 — BASELINE SET

The implementation contains exactly these primary model families:

1. **Transformer sequential baseline** (`moe_variant="none"`)
2. **Conventional MMoE** (`moe_variant="mmoe"`)
3. **ARV-MMoE** (`moe_variant="arv_mmoe"`)


## STEP 12 — EMBEDDING NORMALISATION

Apply L2 normalisation to **both** user and item embeddings before any dot-product
scoring, across all models:

```python
if config.normalize_embeddings:
    user_emb  = nn.functional.normalize(user_emb,  p=2, dim=-1)
    item_emb  = nn.functional.normalize(item_emb,  p=2, dim=-1)
```

**Where to apply:**
- Transformer: normalise `last_hidden` (user repr) and item embedding rows before
  scoring during both training and evaluation.

**Why this is necessary:**
Sampled softmax + tied output weights cause embedding norms to grow unbounded during
training. L2 normalisation bounds the dot-product scores to [-1, 1] and dramatically
stabilises sampled-softmax training and keeps dot-product score scale bounded.

---

## STEP 13 — METADATA INTEGRATION (`models/transformer.py`)

Item metadata is optional but enabled by default and is part of the declared model
architecture. The implementation must expose `config.use_metadata` and must support
turning metadata integration off without changing the item IDs, sequence split, or
candidate protocol.

### 13.1 Metadata Projection

Use one shared linear projection from the per-position metadata feature vector into
the Transformer hidden dimension:

```python
self.metadata_projection = (
    nn.Linear(feature_dim, hidden_dim, bias=False)
    if config.use_metadata else None
)
```

For input metadata of shape `(B, S, feature_dim)`, compute:

```python
metadata_hidden = self.metadata_projection(metadata.to(dtype=torch.bfloat16))
x = item_embeddings + positional_embeddings + metadata_hidden
```

Metadata at padding positions is already zeroed during tensorisation. A genuinely
zero-metadata cold item therefore contributes a zero metadata vector rather than a
learned missing-item signal. When `use_metadata=False`, omit the projection from the
forward computation and use only `item_embeddings + positional_embeddings`.

### 13.2 Forward Interface

```python
def forward(self, seq, metadata=None, return_all_hidden=False):
    # seq:      (B, S) integer item IDs
    # metadata: (B, S, feature_dim) BF16 tensor when use_metadata=True
    ...
```

When `use_metadata=True`, require a metadata tensor and validate its final dimension
against `feature_dim`. When `use_metadata=False`, metadata may be omitted or supplied
but must not affect the computation. The same metadata integration is used during
training, validation, personalization, federated simulation, and evaluation unless
a declared ablation explicitly sets `use_metadata=False`.

### 13.3 Metadata Ablation

Ablation H1 compares `use_metadata=True` versus `False` with the same checkpoint
protocol, split, candidates, seed, and remaining configuration fixed.


## STEP 14 — LR SCHEDULER (`training/pretrain.py`)

Apply a linear warmup followed by cosine decay for every trainable model configuration.

```python
def build_scheduler(optimizer, config, total_steps: int):
    """
    Linear warmup for first `warmup_ratio` fraction of steps,
    then cosine decay to 0.
    """
    warmup_steps = int(total_steps * config.warmup_ratio)  # default: 5%

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / \
                   max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

`total_steps = num_batches_per_epoch * config.pretrain_epochs`

Call `scheduler.step()` **after every batch** (not after every epoch).
Log current LR at the end of each epoch.

---

## STEP 15 — CENTRAL PRETRAINING (`training/pretrain.py`)

### 15.1 Independent Per-Model Training

Each model is trained independently via `--model` CLI flag. No cross-model state.

### 15.2 Per-Model Checkpoint Directories

| Model   | Directory                        |
|---------|----------------------------------|
| T-Small | `outputs/checkpoints/t_small/`  |
| T-Base  | `outputs/checkpoints/t_base/`   |
| T-Large | `outputs/checkpoints/t_large/`  |

Per epoch: `outputs/checkpoints/{model}/epoch_{N:03d}.pt`  
Best: `outputs/checkpoints/{model}/best.pt`

### 15.3 Checkpoint Format

```python
torch.save({
    'epoch':                epoch,
    'model_state_dict':     model.state_dict(),      # includes MoE + GradNorm log_lambdas
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'gradnorm_state_dict':  gradnorm.state_dict() if config.use_gradnorm else None,
    'val_ndcg':             val_ndcg,
    'val_mrr':              val_mrr,
    'val_recall':           val_recall,
    'val_precision':        val_precision,
    'config':               asdict(config)
}, checkpoint_path)
```

### 15.3.1 Best-Checkpoint Selection

`best.pt` is selected **only by validation NDCG@10**. Update `best.pt` when
`val_ndcg` strictly improves over the best value seen so far. Do not use MRR,
Recall, Precision, training loss, routing diagnostics, or any test metric for
checkpoint selection. The full test set remains untouched until the model
configuration and best checkpoint are fixed.

Store `best_epoch`, `best_val_ndcg`, `best_val_mrr`, `best_val_recall`,
`best_val_precision`, `seed`, `model_name`, `moe_variant`, `use_gradnorm`, and
`num_experts` in the checkpoint metadata/manifest.

### 15.4 Training Loop Requirements

- Never refresh the global negative pool per epoch; load the prebuilt shared pool and reuse it.
- Call `scheduler.step()` after every batch.
- Log: epoch, train loss, val NDCG@10, MRR@10, Recall@10, Precision@10,
  current LR, and current GradNorm λ values.
- Log to `outputs/logs/pretrain_{model_name}.log`.
- For transformers with `use_masked_lm=True`: apply masking before forward pass,
  collect all hidden states, compute `MaskedItemLoss`.
- For transformers with `use_contrastive=True`: generate augmented view, run second
  forward pass, compute `InfoNCELoss`.
- Combine all active losses per §7 formulas.

### 15.4.1 Forward Path (ARV-MMoE + GradNorm)

When `config.moe_variant != "none"`, the per-batch forward/loss path is:

```python
all_hidden = model.transformer(seq, metadata=metadata, return_all_hidden=True)  # (B,S,D)
pooled_hidden = extract_last_nonpad(all_hidden, seq_lens)                       # (B,D)

active_losses = {}
routing_snapshots = {}
for objective_idx, name in enumerate(OBJECTIVE_ORDER):
    if not objective_enabled(name, config, batch=batch):
        continue
    Z_ut, r_ut = model.moe(all_hidden, pooled_hidden, objective_idx)
    active_losses[name] = compute_objective_loss(name, Z_ut, batch, model, config)
    routing_snapshots[name] = r_ut
    log_routing_snapshot(name, r_ut)   # feeds Step 21.5 diagnostics buffer

L_total = (gradnorm.weighted_losses(active_losses) if config.use_gradnorm
           else static_weighted_sum(active_losses, config))
```

Only enabled objectives (`use_masked_lm`, `multi_step_k > 0`, `use_contrastive`,
and the interval gating from §7.4 for contrastive) contribute a non-zero loss;
next-item prediction is always enabled. Every routing distribution actually computed for an active objective is appended
to a per-epoch buffer consumed by the routing diagnostics in Step 21.5 — do not
discard it after the loss is computed.

### 15.5 BF16 training

```python
optimizer.zero_grad(set_to_none=True)
all_hidden, pooled_hidden = model.encode(seq, metadata=metadata_batch)
losses, routing_snapshots = model.compute_all_objective_losses(
    all_hidden, pooled_hidden, batch, config
)
L_total = (
    gradnorm.weighted_losses(losses)
    if config.use_gradnorm
    else static_weighted_sum(losses, config)
)
L_total.backward(retain_graph=config.use_gradnorm)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

All model computation is BF16. Do not use FP16 autocast or GradScaler. Gradient/finite-value assertions must be enabled. Do not silently change dtype if a numerical issue occurs.


## STEP 16 — LOCAL PERSONALIZATION (`training/local_train.py`)

### 16.1 Scope

Sample `config.personalization_eval_users = 1000` users randomly (seeded).

### 16.2 Function Signature

```python
def run_local_personalization(
    config: Config,
    backbone_path: str,
    train_user_history: dict,    # train items only — for training negatives
    eval_user_history: dict,     # all items — for eval negatives
    train_dataset: RecDataset,
    test_dataset: RecDataset,
    num_items: int,
    cold_item_set: set,          # NEW: for cold-start metrics
):
```

### 16.3 Per-User Procedure

1. Load backbone from `backbone_path`.
2. Inject LoRA.
3. Train LoRA for `local_epochs = 3` using `train_user_history` for negatives.
4. Evaluate on test split with `eval_user_history` for negatives.
5. Also evaluate backbone without LoRA on same test split.
6. Record per-user: NDCG@10 with LoRA, NDCG@10 without LoRA, gain.
7. Also record whether the test item is cold or warm (for cold-start split).

### 16.4 Global vs Personalised Comparison Report

After all users are evaluated, compute and save to
`outputs/logs/personalization_gain_{model}.json`:

```json
{
  "num_users_evaluated": 987,
  "num_users_skipped": 13,
  "with_lora":    {"ndcg_mean": 0.43, "ndcg_std": 0.11, "hr_mean": 0.55},
  "without_lora": {"ndcg_mean": 0.38, "ndcg_std": 0.10, "hr_mean": 0.49},
  "gain":         {"ndcg_mean": 0.05, "ndcg_std": 0.04},
  "per_user_gains": [0.02, 0.08, -0.01, ...]  // all 987 values for histogram
}
```

The `per_user_gains` array is required for plotting the gain histogram.

Also generate and save **two plots** to `outputs/plots/`:

1. `personalization_gain_histogram_{model}.png` — histogram of per-user NDCG gains.
   - X-axis: NDCG gain (can be negative for users hurt by personalisation).
   - Y-axis: number of users.
   - Include vertical dashed line at x=0 and at the mean.

2. `global_vs_personalized_{model}.png` — scatter plot.
   - X-axis: NDCG without LoRA (per user).
   - Y-axis: NDCG with LoRA (per user).
   - Include y=x diagonal line (no gain line).
   - Points above the line = personalisation helped.

### 16.5 Local Personalization Training Constraints (MANDATORY)

Every non-federated personalization user starts from the **same frozen global checkpoint** and receives a **fresh, newly initialized local LoRA adapter**. Never reuse one user's LoRA state for another user. This is distinct from the federated experiment, where the aggregated global LoRA state intentionally persists across rounds.

The frozen global checkpoint checksum must be identical for every personalization user before local training.


To ensure correct **LoRA-based federated personalization**, the following rules MUST be enforced during **all local client training rounds**:

**Global / Local Parameter Split (per Step 3.4.2 of the paper)**

| Category | Components |
|---|---|
| **Global / Shared** (frozen during local training) | Transformer backbone, item/position embeddings, `SharedExperts` (Ek), ARV router `(Wr, br)` or conventional MMoE gates, objective embeddings `(et)`, shared task heads, GradNorm λ values |
| **Local / Client-specific** (trainable during local training) | LoRA adapters only |

**Parameter Freezing**
- All **global parameters** — transformer layers, embeddings, LayerNorm, FFN,
  output projection, **and the shared experts, the ARV router
  `(Wr, br)` (or conventional MMoE gates), and the objective embeddings
  `(et)`** — MUST be frozen.
- Set `requires_grad = False` for every one of these parameters before constructing the optimizer.
- Only **LoRA adapter parameters** are allowed to be trainable.
- GradNorm's `log_lambdas` are also frozen during local/federated rounds — λ
  balancing only happens during central pretraining.

**Optimizer Construction**
- The optimizer MUST be initialized using **LoRA parameters only**.
- Backbone parameters MUST NOT be passed to the optimizer under any circumstance.
- Recommended setup:
  - Optimizer: `AdamW`
  - Learning rate: higher than backbone pretraining LR (e.g., `3e-3`)
  - Weight decay: `0` for LoRA parameters.

**Gradient Safety Checks**
- Implement a runtime assertion before training:
  - Verify that the number of trainable backbone parameters is zero.
- After backward pass, optionally assert that no backbone gradients are populated.

**Communication Payload Constraints**
- Clients MUST upload **only LoRA adapter weights** to the server.
- Backbone weights MUST NEVER be transmitted during federated rounds.
- The **ARV router `(Wr, br)`, shared experts `(Ek)`, objective
  embeddings `(et)`, and conventional MMoE gates MUST NEVER be transmitted**
  either — they are global parameters, identical to the frozen backbone.
  `get_lora_state_dict(model)` MUST filter by LoRA parameter name pattern
  (`"lora_"`, `".A"`, `".B"`) and MUST NOT accidentally include router,
  expert, or objective-embedding parameters even if their `requires_grad`
  state is ever mishandled.
- Communication cost reporting MUST be computed using:
  - Adapter parameter count × parameter dtype size (the MoE additions must not appear in this count).

**Model State Consistency**
- At the start of each federated round, clients MUST load:
  - Latest global backbone weights (frozen), **including the frozen ARV
    router / experts / objective embeddings**
  - Latest aggregated global adapter state (if applicable).

**Correctness Requirement**
- Any implementation that updates backbone parameters during personalization is considered **invalid** and must be rejected.
- Any implementation whose federated payload includes router, expert, or
  objective-embedding parameters is considered **invalid** and must be
  rejected — this would silently break the communication-efficiency claim
  central to RQ4.

LoRA learning rate MUST be tuned independently of backbone pretraining LR.
Typical range: 1e-3 to 1e-2.

LoRA parameters MUST be identified via name pattern ("lora_", ".A", ".B")
or via explicit parameter group returned by inject_lora().

During federated rounds, backbone weights checksum MUST remain identical
to the pretraining checkpoint.

---

## STEP 17 — FEDERATED SIMULATION (`training/federated.py`)

> **Implementation note:** The federated mechanism is:
> `weighted_fedavg`, client state persistence, and communication accounting
> all operate on LoRA state dicts exactly as before. The only new requirement
> is an explicit assertion that the frozen global model broadcast to clients
> each round (Step 17.2) **includes** the ARV router, shared experts, and
> objective embeddings as part of the frozen `Wbase`, and that
> `get_lora_state_dict()` / `client_updates` **never** contain keys belonging
> to `moe.router`, `moe.experts`, `moe.objective_embeddings`, or
> `moe.gates`. Add this assertion to `local_train_client`:
> ```python
> lora_sd = get_lora_state_dict(client_model)
> assert not any(k.startswith(("moe.router", "moe.experts",
>                               "moe.objective_embeddings", "moe.gates"))
>                for k in lora_sd.keys()), \
>     "MoE/router parameters leaked into federated payload"
> ```

### 17.1 Weighted FedAvg

Weight each client's LoRA update by its number of local training samples:

```python
def weighted_fedavg(client_updates: list[dict], client_sample_counts: list[int]):
    """
    client_updates: list of LoRA state dicts from each client
    client_sample_counts: number of training samples per client
    Returns: aggregated LoRA state dict
    """
    total_samples = sum(client_sample_counts)
    aggregated = {}
    for key in client_updates[0].keys():
        aggregated[key] = sum(
            (n / total_samples) * update[key]
            for update, n in zip(client_updates, client_sample_counts)
        )
    return aggregated
```

Use `weighted_fedavg` when `config.use_weighted_fedavg = True` (default).
Fall back to simple mean when `False`.

### 17.2 Client State Persistence

Before round 1, compute a checksum over the frozen global parameters: Transformer backbone, embeddings, shared experts, router/gates, objective embeddings, task heads, and any other non-LoRA state. Before and after every round, assert the checksum is unchanged. Only the aggregated LoRA adapter state may change on the server.


Maintain a `global_lora_state: dict` that persists across all rounds. Every client
loads from this state at the start of their local training:

```python
# Initialise global LoRA state once before round 1
global_lora_state = get_lora_state_dict(model_with_lora)

# Per round:
for round_idx in range(config.federated_rounds):
    sampled_clients = sample_clients(all_user_ids, config.clients_per_round)
    client_updates = []
    client_counts  = []
    for client_id in sampled_clients:
        # Client always resumes from latest global adapter
        load_lora_state_dict(client_model, global_lora_state)
        local_update, n_samples = local_train_client(client_model, client_id, ...)
        client_updates.append(local_update)
        client_counts.append(n_samples)
    global_lora_state = weighted_fedavg(client_updates, client_counts)
```

Clients that did not participate in a round are not affected — they will receive the
latest `global_lora_state` when they are next sampled.

### 17.3 Communication Cost Breakdown

Compute per round:

```python
lora_param_count = count_lora_parameters(model)

# Upload: client sends the resulting LoRA adapter state after local training
upload_kb_per_client   = lora_param_count * 2 / 1024   # BF16

# Download: client receives aggregated global LoRA
download_kb_per_client = lora_param_count * 2 / 1024   # BF16 (same size)

# Total round communication
total_round_kb = (upload_kb_per_client + download_kb_per_client) \
                 * config.clients_per_round
```

Log per round to `outputs/logs/federated_rounds_{model}.json`:

```json
{
  "round": 5,
  "val_ndcg": 0.41,
  "upload_kb_per_client": 80.0,
  "download_kb_per_client": 80.0,
  "total_round_kb": 16000.0,
  "num_clients": 100
}
```

### 17.4 Federated Convergence Plot

After all rounds complete, generate `outputs/plots/federated_convergence_{model}.png`:

- X-axis: FL round number (1 to `federated_rounds`).
- Y-axis: global validation NDCG@10.
- Include a horizontal dashed line showing the non-federated personalised baseline.
- Title: "Federated Convergence — {Model Name}".

---

## STEP 18 — EVALUATION METRICS (`evaluation/metrics.py`)

### 18.1 Fixed candidate protocol

For every evaluation example:

1. Load the pre-stored candidate set generated before model comparison.
2. Candidate sets contain exactly one ground-truth target plus the configured negatives after filtering.
3. Negatives exclude the target, padding ID, mask ID, and items prohibited by the evaluation-history rule.
4. The same candidate IDs are used for Transformer, MMoE, and ARV-MMoE.
5. Score candidates with the same serving path and rerank using the full model score.

If ANN retrieval is enabled, retrieve the configured candidate count, filter invalid/seen items, force-include the target, replace the lowest-scoring candidate if necessary, then rerank with the full model. Report the fraction of examples requiring target force-inclusion.

### 18.2 Metrics

Report:

| Metric | Definition for one relevant target |
|---|---|
| NDCG@10 | `1/log2(rank+1)` if target rank ≤10 else 0 |
| MRR@10 | `1/rank` if rank ≤10 else 0 |
| Recall@10 | `1` if target rank ≤10 else 0 |
| Precision@10 | `1/10` if target rank ≤10 else 0 |

Recall@10 equals HR@10 for this single-target protocol. If HR@10 is logged for continuity, state explicitly that it is numerically identical to Recall@10.

Return per-user/example values plus mean and standard deviation. For final multi-seed results, report mean ± std across independent seeds; do not manufacture standard deviations from repeated evaluation of one seed.


## STEP 19 — COLD-START EVALUATION (`evaluation/cold_start.py`)

### 19.1 Protocol

Cold items are created by the strict protocol in Step 4: an item must have its first observation strictly after the global training cutoff. The implementation must assert that no cold item appears in training.

Evaluate:
1. **Metadata ON:** model receives the available item metadata tensor.
2. **Metadata OFF:** model receives zero metadata for all items.

### 19.2 Item Classification

For each test sample, classify the target as cold or warm using the persisted `cold_item_set` and `warm_item_set`.

### 19.3 Metrics

Report NDCG@10, MRR@10, Recall@10, and Precision@10 separately for cold and warm targets, for metadata ON/OFF. Save counts and per-example values as well as aggregate means/stds.

Save to `outputs/logs/cold_start_{model}.json`. Generate `outputs/plots/cold_vs_warm_{model}.png` comparing cold/warm × metadata ON/OFF for the four primary metrics.

## STEP 20 — LONG-TAIL EVALUATION (`evaluation/longtail.py`)

### 20.1 Popularity Buckets

Compute item popularity from the training set only. Buckets must partition **all known item IDs**, including items with zero training interactions.

```python
popularity = Counter(item for seq in train_sequences.values() for item in seq)
all_item_ids = set(range(1, num_items + 1))
sorted_items = sorted(popularity.keys(), key=lambda x: (-popularity[x], x))
n = len(sorted_items)
head_items = set(sorted_items[:int(n * config.longtail_head_pct)])
tail_items = set(sorted_items[int(n * (1 - config.longtail_tail_pct)):])
mid_items = set(sorted_items) - head_items - tail_items
# Explicitly classify zero-training items as tail.
tail_items |= (all_item_ids - set(popularity.keys()))
assert head_items.isdisjoint(mid_items)
assert head_items.isdisjoint(tail_items)
assert mid_items.isdisjoint(tail_items)
assert head_items | mid_items | tail_items == all_item_ids
```

Save bucket assignments to `data/processed/popularity_buckets.pkl`.

### 20.2 Metrics Per Bucket

For each test sample, classify the target item by bucket and accumulate separately:

| Bucket | Definition           | Expected behaviour |
|--------|----------------------|--------------------|
| Head   | Top 20% by popularity | High NDCG (model biased toward popular) |
| Mid    | Middle 20%           | Moderate NDCG |
| Tail   | Bottom 60%           | Low NDCG (long-tail challenge) |

Save to `outputs/logs/longtail_{model}.json`.

Generate `outputs/plots/longtail_{model}.png`:
- Bar chart with 3 groups (head / mid / tail), 4 bars per group (NDCG@10, MRR@10, Recall@10, Precision@10).
- X-axis: popularity bucket. Y-axis: metric value.

---

## STEP 21 — LATENCY BENCHMARKING (`evaluation/flops.py`)

### 21.1 Full Specification

For each model variant, run the following benchmarking procedure:

```python
def benchmark_latency(model, config, device) -> dict:
    model.eval()
    # Synthetic single-sample input
    dummy_seq  = torch.zeros(1, config.max_seq_length, dtype=torch.long, device=device)
    dummy_meta = torch.zeros(1, config.max_seq_length, feature_dim,
                             dtype=torch.bfloat16, device=device)

    # Warm-up: discard first N runs to stabilise MPS JIT
    for _ in range(config.latency_warmup_runs):    # default: 20
        with torch.no_grad():
            _ = model(dummy_seq, metadata=dummy_meta)

    def synchronize_device():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    # Explicit synchronisation is mandatory because both MPS and CUDA execute
    # kernels asynchronously.
    synchronize_device()

    # Timed runs
    times = []
    for _ in range(config.latency_timed_runs):     # default: 100
        synchronize_device()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_seq, metadata=dummy_meta)
        synchronize_device()
        times.append((time.perf_counter() - start) * 1000)   # ms

    return {
        "mean_ms":   np.mean(times),
        "std_ms":    np.std(times),
        "min_ms":    np.min(times),
        "p95_ms":    np.percentile(times, 95),
        "ms_per_recommendation": np.mean(times)   # batch=1, so 1 rec per forward pass
    }
```

**Device synchronisation is mandatory** before and after each timed inference
measurement. MPS requires `torch.mps.synchronize()` and CUDA requires
`torch.cuda.synchronize(device)`; without this, asynchronous execution makes the
measured latency invalid.

> **Implementation note:** `model(dummy_seq, metadata=dummy_meta)` must exercise the full
> deployed serving path, including the `erec ≡ enext` ARV routing pass (Step
> 8.6) whenever `moe_variant != "none"`. Do not benchmark a stripped-down
> forward pass that skips routing — the reported latency must reflect what
> actually ships.

### 21.2 Efficiency Report

Log to `outputs/logs/efficiency_{model}.json`:

```json
{
  "total_parameters": 6200000,
  "trainable_parameters": 6200000,
  "lora_parameters": 40960,
  "router_parameters": 5248,
  "expert_parameters": 1050624,
  "moe_variant": "arv_mmoe",
  "num_experts": 4,
  "model_size_fp32_mb": 23.7,
  "model_size_bf16_mb": 11.8,
  "flops_per_inference": 125000000,
  "latency_mean_ms": 4.2,
  "latency_std_ms": 0.3,
  "latency_p95_ms": 4.9,
  "latency_routing_overhead_ms": 0.3,
  "ms_per_recommendation": 4.2
}
```

`router_parameters` and `expert_parameters` come from
`count_router_parameters()` (Step 8.5.5) and are `0` when `moe_variant =
"none"`. `latency_routing_overhead_ms` is measured by running
`benchmark_latency` twice — once with the full model, once with a
`moe_variant="none"` copy sharing the same backbone weights — and reporting
the difference; this isolates the inference cost specifically attributable to
routing (needed for RQ5's efficiency trade-off table).

---

## STEP 21.5 — ROUTING DIAGNOSTICS (`evaluation/routing_diagnostics.py`)

Routing is **not** assumed to be meaningfully adaptive — it must be measured.
This step consumes the `r_ut` routing-weight buffers accumulated during
training (Step 15.4.1) and/or recomputed in a dedicated evaluation pass.

### 21.5.1 Routing Entropy

Report both per-user/per-example distributions and aggregate statistics: mean entropy, standard deviation of entropy, and entropy by objective. Do not interpret low entropy as proof of usefulness.

```python
def routing_entropy(r: torch.Tensor) -> torch.Tensor:
    # r: (N, K) routing distributions
    return -(r * torch.log(r.clamp_min(1e-12))).sum(dim=-1)   # (N,) -- Eq. (13)
```

Report mean entropy (a) across all user sequences, (b) per objective, (c) per
`moe_variant` (none has no entropy; mmoe vs arv_mmoe are compared).

### 21.5.2 Expert Utilization

```python
def expert_utilization(r: torch.Tensor) -> torch.Tensor:
    # r: (N, K) -> (K,) fraction of total routing mass per expert -- Eq. (14)
    return r.mean(dim=0)
```

### 21.5.3 Objective-Specific Routing Differences

```python
def objective_routing_diff(mean_routing_per_objective: dict) -> dict:
    """
    mean_routing_per_objective: {"next_item": (K,), "masked_item": (K,), ...}
    Returns pairwise cosine similarity and KL divergence -- Eq. (15).
    """
    results = {}
    names = list(mean_routing_per_objective.keys())
    for i, t1 in enumerate(names):
        for t2 in names[i+1:]:
            r1, r2 = mean_routing_per_objective[t1], mean_routing_per_objective[t2]
            cos_sim = torch.nn.functional.cosine_similarity(r1, r2, dim=0).item()
            kl = (r1 * torch.log((r1.clamp_min(1e-12)) / r2.clamp_min(1e-12))).sum().item()
            results[f"{t1}_vs_{t2}"] = {"cosine_similarity": cos_sim, "kl_divergence": kl}
    return results
```

Low cosine similarity / high KL divergence between objectives is evidence the
router differentiates expert allocation by objective; high similarity / low
divergence suggests the routing is not truly objective-adaptive and should be
reported as such, not explained away.

### 21.5.4 Objective × Expert Heatmap

Build a `(num_objectives, num_experts)` matrix of mean routing weights and
save `outputs/plots/objective_expert_heatmap_{model}.png` (annotated heatmap,
rows = objective, columns = expert, colourbar = mean routing weight).

### 21.5.5 Collapse Tests

Report:
- fraction of routing distributions near-uniform, using `uniform_collapse_cv_threshold`;
- fraction of examples/users dominated by one expert, using `single_expert_collapse_threshold`;
- per-objective utilization mean/std.

A non-uniform router is not automatically useful. Collapse diagnostics only describe routing behavior.

### 21.5.6 Report & Plots

Save `outputs/logs/routing_diagnostics_{model}.json`:

```json
{
  "moe_variant": "arv_mmoe",
  "mean_entropy_overall": 1.12,
  "mean_entropy_per_objective": {"next_item": 0.95, "masked_item": 1.30, "multi_step": 1.05, "contrastive": 1.20},
  "expert_utilization": [0.31, 0.22, 0.28, 0.19],
  "collapse_tests": {"coefficient_of_variation": 0.19, "uniform_collapse": false, "single_expert_collapse": false},
  "objective_routing_diff": { "next_item_vs_masked_item": {"cosine_similarity": 0.41, "kl_divergence": 0.37}, "...": "..." }
}
```

Generate:
- `outputs/plots/routing_entropy_{model}.png` — bar chart of mean entropy per objective, one bar group per `moe_variant`.
- `outputs/plots/expert_utilization_{model}.png` — bar chart of `U_k` for `k=1..K`, with a dashed line at `1/K` (uniform reference).
- `outputs/plots/objective_expert_heatmap_{model}.png` — as in 21.5.4.

Run this diagnostic every `config.routing_diag_interval_epochs` epochs during
training (buffer reset each interval) and once more, comprehensively, at the
end of training on the full validation set.

---

## STEP 21.6 — GRADIENT-INTERFERENCE DIAGNOSTICS (`evaluation/gradient_diagnostics.py`)

Tests whether ARV-MMoE and/or GradNorm measurably change the distribution of
conflicting gradients (RQ3). Computed periodically (`config.grad_diag_interval_steps`)
on `config.grad_diag_num_batches` freshly-drawn batches, never on the same
batch used for the actual optimizer step (avoid perturbing training).

### 21.6.1 Gradient Cosine Similarity

```python
def gradient_cosine_similarity(losses: list[torch.Tensor], shared_params) -> torch.Tensor:
    """
    losses: per-objective losses computed on a held-out diagnostic batch.
    Returns a (T, T) matrix C_ij = cos(grad L_i, grad L_j) w.r.t. shared_params -- Eq. (16).
    """
    grads = []
    for L in losses:
        g = torch.autograd.grad(L, shared_params, retain_graph=True, allow_unused=True)
        flat = torch.cat([gi.flatten() for gi in g if gi is not None])
        grads.append(flat)
    T = len(grads)
    C = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            C[i, j] = torch.nn.functional.cosine_similarity(
                grads[i].unsqueeze(0), grads[j].unsqueeze(0)
            )
    return C
```

Negative off-diagonal entries are direct evidence of conflicting gradients.
Non-negative entries do **not** prove the absence of interference — only that
gradients are not locally opposed along the measured direction. Report the
full matrix and the mean off-diagonal value.

### 21.6.2 Per-Objective Gradient Norms

```python
def per_objective_grad_norms(losses: list[torch.Tensor], shared_params) -> dict:
    norms = []
    for L in losses:
        g = torch.autograd.grad(L, shared_params, retain_graph=True, allow_unused=True)
        norms.append(torch.norm(torch.cat([gi.flatten() for gi in g if gi is not None])).item())
    cv = float(np.std(norms) / (np.mean(norms) + 1e-8))
    return {"norms": norms, "coefficient_of_variation": cv}
```

Large `coefficient_of_variation` indicates gradient-scale imbalance across objectives.
It is not, by itself, proof of directional interference; interpret it together with
the cosine-similarity matrix.

### 21.6.3 Convergence Behaviour

Track per-objective loss curves and NDCG@10/MRR@10/Recall@10/Precision@10 over training steps; flag
diverging or oscillating curves as evidence of unresolved interference. Reuse
the training logs from Step 15 — no new logging mechanism required, just a
dedicated plot.

### 21.6.4 Report & Plots

Save `outputs/logs/gradient_diagnostics_{model}.json` with the cosine-similarity
matrix, mean off-diagonal similarity, per-objective norms, and CV, computed for
each `moe_variant` under comparison (multi-objective Transformer with no
routing, conventional MMoE, ARV-MMoE, ARV-MMoE+GradNorm).

Generate:
- `outputs/plots/gradient_cosine_similarity_{model}.png` — annotated heatmap of the (T,T) matrix.
- `outputs/plots/gradient_norms_{model}.png` — bar chart of per-objective gradient norms, with CV in the title.

ARV-MMoE is not claimed to eliminate interference — only to reduce or
redistribute it via differentiated expert pathways; GradNorm is not claimed to
eliminate it either — only to balance optimization contribution by magnitude.
Report the measured numbers and let the comparison across variants speak for
itself; do not editorialise beyond what the diagnostics show.

---

## STEP 21.7 — ABLATION STUDIES A–H (`training/ablation_runner.py`)

Implements every ablation from Table 1 / §3.6 of the paper as a runnable,
config-driven experiment. Every ablation is executable without editing Python source
code: `python main.py --stage ablation --study {A..H}` (Step 22) selects the study,
and the runner resolves all study-specific sub-configurations internally. Every
resolved configuration must be saved with the result. Each study writes its results to
`outputs/ablations/{study}_{model}.json` and, where noted, a matching plot.

### Ablation A — Architecture Progression (answers RQ1, RQ2, RQ3, RQ4)

Train and evaluate six configurations **in order**, with each configuration explicitly defined below:

| # | Configuration | Key config flags |
|---|---|---|
| 1 | Transformer + next-item only (single-objective baseline) | `use_masked_lm=False, multi_step_k=0, use_contrastive=False, moe_variant="none"` |
| 2 | Transformer + all 4 objectives, no routing | all objectives on, `moe_variant="none"`, static weights |
| 3 | Transformer + conventional task-specific MMoE | `moe_variant="mmoe"`, `use_gradnorm=False` |
| 4 | Transformer + ARV-MMoE | `moe_variant="arv_mmoe"`, `use_gradnorm=False` |
| 5 | Transformer + ARV-MMoE + GradNorm | `moe_variant="arv_mmoe"`, `use_gradnorm=True` |
| 6 | Federated ARV-MMoE + GradNorm + LoRA | config 5 + federated personalization (Step 16/17) |

Save `outputs/plots/architecture_progression.png` — bar or line chart of
NDCG@10/MRR@10/Recall@10/Precision@10 across configs 1–6, one figure shared across model sizes.

### Ablation B — Routing Formulation (answers RQ2)

Compares `config.router_conditioning = "objective_conditioned"` (proposed,
router sees `[Hu; et]`) against `"input_only"` (router sees `Hu` alone). Both
runs use `moe_variant="arv_mmoe"`, identical `K`, identical seed. Report
NDCG@10/MRR@10/Recall@10/Precision@10 delta and routing diagnostics (Step 21.5) for both.

### Ablation C — MMoE vs ARV-MMoE (answers RQ2)

Run `moe_variant="mmoe"` vs `moe_variant="arv_mmoe"` with
`match_mmoe_expert_count=True`. Report, per Step 8.5.5:
(i) total trainable parameters, (ii) expert parameters, (iii)
router/gating parameters, (iv) inference latency, alongside NDCG@10/MRR@10/Recall@10/Precision@10.
Generate `outputs/plots/mmoe_vs_arvmmoe_comparison.png` (grouped bars: NDCG@10, MRR@10, Recall@10, Precision@10, latency, per variant).

### Ablation D — ARV vs GradNorm (answers RQ3)

Four configurations, all with `moe_variant="arv_mmoe"` unless noted:

| Config | `moe_variant` | `use_gradnorm` |
|---|---|---|
| (i) Neither | `"none"` | `False` |
| (ii) ARV-MMoE only | `"arv_mmoe"` | `False` |
| (iii) GradNorm only | `"none"` | `True` (GradNorm balances the raw-hidden heads) |
| (iv) ARV-MMoE + GradNorm | `"arv_mmoe"` | `True` |

Report NDCG@10/MRR@10/Recall@10/Precision@10 and the full gradient-interference diagnostics (Step
21.6) for all four, to test whether the two mechanisms address complementary
sources of interference.

### Ablation E — Personalization (answers RQ4)

Three configurations: (a) global model only, (b) global + LoRA, (c) global +
LoRA + ARV (context-adaptive routing with LoRA adaptation, i.e. the full ARV-MMoE system). Reuse Step 16's personalization-gain report and
`global_vs_personalized_{model}.png`; add a third series to that plot for
config (c). Clarify in the report that ARV is context-adaptive routing (not a
persistent per-client personalization vector) — LoRA remains the sole
federated client-adaptation mechanism.

### Ablation F — Objective Ablation

Use the full 4-objective ARV-MMoE+GradNorm configuration as the reference run,
then remove each **auxiliary** objective individually: (1) MIM, (2) multi-step
forecasting, and (3) contrastive learning. Keep next-item prediction enabled in
all F runs because `e_next` is the trained serving objective embedding and
`e_rec ≡ e_next` at inference; disabling next-item would make the serving
embedding untrained and invalidate the evaluation protocol.

For each auxiliary ablation, disable the objective through its configuration
flag (`use_masked_lm=False`, `multi_step_k=0`, or `use_contrastive=False`). The
disabled objective must be excluded from active GradNorm, contributes zero to the
objective-weighted loss, and active GradNorm lambdas are renormalized. Report the
NDCG@10/MRR@10/Recall@10/Precision@10 delta versus the full reference.
Four runs total: one full reference + three single-objective-removed runs.

### Ablation G — Expert-Count Ablation

Sweep `config.expert_count_grid = [2, 4, 6, 8]` with `moe_variant="arv_mmoe"`,
all else fixed. Report NDCG@10/MRR@10/Recall@10/Precision@10, inference latency, and routing
diagnostics per `K`. Generate `outputs/plots/expert_count_ablation.png` — two
line plots sharing an x-axis (`K`): NDCG@10 (left y-axis) and latency (right
y-axis).

### Ablation H — Additional System-Level Ablations

Run these as explicit config-driven sub-experiments against the final system
(`moe_variant="arv_mmoe"`, `use_gradnorm=True`):

| H sub-study | Required flag/value sweep |
|---|---|
| H1 Metadata | `use_metadata=True` vs `False` |
| H2 Sequence length | `max_seq_length` over `config.sequence_length_grid` |
| H3 Cold-start | `cold_start_ratio` / metadata ON vs OFF using the fixed cold-start protocol |
| H4 Long-tail | fixed training-derived popularity buckets from Step 20 |
| H5 Latency | fixed serving path with the Step 21 benchmark configuration |
| H6 Communication | fixed LoRA payload and FedAvg configuration from Step 17 |

Each H sub-study must be selectable/resolved by the ablation runner without source edits.

### 21.7.1 CLI & Output Contract

No ablation may require commenting/uncommenting code or manually editing source files.
All ablation choices must be expressible as configuration flags/values and must be
logged in the resolved experiment configuration. The runner may set these values
programmatically from `--study`, but no source-code edits are permitted. `--study all` must expand to the
complete predefined A–H suite.

```
python main.py --stage ablation --study A --model t-base
python main.py --stage ablation --study G --model t-base
python main.py --stage ablation --study all --model t-base   # runs A-H sequentially
```

Every study writes `outputs/ablations/{study}_{model}.json` containing, at
minimum: the configurations compared, their config diffs, NDCG@10/MRR@10/Recall@10/Precision@10 (mean
± std), and a pointer to any generated plot file. `--study all` also writes
`outputs/ablations/summary_{model}.json` aggregating every study's headline
number for inclusion in the final report.

---

## STEP 22 — MAIN ENTRYPOINT (`main.py`)

### 22.1 CLI Design

```
python main.py --stage preprocess
python main.py --stage pretrain    --model {t-small|t-base|t-large}
                                   [--epochs N] [--batch_size B]
                                   [--moe_variant {none|mmoe|arv_mmoe}]
                                   [--num_experts K] [--use_gradnorm | --no_gradnorm]
python main.py --stage personalize --model {t-small|t-base|t-large}
python main.py --stage federated   --model {t-small|t-base|t-large}
                                   [--epochs N]
python main.py --stage evaluate    --model {t-small|t-base|t-large}
python main.py --stage cold_start  --model {t-small|t-base|t-large}
python main.py --stage longtail    --model {t-small|t-base|t-large}
python main.py --stage latency     --model {t-small|t-base|t-large}
python main.py --stage routing_diagnostics  --model {t-small|t-base|t-large}
python main.py --stage gradient_diagnostics --model {t-small|t-base|t-large}
python main.py --stage ablation    --study {A|B|C|D|E|F|G|H|all}
                                   --model {t-small|t-base|t-large}
python main.py --stage plots
python main.py --stage all         --model {t-small|t-base|t-large}
                                   [--epochs N] [--batch_size B]
```

`--stage all` runs the full pipeline (preprocess → pretrain → personalize →
federated → evaluate → cold_start → longtail → latency → routing_diagnostics →
gradient_diagnostics → plots) using `config.moe_variant` and
`config.use_gradnorm` as set in `configs/config.py` (or overridden via CLI).
For final paper runs, execute the complete pipeline independently for every
seed in `config.final_seeds` and aggregate only those independent seed results;
do not overwrite one seed's outputs with another seed's artifacts.
It does **not** run the ablation suite — that is always explicit via
`--stage ablation` since it trains multiple model variants.

### 22.2 New MoE/GradNorm/Ablation CLI Flags

```python
parser.add_argument("--moe_variant", type=str, default=None,
    choices=["none", "mmoe", "arv_mmoe"],
    help="Override config.moe_variant")
parser.add_argument("--num_experts", type=int, default=None,
    help="Override config.num_experts (K)")
parser.add_argument("--use_gradnorm", dest="use_gradnorm", action="store_true", default=None,
    help="Force GradNorm on (overrides config.use_gradnorm)")
parser.add_argument("--no_gradnorm", dest="use_gradnorm", action="store_false",
    help="Force GradNorm off (overrides config.use_gradnorm)")
parser.add_argument("--study", type=str, default=None,
    choices=["A", "B", "C", "D", "E", "F", "G", "H", "all"],
    help="Which ablation study to run (--stage ablation only)")
```

### 22.3 `--epochs` and `--batch_size` Flags

```python
parser.add_argument("--epochs",     type=int, default=None,
    help="Override config.pretrain_epochs (pretrain/all) or "
         "config.federated_rounds (number of FL rounds) for the federated stage")
parser.add_argument("--batch_size", type=int, default=None,
    help="Override config.batch_size for all stages")
```

Log which config values were overridden and the original vs new values.

### 22.4 Evaluate — Skip Models Without Weights

```python
def run_evaluate(model_name: str, config: Config):
    best_ckpt = Path(config.checkpoint_dir) / model_name / "best.pt"
    if not best_ckpt.exists():
        print(f"[SKIP] No checkpoint found for '{model_name}' at {best_ckpt}.")
        print(f"       Run: python main.py --stage pretrain --model {model_name}")
        return
    # Load and evaluate ...
```

Never raise an exception on missing checkpoint — always print the skip message and
the remedy command, then return.

### 22.5 Startup Routine (every stage)

```python
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

with open(Path(config.log_dir) / "run_config.json", "w") as f:
    json.dump(asdict(config), f, indent=2)
```

---

## STEP 23 — TRADEOFF EXPERIMENTS & PLOTS (`outputs/plots/`)

All plots: PNG, 150 DPI, labelled axes, legend, consistent palette.

### 23.1 Core Tradeoff Plots

| File | X-axis | Y-axis | Models |
|---|---|---|---|
| `accuracy_vs_model_size.png` | Size MB (BF16) | NDCG@10 | Transformer, T-Small, T-Base, T-Large, T-Base+LoRA |
| `accuracy_vs_communication.png` | Comm. cost/round (KB) | NDCG@10 | T-Small+LoRA, T-Base+LoRA, T-Large+LoRA |
| `accuracy_vs_flops.png` | FLOPs | NDCG@10 | T-Small, T-Base, T-Large |
| `personalization_gain_vs_lora_rank.png` | LoRA rank [2,4,8,16] | Mean NDCG gain | T-Base+LoRA |


### 23.2 More Plots

| File | Content |
|---|---|
| `federated_convergence_{model}.png` | Val NDCG@10 vs FL round; dashed baseline |
| `personalization_gain_histogram_{model}.png` | Histogram of per-user NDCG gains |
| `global_vs_personalized_{model}.png` | Scatter: NDCG without vs with LoRA per user (3 series with Ablation E) |
| `cold_vs_warm_{model}.png` | Grouped bar: cold/warm × metadata on/off |
| `longtail_{model}.png` | Bar chart: NDCG/MRR/Recall/Precision for head / mid / tail items |

### 23.3 ARV-MMoE / GradNorm / Ablation Plots

| File | Content | Produced by |
|---|---|---|
| `architecture_progression.png` | NDCG@10/MRR@10/Recall@10/Precision@10 across the 6 configs of Ablation A | Step 21.7 (Ablation A) |
| `mmoe_vs_arvmmoe_comparison.png` | Grouped bars: NDCG@10, MRR@10, Recall@10, Precision@10, latency, param counts — MMoE vs ARV-MMoE | Step 21.7 (Ablation C) |
| `expert_count_ablation.png` | NDCG@10 (left axis) & latency (right axis) vs `K ∈ {2,4,6,8}` | Step 21.7 (Ablation G) |
| `gradnorm_lambda_trajectory_{model}.png` | λₜ per objective vs training step | Step 7.5 log |
| `routing_entropy_{model}.png` | Mean routing entropy per objective, grouped by `moe_variant` | Step 21.5 |
| `expert_utilization_{model}.png` | Bar chart of `U_k`, dashed line at `1/K` | Step 21.5 |
| `objective_expert_heatmap_{model}.png` | Heatmap: objective (rows) × expert (cols) mean routing weight | Step 21.5 |
| `gradient_cosine_similarity_{model}.png` | Heatmap of pairwise objective gradient cosine similarity | Step 21.6 |
| `gradient_norms_{model}.png` | Bar chart of per-objective gradient norms, CV in title | Step 21.6 |

---

## STEP 24 — REPRODUCIBILITY CHECKLIST

- [ ] `torch.manual_seed`, `np.random.seed`, and `random.seed` set from `config.seed`.
- [ ] Final runs enumerate independent seeds from `config.final_seeds`.
- [ ] `batch_size`, `num_workers`, `max_seq_length`, dtype, and attention backend recorded.
- [ ] Kaggle dataset identifier and dataset fingerprint recorded.
- [ ] Global negative pool fingerprint recorded and reused.
- [ ] Fixed evaluation candidate-set fingerprint recorded and reused across models.
- [ ] `train_user_history is not eval_user_history` asserted.
- [ ] Test data is untouched until checkpoint/model configuration is frozen.
- [ ] `sum(active GradNorm lambdas) == number of active objectives` within tolerance.
- [ ] LoRA state dict contains no global MoE/router/expert/objective-embedding keys.
- [ ] Global frozen-state checksum is unchanged through every local/federated round.
- [ ] `OBJECTIVE_ORDER` is identical across model, loss, training, and diagnostic modules.
- [ ] MMoE/ARV-MMoE parameter reports are generated for every comparison.
- [ ] Routing diagnostics and gradient diagnostics use declared diagnostic data, not optimizer-step batches where prohibited.
- [ ] MPS FlexAttention self-test passes before MPS smoke training.
- [ ] H100 FlashAttention-4 self-test passes before full training.
- [ ] BF16 is used throughout; no silent dtype substitution occurs.
- [ ] All final results include the exact seed list that contributed to the aggregate.


## FINAL IMPLEMENTATION ORDER

1. Environment/dependency validation
2. Kaggle credential and dataset acquisition validation
3. Dataset integrity and fingerprint validation
4. Chronological split/cold-start unit tests
5. Sequence, left-padding, and multi-step indexing tests
6. Global negative-pool construction and candidate-set tests
7. Pre-tensorised dataset/DataLoader tests
8. Transformer + MPS FlexAttention self-test
9. Transformer baseline one-batch smoke test
10. Loss-function unit tests
11. ARV-MMoE / conventional MMoE shape and sharing tests
12. GradNorm active-objective/lambda/anchor tests
13. LoRA exact-key/global-freeze tests
14. Evaluation metric toy tests including MRR/Recall/Precision
15. ANN target force-inclusion toy test
16. One-batch end-to-end smoke test on Mac
17. One-epoch T-Small smoke test on Mac
18. H100 environment + FlashAttention-4 self-test
19. Full central pretraining on H100
20. Validation-only checkpoint selection
21. 1,000-user local personalization
22. Federated LoRA simulation
23. Final test evaluation
24. Routing / gradient / long-tail / cold-start diagnostics
25. Ablations A–H
26. Independent final seeds
27. Aggregate results and plots
28. Final results manifest

Do not jump from code generation directly to the full MovieLens-20M experiment.

## FINAL DELIVERABLES

| Artifact | Location |
|---|---|
| Full modular codebase | `project_root/` |
| Best + epoch checkpoints per model | `outputs/checkpoints/{model}/` |
| Pretrain logs per model | `outputs/logs/pretrain_{model}.log` |
| Personalisation gain report | `outputs/logs/personalization_gain_{model}.json` |
| Federated round metrics | `outputs/logs/federated_rounds_{model}.json` |
| Cold-start report | `outputs/logs/cold_start_{model}.json` |
| Long-tail report | `outputs/logs/longtail_{model}.json` |
| Efficiency report | `outputs/logs/efficiency_{model}.json` |
| Run config snapshot | `outputs/logs/run_config.json` |
| Item metadata matrix | `data/processed/item_metadata.npy` |
| Popularity bucket assignments | `data/processed/popularity_buckets.pkl` |
| Cold / warm item sets | `data/processed/cold_items.pkl`, `warm_items.pkl` |
| all required plot types | `outputs/plots/` |
| **GradNorm λ trajectory log | `outputs/logs/gradnorm_{model}.jsonl` |
| **MoE parameter comparison report | `outputs/logs/moe_parameter_report_{model}.json` |
| **Routing diagnostics report | `outputs/logs/routing_diagnostics_{model}.json` |
| **Gradient-interference diagnostics report | `outputs/logs/gradient_diagnostics_{model}.json` |
| **Per-study ablation results (A–H) | `outputs/ablations/{study}_{model}.json`, `outputs/ablations/summary_{model}.json` |
| **all required ARV-MMoE/GradNorm/ablation plots | `outputs/plots/` (Step 23.3) |

---

## STRICT RULES

1. Do not change architecture unless explicitly instructed.
2. Do not simplify model dimensions.
3. Do not remove LoRA.
4. No external recommendation libraries (for example, RecBole, Surprise, or pytorch-fm).
5. All model components implemented manually in PyTorch.
6. Personalisation evaluated on 1,000-user sample only — not all 138K users.
7. All file paths relative to `project_root/`.
8. Never hardcode `.cuda()` — always use `get_device(config)`.
9. `train_user_history` and `eval_user_history` always passed as separate arguments.
10. `evaluate` stage silently skips any model without `best.pt` — no exception.
11. Positional embeddings are learnable `nn.Embedding` — not sinusoidal.
12. Causal mask + key-padding validity must be enforced in every attention call using the representation required by the selected optimized backend.
13. Tied output layer: `output_proj.weight` must reference the item embedding weight.
14. `uv venv` is the only permitted environment manager.
15. Calibration is optional and never claimed to change ranking; if used, temperature is fit on validation only.
16. `torch.mps.synchronize()` called before each latency timing measurement on MPS.
17. `weighted_fedavg` is the default aggregation — not simple mean.
18. `global_lora_state` persists across all FL rounds — clients always resume from it.
19. `multi_step_k=0` disables multi-step prediction; `use_ann=False` disables ANN.
20. **The ARV router `(Wr, br)`, conventional MMoE gates, shared experts
    `(Ek)`, and objective embeddings `(et)` are **global** parameters — they
    must never appear in a LoRA state dict or a federated communication
    payload, and must never be updated during local/federated training.
21. **At inference, `erec ≡ enext` exactly (Step 8.6) — never introduce
    a separate, untrained inference-only objective embedding.
22. **The same expert outputs `Ek(Hu)` are computed once and reused
    across all four objectives — only the mixture (routing) weights `r_{u,t}`
    differ per objective; do not recompute or re-parameterise experts per
    objective.
23. **GradNorm's `log_lambdas` are optimised by their own optimizer,
    separate from the main model optimizer, and rescaled so the sum of active
    lambdas equals the number of active objectives after every update.
24. **Every MMoE-vs-ARV-MMoE comparison (Ablation C) must report total,
    expert, and router/gate parameter counts (`count_router_parameters()`)
    even when exact parameter matching is not achievable — never claim a
    comparison is "parameter-matched" without the report backing it.
25. **`moe_variant="none"` must remain a fully working direct-hidden-state path
    for Ablation A configs 1–2; this path consumes the Transformer hidden
    representations without an MoE layer.
