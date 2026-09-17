# Data Pipeline — Requirements

## Overview

Acquire the MovieLens-20M dataset from Kaggle, preprocess it into implicit interaction sequences with chronological splits, extract item metadata, identify cold-start items, generate pre-tensorised training/validation/test datasets, build augmentation strategies, and construct a deterministic global negative pool and fixed evaluation candidate sets.

## Problem Statement

The recommendation system requires properly split, tensorised data with correct chronological ordering, cold-start holdout, and shared evaluation infrastructure. Data leakage (using test data for training decisions), incorrect ID mapping, or inconsistent candidate sets between models would invalidate all experimental results.

## Goals

- G1: Automated dataset acquisition from Kaggle with caching
- G2: Correct chronological train/val/test splitting on complete histories before truncation
- G3: Strict cold-start item holdout with verifiable guarantees
- G4: Dense item metadata matrix aligned to the canonical ID scheme
- G5: Pre-tensorised datasets that require only tensor indexing at training time
- G6: Deterministic, reusable global negative pool (warm items only)
- G7: Fixed evaluation candidate sets shared across all models
- G8: Three sequence augmentation strategies for contrastive learning

## Non-Goals

- Building a real-time data ingestion pipeline
- Supporting datasets other than MovieLens-20M
- Distributed data preprocessing

## Functional Requirements

### FR-1: Kaggle Acquisition (Step 1)
- Automatic download via Kaggle API using configured dataset identifier
- Credential storage outside repository, never logged
- Cached/local dataset reuse when files exist and `force_download=False`
- Validate all 6 CSV files: `rating.csv` (~20M rows), `movie.csv` (~27K), `tag.csv` (~465K), `genome_scores.csv` (~11.7M), `genome_tags.csv` (~1.1K), `link.csv` (~27K)
- Verify required columns per file; raise clear error on missing file/column
- Compute and save dataset fingerprint/metadata manifest

### FR-2: Ratings Pipeline (Step 4.2)
- Validate columns `[userId, movieId, rating, timestamp]`
- Keep rows where `rating >= config.implicit_threshold` (default 4.0)
- Remove users with < `config.min_interactions` (default 5) implicit interactions
- Sort per user by timestamp ascending
- Map userId → contiguous integers from 0
- Map movieId → contiguous integers from 1 (NOT 0 — 0 is padding)
- Store as `{user_id: [item_id, ...]}` in chronological order
- Derive and save `dataset_stats.json`: num_items, PADDING_ID=0, MASK_ID=num_items+1, VOCAB_SIZE=num_items+2
- Skip users with < 3 items after filtering; log total skipped

### FR-3: Cold-Start Holdout (Step 4.3)
- Determine complete chronological train/val/test split on untruncated histories first
- Define global training cutoff = latest timestamp among training-split interactions
- Two holdout strategies are supported, selected by `config.cold_start_strategy`:
  - **`"first_appearance_rank"` (DEFAULT)** — order all items by
    `(first_seen_timestamp, original_movie_id)` and hold out the latest-appearing
    `cold_start_ratio` fraction. Non-empty by construction.
  - **`"timestamp_cutoff"`** — hold out items whose first observation is strictly after
    the global training cutoff, selecting up to `cold_start_ratio` of those candidates in
    the same order. May be empty on a given dataset; an empty holdout is a hard error, not
    a warning.
- Config validation must reject any other value; the strategy that ran must be recorded in
  `preprocessing_manifest.json`
- If fewer candidates exist than the ratio requests, use the achievable count and report it
- Under either strategy: remove cold items from the training split only; preserve in val/test
- Assert: `cold_item_set.isdisjoint(train_interacted_items)`, plus set partition assertions
- Save cold/warm item sets; log all relevant counts

**Rationale (REV-068, owner decision 2026-09-17):** this requirement previously described
only the timestamp protocol. On MovieLens-20M essentially every item appears early in the
timeline, so `timestamp_cutoff` selects the **empty set** — `evaluation/cold_start.py`
classified 100% of targets as warm and the entire cold-start research question became
vacuous. `first_appearance_rank` preserves the substantive semantics ("these items are new
relative to the rest of the catalogue"), keeps the disjoint-from-training guarantee because
selected items are stripped from the training sequences, and is non-empty on any dataset.
Both strategies remain implemented and selectable; the default changed, and this text now
says so. The paper's method section must describe `first_appearance_rank`, not the
timestamp rule.

### FR-4: Item Metadata (Step 4.4)
- Dense matrix shape `(num_items + 1, feature_dim)` where `feature_dim = genome_top_tags + num_genres`
- Row 0 = all-zero padding row; rows 1..num_items aligned to remapped IDs
- Genome tag features: top 20 tags by relevance per item
- Genre features: one-hot encoded from pipe-delimited genres; `(no genres listed)` → zero vector
- Cold items get metadata if data exists; zero-metadata items stay all-zeros
- Log zero-metadata cold item count
- Save to `data/processed/item_metadata.npy`

### FR-5: Sequence Generation (Step 6)
- For every user: filter → sort complete history → determine split boundary on complete history → apply cold-item holdout to training portion → generate sliding-window examples → truncate per-example context to `max_seq_length`
- Never truncate complete history before determining split
- Left-padding to fixed length S with pad ID = 0
- Multi-step targets: `[history[j+1], ..., history[j+k]]`; no target may cross split boundary
- Separate `train_user_history` and `eval_user_history` dictionaries; assert they are distinct objects

### FR-6: Pre-Tensorisation (Step 6.5)
- Construct tensors once; `__getitem__` must only index tensors
- `RecDataset` with fields: sequences, targets, user_ids, lengths, targets_k (optional), metadata (optional)
- Cache tensorised arrays under `data/processed/`; load rather than rebuild each epoch

### FR-7: Augmentation (Step 5)
- Three strategies: random_crop, random_mask, subsequence_sample
- `augment_sequence()` applies one at random with equal probability
- `mask_token = num_items + 1` (never `num_items`)
- Used by contrastive loss and masked item modeling

### FR-8: Global Negative Pool (Step 6.7)
- Deterministic pool from warm items only; cold items never appear as training negatives
- Generated once with fixed seed; saved and reused if fingerprint matches
- Per-example filtering removes: positive target, padding ID, mask ID, history-forbidden items
- If pool cannot provide enough valid negatives, fail with clear error
- Fixed evaluation candidate sets generated once per eval split; shared across all compared models

## Edge Cases

- User with exactly `min_interactions` items: included
- User with < 3 items after filtering: skipped with logged count
- Cold item with no genome/genre data: zero-metadata row (logged)
- Fewer cold candidates than requested ratio: use achievable count, report
- Kaggle credentials unavailable + no local dataset: fail with auth message, never substitute
- Rating file with unexpected column names: fail with clear error

## Constraints

- All item IDs in range [1, num_items]; padding=0, mask=num_items+1
- `num_workers=0` by default; no worker initialization when 0
- Dataset fingerprint must be computed and recorded for reproducibility
- `train_ratio=0.80, val_ratio=0.10` (test = 0.10 implicit)

## Dependencies

- `pandas`, `numpy`, `kagglehub`, `torch`, `scikit-learn`
- `configs/config.py` for all parameters
- `utils.py` for device/seed helpers

## Acceptance Criteria

- **AC-01:** All 6 CSV files downloaded, validated, and present under `data/raw/`
- **AC-02:** Item IDs are contiguous 1..N with 0 reserved for padding and N+1 for mask
- **AC-03:** The cold-item set is non-empty, selected by the configured `cold_start_strategy`, recorded in `preprocessing_manifest.json`, and disjoint from all training interactions. Under `"first_appearance_rank"` (the default) the held-out items are the latest-appearing `cold_start_ratio` fraction by first-appearance order; under `"timestamp_cutoff"` their first observation is strictly after the global training cutoff. Under either strategy, no cold item appears in training data
- **AC-04:** `train_user_history is not eval_user_history` assertion holds
- **AC-05:** Pre-tensorised datasets load without rebuilding tensors; `__getitem__` performs only indexing
- **AC-06:** Global negative pool contains only warm items; deterministic across runs with same seed
- **AC-07:** Fixed evaluation candidate sets are identical across all model evaluations
- **AC-08:** Augmentation strategies produce valid sequences (no real-item collisions with mask token)
- **AC-09:** Metadata matrix row i corresponds to remapped item ID i; row 0 is all-zeros
- **AC-10:** Dataset fingerprint and stats saved to `data/processed/dataset_stats.json`
