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
- Three holdout strategies are supported, selected by `config.cold_start_strategy`:
  - **`"unseen_in_training"` (DEFAULT, owner decision 2026-09-18)** — an item is cold iff
    it has **zero interactions in any user's training portion**:
    `cold = all_items - train_interacted_items`. Nothing is deleted; the set is whatever
    the chronological split already produces. `cold_start_ratio` does **not** apply — the
    size is data-determined. Optionally tightened with a first-appearance threshold
    (`cold = zero-training-interactions AND first_seen > D`) when a strictly temporal set
    is wanted; when no threshold is configured, none is applied.
  - **`"first_appearance_rank"`** — order all items by
    `(first_seen_timestamp, original_movie_id)` and hold out the latest-appearing
    `cold_start_ratio` fraction. Non-empty by construction. **Induced:** selected items
    that do have training interactions have them deleted.
  - **`"timestamp_cutoff"`** — hold out items whose first observation is strictly after
    the global training cutoff, selecting up to `cold_start_ratio` of those candidates in
    the same order. May be empty on a given dataset; an empty holdout is a hard error, not
    a warning. **Unusable under the per-user split protocol — see the rationale below.**
- Config validation must reject any other value; the strategy that ran must be recorded in
  `preprocessing_manifest.json`
- If fewer candidates exist than the ratio requests, use the achievable count and report it
- Remove cold items from the training split only; preserve in val/test. Under
  `"unseen_in_training"` this step is a **no-op by construction** — cold items have no
  training interactions to remove — and that is the point of the strategy
- Assert: `cold_item_set.isdisjoint(train_interacted_items)`, plus set partition assertions
- Save cold/warm item sets; log all relevant counts

**Rationale (owner decision 2026-09-18) — supersedes the REV-068 rationale below.**

The previous rationale recorded a **false cause** and is corrected here. It stated that
"on MovieLens-20M essentially every item appears early in the timeline, so
`timestamp_cutoff` selects the empty set". The empty set is real — measured: **exactly 0
items**. The stated cause is not. Measured on the real data, **9,009 items (43.5% of the
catalogue) are first seen after a global date at the 80th percentile of interactions
(2009-10-18)**. Items keep arriving throughout the decade.

The actual cause is the **cutoff definition interacting with the per-user split**:
`global_training_cutoff = max(all_train_timestamps)` is the maximum over *every user's*
training portion. Because each user is cut at their own 80% point and user activity windows
vary enormously, that maximum lands at **2015-03-31 04:45 — 0.1 days before the dataset's
final interaction**. Nothing can be first seen after it. The set is empty *by construction
of the cutoff*, not because of the item timeline:

| percentile of per-user training-end timestamps | date |
|---|---|
| p50 | 2003-02-18 |
| p90 | 2012-06-05 |
| p99 | 2015-01-25 |
| **p100 — what the code uses** | **2015-03-31** |

A genuine global-date temporal split would restore the timestamp protocol, but it costs
**96% of evaluable users**: at a p80 global cutoff only **5,539 of 136,677 users (4.1%)**
retain both training and evaluation data, because MovieLens users are bursty. That price is
not worth paying for the whole pipeline.

`"unseen_in_training"` obtains the intended semantics without it. An item is cold exactly
when the model never saw it, which under a chronological split is what happens to items
that entered the catalogue late. Measured on the real data: **3,010 items (14.5% of the
catalogue), all 3,010 appearing in val/test and therefore all evaluable**, with
first-appearance dates clustering late — p10 2010-08-17, **median 2013-09-20**, p90
2015-02-02, against a dataset ending 2015-03.

Why this is preferred over `first_appearance_rank`, which it replaces as the default: that
strategy produces an **induced** cold set. Measured: **927 of its 2,072 cold items (44.7%)
had real training interactions deleted** to make them cold — 6,634 interactions across
2,192 users, one item losing 619. `"unseen_in_training"` deletes nothing, restores those
2,192 users' training data, retains all 136,677 users, and makes the
`cold_item_set.isdisjoint(train_interacted_items)` guarantee hold trivially and exactly
rather than by construction-after-deletion.

**The paper's method section must describe `unseen_in_training`**, and must state that
cold items are those with no training-set interactions under the chronological split —
not a global-date rule, and not an induced holdout. If a strictly temporal characterisation
is wanted, apply the optional first-appearance threshold and report it: ~10% of the set
first appeared before 2010-08, so without a threshold the set is dominated by, but not
exclusively, recent items.

*Superseded rationale (REV-068, owner 2026-09-17), retained for history:*
`first_appearance_rank` was made the default because `timestamp_cutoff` selected the empty
set. That observation was correct; its stated cause was not, and the replacement chosen at
the time was induced rather than natural. Both strategies remain implemented and
selectable.

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
- **AC-03:** The cold-item set is non-empty, selected by the configured `cold_start_strategy`, recorded in `preprocessing_manifest.json`, and disjoint from all training interactions. Under `"unseen_in_training"` (the default) the cold set equals `all_items - train_interacted_items` exactly, and **no training interaction is deleted to produce it** — verified by comparing the training split before and after the holdout step and asserting they are identical. Under `"first_appearance_rank"` the held-out items are the latest-appearing `cold_start_ratio` fraction by first-appearance order; under `"timestamp_cutoff"` their first observation is strictly after the global training cutoff. Under every strategy, no cold item appears in training data
- **AC-04:** `train_user_history is not eval_user_history` assertion holds
- **AC-05:** Pre-tensorised datasets load without rebuilding tensors; `__getitem__` performs only indexing
- **AC-06:** Global negative pool contains only warm items; deterministic across runs with same seed
- **AC-07:** Fixed evaluation candidate sets are identical across all model evaluations
- **AC-08:** Augmentation strategies produce valid sequences (no real-item collisions with mask token)
- **AC-09:** Metadata matrix row i corresponds to remapped item ID i; row 0 is all-zeros
- **AC-10:** Dataset fingerprint and stats saved to `data/processed/dataset_stats.json`
