# Data Pipeline — Tasks

Ordered implementation tasks. Each task traces back to requirements/design. Tasks are grouped by dependency chain. Tests gate downstream work per steering/testing.md.

## Phase 1: Foundation

- [x] **T1.1: Create `configs/config.py`** — Implement the full `Config` dataclass with all fields specified in Step 3 of the prompt. Include all data paths, preprocessing thresholds, split ratios, sequence parameters, negative pool settings, genome config, and output directories. (FR-all, AC-10)

- [x] **T1.2: Create `utils.py`** — Implement `get_device(config)`, `resolve_dtype(config)` (must raise if not "bf16"), and global seed setup function (`torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic`). Include `seed_worker` function for DataLoader. (FR-all)

- [x] **T1.3: Create `requirements.txt`** — List exact dependency versions: `torch>=2.9.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `scikit-learn>=1.3.0`, `matplotlib>=3.7.0`, `tqdm>=4.65.0`, `kagglehub>=0.3.0`. (FR-1)

## Phase 2: Dataset Acquisition

- [x] **T2.1: Implement Kaggle acquisition in `data/preprocessing.py`** — Check for existing raw files under `config.raw_data_dir`. If missing and `auto_download_dataset=True`, authenticate via Kaggle API, download `grouplens/movielens-20m-dataset`, extract archive, locate 6 CSVs, copy to `data/raw/`. If credentials unavailable, fail with clear auth message. Validate file presence, required columns, and approximate row counts. Compute and save dataset fingerprint. (FR-1, AC-01)

- [x] **T2.2: Test dataset acquisition** — Unit test: mock Kaggle API responses; verify file validation catches missing files and columns. Integration test (optional, requires credentials): verify actual download succeeds and all 6 files are valid. (AC-01)

## Phase 3: Ratings Processing

- [x] **T3.1: Implement ratings pipeline in `data/preprocessing.py`** — Load `rating.csv`, validate columns, filter `rating >= implicit_threshold`, remove users with < `min_interactions`, sort per user by timestamp, remap userId → 0-indexed contiguous, remap movieId → 1-indexed contiguous (0=pad), build `{user_id: [item_ids]}` dict, compute `num_items`, `PADDING_ID=0`, `MASK_ID=num_items+1`, `VOCAB_SIZE=num_items+2`, skip users with < 3 items post-filter (log count), save `dataset_stats.json`. (FR-2, AC-02, AC-10)

- [x] **T3.2: Test ratings pipeline** — Unit test: synthetic ratings DataFrame → verify IDs are contiguous, 0 reserved for pad, num_items+1 for mask, min_interactions filter correct, chronological order within users. (AC-02)

## Phase 4: Cold-Start Holdout

- [x] **T4.1: Implement chronological splitting** — For each user's complete history, determine train/val/test boundaries using `train_ratio=0.80, val_ratio=0.10`. Compute `global_training_cutoff` = max timestamp in training split across all users. Implement as a function in `data/preprocessing.py` or `data/sequence_generator.py`. (FR-3, FR-5)

- [x] **T4.2: Implement cold-start holdout in `data/preprocessing.py`** — Find items with `first_seen > global_training_cutoff`. Select up to `cold_start_ratio` deterministically by `(first_seen, original_movie_id)`. Remove cold items from training split only. Assert: `cold_item_set.isdisjoint(train_interacted_items)`, `cold | warm == all_items`, `cold ∩ warm == ∅`. Save cold/warm sets to `data/processed/`. Log all counts. (FR-3, AC-03)

- [x] **T4.3: Test cold-start holdout** — Unit test: synthetic dataset with known timestamps → verify cold items are strictly post-cutoff, removed from training, present in val/test, set partition assertions hold. (AC-03)

- [x] **T4.4: Implement `"unseen_in_training"` as the cold-start default (REV-137, amended
  FR-3 / AC-03)** — Owner decision 2026-09-18, made after the REV-068 rationale was measured
  and found to record a **false cause**. An item is cold iff it has zero interactions in any
  user's training portion: `cold = all_items - train_interacted_items`. Nothing is deleted;
  `cold_start_ratio` does not apply. (FR-3, AC-03)
  - `data/preprocessing.py` — `identify_cold_items` gains the branch, computed against the
    union over the **unfiltered** `train_seqs` the function already receives. No new
    plumbing. The optional first-appearance threshold FR-3 describes is applied only when
    `config.cold_start_first_seen_after` is set; when it is `None`, none is applied.
  - `configs/config.py` — `cold_start_strategy` default is now `"unseen_in_training"`,
    `Config.validate` accepts exactly the three documented values and its error message
    names all three, and the new `cold_start_first_seen_after: Optional[float] = None`
    field carries the optional threshold. `cold_start_ratio` is annotated as applying only
    to the other two strategies.
  - **AC-03's new clause is ASSERTED, not assumed.** Under this strategy the removal step
    must be a no-op, so `filtered_train == train_seqs` is asserted directly, with a message
    naming the before/after user and interaction counts. This is what distinguishes a
    natural cold set from an induced one. `preprocessing_manifest.json` additionally records
    `train_interactions_before_holdout` and `train_interactions_removed_by_holdout`, so the
    question "was anything deleted to make this cold set?" is answerable **from the
    artifact** rather than from the code that wrote it.
  - **Non-vacuity verified, not claimed.** (a) Reinstating `first_appearance_rank` as the
    default makes `test_unseen_in_training_is_the_configured_default` FAIL; restored, it
    passes. (b) Injecting a selection that deletes a training interaction makes the AC-03
    assertion FIRE with the count in the message; restored, it does not. Both were run.
  - Tests in `tests/test_preprocessing.py::TestColdStartHoldout` (5 new): the configured
    default; `cold == all_items - train_interacted` exactly; **the no-deletion clause,
    which asserts the superseded default DOES delete on the same fixture** so the test
    discriminates between the two protocols rather than merely documenting one;
    `cold_start_ratio` does not move the set; the optional threshold tightens it and is
    still non-deleting.
  - **RUN AND VERIFIED (2026-09-18)** — `uv run python main.py --stage preprocess`, 107 s,
    exit 0. Every figure below read back off the artifacts, not the console:
      * **3,010 cold / 17,710 warm** over 20,720 items, against 2,072 / 18,648 before —
        matching REV-137's measured prediction exactly, including `warm 18,648 -> 17,710`.
      * **`train_interactions_removed_by_holdout: 0`**, against the 6,634 the superseded
        default removed across 2,192 users. All **136,677** users retain training history.
      * `cold == set(1..20720) - train_interacted_items` verified True by direct set
        comparison; `cold ∩ train_interacted = ∅`; `cold ∩ warm = ∅`; `cold ∪ warm ==
        {1..20720}`.
      * `preprocessing_manifest.json` records `cold_start_strategy: "unseen_in_training"`.
      * Negative pool rebuilt over exactly the 17,710 warm items (measured: 17,710 distinct,
        all ⊆ warm); both candidate matrices and `popularity_buckets.pkl` regenerated —
        the latter measured at head 4,144 / mid 4,144 / tail 12,432, so T3.4's
        catalogue-wide split survives the change.
  - **CASCADE COMPLETED.** This changes the training data, so the re-run went all the way
    through: preprocess → pretrain gate → personalize → federated → evaluate, cold_start,
    longtail, latency, routing_diagnostics, gradient_diagnostics. Sequenced with
    `training-pipeline` REV-129/REV-136 and `personalization-federated` REV-131 so one pass
    served all three, and landed **before** BUILD_ORDER Phase 14 as REV-137 directs. The
    authoritative record of that run is `specs/training-pipeline` T6.3.
  - **For the paper's method section**, per FR-3's rationale: cold items are those with no
    training-set interactions under the chronological split — not a global-date rule, and
    not an induced holdout. If a strictly temporal characterisation is wanted, set
    `cold_start_first_seen_after` and report it; without a threshold the set is dominated
    by, but not exclusively, recent items.

- [x] **T4.5: Remove the dead `item_first_seen_orig_id` local (REV-121)** — Declared in
  `identify_cold_items`, written nowhere and read nowhere; a leftover from the
  deterministic-ordering work D3 describes. The ordering is correct without it: `item_map`
  is built from `sorted(df["movieId"].unique())` with `idx + 1`, so the remapped ID is
  strictly monotone in the original movie ID and the code's `(first_seen, remapped_item_id)`
  sort induces exactly D3's stated `(first_seen, original_movie_id)` order. Deleted. INFO
  severity, no behavioural change. (FR-3)

## Phase 5: Metadata

- [x] **T5.1: Implement metadata extraction in `data/preprocessing.py`** — Load `genome_scores.csv` and `movie.csv`. For each item: extract top `genome_top_tags=20` genome features by relevance, one-hot encode genres. Build a `(VOCAB_SIZE, feature_dim)` = `(num_items+2, feature_dim)` matrix with row 0 all-zeros, rows 1..num_items aligned to remapped IDs, and the MASK_ID row (`num_items+1`) present and zero. Handle zero-metadata cold items (leave as zeros, log count). Save to `data/processed/item_metadata.npy`. (FR-4, AC-09)
  - BUILD note (REV-072, task-text sync): this said `(num_items+1, feature_dim)`, which
    predates the REV-072 widening to the full embedding vocabulary — one row short, so the
    MASK_ID row was missing and a metadata gather on a masked position indexed out of
    range. The code and the artifact have carried the wider shape since REV-072 and REV-090a
    (measured `(20722, 39)` against `VOCAB_SIZE=20722`); only this text was stale. The
    change is to the description, not to the requirement.

- [x] **T5.2: Test metadata extraction** — Unit test: small synthetic data → verify matrix shape, row 0 is zeros, known items have correct features, zero-metadata items are zeros. (AC-09)

## Phase 6: Augmentation

- [x] **T6.1: Implement `data/augmentation.py`** — `random_crop(seq, crop_ratio, rng)`: contiguous subsequence ≥ ceil(len*ratio). `random_mask(seq, mask_ratio, mask_token, rng)`: replace items with mask_token, keep ≥ 1 unmasked. `subsequence_sample(seq, rng)`: random prefix of length 1..len-1. `augment_sequence(seq, config, rng, mask_token)`: equal probability among three. `mask_token = num_items + 1` always. (FR-7, AC-08)

- [x] **T6.2: Test augmentation** — Unit test: verify crop returns valid subsequence, mask uses MASK_ID (not num_items), at least 1 unmasked item, subsequence is shorter than original. (AC-08)

## Phase 7: Sequence Generation & Tensorisation

- [x] **T7.1: Implement sequence generation in `data/sequence_generator.py`** — For each user: take complete history → apply split boundaries from T4.1 → apply cold holdout to training → generate sliding-window examples within each split → left-pad each example to `max_seq_length` → generate multi-step targets (no cross-split). Build separate `train_user_history` and `eval_user_history` dicts; assert they are distinct objects by identity. (FR-5, AC-04)

- [x] **T7.2: Implement `RecDataset` and pre-tensorisation** — `RecDataset(sequences, targets, user_ids, lengths, targets_k, metadata)` with `__getitem__` performing only tensor indexing. Cache tensorised arrays under `data/processed/`. Implement load-if-cached logic. (FR-6, AC-05)

- [x] **T7.3: Implement DataLoader construction** — Use `config.batch_size`, `config.num_workers=0`, `config.pin_memory`, `config.persistent_workers`. No worker initialization when `num_workers=0`. Include `seed_worker` from utils.py. (FR-6)

- [x] **T7.4: Test sequence generation** — Unit test: synthetic user histories → verify split-before-truncate order, left-padding alignment, multi-step targets don't cross splits, separate history dicts are distinct objects. (AC-04, AC-05)

## Phase 8: Negative Pool & Candidate Sets

- [x] **T8.1: Implement `GlobalNegativePool` in `data/sequence_generator.py`** — Construct from sorted warm item IDs with fixed seed. `build_once()` generates pool of size `pool_size`. `sample(n, rng)` draws from pool. Save to `data/processed/global_negative_pool.npy`. Reuse if fingerprint matches. (FR-8, AC-06)

- [x] **T8.2: Implement per-example negative filtering** — For each training example: filter pool to remove positive target, PADDING_ID, MASK_ID, and training-history-forbidden items. If insufficient valid negatives, wrap through pool; if still insufficient, fail with clear error. Produce exactly `num_sampled_negatives` valid negatives. (FR-8, AC-06)

- [x] **T8.3: Implement fixed evaluation candidate sets** — Generate once per eval split: one target + filtered negatives per example. Save to `data/processed/`. Every model evaluation loads these same sets. (FR-8, AC-07)

- [x] **T8.4: Test negative pool and candidate sets** — Unit test: verify pool is warm-only, deterministic across runs, pad/mask excluded. Verify eval candidate sets are identical when loaded by different model configs. (AC-06, AC-07)

## Phase 9: Integration

- [x] **T9.1: End-to-end preprocessing integration test** — Run full pipeline on MovieLens-20M (or a sampled subset for CI). Verify all output files in `data/processed/` are present and valid: `user_sequences.pkl`, `item_mapping.pkl`, `user_mapping.pkl`, `item_metadata.npy`, `cold_items.pkl`, `warm_items.pkl`, `dataset_stats.json`, tensorised datasets, negative pool, candidate sets. (AC-01 through AC-10)
  - BUILD note (REV-090a, 2026-09-17): `uv run python main.py --stage preprocess` was re-run against the current source, because every artifact under `data/processed/` predated the REV-072 / REV-086 / REV-088 fixes and demonstrably contradicted them. 108 s wall, **11.0 GB peak RSS** (`/usr/bin/time -l`, maximum resident set size 11,766,431,744 B). The three facts the finding named are now verified against the written artifacts: `item_metadata.npy` is (20722, 39) against `VOCAB_SIZE=20722` (the MASK_ID row is present); `train_examples['sequences']` and `['lengths']` are int32 ndarrays, not Python lists; and `lengths[i] == (sequences[i] != 0).sum()` for all of the first 5,000 of 7,794,505 examples (it was 1,232 / 2,000 mismatched before). Fingerprint 1ba739f1c65b3c4f.
