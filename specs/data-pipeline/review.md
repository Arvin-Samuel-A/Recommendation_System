# Review — Data Pipeline

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/data-pipeline` — `configs/config.py`, `utils.py`, `data/preprocessing.py`, `data/sequence_generator.py`, `data/augmentation.py`: acquisition, ratings pipeline, chronological split, cold-start holdout, metadata extraction, sequence generation, tensorisation, global negative pool, fixed evaluation candidate sets.

## Review Scope

- Spec directory: `specs/data-pipeline/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master` (`origin/master` at the same commit). All feature work is uncommitted.
- Files inspected: `data/preprocessing.py` (676 lines, modified), `data/sequence_generator.py` (681 lines, modified), `data/augmentation.py`, `configs/config.py`, `utils.py`, `tests/test_preprocessing.py`, `tests/test_sequence_generator.py`, `tests/test_augmentation.py`
- The real artifacts under `data/processed/` (full MovieLens-20M, 136,677 users / 20,720 items) were loaded and checked directly, not only through the tests.
- Re-review of the prior `review.md`. `preprocessing.py` (21:24) and `sequence_generator.py` (21:23) were both modified after it; the `data/processed/` artifacts were written at 01:14–01:27, i.e. **before** those edits.
- Date: 2026-09-15

## Summary

Every prior finding against this spec that was a defect in `data/` has been fixed and the fixes verify: the dense `(N, S, F)` metadata block is gone (REV-058), `rating.csv` is read once with narrow dtypes (REV-085), the split dicts are preallocated int32 arrays (REV-086), candidate sets are written into an on-disk memmap and fingerprinted in row blocks (REV-087), and the metadata matrix now spans the full vocabulary including MASK_ID (REV-072). All ten acceptance criteria were checked against the real artifacts and hold.

What remains is one unresolved spec/implementation divergence that is the owner's to settle (REV-068, the cold-start protocol), plus three new low-severity items: a `lengths` column that is wrong for every training example but the last, a quadratic scan in metadata extraction, and the fact that the artifacts on disk predate the current source, so the real-scale memory claims behind REV-086/087 have not actually been re-demonstrated.

## Git Changes

`git status` (relevant subset):

| File | State |
|---|---|
| `configs/config.py` | modified |
| `data/preprocessing.py` | modified |
| `data/sequence_generator.py` | modified |
| `tests/test_preprocessing.py`, `tests/test_sequence_generator.py` | modified |
| `specs/data-pipeline/tasks.md` | modified |
| `specs/data-pipeline/review.md` | untracked (this file) |

`data/augmentation.py`, `utils.py` and `tests/test_augmentation.py` are unchanged from `32b0904`.

**Scope assessment:** in scope. No unexpected files, no new dependencies, no committed artifacts. `data/processed/` and `data/raw/` are untracked working data and correctly not staged.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Kaggle acquisition | PASS | `acquire_dataset` ([preprocessing.py:83](data/preprocessing.py:83)) caches when all 6 CSVs exist, validates columns per `EXPECTED_FILES`, raises a credential-setup message on download failure, writes `dataset_manifest.json`. `data/processed/dataset_manifest.json` exists with fingerprint `1ba739f1c65b3c4f`. |
| FR-2 Ratings pipeline | PASS | `_read_ratings` + `process_ratings` ([preprocessing.py:160](data/preprocessing.py:160)). Verified on the real artifacts: `item_mapping.pkl` values are contiguous `1..20720`; `dataset_stats.json` records PADDING_ID 0, MASK_ID 20721, VOCAB_SIZE 20722, `skipped_users` 0. |
| FR-3 Cold-start holdout | PARTIAL | Mechanism correct and asserted ([preprocessing.py:400-409](data/preprocessing.py:400)); verified on artifacts: 2,072 cold items, disjoint from all training interactions, present in val (543) and test (1,421). But the selection protocol implemented by default is `first_appearance_rank`, not the timestamp rule FR-3 describes — see REV-068. |
| FR-4 Item metadata | PASS | `extract_metadata` ([preprocessing.py:426](data/preprocessing.py:426)) allocates `(num_items + 2, feature_dim)`, asserts row 0 and row `num_items+1` are zero, logs the zero-metadata count, saves to `data/processed/item_metadata.npy`. |
| FR-5 Sequence generation | PASS | `compute_chronological_split` runs on complete histories before `identify_cold_items` filters training, and `generate_sequences` truncates only per example ([sequence_generator.py:199](data/sequence_generator.py:199)). Multi-step targets carry `split_end_idx`. `train_user_history is not eval_user_history` asserted at :268. |
| FR-6 Pre-tensorisation | PASS | `RecDataset.__getitem__` ([sequence_generator.py:103](data/sequence_generator.py:103)) is pure indexing plus one gather from the `(vocab, F)` matrix; `save_tensorised`/`load_tensorised` cache under `data/processed/`. See REV-088 for the `lengths` field's contents. |
| FR-7 Augmentation | PASS | `data/augmentation.py` unchanged; `tests/test_augmentation.py` covers crop/mask/subsequence and asserts `mask_token != num_items`. |
| FR-8 Negative pool & candidate sets | PASS | `GlobalNegativePool` ([sequence_generator.py:370](data/sequence_generator.py:370)) with fingerprint verification on load; per-example filtering in `filter_negatives` and `sample_training_negatives`; candidate sets generated once in `run_preprocessing` and persisted. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 six CSVs present and validated | PASS | `data/raw/` holds all six; `_validate_files` checks each required column; `dataset_manifest.json` written. |
| AC-02 IDs contiguous 1..N, 0=pad, N+1=mask | PASS | Loaded `item_mapping.pkl`: `min=1`, `max=20720`, `len=20720`. `dataset_stats.json` matches. |
| AC-03 cold items post-cutoff, absent from training | PARTIAL | Disjointness verified directly: `cold_item_set & train_interacted_items == ∅` over the real `train_user_history.pkl`. The "first observation strictly after the global training cutoff" half is not what the default strategy implements (REV-068). |
| AC-04 `train_user_history is not eval_user_history` | PASS | Asserted at [sequence_generator.py:268](data/sequence_generator.py:268) and again at [pretrain.py:722](training/pretrain.py:722); confirmed on the loaded pickles. |
| AC-05 tensorised datasets load without rebuilding | PASS | `load_tensorised` reads `*_tensorised.pt` directly; `tests/test_sequence_generator.py::TestLazyMetadataGather` asserts the resident tensor is the `(vocab, F)` matrix, not `(N, S, F)`. |
| AC-06 pool warm-only and deterministic | PASS | Loaded `global_negative_pool.npy` (207,200 entries): every unique value is in `warm_items.pkl`; 0 and 20721 absent. `build_once` is seeded; `load` recomputes and compares the fingerprint. |
| AC-07 fixed candidate sets identical across models | PARTIAL | `test_candidates.npy` is `(1,059,494, 101)`, matches the test split row count, target in column 0 verified for the first 1,000 rows, no pad/mask in row 0. But four downstream consumers regenerate rather than load these sets — see evaluation-diagnostics REV-093 and personalization-federated REV-096. |
| AC-08 augmentation produces valid sequences | PASS | `tests/test_augmentation.py::TestRandomMask::test_mask_token_not_num_items`. |
| AC-09 metadata row alignment | PARTIAL | The code is correct: `(num_items + 2, F)`, row 0 and MASK_ID row asserted zero. The artifact on disk is `(20721, 39)` — one row short, written by the pre-REV-072 code. `align_metadata_matrix` pads it on load, so nothing breaks; see REV-090. |
| AC-10 fingerprint and stats saved | PASS | `dataset_stats.json` and `preprocessing_manifest.json` both present, the latter recording pool size 207,200, pool fingerprint, both candidate-set fingerprints, cold-start strategy and seed. |

## Findings

```
REV-068
Severity: MEDIUM | Status: OPEN | Requirement: FR-3 | AC: AC-03
File: data/preprocessing.py:354-375 (`identify_cold_items`); configs/config.py:34 default
Problem: `cold_start_strategy` defaults to "first_appearance_rank", which holds out the
  latest-appearing `cold_start_ratio` fraction of items by rank. FR-3 and AC-03 describe a
  different protocol: "items whose first observation is strictly after the global training
  cutoff". Both strategies are implemented and the divergence is documented in the source
  and in `preprocessing_manifest.json`, but `requirements.md` still describes only the
  timestamp rule, and the reason for the change (the timestamp set is empty on
  MovieLens-20M) is recorded nowhere in the spec.
Impact: A reader of `requirements.md` will believe the cold-start protocol is the timestamp
  one. The cold-start research claim rests on which items are held out, so the paper's
  method section would be written from the wrong description.
Correction: This is a spec decision, not a BUILD fix. Raise it with the spec owner so FR-3
  and AC-03 can be reworded to describe both strategies and name the default, with the
  empty-set finding recorded as the rationale. BUILD must not edit requirements.md. The
  code needs no change: the loud `RuntimeError` on an empty holdout
  (preprocessing.py:379-387) is the right guard.
```

```
REV-088
Severity: LOW | Status: OPEN | Requirement: FR-6 | AC: AC-05
File: data/sequence_generator.py:233-238 (`generate_sequences`, training branch)
Problem: `length = min(len(train_items) - 1, S)` is computed once per user and written to
  `train_examples["lengths"]` for every one of that user's sliding-window examples. Each
  example i has a context of length i, so the stored value is correct only for the last
  example. Verified on a synthetic user with 6 training items: all five examples record
  `lengths=5` while the true non-pad counts are 1, 2, 3, 4, 5. The val and test branches
  (:249, :259) compute the per-example value correctly.
Impact: Inert today — nothing in `models/`, `training/` or `evaluation/` reads `lengths`,
  and `conventions.md` forbids `seq_lens`-based position selection, so nothing should. But
  `lengths` is a named `RecDataset` field under FR-6 and is persisted into
  `train_tensorised.pt`, so the first consumer to trust it gets wrong values silently.
Correction: Write the per-example length inside the zip loop --
  `train_examples["lengths"][ti] = min(len(context), S)` where `context` is the example's
  own context -- matching what the val/test branches already do. Add a unit test asserting
  `lengths[i] == (sequences[i] != 0).sum()` for a multi-example user.
```

```
REV-089
Severity: LOW | Status: OPEN | Requirement: FR-4 | AC: AC-09
File: data/preprocessing.py:476 (`extract_metadata`)
Problem: `movie_row = movie_df[movie_df["movieId"] == orig_id]` runs a full linear scan of
  `movie.csv` once per item, inside a loop over all 20,720 items -- ~565 million element
  comparisons and 20,720 intermediate boolean masks and frames. `genome_grouped.get_group`
  plus a per-item `nlargest` sits in the same loop.
Impact: Preprocessing-only, paid once per dataset build, but it is minutes of avoidable
  work and allocator churn in the phase that is already the pipeline's memory peak. No
  correctness effect.
Correction: Build the genre lookup once outside the loop --
  `genres_by_id = movie_df.set_index("movieId")["genres"].to_dict()` -- and index it, or
  vectorise the whole genre one-hot with `str.get_dummies("|")` reindexed onto the remapped
  IDs. Same for the genome side: one `groupby("movieId")["relevance"].nlargest(top_k)`
  rather than a per-item `get_group`.
```

```
REV-090
Severity: LOW | Status: OPEN | Requirement: FR-4, FR-6 | AC: AC-09
File: data/processed/ (artifacts); data/sequence_generator.py:82-84 (`RecDataset.__init__`)
Problem: Two related points.
  (a) The artifacts in `data/processed/` were written at 01:14-01:27, while
      `data/preprocessing.py` and `data/sequence_generator.py` were last modified at
      21:23-21:24. They are the output of the PRE-fix code: `item_metadata.npy` is
      `(20721, 39)` -- `num_items + 1` rows, so it has no MASK_ID row -- and
      `train/val/test_examples.pkl` hold Python lists rather than the int32 arrays
      `_alloc_split` now produces. Nothing is broken (`align_metadata_matrix` pads the
      short matrix on load, and `tensorise_examples` still accepts lists), but the
      real-scale memory behaviour claimed for REV-086/REV-087 has not been demonstrated by
      an actual run of the current code.
  (b) `RecDataset.__init__` calls `align_metadata_matrix(metadata_matrix)` with no
      `vocab_size`, so it does NOT pad. A caller that hands a raw `(num_items + 1, F)`
      array straight to `tensorise_examples`/`RecDataset` -- which is exactly what the
      on-disk artifact is -- would IndexError on the first masked sequence. The pipeline
      avoids this only because every live path goes through `load_metadata_matrix`, which
      does pad.
Impact: (a) leaves the headline memory fixes unverified at the scale they were written for.
  (b) is a latent trap one call site away from firing.
Correction: Re-run `python main.py --stage preprocess` with the current code and record
  peak RSS, so `data/processed/` matches the source and the memory claims have evidence.
  For (b), give `RecDataset.__init__` the vocabulary it needs (pass `vocab_size`, or derive
  it from `sequences.max() + 1`) so the alignment is unconditional wherever the matrix
  enters.
```

## Design Compliance

Follows `design.md` as written. D1 (split-before-truncate) is respected: `compute_chronological_split` runs on complete histories and `identify_cold_items` filters the training portion only, before `generate_sequences` truncates per example. D4 (ID scheme), D5 (pre-tensorisation), D6 (one deterministic pool) and D7 (candidate sets generated once) are all implemented as described.

Two deliberate deviations, both documented in the source:

- **D2/D3 cold-item selection.** `first_appearance_rank` is a second strategy design.md does not describe. It is config-selectable and the manifest records which was used, so it is not a silent redesign — but the spec still needs updating (REV-068).
- **REV-072 vocabulary alignment.** `align_metadata_matrix` / `load_metadata_matrix` are new helpers not in design.md's file table. They are a natural extension of D5 and are consumed uniformly by every call site; no concern.

No unapproved dependencies. `pandas`, `numpy`, `torch`, `kagglehub` only, as FR/Dependencies list.

## Task Completion

All 23 tasks in `specs/data-pipeline/tasks.md` are marked `[x]`. Verified:

| Task | Verdict |
|---|---|
| T1.1–T1.3 foundation | Complete. `Config` carries every field the downstream specs read; `utils.py` provides `get_device`, `resolve_dtype` (raises off bf16), `set_seed`, `seed_worker`. |
| T2.1–T2.2 acquisition | Complete. `tests/test_preprocessing.py::TestFileValidation` covers the missing-file and missing-column paths. |
| T3.1–T3.2 ratings | Complete, and confirmed against the real artifacts. |
| T4.1–T4.3 cold-start | Complete as implemented; the protocol itself is REV-068. `TestColdStartHoldout` covers both strategies including the empty-set case. |
| T5.1–T5.2 metadata | Complete. `TestMetadataExtraction` asserts shape, zero pad row and zero-metadata items. |
| T6.1–T6.2 augmentation | Complete. |
| T7.1–T7.4 sequences/tensorisation | Complete except the `lengths` contents (REV-088). `TestGenerateSequences::test_split_before_truncate` and `test_separate_history_dicts` are real tests of the properties they name. |
| T8.1–T8.4 pool & candidates | Complete. Fingerprint verification, pad/mask exclusion and determinism all covered. |
| T9.1 end-to-end integration | Complete in substance — `TestEndToEndPreprocessing` runs against the real `data/processed/` and checks every named output file, the cold/train disjointness, candidate-set/split alignment and fingerprints. Caveat: it is validating artifacts produced by the previous revision of the code (REV-090a). |

No false-complete task claims found in this spec.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Both failures are `transformer-backbone` REV-062's armed MPS backend gate; neither is in this spec. |
| `uv run python -m pytest tests/test_preprocessing.py tests/test_sequence_generator.py tests/test_augmentation.py` | All pass (included in the run above). |
| Direct artifact check: `item_mapping.pkl` contiguity | `min=1 max=20720 len=20720` — AC-02 holds. |
| Direct artifact check: `cold_items.pkl` vs `train_user_history.pkl` | Disjoint; 2,072 cold items; 543 in val, 1,421 in test — AC-03 disjointness holds. |
| Direct artifact check: `global_negative_pool.npy` | 207,200 entries, all in `warm_items.pkl`, no 0, no 20721 — AC-06 holds. |
| Direct artifact check: `test_candidates.npy` | `(1059494, 101)`, matches `test_examples.pkl` row count; column 0 equals the target for the first 1,000 rows; row 0 has no duplicates and no pad/mask — AC-07 structural half holds. |
| Direct artifact check: `item_metadata.npy` | `(20721, 39)`, row 0 all-zero, no MASK_ID row — REV-090a. |
| `generate_sequences` on a synthetic 6-item user (scratch script) | `lengths` = 5 for all five examples; true non-pad counts 1..5 — REV-088 confirmed. |

## Final Assessment

The data pipeline is in good shape and the prior CRITICAL is genuinely fixed. BUILD should:

1. Fix **REV-088** (per-example `lengths` in the training split) — small, self-contained, and add the assertion test.
2. Fix **REV-090b** (make `RecDataset`'s metadata alignment unconditional) and then re-run `--stage preprocess` so `data/processed/` matches the current source and the REV-086/087 memory claims have a measured run behind them (**REV-090a**).
3. Optionally fix **REV-089** (vectorise metadata extraction) while re-running preprocessing anyway.
4. Do **not** act on **REV-068** — escalate the cold-start protocol wording to the spec owner and leave `requirements.md` alone.

AC-03 and AC-09 stay PARTIAL until REV-068 is settled by the owner and preprocessing is re-run; AC-07 stays PARTIAL until the downstream consumers stop regenerating candidate sets (tracked in the evaluation-diagnostics and personalization-federated reviews, not here).
