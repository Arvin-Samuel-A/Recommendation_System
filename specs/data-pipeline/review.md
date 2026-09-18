# Review — Data Pipeline

## Review Status

**PASS**

(Was `CHANGES_REQUIRED`. REV-137 is implemented and every figure it predicted was
re-measured here against the regenerated artifacts — all of them match exactly. REV-121 is
also closed. No finding in this spec is open.)

## Spec Amendments — Status

The 2026-09-18 owner amendments (authored by REVIEW at the owner's explicit instruction,
recorded so the spec's provenance is as traceable as the provenance it now requires of
artifacts) are all implemented:

| File | Change | Status |
|---|---|---|
| `requirements.md` FR-3 | `"unseen_in_training"` as DEFAULT; REV-068 rationale corrected | **Implemented and verified** |
| `requirements.md` AC-03 | "no training interaction is deleted" clause | **Implemented as a runtime assertion, and the assertion fires when broken — verified** |
| `design.md` D2 / Phase 5 | Three strategies; metadata shape corrected | doc only |
| `steering/code-quality.md` | "Artifact Provenance" + "Coverage: a property, not a list" | implemented; the coverage gap (REV-138) is now closed |

## Feature

`specs/data-pipeline` — `configs/config.py`, `utils.py`, `data/preprocessing.py`,
`data/sequence_generator.py`, `data/augmentation.py`: Kaggle acquisition, ratings
pipeline, chronological split, cold-start holdout, metadata extraction, sequence
generation, tensorisation, global negative pool, fixed evaluation candidate sets.

## Review Scope

- Spec directory: `specs/data-pipeline/` (requirements.md, design.md, tasks.md). All
  three unmodified in the working tree.
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`,
  `testing.md`.
- Baseline: `master` at `0bc67ee`, tracked by `origin/master` at the same commit. 34
  files modified in the working tree after BUILD's REV-137 / REV-136 round; this spec owns
  `data/preprocessing.py`, `configs/config.py` and `tests/test_preprocessing.py`. No
  untracked files.
- Files inspected: `data/preprocessing.py` (689 lines, read in full),
  `data/sequence_generator.py` (748 lines, read in full), `data/augmentation.py`,
  plus `tests/test_preprocessing.py`, `tests/test_sequence_generator.py`,
  `tests/test_augmentation.py`.
- The real artifacts under `data/processed/` were loaded and measured directly, not
  inferred from the code that writes them.
- Re-review (third pass). Prior findings carried in: REV-137 (MEDIUM, OPEN), REV-121
  (INFO, OPEN), REV-068 (superseded), REV-088, REV-089, REV-090 (RESOLVED).
- Date: 2026-09-18 (post-BUILD-fix)

## Summary

Nothing for BUILD. REV-137 is implemented correctly and the artifacts were fully
regenerated.

**Every figure REV-137 predicted was re-measured here and matches exactly** — 3,010 cold /
17,710 warm (against the finding's predicted 3,010 and the predicted warm transition
18,648 → 17,710), `train_interactions_removed_by_holdout: 0` against the 6,634 the
superseded default removed, and all 136,677 users retaining training history. The cold set
equals `{1..20720} - train_interacted_items` by direct set comparison, not by inference.

Two things BUILD did that are worth naming, because both are the harder option:

1. **The no-deletion clause is asserted, not assumed.** `preprocessing.py:427-435` asserts
   `filtered_train == train_seqs` under this strategy, and the manifest additionally
   records `train_interactions_before_holdout` / `train_interactions_removed_by_holdout`
   so the question is answerable *from the artifact*. I verified the assertion fires:
   injecting a selection that deletes one interaction produced the named error with the
   before/after counts.
2. **The discriminating test is self-checking.** `test_unseen_in_training_deletes_no_training_interaction`
   asserts, in the same test, that the *superseded* strategy DOES delete on the same
   fixture — so the test cannot quietly become vacuous if the fixture drifts. Independently
   confirmed: reverting the default fails 4 tests; restoring passes 9.

REV-121 (the dead local) is also gone.

The cold-start question is no longer vacuous, which was the original point:
`cold_start_t_small.json` now shows cold NDCG 0.0417/0.0414 (metadata on/off) against warm
0.0866/0.0770 — a real gap on a real cold set.

## Git Changes

This spec's files: `data/preprocessing.py` (+89/-33 — the new strategy branch, the AC-03
assertion, the manifest fields, and the REV-121 dead-variable deletion), `configs/config.py`
(the new default, the third accepted value, `cold_start_first_seen_after`), and
`tests/test_preprocessing.py` (+118 — five new `TestColdStartHoldout` cases).

**Scope assessment:** proportionate. The diff is confined to `identify_cold_items`, the two
config fields, and the tests. Nothing opportunistic rode along, and the two prior strategies
are left byte-equivalent in behaviour — verified by their existing tests still passing
unchanged.

`data/processed/` is gitignored, so the regeneration produces no tracked diff, which is why
the verification below measures the artifacts directly.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Kaggle acquisition + validation | PASS | `_validate_files` raises `FileNotFoundError` naming the path and `ValueError` naming the file and the missing column (`preprocessing.py:54-65`); `_compute_fingerprint` is size-based and deterministic. `dataset_manifest.json` present with fingerprint `1ba739f1c65b3c4f` and per-file row counts for all 6 CSVs. Credential failure raises with the `~/.kaggle/kaggle.json` setup steps and never substitutes a dataset |
| FR-2 Ratings pipeline | PASS | `dataset_stats.json` read directly: `num_users=136677`, `num_items=20720`, `PADDING_ID=0`, `MASK_ID=20721`, `VOCAB_SIZE=20722`, `total_interactions=9990682`, `skipped_users=0`. Items map from `idx + 1` (`preprocessing.py:222`), users from `0`. Sort is `["userId","timestamp"]`, `kind="stable"` |
| FR-3 Cold-start holdout (amended 2026-09-18) | PASS | All three strategies implemented and selectable; `Config.validate` accepts exactly the three documented values and its message names all three; unknown values raise; an empty holdout raises `RuntimeError`. `"unseen_in_training"` computes `all_items - pre_holdout_train_items` against the **unfiltered** `train_seqs`, which is the correct quantity. The optional `cold_start_first_seen_after` threshold is applied only when set. Measured on the manifest: `cold_start_strategy: "unseen_in_training"` |
| FR-4 Item metadata | PASS | `item_metadata.npy` measured at `(20722, 39)` = `VOCAB_SIZE × (20 genome + 19 genre)`. Row 0 all-zero and row 20721 (MASK_ID) all-zero, both measured on the artifact and asserted in code (`preprocessing.py:502-506`). `(no genres listed)` maps to a zero genre vector; zero-metadata items are counted and logged |
| FR-5 Sequence generation | PASS | `compute_chronological_split` runs on complete untruncated histories and truncation happens only inside `_left_pad`/`_fill_left_padded`, after the boundary is fixed. Multi-step targets zero out past `split_end_idx`. `train_user_history is not eval_user_history` asserted at `sequence_generator.py:277` |
| FR-6 Pre-tensorisation | PASS | `RecDataset.__getitem__` performs only indexing plus one `(S, F)` gather from the resident `(vocab, F)` matrix (the REV-058 design). Splits are preallocated `int32` ndarrays; `train_examples["sequences"]` measured as an int32 ndarray, not a list of lists |
| FR-7 Augmentation | PASS | Three strategies at equal probability in `augment_sequence`; `mask_token` is passed in as `num_items + 1` and never `num_items`; `tests/test_augmentation.py::test_mask_token_not_num_items` pins this specifically |
| FR-8 Global negative pool + fixed candidate sets | PASS | Pool re-measured after regeneration: 207,200 draws over exactly the **17,710** warm items (was 18,648 — it tracks the new cold set), `set(pool) ⊆ warm`, pad absent, mask absent, seed 42 recorded in `.meta.json`, fingerprint verified on load. `filter_negatives` and `sample_training_negatives` both exclude target/pad/mask/history and raise a named error on exhaustion |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 6 CSVs downloaded, validated, present | PASS | `dataset_manifest.json` lists all 6 with row counts (`rating.csv` 20,000,263; `genome_scores.csv` 11,709,768). `tests/test_preprocessing.py::TestFileValidation` covers missing file and missing column |
| AC-02 IDs contiguous 1..N, 0 pad, N+1 mask | PASS | `dataset_stats.json` as above; `test_contiguous_item_ids`, `test_padding_and_mask_ids` on synthetic frames |
| AC-03 cold set non-empty, strategy-selected, recorded, disjoint from training, **nothing deleted** (amended 2026-09-18) | PASS | Re-measured on the regenerated artifacts: **3,010 cold / 17,710 warm**; `cold == {1..20720} - train_interacted_items` **exactly** (direct set comparison, True); `cold ∩ train_interacted = ∅`; `cold ∩ warm = ∅`; `cold ∪ warm == {1..20720}`; all **136,677** users retain training history. The no-deletion clause holds and is *checkable from the artifact*: `train_interactions_removed_by_holdout: 0` against `train_interactions_before_holdout: 7937816`. It is also asserted at runtime (`preprocessing.py:427-435`), and **I verified the assertion fires** — injecting a selection that deletes one interaction produced the named error with before/after counts |
| AC-04 `train_user_history is not eval_user_history` | PASS | Identity assertion in `generate_sequences`; `test_separate_history_dicts` verifies both the identity and that the contents differ |
| AC-05 tensorised load, index-only `__getitem__` | PASS | `test_tensorised_datasets_loadable` loads all three splits; `TestLazyMetadataGather::test_resident_tensor_is_the_vocab_matrix_not_n_s_f` pins that no dense `(N, S, F)` block is built. `lengths` verified per-example, not per-user |
| AC-06 pool warm-only, deterministic | PASS | Re-measured after regeneration: 17,710 distinct draws, all ⊆ warm (see FR-8). `test_deterministic` rebuilds under the same seed and compares |
| AC-07 identical candidate sets across all evaluations | PASS | Every production consumer routes through `load_or_build_eval_candidates` (cold_start, longtail, federated, local_train) or a load-first branch with a warned, `config.seed`-derived fallback (`main.py:256`, `pretrain.py:622`); `ablation_runner.py:320` loads unconditionally. Stored `test_candidates.npy` is `(1059494, 101)` int64 and its row count equals the test split's 1,059,494 examples. `load_eval_candidates` recomputes and compares the block fingerprint on every load |
| AC-08 augmentation validity | PASS | `tests/test_augmentation.py` — crop contiguity and minimum length, mask uses MASK_ID with ≥1 unmasked, subsequence is a strict prefix, all three strategies reachable |
| AC-09 metadata row i ↔ remapped ID i, row 0 zeros | PASS | Matrix written by `metadata_matrix[remapped_id] = row_features` keyed off `item_map`; row 0 measured all-zero on the artifact and asserted in code |
| AC-10 fingerprint + stats saved | PASS | `dataset_stats.json` and `preprocessing_manifest.json` both present and populated; fingerprint `1ba739f1c65b3c4f` appears in both the dataset manifest and the preprocessing manifest |

## Findings

```
REV-137
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-3 (amended) | AC: AC-03 (amended)
File: data/preprocessing.py:342-435; configs/config.py:28-49, :302-310
Resolution: implemented as specified, and verified against the regenerated artifacts rather
  than the diff. Every predicted figure matches exactly:
    * 3,010 cold / 17,710 warm over 20,720 items -- including the predicted warm transition
      18,648 -> 17,710.
    * `train_interactions_removed_by_holdout: 0` against `train_interactions_before_holdout:
      7937816`, where the superseded default removed 6,634 across 2,192 users.
    * `cold == {1..20720} - train_interacted_items` by DIRECT SET COMPARISON (True), not by
      inference from counts. `cold ∩ train_interacted = ∅`, `cold ∩ warm = ∅`,
      `cold ∪ warm == {1..20720}`.
    * All 136,677 users retain training history.
    * Manifest records `cold_start_strategy: "unseen_in_training"`.
    * Cascade completed: negative pool rebuilt over exactly the 17,710 warm items, both
      candidate matrices regenerated, and `popularity_buckets.pkl` re-measured at
      4,144 / 4,144 / 12,432 -- so T3.4's catalogue-wide split survives the change.
  The implementation is correct in the detail that matters: `pre_holdout_train_items` is the
  union over the UNFILTERED `train_seqs`, which is the quantity the definition requires. The
  optional `cold_start_first_seen_after` threshold is applied only when set.
Verification of BUILD's two non-vacuity claims, both reproduced independently here rather
  than accepted:
    (a) Reverting `cold_start_strategy` to `"first_appearance_rank"` fails 4 tests in
        `TestColdStartHoldout`; restoring passes 9.
    (b) Injecting a selection that deletes one training interaction makes the AC-03
        assertion FIRE, with the message naming users before=1 after=1 and interactions
        before=8 after=7. Restored, it does not fire.
  Source restored after both; `git diff` on `configs/config.py` and `data/preprocessing.py`
  matches BUILD's committed state.
Worth recording as good practice: `test_unseen_in_training_deletes_no_training_interaction`
  asserts IN THE SAME TEST that the superseded strategy does delete on the same fixture, so
  it cannot silently become vacuous if the fixture drifts. That is a stronger construction
  than the task asked for, and it is the direct lesson of REV-105, where a test that
  asserted only `torch.isfinite` held under both candidate behaviours for three reviews.
Downstream: the cold-start question is no longer vacuous. `cold_start_t_small.json` reports
  cold NDCG 0.04165 / 0.04143 (metadata on / off) against warm 0.08658 / 0.07695 -- a real
  gap measured on a natural cold set.
```

```
REV-068
Severity: MEDIUM | Status: NOT_APPLICABLE (superseded by REV-137) | Requirement: FR-3 | AC: AC-03
File: specs/data-pipeline/requirements.md (FR-3, AC-03); data/preprocessing.py:346-380
Re-verified this pass, not carried on trust. FR-3's amended text was read clause by
clause against the implementation: both strategies present and selectable; config
validation rejects any other value (the `else` branch raises `ValueError` naming the two
valid choices); the strategy that ran is recorded in `preprocessing_manifest.json`
(measured: "first_appearance_rank"); an empty holdout raises `RuntimeError` rather than
logging. Every clause of the 2026-09-17 text holds.
SUPERSEDED 2026-09-18 by REV-137. This finding closed the gap between FR-3's text and the
code, and it did so correctly -- but neither it nor the three reviews after it questioned
the *reason* FR-3 gave. That reason ("essentially every item appears early in the timeline")
is false, and the strategy it justified is induced rather than natural. The lesson is the
one this finding illustrates by omission: a rationale is a claim about the data, and a claim
about the data can be measured. This one never was until REV-137.
```

```
REV-088
Severity: LOW | Status: RESOLVED | Requirement: FR-6 | AC: AC-05
File: data/sequence_generator.py:255-262
Re-verified. `train_examples["lengths"][ti] = min(j + 1, S)` is inside the per-example
zip loop, and `j + 1` is exactly the context length for the sliding window's example j.
Regression tests `test_train_lengths_are_per_example` and
`test_train_lengths_saturate_at_max_seq_length` both pass.
```

```
REV-089
Severity: LOW | Status: RESOLVED | Requirement: FR-4 | AC: AC-09
File: data/preprocessing.py:470-491 (`extract_metadata`)
Re-verified. `genres_by_id` is one `drop_duplicates(...).set_index(...).to_dict()` and
the genome side is one `groupby(...).nlargest(top_k)`; neither the per-item linear scan
of movie.csv nor the per-item `get_group` remains. The loop over `item_map` is now
dictionary lookups only.
```

```
REV-090
Severity: MEDIUM | Status: RESOLVED (both parts) | Requirement: FR-4, FR-6 | AC: AC-05, AC-09
File: data/processed/ (artifacts); data/sequence_generator.py:75-93
Re-verified against the written files, which is where this finding lived:
  (a) `item_metadata.npy` is (20722, 39) against VOCAB_SIZE 20722 — the MASK_ID row is
      present and all-zero. `train_examples["sequences"]` and `["lengths"]` are int32
      ndarrays. `val_examples` / `test_examples` `lengths` measured against non-pad
      counts: minimum length 2 (val) and 3 (test), zero rows with a length of 0, and the
      minimum non-pad count per row matches the minimum length in both splits.
  (b) `RecDataset.__init__` still calls `align_metadata_matrix(...)` unconditionally,
      with the vocabulary implied by the stored sequences as the target — so a raw
      pre-REV-072 artifact handed in directly is padded rather than raising.
```

```
REV-121
Severity: INFO | Status: RESOLVED | Requirement: FR-3 | AC: AC-03
File: data/preprocessing.py — identify_cold_items()
Resolution: the dead `item_first_seen_orig_id` local is deleted; a grep for the name over
  the file returns 0 matches. Folded into the REV-137 change, as suggested.
Superseded problem statement, retained for context:
Problem: `item_first_seen_orig_id: Dict[int, int] = {}` is declared and then neither
written nor read. It is a leftover from the deterministic-ordering work that D3
describes; the ordering itself is correct without it (see Design Compliance below).
Impact: none on behaviour. A reader looking for where original movie IDs enter the
cold-start ordering finds a dict that never receives one.
Correction: delete the line. INFO, not a defect — recorded so it is not rediscovered.
```

## Design Compliance

Compliant across D1–D7.

- **D1 split-before-truncate** — honoured: `compute_chronological_split` takes complete
  histories and the only truncation is per-example, inside the padding helpers.
- **D2/D3 cold-start identification and deterministic ordering** — D2 and D3 describe the
  `timestamp_cutoff` protocol and `(first_seen_timestamp, original_movie_id)` ordering.
  The default is now `first_appearance_rank` per the owner's FR-3 amendment, and the sort
  key is `(first_seen, remapped_item_id)` rather than the original movie ID. **Neither is
  a divergence in substance:** `design.md` is the older document and `requirements.md` is
  authoritative on the strategy; and because `item_map` is built from
  `sorted(df["movieId"].unique())` with `idx + 1`, the remapped ID is strictly monotone in
  the original movie ID, so the two tie-break keys induce exactly the same order. Worth
  noting only because `design.md` still reads as if `timestamp_cutoff` were the protocol.
- **D4 ID scheme** — enforced at remapping and respected by the negative sampler, the
  candidate generator and the metadata matrix, all three verified against the artifacts.
- **D5 pre-tensorisation** — demonstrated, not merely asserted: the cached arrays are
  int32 ndarrays and `load_tensorised` reconstructs `RecDataset` without rebuilding.
- **D6/D7 pool and fixed candidate sets** — one pool, one seed, fingerprint-checked on
  every load; candidate sets generated once by preprocessing and loaded everywhere else.

`num_workers=0` with `worker_init_fn=None` is respected (`build_dataloader`), per
steering's "no worker initialization when `num_workers=0`".

## Task Completion

All tasks T1.1–T9.1 are marked `[x]`. Verified this pass:

| Task | Verdict |
|---|---|
| T4.2 cold-start holdout | Complete. All four assertions present and the artifacts satisfy them |
| T5.1 metadata extraction | Complete, **and the task text is now correct.** The previous review carried this as an INFO: the text said `(num_items+1, feature_dim)` while the code and artifact had carried the wider vocabulary shape since REV-072. BUILD's note now records the widening explicitly and the measured `(20722, 39)`. Text and code agree |
| T7.1 / T7.2 sequences + `RecDataset` | Complete. Preallocation counts are cross-checked at runtime by an assertion comparing filled against allocated counts |
| T8.1 / T8.2 / T8.3 pool, filtering, fixed candidate sets | Complete. `sample()` exists (the T8.1 method a prior finding noted was missing), fingerprint verification is enforced on load, and the candidate sets are on disk |
| T9.1 end-to-end preprocessing integration | Complete. The BUILD note's three measurable claims were re-measured here and all three hold. `TestEndToEndPreprocessing` (8 tests, `slow`-marked but not deselected by default) exercises the real artifacts |

No false-complete tasks.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_preprocessing.py tests/test_sequence_generator.py tests/test_augmentation.py -q` | **72 passed**, 3.20 s |
| Load `data/processed/dataset_stats.json` (direct) | `num_items=20720`, `MASK_ID=20721`, `VOCAB_SIZE=20722`, `skipped_users=0` |
| Load `data/processed/preprocessing_manifest.json` (direct) | `cold_start_strategy="first_appearance_rank"`, 2,072 cold / 18,648 warm, `seed=42`, pool and candidate fingerprints present |
| Load `item_metadata.npy` (direct, mmap) | `(20722, 39)`; row 0 all-zero; row 20721 all-zero |
| Load `cold_items.pkl` + `warm_items.pkl` (direct) | overlap 0; union `== {1..20720}` |
| Load `global_negative_pool.npy` + meta (direct) | `(207200,)` int64, 18,648 distinct, `⊆ warm`, pad absent, mask absent |
| Load `val_examples.pkl` / `test_examples.pkl` (direct) | 993,372 / 1,059,494 rows; min length 2 / 3; zero rows with length 0; min non-pad count per row equals min length in both |
| `grep` every `generate_eval_candidates` / `load_eval_candidates` call site | 6 production consumers: 4 via `load_or_build_eval_candidates`, 2 load-first with warned `config.seed` fallbacks, 1 (`ablation_runner`) load-only |

## Final Assessment

**Nothing for BUILD. This spec is accepted.**

REV-137 landed correctly and completely, and it is the rare case where a finding's predicted
numbers and the implementation's measured numbers agree to the item. The cold set is now
natural rather than induced: nothing is deleted, every user keeps their training history,
and the guarantee that cold items are unseen in training holds *by definition* rather than
by construction-after-deletion — which is checkable from the manifest, not just from the
code.

The two things I would have been most suspicious of, I checked by breaking them: the AC-03
assertion fires when a deletion is injected, and the default-strategy tests fail when the
default is reverted. Both behaved correctly.

REV-121 is closed. REV-068 stays `NOT_APPLICABLE`, superseded.

**The provenance coverage gap is closed.** The original enumeration omitted `evaluation/`,
`main.py` and `utils.py`; the owner replaced it with an exclusion rule and BUILD implemented
it (`specs/training-pipeline` REV-138, now RESOLVED). It never affected this spec's verdict —
`data/preprocessing.py` and `configs/config.py` were covered throughout — but the rule it
belonged to was one I wrote too narrowly, and that is now corrected.

**A methodological note carried forward.** REV-068 closed the gap between FR-3's text and
the code, correctly, and three subsequent reviews re-confirmed it — none tested the claim
the rationale made about the data. It was false, and it justified a protocol that silently
deleted 6,634 training interactions. A rationale that asserts something measurable should be
measured at least once. This round is the correction; the guard against a repeat is that
FR-3 now carries its evidence inline.
