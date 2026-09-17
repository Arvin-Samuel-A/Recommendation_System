# Review — Data Pipeline

## Review Status

**PASS**

(Was `CHANGES_REQUIRED`. The sole open finding, REV-068, was a spec-wording gap that only
the owner could close; the owner closed it on 2026-09-17. No finding in this spec is open
and BUILD has nothing to do here.)

## Feature

`specs/data-pipeline` — `configs/config.py`, `utils.py`, `data/preprocessing.py`,
`data/sequence_generator.py`, `data/augmentation.py`: Kaggle acquisition, ratings
pipeline, chronological split, cold-start holdout, metadata extraction, sequence
generation, tensorisation, global negative pool, fixed evaluation candidate sets.

## Review Scope

- Spec directory: `specs/data-pipeline/` (requirements.md, design.md, tasks.md).
  `requirements.md` and `design.md` are unmodified since the last review; `tasks.md` has
  one added line (the T9.1 BUILD note).
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. 38 tracked files modified, no untracked files.
- **No source file owned by this spec is modified in the working tree** — `data/*.py` are
  at their committed state, so the code below is verified against `HEAD`'s content. What
  changed since the last review is the *data*: every artifact under `data/processed/` was
  regenerated.
- Files inspected: `data/preprocessing.py`, `data/sequence_generator.py`,
  `data/augmentation.py`, `configs/config.py`, `utils.py`, and the three test modules.
- The real artifacts under `data/processed/` (full MovieLens-20M, 136,677 users /
  20,720 items / 7,794,505 training examples) were loaded and measured directly.
- Re-review. Prior findings carried in: REV-068 (MEDIUM, OPEN), REV-088 (RESOLVED),
  REV-089 (RESOLVED), REV-090 (MEDIUM, part a OPEN).
- Date: 2026-09-17

## Summary

REV-090a is fixed and verified. `--stage preprocess` was re-run against the current
source (artifacts now dated 09-17 00:13–00:15, source 09-15 23:13–23:21), and all three
facts the finding named have flipped when measured against the written files:
`item_metadata.npy` is `(20722, 39)` against `VOCAB_SIZE=20722` (the MASK_ID row is
present), `train_examples["sequences"]` and `["lengths"]` are `int32` ndarrays rather than
Python lists, and `lengths[i] == (sequences[i] != 0).sum()` holds for all 5,000 sampled
examples — it was 1,232/2,000 mismatched last pass. The repository's data no longer
contradicts its own source.

One ordering detail was checked and is benign: `configs/config.py` was modified at
00:30, *after* the 00:13–00:15 preprocess run. The diff touches only
`federated_eval_examples`, the `max_steps_per_epoch` comment, and the
`attention_backend` validator — no field that feeds preprocessing — so the artifacts are
not stale with respect to it.

REV-068 is unchanged and is the only thing left. It is a spec-wording decision for the
owner; the code and the manifest are already honest about it.

## Git Changes

No source file owned by this spec appears in `git status`. The modified files belong to
`transformer-backbone`, `loss-functions`, `training-pipeline`,
`personalization-federated`, `evaluation-diagnostics`, `ablations-cli`, and the owner's
steering amendment. `specs/data-pipeline/tasks.md` gains one line: the T9.1 BUILD note
recording the preprocess re-run.

**Scope assessment:** nothing unexpected, nothing missing. The re-run this spec needed
produces no tracked diff by design — `data/processed/` is gitignored — so the evidence is
the artifacts themselves, which is where it was checked.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Kaggle acquisition + validation | PASS | `validate_raw_files` raises `FileNotFoundError`/`ValueError` naming the file and the column; `tests/test_preprocessing.py` covers both; `dataset_manifest.json` present with fingerprint |
| FR-2 Ratings pipeline | PASS | `dataset_stats.json`: `num_users=136677`, `num_items=20720`, `PADDING_ID=0`, `MASK_ID=20721`, `VOCAB_SIZE=20722`, `skipped_users=0`; ID-scheme unit tests on synthetic frames |
| FR-3 Cold-start holdout | PASS | FR-3 now documents both strategies and names `first_appearance_rank` as the default (owner, 2026-09-17). The implementation matches the amended text clause for clause: both strategies present, `configs/config.py:287` validates the choice, `preprocessing_manifest.json` records which ran, and an empty holdout raises rather than warning |
| FR-4 Item metadata | PASS | `item_metadata.npy` is `(20722, 39)` = `VOCAB_SIZE × (20 genome + 19 genre)`; row 0 verified all-zero. The artifact now matches the REV-072 widening |
| FR-5 Sequence generation | PASS | Split-before-truncate, left-padding, no cross-split multi-step targets, distinct history dicts — covered in `tests/test_sequence_generator.py` |
| FR-6 Pre-tensorisation | PASS | `RecDataset.__getitem__` is index-only; `__init__` aligns the metadata matrix unconditionally; stored arrays are `int32` ndarrays of shape `(7794505, 50)` |
| FR-7 Augmentation | PASS | Three strategies at equal probability, `mask_token = num_items + 1`; `tests/test_augmentation.py` |
| FR-8 Negative pool + fixed candidate sets | PASS | Pool measured: 207,200 draws over exactly the 18,648 warm items, no pad, no mask, fingerprint `3806cd334202b326`; `load_or_build_eval_candidates` is the single entry point for every consumer |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 6 CSVs validated | PASS | `dataset_manifest.json` present; validation tests cover missing file and missing column |
| AC-02 IDs contiguous 1..N, 0 pad, N+1 mask | PASS | `dataset_stats.json` as above; unit tests on synthetic frames |
| AC-03 cold-item set non-empty, strategy-selected, recorded, disjoint from training | PASS | AC-03 was reworded by the owner to describe both strategies. Measured against the artifacts: 2,072 cold items (non-empty), `cold_start_strategy: "first_appearance_rank"` recorded in `preprocessing_manifest.json`, warm ∩ cold = ∅, and `cold_item_set.isdisjoint(train_interacted_items)` asserted at `preprocessing.py:399` |
| AC-04 `train_user_history is not eval_user_history` | PASS | Identity assertion in `generate_sequences`; covered in `tests/test_sequence_generator.py` |
| AC-05 tensorised load, index-only `__getitem__` | PASS | Verified against the real artifact: `sequences`/`lengths` are int32 ndarrays, not lists. `__getitem__` indexes and performs one `(S, F)` gather from the `(vocab, F)` matrix (the REV-058 design) |
| AC-06 pool warm-only, deterministic | PASS | Measured directly: `set(unique(pool)) ⊆ warm_items`, pad and mask absent, warm ∩ cold = ∅ (18,648 / 2,072), seed 42 recorded in `.meta.json` |
| AC-07 identical candidate sets across evaluations | PASS | Every production site routes through `load_or_build_eval_candidates`; `tests/test_diagnostics.py::TestCandidateSetConsistency` asserts the rows the long-tail and cold-start evaluators hand to `score_candidates_batch` are the stored `test_candidates.npy` rows. Artifact measured: `(1059494, 101)` int64, row 0 duplicate-free, no pad/mask |
| AC-08 augmentation validity | PASS | `tests/test_augmentation.py` |
| AC-09 metadata row alignment | PASS | `item_metadata.npy.shape[0] == 20722 == VOCAB_SIZE`; row 0 all-zeros. The one-row shortfall flagged last pass is gone |
| AC-10 fingerprint + stats saved | PASS | `dataset_stats.json` and `preprocessing_manifest.json` present and populated; fingerprint `1ba739f1c65b3c4f` |

## Findings

```
REV-068
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-3 | AC: AC-03
File: specs/data-pipeline/requirements.md (FR-3, AC-03); data/preprocessing.py:354-387
Resolution: closed by the spec owner on 2026-09-17, which is the only way it could be
  closed -- the code was never wrong, the text was. FR-3 now documents BOTH strategies,
  names `first_appearance_rank` as the default, requires config validation to reject any
  other value, requires the strategy that ran to be recorded in
  `preprocessing_manifest.json`, and states that an empty holdout is a hard error rather
  than a warning. The rationale block records why the default changed: `timestamp_cutoff`
  selects the empty set on MovieLens-20M, which made `evaluation/cold_start.py` classify
  100% of targets as warm and rendered the cold-start research question vacuous. AC-03 was
  reworded to match and now asserts non-emptiness, strategy recording and
  disjoint-from-training rather than the timestamp property alone.
  Verified against the implementation after the edit: every clause of the amended FR-3 is
  already satisfied by `identify_cold_items` and `Config.validate`, so this closes with no
  BUILD work. The amendment also fixes the paper risk the finding actually cared about --
  the method section will now be written from a description of what runs.
```

```
REV-088
Severity: LOW | Status: RESOLVED | Requirement: FR-6 | AC: AC-05
File: data/sequence_generator.py:242-252
Resolution (code): confirmed still correct -- `train_examples["lengths"][ti] =
  min(j + 1, S)` is inside the zip loop. Resolution (artifact): the stored data now
  agrees. `lengths[i] == (sequences[i] != 0).sum()` for all 5,000 sampled rows of
  7,794,505; it was 1,232/2,000 mismatched last pass. Regression tests
  `test_train_lengths_are_per_example` and
  `test_train_lengths_saturate_at_max_seq_length` both pass.
```

```
REV-089
Severity: LOW | Status: RESOLVED | Requirement: FR-4 | AC: AC-09
File: data/preprocessing.py:470-491 (`extract_metadata`)
Resolution: unchanged from the last review -- `genres_by_id` is built once via
  `drop_duplicates(...).to_dict()` and the genome side is one
  `groupby(...).nlargest(top_k)`. Re-confirmed; the 108 s end-to-end preprocess wall
  time is consistent with the linear-scan version being gone.
```

```
REV-090
Severity: MEDIUM | Status: RESOLVED (both parts) | Requirement: FR-4, FR-6 | AC: AC-05, AC-09
File: data/processed/ (artifacts); data/sequence_generator.py:75-93
Resolution (a): `uv run python main.py --stage preprocess` was re-run against the current
  source. Artifacts are dated 09-17 00:13-00:15 against source mtimes of 09-15
  23:13/23:21. All three facts the finding required to flip, measured directly against
  the written files this pass:
    - `item_metadata.npy` is (20722, 39); `VOCAB_SIZE` is 20722. MASK_ID row present.
    - `train_examples["sequences"]` is an int32 ndarray (7794505, 50), `["lengths"]` an
      int32 ndarray (7794505,) -- not Python lists. The REV-086 memory fix has now been
      exercised at real scale.
    - 0 / 5,000 sampled `lengths` mismatches.
  BUILD recorded 11.0 GB peak RSS for the run in the T9.1 note, which is the evidence
  REV-086/REV-087 were missing.
Resolution (b): `RecDataset.__init__` still calls `align_metadata_matrix(...)`
  unconditionally; re-confirmed.
Note: `configs/config.py` was edited at 00:30, after the run. The diff touches only
  `federated_eval_examples`, a comment on `max_steps_per_epoch`, and the
  `attention_backend` validator -- no preprocessing input -- so the artifacts are not
  stale against it. Worth keeping in mind as a general hazard, not a finding.
```

## Design Compliance

Compliant. The ID scheme (pad=0, items=1..N, mask=N+1) is enforced at remapping and
respected by the negative sampler, the candidate generator and the metadata matrix — and
as of this pass it is respected by the artifacts as well, which is what D4 actually
claims. `__getitem__` stays index-only per FR-6 and the REV-058 sparse-metadata design.
D5's "constructed once and cached" is now demonstrated rather than asserted.
`num_workers=0` with no worker init is respected.

The one live divergence from the written spec remains REV-068's cold-start strategy,
documented in the source and the manifest but not in `requirements.md`.

## Task Completion

All tasks T1.1–T9.1 are marked `[x]`. Spot-checked this pass:

| Task | Verdict |
|---|---|
| T5.1 metadata extraction | Code correct and efficient. The task text still says "Build `(num_items+1, feature_dim)` matrix" while REV-072 widened the code — and now the artifact — to the full vocabulary. **Task text is stale, not the code.** Flagging rather than editing; `tasks.md` is not mine to change |
| T7.1/T7.2 sequences + `RecDataset` | Verified correct in code and against the regenerated artifacts |
| T8.1/T8.3 pool + fixed candidate sets | Verified against the artifacts, not only the generator |
| T9.1 end-to-end preprocessing integration | **Now genuinely complete.** Was false-complete last pass. The BUILD note records the re-run, the wall time, the peak RSS and the fingerprint, and the three claims it makes were independently re-measured here and all hold |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| `uv run python -m pytest -q -m slow --collect-only` | 8 slow tests collected, 259 deselected — confirming `slow` is *not* deselected in the default run above |
| Load `data/processed/dataset_stats.json` (direct) | `num_items=20720`, `MASK_ID=20721`, `VOCAB_SIZE=20722`, `skipped_users=0` |
| Load `data/processed/item_metadata.npy` (direct) | `(20722, 39)`, row 0 all-zero — REV-090a fact 1 flipped |
| Load `data/processed/train_examples.pkl` (direct) | `sequences` int32 ndarray `(7794505, 50)`, `lengths` int32 ndarray — REV-090a fact 2 flipped |
| Compare `lengths` to non-pad counts over 5,000 rows (direct) | **0 mismatches** — REV-090a fact 3 flipped |
| Load pool + warm/cold sets (direct) | 207,200 draws over exactly 18,648 warm items; pad/mask absent; warm ∩ cold = ∅ |
| Load `test_candidates.npy` (direct, mmap) | `(1059494, 101)` int64; row 0 duplicate-free, no pad/mask |
| `ls -la data/processed/` vs `stat` on `data/*.py`, `configs/config.py` | artifacts 09-17 00:13–00:15; data source 09-15 23:13–23:21; config 09-17 00:30 (diff is preprocessing-irrelevant) |

## Final Assessment

**Nothing for BUILD, and nothing left for the owner.** Every finding in this spec is
resolved and independently verified against the real artifacts rather than the tests alone.

The last two rounds closed this spec's two remaining gaps from opposite directions:
REV-090a was a BUILD gap (the on-disk data contradicted its own source) and was fixed by
re-running `--stage preprocess`; REV-068 was an owner gap (the requirement described a
protocol that had been abandoned for a good reason) and was fixed by amending FR-3 and
AC-03. Neither could have been closed by the other party, which is why both sat open
across several reviews.

Two INFO observations carried for the record, neither requiring action:

1. **T5.1 task text** still describes the pre-REV-072 `(num_items+1, feature_dim)` matrix
   shape. `tasks.md` is BUILD's to correct, not mine.
2. `configs/config.py` was edited after the preprocess run. The diff touches nothing that
   feeds preprocessing, so the artifacts are not stale — but the general hazard (config
   drifting ahead of the data generated from it) has no guard beyond the fingerprint, and
   the fingerprint covers dataset identity rather than config identity.
