# Data Pipeline — Design

## Relevant Existing Architecture

Greenfield project — no existing code. Design follows the file structure from Prompt_v6.2 Step 2.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `data/preprocessing.py` | Create | Kaggle acquisition, ratings pipeline, cold-start holdout, metadata extraction |
| `data/sequence_generator.py` | Create | Chronological splitting, sliding-window examples, tensorisation, negative pool, candidate sets |
| `data/augmentation.py` | Create | Three augmentation strategies + pipeline |
| `configs/config.py` | Create | All data parameters (paths, thresholds, ratios, pool sizes) |
| `utils.py` | Create | `get_device()`, `resolve_dtype()`, seed utilities |

## Data Flow Architecture

```
Phase 1: Acquisition & Validation (preprocessing.py)
  Kaggle API → download → extract → validate 6 CSVs → fingerprint

Phase 2: Ratings Processing (preprocessing.py)
  rating.csv → implicit filter → min_interactions filter → sort → ID remap → user_sequences

Phase 3: Splitting (sequence_generator.py)
  Complete user histories → chronological train/val/test boundaries
  → global_training_cutoff → cold-start holdout on training portion

Phase 4: Cold-Start (preprocessing.py, called before sequence generation)
  Items first seen after training cutoff → cold candidates
  → deterministic selection → remove from training → save sets

Phase 5: Metadata (preprocessing.py)
  genome_scores + movies → top-k genome tags + one-hot genres
  → dense (num_items+2, feature_dim) matrix = the full embedding vocab;
    row 0 (pad) and row num_items+1 (MASK_ID) both all-zero

Phase 6: Sequence Generation (sequence_generator.py)
  Per user: sliding window within each split → left-pad to max_seq_length
  → multi-step targets (no cross-split) → RecDataset tensors

Phase 7: Negative Pool (sequence_generator.py)
  Warm items → deterministic global pool → per-example filtering
  → fixed eval candidate sets (generated once, saved)
```

## Key Design Decisions

### D1: Split-Before-Truncate Order
The chronological split boundary is determined on the **complete** user history. Only after the split is the per-example context truncated to `max_seq_length`. This prevents information leakage where truncation could move the effective split point.

### D2: Cold-Start Identification

**Amended (owner, 2026-09-18) to match FR-3.** Three strategies exist;
`config.cold_start_strategy` selects between them and the strategy that ran is recorded in
`preprocessing_manifest.json`.

- **`"unseen_in_training"` — the DEFAULT.** An item is cold iff it has zero interactions in any
  user's training portion: `cold = all_items - train_interacted_items`. **Nothing is deleted.**
  The set is whatever the chronological split already produces — 3,010 items (14.5%) on
  MovieLens-20M, all of them appearing in val/test and therefore all evaluable.
  `cold_start_ratio` does not apply. An optional first-appearance threshold tightens the set to
  a strictly temporal one.
- **`"first_appearance_rank"`.** Items ordered by first appearance; the latest-appearing
  `cold_start_ratio` fraction is held out. Non-empty by construction, but **induced** — 44.7% of
  its cold items had real training interactions deleted to make them cold. No longer the default.
- **`"timestamp_cutoff"`.** Cold items are those with `item_first_seen > global_training_cutoff`.
  Still implemented and selectable, but **unusable under this project's split protocol**: because
  the cutoff is `max()` over every user's own 80% point, it lands 0.1 days before the dataset's
  last interaction and the set is empty. An empty holdout is a hard error, not a warning.

The strategies differ in whether the disjoint-from-training guarantee is *achieved* or
*constructed*. `unseen_in_training` achieves it — the guarantee is the definition, so the
"strip cold items from training" step is a no-op. The other two construct it by deleting
training interactions of the items they select.

*This section previously described only the timestamp rule, then only the first-appearance rule.
`requirements.md` FR-3 is authoritative and carries the measured evidence for the 2026-09-18
decision, including why the earlier rationale's stated cause was wrong. The paper's method
section must describe `unseen_in_training`.*

### D3: Deterministic Cold-Item Selection
Cold candidates are ordered by `(first_seen_timestamp, original_movie_id)` for deterministic
selection across runs. Up to `cold_start_ratio` of eligible candidates are selected.

*Implementation note (owner, 2026-09-18): the code sorts by `(first_seen_timestamp, remapped_item_id)`.
This induces exactly the order above, because `item_map` is built from `sorted(df["movieId"].unique())`
with `idx + 1`, so the remapped ID is strictly monotone in the original movie ID. Recorded so the
difference is not later mistaken for a divergence.*

### D4: ID Remapping Scheme
Users start at 0 (for array indexing). Items start at 1 (0 is padding). Mask token is `num_items + 1`. This is a global invariant enforced during remapping and never violated.

### D5: Pre-Tensorisation
Tensors are constructed once and cached to `data/processed/`. `RecDataset.__getitem__` performs only tensor indexing — no string parsing, no filtering, no computation. This eliminates per-epoch overhead.

### D6: Global Negative Pool
One deterministic pool of warm items, generated with a fixed seed offset, saved to disk. Reused across epochs and runs (fingerprint-matched). Per-example filtering (remove target, pad, mask, forbidden items) happens at training time but draws from this fixed pool.

### D7: Fixed Evaluation Candidate Sets
Generated once per evaluation split using the same global pool + declared filtering rules. Saved to disk. Every model evaluation loads the same sets — guaranteeing identical comparison conditions.

## Affected Components

- `configs/config.py` must be created first — all other modules import from it
- `utils.py` must exist for device/dtype resolution
- `data/processed/` directory created by preprocessing
- All training/evaluation modules depend on the outputs listed in Step 4.5

## Error Handling

- Kaggle auth failure: raise clear message naming the credential setup steps
- Missing CSV file/column: raise identifying which file and which column
- Too few cold candidates: log achievable count, proceed (not an error)
- Negative pool exhaustion for an example: raise config error (pool_size too small relative to constraints)

## Testing Strategy

- Unit test ID remapping: verify contiguous range, padding=0, mask=num_items+1
- Unit test cold-start: synthetic ratings with known timestamps; verify cold items are post-cutoff
- Unit test left-padding: verify shape, alignment, pad positions
- Unit test negative pool: verify warm-only, deterministic, pad/mask excluded
- Integration test: full pipeline on small synthetic dataset → verify all output files present and valid
