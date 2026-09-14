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
  → dense (num_items+1, feature_dim) matrix, row 0 = zeros

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
Cold items are identified by `item_first_seen > global_training_cutoff`, not by popularity quantile. The global training cutoff is the latest timestamp in the training split, ensuring cold items are genuinely unseen during any training interaction.

### D3: Deterministic Cold-Item Selection
Cold candidates are ordered by `(first_seen_timestamp, original_movie_id)` for deterministic selection across runs. Up to `cold_start_ratio` of eligible candidates are selected.

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
