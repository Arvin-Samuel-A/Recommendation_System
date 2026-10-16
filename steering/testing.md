# Testing Strategy

## Gate: No Full Training Without Passing Tests

Full MovieLens-20M training is forbidden until ALL of the following pass:
1. Unit tests
2. Integration tests
3. One-batch smoke tests
4. Few-step T-Small smoke test on real data

## Required Test Progression (Implementation Order)

| Phase | Tests | Must pass before |
|---|---|---|
| 1 | Environment/dependency validation | Any code execution |
| 2 | Kaggle credential and dataset acquisition validation | Preprocessing |
| 3 | Dataset integrity and fingerprint validation | Preprocessing |
| 4 | Chronological split / cold-start unit tests | Sequence generation |
| 5 | Sequence, left-padding, multi-step indexing tests | Model training |
| 6 | Global negative-pool construction and candidate-set tests | Training |
| 7 | Pre-tensorised dataset / DataLoader tests | Training |
| 8 | Transformer + MPS SDPA self-test (forward + backward) | Training |
| 9 | Transformer baseline one-batch smoke test | Multi-epoch training |
| 10 | Loss-function unit tests (all four + GradNorm) | Training |
| 11 | ARV-MMoE / conventional MMoE shape and sharing tests | MoE training |
| 12 | GradNorm active-objective / lambda / anchor tests | GradNorm training |
| 13 | LoRA exact-key / global-freeze tests | Personalization |
| 14 | Evaluation metric toy tests (NDCG, MRR, Recall, Precision) | Evaluation |
| 15 | ANN target force-inclusion toy test | ANN evaluation |
| 16 | One-batch end-to-end smoke test on Mac | Full training |
| 17 | Few-step T-Small smoke test on Mac, on real data | Full training |
| 18 | Hopper environment + FlashAttention-3 self-test and parity (`scripts/verify_attention_backend.py --backend flash_attention_3`) | H100 training |

Phase 18 was "H100 environment + FlashAttention-4 self-test"; owner decision 2026-10-15 replaced
FA4 with FlashAttention-3 on Hopper (see `code-quality.md`). Non-Hopper CUDA machines run SDPA and
gate on `scripts/verify_attention_backend.py --backend sdpa`.

## What Each Test Category Verifies

**Unit tests:** Individual functions produce correct outputs for known inputs. Examples:
- ID remapping yields contiguous 1..N with 0 reserved for padding
- Cold items have first observation strictly after training cutoff
- Left-padded sequences have correct shape and alignment
- Multi-step targets at causal positions S-1-step
- Negative pool contains only warm items, excludes pad/mask
- Loss functions produce expected gradients on toy inputs
- LoRA state dict keys match `lora_`/`.A`/`.B` pattern only
- GradNorm lambda renormalization preserves sum constraint
- NDCG/MRR/Recall/Precision on known ranking inputs, **including deliberately tied scores**: ties must never favour the target (`specs/eval-integrity` FR-2)
- Scores returned by evaluation are fp32 under mixed precision
- Under mixed precision, a forward after `optimizer.step()` or `load_state_dict()` reflects the new weights. This is a regression guard for the stale autocast cache, checked inside the real `train_one_step`.

**Integration tests:** Components interoperate correctly:
- Preprocessing → sequence generation → tensorisation pipeline produces valid datasets
- Transformer forward with MoE routing produces correct output shapes
- Full training step (forward + loss + backward + optimizer) converges loss on toy data
- Checkpoint save/load roundtrip preserves model state

**Smoke tests:** Real data, minimal scale:
- One-batch: single batch through full pipeline, loss is finite and decreasing
- Few-step T-Small: a capped run of `run_pretrain` on real MovieLens data
  (`--max_steps N`, a few hundred steps), which must complete the epoch loop, compute
  validation metrics, save a checkpoint and write the per-epoch log

**Owner decision (2026-09-17):** Phase 17 was previously "one-epoch T-Small: full epoch on
real MovieLens data". The full epoch is 30,447 steps (~2 h measured on the dev host) and
what it actually gates is that `run_pretrain` works end to end on real data, which a few
hundred steps establishes at the same confidence. REV-066 correctly refused to let BUILD
relax this gate on its own; this change is the spec owner's, made deliberately and
recorded here so the next review reads it as such and not as a repeat of REV-066.

**What the few-step gate does NOT establish:** training-pipeline AC-04 ("training loss
decreases over epochs on real data"). Under LR warmup a short window shows loss *rising*
(measured: 100 steps 26.23 -> 28.51; 400 steps 27.32 -> 29.15), so AC-04 is only
observable over a real multi-epoch run and remains outstanding until full training. AC-04
is unchanged; the gate no longer claims to cover it.

## Dev-Host Test Hygiene (owner, 2026-10-15)

The full suite writes ~16 GB to pytest's temp dir and has filled the Mac's disk twice. Therefore:

- **Run targeted test files only**: `uv run pytest tests/<file>.py -m "not slow" -q` (narrow further
  with `-k`). Never run `pytest` without a path on the dev host. Check `df -h .` before anything
  that writes data.
- **New tests use tiny synthetic data** — in-memory tensors, or the shared tiny-dataset fixture in
  `tests/conftest.py` (synthetic raw CSVs run through the real `run_preprocessing`, < 20 MB).
  Never copy `data/processed/` or any real artifact into a test.
- The former disk hog is fixed (owner-approved, `specs/ablation-run` T0.2):
  `tests/test_training.py::_build_subset_processed_dir` now subsets the int32 NumPy
  `*_examples.pkl` arrays too, so one call writes ~57 MB instead of ~2.3 GB. Its `slow` tests
  still read the real `data/processed/`, so the targeted-files rule above stands.
- Always `uv run python` / `uv run pytest`.
- GPU-only behaviour (FlashAttention-3 kernels) is tested on the dev host through pure-torch
  reference stand-ins; tests that need the real kernel are `skipif`-gated on capability and import,
  and are run on the target machine.

## Critical Assertions (Runtime)

These assertions must be active during all training/evaluation:
- `train_user_history is not eval_user_history`
- `cold_item_set.isdisjoint(train_interacted_items)`
- GradNorm: `sum(active_lambdas) == num_active_objectives` within tolerance
- LoRA state dict: no keys starting with `moe.router`, `moe.experts`, `moe.objective_embeddings`, `moe.gates`
- Federated: global frozen-state checksum unchanged through every round
- All losses, lambdas, and gradients are finite
- Evaluation scores are fp32 (`score_candidates_batch` asserts it); ranking metrics reject NaN scores

## Seed and Reproducibility

- Dev/smoke: single seed `42`
- Final paper: independent seeds `[42, 123, 456]`
- All random state sources seeded: `torch.manual_seed`, `np.random.seed`, `random.seed`
- `torch.backends.cudnn.deterministic = True`
- Worker seeding function for DataLoader when `num_workers > 0`
- `num_workers > 0` must yield the same batch sequence as `num_workers = 0` for the same seed
  (tested); `persistent_workers` stays off for training (see `code-quality.md`)
- An experiment's **data** sample has its own seed (`user_sample_seed`), independent of the
  training `seed`, so re-seeding training never changes which users are in the dataset
- Resume from `last.pt` must reproduce an uninterrupted run exactly on CPU (tested); on CUDA it
  reproduces the state, LR schedule and data order (kernels with atomics are not bit-deterministic)
