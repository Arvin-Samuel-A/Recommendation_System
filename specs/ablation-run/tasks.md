# Ablation Run — Tasks

Ordered. Depends on every implemented spec (data-pipeline, transformer-backbone, training-pipeline,
ablations-cli). Each task lists the requirement/design it implements, what to change, and a
**Done when** condition verifiable on the Mac (no CUDA). Tasks marked *(target machine)* in an AC are
verified later by the owner, not by BUILD.

## Rules for BUILD on this feature (read before T0.1)

- **Never run the whole test suite.** It writes ~16 GB to the temp dir and has filled the disk
  twice. Run only the files a task names, as `uv run pytest tests/<file>.py -m "not slow" -q`
  (add `-k` to narrow further). Do not run `-m slow` tests unless a task says so.
- New tests use the tiny synthetic fixture from T0.1 or in-memory tensors. **Never copy
  `data/processed/`** (or any real artifact) into a test.
- Always `uv run python` / `uv run pytest`. Before any real-data command (T9.2), check
  `df -h .` shows ≥ 15 GB free; stop and report if not.
- No multi-hour runs. Real-data runs are capped (`--max_steps`, ≤ 2 epochs) and only in T9.2.
- `tests/test_preprocessing.py::TestUserSampling` already exists, uncommitted, written ahead of this
  spec. It is the target interface for T1.1/T1.2/T2.1 — make it pass; do not weaken it.
- T3.2 (golden) must be committed to the working tree and green **before** starting T5.1. Record in
  its BUILD note the `git diff --stat` at the moment the constants were captured.
- Do not commit or push; the orchestrator does that.

## Phase 0 — Test infrastructure

- [x] **T0.1: Tiny synthetic processed-data fixture** — Create `tests/conftest.py` with a
  session-scoped factory fixture `tiny_processed_dir(**overrides)` and a helper
  `tiny_config(processed_dir, out_dir, **overrides)` exactly as design D14 describes: six synthetic
  raw CSVs (~60 users × ~150 items, string timestamps, items 141–150 only as some users' final two
  interactions so the `unseen_in_training` cold set is non-empty), then the real
  `data.preprocessing.run_preprocessing` with `max_seq_length=8`, `sequence_length_grid=[4, 8, 16]`,
  `num_sampled_negatives=10`, `seed=42`, `device="cpu"`; outputs under `tmp_path_factory`. Cache one
  directory per distinct override set. Add `tests/test_fixture_tiny.py` asserting the directory
  contains every artifact `run_pretrain` reads, `dataset_stats.json["num_users"] > 0`, the cold set
  is non-empty, and the whole directory is < 20 MB. (D14; supports AC-11, AC-15–17, AC-21–23, AC-26–28)
  - **Done when:** `uv run pytest tests/test_fixture_tiny.py -q` passes in < 60 s; no file under
    `data/` is read or written by the fixture (assert `raw_data_dir`/`processed_data_dir` are under
    the temp root).
  - *BUILD note:* 4 passed in 0.49 s. Fixture: 60 users, 150 items (10 cold), processed dir
    ~0.35 MB. `tiny_config` also lives in `tests/conftest.py` (import as
    `from tests.conftest import tiny_config`).

- [x] **T0.2: Stop the slow-test helper copying ~2 GB per call** *(owner-approved extra,
  added by BUILD 2026-10-15; requirements "Out-of-scope findings" first bullet)* — In
  `tests/test_training.py::_build_subset_processed_dir`, subset `*_examples.pkl` values that are
  NumPy arrays (they have been int32 arrays since REV-086), not only lists, so each call writes
  the first `n` rows instead of the full arrays. The subset rows are the same rows the tensorised
  splits and the candidate matrices are cut to, so example `i` still matches candidate row `i`.
  No other helper had the list-only pattern (`grep "isinstance(v, list)" tests/`).
  - **Done when:** one call writes < 100 MB (was ~2.3 GB) and the non-slow test that uses it
    passes.
  - *BUILD note:* measured one default call (`n_train=512, n_val=128, n_test=128`): 57.2 MB total,
    `train_examples.pkl` 115 KB (shape `(512, 50)`), `val/test_examples.pkl` 29 KB each; before the
    fix those three files were copied whole (1.75 GB + 0.22 GB + 0.24 GB). The remaining ~51 MB is
    the two user-history pickles, kept as full copies on purpose (tests rely on them being real,
    distinct files; a hardlink would let an in-place write reach `data/processed`).
    `uv run pytest tests/test_training.py -m "not slow" -k build_once -q` → 1 passed.

## Phase 1 — Config, CLI, fingerprint

- [x] **T1.1: New config fields and validation** — In `configs/config.py` add `min_lr_ratio: float =
  0.1`, `user_sample_fraction: float = 1.0`, `user_sample_seed: int = 42`,
  `resume_training: bool = True`, each with a short comment naming this spec. `validate()` rejects
  `min_lr_ratio` outside `[0, 1]` and `user_sample_fraction` outside `(0, 1]`, naming the field.
  Do **not** touch the attention-backend enum here (T4.2). (FR-5, FR-6, FR-8; AC-14, AC-15)
  - **Done when:** `uv run pytest tests/test_preprocessing.py -k "invalid_fraction" -q` passes and a
    new `tests/test_cli.py::TestAblationRunConfig` asserts the four defaults, that
    `Config().dtype == "bf16"` and `Config().mixed_precision is False` (AC-01 default half), and
    the four rejection cases (`min_lr_ratio` −0.1 / 1.1, fraction 0 / 1.5).
  - *BUILD note:* `-k "invalid_fraction or test_cli_moves"` 2 passed;
    `tests/test_cli.py::TestAblationRunConfig` 10 passed (adds boundary-acceptance cases).

- [x] **T1.2: CLI flags and derived processed directory** — In `main.py` add `--user_sample_fraction`
  (float), `--user_sample_seed` (int), `--sequence_length_grid` (`nargs="+"`, int),
  `--attention_backend` (choices = `models.transformer.VALID_BACKENDS`, imported, so T4.2's rename
  flows through), `--num_workers` (int), `--min_lr_ratio` (float), `--no_resume`
  (store_true → `resume_training=False`). All default to `None`/`False` so config values stay
  authoritative, and each applied override is logged `CLI override: field old -> new` through the
  existing list. Implement the FR-6 directory derivation as a module-level function
  `derive_processed_dir(config) -> Optional[str]` (reused by T8.1), applied at the end of
  `apply_cli_overrides` (and logged as an override) only when `processed_data_dir` equals the
  `Config` default. (FR-6, FR-7, FR-8; D6; AC-15)
  - **Done when:** `uv run pytest tests/test_preprocessing.py::TestUserSampling::test_cli_moves_a_subset_to_its_own_processed_dir -q`
    passes, and new `tests/test_cli.py` tests cover: each flag applied and logged; absent flags leave
    config untouched; `--user_sample_fraction 0.25 --user_sample_seed 7` →
    `data/processed_users25pct_seed7`; an explicitly non-default `processed_data_dir` is never
    rewritten; `--no_resume` → `resume_training is False`.
    Run: `uv run pytest tests/test_cli.py -m "not slow" -q`.
  - *BUILD note:* `tests/test_cli.py -m "not slow"` 63 passed (incl. new `TestAblationRunCLI`,
    7 tests); `TestUserSampling::test_cli_moves_a_subset_to_its_own_processed_dir` passed.
    `derive_processed_dir` derives from the *default's* parent (`data/`), so the result is
    `data/processed_users{pct}pct[_seed{S}]`.

- [x] **T1.3: Training-fingerprint hygiene** — In `training/ablation_runner.py` add
  `_RUNTIME_ONLY_FIELDS = frozenset({"num_workers", "pin_memory", "resume_training"})` and exclude it
  in `_training_fingerprint`, with a comment citing design D4/D13 (why `persistent_workers` stays
  in). (FR-13; D13; AC-29)
  - **Done when:** a new `tests/test_ablation_robustness.py::TestFingerprint` asserts the fingerprint
    is unchanged by `num_workers`, `pin_memory`, `resume_training` and changed by
    `persistent_workers`, `min_lr_ratio`, `user_sample_fraction`, `user_sample_seed`,
    `attention_backend`; `uv run pytest tests/test_ablation_robustness.py -q` passes.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py -q` → 8 passed (`TestFingerprint`: 3
    unchanged-by, 5 changed-by cases).

## Phase 2 — Data: sampling, stored width, narrowing, guards

- [x] **T2.1: Seeded user sampling** — Implement D6 in `data/preprocessing.py`: sampling in
  `_read_ratings` after the min-interactions filter, the optional `sampling_info` dict,
  `process_ratings` merging it into `stats`/`dataset_stats.json`, `run_preprocessing` calling
  `config.validate()` first and copying the five fields into `preprocessing_manifest.json`.
  (FR-6; D6; AC-15)
  - **Done when:** `uv run pytest tests/test_preprocessing.py -m "not slow" -q` passes (including
    all of `TestUserSampling`), plus new tests on `tiny_processed_dir(user_sample_fraction=0.5)`:
    the five fields are in both JSON files with correct counts; every user id in `splits.pkl`,
    `train_user_history.pkl` and `test_examples.pkl` maps to a sampled original id (via
    `user_mapping.pkl`); `num_items` equals the number of distinct movies those users rated after
    filtering; and with `user_sample_fraction=1.0` `_read_ratings` never constructs a
    `RandomState` (monkeypatch it to raise) and returns every valid user.
  - *BUILD note:* `uv run pytest tests/test_preprocessing.py -m "not slow" -q` → 45 passed (all 5
    `TestUserSampling` + 4 new `TestUserSamplingOnTheTinyFixture`). One pre-existing test wrapper
    (`TestSingleRatingPass::...counting`) now forwards `*args, **kwargs`, because `_read_ratings`
    gained D6's optional `sampling_info` argument; its assertion is unchanged.
    Interpretation: the five sampling fields are written to both JSON files for every run, with
    `user_sample_fraction: 1.0`, equal before/after counts and `user_sample_fingerprint: null`
    for the full dataset. Every data artifact is unchanged at 1.0 (the sampling block is skipped,
    as tested with `RandomState` booby-trapped); only the two JSON records gain these fields.

- [x] **T2.2: Store at the widest grid length; width guard on load** — Implement D7's
  `stored_width` in `run_preprocessing` / `generate_sequences(..., width=None)`, the manifest's
  `stored_seq_length`, and the `ValueError` in `load_tensorised` when the requested width exceeds
  the stored width. (FR-7; D7; AC-16, AC-19)
  - **Done when:** new tests in `tests/test_sequence_generator.py` pass: on the tiny fixture every
    tensorised split and every `*_examples.pkl` has width 16 and the manifest says 16; for each split,
    `load_tensorised(..., max_seq_length=8)` sequences and lengths equal those of
    `generate_sequences(..., width=8)` on the same `splits.pkl` input (`torch.equal`); requesting 17
    raises `ValueError` mentioning `--stage preprocess` and `sequence_length_grid`. Run:
    `uv run pytest tests/test_sequence_generator.py -m "not slow" -q`.
  - *BUILD note:* `uv run pytest tests/test_sequence_generator.py -m "not slow" -q` → 34 passed (new
    `TestStoredWidth`, 4 tests: width 16 for all three tensorised splits and `*_examples.pkl`,
    manifest `stored_seq_length: 16`; `load_tensorised(..., 8)` and `load_examples(..., 8)` equal
    `generate_sequences(..., width=8)` on `splits.pkl` for every split, sequences and lengths;
    width 17 raises naming `--stage preprocess` and `sequence_length_grid`). The width rule lives
    in `data.sequence_generator.stored_seq_length(config)`. `narrow_examples`/`load_examples` were
    added here rather than in T2.3 because the AC-16 equivalence covers `*_examples.pkl` too.

- [x] **T2.3: Narrow every reader** — Add `narrow_examples` / `load_examples` to
  `data/sequence_generator.py` (D7), then route every reader through narrowing: `main.py`
  (:276 evaluate, :530 routing diagnostics, :572 gradient diagnostics), `training/federated.py`
  (:134 val tensorised, :177 test examples, :187 val examples), `training/local_train.py` (:229,
  :238), `evaluation/cold_start.py` (:64), `evaluation/longtail.py` (:119), and
  `scripts/estimate_training_time.py` (:232-236 hardcoded 50 → `cfg.max_seq_length`). Line numbers
  are as of HEAD `0a91094`. (FR-7; D7; AC-17)
  - **Done when:** (1) a new static test `tests/test_sequence_generator.py::TestEveryReaderNarrows`
    (AST walk over `git ls-files '*.py'` minus `tests/`) fails on HEAD's code and passes after the
    change — record the failing names it printed before the fix in the BUILD note; (2) a runtime test
    on the tiny fixture (stored width 16) with a t-small model built at `max_seq_length=8` runs
    `run_evaluate_stage`, `run_routing_diagnostics_stage`, `run_gradient_diagnostics_stage`
    (`grad_diag_num_batches=1`), `run_local_personalization` (`personalization_eval_users=2`,
    `local_epochs=1`), `run_federated` (`federated_rounds=1`, `clients_per_round=2`), cold-start and
    long-tail evaluation without error, against a checkpoint produced by a 2-step `run_pretrain`
    on the same fixture. Run: `uv run pytest tests/test_sequence_generator.py -m "not slow" -q`.
  - *BUILD note:* (1) `TestEveryReaderNarrows::test_static_every_reader_goes_through_narrowing` run BEFORE the
    reader changes printed 10 offenders (line numbers as of this working tree, which had grown
    `main.py` by the T1.2 flags):
    `evaluation/cold_start.py:64`, `evaluation/longtail.py:119` (open `*_examples.pkl`);
    `main.py:355` (evaluate), `main.py:609` (routing diagnostics), `main.py:651` (gradient
    diagnostics) (`load_tensorised` without `max_seq_length`); `training/federated.py:134`
    (`load_tensorised`), `:177`, `:187` (open `*_examples.pkl`); `training/local_train.py:229`,
    `:238` (open `*_examples.pkl`). After the change it passes; a second test proves the audit
    flags both defect shapes (incl. an f-string `{s}_examples.pkl`). The estimate script's
    hardcoded `max_seq_length=50` (not an audit shape) now reads `cfg.max_seq_length`, cached
    per width.
    (2) `TestNarrowedReadersRunEndToEnd` (2-step `run_pretrain` on the width-16 fixture, model
    width 8, then evaluate, routing diagnostics, gradient diagnostics, personalization (2 users),
    federated (1 round, 2 clients), cold-start and long-tail) passes in 37 s. Sanity check that it
    is load-bearing: with `cold_start.load_examples` swapped for a raw unpickle, cold-start raised
    `IndexError: index out of range in self`.
    `uv run pytest tests/test_sequence_generator.py -m "not slow" -q` → 37 passed.

- [x] **T2.4: Position guard in the model** — `TransformerModel.forward` raises `ValueError` naming
  both widths when `seq.shape[1] > position_embedding.num_embeddings` (D7). (FR-7; AC-18)
  - **Done when:** a new test in `tests/test_transformer.py` passes for `(2, max_seq_length + 1)`
    and a width-`max_seq_length` input still runs; `uv run pytest tests/test_transformer.py -m "not slow" -q` passes.
  - *BUILD note:* `uv run pytest tests/test_transformer.py -m "not slow" -q` → 32 passed (new
    `TestPositionGuard`: `(2, S+1)` raises `ValueError` naming `S+1` and `max_seq_length=S`;
    width `S` still runs). The check reads only `seq.shape`, so it adds no device sync.

- [x] **T2.5: Processed-directory consistency check** — Implement
  `data.preprocessing.check_processed_dir(config)` (D6) and call it at the top of `run_pretrain`
  (after `validate`) and at the top of `run_ablation`. (FR-7; D6; AC-20)
  - **Done when:** tests in `tests/test_ablation_robustness.py` show: a tiny dir built with
    fraction 0.5 used by a config with fraction 1.0 (and with a different `user_sample_seed`) makes
    both `run_pretrain` and `run_ablation` raise `ValueError` before any training step (spy on
    `train_one_step`); a dir without a manifest logs a warning and proceeds.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py -q` → 12 passed (new `TestProcessedDirCheck`:
    fraction-0.5 dir used with fraction 1.0, and with fraction 0.5 / seed 7, makes both
    `run_pretrain` and `run_ablation(..., "B")` raise `ValueError` naming `--stage preprocess`
    with a `train_one_step` spy at 0 calls; matching sample accepted; a copy of the tiny dir
    without its manifest logs the warning and trains).

## Phase 3 — LR floor, then freeze behaviour with a golden test

- [x] **T3.1: Cosine LR floor** — Implement D5 in `training/pretrain.py::build_scheduler`. Replace
  `tests/test_training.py::TestLRScheduler::test_cosine_reaches_near_zero` with
  `test_cosine_decays_to_floor` (final LR == `min_lr_ratio × lr`, rel 1e-6; no post-warmup LR below
  it), and add `test_zero_floor_matches_previous_schedule` (inline the old lambda in the test and
  compare every step for `total_steps=100` exactly) and a fingerprint-changes test. Confirm by
  reading `training/local_train.py` and `training/federated.py` that no scheduler exists there and
  say so in the BUILD note. (FR-5; D5; AC-14)
  - **Done when:** `uv run pytest tests/test_training.py -m "not slow" -k "LRScheduler" -q` passes.
  - *BUILD note:* `uv run pytest tests/test_training.py -m "not slow" -k "LRScheduler" -q` → 5 passed
    (`test_cosine_decays_to_floor` replaces `test_cosine_reaches_near_zero`;
    `test_zero_floor_matches_previous_schedule` compares every step 0..100 with `==` against the
    HEAD lambda inlined; `test_floor_is_part_of_the_training_fingerprint`). Read-check: the only
    optimizer in `training/local_train.py` is `AdamW(lora_params, lr=config.lora_lr, ...)` at
    :113 with no scheduler; `training/federated.py` builds no optimizer or scheduler of its own
    (it calls `train_user_lora`). No `LambdaLR`/`lr_scheduler` in either file.

- [x] **T3.2: Golden characterisation test (before any FR-4/FR-8 change)** — Add
  `tests/test_determinism.py::test_golden_tiny_run`: `set_seed(42)`, `torch.set_num_threads(1)`,
  `run_pretrain` on the tiny fixture with a fully pinned config (t-small, `pretrain_epochs=2`,
  `max_steps_per_epoch=6`, `batch_size=16`, `num_workers=0`, contrastive and GradNorm on,
  `min_lr_ratio=0.1`, pure bf16 on CPU), capturing via a thin spy the per-step loss values and LRs;
  assert SHA-256 of the concatenated final parameters (model, MoE, loss fns, in `state_dict` order,
  as raw bytes), the loss trace (rounded to 7 significant digits) and the LR trace equal recorded
  constants. Record `torch.__version__` and `platform.machine()`; `pytest.skip` on mismatch. Run it
  twice in one session to show it is deterministic. (FR-4; D4; AC-11)
  - **Done when:** `uv run pytest tests/test_determinism.py -q` passes, the constants are in the
    file, and the BUILD note records the `git diff --stat` at capture time showing no FR-4/FR-8 code
    change yet.
  - *BUILD note:* Constants captured with `GOLDEN_CAPTURE=1` (both parametrised repeats printed byte-identical
    values in one session) and recorded in `tests/test_determinism.py`; then
    `uv run pytest tests/test_determinism.py -q` → 2 passed (31.9 s). Captured on torch 2.14.0 /
    arm64 (skips elsewhere). Hash covers model, MoE and loss-fn `state_dict`s plus GradNorm
    `log_lambdas` (one more than the spec lists). `git diff --stat` at capture, code files only:
    `configs/config.py +21`, `data/preprocessing.py +115`, `data/sequence_generator.py +77`,
    `evaluation/cold_start.py 8`, `evaluation/longtail.py 8`, `main.py +87`,
    `models/transformer.py +10`, `scripts/estimate_training_time.py 15`,
    `training/ablation_runner.py 14`, `training/federated.py 17`, `training/local_train.py 13`,
    `training/pretrain.py 13` (= T2.5 `check_processed_dir` call + T3.1 LR floor only);
    `training/gradnorm.py` and `utils.py` untouched. No FR-4/FR-8 change existed yet.
  - *BUILD note:* See T5.1 note: the constants above were re-captured once (hash `a0cef99f…`), on a tree with no
    FR-4/FR-8 code, after T4.2's probe was made RNG-neutral; the first capture (`3e5e596f…`) is
    superseded.

## Phase 4 — FlashAttention-3

- [x] **T4.1: Unpad/repad and the varlen attention core** — In `models/transformer.py` add
  `VarlenMeta`, `varlen_meta`, `unpad`, `repad`, `varlen_attention(..., varlen_fn, meta=None)` and
  the per-`padding_mask` meta cache on `AttentionBackend` (D2). Pure torch; no FA import. (FR-2; D2;
  AC-05)
  - **Done when:** new `tests/test_transformer.py::TestVarlenCore` passes on CPU: roundtrip
    (real positions equal, pads exactly 0); `cu_seqlens == [0, S, S+5, S+6]` and `max_seqlen == S`
    for lengths {S, 5, 1}; `all_real` true for an unpadded mask; with a per-sequence SDPA reference
    `varlen_fn`, output and `q/k/v` gradients equal masked SDPA at real positions (fp32, atol 1e-5);
    the reference receives `causal=True`; a 3-layer forward through `TransformerModel` with a
    counting `varlen_meta` spy computes it once.
  - *BUILD note:* `tests/test_transformer.py::TestVarlenCore` → 5 passed on CPU: roundtrip (real positions
    equal, pads exactly 0); `cu_seqlens == [0, 8, 13, 14]` (int32), `max_seqlen == 8` for lengths
    {8, 5, 1}; `all_real` for an unpadded mask; fp32 output and q/k/v gradients equal masked
    SDPA at real positions (atol 1e-5) and the reference received `causal=True`; a 3-layer
    t-small `TransformerModel` forward computes `varlen_meta` once (twice for two forwards).
    Ordering note: the "3-layer forward through `TransformerModel`" check can only reach the
    varlen path through the FA3 dispatch, so T4.1's code and T4.2's dispatch were written in one
    edit and T4.1 was verified after it. `VarlenMeta.indices` is `None` when `all_real` (no
    second sync); `cu_seqlens` is always computed.

- [x] **T4.2: Replace `flash_attention_4` with `flash_attention_3`** — Implement D2's `_import_fa3`,
  `_fa3_output`, dtype check, dense/varlen dispatch (varlen via T4.1 with FA3's varlen function),
  `preferred_backend(device_type, capability)`, probe head dim from config, `VALID_BACKENDS =
  ("auto", "flash_attention_3", "sdpa")`, and the same set in `Config.validate()` (message names the
  three choices only). Delete `_flash_attention_4*`. Migrate the FA4 tests in
  `tests/test_transformer.py` (`TestAttentionBackendGate`, `TestNarrowedBackendSurface`,
  `TestFlashAttentionVarlen`, `TestCausalIntentReachesEveryBackend`) to a reference
  `flash_attn_interface` module (two variants: returns `out`; returns `(out, lse)`). Add a
  `skipif`-gated real-kernel test (CUDA capability 9.x and importable FA3) that compares FA3 to SDPA
  on a left-padded batch. Update docstrings/comments that say FA4. (FR-2; D2; AC-02–AC-07)
  - **Done when:** `uv run pytest tests/test_transformer.py -m "not slow" -q` passes with the
    real-kernel test reported as skipped; `grep -rn "flash_attention_4\|from flash_attn import\|import flash_attn$" --include=*.py configs data models training evaluation main.py utils.py scripts`
    returns nothing (tests may still name `flash_attention_4` in the rejection test); a test proves
    `import models.transformer` and an `sdpa` model build work with
    `sys.modules["flash_attn_interface"] = None`.
  - *BUILD note:* `uv run pytest tests/test_transformer.py -m "not slow" -q -rs` → 53 passed, 1 skipped (the
    skip is `TestRealFlashAttention3Kernel`: "needs CUDA capability 9.x and
    flash_attn_interface"). Migrated: `TestAttentionBackendGate` (FA3 pinned as the preferred
    kernel), `TestNarrowedBackendSurface` (message names the three choices),
    `TestFlashAttentionVarlen` → `TestVarlenCore` + `TestFlashAttention3Backend` (reference
    `flash_attn_interface` returning `out` and `(out, lse)`; dense call for an all-real batch;
    varlen call with `cu_seqlens [0, 1, 9]` for rows of length 1 and 8; fp32 raises naming
    `float32`; probe runs at head dim 80 for t-base; mocked Hopper + FA3 missing → `sdpa` with a
    reason naming `flash_attention_3`, or `AttentionBackendError` when required; explicit FA3
    without the package raises), `TestCausalIntentReachesEveryBackend` (FA3 branch in bf16).
    AC-04 is a fresh-interpreter subprocess with `sys.modules["flash_attn_interface"] = None`
    (import + sdpa build). The task's grep returns nothing (exit 1). The FA3 dtype check runs
    before the import, so an fp32 call names the dtype even where FA3 is installed.

- [x] **T4.3: Record the runtime backend per arm** — In `_run_single_config` add the
  `result["runtime"]` block (D2 "Runtime record"). (FR-2; AC-30)
  - **Done when:** a test in `tests/test_ablation_robustness.py` runs one arm of a study on the tiny
    fixture (`max_steps_per_epoch=2`, `pretrain_epochs=1`) and finds `runtime.attention_backend ==
    "sdpa"`, `runtime.device == "cpu"` and a non-empty `runtime.device_name` in the result.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py -q` → 13 passed. `TestRuntimeRecord` runs
    study B's first arm on the tiny fixture (1 epoch, 2 steps): `runtime.attention_backend ==
    "sdpa"`, `runtime.device == "cpu"`, non-empty `device_name` (`platform.processor()`, here
    "arm"), plus `dtype`/`mixed_precision`/`attention_fallback_reason` per D2.

- [x] **T4.4: Backend verification script** — Add `scripts/verify_attention_backend.py` per D3
  (exit codes 0/1/2, JSON report keys as listed). (FR-3; D3; AC-08)
  - **Done when:** new `tests/test_scripts.py` runs it as a subprocess on the Mac:
    `--backend sdpa --model t-base --seq-lens 50 --report <tmp>` exits 0 and the report has every key
    with `passed: true`; `--backend flash_attention_3` exits 2 and prints that FA3 is unavailable.
    Run: `uv run pytest tests/test_scripts.py -q`.
  - *BUILD note:* `uv run pytest tests/test_scripts.py -q` → 2 passed (sdpa `--seq-lens 50` exits 0, report has
    every D3 key, 4 cases all `passed: true`; `--backend flash_attention_3` exits 2 and prints
    "FlashAttention-3 is unavailable on this machine: ..."). By hand on the Mac (MPS, defaults
    head dims 64,80 × lengths 50,200): exit 0, max out diff 7.6e-3, max grad diff 2.2e-2 against
    fp32 SDPA (passes `atol=2e-2, rtol=2e-2`; recorded in the report's `tolerance`). Extra
    report keys: `model`, `self_test`, `exit_code`, `lengths` per case, `python`, `generated_at`.

- [x] **T4.5: FA3 source and install scripts** — Add `scripts/fetch_fa3_source.sh` and
  `scripts/install_fa3.sh` per D3 (preflight with named failures, `--dry-run`, skip-if-installed,
  flag-name check against `hopper/setup.py`, detached-friendly logging, final verification call).
  Both `set -euo pipefail`, executable bit set. (FR-3; D3; AC-09)
  - **Done when:** tests in `tests/test_scripts.py` show `bash -n` passes for both;
    `bash scripts/install_fa3.sh <any path> --dry-run` on the Mac exits non-zero, names the failed
    check (CUDA/Hopper), and creates no build directory (assert with a `--build-dir` in `tmp_path`);
    the install script text invokes `scripts/verify_attention_backend.py --backend flash_attention_3`.
    `fetch_fa3_source.sh` is **not** executed in tests (network); its `--help` exits 0.
  - *BUILD note:* `uv run pytest tests/test_scripts.py -q` → 7 passed: `bash -n` + executable bit +
    `set -euo pipefail` for both; `fetch_fa3_source.sh --help` exits 0 (never executed
    otherwise); `install_fa3.sh <missing tarball> --dry-run --build-dir <tmp>` with
    `PYTHON=<venv python>` exits 2 with "PREFLIGHT FAILED: CUDA -- torch.cuda.is_available() is
    False ..." and creates no build dir; the script ends by running
    `scripts/verify_attention_backend.py --backend flash_attention_3` and exits with its status.
    The test passes `PYTHON` because the Mac's bare `python3` has no torch (the preflight would
    otherwise stop one check earlier, at "torch"). The tarball's existence is checked only when a
    build is needed (after the skip-if-installed check). Not verifiable here: the build itself and
    the `hopper/setup.py` flag-name check (target machine).

## Phase 5 — Utilisation without changing training (golden must stay green)

- [x] **T5.1: Worker loading and CPU-side negatives** — In `run_pretrain`'s step loop attach
  negatives to the CPU batch, then move tensors with `non_blocking` when CUDA and pinned (D4). Do the
  same at the `diag`/gradient-diagnostics negative sampler call sites that receive device batches if
  any. Leave `persistent_workers` default False. (FR-4; D4; AC-10, AC-12)
  - **Done when:** in `tests/test_determinism.py`: `test_workers_yield_identical_batches` (two epochs
    over the tiny train split, `num_workers` 0 vs 2, `pin_memory=True`, every key `torch.equal`)
    passes; `test_cpu_side_negatives_identical` (same `RandomState` state, CPU vs "device" batch —
    on the Mac use a batch round-tripped through `.to("cpu")` copies plus, if MPS is available, an
    MPS batch) passes; a spy shows `run_pretrain` makes no `.cpu()` call on `targets`/`user_ids`
    inside the step loop; `test_golden_tiny_run` still passes; and a tiny `run_pretrain` with
    `num_workers=2` produces the golden parameter hash. Run: `uv run pytest tests/test_determinism.py -q`.
  - *BUILD note:* DEVIATION (golden re-captured once, before any FR-4 code was live). Starting T5.1 the
    golden failed at step 0. Bisected: with the T5.1 loop edits reverted it still failed, and
    with the probe's head dim set back to 16 it passed — T4.2's D2 probe change (head dim 16 →
    `hidden_dim // num_heads`) drew more numbers from the GLOBAL torch RNG during
    `AttentionBackend` construction, which precedes weight init, so the same seed initialised
    different weights. T4.2 sits between T3.2 and T5.1 in the task order, so the spec's own
    ordering broke the golden. Fix (implementation-level): `_probe` now draws q/k/v from its own
    seeded CPU generator, so construction never consumes the global RNG (this also stops `auto`
    fallbacks, which probe twice, from initialising differently to an explicit backend). The
    golden was then re-captured on a tree with ZERO FR-4/FR-8 code — `training/pretrain.py` back
    to its 13-line T2.5+T3.1 diff, `evaluation/gradient_diagnostics.py` at HEAD — both repeats
    identical, new hash `a0cef99f…18b5b` (LR trace unchanged); then the T5.1 edits were re-applied.
    T5.1: negatives attached to the loader's CPU batch, then one `.to(device, non_blocking=cuda
    and pin_memory)`; `attach_training_negatives` reads host tensors through `_host_numpy`
    (zero-copy view on CPU); gradient diagnostics also sample before the device copy.
    `uv run pytest tests/test_determinism.py -q` → 6 passed: golden ×2,
    `test_workers_yield_identical_batches` (2 epochs, 0 vs 2 workers + `pin_memory=True`, every
    key `torch.equal`), `test_cpu_side_negatives_identical` (CPU, `.to("cpu")` copy and MPS
    batch, same `RandomState(1234)`), `test_step_loop_samples_negatives_before_the_device_copy`
    (on MPS: the sampler always received CPU `targets`/`user_ids`), and
    `test_golden_hash_with_two_workers` (golden hash, losses and LRs with `num_workers=2`).

- [x] **T5.2: Fewer per-step host syncs** — Implement D4's batched finite checks (A and B), 0-d
  tensor loss values materialised once per epoch, removal of per-step `assert_finite_step` calls,
  and branch-free GradNorm renormalisation in `training/gradnorm.py`. Update
  `scripts/estimate_training_time.py` only if it breaks. (FR-4; D4; AC-11, AC-13)
  - **Done when:** `test_golden_tiny_run` still passes; new tests pass: a NaN-producing loss (patch
    a loss fn to return `nan`) raises `AssertionError` naming that loss and every parameter is
    `torch.equal` to before the step; a non-finite grad norm raises; non-finite lambdas raise;
    vectorised renorm `torch.equal` to the old loop (inlined in the test) over 100 random lambda
    vectors and active subsets. Run: `uv run pytest tests/test_determinism.py tests/test_losses.py tests/test_training.py -m "not slow" -q`.
  - *BUILD note:* `uv run pytest tests/test_determinism.py tests/test_losses.py tests/test_training.py -m "not
    slow" -q` → 72 passed (golden ×2 and golden-with-workers still green). New:
    `TestBatchedFiniteChecks` (NaN `multi_step` raises "multi_step loss is not finite" and every
    model/MoE/loss-fn parameter is `torch.equal` to before; an inf grad norm raises "Non-finite
    gradient" without stepping; NaN lambdas raise "lambdas not finite" BEFORE the optimizer step —
    they used to be checked after it, in `assert_finite_step`; the step returns 0-d tensors that
    still work with `math.isfinite`/`format`) and `TestVectorisedRenormalisation` (100 random
    lambda vectors × random active subsets, `torch.equal` to the HEAD loop inlined). The renorm
    is now `GradNormController._renormalize` and keeps the old values via `torch.where` when
    renorm ≤ 0, as the branch did. `_skipped` entries are no longer averaged per epoch (they were
    a list of names; averaging them would have raised). `scripts/estimate_training_time.py` did
    not break and is unchanged by this task. The pre-existing `assert_finite_step` tests still pass.
  - *BUILD note:* Follow-up (found during the final audit): the slow, real-data
    `tests/test_training.py::TestFewStepsSmoke::test_few_steps_t_small` called `np.isfinite` on the
    step's loss value, which is now a 0-d tensor — on MPS `np.isfinite(mps_tensor)` raises
    `TypeError` (confirmed with a one-liner). It now appends `float(...)`; assertion unchanged. Not
    run (slow test, hygiene rule); the non-slow `tests/test_training.py` → 26 passed after the edit.

## Phase 6 — Exact resume

- [x] **T6.1: Atomic checkpoint writes and `last.pt`** — Add `utils.atomic_torch_save`; use it in
  `save_checkpoint`; write `last.pt` with the D8 `resume_state` at the end of every epoch (after
  `best.pt`). (FR-8; D8; AC-23)
  - **Done when:** new `tests/test_resume.py` tests pass: after a 2-epoch tiny run `last.pt` exists
    with every `resume_state` key and `epoch == 1`; patching `torch.save` to write partial bytes then
    raise leaves the previous `last.pt` loadable and no file named `last.pt` is partial (and no
    `*.tmp-*` remains after a successful save); `test_golden_tiny_run` still passes.
    Run: `uv run pytest tests/test_resume.py tests/test_determinism.py -q`.
  - *BUILD note:* `uv run pytest tests/test_resume.py tests/test_determinism.py -q` → 8 passed (golden still
    green). `TestLastCheckpoint`: after a 2-epoch tiny run `last.pt` has every D8 `resume_state`
    key (+ RNG sub-keys, both loader generators), `epoch == 1`, `global_step == total_steps == 8`,
    and the full `save_checkpoint` payload; no `*.tmp-*` remains. With `torch.save` patched to
    write partial bytes and raise, `atomic_torch_save` over `last.pt`/`best.pt` and
    `save_checkpoint` over `last.pt` all raise, both files keep their size and reload with the
    same weights, and no temp file is left. `epoch_NNN.pt`/`best.pt`/`last.pt` all go through
    `utils.atomic_torch_save` (tmp + fsync + `os.replace`). Also saved (beyond D8): the global
    Python and NumPy RNG states.

- [x] **T6.2: Resume in `run_pretrain`** — Implement D8's resume flow: detection gated by
  `resume_training`, refusal on fingerprint / `total_steps` / `source_revision` mismatch, restoration
  of every object and RNG/generator state after construction, `range(start_epoch, …)`, completed-run
  short-circuit, log/JSONL hygiene (`setup_epoch_logger(..., append=...)`). (FR-8; D8; AC-21, AC-22)
  - **Done when:** in `tests/test_resume.py`: an uninterrupted 3-epoch tiny run vs. one interrupted
    by an exception injected at the first step of epoch 2 and then re-run → identical final
    parameter hash, `best_val_ndcg`, `best_epoch`, per-step LR trace, `scheduler.last_epoch ==
    global_step` at resume, one log line per epoch, no duplicate GradNorm steps; the same with
    `num_workers=2`; a changed `learning_rate`, changed `pretrain_epochs`, and a monkeypatched
    `get_source_revision` each raise with the checkpoint path in the message; `resume_training=False`
    retrains from scratch (spy counts all steps); re-running a completed run executes 0 steps and
    returns the same `best_val_ndcg`/`best_epoch`. `test_golden_tiny_run` still passes.
  - *BUILD note:* `uv run pytest tests/test_resume.py -q` → 12 passed (92 s); `tests/test_determinism.py` → 6
    passed (golden green after resume landed). `TestResume`: failure injected at the first step of
    epoch 2 (call 9 of 12), re-run → 4 calls only, `scheduler.last_epoch == 8` on entry, LR
    trace (8 + 4) == uninterrupted, final parameter hash, `best_val_ndcg`, `best_epoch` and
    epoch means identical, epoch log lines exactly `0,1,2`, GradNorm JSONL steps unique — for
    `num_workers` 0 and 2 (both compared to the 0-worker reference); the resume log line reads
    "Resuming t-small from … at epoch 1 (global step 4)"; changed `learning_rate` /
    `pretrain_epochs` ("training_fingerprint differs") and a patched `get_source_revision`
    ("source_revision.source_hash differs") are refused with the `last.pt` path and
    `--no_resume` in the message, and `total_steps` alone is refused at the loader; an unchanged
    copy in another dir resumes (non-vacuity); `resume_training=False` retrains all 12 steps to the
    same hash; a completed run trains 0 steps and returns the same results.
    Test-design finding: `raw_data_dir` is not a `_PATH_FIELDS` member, so it is part of the
    training fingerprint; the tests pin it. Not changed (fingerprint semantics are the spec's),
    but worth knowing: the same data preprocessed from a different raw path fingerprints apart.
    Note on GradNorm JSONL: lambdas are logged every 100 global steps, so a 12-step run has one
    record (step 0); the no-duplicates check is real but small.

## Phase 7 — Running a study robustly

- [x] **T7.1: OOM classification, exit 75, re-raise in the runner** — Add `utils.is_oom_error`,
  `utils.OOM_EXIT_CODE = 75`; wrap the stage call in `main.main`; re-raise OOM in `run_ablation`'s
  per-arm handler (D9). (FR-9; D9; AC-24)
  - **Done when:** tests in `tests/test_ablation_robustness.py` pass for the predicate cases in AC-24;
    `main.main(["--stage", "pretrain", ...])` with the stage monkeypatched to raise
    `torch.OutOfMemoryError("CUDA out of memory")` raises `SystemExit` with code 75, and with a
    `ValueError` re-raises the `ValueError`; `run_ablation` with `_run_single_config` patched to raise
    OOM on arm 2 propagates it, and with a `KeyError` records `status: "failed"` and runs arm 3.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py -q` → 24 passed. `TestOOMHandling`: the
    predicate is true for `torch.OutOfMemoryError`, "CUDA out of memory" and "MPS backend out of
    memory" `RuntimeError`s, false for another `RuntimeError`, host `MemoryError` and a
    `ValueError` mentioning OOM; `main.main(["--stage", "pretrain", ...])` with the stage patched
    to raise OOM → `SystemExit(75)`, with a `ValueError` → the `ValueError`; `run_ablation("D")`
    with `_run_single_config` raising OOM on arm 2 propagates it, with `KeyError` records arm 2 as
    `status: "failed"` and still runs arms 3 and 4. (`save_config_snapshot` is patched out in the
    `main` tests so nothing is written under `outputs/`.)

- [x] **T7.2: Retry launcher** — Add `scripts/run_with_retry.py` per D9. (FR-9; D9; AC-25)
  - **Done when:** `tests/test_scripts.py` runs it with `--wait-seconds 0` around small
    `uv run python -c` children driven by a counter file: exits 75 twice then 0 → wrapper exits 0
    after 3 attempts (attempt lines in output); always 75 with `--max-retries 2` → exits 75 after 3
    attempts; exits 3 → wrapper exits 3 after 1 attempt; SIGTERM to the wrapper terminates a sleeping
    child and the wrapper exits non-zero without retrying.
  - *BUILD note:* `uv run pytest tests/test_scripts.py -q -k RunWithRetry` → 4 passed: 75, 75, 0 → wrapper
    exits 0 after 3 attempts ("attempt 3/21" logged, no 4th); always 75 with `--max-retries 2` →
    exits 75 after 3 attempts; exit 3 → exits 3 after 1 attempt; SIGTERM to the wrapper while the
    child sleeps → wrapper exits 143, no "attempt 2", the child process is gone. Children are run
    with the venv's `sys.executable -c` (same interpreter `uv run python` resolves to) to keep the
    test fast. The script imports nothing from the repo (75 is mirrored), so it runs under any
    Python on the target machines.

- [x] **T7.3: Backbone store lock** — Add `backbone_lock` (D10) and move
  `find_backbone → run_pretrain → register_backbone` in `_run_single_config` under it with a re-check
  after acquisition. (FR-10; D10; AC-26)
  - **Done when:** tests in `tests/test_ablation_robustness.py`: a subprocess acquires the lock and
    waits on a sentinel file; `backbone_lock(..., timeout=0.5)` raises `TimeoutError`; the test then
    registers a backbone for that fingerprint, releases the subprocess, and a `_run_single_config`
    (with `run_pretrain` patched to fail if called) acquires the lock and reuses the backbone; the
    lock file contains the holder pid while held.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py -q` → 26 passed. `TestBackboneLock`: a
    subprocess holds the lock (the lock file's first field is its pid while held);
    `backbone_lock(..., timeout=0.5)` raises `TimeoutError`; a `_run_single_config` started in a
    thread (with `run_pretrain` patched to fail) blocks on the lock; the test then registers a
    backbone for that fingerprint and releases the holder; the waiter re-checks after acquiring,
    reuses it (`shared_backbone_with == "holder_process"`) and never calls `run_pretrain`. Extra:
    a SIGKILLed holder leaves no stale lock. Poll interval is `BACKBONE_LOCK_POLL_SECONDS` (5 s;
    the test shortens it). `--no_train` takes no lock (the lock is only entered when
    `train=True` and no backbone is registered). `tiny_config` now uses the fixture's own raw dir
    (not one per output dir), since `raw_data_dir` is part of the fingerprint.

- [x] **T7.4: Persist arms; seed-namespaced results** — Implement D11 in `run_ablation`
  (`_results_root`, `_arm_key`, reuse, atomic arm writes) and move study JSON/summary under
  `seed_{seed}/` (the superseding note in `specs/ablations-cli/requirements.md` FR-5 is already in
  place; update any test that asserted the old path). (FR-11; D11; AC-27)
  - **Done when:** tests in `tests/test_ablation_robustness.py` on the tiny fixture with study `B`
    (`max_steps_per_epoch=2`, `pretrain_epochs=1`, `ablation_eval_examples=32`): first run writes
    `seed_42/B_t_small.json` and two arm files; the second run calls neither `run_pretrain` nor
    `evaluate_model` (spies) and writes an identical study JSON apart from `reused_from_previous_run`;
    changing `ablation_eval_examples` re-runs both arms' evaluation but not training (store reuse);
    `resume_training=False` re-runs evaluation for both. Run:
    `uv run pytest tests/test_ablation_robustness.py tests/test_cli.py -m "not slow" -q`.
  - *BUILD note:* `uv run pytest tests/test_ablation_robustness.py tests/test_cli.py -m "not slow" -q` → 92
    passed. `TestArmPersistence` (tiny fixture, study B, 1 epoch × 2 steps,
    `ablation_eval_examples=32`): run 1 → `seed_42/B_t_small.json` + `arms/B_t_small/{input_only,
    objective_conditioned}.json`, `run_pretrain` ×2 and `evaluate_model` ×2; run 2 → 0 and 0, and
    the study JSON equals run 1's once `reused_from_previous_run` is dropped; changing
    `ablation_eval_examples` → `evaluate_model` ×2, `run_pretrain` ×0 (backbone store), 16 examples
    scored; `resume_training=False` → `evaluate_model` ×2 again. A changed `source_hash` changes the
    arm key. No existing test asserted the old flat path. Each persisted arm carries its `arm_key`
    (also in the study JSON); arm files and the study JSON are written atomically.

## Phase 8 — Packaging and documentation

- [x] **T8.1: Packaging and launch generator** — Add `scripts/package_ablation_run.py` per D12
  (shared command builder, manifest validation, `bash -n` of generated scripts, tracked-files code
  tarball + `launch/`, data tarball, printed one-liners, no hardcoded hosts). (FR-12; D12; AC-01,
  AC-28)
  - **Done when:** `tests/test_scripts.py` runs it against `tiny_processed_dir(user_sample_fraction=0.5)`
    with `--processed-dir` and `--out tmp_path`: exit 0; code tarball members == `git ls-files` ∪
    `launch/*` (no `data/`, `outputs/`, `.venv/`, `dist/`); data tarball contains the manifest;
    `run_h100.sh` runs the FA3 verification first then studies A, D, E, H in order, `run_aws.sh`
    activates `/opt/pytorch` and runs B, C, F, G; every ablation command has `--model t-base --seed 42
    --user_sample_fraction <f> --user_sample_seed 42 --dtype bf16 --mixed_precision --batch_size 256
    --attention_backend <machine backend> --num_workers 8` and goes through `run_with_retry.py`;
    `LAUNCH.md` contains both `setsid nohup bash launch/run_<m>.sh > logs/run_<m>.log 2>&1 < /dev/null &`
    one-liners; a manifest with a different fraction or `stored_seq_length` < 200 makes it exit
    non-zero.
  - *BUILD note:* `uv run pytest tests/test_scripts.py -q` → 14 passed. `TestPackageAblationRun`: exit 0;
    code tarball members == existing `git ls-files` ∪ `launch/{run_h100.sh,run_aws.sh,LAUNCH.md}`
    with nothing under `data/raw`, `data/processed*`, `outputs/`, `.venv/`, `dist/`; data tarball has
    `data/processed_users50pct/preprocessing_manifest.json`; `run_h100.sh` runs the FA3
    verification before `for STUDY in A D E H`, `run_aws.sh` sources `/opt/pytorch/bin/activate`
    and runs `B C F G`; every ablation command (scripts and LAUNCH.md) goes through
    `run_with_retry.py` and carries `--model t-base --seed 42 --user_sample_fraction <f>
    --user_sample_seed 42 --dtype bf16 --mixed_precision --batch_size 256 --attention_backend
    <machine backend> --num_workers 8`; LAUNCH.md has both `setsid nohup ... < /dev/null &`
    one-liners and `<H100_SSH>`/`<AWS_SSH>` placeholders; fraction 0.25 against a 0.5 dir and a
    stored width 16 < 200 both exit non-zero.
    Deviations: (1) the positive test builds the tiny fixture with the run's default grid
    (`sequence_length_grid=[25, 50, 100, 200]` → stored width 200) because the manifest check
    validates against the grid the launched runs will use; with the plain fixture (width 16) the
    packager correctly refuses, which is the negative test. (2) The data tarball is named and
    rooted by the DERIVED directory name (`processed_users{pct}pct`, what `main.py` resolves on
    the target), not the source dir's basename — they coincide for the default
    `--processed-dir`; with the tiny fixture (`.../processed`) D12's basename rule would ship data
    the launched commands could not find. (3) The packager warns that
    `scripts/run_with_retry.py`, `scripts/verify_attention_backend.py` and
    `scripts/install_fa3.sh` are not yet git-tracked (they are new, uncommitted files) — the code
    tarball only contains them after the orchestrator commits. Also added
    `data/processed_users*/` and `/logs/` to `.gitignore` so the sampled data and run logs are
    never committed.

- [x] **T8.2: README — CLI reference and runbook** — Add the new flags to the README CLI table and a
  short "Ablation run (10% users)" section: preprocess command, `package_ablation_run.py`, FA3
  install/verify, launch, monitor (`tail -f logs/...`), resume/retry behaviour, collecting
  `outputs/ablations/seed_42/`. Do not duplicate `LAUNCH.md`; link to the generator. (FR-3, FR-12)
  - **Done when:** every flag added in T1.2 appears in the README table, and every script path the
    README names exists (`ls` them).
  - *BUILD note:* All seven T1.2 flags are rows of the README CLI table (one match each; the table also gained
    the pre-existing but undocumented `--no_train`, `--seed`, `--dtype`, `--mixed_precision`,
    `--max_steps`, personalization overrides, and the exit-75 note). New section
    "Ablation run (10% users)": preprocess, FA3 fetch/install/verify, `package_ablation_run.py`,
    detached launch, monitor, resume/retry behaviour, collecting `outputs/ablations/seed_42/`;
    per-machine copy/extract details are left to the generated `launch/LAUNCH.md`. The
    "Ablation Results" table now shows the `seed_{seed}/` paths and `arms/`. Every script path
    the README names exists (`ls`: fetch_fa3_source.sh,
    install_fa3.sh, package_ablation_run.py, run_with_retry.py, verify_attention_backend.py).

## Phase 9 — Mac verification

- [x] **T9.1: Targeted test sweep** — Run, one file at a time, `uv run pytest <file> -m "not slow" -q`
  for: `test_fixture_tiny.py`, `test_cli.py`, `test_preprocessing.py`, `test_sequence_generator.py`,
  `test_transformer.py`, `test_training.py`, `test_losses.py`, `test_determinism.py`,
  `test_resume.py`, `test_ablation_robustness.py`, `test_scripts.py`, `test_federated.py`,
  `test_diagnostics.py`, `test_lora.py`, `test_moe.py`, `test_smoke.py`. Check free disk before and
  after each file. (All ACs' test halves)
  - **Done when:** every file passes; the BUILD note lists pass/skip counts per file and the
    real-kernel FA3 test is among the skips.
  - *BUILD note:* One file at a time, `uv run pytest tests/<file>.py -m "not slow" -q -rs`, free disk (`df -k /`) before/after each:
    `test_fixture_tiny` | 4 passed in 0.48s | free_before=62G free_after=62G;
    `test_cli` | 63 passed in 1.56s | free_before=62G free_after=62G;
    `test_preprocessing` | 45 passed in 3.37s | free_before=62G free_after=62G;
    `test_sequence_generator` | 37 passed, 1 warning in 8.51s | free_before=62G free_after=62G;
    `test_transformer` | 53 passed, 1 skipped in 1.52s | free_before=62G free_after=62G;
    `test_training` | 26 passed, 2 deselected, 2 warnings in 28.80s | free_before=62G free_after=61G;
    `test_losses` | 40 passed in 1.62s | free_before=61G free_after=61G;
    `test_determinism` | 6 passed, 2 warnings in 50.68s | free_before=61G free_after=61G;
    `test_resume` | 12 passed in 92.16s (0:01:32) | free_before=61G free_after=59G;
    `test_ablation_robustness` | 29 passed in 18.10s | free_before=59G free_after=59G;
    `test_scripts` | 14 passed in 5.75s | free_before=59G free_after=60G;
    `test_federated` | 13 passed, 6 deselected in 1.10s | free_before=60G free_after=60G;
    `test_diagnostics` | 22 passed in 3.15s | free_before=60G free_after=61G;
    `test_lora` | 7 passed in 1.07s | free_before=61G free_after=61G;
    `test_moe` | 19 passed in 0.67s | free_before=61G free_after=62G;
    `test_smoke` | 2 passed in 1.15s | free_before=62G free_after=62G;
    The single skip is `TestRealFlashAttention3Kernel::test_left_padded_parity_against_sdpa` ("needs CUDA capability 9.x and flash_attn_interface"). Deselected = `slow` (real-data) tests, not run per the hygiene rule. Disk never dropped below 59 GB free.

- [x] **T9.2: Real-data capped smoke (with orchestrator approval for disk use)** — With ≥ 15 GB free:
  (a) `uv run python main.py --stage preprocess --user_sample_fraction 0.1 --seed 42` → check
  `data/processed_users10pct/preprocessing_manifest.json` (fraction 0.1, seed 42,
  `stored_seq_length` 200) and `du -sh` it; (b) `uv run python -u main.py --stage pretrain --model
  t-small --user_sample_fraction 0.1 --epochs 2 --max_steps 30 --dtype bf16 --mixed_precision
  --num_workers 2 > <scratch>/smoke.log 2>&1 &`, poll until `last.pt` for epoch 0 exists, `kill -9`
  the process, re-run the same command in the foreground and confirm the log says
  `Resuming ... at epoch 1` and the run completes; (c) `--stage ablation --study B --model t-small
  --user_sample_fraction 0.1 --epochs 1 --max_steps 10 --dtype bf16 --mixed_precision` twice; the
  second run logs both arms as reused; (d) `uv run python scripts/package_ablation_run.py --skip-data
  --out <scratch>/dist` and inspect `LAUNCH.md`; (e) delete the smoke outputs under `outputs/` that
  these runs created (list them first). (AC-31)
  - **Done when:** (a)–(d) hold as stated, with the evidence (manifest excerpt, resume log line,
    reuse log lines, emitted commands) quoted from the files in the BUILD note, not from memory.
  - *BUILD note:* Free disk before: 62 GB (`df -h /`). `outputs/` (115 MB) was backed up first, because the
    t-small pretrain smoke writes into the same `outputs/checkpoints/t_small/` and
    `outputs/logs/*_t_small*` files earlier runs left there.
    (a) `uv run python -u main.py --stage preprocess --user_sample_fraction 0.1 --seed 42` → exit 0
    in 30 s; log: "CLI override: processed_data_dir data/processed -> data/processed_users10pct",
    "User sample (fraction=0.1, seed=42): 136677 -> 13668 users, 1011935 rows", "Storing sequences
    at width 200". `data/processed_users10pct/preprocessing_manifest.json` (read back):
    `"user_sample_fraction": 0.1, "user_sample_seed": 42, "stored_seq_length": 200,
    "num_users_before_sampling": 136677, "num_users_after_sampling": 13668,
    "user_sample_fingerprint": "d7636927d8f9e458"`, 13530 items, 1812 cold. `du -sh`: 2.5G. The
    directory is KEPT (it is what `package_ablation_run.py` ships) and is now git-ignored.
    (b) `uv run python -u main.py --stage pretrain --model t-small --user_sample_fraction 0.1
    --epochs 2 --max_steps 30 --dtype bf16 --mixed_precision --num_workers 2` in the background
    (MPS); polled until `outputs/checkpoints/t_small/last.pt` existed, then `kill -9`.
    CAVEAT, measured: `kill -9` of the `uv run` PID killed only `uv`; its python child (pid 13487,
    re-parented to 1) and its two DataLoader workers kept running, and were `kill -9`-ed directly
    ~35 s later, still inside epoch 1 (`last.pt` read back: `epoch 0, global_step 30, total_steps
    60`). Re-running the same command in the foreground: exit 0; log line "Resuming t-small from
    outputs/checkpoints/t_small/last.pt at epoch 1 (global step 30)", then "Epoch 1:
    train_loss=26.6237 val_ndcg=0.1150"; the per-epoch log has exactly two lines, `epoch=0`
    (00:07:52, first process) and `epoch=1` (00:08:51, resumed). No leftover processes.
    (c) `--stage ablation --study B --model t-small --user_sample_fraction 0.1 --epochs 1
    --max_steps 10 --dtype bf16 --mixed_precision`, twice, both exit 0. Run 2 log: "objective_conditioned:
    reusing the persisted result outputs/ablations/seed_42/arms/B_t_small/objective_conditioned.json
    (arm_key e6d620adb00fc68f)" and "input_only: reusing ... input_only.json (arm_key
    c4511fa14b42ef74)"; `seed_42/B_t_small.json` read back: both arms `status ok`,
    `reused_from_previous_run: true`, `runtime: {attention_backend: sdpa, device: mps,
    device_name: arm, dtype: bf16, mixed_precision: true}`, NDCG 0.0750 / 0.0842 on 5000 examples.
    (d) `uv run python scripts/package_ablation_run.py --skip-data --out <scratch>/dist` → exit 0;
    `code.tar.gz` 87 members (84 tracked + 3 launch files); both scripts pass `bash -n`; LAUNCH.md
    command: `python3 -u scripts/run_with_retry.py --max-retries 20 --wait-seconds 300 -- python3
    -u main.py --stage ablation --study A --model t-base --seed 42 --user_sample_fraction 0.1
    --user_sample_seed 42 --dtype bf16 --mixed_precision --batch_size 256 --attention_backend
    flash_attention_3 --num_workers 8`; one-liners `mkdir -p logs && setsid nohup bash
    launch/run_h100.sh > logs/run_h100.log 2>&1 < /dev/null &` (and `run_aws`). Reading it turned
    up a wrong stop instruction (killing only the wrapper would let the launch loop start the next
    study); fixed to end the launch script first, re-packaged, packaging tests re-run (3 passed).
    The packager warned, correctly, that the three new scripts are not git-tracked yet, so this
    tarball lacks them until the orchestrator commits.
    (e) Created/changed under `outputs/` (from `diff -rq` against the backup):
    `ablations/seed_42/`, `checkpoints/ablations/`, `checkpoints/t_small/{epoch_001.pt,last.pt}`
    (new), `checkpoints/t_small/{best.pt,epoch_000.pt}`, `logs/{gradnorm_t_small.jsonl,
    pretrain_t_small.log,routing_diagnostics_t_small.json,run_config.json}` (overwritten),
    `logs/moe_parameter_report_t_small_B_{input_only,objective_conditioned}.json`,
    `plots/ablation_B_t_small.png` (new) and four `*_t_small.png` plots (overwritten). Evidence was
    copied to the scratchpad, then `outputs/` was restored from the backup; `diff -rq` against
    the backup is now empty.

## Review fixes (round 2, from `review.md`)

Each fix below was verified with the targeted test files, one at a time, `-m "not slow"`:
`test_scripts.py` 18 passed, `test_resume.py` 13 passed, `test_ablation_robustness.py` 31
passed, `test_determinism.py` 6 passed (golden unchanged), `test_cli.py` 63 passed,
`test_training.py` 26 passed.

- [x] **REV-001 — packager refuses uncommitted launch code.** `scripts/package_ablation_run.py`
  now runs `launch_readiness_problems()` before writing anything. It exits non-zero, listing
  each problem and naming `--allow-dirty`, when:
  - `main.py`, `scripts/run_with_retry.py`, `scripts/verify_attention_backend.py` or
    `scripts/install_fa3.sh` is not tracked;
  - any of them, or any source the provenance rule covers (every first-party `.py` outside
    `tests/`, i.e. what `main.py` imports), is modified or untracked.

  `--allow-dirty` ships the working tree with a warning and adds untracked launch scripts to
  the tarball. New `TestPackagerRefusesUncommittedCode` copies the tree into a throwaway git
  repo and commits it, so it does not depend on this repo's commit state:
  - the committed copy packages without the flag, and the tarball contains all four files;
  - an edited `run_with_retry.py`, an untracked `install_fa3.sh` and an edited
    `training/pretrain.py` are each named, with nothing written;
  - `--allow-dirty` then succeeds.

  The in-repo packaging test passes `--allow-dirty`. Today, without the flag, the real tree is
  refused (three untracked scripts plus every modified source). After the orchestrator's
  commit it passes.
- [x] **REV-002 — per-arm training logs.** `_run_single_config` trains with
  `cfg.log_dir = training_log_dir(checkpoint_root, fingerprint)`, which is
  `<store>/<fingerprint>/logs`, the same namespace as the arm's checkpoints. The arm's own
  `log_dir` is restored afterwards for its post-training reports, and the result records
  `training_log_dir`. New `TestPerArmTrainingLogs` (study B, 2 epochs):
  - each arm's own `pretrain_t_small.log` has epochs `0,1`;
  - its GradNorm JSONL has step `[0]`;
  - the shared per-model files no longer exist.

  LAUNCH.md's collect command and the README now include
  `outputs/checkpoints/ablations/by_fingerprint/*/logs`.
- [x] **REV-003 — resume log-cleanup test.** The GradNorm log interval is now the module
  constant `GRADNORM_LOG_INTERVAL_STEPS` (100, unchanged), so tests can patch it. New
  `TestResumeLogCleanup` runs with the interval set to 2, and makes `save_checkpoint` raise for
  the final epoch's `last.pt`, after that epoch's log line and GradNorm records are written:
  - before resume the logs hold epochs `0,1,2` and steps `0..10`, so the test is non-vacuous;
  - the resume redoes epoch 2 once (4 steps);
  - the logs are still exactly epochs `0,1,2` and steps `[0,2,4,6,8,10]`.

  It would fail if either truncation function were a no-op.
- [x] **REV-004 — single-arm selectivity test.** New `TestSingleArmRerun` changes
  `ablation_eval_examples` for `input_only` only:
  - `run_pretrain` ×0, `evaluate_model` ×1;
  - `objective_conditioned` is `reused_from_previous_run` (32 examples);
  - `input_only` is re-evaluated (16 examples).
- [x] **REV-005 — safe venv activation.** In the generated `run_aws.sh`:
  - `set +u` / `source /opt/pytorch/bin/activate` / `set -u`;
  - then `"$PY" -c "import torch" || { …; exit 1; }` before the study loop.

  The packaging test asserts both, in that order. The script passes `bash -n`.
- [x] **REV-006 — distinct build-failure exit code.** `scripts/install_fa3.sh` exit codes:

  | Code | Meaning |
  |---|---|
  | 0 | built (or already installed) and verified |
  | 1 | parity failure |
  | 2 | preflight failed or FA3 unavailable |
  | 3 | extraction or `pip install` of `hopper/` failed (`BUILD FAILED`, no verification run) |

  A new `--log-dir` option defaults to `logs/`. The codes are documented in the script header,
  in the README and in the generated LAUNCH.md. New `TestFA3InstallExitCodes` runs the real
  script against stub `python`/`nvcc`/`ninja` that impersonate a Hopper box:
  - a failing build exits 3, its output lands in `fa3_build.log`, and the verification does
    not run;
  - `--dry-run` passes the preflight and creates no build directory.
- [x] **Stale disk-hog text.** `steering/testing.md` and `requirements.md` (Non-Goals and
  "Out-of-scope findings") now say the subset-helper disk hog was fixed by owner-approved T0.2.
- **REV-007 (left as is; recommendation only).** `atol=rtol=2e-2` is an element-wise bound,
  and its margin shrinks with tensor magnitude. On MPS, bf16 SDPA vs the fp32 reference
  (t-base heads, head dims 64/80, lengths 50/200, padded and unpadded) measured:

  | Metric (real positions) | Output | Worst gradient |
  |---|---|---|
  | Relative Frobenius error `‖a−b‖/‖b‖` | ≤ 1.6e-3 | ≤ 2.4e-3 |
  | Cosine similarity | ≥ 0.999999 | ≥ 0.999997 |
  | Max absolute diff | up to 2.2e-2 (`dv`, with \|ref\| up to 5.4) | |

  Recommendation: gate each tensor on relative Frobenius error ≤ 1e-2 and cosine ≥ 0.9999,
  and keep reporting max-abs. That is about 4× headroom over the measured bf16 error and still
  rejects any real defect: a wrong mask or non-causal attention gives a relative error of
  order 1. It is not meaningfully weaker. Not changed, as instructed.
- REV-008, REV-009, REV-010: left as they are (INFO).
