# Training Pipeline — Tasks

Ordered tasks. Depends on all model specs (transformer-backbone, moe-routing, loss-functions) and data-pipeline.

## Phase 1: LR Scheduler

- [x] **T1.1: Implement `build_scheduler` in `training/pretrain.py`** — Linear warmup for `warmup_ratio` fraction of `total_steps`, then cosine decay to 0. Use `LambdaLR`. `total_steps = num_batches_per_epoch * pretrain_epochs`. (FR-1, AC-01)

- [x] **T1.2: Test LR scheduler** — Verify warmup increases linearly over warmup steps. Verify cosine decay reaches near-zero at end. Verify per-batch stepping (not per-epoch). (AC-01)

## Phase 2: Training Loop

- [x] **T2.1: Implement model construction function** — Build Transformer + MoELayer based on model name and config. Initialize all parameters in BF16. Move to device via `get_device(config)`. (FR-7)

- [x] **T2.2: Implement training loop in `training/pretrain.py`** — Per epoch: iterate batches → apply masking (if enabled) → transformer forward (return_all_hidden=True) → extract pooled_hidden → per-objective MoE routing + loss → GradNorm or static weighted sum → backward(retain_graph=use_gradnorm) → gradnorm.update (if enabled) → clip_grad_norm(1.0) → optimizer.step → scheduler.step. Accumulate routing snapshots for diagnostics buffer. Load global negative pool once (never per-epoch). (FR-5, FR-6, AC-04, AC-05, AC-06)

- [x] **T2.3: Implement contrastive loss integration** — On contrastive-interval batches: generate augmented sequence via `augment_sequence()`, run second forward pass through same model, L2-normalize embeddings, compute InfoNCE. On non-interval batches: zero tensor, exclude from GradNorm active set. (FR-5)

- [x] **T2.4: Implement validation step** — After each epoch: model.eval(), run evaluation on validation dataset using fixed candidate sets and `eval_user_history`. Report NDCG@10, MRR@10, Recall@10, Precision@10. (FR-8, AC-02)

## Phase 3: Checkpointing

- [x] **T3.1: Implement checkpoint save** — Save dict with: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, gradnorm_state_dict (if enabled), val metrics, config. Per-epoch: `outputs/checkpoints/{model}/epoch_{N:03d}.pt`. (FR-3)

- [x] **T3.2: Implement best checkpoint selection** — Track `best_val_ndcg`. Update `best.pt` only on strict NDCG improvement. Store best_epoch and all best_val metrics in checkpoint metadata. Never use MRR, Recall, Precision, training loss, or test metrics for selection. (FR-4, AC-02)

- [x] **T3.3: Implement checkpoint loading** — Load full state: model, optimizer, scheduler, gradnorm. Support resuming training from a checkpoint. (FR-3)

## Phase 4: Logging

- [x] **T4.1: Implement per-epoch logging** — Log to `outputs/logs/pretrain_{model_name}.log`: epoch, train loss (total + per-objective), val NDCG@10, MRR@10, Recall@10, Precision@10, current LR, GradNorm lambda values (per objective). (FR-5, AC-07)

- [x] **T4.2: Implement GradNorm lambda trajectory logging** — Log per-objective lambda values at configured intervals to `outputs/logs/gradnorm_{model}.jsonl`. At end of training, generate `outputs/plots/gradnorm_lambda_trajectory_{model}.png`. (FR-9)

- [x] **T4.3: Implement routing snapshot accumulation** — Per batch, for each active objective: detach r_ut and append to per-epoch buffer. Reset buffer at each diagnostic interval. Feed buffer to routing diagnostics (evaluation/routing_diagnostics.py). (FR-6, AC-06)

## Phase 5: Integration

- [x] **T5.1: Implement `run_pretrain` function** — Orchestrate: load datasets → build model → build optimizer/scheduler/gradnorm → training loop → validation → checkpointing → logging. Accept config and model_name. (FR-all)

- [x] **T5.2: BF16 assertions** — Add assertions verifying all model parameters, outputs, and losses are bfloat16. Add finite-value assertions on losses, lambdas, and gradients on every step. (FR-7, AC-08)

## Phase 6: Testing

- [x] **T6.1: Checkpoint roundtrip test** — Save checkpoint → load → verify model produces identical output on same input. Verify gradnorm state preserved. (AC-03)

- [x] **T6.2: One-batch end-to-end smoke test** — Run one complete training iteration on real data shapes. Verify: loss is finite, gradients exist, no NaN/Inf, optimizer step completes, scheduler steps. (Smoke test gate per steering/testing.md Phase 16)
  - BUILD note (REV-062): the test exists and its training assertions pass. The gate is
    armed (`require_attention_backend=True`) and was failing because the then-sanctioned
    MPS kernel, FlexAttention, has no backward pass on MPS in PyTorch 2.14.0.
  - PLAN note (REV-062, RESOLVED -- owner decision): resolution (a) was taken. SDPA is
    now the sanctioned MPS/CPU kernel (`steering/code-quality.md` "Attention Backend
    Discipline"; `specs/transformer-backbone` FR-4 / D2b), and FlexAttention is removed
    from the backend surface by `transformer-backbone T1.3`. **Keep
    `require_attention_backend = True` here** -- the arming was never the problem, and
    the gate now passes because the preferred backend genuinely executes. Check this
    task off only after re-running the gate unmodified against the narrowed backend
    surface.
  - BUILD note (gate re-run): done. `tests/test_smoke.py` was changed on comment lines
    ONLY (`git diff` shows no non-comment edit); `require_attention_backend = True` is
    still set and no assertion was touched. The gate passes on MPS with the preferred
    backend resolving to `sdpa` and no `fallback_reason` recorded. Full suite: 256
    passed, 0 failed.

- [x] **T6.3: Few-step T-Small smoke test on real data** — Run `run_pretrain` on the real
  MovieLens `data/processed` with a capped step budget
  (`--stage pretrain --model t-small --epochs 1 --max_steps N`, a few hundred steps).
  Verify: validation metrics computed, checkpoint saved, log written.
  (Smoke test gate per steering/testing.md Phase 17)
  - **Owner decision (2026-09-17):** this task previously read "Run one full epoch of
    T-Small on real MovieLens data ... Verify: training loss decreases, validation
    metrics computed, checkpoint saved, log written." The spec owner has reduced the gate
    to a few-step run on real data; `steering/testing.md` Phase 17 and `BUILD_ORDER`
    step 11.2 were updated to match, and both record the decision.
    Context for the next review: REV-066 flagged an earlier, unauthorised version of this
    same relaxation, where BUILD rewrote the gate to fit what it had run. This one is the
    owner's, made explicitly. `--max_steps` was added to `parse_args` so the gate is
    runnable without editing source (`conventions.md`, "Config-Driven Design").
  - **AC-04 is NOT covered by this gate.** "Training loss decreases over epochs on real
    data" cannot be observed over a few hundred steps: under LR warmup both dev-host
    probes showed loss *rising* (100 steps 26.23 -> 28.51; 400 steps 27.32 -> 29.15).
    AC-04 is unchanged and stays outstanding until full training (BUILD_ORDER Phase 14).
    The reduced gate no longer claims it.
  - Dev-host measurement for the record (2026-09-17, after loss-functions REV-120): the
    uncapped epoch is 30,447 steps at ~0.235 s/step ≈ 2.0 h, down from the 0.641 s/step /
    ~5.5 h measured before the factored-`G_t` change.
  - `tests/test_training.py::TestFewStepSmokeViaRunPretrain` is the in-suite form of this
    gate: the same capped `run_pretrain` path against a subset of the real
    `data/processed`. It runs on every `pytest` invocation and is kept.
  - **RE-RUN AND VERIFIED (2026-09-18), superseding the 2026-09-17 record below.**
    REV-129 found the previous note's metrics did not match the files on disk, and that
    the repository gave two contradictory accounts of whether the certified checkpoint
    predated the multi-step objective change. Both are resolved by re-running the gate
    against current source rather than by arguing about the old one.

        uv run python main.py --stage preprocess
        uv run python main.py --stage pretrain --model t-small --epochs 1 --max_steps 200

    The preprocess re-run is required, not incidental: `data-pipeline` REV-137 changed the
    cold-start default to `"unseen_in_training"`, which changes the TRAINING DATA (6,634
    previously-deleted interactions are restored). Pretrain: 15:25:59 -> 15:29:08, 189 s
    wall, exit 0.

    Every value below was read back out of the artifact AFTER the run, never transcribed
    from console output — that is REV-130's correction and the reason the previous note
    drifted. If a later run overwrites these files, this note becomes visibly wrong rather
    than plausibly right.

    1. **Validation metrics computed** over the FULL 993,372-example validation split
       against the stored `val_candidates.npy`. Read from `outputs/logs/pretrain_t_small.log`
       and cross-checked against `best.pt`, which agrees to all six digits:
       `val_ndcg@10=0.091145`, `val_mrr@10=0.064986`, `val_recall@10=0.178565`,
       `val_precision@10=0.017856`.
       Per-objective training losses from the same line: `train_loss_total=28.176922`,
       `nextitem=4.594844`, `maskeditem=9.900937`, `multistep=9.905000`,
       `contrastive=4.433012`.
    2. **Checkpoint saved** — `outputs/checkpoints/t_small/best.pt` and `epoch_000.pt`,
       both carrying the full FR-3 metadata set plus GradNorm state, and now
       `source_revision` (see T6.5).
    3. **Log written** — `outputs/logs/pretrain_t_small.log` with every AC-07 field:
       per-objective losses, all four validation metrics, LR and the four λ values.
    Also regenerated from this checkpoint: `test_metrics_t_small.json` (test
    `ndcg=0.087029`, `mrr=0.061676`, `recall=0.171841`, `precision=0.017184`),
    `cold_start_t_small.json`, `longtail_t_small.json`, `efficiency_t_small.json`,
    `routing_diagnostics_t_small.json`, `gradient_diagnostics_t_small.json`,
    `personalization_gain_t_small.json`, `federated_rounds_t_small.json`, and all 11 plots.

  - **The three contradictory provenance accounts are resolved by deletion, not by
    argument.** REV-129 / REV-131 recorded that `specs/loss-functions` T3.7 said this
    `best.pt` "was trained under the old objective" while
    `specs/personalization-federated` T6.1 said it was the regenerated one, both written
    into commit `0bc67ee`. Neither claim is about the current `best.pt`, which was written
    at 15:29 on 2026-09-18 by the command above, against source that demonstrably carries
    `pos = S - 1` (pinned by `tests/test_losses.py::test_every_step_reads_position_S_minus_1`,
    which REV-105 verified fails on the superseded rule). Those two notes have been
    rewritten to point here rather than left standing. **One account, and it is this one.**

  - **Why this cannot recur (T6.5 / AC-09).** The dispute was unresolvable because
    provenance lived only in prose, and `config` — the one thing the checkpoint did
    record — was byte-identical across the change. Every artifact above now carries
    `source_revision`, verified by reading the checkpoint and each derived file and
    comparing the fields:
    `{"commit": "0bc67ee263063b562b93667f7f7523147474c979", "dirty": true,
    "source_hash": "3a59e7cb70a8be69"}`.
    `dirty: true` is correct and deliberate: REV-137 and REV-136 are in the working tree
    and uncommitted, so attributing this run to `0bc67ee` alone would be false. The
    `source_hash` is what distinguishes it.

  - Two things a reader of that log should not misread:
    - `lr=0.00000000` is correct, not a defect. `max_steps_per_epoch` also shrinks
      `total_steps`, so the capped run executes the WHOLE schedule: warmup 0 -> 1e-3 over
      20 steps, then cosine decay back to 0 at step 200. The logged value is the
      schedule's endpoint.
    - The λ values do not sum to 4. Contrastive is inactive on the final step
      (`contrastive_interval`), and the three active λ sum to 3.000001 — which is AC-05's
      `sum(active) == num_active` holding within tolerance, not violating.
    - `train_loss_total=28.176922` is HIGHER than the first-window mean. This is the
      documented AC-04 non-coverage, not a regression: under LR warmup a short window
      shows loss rising. AC-04 stays outstanding until Phase 14.

  - *Superseded record (2026-09-17), retained for history:* reported
    `val_ndcg@10=0.090022 / mrr 0.064059 / recall 0.176741 / precision 0.017674` against a
    `data/processed` built under the `first_appearance_rank` cold-start default. Those
    numbers never matched the log that survived (`0.097165 / 0.070113 / ...`), which is
    what REV-129 caught. Both that run and the intermediate one it was confused with are
    now superseded by the 2026-09-18 re-run: the training data itself has changed.

- [x] **T6.5: Artifact provenance — `source_revision` (FR-3 amended, AC-09 new)** —
  REV-136 / `steering/code-quality.md` "Artifact Provenance" (owner, 2026-09-18).
  - `utils.get_source_revision()` resolves the revision once per process: the git commit
    when the tree is clean, otherwise the commit PLUS a SHA-256 over every covered `.py`
    source, with `dirty: true`. A dirty tree is never attributed to its nearest commit —
    that is precisely the inference that failed in REV-129, where commit time was not
    edit time.
  - Written into every checkpoint by `save_checkpoint` (`training/pretrain.py:107`).
  - Echoed VERBATIM by every artifact derived from a checkpoint. Downstream stages read it
    off the loaded checkpoint via `utils.artifact_source_revision()` rather than
    recomputing it, so an artifact reports the model's provenance and not the provenance of
    whatever code happened to evaluate it. Covered: `test_metrics_*`, `cold_start_*`,
    `longtail_*`, `efficiency_*`, `routing_diagnostics_*`, `gradient_diagnostics_*`,
    `personalization_gain_*`, `federated_rounds_*` (per round record, since that file is a
    LIST read by index at `main.py`'s plots stage), each per-arm ablation result, the
    per-epoch log and the λ JSONL.
  - A checkpoint written BEFORE this landed degrades to an explicit
    `"the provenance of this artifact is NOT established"` note rather than borrowing the
    running code's revision. Substituting would assert a provenance that does not hold —
    the silent-substitution failure `steering/code-quality.md` forbids.
  - **Verified as AC-09 specifies** — by reading a real checkpoint AND each real derived
    artifact and comparing the fields, not by asserting the writing code exists (REV-136's
    explicit instruction, and the lesson of REV-135's two dead stages). All 8 derived
    artifacts plus all 3 federated round records MATCH the checkpoint exactly; measured
    2026-09-18 after the re-run above.
  - **Coverage is now defined by EXCLUSION, not enumeration** — REV-138, implemented
    2026-09-18 against `steering/code-quality.md` "Coverage: a property, not a list"
    (owner amendment). The governing rule is the property: `source_revision` must change
    whenever any code that can change a published artifact changes. Covered is therefore
    every first-party `.py` tracked by git EXCEPT `tests/`, `.venv/`, `__pycache__/`,
    `data/raw/` and `data/processed/`. `utils._covered_source_files()` enumerates that set
    from `git ls-files` rather than from a fixed directory list, so top-level modules and
    packages added later are covered without anyone remembering to add them. Measured on
    this repo: **27 covered files**, including all six `evaluation/` modules, `main.py`
    and `utils.py`.
  - **The dirtiness check and the content hash share one predicate**,
    `utils._is_covered_source()` — not two rules that happen to agree. The previous
    `path.startswith(SOURCE_REVISION_DIRS)` string test is gone, and
    `_porcelain_paths()` feeds `git status --porcelain` entries (renames split on
    `->`, untracked `??` entries skipped, since the rule excludes anything not tracked)
    through that same predicate.
  - **Correction to the superseded gap note (REV-139).** The previous version of this
    bullet claimed that an edit to an uncovered file "still surfaces as `dirty: true`
    whenever the working tree is otherwise clean". **That was false**, and the same bullet
    contained the fact refuting it: the implementation followed the enumeration literally
    for the dirty check too, so `"main.py".startswith(("models","training","data",
    "configs"))` was `False` and an edit confined to an uncovered path moved NEITHER the
    hash NOR the flag — lost silently, not loudly. The note also omitted the consequential
    omission, `evaluation/`, which is where every derived artifact this project publishes
    is produced. Both are recorded here rather than quietly deleted, because a later
    reader would otherwise have no way to tell the claim had ever been made.
  - Measured after the change (`utils._hash_source_tree`, probe appended then restored
    byte-for-byte): editing `evaluation/longtail.py` moves the hash — **True**; `main.py`
    — **True**; `utils.py` — **True**; `evaluation/metrics.py` — **True**; editing
    `tests/test_training.py` — **False**, as the rule requires.
  - **Runtime verification of the widened rule** (2026-09-18, into a scratch output dir so
    no published artifact was touched): a real `run_pretrain` on the subset `processed`
    dir wrote `best.pt` carrying
    `{"commit": "0bc67ee…", "dirty": true, "source_hash": "4bc4131eb2ecd201"}` — read back
    OUT of the checkpoint after the run, not from console output. That hash differs from
    the pre-widening `3a59e7cb70a8be69`, which is the proof the new rule is what actually
    ran. `routing_diagnostics_t_small.json` echoed it verbatim, and the epoch log carried
    `source_hash=4bc4131eb2ecd201`. Appending a line to `evaluation/longtail.py` in the
    same process then moved it to `da421a8e2f4aa2aa` and removing the line restored it —
    the discriminating power REV-138 was filed for, measured end to end rather than
    inferred from the unit tests.
  - **Existing artifacts were NOT re-stamped**, per REV-138's explicit instruction. Their
    `source_hash` remains a true statement about the directories it covered; widening
    changes future identifiers only. The `best.pt` cascade of 2026-09-18 therefore still
    carries `source_hash: "3a59e7cb70a8be69"`, which is the pre-widening value and is
    correct for that run.
  - **Residual, stated rather than mitigated.** An untracked `.py` is outside coverage by
    the rule's third bullet ("anything not tracked by git"), so it moves neither the hash
    nor the `dirty` flag. On this repo the point is currently moot — `git ls-files
    --others --exclude-standard` is empty, and `data/raw/`, `data/processed/`, `.venv/`
    and `__pycache__/` are gitignored so they could not be tracked in any case — but it is
    the rule's boundary and is recorded as measured fact, not offered as a mitigation.
  - **`moe_parameter_report_*.json` is now stamped too** (REV-136's closing note; owner
    approved 2026-09-18). It sits in `models/moe.py`, owned by `specs/moe-routing`, and is
    recorded under that spec's T2.3 as well. It carries the **running** revision rather
    than a checkpoint's, because it is a pure function of `config` and AC-09's
    checkpoint-echo semantics do not reach it — calling `artifact_source_revision()` there
    would assert a model provenance the file does not have. The two already-published
    reports were not regenerated, per REV-138.
  - **Note for the owner, not acted on:** `requirements.md` FR-3 still describes the hash
    as covering "`models/`, `training/`, `data/` and `configs/`". That wording is now
    superseded by the steering amendment. Per §9 the spec is not rewritten by BUILD to
    match the implementation, so it is flagged here instead.
  - Tests: `tests/test_training.py::TestArtifactProvenance` (6 tests: the revision
    identifies the running source; a source change moves the hash so two runs from one
    dirty tree are distinguishable — the probe now edits `evaluation/longtail.py`, the
    file the allowlist silently missed; a pre-AC-09 checkpoint is recorded as
    unestablished rather than attributed; every named writer emits the field, checked by
    writing real files and reading the field back; plus the two REV-138 additions —
    **`test_hashed_set_equals_the_set_the_coverage_rule_defines`**, which records the
    files `_hash_source_tree` ACTUALLY opens and compares them to a set derived
    independently in the test from `git ls-files`, and
    **`test_dirtiness_and_hash_share_one_coverage_rule`**, which drives synthetic
    `git status --porcelain` output through `get_source_revision` and pins the dirty verdict
    for `evaluation/`, `main.py`, `utils.py`, renames, deletions, `tests/`, `specs/` and
    untracked paths). Both additions are load-bearing, verified by reintroducing the
    defects: reinstating the allowlist fails 3 of the 6; reverting only the dirty
    predicate to the string-prefix test fails
    `test_dirtiness_and_hash_share_one_coverage_rule` alone. The completeness test is why
    this cannot ship green while incomplete — asserting that *some* change moves the hash
    proves the mechanism, not the coverage. `TestFewStepSmokeViaRunPretrain` additionally
    asserts the checkpoint→artifact echo end-to-end on every `pytest` invocation.

- [x] **T6.4: Negative pool reuse test** — Verify the negative pool is loaded from disk (not regenerated) on second run with same fingerprint. (AC-05)
