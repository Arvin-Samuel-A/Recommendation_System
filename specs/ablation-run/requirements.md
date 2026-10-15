# Ablation Run (10% users, t-base, one seed) — Requirements

## Overview

Make the repository ready to run **every ablation study A–H at t-base, one seed (42), on a seeded
10% sample of users, in mixed bf16, at batch size 256**, split across two machines (shared H100 →
studies A, D, E, H; AWS VM → B, C, F, G; the base config trains once on each), and hand the owner
the exact artifacts and detached launch commands to start it.

The run itself happens on the target machines, which BUILD cannot reach. This feature delivers the
code, tests, tooling and capped Mac verification that make that run correct, resumable and
reproducible.

Source: `HANDOFF-ablation-run.md` (findings at `0a91094`, re-verified at HEAD — see
`design.md` "Re-verified findings") and the owner's answers of 2026-10-15, which are binding here.

## Problem Statement

At HEAD the ablation cannot be run as the owner requested:

1. **Precision.** The default stores weights in pure bf16, which plateaus (measured on a T4,
   t-small, 360 steps: next-item loss 4.56 flat after ~120 steps, against 4.436 and still falling
   under mixed precision). The run must use `--dtype bf16 --mixed_precision`.
2. **Attention.** The CUDA backend is named `flash_attention_4` but imports the FlashAttention-2
   `flash_attn` package. FlashAttention-3 (`flash_attn_interface`, built from the
   flash-attention repo's `hopper/` directory) is a different package, runs only on Hopper (sm_90),
   and ships no `bert_padding` helpers — yet every training batch is left-padded, so the
   variable-length path is the hot path.
3. **LR schedule.** `build_scheduler` decays the cosine to exactly 0 at the final step.
4. **Data.** There is no way to train on a subset of users.
5. **Sequence width.** Tensorised splits are stored at `max_seq_length` (50), and loaders can only
   narrow, so Ablation H2's seq100/seq200 arms silently train on the same 50 items as seq50. Several
   loaders do not narrow at all, so storing wider data would break them.
6. **Fragility on a shared GPU.** `run_pretrain` checkpoints every epoch but cannot resume; other
   users can OOM a multi-hour run and lose it. `run_ablation` swallows every per-arm exception,
   including OOM, and the cross-study backbone store has no lock.
7. **No run packaging.** The H100 has no git; code and data travel by tarball, and there is no
   reproducible way to build them or to produce the launch commands.

## Goals

- G1: Mixed bf16 for every ablation command; project default untouched.
- G2: A correctly named FlashAttention-3 backend on Hopper, SDPA everywhere else, with our own
  unpad/repad for padded batches, importable-optional and tested on the Mac.
- G3: FA3 install + verification tooling for the H100 (torch 2.14.1+cu130, CUDA 13, Python 3.12,
  no git).
- G4: Higher GPU utilisation **without changing training semantics by a single bit**.
- G5: Cosine LR decays to a configurable floor (default 10% of peak).
- G6: Seeded user sampling, recorded in the processed artifacts and in the training fingerprint.
- G7: Tensorised data stored wide enough for every H2 arm, with every loader narrowing and a loud
  failure when a model or loader would see the wrong width.
- G8: Exact resume of `run_pretrain` from its latest checkpoint, an OOM-aware retry launcher, a
  lock on the backbone store, and per-arm progress persistence so a restarted study resumes.
- G9: One command that builds the `dist/` tarballs and emits the exact detached launch commands for
  both machines.

## Non-Goals

- Changing the batch size, LR scaling, data-parallel shards, multiple arms per GPU, or any other
  batch/run-level parallelism (owner decision 3). No batch-size probing.
- GPU negative sampling (see design D4 for why it fails the owner's bit-identity bar).
- Changing the project default precision (`dtype="bf16"`, `mixed_precision=False` stays).
- Running the ablation itself, any multi-hour job, or anything on the H100/AWS machines.
- Mid-epoch resume (epoch-granular resume only; design D8).
- Resume of `run_federated` / `run_local_personalization` internals (an arm that fails in those
  stages re-runs them; its backbone is reused — design D11).
- Multi-seed runs, `--study all` summary across machines, full-data (100% users) runs.
- ~~Fixing the disk-hungry slow-test helper `_build_subset_processed_dir`~~ — superseded: the
  owner approved the fix during BUILD, and it landed as task T0.2 (see "Out-of-scope findings").

## Functional Requirements

### FR-1: Precision for the ablation run (owner decision 1)
- Every ablation command emitted by this feature carries `--dtype bf16 --mixed_precision`.
- `Config` defaults are unchanged: `dtype="bf16"`, `mixed_precision=False`.

### FR-2: FlashAttention-3 backend (owner decision 2)
- The sanctioned backends are exactly `auto`, `flash_attention_3`, `sdpa`. `flash_attention_4` is
  removed (it called the FA2 API); selecting it, or `flex_attention`, is rejected by
  `Config.validate()` with a message naming only the supported choices.
- Preferred backend: CUDA device with compute capability major 9 (Hopper) → `flash_attention_3`;
  any other CUDA device → `sdpa` (sanctioned, not a fallback); MPS/CPU → `sdpa`.
- `auto` on Hopper with FA3 unavailable falls back to `sdpa` with a recorded `fallback_reason`,
  unless `require_attention_backend=True`, which raises. An **explicitly** requested backend that
  cannot execute raises `AttentionBackendError` (no fallback).
- FA3 is imported lazily, from `flash_attn_interface`, only when the FA3 backend is probed or used.
  No import of `models.transformer`, no test collection, and no Mac code path requires it.
- Unpadded batches use FA3's dense call; padded batches are unpadded to a ragged buffer with
  cumulative sequence lengths, passed to FA3's variable-length call, and repadded with zeros at pad
  positions. The unpad/repad is our own code (FA3 has no `bert_padding`), is differentiable, and
  makes no left-padding assumption.
- Causality reaches the FA3 kernel as `causal=True` on both the dense and varlen calls.
- FA3 receives only bf16/fp16 inputs; anything else raises `AttentionBackendError` with a clear
  message instead of a kernel error.
- The startup probe exercises the model's real head dimension (`hidden_dim // num_heads`) on both
  the unpadded and the left-padded branch, forward and backward.
- Every ablation arm result records the attention backend that actually ran and the device it ran
  on, so reviewers can check that all arms of a study shared one backend.

### FR-3: FA3 installation and verification tooling (owner decision 2)
- A Mac-side script fetches the flash-attention source (with submodules) at a given ref and packs
  it into a tarball under `dist/`, recording the exact commit.
- An H100-side script runs preflight checks (Hopper GPU, torch and its CUDA version, a matching
  `nvcc`, build prerequisites, disk), skips the build if FA3 already imports (unless forced),
  builds `hopper/` against the installed torch with only the kernel variants this project needs,
  and finishes by running the verification step. It has a `--dry-run` mode that runs only preflight.
- A verification script checks a backend on the current machine: self-test, then numeric parity
  of outputs and gradients against an fp32 SDPA reference on left-padded and unpadded batches at
  head dims 64 and 80 and sequence lengths 50 and 200; writes a JSON report and exits 0 (pass),
  1 (parity failure) or 2 (backend unavailable).
- README documents the procedure, including detached execution of the build.

### FR-4: Utilisation levers that keep training bit-for-bit identical (owner decision 3)
- Batch size stays 256 for every arm. No batch probing, no data-parallel shards, no concurrent arms.
- `--num_workers N` loads batches in worker processes with pinned memory; the sequence of batches
  (every tensor) is identical to `num_workers=0` for the same seed, across epochs.
- Training negatives are sampled from the CPU-side batch before the host→device copy (today the
  batch is copied to the device and its targets/user ids copied back each step). The negatives are
  identical for the same RNG state.
- Per-step host↔device syncs in the training hot path are reduced where the result is provably
  identical: per-loss `.item()` calls are replaced by per-epoch materialisation; the per-step finite
  checks are batched into stacked checks; GradNorm's renormalisation no longer branches on a device
  value. Every training step is still checked for finite losses, total loss, gradient norm and
  active lambdas, and a non-finite value still aborts the step before `optimizer.step()`.
- A characterisation test, recorded **before** any FR-4 change, proves the parameters, loss trace
  and LR trace of a tiny CPU run are unchanged by FR-4.

### FR-5: Cosine LR floor (owner decision 4)
- New config field `min_lr_ratio` (default `0.1`), settable by `--min_lr_ratio`, validated to
  `[0, 1]`.
- After linear warmup (unchanged), the LR multiplier follows
  `min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * progress))` with progress clamped to
  `[0, 1]`; it reaches `min_lr_ratio` at the final step and never goes below it.
- `min_lr_ratio = 0` reproduces the previous schedule exactly.
- `min_lr_ratio` is part of the training fingerprint.
- LoRA personalization and the federated simulation use constant-LR optimizers with no scheduler;
  they are unchanged (verified, design D5).

### FR-6: Seeded user sampling (owner decision 5)
- New config fields `user_sample_fraction` (default `1.0` = no sampling) and `user_sample_seed`
  (default `42`, independent of the training `seed`), settable by `--user_sample_fraction` /
  `--user_sample_seed`; `user_sample_fraction` validated to `(0, 1]`.
- Sampling happens in `data/preprocessing.py::_read_ratings` **after** the rating filter and the
  min-interactions filter: exactly `max(1, round(fraction × N_valid))` users are drawn without
  replacement from the sorted valid users by an RNG seeded with `user_sample_seed`, keeping each
  sampled user's full filtered history.
- Everything downstream (ID maps, splits, cold/warm items, metadata rows, negative pool, eval
  candidates, popularity buckets, histories, tensorised splits) is rebuilt from the subset — by
  construction, since all of it derives from `_read_ratings`.
- When `--user_sample_fraction` < 1 and `processed_data_dir` is still the default, the processed
  directory becomes `data/processed_users{pct}pct` (`pct` = `format(round(fraction*100, 6), "g")`),
  with suffix `_seed{S}` when `user_sample_seed` differs from its default. A sampled dataset never
  overwrites `data/processed`.
- `dataset_stats.json` and `preprocessing_manifest.json` record `user_sample_fraction`,
  `user_sample_seed`, the user count before and after sampling, and a `user_sample_fingerprint`
  (hash of the sorted original user ids kept). Both fields are in the training fingerprint.

### FR-7: Stored sequence width and narrowing (owner decision 6, option A)
- Preprocessing stores every split (tensorised `.pt` and `*_examples.pkl`) at
  `max(sequence_length_grid ∪ {max_seq_length})` — 200 by default — and records it as
  `stored_seq_length` in the manifest. `--sequence_length_grid` is settable from the CLI.
- Every reader narrows to the consuming config's `max_seq_length` (keeping the most recent
  positions and clamping `lengths`), through one shared helper per artifact type. Narrowing from the
  stored width to N is identical to having generated the data at width N.
- `load_tensorised` (and the examples helper) raise a `ValueError` naming the remedy when the
  requested width exceeds the stored width — a seq200 arm must never silently train on 50 items.
- `TransformerModel.forward` raises a `ValueError` naming both numbers when it receives more
  positions than its `max_seq_length`, before any embedding lookup.
- `run_pretrain` and `run_ablation` verify that the processed directory's manifest matches the
  config's `user_sample_fraction` / `user_sample_seed` and fail before training if not (a missing
  manifest field is read as the pre-feature default).

### FR-8: Resume from checkpoint (owner decision 7)
- Every checkpoint write is atomic (temporary file + `os.replace`).
- At the end of every epoch `run_pretrain` writes `last.pt` in the run's checkpoint directory with
  everything needed to continue exactly: model, MoE, loss-function, optimizer, scheduler,
  GradScaler (when enabled) and GradNorm state; the Python, NumPy, masking-generator, global torch
  and accelerator RNG states; the train- and diagnostic-loader generator states (data order);
  epoch, global step, best val NDCG/epoch, per-epoch loss history; `total_steps`; the training
  fingerprint; and the `source_revision`.
- When `resume_training` is true (default; `--no_resume` turns it off) and `last.pt` exists,
  `run_pretrain` resumes at the next epoch. The resumed run's LR at every global step, its batch
  order, and (on CPU) its final parameters equal those of an uninterrupted run.
- Resume refuses, with a message naming the checkpoint and the remedy, if the stored training
  fingerprint, `total_steps` or `source_revision` differ from the current run.
- A run whose `last.pt` already covers every epoch performs no training steps and returns the
  stored result.
- On resume, the per-epoch log and the GradNorm JSONL contain no duplicate epochs/steps.

### FR-9: OOM handling and automatic retry (owner decision 7)
- One predicate classifies out-of-memory errors (`torch.OutOfMemoryError`, and `RuntimeError`
  messages containing "out of memory" for CUDA and MPS).
- `main.py` exits with status **75** when a stage raises an OOM error.
- `run_ablation` re-raises an OOM error instead of recording the arm as failed; other exceptions
  keep today's record-and-continue behaviour.
- `scripts/run_with_retry.py` runs a command; on exit 75 it waits and re-runs it, up to a maximum;
  on 0 it stops successfully; on any other status it stops immediately with that status. Each
  attempt is logged with a timestamp. SIGTERM/SIGINT stop the child and the retry loop.

### FR-10: Backbone store lock (owner decision 8)
- Training a fingerprint in `_run_single_config` happens under an exclusive OS file lock on that
  fingerprint's store directory. After acquiring the lock the store is re-checked, so a process
  that waited reuses the backbone the holder registered instead of retraining it. Locks die with
  their process (no stale locks). Waiting is logged once with the holder's pid.

### FR-11: Ablation progress persistence and seed-namespaced results
- Each completed arm's result is persisted as it finishes. A re-run of the same study (e.g. after an
  OOM restart) reuses a persisted arm instead of re-evaluating it when its arm key — the resolved
  config without path/runtime-only fields, study, label, model, and the current `source_revision` —
  matches. `--no_resume` disables reuse. Backbone-store reuse is unaffected by `--no_resume`.
- Study results, persisted arms and the summary are written under `{ablation_dir}/seed_{seed}/`.

### FR-12: Run packaging and launch commands (owner decision 9)
- `scripts/package_ablation_run.py` refuses to run unless the sampled processed directory exists and
  its manifest matches the requested fraction/seed with `stored_seq_length ≥ max(sequence_length_grid)`.
- It writes `dist/code.tar.gz` (all git-tracked files plus the generated `launch/` directory; no
  data, outputs, virtualenv or untracked files), `dist/<processed dir name>.tar.gz`, and the
  `launch/` directory: `run_h100.sh`, `run_aws.sh`, `LAUNCH.md`.
- `run_h100.sh` first runs the FA3 verification and aborts if it fails, then runs studies A, D, E, H
  **sequentially**; `run_aws.sh` activates `/opt/pytorch` and runs B, C, F, G sequentially. Each
  study runs through the retry wrapper with: `--stage ablation --study X --model t-base --seed 42
  --user_sample_fraction 0.1 --user_sample_seed 42 --dtype bf16 --mixed_precision --batch_size 256
  --attention_backend <machine backend> --num_workers <N>`, logging to its own file. A failed study
  does not stop the next one.
- `LAUNCH.md` gives, per machine: copy, extract, (H100) FA3 install, and the detached launch
  one-liner in the form `setsid nohup ... > log 2>&1 < /dev/null &`, plus monitoring and
  result-collection commands. Host names/addresses are parameters with placeholders; none are
  hardcoded in the repository.
- The base config is not a separate step: the first study on each machine that needs it trains it,
  and the store serves it to every later arm on that machine.

### FR-13: Training fingerprint hygiene
- `num_workers`, `pin_memory` and `resume_training` are excluded from `_training_fingerprint`
  (they cannot change the trained model). `persistent_workers` stays in it (design D4).

## Expected Behavior

- On the Mac (no CUDA): every new test passes; `auto` resolves to SDPA; FA3 is never imported.
- On the H100: `--attention_backend flash_attention_3` runs FA3 or fails at startup; never SDPA.
- On the AWS VM: `--attention_backend sdpa` runs SDPA regardless of GPU type.
- An OOM anywhere in a study exits 75; the wrapper waits and restarts; completed arms are reused,
  the interrupted arm resumes from its last completed epoch.

## Edge Cases

- `user_sample_fraction` so small that `round(f × N)` is 0 → one user is kept (the cold-start
  holdout may then be empty and preprocessing fails loudly as it does today).
- `user_sample_fraction = 1.0` → output byte-identical to today's preprocessing for the same config.
- A sampled user with fewer than 3 items after filtering is skipped downstream as today; the
  manifest counts users after the filters and after sampling, not after this skip.
- An all-real (unpadded) batch on FA3 takes the dense call; a batch with a length-1 row and a
  full-length row takes the varlen call.
- `last.pt` from an older code revision → resume refused with a remedy; deleting it or `--no_resume`
  restarts that run.
- Crash between `best.pt` and `last.pt` writes → the epoch is redone on resume; because training is
  deterministic given the restored state, the redone epoch reproduces the same `best.pt`.
- Two processes on one machine reaching the same fingerprint → the second waits, then reuses.
- A host-RAM OOM kill (SIGKILL) is not exit 75 and is not retried.
- An old processed directory (stored width 50, no sampling fields) used by an H2 seq100 arm →
  `ValueError` from the width guard, recorded as that arm's failure.

## Constraints

- Explicit owner decisions 1–9 of 2026-10-15 (see Overview) override steering and earlier specs;
  conflicts are recorded in steering and in `design.md`.
- No new Python dependencies; FA3 is an optional, machine-local install, never in
  `requirements.txt`.
- **Dev-host test hygiene (binding on BUILD):** run only the test files a task names, always with
  `uv run pytest <file> -m "not slow"` unless the task says otherwise; never run the whole suite
  (it writes ~16 GB to temp and has filled the disk twice). New tests use tiny synthetic data built
  in a temp dir (the shared fixture from T0.1), never copies of `data/processed/`. Always
  `uv run python` / `uv run pytest`.
- Mac runs are short and capped (`--max_steps`, few epochs); no multi-hour job.
- Code tarballs carry no git metadata, so `source_revision` on the target machines is
  `{commit: null, dirty: true, source_hash}`; the hash is what identifies the code.

## Dependencies

- `training-pipeline`, `transformer-backbone`, `data-pipeline`, `ablations-cli` (all implemented).
- Superseded clauses: transformer-backbone FR-4/AC-03 (FA4 → FA3), training-pipeline FR-1 ("cosine
  decay to 0" → floor), ablations-cli FR-5 result path and its "no incremental resume" edge case.
  Each carries a pointer to this spec.
- Target machines: shared H100 PCIe (Python 3.12.3, torch 2.14.1+cu130), AWS Deep Learning AMI
  (`/opt/pytorch`, GPU type may change). Not reachable from BUILD.

## Out-of-scope findings (named, not fixed)

- `tests/test_training.py::_build_subset_processed_dir` subsets `*_examples.pkl` values only when
  they are Python lists; since REV-086 they are int32 NumPy arrays, so every call copies the full
  ~2 GB arrays. This is the main reason the full suite writes ~16 GB. **Fixed** (owner-approved,
  tasks.md T0.2): NumPy arrays are subset too; one call now writes ~57 MB.
- `training-pipeline` T3.3 is checked `[x]` ("Support resuming training from a checkpoint") but no
  code path resumed a run; FR-8 delivers it.
- `ablation_eval_examples = 5000` scores the first 5000 test examples, which are ordered by user id
  (~650 users at 10%). Combined with one seed, small NDCG differences between arms are not reliable
  (the handoff's caveat).

## Acceptance Criteria

All are verifiable on the Mac unless marked *(target machine)*; those are verified by the owner and
are not BUILD gates.

- **AC-01:** `Config()` still has `dtype == "bf16"` and `mixed_precision is False`; every ablation
  command in the generated `run_h100.sh`, `run_aws.sh` and `LAUNCH.md` contains
  `--dtype bf16 --mixed_precision` and `--batch_size 256`.
- **AC-02:** `Config.validate()` accepts `attention_backend` ∈ {`auto`, `flash_attention_3`,
  `sdpa`} and rejects `flash_attention_4` and `flex_attention` with a message that names exactly the
  three supported choices; `VALID_BACKENDS` equals that set and no `_flash_attention_4*` code remains.
- **AC-03:** The preferred-backend function returns `flash_attention_3` for (cuda, capability 9.x)
  and `sdpa` for (cuda, 8.x), (cuda, 7.5), mps and cpu. With a mocked Hopper device and FA3
  unavailable, `auto` resolves to `sdpa` with a `fallback_reason` naming `flash_attention_3`, and
  raises `AttentionBackendError` when `require_attention_backend=True`. An explicit
  `flash_attention_3` request on the Mac raises `AttentionBackendError`.
- **AC-04:** With `flash_attn_interface` absent, `import models.transformer`, building a model with
  `sdpa`, and collecting `tests/test_transformer.py` all succeed.
- **AC-05:** Unpad/repad on CPU: `repad(unpad(x))` equals `x` at real positions and is exactly 0 at
  pad positions; `cu_seqlens` and `max_seqlen` are correct for rows of lengths {S, 5, 1}; with a
  pure-torch reference varlen kernel, the FA3 varlen path's output **and** the gradients w.r.t.
  q, k, v equal masked SDPA at real positions (fp32, atol 1e-5); the kernel receives `causal=True`;
  for one model forward over L layers the varlen metadata is computed once, not L times.
- **AC-06:** The FA3 path works with a stub kernel that returns `out` and with one that returns
  `(out, softmax_lse)`; an fp32 input raises `AttentionBackendError` mentioning the dtype.
- **AC-07:** A real-kernel FA3 test exists and is skipped (not failed) unless CUDA capability 9.x
  and `flash_attn_interface` are both available. *(target machine: it passes on the H100.)*
- **AC-08:** `uv run python scripts/verify_attention_backend.py --backend sdpa --model t-base`
  exits 0 on the Mac and writes a JSON report containing torch version, device, backend, head dims,
  sequence lengths and max abs output/gradient differences; `--backend flash_attention_3` on the Mac
  exits 2 with a message saying FA3 is unavailable. *(target machine: `--backend
  flash_attention_3` exits 0 on the H100.)*
- **AC-09:** `bash -n` passes for `scripts/fetch_fa3_source.sh`, `scripts/install_fa3.sh` and both
  generated launch scripts; `scripts/install_fa3.sh --dry-run` on the Mac exits non-zero naming the
  failed preflight check and creates no build directory; the install script ends by invoking the
  verification script. *(target machine: FA3 builds and verifies on the H100.)*
- **AC-10:** For the same seed, the batches yielded over two epochs by `build_dataloader` with
  `num_workers=2, pin_memory=True` are `torch.equal` on every key to those with `num_workers=0`.
- **AC-11:** The golden characterisation test (tiny synthetic data, CPU, fixed config) records the
  final-parameter hash, per-step loss trace and per-step LR trace before any FR-4 change, and still
  passes after all FR-4 and FR-8 changes. It skips (not fails) on a different torch version or
  platform than the one recorded.
- **AC-12:** For the same `RandomState` state, negatives sampled from the CPU batch equal those
  sampled from the same batch on the device; `run_pretrain` no longer copies `targets`/`user_ids`
  from the device to the host in its step loop.
- **AC-13:** A step whose loss is NaN raises `AssertionError` and leaves every model parameter
  unchanged; a non-finite gradient norm still raises on the non-scaler path; non-finite GradNorm
  lambdas still raise. The vectorised GradNorm renormalisation is `torch.equal` to the previous
  per-index loop on random inputs.
- **AC-14:** With `min_lr_ratio = 0.1`, the LR after the final scheduler step equals
  `0.1 × learning_rate` (rel. tol 1e-6) and no post-warmup LR is below it; with
  `min_lr_ratio = 0.0` every per-step multiplier equals the previous formula exactly;
  `Config.validate()` rejects −0.1 and 1.1; changing `min_lr_ratio` changes `_training_fingerprint`.
  `tests/test_training.py::TestLRScheduler::test_cosine_reaches_near_zero` is replaced by a floor
  test.
- **AC-15:** User sampling (the pre-written `tests/test_preprocessing.py::TestUserSampling` plus
  new tests): exactly `round(f × N)` users with full histories; the subset depends only on
  `user_sample_seed`; `f = 1.0` changes nothing; `validate()` rejects 0 and 1.5;
  `--user_sample_fraction 0.1` moves `processed_data_dir` to `data/processed_users10pct`; on the tiny
  fixture, `dataset_stats.json` and the manifest carry the five sampling fields, every user in the
  splits and histories is a sampled user, and `num_items` equals the number of distinct items those
  users rated.
- **AC-16:** On the tiny fixture with `sequence_length_grid=[4, 8, 16]`, `max_seq_length=8`, every
  stored split has width 16 and the manifest says `stored_seq_length: 16`; narrowing each split to 8
  is `torch.equal`/`np.array_equal` (sequences and lengths) to the same split generated at width 8.
- **AC-17:** A static test asserts that every `load_tensorised(` call outside `tests/` passes
  `max_seq_length=` and every `*_examples.pkl` read outside `data/preprocessing.py` goes through the
  narrowing helper; the evaluate, routing-diagnostics, gradient-diagnostics, personalize (2 users),
  federated (1 round, 2 clients), cold-start and long-tail paths run on the tiny fixture stored at
  width 16 with a width-8 model without error.
- **AC-18:** `TransformerModel.forward` on a `(B, max_seq_length + 1)` input raises `ValueError`
  naming both widths.
- **AC-19:** `load_tensorised(path, max_seq_length=W+1)` on data stored at width W raises
  `ValueError` naming `--stage preprocess` and `sequence_length_grid`.
- **AC-20:** A processed directory whose manifest has a different `user_sample_fraction` or
  `user_sample_seed` from the config makes `run_pretrain` and `run_ablation` raise before any
  training step.
- **AC-21:** Tiny CPU run, 3 epochs: interrupted at the first step of epoch 2 and re-run, it resumes
  at epoch 2 (no repeated epoch 0/1 steps) and its final parameters, `best_val_ndcg`, `best_epoch`
  and per-step LR trace are identical to an uninterrupted run; at resume
  `scheduler.last_epoch == global_step`; the per-epoch log has exactly one line per epoch and the
  GradNorm JSONL has no duplicate steps. Also passes with `num_workers=2`.
- **AC-22:** Resume is refused (with the checkpoint path and remedy in the message) when the stored
  fingerprint, `total_steps` or `source_revision` differ; `--no_resume` ignores `last.pt` and trains
  from scratch; re-running a completed run executes zero training steps and returns the same
  `best_val_ndcg`/`best_epoch`.
- **AC-23:** If the process dies while a checkpoint is being written (simulated by making the save
  raise mid-write), the previous `last.pt`/`best.pt` remain loadable and no partial file has the
  final name.
- **AC-24:** The OOM predicate is true for `torch.OutOfMemoryError(...)`,
  `RuntimeError("CUDA out of memory ...")` and `RuntimeError("MPS backend out of memory ...")` and
  false for other `RuntimeError`s, for host `MemoryError`, and for `ValueError`; `main.main([...])` raises
  `SystemExit(75)` when the stage raises OOM; `run_ablation` propagates an arm's OOM and still
  records a non-OOM arm failure as `status: "failed"`.
- **AC-25:** `run_with_retry.py` re-runs a child that exits 75 until it exits 0 (attempt count
  logged), stops after `--max-retries` with status 75, and stops immediately with the child's status
  for any other non-zero exit, without retrying.
- **AC-26:** While a subprocess holds a fingerprint's store lock, acquiring it with a 0.5 s timeout
  raises `TimeoutError`; when a waiting `_run_single_config` acquires the lock after the holder
  registered the backbone, it reuses that backbone and `run_pretrain` is never called.
- **AC-27:** Re-running a study whose arms all completed calls neither `run_pretrain` nor
  `evaluate_model` for those arms and reproduces the same study JSON; changing a post-training field
  of one arm (or the source revision) re-runs only that arm; `--no_resume` re-runs every arm's
  evaluation. Results land in `{ablation_dir}/seed_{seed}/{study}_{model}.json`.
- **AC-28:** On the tiny fixture `package_ablation_run.py` produces a code tarball whose member list
  is exactly the git-tracked files plus `launch/*` (no `data/`, `outputs/`, `.venv/`, `dist/`), a
  data tarball of the processed directory, and launch scripts in which the H100 runs studies
  A, D, E, H and AWS runs B, C, F, G in that order with the FR-12 flag set, and launch one-liners in
  `setsid nohup ... > <log> 2>&1 < /dev/null &` form; it exits non-zero when the manifest's
  fraction/seed/stored width do not match.
- **AC-29:** `_training_fingerprint` is unchanged by `num_workers`, `pin_memory` and
  `resume_training`, and changed by `persistent_workers`, `min_lr_ratio`, `user_sample_fraction`,
  `user_sample_seed` and `attention_backend`.
- **AC-30:** Each arm result in a study JSON carries `runtime.attention_backend`,
  `runtime.device` and `runtime.device_name`.
- **AC-31 (Mac smoke, real data, capped):** (a) `--stage preprocess --user_sample_fraction 0.1`
  completes and the manifest shows fraction 0.1, seed 42 and `stored_seq_length` 200; (b) a capped
  `--stage pretrain --model t-small` on that directory with `--dtype bf16 --mixed_precision
  --num_workers 2` completes; killed with `kill -9` after its first `last.pt` and re-run with the
  same command, it logs a resume at epoch 1 and completes; (c) a capped `--stage ablation --study B`
  run twice: the second run reuses both arms; (d) `package_ablation_run.py` builds `dist/` and the
  emitted commands match FR-12.
