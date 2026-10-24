# Code Quality

## Dtype Discipline

- Model and data computation uses `torch.bfloat16` by default. FP16 is allowed only under `mixed_precision=True` (fp32 master weights, autocast, GradScaler); pure FP16 weights are rejected by `Config.validate`.
- `resolve_dtype(config)` must raise for an unknown dtype and for `"fp16"` without `mixed_precision` (this line previously read "must raise if dtype is not bf16", which stopped matching the code when optional mixed precision landed in `c293616`).
- Experiment runs (ablations, final runs) use `--dtype bf16 --mixed_precision` — owner decision 2026-10-15; see `architecture.md`. The default is not changed to get there.
- Never silently change dtype if a numerical issue occurs — fail explicitly.

### Autocast scoping

These rules come from `specs/eval-integrity` FR-3, owner-directed on 2026-10-16.

- **One region per forward.** Each function that runs a model forward opens its own
  `utils.autocast_context(config, device_type=<the model's parameter device type>)` around that
  forward and its loss computation, and nothing else.
  - Backward, GradNorm updates, gradient clipping and `optimizer.step()` run **outside** autocast.
  - There is never a stage-wide or loop-wide region.
- **The weight cache stays off.** `autocast_context` uses `cache_enabled=False`, and code never
  uses a raw `torch.autocast` with the cache enabled around more than one forward.
  - Why: autocast caches the bf16 cast of every fp32 leaf parameter with `requires_grad` and clears
    the cache only when the outermost region exits. `optimizer.step()` and `load_state_dict()`
    update parameters in place, so a cached cast goes stale.
  - What happened: under the former stage-wide region in `main.py`, every `nn.Linear` forwarded
    its first-step weights for a whole run (measured: forward change 0.0 after three AdamW steps).
- **Autocast is a compute-dtype choice, never a scoring choice.** Ranking scores (L2
  normalisation and dot products) are computed in fp32 outside autocast. See "Evaluation
  Integrity".

## Device Handling

- Always use `get_device(config)` → `torch.device`. Never call `.cuda()` or `.to("cuda")` directly.
- `config.device = "auto"` resolves CUDA > MPS > CPU.
- Device synchronization is mandatory before latency timing: `torch.mps.synchronize()` on MPS, `torch.cuda.synchronize(device)` on CUDA.

## Attention Backend Discipline

- Sanctioned backends: `auto`, `flash_attention_3`, `sdpa`. **Owner decision 2026-10-15** (supersedes "CUDA/H100 → FlashAttention-4"): Hopper CUDA (compute capability 9.x) → FlashAttention-3 (`flash_attn_interface`); any other CUDA GPU → SDPA; MPS/CPU → SDPA. `"auto"` resolves by hardware. The former `flash_attention_4` backend was misnamed — it called the FA2 `flash_attn` API — and is removed.
- SDPA is the sanctioned kernel on MPS/CPU and non-Hopper CUDA, not a fallback. FlexAttention is not a supported backend in this project: it cannot run a backward pass on MPS in the pinned PyTorch build, so it can never satisfy the training gates on the dev host.
- FlashAttention-3 is an **optional, lazily imported** dependency: no module import, test collection or dev-host code path may require it. Tests exercise our side of the FA3 path (unpad/repad, causality, dispatch) with a pure-torch reference module; real-kernel tests are skipped unless Hopper + FA3 are present.
- Padded batches on FA3 go through our own differentiable unpad → varlen kernel → repad (FA3 ships no `bert_padding`). Causality is passed to every kernel as a flag (REV-122), never encoded in a mask only one backend reads.
- Each backend must pass a startup self-test — forward **and** backward, unpadded **and** left-padded, at the model's real head dimension — before training.
- A run that requests `flash_attention_3` explicitly must fail fast if it cannot initialize — no silent fallback. Experiment run commands always name the backend explicitly.
- MPS smoke test must fail fast if SDPA cannot execute — no silent fallback.
- Arms that are compared with each other must have run on the same backend; every ablation arm result records the backend and device it actually ran on.
- "No silent fallback" is about *honesty of the manifest*, not about kernel preference: the backend that actually ran is the backend that gets recorded, and `require_attention_backend=True` turns any substitution into a hard failure.
- Record PyTorch version, device, dtype, and resolved attention backend in every run manifest.

## Artifact Provenance

**Owner decision (2026-09-18).** Every artifact must be able to say which code produced it, from
the file itself and without reference to prose.

- Every checkpoint records a `source_revision`: the git commit when the tree is clean; otherwise
  the commit plus a content hash over the covered sources and a `dirty: true` marker. A run from a
  dirty tree is recorded as such, never silently as its nearest commit.
- Every artifact derived from a checkpoint — metrics, diagnostics, reports, round logs — echoes
  that checkpoint's `source_revision` verbatim.
- A verification note that quotes a number quotes it by reading the artifact back after the run,
  never from console output. A run that is later superseded then leaves a record that is visibly
  wrong rather than plausibly right.

### Coverage: a property, not a list (amended 2026-09-18)

**The governing rule is a property: `source_revision` must change whenever any code that can
change a published artifact changes.** Everything below is how that property is realised; if the
two ever conflict, the property wins and the realisation is the thing that is wrong.

Coverage is therefore defined by **exclusion, not enumeration**: every first-party `.py` file in
the repository is covered, except

- `tests/` — a test cannot change a published artifact;
- `.venv/`, `__pycache__/`, and anything not tracked by git;
- `data/raw/` and `data/processed/` — data, not code.

An allowlist was tried first and failed within one round: it named `models/`, `training/`,
`data/`, `configs/` and thereby omitted `evaluation/`, which is where every derived artifact in
this project is actually produced. Measured at the time: editing `evaluation/longtail.py` left
`source_revision` byte-identical. A denylist covers new code by default, so drift now requires
someone to write an explicit exclusion rather than merely to forget one.

Two constraints that follow, and are not optional:

- **The dirtiness check and the content hash use the same coverage rule**, so the two halves of a
  revision can never disagree about what "this code" means. A change to a covered file that moves
  neither the hash nor the `dirty` flag is a defect in the rule's implementation.
- **Coverage completeness is tested, not assumed.** A test asserts that the set of files actually
  hashed equals the set this rule defines. Asserting only that *some* change moves the hash proves
  the mechanism works, not that it covers what it must — that is precisely the gap that let the
  allowlist ship green.

**Why this exists.** Commit time is not edit time, and a change landing between a run and its
write-up leaves no trace in the artifact. That happened: three task notes gave three different
accounts of which objective trained a single checkpoint, none matched the files, and the question
could not be settled from the repository at all — the checkpoint stores `config`, which the change
in question did not touch. Provenance belongs in the artifact, not in prose about the artifact.

## Evaluation Integrity

These rules come from `specs/eval-integrity`, 2026-10-16. They apply to every path that reports a
ranking metric.

- **fp32 scores.** The forward may run in bf16, but scores are fp32. All scoring goes through
  `evaluation/metrics.py::score_catalogue_batch` (the full-catalogue protocol) or
  `score_candidates_batch` (explicit candidate rows, e.g. after ANN retrieval), and any other
  scoring site (`models/ann_retrieval.py`) follows the same rule. A guard test lists the
  allowed scoring sites.
  - Measured: bf16 cosines tied the target in 41–52% of examples and inflated NDCG@10 by up to
    0.08, by different amounts per arm.
- **Ties never favour the target.** Metrics are the expectation under random tie-breaking (the
  tie-averaged form in `compute_metrics_batch`).
- **A popularity floor beside every reported ranking result.** Every ranking result file and every
  ablation study carries a popularity-only reference: training-interaction counts, scored on the
  same rows and candidates. A gain is read against that floor, not against zero.
- **Comparisons use the full split.** Arms are compared on the full test split. A sample is
  allowed only when explicitly requested, and it must be seeded and recorded (method, size, seed,
  index hash) in the result file. Never score a prefix of a user-sorted split.
- **Self-describing results.** Result files carry an `eval_protocol` stamp. A number from an
  unstamped (pre-2026-10-16) file is not comparable with a stamped one.

## Configuration as Single Source of Truth

- `configs/config.py` (`Config` dataclass) holds all hyperparameters.
- No hardcoded hyperparameters in model/training/evaluation code.
- CLI flags override config values; log which values were overridden and original vs. new.

## Numerical Safety

- Assert finite losses, finite GradNorm loss, and finite active lambdas on every training step. The checks may be **batched** into stacked tensors (one host sync per check point) but never deferred past that step's `optimizer.step()`.
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)` on every step.
- `optimizer.zero_grad(set_to_none=True)` — use `set_to_none` for memory efficiency.

## Training Hot Path and Reproducibility

- Throughput changes must leave training semantics **bit-for-bit identical** (owner decision 2026-10-15): same batches, same RNG draws, same arithmetic. A change that alters an RNG stream (e.g. moving negative sampling to a device generator) or a kernel choice is a semantic change, not an optimisation.
- A **correctness fix** that changes what training produces for an unchanged config (for example `specs/eval-integrity` FR-3, the autocast scoping) is not a throughput change. It must bump `TRAINING_SEMANTICS_VERSION` in the training fingerprint, so backbones trained before the fix are never reused. The pure-dtype golden test must still pass with its constants untouched.
- Batch size is a training hyperparameter, not a utilisation knob: it sets InfoNCE's in-batch negatives, the number of optimizer and GradNorm steps, and the LR schedule. Do not change it for throughput.
- No per-step `.item()` / `.cpu()` / Python `bool()` of a device tensor in the training loop beyond the batched finite checks and syncs that are inherent to the computation (data-dependent shapes, the Python-side augmentation). Per-step loss values stay on the device and are materialised once per epoch.
- DataLoader workers must yield the same batch sequence as `num_workers=0` (tested). `persistent_workers` stays `False` for training: a persistent iterator does not redraw its base seed between epochs, which changes the loader generator's consumption and so the data order.
- Characterisation ("golden") tests that pin a tiny run's parameters, loss trace and LR trace are captured **before** a hot-path refactor and must stay green through it.

## Checkpoints and Resume

- Every checkpoint is written atomically (temporary file + `os.replace`); a crash mid-write never leaves a truncated file under a checkpoint's final name.
- Resume restores **all** state that affects the next step: model, MoE, loss-function parameters, optimizer, scheduler, GradScaler, GradNorm (incl. its optimizer and initial losses), every RNG (Python, NumPy, torch generators, global CPU/accelerator) and the DataLoader generators.
- Resume refuses — loudly, naming the checkpoint and the remedy — when the training fingerprint, `total_steps` or `source_revision` differ. A run assembled from two code revisions would violate "Artifact Provenance" below.

## Error Handling

- Missing Kaggle credentials with no local dataset → fail with clear auth message; never substitute another dataset.
- Missing checkpoint at evaluate stage → print skip message with remedy command, return — never raise exception.
- Negative pool exhaustion → fail with clear dataset/configuration error.
- `contrastive_interval = 0` is invalid; use `use_contrastive = False` to disable.
- Out-of-memory → the process exits with status 75 (`utils.OOM_EXIT_CODE`) so a launcher can retry; code that catches broad exceptions per item (e.g. per ablation arm) must re-raise OOM rather than record it as an ordinary failure.
- Data shape mismatches fail loudly at the boundary: a loader asked for more positions than are stored, a model fed more positions than it was built for, or a processed directory whose sampling parameters differ from the config all raise with the remedy command in the message.

## Anti-Patterns (Forbidden)

- No external recommendation libraries (RecBole, Surprise, pytorch-fm, etc.)
- No sinusoidal positional embeddings (learnable `nn.Embedding` only)
- GradScaler only on the fp16 mixed-precision path; never with pure-dtype weights
- No `num_items` as mask token (that collides with the last real item ID)
- No test data for model selection, threshold tuning, or diagnostic decisions
- No silent dtype/backend substitution
- No autocast region spanning more than one forward; no scoring under autocast or in bf16
- No evaluation on a prefix of a user-sorted split; no ranking metric that resolves ties in the target's favour
- No worker initialization when `num_workers=0`
- No per-epoch negative pool regeneration
- No cross-model state sharing between independently trained models
