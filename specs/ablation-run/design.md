# Ablation Run (10% users, t-base, one seed) — Design

## Re-verified findings (HEAD = `0a91094` + uncommitted `tests/test_preprocessing.py`)

Every handoff finding was re-read against the code. Line numbers below are current.

| Handoff claim | At HEAD | Notes |
|---|---|---|
| `flash_attention_4` branch imports the FA2 `flash_attn` API | `models/transformer.py:202-277` (`flash_attn_func`, `flash_attn_varlen_func`, `flash_attn.bert_padding`) | confirmed; also `VALID_BACKENDS` at :16 and `configs/config.py:331-335` |
| Training batches are left-padded; varlen is the hot path | `transformer.py:471-475` passes `padding_mask` whenever any pad exists | confirmed |
| Attention dropout is applied after attention | `transformer.py:352-355` | confirmed; FA3's lack of in-kernel dropout is irrelevant |
| `build_scheduler` reaches 0 | `training/pretrain.py:52-61` | confirmed; progress is also unclamped beyond `total_steps` |
| `test_cosine_reaches_near_zero` asserts `< 1e-5` | `tests/test_training.py:46-56` | confirmed |
| Rating filter / min-interactions filter | `data/preprocessing.py:192` / `:198-200` | confirmed |
| Tensorised at `config.max_seq_length` | `data/sequence_generator.py:219` | confirmed; `load_tensorised` (:332-362) can only narrow and silently returns narrower data |
| Loaders that do not narrow | `main.py:276`, `main.py:530`, `main.py:572`, `training/federated.py:134` | confirmed |
| … and the audit the handoff asked for | `training/local_train.py:229,238` (test/val `*_examples.pkl` sequences fed to the model), `training/federated.py:177,187` (same), `evaluation/cold_start.py:64`, `evaluation/longtail.py:119`, `scripts/estimate_training_time.py:232-236` (hardcodes width 50) | **new**: every `*_examples.pkl` reader feeds stored-width sequences to the model |
| Already narrowing | `training/pretrain.py:719-726`, `training/ablation_runner.py:391,439,516` | confirmed |
| `user_sequences.pkl` readers | written at `preprocessing.py:261`; **no reader** outside tests | nothing to fix |
| `run_pretrain` saves every epoch, cannot resume | `pretrain.py:923-939`; `load_checkpoint` (:129-151) restores optimizer/scheduler/GradNorm but nothing orchestrates a resume; the epoch log is opened `mode="w"` (:404) and the GradNorm JSONL is deleted at start (:830-832) | confirmed; training-pipeline T3.3 is checked but unimplemented |
| `run_ablation` | per-arm `except Exception` at `ablation_runner.py:567-576` records **every** failure, including OOM, and continues | **new**: an OOM never reaches a process-level retry |
| Backbone store has no lock | `ablation_runner.py:238-283` | confirmed |
| Results path not seed-namespaced | `ablation_runner.py:854-859` | confirmed |
| LoRA/federated schedules | `local_train.py` AdamW at constant `lora_lr`; no `LambdaLR` anywhere but `pretrain.py` | confirmed: no other schedule |
| Uncommitted `TestUserSampling` | `tests/test_preprocessing.py:712-768` expects `user_sample_fraction`, `user_sample_seed` (independent of `seed`), `--user_sample_fraction`, `--sequence_length_grid`, dir `data/processed_users10pct` | adopted as the target interface (FR-6) |

Per-step host syncs in the training hot path at HEAD (input to D4):

| Site | Syncs/step | Disposition |
|---|---|---|
| `pretrain.py:842` then `:620-621` — batch copied to device, then `targets`/`user_ids` copied back for negative sampling | 2 | removed (sample on the CPU batch) |
| `pretrain.py:296` `assert torch.isfinite(loss)` per active loss | ≤4 | batched into one stacked check |
| `pretrain.py:322` `assert torch.isfinite(L_total)` | 1 | folded into the same stacked check |
| `pretrain.py:356` grad-norm finite check | 1 | kept, folded with the lambda check |
| `pretrain.py:360-361` `.item()` per loss + total | ≤5 | removed (0-d tensors, materialised per epoch) |
| `pretrain.py:393-394` lambdas finite (`assert_finite_step`) | 1 | folded with the grad-norm check |
| `gradnorm.py` `if renorm > 0:` | 1 | removed (branch-free, identical arithmetic) |
| `transformer.py:472` `if has_padding` | 1 per forward | **kept**: removing it routes unpadded batches through SDPA's masked path, a different kernel, so not bit-identical |
| `losses.py` `MaskedItemLoss` boolean indexing, `MultiStepLoss` `valid_mask.sum() == 0` + indexing | 1 + 2×k | **kept**: data-dependent shapes; replacing them changes the arithmetic |
| `pretrain.py:263` `seq.cpu().tolist()` on contrastive steps | 1 every 4th step | **kept**: the augmentation runs in Python |

## Relevant Existing Architecture (reused)

- `configs/config.py::Config` — single source of truth; `validate()`; `asdict(config)` feeds both
  the run manifest and the training fingerprint.
- `main.py` — `parse_args` / `apply_cli_overrides` (logs `CLI override: field old -> new`),
  `_resolve_before_snapshot`, `save_config_snapshot`, one `autocast_context` around every stage.
  *Superseded 2026-10-16 by `specs/eval-integrity` FR-3.* The stage-wide region froze every
  `nn.Linear` at its first-step bf16 cast through autocast's weight cache. Each forward now opens
  its own region, with the cache off.
- `data/preprocessing.py::run_preprocessing` — single pass from `_read_ratings` through splits,
  cold items, metadata, sequences, pool, candidates, popularity buckets, manifest.
- `data/sequence_generator.py` — `generate_sequences` (int32 preallocated arrays),
  `tensorise_examples`, `save_tensorised`/`load_tensorised` (narrowing), `build_dataloader`
  (seeded `torch.Generator`, `seed_worker`), `sample_training_negatives` (bitmap rejection sampler).
- `models/transformer.py::AttentionBackend` — resolve → probe (fwd+bwd, unpadded and left-padded)
  → `_dispatch(backend, q, k, v, padding_mask, is_causal)`; causal intent travels as a flag
  (REV-122) and each backend formats its own mask.
- `training/pretrain.py::run_pretrain` / `train_one_step` / `save_checkpoint` / `load_checkpoint`.
- `training/ablation_runner.py` — `STUDY_CONFIGS`, `_training_fingerprint`, backbone store
  (`backbone_store_dir`, `find_backbone`, `register_backbone` with atomic marker),
  `_run_single_config`, `run_ablation`.
- `utils.py` — `get_device`, `resolve_dtype`, `resolve_param_dtype`, `autocast_context`,
  `make_grad_scaler`, `set_seed`, `get_source_revision` (works without git: hash only).
- Test pattern for FlashAttention without CUDA: a pure-torch reference module installed into
  `sys.modules` (`tests/test_transformer.py::TestFlashAttentionVarlen._install_reference_flash_attn`).

## Files

| File | Action | Role |
|---|---|---|
| `configs/config.py` | modify | new fields `min_lr_ratio`, `user_sample_fraction`, `user_sample_seed`, `resume_training`; validation; backend enum |
| `main.py` | modify | new CLI flags, derived processed dir, OOM exit code, narrowing at :276/:530/:572 |
| `utils.py` | modify | `is_oom_error`, `OOM_EXIT_CODE = 75`, `atomic_torch_save` |
| `data/preprocessing.py` | modify | user sampling, stored width, manifest fields, `check_processed_dir` |
| `data/sequence_generator.py` | modify | width guard in `load_tensorised`, `narrow_examples`/`load_examples` helpers |
| `models/transformer.py` | modify | FA3 backend, unpad/repad, preferred-backend function, probe head dim, position guard |
| `training/pretrain.py` | modify | LR floor, CPU-side negatives, sync reduction, `last.pt`, resume |
| `training/gradnorm.py` | modify | branch-free renormalisation |
| `training/ablation_runner.py` | modify | fingerprint exclusions, store lock, OOM re-raise, arm persistence, seed-namespaced paths, runtime record, processed-dir check |
| `training/local_train.py`, `training/federated.py`, `evaluation/cold_start.py`, `evaluation/longtail.py`, `scripts/estimate_training_time.py` | modify | route every sequence read through the narrowing helpers |
| `scripts/verify_attention_backend.py` | new | backend self-test + parity report |
| `scripts/fetch_fa3_source.sh` | new | Mac: flash-attention source tarball |
| `scripts/install_fa3.sh` | new | H100: preflight, build `hopper/`, verify |
| `scripts/run_with_retry.py` | new | OOM-aware retry launcher |
| `scripts/package_ablation_run.py` | new | `dist/` tarballs + launch scripts + `LAUNCH.md` |
| `tests/conftest.py` | new | tiny synthetic raw data → `run_preprocessing` fixture |
| `tests/test_determinism.py` | new | golden characterisation + workers equivalence |
| `tests/test_resume.py` | new | FR-8 |
| `tests/test_ablation_robustness.py` | new | FR-9 (in-process parts), FR-10, FR-11, FR-13 |
| `tests/test_scripts.py` | new | retry wrapper, packaging, verify script, `bash -n` |
| `tests/test_transformer.py`, `test_training.py`, `test_preprocessing.py`, `test_sequence_generator.py`, `test_cli.py` | modify | FA4→FA3 migration, LR floor, sampling/width, CLI |
| `README.md` | modify | CLI reference + runbook for the new scripts (steering-conflict edits already made by PLAN) |

No new third-party dependency. `fcntl` and `tarfile` are stdlib.

## Design Decisions

### D1 — Precision: CLI flags, not a default change (FR-1)
The ablation commands pass `--dtype bf16 --mixed_precision`; `Config` defaults stay. Mixed
precision is already implemented end to end (`c293616`). *Alternative rejected by the owner:* make
mixed bf16 the default.

### D2 — FlashAttention-3 backend with our own unpad/repad (FR-2)

**Naming and import.** Backend `flash_attention_3` replaces `flash_attention_4`. One helper
`_import_fa3()` does `import flash_attn_interface` and returns `(flash_attn_func,
flash_attn_varlen_func)`; an `ImportError` becomes `AttentionBackendError("FlashAttention-3
(flash_attn_interface) is not installed ...; see scripts/install_fa3.sh")`. The import is only ever
reached from `_probe`/`_dispatch` of the FA3 backend, so the Mac never touches it.

**Return value.** FA3 versions differ in whether `flash_attn_func`/`flash_attn_varlen_func` return
`out` or `(out, softmax_lse)`. `_fa3_output(result)` takes `result[0]` for a tuple, else `result`.

**Dtype.** FA3 kernels take bf16/fp16 only. The FA3 dispatch checks `q.dtype` and raises
`AttentionBackendError` otherwise. Under the ablation's mixed precision `q/k/v` come out of
autocast-ed `nn.Linear` in bf16.

**Dense path** (no padding mask, or a mask with no pad): `(B,H,S,D) → (B,S,H,D)` (a view back to
the contiguous projection layout), `flash_attn_func(q, k, v, causal=is_causal)`, transpose back.

**Varlen path** (any pad). Pure-torch, CPU-testable functions in `models/transformer.py`:

```python
@dataclass
class VarlenMeta:            # computed once per padding_mask object
    indices: Tensor          # (N_real,) int64, flat positions b*S+s of real tokens, row-major
    cu_seqlens: Tensor       # (B+1,) int32, on device
    max_seqlen: int          # host int, exact
    all_real: bool           # True -> caller uses the dense path

def varlen_meta(padding_mask) -> VarlenMeta
def unpad(x_bshd, meta) -> Tensor        # (N_real, H, D) via index_select (differentiable)
def repad(x_un, meta, B, S) -> Tensor    # zeros(B*S,H,D).index_copy(0, indices, x_un) -> (B,S,H,D)
def varlen_attention(q, k, v, padding_mask, is_causal, varlen_fn, meta=None) -> Tensor  # (B,H,S,D)
```

`varlen_meta` reads `seqlens = padding_mask.sum(1, dtype=int32)` to the host once (one sync: gives
`max_seqlen`, and `all_real` when every row is full), and only when a pad exists computes
`indices = nonzero(padding_mask.flatten())` (a second sync). It assumes nothing about where pads
are, so a correct result does not depend on the left-padding convention — but under left padding
each row's real tokens are contiguous and in order, which is what makes causal attention over the
ragged buffer equal to causal attention with a key-padding mask.

`varlen_fn` is injected: production passes FA3's `flash_attn_varlen_func`; tests pass a pure-torch
reference (per-sequence SDPA). It is called with keyword arguments
`cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=m, max_seqlen_k=m, causal=is_causal`.

**One metadata computation per forward.** All L blocks receive the same `padding_mask` object from
`TransformerModel.forward`. `AttentionBackend` caches `(padding_mask, VarlenMeta)` and reuses the
meta when `cached_mask is padding_mask`. The cache holds a strong reference, so the object cannot be
freed and its identity cannot be reused by a different mask while cached. This cuts the varlen syncs
from 2·L to 2 per forward.

Pad positions come back as exact zeros (SDPA leaves its own values there). Pad positions never act
as keys for real queries and no loss reads them, so this is not a semantic difference; parity tests
compare real positions only.

*Alternatives considered:* (a) keep FA2's `bert_padding` — FA2 is not FA3 and may not be installed
alongside it; (b) pass the padded buffer with `seqused_q/k` — those mean "first L tokens of each
segment", i.e. right padding, so left-padded rows would attend to pads; (c) pass `max_seqlen=S` to
save a sync — the kernel contract for an upper bound is unverified on the target build, and a wrong
result there is silent; correctness first.

**Backend selection.** A pure function `preferred_backend(device_type: str, capability:
Optional[Tuple[int, int]]) -> str` (cuda & major == 9 → `flash_attention_3`; otherwise `sdpa`);
`AttentionBackend._preferred_backend` calls it with `torch.cuda.get_device_capability(self.device)`
on CUDA. Tests call the function directly and monkeypatch the method for resolution tests.

**Probe.** `_probe` uses `D = config.hidden_dim // config.num_heads` (80 at t-base, 64 at t-small
and t-large) instead of the hardcoded 16, so the startup gate exercises the kernel instance training
uses. `B, H, S` stay small.

**Runtime record.** `_run_single_config` adds
`result["runtime"] = {"attention_backend": model.backend.backend_name, "attention_fallback_reason":
..., "device": str(device), "device_name": torch.cuda.get_device_name(device) if cuda else
platform.processor() or device.type, "dtype": cfg.dtype, "mixed_precision": cfg.mixed_precision}`.

**One backend per study.** Enforced operationally: each study runs on one machine (D12) with an
explicit `--attention_backend`, which is also part of the training fingerprint, so an arm trained
under one backend is never served to an arm requesting another.

### D3 — FA3 install and verification tooling (FR-3)

`scripts/verify_attention_backend.py` (Python, runs anywhere):
`--backend {sdpa,flash_attention_3} --model {t-small,t-base,t-large} [--head-dims 64,80]
[--seq-lens 50,200] [--report PATH]`. Steps: build `AttentionBackend` with the explicit backend
(→ exit 2 with the reason on `AttentionBackendError`); `self_test()`; for each head dim and length,
random bf16 q/k/v with left-padded rows of lengths {S, S/2, 1} plus an unpadded batch, run the
backend forward+backward and an fp32 SDPA reference (masked) on the same device; compare real
positions; tolerance `atol=2e-2, rtol=2e-2` for bf16 outputs and grads (exact tolerance recorded in
the report). Report keys: `torch_version`, `torch_cuda`, `device`, `device_name`, `capability`,
`backend`, `backend_module` (`flash_attn_interface.__file__` / version when FA3), `cases[]` with
`head_dim, seq_len, padded, max_abs_out_diff, max_abs_grad_diff, passed`, `passed`. Exit 0/1/2.

`scripts/fetch_fa3_source.sh [REF] [OUT_DIR]` (Mac; default `REF=main`, `OUT_DIR=dist`):
`git clone --depth 1 --branch REF --recursive --shallow-submodules
https://github.com/Dao-AILab/flash-attention.git` into a temp dir, write the resolved commit to
`FA3_SOURCE_COMMIT`, drop `.git` directories, `tar czf OUT_DIR/flash-attention-src.tar.gz`, print
size and commit. (`csrc/cutlass` is a submodule FA3 needs — hence `--recursive`.)

`scripts/install_fa3.sh TARBALL [--dry-run] [--force] [--max-jobs N] [--build-dir DIR]` (H100):
1. Preflight, each with a named failure message and exit 2: `python` ≥ 3.10; `import torch`;
   `torch.cuda.is_available()`; capability major == 9; `torch.version.cuda` major equals the major
   of `nvcc --version` (searching `PATH` then `$CUDA_HOME/bin`); `ninja` and `packaging`
   importable/available; ≥ 20 GB free in the build dir. `--dry-run` stops here and prints the build
   command it would run.
2. If `python -c "import flash_attn_interface"` succeeds and not `--force`: skip to step 5.
3. Extract to `BUILD_DIR` (default `$HOME/fa3_build`); `cd flash-attention/hopper`.
4. Export `FLASH_ATTENTION_DISABLE_{FP8,FP16,PAGEDKV,APPENDKV,SOFTCAP,LOCAL,SPLIT,SM80,HDIM192,
   HDIM256}=TRUE` (keep HDIM64/96/128 — head dim 80 rounds up to the 96 kernels — VARLEN and
   BACKWARD). For each name, warn if it does not occur in `hopper/setup.py` (a renamed flag would
   otherwise silently build everything). `FA3_KEEP_FP16=1` keeps fp16. `MAX_JOBS` defaults to
   `min(nproc/2, RAM_GB/8)`. Build with
   `nice -n 10 python -m pip install --no-build-isolation --no-deps -v .` (builds against the
   installed torch; never upgrades it), tee'd to `logs/fa3_build.log`.
5. Run `python scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base`
   from the repository root; its exit status is the script's.

The build is long and runs on a shared box, so `LAUNCH.md` runs it detached with
`setsid nohup bash scripts/install_fa3.sh ... > logs/fa3_install.log 2>&1 < /dev/null &`.

### D4 — Utilisation without changing a bit (FR-4, FR-13)

**DataLoader workers.** `--num_workers N` sets `config.num_workers`; `build_dataloader` already
passes `worker_init_fn=seed_worker`, `pin_memory`, and a seeded `generator`. `RecDataset` is
deterministic and index-only, so worker processes produce the same batches. The loader draws
`_base_seed` from its generator in the iterator constructor and the permutation on first `next`, in
the same order for single- and multi-process iterators — which is why the order is identical.
`persistent_workers` must stay **False**: a persistent iterator is `_reset` between epochs without
redrawing `_base_seed`, so the generator is consumed differently from epoch 2 on and the data order
changes. That is also why `persistent_workers` stays in the fingerprint while `num_workers` and
`pin_memory` leave it.

**Host→device copies.** In the step loop, negatives are attached to the CPU batch first
(`attach_training_negatives` on CPU tensors: same inputs, same `neg_rng` draws, same output), then
every tensor is moved with `non_blocking=(device.type == "cuda" and config.pin_memory)`.

**Finite checks, batched.** In `train_one_step`:
- Check A (before backward): `torch.isfinite(torch.stack([l.float() for l in active_losses.values()]
  + [L_total.float()])).all()` — one sync; on failure, compute per-loss values only then, to keep the
  current error message naming the offending loss.
- Check B (after clipping, before `optimizer.step()`, non-scaler path): stacked check of
  `total_grad_norm` and, when GradNorm is on, `gradnorm.lambdas` — one sync. With a GradScaler the
  grad-norm part is skipped exactly as today (the scaler skips the step), the lambda part stays.
Both still run on every step and still abort before the optimizer step, satisfying
`steering/code-quality.md` "Numerical Safety".

**Loss values.** `train_one_step` returns `{name: loss.detach()}` (0-d tensors) plus `"total"` and
the `_skipped` list. `run_pretrain` keeps per-epoch lists of these tensors and materialises them
once per epoch (`torch.stack(...).tolist()`), producing the same floats for the per-epoch means,
loss windows and logs. `assert_finite_step` no longer runs per step (its checks moved into A/B);
callers outside `run_pretrain` (`scripts/estimate_training_time.py`, tests) keep working because a
0-d tensor supports `float()`, `math.isfinite()` and `format()`.

**GradNorm renormalisation.** Replace the `if renorm > 0:` branch and per-index loop with
`idx = tensor(active_indices); lam = self.lambdas[idx]; scale = torch.where(renorm > 0, num_active /
renorm, 1.0); self.log_lambdas[idx] = (lam * scale).log()`. Same elementwise arithmetic, no host
branch; a test compares it with the old loop via `torch.equal`.

**Proof of identity.** `tests/test_determinism.py::test_golden_tiny_run` is written and its
constants recorded **before** any D4 change (task T3.2), from a tiny CPU run with a fully pinned
config; it must stay green through D4 and D8. It records `torch.__version__` and `platform.machine()`
and skips elsewhere. A separate test proves AC-10 (workers) directly.

*Rejected levers:* larger batch / LR rescaling / data-parallel / concurrent arms (owner decision 3);
batch probing (owner); **GPU negative sampling** — it would draw from a device RNG stream, so the
negatives (and every downstream number) differ from the CPU sampler even though the distribution is
the same; the owner's bar is bit-identical training semantics, which it cannot meet; removing the
`has_padding`, MIM and multi-step syncs (table above).

### D5 — Cosine LR floor (FR-5)
```python
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
```
With `min_lr_ratio = 0` this is the old expression for every `step ≤ total_steps` (the old
`max(0.0, ·)` is a no-op there because the cosine term is ≥ 0); the clamp only changes steps past
`total_steps`, which `run_pretrain` never reaches. LoRA (`train_user_lora`) and federated clients
build `AdamW(lr=config.lora_lr)` with no scheduler — nothing to change; T3.1 records the check.

### D6 — User sampling (FR-6)
In `_read_ratings(config, sampling_info: Optional[dict] = None)`, after the min-interactions
filter:
```python
fraction = config.user_sample_fraction
if fraction < 1.0:
    users = np.sort(df["userId"].unique())
    n_keep = max(1, int(round(fraction * len(users))))
    keep = np.random.RandomState(config.user_sample_seed).choice(users, size=n_keep, replace=False)
    df = df[df["userId"].isin(keep)]
```
`sampling_info` (when given) receives `user_sample_fraction`, `user_sample_seed`,
`num_users_before_sampling`, `num_users_after_sampling`, `user_sample_fingerprint`
(`sha256` of the sorted kept original ids as int64 bytes, first 16 hex). `process_ratings` passes a
dict and merges it into `stats` (→ `dataset_stats.json`); `run_preprocessing` copies the same fields
into `preprocessing_manifest.json`. `get_timestamps_per_user` calls `_read_ratings` too and gets the
same subset (same seed, same sorted input). With `fraction == 1.0` the code path is skipped
entirely, so output is unchanged. `run_preprocessing` calls `config.validate()` first.

A separate `user_sample_seed` keeps the data subset fixed while the training `seed` varies (the
pre-written test requires this). `apply_cli_overrides` derives the directory (FR-6) only when
`processed_data_dir == Config.processed_data_dir` (the default), and logs it as an override.

`check_processed_dir(config)` in `data/preprocessing.py`: read `preprocessing_manifest.json` if
present; compare `user_sample_fraction` (absent → 1.0) and, when < 1, `user_sample_seed`; raise
`ValueError` naming both values and the `--stage preprocess` command otherwise. Called at the top of
`run_pretrain` (after `validate`) and once at the top of `run_ablation`. Absent manifest → warning
only (slow tests build manifest-less subsets).

### D7 — Stored width and narrowing (FR-7)
- `stored_width = max([config.max_seq_length] + list(config.sequence_length_grid))`;
  `generate_sequences(..., width: Optional[int] = None)` uses it instead of `config.max_seq_length`
  (default `None` → `config.max_seq_length`, keeping existing callers/tests valid).
  `run_preprocessing` passes `stored_width` and writes `"stored_seq_length"` to the manifest.
- `sequence_generator.narrow_examples(examples: dict, max_seq_length) -> dict` returns a shallow
  copy whose `sequences` are `[:, -N:]` (NumPy view or list slice per row) and `lengths`
  `np.minimum(lengths, N)`; raises `ValueError` if `N` exceeds the stored width.
  `load_examples(path, max_seq_length)` = unpickle + `narrow_examples`.
- `load_tensorised(..., max_seq_length)` raises `ValueError` when `max_seq_length` exceeds the
  stored width ("stored at W, requested N: re-run `--stage preprocess` with `--sequence_length_grid`
  covering N").
- Every reader passes `config.max_seq_length`: `main.py:276,530,572`, `federated.py:134,177,187`,
  `local_train.py:229,238`, `cold_start.py:64`, `longtail.py:119`; the estimate script uses
  `cfg.max_seq_length` instead of 50. The static audit test (AC-17) walks the AST of every
  git-tracked non-test `.py` file: each `load_tensorised(...)` call must have a `max_seq_length`
  keyword, and no `open(...)` call whose arguments contain a string constant ending in
  `_examples.pkl` may appear outside `data/preprocessing.py` (the writer) — readers must call
  `load_examples(...)`, which opens the file itself. (Pattern: `tests/test_cli.py`
  `TestEvaluationStagesResolveTheirNames` guards a defect class the same way.)
- `TransformerModel.forward`: `if S > self.position_embedding.num_embeddings: raise ValueError(
  f"input has {S} positions but the model was built with max_seq_length=
  {self.position_embedding.num_embeddings} ...")` — a shape check, no sync. On CUDA the alternative
  is a device-side assert that poisons the context.
- Narrowing equivalence holds because `_left_pad`/`_fill_left_padded` keep the tail and
  `lengths = min(len(context), S)`; slicing the tail of a wider row and clamping its length yields
  the same values (AC-16 proves it on every split).
- Memory note: at 10% users, train tensorised at width 200 is ~1.3 GB (int64); full data would be
  ~13 GB — acceptable now, a known cost for the later full-data run.

### D8 — Epoch-granular exact resume (FR-8)

**Atomic writes.** `utils.atomic_torch_save(obj, path)`: save to `path + ".tmp-<pid>"`, `fsync`,
`os.replace`. `save_checkpoint` and the new `last.pt` write use it.

**What `last.pt` holds** (written after `best.pt` at the end of every epoch): the full
`save_checkpoint` payload (model, MoE, loss fns, optimizer, scheduler, GradNorm incl. its optimizer
and `initial_losses`, config, `source_revision`) plus

```python
"resume_state": {
    "format": 1, "epoch": epoch, "global_step": global_step, "total_steps": total_steps,
    "training_fingerprint": _training_fingerprint(config),
    "best_val_ndcg": ..., "best_epoch": ..., "epoch_mean_losses": [...], "epoch_loss_windows": [...],
    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
    "rng": {"python": rng.getstate(), "numpy_neg": neg_rng.get_state(),
            "torch_mask": mask_rng.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if cuda else None,
            "torch_mps": torch.mps.get_rng_state() if mps else None},
    "loader_generators": {"train": train_loader.generator.get_state(),
                          "diag": diag_loader.generator.get_state()},
}
```
`_training_fingerprint` is imported lazily from `training.ablation_runner` (no import cycle:
`ablation_runner` imports `pretrain` only inside functions).

**Resume flow in `run_pretrain`** (after every object is constructed, before the epoch loop): if
`config.resume_training` and `<ckpt_dir>/last.pt` exists → load; refuse (`RuntimeError` naming the
file, the mismatching field and "delete it or pass --no_resume") if the fingerprint, `total_steps`,
or `source_revision` (`commit`, `source_hash`, `dirty`) differ; restore every object and RNG above;
`start_epoch = epoch + 1`; log `Resuming <model> from <path> at epoch <start_epoch> (global step
<g>)`. The epoch loop runs `range(start_epoch, pretrain_epochs)`; a completed run skips straight to
the end-of-training routing pass and returns the restored result.

Restoring the global torch RNG **after** model construction matters: construction consumed it; the
restored state is what dropout in the next epoch must see.

**Log hygiene on resume.** `setup_epoch_logger(..., append=True)` (no truncation) after rewriting the
file without lines whose `epoch=` is ≥ `start_epoch`; the GradNorm JSONL is rewritten keeping only
records with `step < global_step` instead of being deleted.

**Why epoch-granular.** It restores the loader exactly by generator state alone. Mid-epoch resume
would need a resumable sampler, partial epoch-loss lists and routing buffers; at 10% users an epoch
is ~3,045 steps (≈6–8 min at the measured 125–146 ms/step), so the most an OOM can cost is one such
epoch. The routing buffer is not persisted: it is a FIFO of the last 256 batches and an epoch has
far more, so after an epoch boundary it refills identically.

**Bit-identity scope.** On CPU the resumed run is bit-identical (AC-21). On CUDA, kernels with
atomics (embedding backward, FA3 backward) are not bit-deterministic run to run anyway; resume
guarantees the same state, LR schedule and data order.

### D9 — OOM: exit code 75 + external retry (FR-9)
`utils.is_oom_error(exc)`: `isinstance(exc, torch.OutOfMemoryError)` or (`isinstance(exc,
RuntimeError)` and `"out of memory" in str(exc).lower()`). `main.main` wraps the stage call:
on an OOM error it logs `OOM in stage <s>: <msg> -- exiting with status 75 for the retry launcher`
and `raise SystemExit(OOM_EXIT_CODE)`. `run_ablation`'s per-arm handler re-raises when
`is_oom_error(exc)`.

`scripts/run_with_retry.py [--max-retries 20] [--wait-seconds 300] -- CMD...`: `subprocess.Popen`
with inherited stdout/stderr (so the nohup log captures the child), `wait()`, loop on status 75,
`time.sleep` between attempts, print `[run_with_retry <iso time>] attempt k/N ...` with flush.
SIGTERM/SIGINT handlers terminate the child and exit 143/130 without retrying.

*Alternatives:* in-process retry (rejected: a fragmented CUDA allocator and half-built state are
exactly what a fresh process avoids); parsing logs for "out of memory" (rejected: brittle, and the
exit code is cheaper and testable).

### D10 — Store lock (FR-10)
```python
@contextmanager
def backbone_lock(checkpoint_root, fingerprint, timeout=None, poll=5.0):
    store = backbone_store_dir(checkpoint_root, fingerprint); os.makedirs(store, exist_ok=True)
    fd = os.open(os.path.join(store, ".train.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    # flock(LOCK_EX | LOCK_NB) in a loop; on first EWOULDBLOCK log the holder pid read from the file
    # once; raise TimeoutError after `timeout`; on success write "<pid> <host> <iso time>"
    ...  # finally: flock(LOCK_UN); os.close(fd)
```
In `_run_single_config`, the `find_backbone → run_pretrain → register_backbone` sequence moves inside
`with backbone_lock(...)`, with `find_backbone` called again after acquisition. `--no_train` takes
no lock. `flock` locks are released by the kernel when the process dies, so an OOM-killed run never
leaves a stale lock. *Alternative rejected:* `O_EXCL` lock files (stale after a crash).

### D11 — Arm persistence and seed namespace (FR-11)
- `_results_root(config) = os.path.join(config.ablation_dir, f"seed_{config.seed}")`; study JSON,
  summary and arms go under it.
- `_arm_key(cfg, study, label, model_name)`: `sha256(json.dumps({"config": {asdict(cfg) minus
  _PATH_FIELDS minus _RUNTIME_ONLY_FIELDS}, "study", "label", "model", "source": {commit,
  source_hash, dirty}}, sort_keys=True, default=str))[:16]`.
- In `run_ablation`, per arm: if `config.resume_training` and
  `<root>/arms/{study}_{mn}/{label}.json` has `status == "ok"` and the same `arm_key`, load it,
  mark `"reused_from_previous_run": true`, and skip `_run_single_config`; otherwise run it and, on
  `status == "ok"`, write the arm file atomically (tmp + `os.replace`) with its `arm_key`.
- `run_federated` / `run_local_personalization` are not made resumable: they run after the backbone
  exists, take minutes rather than hours at 10%, and an arm interrupted there simply re-runs them
  with its backbone reused from the store.

### D12 — Packaging and launch (FR-12)
`scripts/package_ablation_run.py` arguments (defaults in brackets): `--user-sample-fraction [0.1]`,
`--user-sample-seed [42]`, `--seed [42]`, `--model [t-base]`, `--batch-size [256]`,
`--num-workers [8]`, `--h100-studies [A,D,E,H]`, `--aws-studies [B,C,F,G]`,
`--h100-backend [flash_attention_3]`, `--aws-backend [sdpa]`, `--h100-python [python3]`,
`--aws-venv [/opt/pytorch]`, `--max-retries [20]`, `--retry-wait [300]`, `--out [dist]`,
`--processed-dir` (default: derived exactly as `apply_cli_overrides` derives it — the function is
shared, not duplicated), `--h100-ssh` / `--aws-ssh` (optional; placeholders in `LAUNCH.md`
otherwise), `--skip-data`.

Steps: validate the processed manifest (fraction, seed, `stored_seq_length ≥
max(sequence_length_grid)`) → render `launch/run_h100.sh`, `launch/run_aws.sh`, `launch/LAUNCH.md`
from one command-builder function (so scripts and docs cannot drift) → `bash -n` each script →
write `dist/code.tar.gz` from `git ls-files -z` (working-tree contents) plus `launch/` → write
`dist/<processed dir basename>.tar.gz` (members under `data/<basename>/`) → print the launch
one-liners and tarball sizes. It warns (does not refuse) if covered sources are dirty: the
`source_hash` travels with the code and identifies it.

`run_h100.sh` shape:
```bash
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs
PY="${PYTHON:-python3}"
"$PY" -u scripts/verify_attention_backend.py --backend flash_attention_3 --model t-base \
    --report logs/verify_fa3.json > logs/verify_fa3.log 2>&1 || { echo "FA3 verification failed"; exit 1; }
for STUDY in A D E H; do
  echo "$(date -Is) study ${STUDY} start"
  "$PY" -u scripts/run_with_retry.py --max-retries 20 --wait-seconds 300 -- \
    "$PY" -u main.py --stage ablation --study "${STUDY}" --model t-base --seed 42 \
      --user_sample_fraction 0.1 --user_sample_seed 42 --dtype bf16 --mixed_precision \
      --batch_size 256 --attention_backend flash_attention_3 --num_workers 8 \
      > "logs/ablation_${STUDY}_t_base_seed42.log" 2>&1
  echo "$(date -Is) study ${STUDY} exit $?"
done
```
`run_aws.sh` is the same with `source /opt/pytorch/bin/activate`, `PY=python`, no FA3 gate,
`--attention_backend sdpa`, studies B C F G. Launch one-liner (per machine, from the extracted
directory): `mkdir -p logs && setsid nohup bash launch/run_h100.sh > logs/run_h100.log 2>&1 <
/dev/null &`.

### D13 — Fingerprint hygiene (FR-13)
`_RUNTIME_ONLY_FIELDS = frozenset({"num_workers", "pin_memory", "resume_training"})` is excluded in
`_training_fingerprint` alongside the existing sets. Adding new `Config` fields changes every
fingerprint, so backbones registered before this feature are not reused — intended, since the LR
floor changes training anyway.

### D14 — Test strategy
- **Tiny synthetic data (T0.1).** `tests/conftest.py` provides a session-scoped
  `tiny_processed_dir` factory: writes the six raw CSVs for ~60 users × ~150 items (ratings 5.0,
  string timestamps increasing per user; items 141–150 occur only as some users' last two
  interactions so the `unseen_in_training` cold set is non-empty; genome/genre rows for every item)
  into `tmp_path_factory`, then runs the real `run_preprocessing` with `max_seq_length=8`,
  `sequence_length_grid=[4, 8, 16]`, `num_sampled_negatives=10`, `seed=42`, optionally
  `user_sample_fraction`. A `tiny_config(processed_dir, tmp)` helper returns a CPU config (t-small,
  `batch_size=16`, `pretrain_epochs=3`, `max_steps_per_epoch=4`, outputs under `tmp`). The fixture
  asserts the directory stays < 20 MB.
- **Golden (T3.2)** and **workers** (T5.1) in `tests/test_determinism.py`; tests call
  `utils.set_seed(cfg.seed)` before each run and may pin `torch.set_num_threads(1)`.
- **FA3 without CUDA:** reference `flash_attn_interface` module in `sys.modules`; the real-kernel
  test is `skipif` unless capability 9.x and the import succeed.
- **Scripts:** run as subprocesses with `uv run python` / `bash -n`; packaging runs against the tiny
  fixture with `--out` in `tmp_path`.
- **Execution:** only the files a task names, with `-m "not slow"`; never the full suite.

## Data Flow (changes in bold)

```
rating.csv → filters → **seeded user sample** → ID maps → splits → cold/warm → metadata
  → sequences **at stored width max(grid ∪ {max_seq_length})** → tensorised + *_examples.pkl
  → pool, candidates, popularity → manifest **(+sampling fields, stored_seq_length)**

run_ablation → **check_processed_dir** → per arm: **reuse persisted arm?** → fingerprint
  → **lock** → find_backbone | run_pretrain(**resume from last.pt**) → register → unlock
  → evaluate (**narrowed loaders**) → extras → **persist arm** → seed_{seed}/{study}_{model}.json
OOM anywhere → **exit 75** → run_with_retry waits → same command → reused arms + resumed arm
```

## Error Handling

| Condition | Behaviour |
|---|---|
| Explicit `flash_attention_3` unavailable / not Hopper | `AttentionBackendError` at model construction (fail fast) |
| FA3 input not bf16/fp16 | `AttentionBackendError` naming the dtype |
| Requested width > stored width | `ValueError` with remedy (loader) |
| Model input wider than `max_seq_length` | `ValueError` naming both widths |
| Manifest sampling ≠ config | `ValueError` with the preprocess command |
| `last.pt` from another config/code | `RuntimeError` naming file, field, remedy |
| OOM | exit 75 → retry launcher; in-study arms re-raise |
| Other arm failure | recorded `status: "failed"`, study continues (unchanged) |
| Lock held | wait (logged once); `TimeoutError` only when a timeout is passed |

## Security / Operational Notes
- No credentials, hostnames or keys are written into the repository; `LAUNCH.md` uses placeholders
  unless `--h100-ssh/--aws-ssh` are given at packaging time (and `dist/` is git-excluded).
- The FA3 build uses `--no-build-isolation --no-deps` so it cannot replace the machine's torch.

## Steering Conflicts Resolved (owner decisions 2026-10-15)
- "CUDA/H100 → FlashAttention-4, fail fast" → FA3 on Hopper, SDPA on other CUDA, fail fast only
  when FA3 is explicitly requested (`steering/code-quality.md`, `architecture.md`, `testing.md`,
  README updated by PLAN).
- Experiment runs use mixed bf16; default unchanged (`architecture.md`, `code-quality.md`).
- New durable rules: stored width vs model width (`conventions.md`), dev-host test hygiene and tiny
  data (`testing.md`), batched per-step finite checks and atomic checkpoints (`code-quality.md`).

---

# Extension design: splitting a study's arms across GPUs (FR-14–FR-18, owner 2026-10-16)

Built on the code at `e0729ab` (re-read for this extension; see requirements "Re-verified at
`e0729ab`"). Reused as-is: `resolve_study_configs` / `_study_entries`, `_training_fingerprint`,
`backbone_lock` + `find_backbone` re-check, `run_pretrain` resume from `last.pt`, per-arm files
(`_arm_path`, `_arm_key`, `_load_persisted_arm`, `_write_json_atomic`), `_save_study_results`,
`_plot_study`, `_STUDY_PLOTS`, `_save_summary`, `main.is_oom_error` → exit 75,
`scripts/run_with_retry.py`, and the packager's command builder, readiness check and tarballs.

## Files (extension)

| File | Action | Role |
|---|---|---|
| `training/ablation_runner.py` | modify | `parse_arms`/`select_study_configs`, `resolve_dataset_fields`, generic `file_lock` (backbone lock delegates), study lock + `write_study_outputs`, `assess_study`/`collect_study`, failed-arm persistence, arm-scoped side outputs, `run_ablation(..., arms=None)` |
| `main.py` | modify | `--arms`, `--allow_incomplete`, stage `ablation_collect`, atomic `run_config.json` with the selection |
| `utils.py` | modify | `atomic_write_json` (ablation runner's `_write_json_atomic` delegates to it) |
| `training/pretrain.py` | modify | device-count-tolerant CUDA RNG restore; save the current CUDA device index |
| `scripts/package_ablation_run.py` | modify | GPU options, planner (groups, cost model, LPT), per-GPU scripts, driver, plan JSON, LAUNCH.md |
| `tests/test_ablation_parallel.py` | new | FR-14–FR-17 in-process tests and real-subprocess concurrency tests |
| `tests/test_scripts.py` | modify | FR-18 packager tests |
| `tests/test_resume.py` | modify | AC-41 |
| `README.md` | modify | `--arms`, `ablation_collect`, multi-GPU packaging |

## D15 — Arm selection (FR-14)

```python
def parse_arms(text: Optional[str]) -> Optional[List[str]]   # None -> None; "a, b,a" -> ["a","b"]; "" / " , " -> ValueError
def select_study_configs(study, base_config, arms) -> List[Config]
    # arms None -> resolve_study_configs(study, base_config) unchanged
    # study == "all" -> ValueError("--arms needs exactly one --study ...")
    # unknown -> ValueError(f"Study {study} has no arm(s) {bad}; valid labels: {labels}")
    # else the resolved configs for `arms`, in the given order
```
`run_ablation(config, model_name, study="all", train=True, arms=None)` calls
`select_study_configs` **first** (before `check_processed_dir`), so a typo costs nothing.
Labels come from `_study_entries`, so grid labels follow the run's `sequence_length_grid` /
`expert_count_grid`. `main.py`: `--arms` (str, default `None`); `main()` raises `ValueError` if
`--arms` is given with a stage other than `ablation`; `run_ablation_stage` passes
`arms=parse_arms(args.arms)`. A `ValueError` exits 1 through Python's default handler, which
`run_with_retry.py` does not retry. `--arms` is not a `Config` field: like `--study`, it selects
runs and must not enter the fingerprint or `arm_key`.

## D16 — Study outputs written once, under a lock (FR-15)

- `file_lock(path, timeout=None, poll=..., on_wait=None)`: the `flock` loop extracted from
  `backbone_lock`, which keeps its signature, log message and behaviour (its tests stay green).
- `_study_lock(config, study, model_name)` = `file_lock({root}/arms/{study}_{mn}/.study.lock)`.
- `write_study_outputs(config, model_name, study, results)`: under the study lock,
  `_save_study_results` (already atomic) → `_plot_study` → `_STUDY_PLOTS[study]` (exceptions in
  named figures still only logged). The full-run path (no `--arms`) calls it in place of the
  three calls it makes today: same files, same content plus `code_revision`.
- Arm persistence: before writing, every result gets `arm_key` and
  `code_revision = get_source_revision()`; the arm file is written for `ok` **and** `failed`
  results (the `except` branch builds the failed record as today, then persists it). OOM still
  re-raises before anything is written.
- Why per-arm files + assembly rather than merging into the study JSON under a lock: the arm files
  already exist and are written atomically by exactly one process each; a study JSON that grows
  arm by arm would be a partial study that plots and readers could mistake for a complete one.

## D17 — Assessment and collection (FR-15)

```python
@dataclass
class StudyAssessment:
    labels: List[str]                 # expected, study order
    ok: Dict[str, dict]; failed: Dict[str, dict]
    missing: List[str]; stale: List[str]
    inconsistent: List[str]           # human-readable "field X differs: a=..., b=..."
    @property
    def complete(self): return not (self.missing or self.stale or self.failed or self.inconsistent)

def assess_study(config, model_name, study) -> StudyAssessment
def collect_study(config, model_name, study, allow_incomplete=False) -> StudyAssessment
```
- `assess_study` resolves the expected labels with `resolve_study_configs(study, config)`, reads
  each arm file, and classifies: no file → missing; `code_revision.source_hash` ≠ this process's
  `get_source_revision()["source_hash"]` → stale; `status == "failed"` → failed; else ok.
  Comparing `source_hash` only (not `commit`/`dirty`) lets a clean checkout of the same commit
  collect results produced from the git-less tarball (same tracked `.py` files, same hash).
- Consistency over ok + failed records: `runtime.attention_backend` and `runtime.device_name`
  equal across records that have `runtime`; and for every key of `resolved_config` not in
  (union of the study's `_study_overrides` keys) ∪ `_PATH_FIELDS` ∪ `_RUNTIME_ONLY_FIELDS`, the
  value is equal across records. Each violation is one `inconsistent` entry naming the field and
  the labels/values.
- `collect_study` takes the study lock, calls `assess_study` (inside the lock, so it sees every
  file persisted before the lock was granted), then: inconsistent → log, write nothing; complete
  or `allow_incomplete` → build the list in label order (ok record; failed record; or
  `{"label", "study", "status": "missing"}`; stale records are treated as missing), drop
  `reused_from_previous_run`, and write via the unlocked body of `write_study_outputs`.
- **Auto-collect** (`run_ablation` with `arms`): after the arm loop, `collect_study(...)` inside
  `try/except Exception` (logged, never raised — the arms themselves succeeded). If incomplete, log
  `study X: N/M arms complete; missing [...] failed [...] stale [...] -- study JSON and plots
  deferred to the process that completes the study or to --stage ablation_collect`. Safety
  argument: each process persists its arm files before requesting the lock; the files of the last
  arm to be persisted therefore exist when its process assesses, so that process sees a complete
  study. Two processes that both see a complete study write identical outputs one after the other.
- **Stage `ablation_collect`** (`main.run_ablation_collect_stage`): `--study X` or `all`;
  `--allow_incomplete`; no `check_processed_dir`, no data, no model. For `all` it also writes
  `summary_{model}.json` with `_save_summary` over assembled studies plus `"complete": false,
  "missing": [...]` entries for the rest. Exit `SystemExit(1)` if any requested study was not
  assembled. With `--allow_incomplete` a partially assembled study counts as assembled (exit 0,
  warning logged), because the flag means "I accept a partial study"; an inconsistent study is
  never assembled and always exits 1.

## D18 — Waiting on and taking over a backbone (FR-16)

Already implemented by T7.3 and FR-8; this extension pins it with tests and fixes one gap.
- Waiter: `find_backbone` miss → `backbone_lock(timeout=None)` → polls every 5 s, logs the
  holder once → after acquiring, `find_backbone` again → reuse (no `run_pretrain`).
- Holder death: `flock` is released by the kernel (REV-010: after its forked DataLoader workers
  exit, seconds). The waiter then trains; `run_pretrain` resumes the holder's `last.pt` because the
  store directory, fingerprint, `total_steps` and `source_revision` are identical.
- Gap fixed: `_capture_rng_state` also stores `torch_cuda_index = torch.cuda.current_device()`;
  `_restore_rng_state` uses `set_rng_state_all` when the saved list length equals
  `torch.cuda.device_count()` (unchanged path), otherwise sets the saved state at
  `torch_cuda_index` (default 0 for older checkpoints, clamped to the list) on the current device
  and logs a warning. Under `CUDA_VISIBLE_DEVICES=k` every process sees one device, so the normal
  path is unchanged.
- Process placement avoids the idle-waiter case entirely: the planner never splits a fingerprint
  group across GPUs (D20). Waiting remains the correct fallback for hand-written commands.

## D19 — Per-process output isolation (FR-17)

- `_run_single_config(..., scoped_outputs=False)`: when true (`run_ablation` passes
  `arms is not None`), immediately after `resolved_config` is captured, `cfg.log_dir` and
  `cfg.plot_dir` become `{base}/arms/{study}_{label}` for the arm's extras. Training already uses
  its fingerprint log directory (REV-002 swap). `_PATH_FIELDS` keep both out of the fingerprint and
  `arm_key`, and the consistency check excludes them.
- `utils.atomic_write_json(path, payload)` (temp + fsync + `os.replace`);
  `save_config_snapshot` uses it and adds `"selection": {"stage", "study", "arms"}` to the
  manifest (passed in from `main()`; `None` fields when absent).

## D20 — Packager planner and multi-GPU launch (FR-18)

**Machine GPUs.** `gpu_ids(args, machine) -> Optional[List[str]]`: explicit
`--{machine}-gpu-ids` wins (validated: integers, unique, non-empty); else
`--{machine}-gpus` or `--gpus` (≥ 1); a count of 1 without explicit ids returns `None` = the
legacy single-GPU path, rendered by today's `render_script` untouched (AC-43). Count N > 1 →
`["0", ..., str(N-1)]`. `--h100-studies` ∩ `--aws-studies` ≠ ∅ → refuse.

**Resolving arms exactly as the run will.** New shared helper in `training/ablation_runner.py`:
```python
def resolve_dataset_fields(cfg: Config, stats_dir: Optional[str] = None) -> None
    # num_items, num_users, vocab_size from {stats_dir or cfg.processed_data_dir}/dataset_stats.json;
    # feature_dim from load_metadata_matrix(...) -- exactly what _run_single_config does today
```
`_run_single_config` calls it (no behaviour change). The packager builds the base config with
`main.parse_args(<the study command's argv>)` → `apply_cli_overrides` → `apply_model_config`,
so `processed_data_dir` holds the same relative path the remote will use, then for every arm of
`resolve_study_configs(study, base)` calls `resolve_dataset_fields(cfg, stats_dir=<local
processed dir>)` and `_training_fingerprint(cfg)`. AC-45 proves equality with
`_run_single_config`.

**Cost model.**
- Training hours per fingerprint group: if `--estimate-json` is given, the max `run_hours` over
  the group's arms found in `models[<model>].configs[*]` (rows matched by `"study:label"`;
  the estimate must have been produced with the run's `--user-fraction`, epochs and batch);
  otherwise `fallback_training_hours(cfg) = base(S) × (user_sample_fraction / 0.1) ×
  (pretrain_epochs / 20)` with `base(S) = 1.81` for `S ≤ 50` and `4.5 × S / 100` above (handoff:
  16 arms ≤ 50 ≈ 29 h, seq100 ≈ 4.5 h, seq200 ≈ 9 h on one L4 at 10%). The plan records which
  source was used per group.
- Post-training overhead per arm (constants, documented as rough): 0.05 h, +0.25 h when
  `personalization_mode == "lora"` and the study has the `personalization` extra, +0.5 h when
  `system_eval == "communication"`, +0.1 h for the `gradient` extra, +0.05 h for `efficiency` or
  `system_eval == "latency"`. Not scaled by user fraction (fixed user/client counts).

**Assignment.** `lpt_assign(groups, n_gpus) -> List[List[Group]]`: groups sorted by
`(-hours, first_arm_sort_key)`; each to `argmin(load)` (lowest index on ties). Pure function,
unit-tested. Example on the default AWS studies with `--aws-gpus 4` and fallback hours: 9 groups
(one of 4 arms, eight singletons) → loads ≈ {4.0, 3.8, 5.6, 3.8} h, makespan ≈ 5.6 h. With H moved to AWS the
seq200 group (~9 h) sits alone on one GPU and bounds the makespan.

**Commands per GPU.** The GPU's arms sorted by (machine study order, study order); consecutive
arms of one study → one command `study_command(args, backend, py, study=X, arms=[...])`.
`study_command` gains optional `study`/`arms` parameters; with neither it renders today's string
(legacy byte-identity).

**Emitted files** (machine with N > 1):
```bash
# launch/run_aws_gpu2.sh
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."; mkdir -p logs
<today's machine setup block: PY / venv activation + torch import check>
export CUDA_VISIBLE_DEVICES=2
<FA3 gate iff backend is FA3, report logs/verify_fa3_gpu2.json>
"$PY" -u scripts/run_with_retry.py --max-retries 20 --wait-seconds 300 -- \
  "$PY" -u main.py --stage ablation --study B --arms input_only <FR-12 flags> \
  > logs/ablation_t_base_seed42_gpu2_01_B.log 2>&1
echo "$(date -Is) gpu2 command 1 (B: input_only) exit $?"
...
# launch/run_aws.sh (driver)
<setup>
for G in 0 1 2 3; do bash "launch/run_aws_gpu${G}.sh" > "logs/run_aws_gpu${G}.log" 2>&1 & done
wait
for STUDY in B C F G H; do
  "$PY" -u main.py --stage ablation_collect --study "${STUDY}" <FR-12 flags> \
    > "logs/collect_${STUDY}_t_base_seed42.log" 2>&1
  echo "$(date -Is) collect ${STUDY} exit $?"
done
```
`launch/plan_{machine}.json`: `{"machine", "gpu_ids", "hours_source", "groups": [{"fingerprint",
"arms": ["S:label", ...], "training_hours", "overhead_hours", "hours", "gpu"}], "gpus": {id:
{"arms", "hours"}}, "makespan_hours"}`. `LAUNCH.md` gains, for a multi-GPU machine, the plan
table, per-GPU one-liners `mkdir -p logs && setsid nohup bash launch/run_{m}_gpu{id}.sh >
logs/run_{m}_gpu{id}.log 2>&1 < /dev/null &` (for manual use), and the collect commands. The
driver one-liner is unchanged. Every generated script is checked with `bash -n` as today.

*Alternatives considered:* (a) let reusing arms of a group run on other GPUs after the trainer —
needs cross-process dependency scheduling and leaves GPUs idle on the lock; (b) a dynamic work
queue — needs a coordinator process and makes the run harder to reason about and reproduce; a
static, reviewable plan in `plan_*.json` was preferred; (c) ignoring groups — duplicated waiting
on the shared base group (4–9 arms) would serialise most of the machine behind one lock.

## D21 — Tests (extension)

- `tests/test_ablation_parallel.py` (new):
  - in-process: `parse_arms`/`select_study_configs` errors and order (AC-32/33), full-run vs
    collect equality and `code_revision` (AC-34), `assess_study` classification incl. stale
    (patched `source_hash`) and inconsistency (edit a persisted record's `batch_size` /
    `runtime.device_name`) (AC-36), arm-scoped outputs + atomic `run_config.json` (AC-42), OOM
    under `--arms` then re-run (AC-40, in-process like `TestArmPersistence`).
  - real subprocesses: a small driver script written to `tmp_path` that puts the repo on
    `sys.path`, builds `tiny_config(...)` from `tests.conftest`, applies optional monkeypatches
    named by argv (e.g. "hold after epoch 0 until a sentinel file disappears"), and calls
    `run_ablation(...)`/`collect_study(...)`; launched with `uv run python <driver>` and
    `subprocess.Popen`; every wait is bounded (e.g. 120 s) and every child is killed in a
    `finally`. Used for AC-35, AC-37, AC-38, AC-39.
- `tests/test_resume.py`: AC-41 with `torch.cuda` functions monkeypatched (no CUDA needed).
- `tests/test_scripts.py`: AC-43–AC-46 against `tiny_processed_dir(user_sample_fraction=...)`
  with `--processed-dir`, `--skip-data`, `--allow-dirty` where the tree is dirty, `--out tmp_path`.
- Rules unchanged: targeted files only, `-m "not slow"`, tiny data, `uv run`.

## Steering (extension)

`steering/architecture.md` "Run Execution" and `steering/conventions.md` "Output Organization"
are updated by PLAN for: parallel arms on GPUs of one machine, study outputs assembled from
persisted arms under a per-study lock, the `ablation_collect` stage, `.study.lock`. The
ablations-cli Non-Goal "Parallel ablation execution" carries a supersession note.
