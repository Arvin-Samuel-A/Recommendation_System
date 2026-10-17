# Handoff: full ablation run on a 10% user sample (master, bf16)

Context carried over from the previous session, for the PLAN agent. Everything under
"Findings" was measured or read from the code at commit `0a91094`; re-verify before
relying on line numbers.

## The request (owner's words, 2026-10-15)

1. Run the full ablation (all studies A-H).
2. Use FlashAttention-3 everywhere. Install it on the target machine (the shared H100) if missing.
3. Find the maximum utilisation (batch size) that does not OOM.
4. The cosine LR decay must have a minimum floor; today it reaches 0 at the end of training.
5. Use a random 10% of users for now; a full-data run comes later.
6. Fix the stored sequence width, or force the ablation to re-run preprocessing.

Constraints: on `master`, in bf16. Scope agreed earlier: **t-base only, one seed, all
ablations.**

## Repository state

- `master` is at `0a91094`, three commits ahead of `origin/master` (not pushed):
  - `203de3a` cross-study backbone store keyed by training fingerprint
    (`training/ablation_runner.py`: `backbone_store_dir`, `find_backbone`,
    `register_backbone`); multi-seed stage registers its per-seed checkpoints;
    `final_seeds` excluded from the fingerprint.
  - `c293616` optional mixed precision: `--dtype {bf16,fp16}` + `--mixed_precision`
    (fp32 master weights, autocast in every stage via `main.py`, GradScaler for fp16
    only); InfoNCE pinned to fp32; SDPA gets a boolean broadcast mask.
  - `0a91094` `scripts/estimate_training_time.py` (times every unique training config).
- Earlier, already pushed: `aeb658f` removed per-step device syncs.
- Tests: 299 pass. A full `pytest` run writes ~16 GB to pytest's temp dir (several tests copy
  ~2 GB of processed data each); the Mac's disk has filled up twice because of this.
- Locally, always run Python as `uv run python`.

## Findings that bear on each request item

### "bf16" — pure vs mixed (decision needed)
The default (`mixed_precision=False`) stores weights in pure bf16. Measured on a T4, t-small,
360 steps, schedule compressed to those steps, same seed:

| Precision | next-item loss at end |
|---|---|
| pure bf16 (default) | 4.56, flat after ~120 steps |
| bf16 mixed precision | 4.436, still falling |
| fp16 mixed precision | 4.436, still falling |

Cause: bf16 weights round away updates smaller than ~0.4% of the weight. **Recommendation:
interpret "bf16" as `--dtype bf16 --mixed_precision`.** Confirm with the owner. Steering
(`steering/architecture.md`, `code-quality.md`) was updated to allow this as an option, but
the default is still pure bf16.

### Item 1 — ablation scope and machine split
- 34 arms in `STUDY_CONFIGS` resolve to **18 unique training configs per seed** (16 if
  seq100/seq200 are dropped; they are not dropped here). The base config is shared by 13
  arms across A, B, C, D, E, F, G and H.
- Two machines are planned: the shared H100 and an AWS VM whose GPU is not decided yet.
  - Within a study, every arm should train on the same hardware, precision and attention
    backend; otherwise hardware differences show up as ablation effects.
  - A, D and E share backbones (A2=D1, A4=D3, D2=E1=E2), so keep them on one machine.
  - Proposed split: H100 → base, A, D, E, H; AWS → base, B, C, F, G. The base config is
    trained on both machines.
- The backbone store is per machine (on-disk) and has **no lock**: two processes on one
  machine can both start training the same fingerprint. Train the base config first, or
  run studies sequentially per GPU.
- With one seed, the per-seed results-path collision (`outputs/ablations/{study}_{model}.json`
  is not seed-namespaced) does not bite, but it will for later multi-seed runs.
- **Not requested, but strongly advised: resume-from-checkpoint.** `run_pretrain` saves every
  epoch but cannot resume. On the shared H100, other users' memory use can OOM a run
  mid-way and lose it entirely. Runs are ~2–9 h each at 10% (see below).

### Item 2 — FlashAttention-3
- **FA3 runs only on Hopper (sm_90: H100/H200).** "Everywhere" is impossible if the AWS VM
  is an L4/A100. Ask the owner: FA3 on Hopper with SDPA elsewhere? Or require Hopper on AWS
  too? The answer also determines whether the two machines' backends differ, which matters
  for the split above.
- Current code (`models/transformer.py`): backends are `auto`, `flash_attention_4`, `sdpa`
  (validated in `configs/config.py` and `VALID_BACKENDS`). The `flash_attention_4` branch
  actually imports the **`flash_attn` (FA2) package API**: `flash_attn_func`, plus
  `flash_attn_varlen_func` and `flash_attn.bert_padding` for padded batches
  (`_flash_attention_4_varlen`).
- FA3 is a different package: `flash_attn_interface`, built from the `hopper/` directory of
  the flash-attention repo. It is a long source build; compatibility with this torch
  (2.14.1+cu130) and CUDA 13 must be verified on the machine. FA3 does not ship
  `bert_padding`, so padded batches need their own unpad/repad for the varlen call.
- Every training batch is **left-padded**, so the varlen path is the hot path.
- Attention dropout is applied *after* attention (`attn_dropout`), not inside the kernel. FA3's
  lack of attention dropout is therefore not a blocker.
- **Steering conflict:** steering and README say "FlashAttention-4 on H100, fail fast". FA3
  replaces that, so the PLAN agent must update steering explicitly.
- Expected speed impact: at sequence length 50, attention is ~2–3% of the FLOPs, so FA3 changes
  step time by under ~3% there. It matters more for the seq100/seq200 arms.

### Item 3 — maximum utilisation without OOM
Measured at t-base, base config, bf16 mixed precision:

| GPU | batch 256 | batch 512 | larger |
|---|---|---|---|
| L4 (AWS g6.8xlarge, idle) | 125 ms/step, 2,041 ex/s, 3.5 GB | 268 ms, 1,910 ex/s, 6.6 GB | not run in that pass |
| H100 PCIe (shared, busy) | 146 ms/step, 1,753 ex/s, 3.6 GB | OOM (only 43–340 MB free: others held ~80 GB) | OOM |

- On the L4, a larger batch did not raise throughput.
- On the shared H100, the memory available depends on other users, so a batch that fits now
  can OOM later. That argues for headroom plus resume.
- A batch other than 256 changes training: InfoNCE uses in-batch negatives (2B−2), there are
  fewer optimizer steps and GradNorm updates, and the LR needs rescaling. The batch must also
  be identical for every arm within a study. The owner should confirm that changing the batch
  size is acceptable before "max utilisation" means "larger batch".
- Other utilisation levers that leave the batch unchanged:
  - `num_workers > 0`; batch loading is in the main process today.
  - Negative sampling on the GPU; `sample_training_negatives` costs ~8 µs per example on the CPU.
  - Several processes per GPU.
  - The remaining per-step `.item()` syncs.

### Item 4 — cosine LR floor
- `training/pretrain.py` `build_scheduler`: linear warmup, then
  `max(0.0, 0.5*(1+cos(pi*progress)))`, which reaches 0 at the final step.
- `tests/test_training.py::TestLRScheduler::test_cosine_reaches_near_zero` asserts
  `final_lr < 1e-5`. It must be updated together with the change.
- A floor needs a config field (e.g. a ratio of the peak LR) and a decided default. Check
  whether the LoRA/federated optimizers use their own schedules.

### Item 5 — 10% user sampling
- `data/preprocessing.py` `_read_ratings`: rating filter (`implicit_threshold`, ~line 192),
  then the min-interactions filter (~line 199). Sample users **after** these filters, with a
  seeded RNG.
- Everything downstream must be rebuilt from the subset: splits, negative pool, eval
  candidates, cold/warm items, popularity buckets. Record the fraction and its seed in the
  manifest / dataset fingerprint.
- Sample users, keeping each one's full history. Do not sample individual ratings, top items
  or truncated histories.
- Run preprocessing once and copy `data/processed/` to both machines, so they use the same
  sample.

### Item 6 — stored sequence width
- `data/sequence_generator.py` (~line 219) tensorises at `config.max_seq_length` (50).
  `load_tensorised(..., max_seq_length=N)` can only **narrow** (keeps the most recent N).
  So H2's seq100 and seq200 currently train on the same 50 items as seq50.
- Share of training examples with more than 25/50/100/200 prior items: 65% / 47% / 27% / 11%.
  Longer contexts carry real extra signal.
- Option A (recommended): store at `max(sequence_length_grid)` = 200 and narrow per run.
- Option B: re-run preprocessing per H2 arm. That means 4 preprocessing runs, and different
  artifacts per arm. Not recommended.
- **Hazard for option A:** these loaders do **not** pass `max_seq_length`. With 200-wide data
  they would feed 200 positions to a 50-position model:
  - `main.py:276` (test, evaluate stage)
  - `main.py:530` (val, routing diagnostics)
  - `main.py:572` (train, gradient diagnostics)
  - `training/federated.py:134` (val)

  Already narrowing: `training/pretrain.py:719,723` and `training/ablation_runner.py:391,439,516`.
  Also audit `training/local_train.py` and anything reading `user_sequences.pkl`.
- Stored size grows ~4×: `train_tensorised.pt` is 3.3 GB at width 50 on full data, so ~13 GB
  at 200. At 10% it is ~1.3 GB.

## Estimated time for this plan (t-base, 1 seed, 10% users, all arms)

From L4 measurements, with time scaled linearly by the user fraction:

| Part | Hours on one L4 |
|---|---|
| 16 arms at width ≤ 50 | ~29 h |
| seq100 (~2× per step) | ~4.5 h |
| seq200 (~4× per step) | ~9 h |
| **Total** | **~43 h** |

Split across the two machines, with the base trained twice, that is roughly one day each.
Caveat for the owner: one seed on 10% of users is noisy; differences below ~1–2% NDCG
between arms are not reliable.

## Machines and access

- **Shared H100** (institute, reached via VPN; Claude cannot be used while on the VPN):
  - Access: `ssh -p 52247 root@172.16.8.22`. No git or gh on the machine; copy code and data
    with `scp -P 52247` tarballs (`dist/` in the repo holds the last ones; `dist/` is in
    `.git/info/exclude`).
  - Environment: H100 PCIe 85 GB, 96 CPUs, Python 3.12.3, torch 2.14.1+cu130. FA3 status
    unknown.
  - GPU memory and compute are contended by other users.
- **AWS VM**: key `~/Downloads/Arvin.pem`, user `ec2-user` (root is refused).
  - Image: Deep Learning AMI (PyTorch 2.14, Amazon Linux 2023). The venv is
    `/opt/pytorch` (`source /opt/pytorch/bin/activate`); it already has every dependency.
  - Last seen as a g6.8xlarge (1× L4, 23 GB). Its public DNS changes on every stop/start, and
    the owner says the GPU type may change.
- Run long jobs detached with unbuffered output, e.g.
  `setsid nohup python -u ... > log 2>&1 < /dev/null &`. Piping through `grep` inside `tmux`
  lost the output of an earlier run.

## Open questions for the owner (ask before designing)

1. Should "bf16" mean `--dtype bf16 --mixed_precision`? (Recommended, given the plateau above.)
2. FA3 on non-Hopper GPUs is impossible. Should the AWS VM fall back to SDPA, or must it be
   Hopper?
3. Is a batch size other than 256 acceptable, given its effect on the contrastive loss and the
   optimizer? And what LR scaling rule should apply?
4. What default should the cosine LR floor have (e.g. 10% of peak)?
5. Should resume-from-checkpoint be added in this feature? (Strongly advised for the shared H100.)
6. Should `master` be pushed? The H100 has no git, so code still travels by `scp`.
