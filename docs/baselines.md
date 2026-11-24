# External baselines: SASRec, BERT4Rec, GRU4Rec, DuoRec, BSARec

Reviewers will ask how ARV-MMoE compares with published sequential recommenders. These five
run as ablation **study X**, through our data, training feed and evaluation, with each
model's **published hyperparameters, unchanged** (owner rule 2026-11-13: no tuning).

## Where the code is

| Piece | File |
|---|---|
| Architectures and losses | `models/baselines/models.py` (shared blocks in `common.py`) |
| Published hyperparameters, one source per value | `models/baselines/hparams.py` |
| Training loop, prefix feed, evaluation, arm result | `training/baseline_train.py` |
| Study X, fingerprint defaults, dispatch | `training/ablation_runner.py` (`baseline_arm`, `_run_single_config`) |
| Config fields | `configs/config.py`: `baseline_model`, `baseline_params`, `early_stop_patience` |
| Tests | `tests/test_baselines.py` |

The three new config fields default to "our model, no early stopping", and they're
left out of the fingerprint at those defaults, so no existing fingerprint changes.
`--study all` still means A–H. Study X runs only when it's named.

## Reference implementation

All five models follow the BSARec benchmark code: Shin et al., *An Attentive Inductive
Bias for Sequential Recommendation beyond the Self-Attention*, AAAI 2024 (arXiv
2312.10325), <https://github.com/yehjin-shin/BSARec>, commit `c80bdc0`. That repository
has no license, so the models are reimplemented from its behaviour, not copied. Cite it
together with the original papers: SASRec (Kang & McAuley, ICDM 2018), BERT4Rec (Sun et
al., CIKM 2019), GRU4Rec (Hidasi et al., ICLR 2016; Tan et al., DLRS 2016), DuoRec
(Qiu et al., WSDM 2022) and BSARec.

Its paper reports all five on Beauty and ML-1M under full ranking, which makes it the
reference for the port check.

## Hyperparameters (published, unchanged)

ML-20M uses the authors' ML-1M settings. Beauty uses their Beauty settings.

| | SASRec | BERT4Rec | GRU4Rec | DuoRec | BSARec |
|---|---|---|---|---|---|
| hidden size / layers | 64 / 2 | 64 / 2 | 64 / 2 (GRU hidden 64) | 64 / 2 | 64 / 2 |
| heads (ML / Beauty) | 2 / 1 | 2 / 2 | — | 2 / 2 | 4 / 1 |
| dropout (attention, hidden) | 0.5, 0.5 | 0.5, 0.5 | 0.5 (embedding) | 0.5, 0.5 | 0.5, 0.5 |
| max sequence length | 50 | 50 | 50 | 50 | 50 |
| batch size | 256 | 256 | 256 | 256 | 256 |
| optimizer | Adam, lr 1e-3, betas (0.9, 0.999), no weight decay | same | same | same | Adam, **lr 5e-4** |
| loss | BCE, 1 negative | CE (see below) | BPR, 1 negative | CE + 0.1 × InfoNCE (τ = 1, dot) | CE |
| model-specific | — | mask ratio 0.2 | — | `ssl = us_x`, `lmd_sem = 0.1` | α 0.3 / 0.7, c 9 / 5 (ML / Beauty) |

Sources (cited per value in `hparams.py`):

- the `parse_args()` defaults (the README says the baseline hyperparameters are there);
- the paper's setup section (D = 64, N = 50, L = 2, Adam, batch 256);
- Appendix Table 6 for BSARec, with the README's example commands. The released
  `BSARec_Beauty_best.log` agrees with Table 6.

The paper says the baselines were searched "based on their recommended hyperparameters",
but it doesn't publish per-dataset values for them. Where no value is published, the code
default is used. The one exception is SASRec on Beauty, which uses 1 head as in the README.
Neither the reference trainer nor ours uses an LR schedule or gradient clipping. Training
runs in fp32.

## What comes from our pipeline

- **Data:** our leave-one-out splits (`data/processed`, `data/processed_beauty`).
- **Training feed:** each user's whole training history, cut into chunks of 50 predictions
  (`load_train_dataset`).
  - The published losses score one next item from the last position of a left-padded
    sequence. The reference trains on every prefix.
  - BSARec's Fourier filter runs over the whole sequence, padding included, so it isn't
    causal. A chunk can't be scored in one forward pass.
  - So each chunk position with a target becomes one left-padded prefix example
    (`PrefixFeed`). Every training interaction is a target exactly once per epoch, as in
    our own training. On Beauty that gives 131,413 examples per epoch.
- **Evaluation:** full catalogue minus the user's history, fp32 scores, tie-averaged
  NDCG/Recall/MRR@10 (`evaluation/metrics.py`).
  - Validation runs every epoch, and the best epoch is chosen on validation NDCG@10.
  - The test score comes from `best.pt`.
  - Scores are the model's raw dot products, as in the reference. Our models' L2
    normalisation doesn't apply.
- **Epochs:** 20 on ML-20M. On Beauty, early stopping with patience 50 within at most 300
  epochs (`early_stop_patience`), the rule the early-stopped Beauty arms ran under.
- **Seeds, checkpoints, resume, result JSONs:** seeds 42/43/44, the by-fingerprint store,
  `best.pt` and `last.pt`, exact epoch-level resume, and arm JSONs in
  `ablations/seed_{S}/arms/X_t_base/`. Each arm JSON records the parameter count
  (`parameters.total_parameters`), `baseline_params`, the protocol and provenance.
- **Beauty eval width:** Beauty's processed splits are stored at width 20. For width 50,
  `load_eval_split` rebuilds the val and test sequences from the stored histories (the
  ones preprocessing used). It checks that the rebuilt sequences equal the stored rows on
  the last 20 positions.

## Deviations from the reference, and why

1. **Training examples.**
   - The reference uses each user's last 52 items: every prefix, each with a sliding
     context of up to 50 items, including the empty prefix that predicts the first item.
   - We use the whole history in chunks of 50. A target early in a chunk sees only the
     chunk's earlier items, and the first item of a history is never a target.
   - The rule is "our training feed". On Beauty, 5-core histories average 8.9 items, so
     nearly every user fits in one chunk and the two feeds almost coincide.
2. **Model selection and stopping.** We select on validation NDCG@10 with patience 50 (ML:
   20 fixed epochs). The reference selects on validation NDCG@20 with patience 10.
3. **Excluded history items.** The reference sets their scores to 0. We set them to −∞, our
   protocol's rule. With 0, an excluded item can still outrank items with negative scores.
4. **BERT4Rec.**
   - We reproduce what the reference code computes. It masks 20% of all 50 positions,
     padding included, and trains the last position's output to predict the next item.
   - The reference also builds the Cloze loss over the masked positions, but then
     overwrites it, so that loss never reaches the gradient.
   - Inference appends the MASK token, as in the reference.
   - This is the variant behind the published 0.0387. The original BERT4Rec paper trains
     with the Cloze loss.
5. **Unused parameters.** The reference's GRU4Rec also allocates a 50 × 64 position table
   it never uses. We leave it out, so our parameter count is 3,200 lower.
6. **Negatives and DuoRec positives.** These are sampled on the GPU from seeded generators.
   - A negative is uniform over items outside the prefix and the target.
   - DuoRec's same-target sequence is uniform over the other training examples with the
     same target. It's redrawn (up to 10 times) while it equals the example's own input.
     A target seen once uses the example itself, as in the reference.

## How to run

From the run directory on the VM, with the processed data linked in:

```bash
python main.py --stage ablation --dataset_preset amazon_beauty_2014 --study X --arms bsarec --seed 42 --num_workers 0
```

```bash
python main.py --stage ablation --study X --arms bsarec --seed 42
```

The first command runs an arm on Beauty, the second on ML-20M. The arm JSON goes to
`{ablation_dir}/seed_{S}/arms/X_t_base/{arm}.json`, and the per-epoch log to
`checkpoints/ablations/by_fingerprint/{fp}/logs/pretrain_t_base.log`. For the 3-seed table,
`scripts/analysis/agg_3seed.py` now includes study X. `per_user_eval.py` handles baseline
arms, and `paired_tests_3seed.py` then compares each baseline with the reference arm.

## Port check (Beauty, seed 42)

The port was checked by reproducing the BSARec paper's published Beauty numbers before the
3-seed runs. Results are kept out of the repository: see `outputs/baselines-port-check.md`
and `outputs/results_baseline.md`.

## 3-seed runs (launched 2026-11-13 17:52 IST, owner approved)

**Status: complete.** Beauty finished on 2026-11-13 and ML-20M on 2026-11-15, all 30 arms
from source hash `0b12075dc206`.

The runs are started by a gated scheduler, `scripts/vm/baseline_scheduler.py`
(`~/early_stop/` on the VM),
logging to `baseline_scheduler.log`.
- **Queue order:** Beauty seeds 43 and 44, then ML-20M DuoRec for seeds 42–44, then the
  other ML-20M arms.
- **Gating:** it starts an arm on a GPU only when the GPU's free memory, minus a 2 GB
  margin, covers the arm's measured peak. Peaks are 1.8 GB for Beauty, 2.4 GB for ML-20M,
  and 3.0 GB for ML-20M DuoRec.
- **Settings:** it ran at first with a 16 GB reserve on GPU 1 and at most 10 arms. From
  19:01 it ran with no reserve and at most 15 arms (the owner no longer needs GPU 1 space).
- **Restarts:** it skips finished and running arms. After a container restart, run
  `cd /root/early_stop && RESERVE_MIB= MAX_RUNNING=15 setsid nohup python3 -u baseline_scheduler.py >> baseline_scheduler.log 2>&1 < /dev/null &`
  again: unfinished arms resume from `last.pt`.

### Original plan

Every run goes from the frozen VM copy `~/baselines_run`. It holds this branch's code,
source hash `0b12075dc206`, with `data/processed` and `data/processed_beauty` linked in.
Seed 42 on Beauty already ran from there, so all seeds share one provenance. One launcher
covers both datasets, `~/early_stop/launch_baselines.sh`. It runs one process per arm,
through `run_with_retry.py`, which retries after an OOM. Its arguments are environment
variables: `DATASET`, `SEEDS`, `ARMS`, `GPU`.

1. **Beauty, seeds 43 and 44:** 10 arms of about 1 GB each.
   - Each arm takes 1–2.5 h on the shared GPU (DuoRec is the slowest).
   - Five at a time come to about 5 GB; the job takes about 3 h of wall time.

   ```bash
   DATASET=beauty SEEDS="43 44" GPU=1 bash ~/early_stop/launch_baselines.sh
   ```

2. **ML-20M, seeds 42, 43 and 44:** 15 arms.
   - SASRec, BERT4Rec, GRU4Rec and BSARec take about 11 h each. DuoRec takes about 27 h
     (20 epochs; measured with `scripts/vm/time_baseline.py`).
   - That is about 210 process-hours. Each process needs 1.2–1.8 GB.
   - Five at a time come to about 8 GB, and the job takes 2–3 days of wall time if
     per-process speed holds under that concurrency.

   ```bash
   DATASET=ml SEEDS="42 43 44" GPU=1 bash ~/early_stop/launch_baselines.sh
   ```

   Launch it staggered, one seed at a time, as GPU memory allows with the 2 GB margin.
3. **Container restarts (daily around 06:20):** re-running the same launcher command resumes
   every unfinished arm from its `last.pt`, losing at most one epoch. Finished arms are
   reused from their arm JSONs. The existing `auto_recover.sh` hook doesn't know about
   these runs. Either extend it or re-launch by hand after each restart.
4. **Analysis afterwards** (from the run dir):
   - `agg_3seed.py outputs/.../ablations` builds the study X table.
   - `per_user_eval.py OUT ARM_JSON...` saves per-user metrics for the X arms.
   - `paired_tests_3seed.py` compares the X arms with the reference arm's per-user files.
