# Ablation run: errors found and how they were fixed

This file covers the 10%-user t-base A–H ablation on AWS (g6.12xlarge, 4× NVIDIA L4), between 2026-10-16 and 2026-10-17.

| Run | Code | Status |
|---|---|---|
| `~/run10` | `c094b2c` | **Void.** Every result is unusable (see 4.1) |
| `~/run11` | `4247429` (master) | The re-run with every fix below |

## Summary

| # | Problem | Effect | Fix | Commit |
|---|---|---|---|---|
| 1.1 | Old timing estimate understated seq100/seq200 | One GPU would have run 16.6 h, over budget | Re-measured; plan rebalanced | — (operations) |
| 2.1 | LoRA arms crashed on numpy `int32` in the result JSON | 3 arms lost after training finished | Plain Python types plus a JSON safety net | `aa6dc31` |
| 2.2 | A failed arm still exited 0 | The crash went unnoticed for hours | Exit status 70 for a failed arm | `b730d93` |
| 3.1 | Scores computed in bf16, with ties counted in the target's favour | NDCG/MRR inflated, and not equally across arms | fp32 scoring and honest tie handling | `f4d4d8a` |
| 3.2 | Test evaluation used the first 5,000 rows, sorted by user | Only 637 of 13,582 users were evaluated | Full test split by default | `f4d4d8a` |
| 3.3 | No popularity reference | Nobody could see the models only matched popularity | Popularity score in every result | `f4d4d8a` |
| 4.1 | **Stale autocast weight cache** | **All Linear layers frozen at their initial weights; run10 void** | Autocast opened per step, weight cache off | `f4d4d8a` |
| 4.2 | Broken models reusable by fingerprint | A re-run would have reused the broken backbones | Version number added to the fingerprint | `f4d4d8a` |
| 5.1 | Cosine logits with no temperature | Loss floors far above zero; models only tied popularity | `--logit_temperature` (τ = 0.07 in run11) | `fb34968` |
| 5.2 | Contrastive scaled 14× more than the other objectives | Contrastive gradients dominated GradNorm | One τ for all four objectives | `fb34968` |
| 5.3 | GradNorm pinned λ_contrastive near 1 | GradNorm could not balance contrastive | `--gradnorm_renorm_scope`; contrastive every step | `fb34968` |
| 5.4 | `contrastive_interval = 4` was never tested | Unjustifiable value, bursty gradients | Interval set to 1 in run11 | `fb34968` (flag) |
| 6.1 | The packager could not drop one sequence-length arm | No way to skip seq200 | `--sequence-length-grid` pass-through | `4247429` |

---

## 1. Planning and timing

### 1.1 The timing estimate understated the long-sequence arms
- **Symptom.** `estimate_gpu0.json` listed H2_seq100 and H2_seq200 at the same cost as the base config, about 2.2 h and 3.5 GB each. The smoke test then measured seq200 at **12 GB**.
- **Cause.** The estimate was timed on an old copy of the data, where the stored sequence width was smaller. So seq200 was silently cut down to the shorter width.
- **Fix.** Both arms were re-timed on the current code and the 10% data (old code): seq100 took 246 ms/step (4.4 h) and seq200 510 ms/step (9.2 h). Patching those two entries rebalanced the plan from a 16.6 h makespan to 12.3 h.
- **Lesson.** Re-measure the estimate whenever the code or the data changes. After the fixes in this file, the base config takes **169 ms/step** (it was 128) and seq200 **770 ms/step, 13.75 h**. Total: 59 GPU-hours, up from 43.5.

## 2. Failures that went unnoticed

### 2.1 LoRA arms crashed with `TypeError: Object of type int32 is not JSON serializable`
- **Symptom.** A: arv_mmoe_gradnorm_lora, E: global_lora and E: global_lora_arv each trained to the end, then failed while saving their result. Each was recorded as `failed`.
- **Cause.** `training/local_train.py` took `user_id` straight from a numpy array (`test_examples["user_ids"]`) and put it into `per_user_scores`. Python's `json` module refuses numpy integers.
- **Fix (`aa6dc31`).**
  - Every result field is now a plain Python value.
  - `utils.atomic_write_json` converts any remaining numpy or torch value and logs a warning naming the field.
  - A regression test runs E: global_lora on tiny data and checks that its result is saved with status `ok`.
  - An audit ran all 33 arms on tiny data and found no other numpy values in any result.
- **Verified.** In the AWS smoke test, the LoRA result was saved with status `ok` and `user_id` stored as an int.

### 2.2 A failed arm still exited 0
- **Symptom.** The launch scripts and the retry wrapper all reported success while three arms were failing.
- **Fix (`b730d93`).**
  - `--stage ablation` now exits **70** if any arm failed, after finishing the other arms and writing every record.
  - An out-of-memory error still exits 75 and is still retried. 70 is never retried.
  - `ablation_collect` and `ablation_rescore` also exit 70 when an arm failed.

## 3. Evaluation bugs (inflated or unreliable metrics)

### 3.1 bf16 scores and ties counted in the target's favour
- **Symptom.** The NDCG logged during training could not be reproduced from the saved checkpoint, and the gap between validation and test varied a lot from arm to arm.
- **Cause.** Every stage ran inside a single bf16 autocast region, so candidate scores came out in bf16. In **41–52 % of examples** the target scored exactly the same as another candidate, and `compute_metrics_batch` (`rank = 1 + #strictly higher`) counted every tie as a win for the target.
- **Measured.** Baseline checkpoint, full validation set:

  | Scoring | NDCG@10 | MRR@10 | Recall@10 |
  |---|---|---|---|
  | bf16 | 0.4639 | 0.3966 | 0.6834 |
  | fp32 | 0.3880 | 0.2969 | 0.6831 |

  The inflation differed by model: about +0.08 NDCG for the baseline, about +0.02 for the MoE arms. So it distorted comparisons between arms, not only the absolute level.
- **Fix (`f4d4d8a`).** Scores are converted to fp32 in every evaluation path, and each metric is reported as its expected value when ties are broken at random. A test with deliberately tied scores covers this.

### 3.2 The test sample was the first 5,000 rows
- **Cause.** The rows of `test_tensorised.pt` are sorted by user ID, so the first 5,000 covered only **637 of 13,582 users**.
- **Fix (`f4d4d8a`).** The full test split (107,282 examples) is used by default. A seeded random sample is still possible, and its size, seed and an index hash are recorded.

### 3.3 No popularity reference
- **Problem.** Nothing showed that the models were only matching a popularity ranker. Popularity scores 0.374 test NDCG (0.408 on validation) against the same candidates.
- **Fix (`f4d4d8a`).** Every result and every study JSON includes `popularity_reference`.

## 4. Training bug: run10 is void

### 4.1 The stale autocast weight cache froze every Linear layer
- **Cause.** `main.py` wrapped each whole stage in **one** `torch.autocast` region. Inside such a region, PyTorch caches the bf16 copy of each weight the first time it is used and keeps using that copy.
  - The optimizer did update the stored fp32 weights, but the forward pass never used the updates.
  - Every Linear layer (attention, feed-forward, MoE experts and gates, projections, LoRA) therefore stayed at its **initial random weights** during training. Only the embeddings and layer norms actually learned.
  - Gradients were always computed at the stale weights, so the stored fp32 weights drifted blindly.
- **Proof.** One Linear layer, 30 SGD steps under bf16 autocast:

  | Setup | Loss at step 0 | Loss at step 29 | Loss with the stored fp32 weights |
  |---|---|---|---|
  | One autocast region around all steps (old code) | 18.686 | **18.686** | 16,905 |
  | Autocast opened fresh each step (fix) | 18.686 | 0.000 | 0.000 |

- **What it explains in run10.**
  - Models only matched popularity.
  - The logged validation NDCG (0.536 for the baseline) could not be reproduced from the checkpoint (0.388 in fp32).
  - Federated validation NDCG was **exactly 0.3395 in every round**.
  - The ARV/MoE arms did worst, because their path has the most Linear layers.
  - All the GradNorm weight trajectories and gradient-conflict measurements came from a broken network.
- **Fix (`f4d4d8a`).** Autocast is opened per training step and per forward pass, with the weight cache off. After the fix, the logged validation NDCG matches a re-evaluation of the saved checkpoint exactly, and federated NDCG changes from round to round.
- **Consequence.** **Every run10 result is void**, and so is every conclusion drawn from it (about weighting, contrastive harm, PCGrad and gradient conflicts). Re-scoring the old checkpoints was skipped, because it would only have measured broken weights.

### 4.2 The broken models could have been reused
- **Fix (`f4d4d8a`).** The training fingerprint includes an evaluation-protocol / training version number, so a re-run retrains instead of reusing a backbone trained before the fix.

## 5. Model and objective design

### 5.1 Cosine logits with no temperature put a floor under the losses
- **Cause.** Next-item, masked-item and multi-step all scored with L2-normalised cosine similarity and **no temperature**, so every logit lay between −1 and 1. That caps how confident the softmax can get:

  | Loss | Candidates | Chance | Lowest possible at τ = 1 |
  |---|---|---|---|
  | next-item | 1 target + 100 negatives | 4.62 | 2.68 |
  | masked-item / multi-step | about 13.5k items | 9.51 | 7.51 |

- **Fix (`fb34968`).** New flags `--logit_temperature`, `--learnable_logit_scale` and `--logit_scale_max`. Run11 uses **τ = 0.07** for every arm, baseline included. Evaluation needed no change, because ranking is unchanged when every score is divided by the same positive number.
- **Evidence.** AWS smoke test, 2 epochs, full test split, fp32. Popularity scores 0.3736:

  | Arm | τ = 1 | τ = 0.07 |
  |---|---|---|
  | single_obj_baseline | 0.3753 | **0.4986** |
  | arv_mmoe_gradnorm | 0.3754 | **0.6097–0.6127** |

### 5.2 Contrastive was scaled 14× more than the other objectives
- **Cause.** Contrastive alone used τ = 0.07, which scales its logits by 14.3 relative to the other three. Its gradients dominated, and GradNorm compared objectives measured on different scales.
- **Fix.** Using τ = 0.07 for every objective puts all four on the same scale. (`--shared_logit_temperature` exists, but it isn't needed here because contrastive already uses 0.07.)

### 5.3 GradNorm pinned λ_contrastive near 1
- **Cause.** `training/gradnorm.py` rescaled only the objectives active on a given step so that they summed to their count. Contrastive ran on 1 step in 4. On the other 3 steps the remaining weights were rescaled to sum to 3; on each contrastive step all four were rescaled to sum to 4, which pulled λ_contrastive back toward 1. In run10 it sat at 0.90–0.94 in every arm.
- **Fix (`fb34968`).** `--gradnorm_renorm_scope enabled` rescales over every enabled objective. Run11 also sets `--contrastive_interval 1`, so every objective is active on every step and both scopes behave the same.
- **Observed** (smoke test, τ = 0.07, 2 epochs). With `enabled` scope, GradNorm drove λ_contrastive down to 0.0004. With `active` scope it stayed at 0.91. Test NDCG was 0.6127 vs 0.6097, a difference within noise.

### 5.4 `contrastive_interval = 4` was never tested
- **Cause.** It was a cost shortcut taken when the code was first written. The design note said contrastive would "double training time"; in fact it cost about +32 % step time (128 → 169 ms/step, measured). It also made contrastive's gradient arrive in bursts.
- **Decision.** Run11 uses `--contrastive_interval 1`. That needs no justification in the paper, and it avoids both the bursts and the GradNorm pinning.

## 6. Tooling and operations

### 6.1 The packager could not drop H2_seq200
- **Fix (`4247429`).** A new `--sequence-length-grid` option is passed to every rendered `main.py` command, collect included, so the planner and `ablation_collect` expect the same arms for study H. Run11 uses `25 50 100`.

### 6.2 Operational lessons
- **Smoke-test every kind of arm, not only the largest.** The first smoke test covered only seq200 and K8, so the LoRA crash (2.1) went unnoticed until hours into the run.
- **Stopping a run without stopping the instance.** The launch wrapper runs `run_aws.sh; sudo shutdown -h now`. Killing only the training processes lets the wrapper reach the shutdown. Kill the whole process group, wrapper included (`kill -- -<pgid>`), if the instance should stay up.
- **The DNS name changes every time the instance is stopped and started.** Log in as `ec2-user`; root is refused.
- **A detached launch over ssh can make the ssh session hang.** The `setsid nohup` job keeps running regardless, so check from a new session rather than killing it.
- **Short proxy runs can mislead.** A 6k-step local proxy suggested contrastive conflicted with next-item (cosine −0.31). The real, if void, run10 measurements pointed to multi-step instead. Never base a design change on proxy-scale gradient statistics.
