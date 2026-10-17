# Ablation results: run12 (10-epoch A–H, 10% users)

**Run:** AWS g6.12xlarge (4× L4), `~/run12`, 2026-10-17 13:02–19:19 UTC. Code is commit `b673002` (branch `claude/run11-followup-36a3c6`). All 33 arms have status `ok` and all 8 studies were collected. Results were copied to `outputs/aws_2026-10-17/run12/` in the main checkout.

**Settings (every arm):**
- t-base, seed 42, 10% of users (sample seed 42), batch 256, bf16 mixed precision, SDPA.
- **10-epoch schedule**: warmup 10%, cosine decay to 0.1× the peak LR.
- Logit temperature τ = 0.07 for all four objectives; contrastive every step.
- Sequence-length grid 25/50/100.

**Reference model (the "full" ARV-MMoE):**
- ARV routing, 4 experts, objective-conditioned router, metadata on, sequence length 50.
- All four objectives, balanced by **next-item-primary GradNorm**: λ_next_item is fixed at 1, and GradNorm balances masked-item, multi-step and contrastive under a total budget **β = 0.3**. β was chosen on validation in a separate sweep (see below).

**Evaluation:** full test split (107,282 rows), fp32 scoring, ties counted at their expected rank, eval protocol v2. Test metrics are reported at the **best validation epoch**. The popularity-only reference scores **NDCG@10 = 0.374** (MRR 0.294) on the same rows and candidates.

**Read with care:**
- **One seed.** The sampling standard error of a mean NDCG@10 over 107k rows is about 0.001. Seed-to-seed variance hasn't been measured, so differences under about **0.01** should be treated as ties.
- **The single-objective baseline hadn't converged at 10 epochs.** Its validation NDCG was still rising at epoch 9: 0.628 here, against 0.690 on run11's 20-epoch schedule (test 0.657). The multi-objective arms converge by epoch 4–9, so the 10-epoch schedule barely changes them (arv_mmoe: 0.650 here, 0.649 on 20 epochs). The owner chose to keep every arm at 10 epochs, so study A's baseline gap is an *equal-budget* result: the multi-objective models learn faster.

## All arms (test NDCG@10 at the best validation epoch)

Arms with the same training fingerprint share one backbone, so they have identical metrics. The reference backbone `17bcde2f65e8` serves 13 arms.

| Study | Arm | Backbone | Best epoch | Val NDCG | **Test NDCG** | Test MRR | Recall@10 |
|---|---|---|---|---|---|---|---|
| A | single_obj_baseline | aa8ea802 | 9 | 0.628 | **0.594** | 0.516 | 0.840 |
| A | 4obj_no_routing | d08164f7 | 4 | 0.652 | **0.613** | 0.539 | 0.845 |
| A | conv_mmoe | f4b4966b | 5 | 0.688 | **0.648** | 0.579 | 0.867 |
| A | arv_mmoe (equal weights) | c225ac49 | 5 | 0.688 | **0.650** | 0.580 | 0.868 |
| A | arv_mmoe_gradnorm (reference) | 17bcde2f | 8 | 0.686 | **0.649** | 0.580 | 0.865 |
| A | arv_mmoe_gradnorm_lora | 17bcde2f | 8 | 0.686 | **0.649** | 0.580 | 0.865 |
| B | objective_conditioned (reference) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| B | input_only | a03b7fa5 | 9 | 0.684 | **0.647** | 0.577 | 0.865 |
| C | arv_mmoe (reference) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| C | mmoe (expert count matched) | 098432b8 | 9 | 0.683 | **0.645** | 0.575 | 0.864 |
| D | none_no_gradnorm | d08164f7 | 4 | 0.652 | 0.613 | 0.539 | 0.845 |
| D | none_gradnorm | 218045db | 8 | 0.675 | **0.639** | 0.568 | 0.865 |
| D | arv_no_gradnorm | c225ac49 | 5 | 0.688 | 0.650 | 0.580 | 0.868 |
| D | arv_gradnorm (reference) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| E | global_only | 218045db | 8 | 0.675 | 0.639 | 0.568 | 0.865 |
| E | global_lora | 218045db | 8 | 0.675 | 0.639 | 0.568 | 0.865 |
| E | global_lora_arv | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| F | full_reference | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| F | no_masked | 5b1474a1 | 8 | 0.681 | **0.644** | 0.575 | 0.863 |
| F | no_multistep | 4967803f | 8 | 0.687 | **0.651** | 0.583 | 0.865 |
| F | no_contrastive | accef6c8 | 9 | 0.653 | **0.617** | 0.544 | 0.850 |
| G | K2 | aa247e5a | 9 | 0.685 | **0.648** | 0.579 | 0.866 |
| G | K4 (reference) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| G | K6 | 6bbade07 | 9 | 0.682 | **0.645** | 0.575 | 0.861 |
| G | K8 | 4e87152e | 7 | 0.682 | **0.645** | 0.576 | 0.864 |
| H | H1_no_metadata | d61f6837 | 8 | 0.683 | **0.645** | 0.576 | 0.863 |
| H | H2_seq25 | d05c7f4d | 9 | 0.678 | **0.641** | 0.571 | 0.860 |
| H | H2_seq50 (reference) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |
| H | H2_seq100 | 2145e400 | 8 | 0.687 | **0.650** | 0.580 | 0.867 |
| H | H3–H6 (reference backbone) | 17bcde2f | 8 | 0.686 | 0.649 | 0.580 | 0.865 |

## Findings

### 1. Routing is the largest architectural effect (studies A, D)
Routed arms score 0.645–0.650; the unrouted 4-objective model scores 0.613, a gain of **+0.035**. That holds for ARV and conventional MMoE alike.

### 2. ARV routing doesn't beat conventional MMoE (studies A, B, C)
- A, equal weights: ARV 0.650, MMoE 0.648.
- C, expert counts matched: ARV 0.649, MMoE 0.645.
- B: objective-conditioned routing 0.649, input-only 0.647.

All of these are ties at one seed. The paper can't claim that ARV routing or objective conditioning improves accuracy.

### 3. GradNorm and routing are substitutes (study D)

| | equal weights | next-item-primary GradNorm (β = 0.3) |
|---|---|---|
| no routing | 0.613 | **0.639** (+0.027) |
| ARV routing | **0.650** | 0.649 (±0) |

Each removes the same problem, and once one is in place the other adds nothing. The gradient diagnostics (cosine between objectives' gradients, measured on the trained models) show what that problem is:

| Model | next·masked | next·multi-step | **next·contrastive** |
|---|---|---|---|
| no routing, equal weights | +0.021 | +0.092 | **−0.145** |
| no routing, GradNorm | +0.009 | +0.089 | +0.014 |
| ARV, equal weights | +0.048 | +0.150 | −0.003 |
| ARV, GradNorm | +0.078 | +0.155 | −0.017 |

Only the unrouted, equal-weight model has a substantial conflict, between contrastive and next-item. Routing separates it, and GradNorm down-weights contrastive (λ_contrastive ≈ 0.007–0.03). All other pairs are near-orthogonal (|cos| ≤ 0.16), which is why reweighting matters so little once the conflict is gone.

### 4. Contrastive is the one auxiliary objective that matters (study F)
- Removing contrastive: **−0.031** (0.649 → 0.617).
- Removing masked-item: −0.004 (a tie).
- Removing multi-step: +0.002 (a tie).

Taken with finding 3, contrastive both conflicts with next-item *and* is needed. It helps only once routing or GradNorm contains that conflict. This is the most interesting result for the paper.

### 5. Expert count, sequence length and metadata are minor (studies G, H)
- **K** = 2 / 4 / 6 / 8 experts: 0.648 / 0.649 / 0.645 / 0.645. No gain from more experts.
- **Sequence length** 25 / 50 / 100: 0.641 / 0.649 / 0.650. Short sequences cost a little, and 100 adds nothing over 50. Seq200 is deferred (docs/future-scope.md).
- **Metadata off**: 0.645 against 0.649 (a tie in training). At inference on warm items it's worth +0.015 (see finding 7).

### 6. LoRA personalisation and federated training add nothing (studies E, H6)
- **Per-user LoRA gain** on 1,000 users: −0.0002 ± 0.035 without routing, −0.0002 ± 0.047 with ARV.
- **Federated LoRA** (20 rounds, 100 clients per round, 80 KB upload per round, 320 MB in total): final federated validation NDCG 0.678. The federated evaluation subset starts at 0.699 at round 0 and drifts down to about 0.693, so the rounds don't improve the model.
- The personalisation and federated claims aren't supported by this run.

### 7. Cold-start and long-tail (study H, reference model only)

| Segment | Model NDCG@10 | Popularity |
|---|---|---|
| Warm, metadata on | **0.658** | 0.379 |
| Warm, metadata off | 0.643 | 0.379 |
| **Cold, metadata on** | **0.0007** | 0.000 |
| Cold, metadata off | 0.0009 | 0.000 |
| Head items | 0.728 | 0.453 |
| Mid items | 0.384 | 0.000 |
| Tail items | 0.095 | 0.000 |

- **Cold items are effectively never recommended**, and metadata doesn't rescue them. This is a negative result.
- The model does recover mid- and tail-popularity items, where popularity scores 0.
- These segment metrics exist only for the reference backbone, so they compare the model with popularity, not with the baseline.

### 8. Efficiency (studies C, G, H5)
Latency is 3.0–3.3 ms per recommendation for every routed variant (ARV 3.28, MMoE 3.17, K2 3.00, reference 3.14 ms). The reference model has 12.6 M parameters (24 MB in bf16), of which 1,540 are router parameters and 3.28 M are expert parameters.

## Selecting β (next-item-primary GradNorm)

A:arv_mmoe_gradnorm on run11's 20-epoch schedule, scored at the best epoch ≤ 9, in `~/valp/b*`:

| β | Best epoch | Val NDCG | Test NDCG |
|---|---|---|---|
| 0.1 | 9 | 0.688 | 0.652 |
| **0.3** | 9 | **0.694** | **0.656** |
| 1 | 7 | 0.691 | 0.650 |
| 3 | 4 | 0.688 | 0.650 |

β = 0.3 was chosen on validation.

For comparison, the original GradNorm (all four objectives balanced together) drove **λ_next_item to exactly 0 from epoch 12**. It peaked at val 0.682 (epoch 3), scored 0.644 on test (run11), and declined afterwards.

## What the paper can and can't claim (one seed)

**Supported:**
- Routing helps (+0.035).
- The contrastive objective helps (+0.031), but only when its conflict with next-item is contained by routing or GradNorm.
- GradNorm helps when there's no routing (+0.027).
- Next-item-primary GradNorm fixes a real failure of standard GradNorm, which zeroes λ_next_item.
- At an equal 10-epoch budget, multi-objective models beat next-item-only training (0.649 vs 0.594). Say clearly that the baseline isn't converged at 10 epochs: it reaches 0.657 at 20.

**Not supported:**
- ARV over MMoE.
- Objective-conditioned over input-only routing.
- More experts.
- LoRA personalisation or federated gains.
- Cold-start gains.

**Needed before any of these claims are final:**
- 2–3 more seeds for the reference model, arv_mmoe, conv_mmoe, no_contrastive, the two no-routing arms and the baseline.
- A per-user paired test (bootstrap) on the key differences.
- A decision on whether to also report a converged baseline.

## Files
- **Arm records:** `outputs/aws_2026-10-17/run12/outputs/ablations/seed_42/arms/<STUDY>_t_base/<arm>.json`
- **Study JSONs:** `.../seed_42/<STUDY>_t_base.json`
- **Plots:** `outputs/aws_2026-10-17/run12/outputs/plots/`
- **Logs:** `outputs/aws_2026-10-17/run12/logs/` (per-arm), `.../launch/plan_aws.json`
- **β sweep:** `outputs/aws_2026-10-17/valp/`
- **run11, 20-epoch schedule, partial:** `outputs/aws_2026-10-17/run11/`
