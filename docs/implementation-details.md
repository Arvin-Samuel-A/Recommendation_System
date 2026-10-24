# ARV-MMoE: implementation details

How the system is built, as of commit `542135c` (master, 2026-10-18). Every number below is a value from the code or the config, or a measurement from the 10%-user experiments. Results are in docs/ablation-results-run12.md; the bugs found and fixed along the way are in docs/ablation-run-errors-and-fixes.md.

> **Changed since this was written (2026-10-25).** Sections 1, 3 and 6 describe the run12 protocol. The code now:
> - keeps **every rating** (`implicit_threshold = 0.0`) and defaults to the **full dataset**;
> - splits **leave-one-out** (last item test, second-to-last validation, one row per user per split);
> - trains **LLM-style**: each user's training history is cut into chunks of up to `max_seq_length = 200` predictions, with a full-catalogue next-item and multi-step loss at every position and per-position routing for those two objectives; no sampled training negatives;
> - evaluates against the **full catalogue** minus the user's earlier interactions (eval protocol v3); the stored 101-item candidate sets are gone;
> - defaults to next-item-primary GradNorm with **β = 3** and **multi_step_k = 9**;
> - feeds the **masked sequence only to the masked-item objective**; the other three objectives read a separate clean forward (so §3's "one forward pass" no longer holds).
>
> Run12's numbers are not comparable with runs under the new protocol. See `steering/architecture.md`.

**ARV-MMoE** is a sequential recommender for MovieLens-20M with five parts:
- a causal Transformer backbone;
- **four training objectives**: next-item, masked-item, multi-step and contrastive;
- a **Mixture-of-Experts layer with an objective-conditioned router** (Adaptive Routing Vector, "ARV"), so the four objectives share experts but mix them differently;
- **GradNorm** loss balancing, in a next-item-primary form;
- **LoRA adapters**, for per-user personalisation and a simulated federated setting.

Everything is written by hand in PyTorch 2.x; no recommendation library is used. `configs/config.py` is the single source of truth for hyperparameters, and `main.py` is the command-line entry point for every stage.

```
data/       preprocessing, sequence generation, augmentation, tensorised datasets
models/     Transformer backbone, MoE layer (ARV-MMoE / MMoE / none), LoRA, ANN retrieval
training/   pretraining loop, losses, GradNorm, gradient projection, LoRA personalisation,
            federated simulation, ablation runner and rescoring
evaluation/ ranking metrics, popularity reference, cold-start, long-tail, latency/FLOPs,
            routing and gradient diagnostics
scripts/    ablation packaging, timing estimates, retry wrapper, demo
```

---

## 1. Data pipeline (`data/preprocessing.py`, `data/sequence_generator.py`)

### 1.1 Source and filtering
- **Source:** MovieLens-20M from Kaggle (`grouplens/movielens-20m-dataset`). The files used are `rating.csv` (20,000,263 ratings), `movie.csv` (titles and genres) and `genome_scores.csv` (tag relevance).
- **Implicit feedback:** a rating ≥ **4.0** counts as a "like". Lower ratings are dropped.
- **Users** with fewer than **5** likes are dropped.
- **User sample (experiments):** a seeded random **10% of users** (`user_sample_fraction 0.1`, `user_sample_seed 42`), keeping each sampled user's full history. The sample is drawn after the filters, and its seed is independent of the training seed.
  - 136,677 users → **13,668 users**.
  - **13,530 items** and **1,011,935 interactions**.
  - It is stored in `data/processed_users10pct/`.
- Each user's likes are sorted by timestamp.

### 1.2 IDs
- **Items:** remapped to `1..N` (N = 13,530). `0` is **padding** and `N+1` = 13,531 is the **MASK** token, so the vocabulary has 13,532 entries.
- **Users:** remapped to `0..U-1`.
- **Mappings:** both directions are saved (`item_mapping.pkl`, `reverse_item_mapping.pkl`, `user_mapping.pkl`).

### 1.3 Chronological split
Each user's own timeline is split, not a global date:
- **train:** the first 80%;
- **validation:** the next 10%;
- **test:** the last 10%.

This gives **804,062 train interactions**, **100,591 validation rows** and **107,282 test rows**.

### 1.4 Cold and warm items
- **Strategy:** `cold_start_strategy = "unseen_in_training"`. An item is cold if no user has it in their training portion. Nothing is deleted to create cold items.
- **Counts:** 1,812 cold items and 11,718 warm items.
- Cold items can only appear as validation or test targets.

### 1.5 Item metadata (39 dimensions per item)
- **Genome:** the **20 largest tag-relevance values** of the movie from `genome_scores.csv` (`genome_top_tags = 20`), in descending order. Only the magnitudes are kept, not *which* tags they belong to (see §9).
- **Genres:** a one-hot vector over the **19 genres** in `movie.csv`.
- **Storage:** a `(vocab, 39)` matrix (`item_metadata.npy`). The padding and MASK rows are all zero. Metadata is gathered per item when a batch is built, never stored densely per example.

### 1.6 Training examples (sliding window)
- **Inputs and targets:** for every position `i ≥ 1` of a user's training sequence, the input is the items before `i` and the target is item `i`. Inputs are **left-padded** to the sequence length and keep the most recent items.
- **Count:** about **790k training examples**, roughly 3,088 steps per epoch at batch size 256.
- **Multi-step targets:** each example also stores the next **k = 3** items (`multi_step_k`). They're set to 0 (ignored) past the end of the training portion, so the model never peeks into validation.
- **Validation and test rows:** each target is predicted from the user's whole history up to that point.
  - A validation row's context is train plus the earlier validation items.
  - A test row's context is train, validation and the earlier test items.
- **Stored width:** splits are stored **200 positions wide**, and each model narrows them to its own `max_seq_length` (default 50), so the sequence-length study (25 / 50 / 100 / 200) shares one dataset.

### 1.7 Negatives and fixed evaluation candidates
- **Global negative pool:**
  - It covers **warm items only**. Its size is max(10,000, 10 × N) = 135,300: every warm item appears about 11–12 times, shuffled with seed 42.
  - It's built once and fingerprinted.
- **Training negatives:** **100 per example**, drawn from the pool. A draw is rejected if it's the target, the padding or MASK ID, or anything in the user's training history.
- **Evaluation candidates:**
  - Each validation and test row gets a fixed set: the **target plus 100 negatives**, drawn from the pool and excluding the user's history.
  - The sets are generated once (seed 42), stored (`val_candidates.npy`, `test_candidates.npy`, fingerprinted) and reused by every model, so all arms are ranked on identical candidates.

---

## 2. Model (`models/transformer.py`, `models/moe.py`)

### 2.1 Backbone: causal Transformer

| Size | Layers | Hidden | Heads | FFN |
|---|---|---|---|---|
| t-small | 3 | 256 | 4 | 4× |
| **t-base** (all experiments) | **4** | **320** | **4** (head dim 80) | **4×** (1,280) |
| t-large | 5 | 384 | 6 | 4× |

**Input embedding:** item embedding plus a learned position embedding, plus a linear projection of the 39-d metadata (no bias, added only if `use_metadata`), followed by dropout 0.1.

**Blocks:** pre-LayerNorm. Each block is

```
x = x + Dropout(Attn(LN(x)))
x = x + Dropout(FFN(LN(x)))      FFN = Linear(320→1280), GELU, Dropout, Linear(1280→320)
```

**Attention:**
- **Masking:** causal, with key padding masked out.
- **Kernels:**
  - **SDPA**, `torch.nn.functional.scaled_dot_product_attention`, on CPU, MPS and non-Hopper GPUs (the AWS L4s).
  - **FlashAttention-3** on Hopper (H100), with variable-length unpadding.
- **Backend check:** the backend is probed forward and backward at start-up. An explicitly requested backend that can't run raises an error rather than silently falling back.

**Output:** a final LayerNorm. The hidden state at the **last position** is the sequence summary.

**Shared item table:** the item embedding table is also the **output table**, so every objective scores items by **cosine similarity** between a user/position vector and L2-normalised item embeddings (`normalize_embeddings = True`).

### 2.2 Mixture-of-Experts layer
Three variants (`moe_variant`):

| Variant | What it does |
|---|---|
| `none` | The backbone hidden states are used directly. |
| `mmoe` (conventional MMoE) | One softmax gate per objective, `softmax(W_t · h_pooled)`. |
| **`arv_mmoe`** (ours) | **One shared router** conditioned on a learned **objective embedding**: `r_t = softmax(W · [h_pooled ; e_t])`, where `e_t` is a 32-d embedding per objective. `router_conditioning = "input_only"` drops `e_t` (study B). |

- **Experts:** K = **4** expert FFNs (`Linear(320→1280)`, GELU, Dropout, `Linear(1280→320)`).
  - They're shared by all objectives and **computed once per forward pass**; only the mixture weights differ per objective.
  - The mixture is `Z_t = Σ_k r_{t,k} · Expert_k(H)`.
- **Pooled-then-broadcast routing:** each objective's routing distribution comes from the pooled (last-position) hidden state and is applied to every position.
- **At inference**, only the next-item routing is used (`e_rec ≡ e_next`).
- **Parameters (t-base, ARV, K = 4):**
  - 12,575,940 in total (24 MB in bf16);
  - experts 3,283,200;
  - router plus objective embeddings 1,540.

---

## 3. Training objectives (`training/losses.py`, `train_one_step` in `training/pretrain.py`)

**One forward pass per step serves all objectives.**
- The backbone runs **once**, on the *masked* sequence (when masked-item is on).
- The MoE layer is then called once per objective, each with its own routing.

| Objective | Index | Input to the loss | Loss |
|---|---|---|---|
| **next-item** | 0 | last-position MoE output for the next-item objective | **sampled softmax**: cross-entropy over target plus 100 negatives |
| **masked-item** (MIM) | 1 | MoE output at masked positions | **full-catalogue softmax** over all 13,532 items |
| **multi-step** | 2 | last-position MoE output, then one linear projection (no bias) per step | full-catalogue softmax for items t+1, t+2, t+3, averaged over the valid steps |
| **contrastive** | 3 | last-position MoE output of the original view and of an augmented view | **InfoNCE** with in-batch negatives (2B−1 candidates), in fp32 |

- **Scores:** all four objectives score items by **cosine similarity**, and the logits are multiplied by `1/τ`.
  - Run11 and later use **τ = 0.07 for all four** (`--logit_temperature 0.07`).
  - Contrastive uses its own temperature, also 0.07.
  - Why τ is needed: with τ = 1, cosine logits lie in [−1, 1], so the loss can't fall below 2.68 for next-item (101 candidates) or about 7.5 for the full-catalogue objectives. In practice those models only tied the popularity ranker.
  - Optional: a learnable CLIP-style scale per objective (`--learnable_logit_scale`, clamped at 100).
- **Masking (MIM):**
  - Each real item is replaced by MASK with probability **0.15**, and at least one real item is always left unmasked.
  - Masked positions get zero metadata, so the metadata can't reveal the hidden item.
  - **The backbone is causal,** so the hidden state at a masked position sees only the items before it plus the MASK token.
- **Contrastive augmentation:** each sequence's augmented view is made by one of three operations, chosen uniformly:
  - a random crop (keeping at least 60%);
  - random masking (20% of items);
  - a random prefix.
  
  Contrastive runs every step in run11 and later (`--contrastive_interval 1`; the old default was every 4th step).
- **Skipped objectives:** an objective with no valid target in a batch (for example, no masked positions) is left out of that step.

---

## 4. Loss balancing

### 4.1 GradNorm (`training/gradnorm.py`; Chen et al., 2018)
- **What's learned:** weights λ_t, stored as `log λ_t`.
- **Optimiser:** a separate Adam, lr 0.01.
- **Strength:** α = 1.5.
- **Anchor:** the gradient norms are measured on the **last shared Transformer block**, so every MoE variant has the same anchor.
- **Each step:**
  1. `G_t = λ_t · ‖∇_W L_t‖`, computed with a first-order factored form.
  2. Target `G* = mean(G) · (r_t)^α`, where `r_t` is the objective's loss ratio `L_t/L_t(0)` relative to the mean ratio.
  3. Minimise `Σ |G_t − G*|` with respect to λ only. The model's own gradients aren't touched.
  4. Rescale the λs back to a fixed sum.

**The problem.** In standard GradNorm all four λ are balanced together and sum to 4. That drove **λ_next_item to exactly 0** from epoch 12 in run11: the model stopped training the objective it's evaluated on. Validation NDCG peaked at epoch 3 and then fell.

**Next-item-primary GradNorm** (`--gradnorm_primary_objective next_item --gradnorm_aux_budget β`, commit `50a8332`):
- λ_next_item is **fixed at 1** and excluded from GradNorm.
- GradNorm balances only masked-item, multi-step and contrastive, whose λ sum to **β**. The auxiliary λ start at β/3.
- **β = 0.3** was chosen on validation from {0.1, 0.3, 1, 3}. Next-item therefore gets about 77% of the total weight.
- **In run12**, at the reference model's best epoch, the learned auxiliary weights were λ_masked = 0.16, λ_multi-step = 0.13 and λ_contrastive = 0.007.

**Other GradNorm options:**
- `gradnorm_renorm_scope`: `"active"` (the default) rescales the λ of the objectives active this step; `"enabled"` rescales over every enabled objective. The two behave the same when contrastive runs every step.
- Arms with `use_gradnorm = False` use **fixed weights** of 1 for every objective (`masked_lm_weight` etc.).

### 4.2 Gradient projection (implemented, not yet run; `training/grad_projection.py`, docs/gradient-projection.md)
- **Switch:** `grad_projection = "next_item"`, which requires `use_gradnorm = False`.
- **What it does:** each objective gets its own backward pass. Any auxiliary gradient with a negative dot product against next-item's has its component along next-item's gradient removed (an asymmetric PCGrad). Next-item's own gradient is never changed.
- **Cost:** about 1.7× per step.
- **Status:** off by default, and no ablation arm uses it yet.

---

## 5. Training loop (`run_pretrain` in `training/pretrain.py`)

| Setting | Value |
|---|---|
| Optimiser | **AdamW**, lr **1e-3**, weight decay **0.01** (learnable logit scales excluded), fused kernels on GPU |
| Schedule | linear **warmup over 10%** of steps, then **cosine decay to 0.1×** the peak LR |
| Epochs | **10** for the 10%-user ablation (run12); 20 planned for the full-dataset run |
| Batch | **256** (fixed, because in-batch contrastive negatives depend on it) |
| Gradient clipping | global norm **1.0** over model, MoE and loss parameters |
| Dropout | 0.1 everywhere |
| Precision | **mixed bf16**: fp32 master weights, bf16 autocast opened **per step and per forward pass with the weight cache off**. The cache was the run10 bug: a single outer autocast region froze every Linear layer at its first-step weights. Ranking scores are always fp32. |
| Model selection | validation NDCG@10 on the full validation split after every epoch. `best.pt` keeps the best epoch, and **test metrics are reported only for that epoch** |
| Checks | finite-loss and finite-gradient checks before each optimiser step; dtype assertions |
| Resume | `last.pt` (full state, RNGs included) written atomically every epoch; a run resumes from it automatically |
| Seed | 42 for training. The user sample, negatives and candidates have their own fixed seeds |

**Run12 timing** on one NVIDIA L4 at 10% of users:
- 169 ms/step for the reference model, about 9 min/epoch;
- about 2 min/epoch for the next-item-only baseline.

---

## 6. Evaluation (`evaluation/metrics.py` and the other `evaluation/` modules)

### 6.1 Ranking protocol (eval protocol v2)
- **Candidates:** every test row is ranked among its **101 fixed candidates** (target in column 0).
- **Score:** the cosine between the user vector (the next-item MoE output at the last position) and each candidate's L2-normalised embedding, **in fp32**. The training temperature isn't applied, since ranking is unchanged by a positive scale.
- **Metrics:** NDCG@10, MRR@10, Recall@10 and Precision@10, averaged over rows.
- **Ties** are counted at their **expected rank** (tie-averaged, McSherry & Najork 2008). The old rule gave ties to the target, which bf16 scores made frequent and which inflated NDCG by up to 0.08.
- **Rows:** the **whole test split** (107,282 rows) is scored.
- **Popularity reference:** the same rows and candidates ranked by training-interaction count. **NDCG@10 = 0.374.** Every result is compared against it.

### 6.2 Extra evaluations, selected per arm by `system_eval`
- **Cold-start (H3):** test rows split by cold/warm target, scored with metadata on and off (the metadata-off forward skips the metadata input entirely).
- **Long-tail (H4):** items bucketed by training popularity into head (top 20% of the catalogue), mid (next 20%) and tail (bottom 60%, including zero-interaction items).
- **Latency (H5):** milliseconds per recommendation (20 warmup and 100 timed runs), plus parameter counts and an analytical FLOPs estimate (`evaluation/flops.py`).
- **Communication (H6):** the federated simulation of §7.2 with byte accounting.
- **Gradient diagnostics (study D):** cosine similarity and norm of every objective's gradient on the anchor layer, over 10 fresh batches.
- **Routing diagnostics:** expert-usage entropy and per-objective routing over the validation set.

### 6.3 Ablation runner (`training/ablation_runner.py`)
- **Studies:** eight, **A–H**, 33 arms in total:
  - A: architecture progression;
  - B: router conditioning;
  - C: ARV vs MMoE;
  - D: routing × GradNorm;
  - E: personalisation;
  - F: leave-one-objective-out;
  - G: number of experts;
  - H: metadata, sequence length and system evaluations.
- **Fingerprints:** each arm's training-relevant config is hashed. Arms with the same fingerprint share one trained backbone, stored under `by_fingerprint/<fp>/`. In run12 the reference backbone `17bcde2f65e8` serves 13 arms.
- **Results:** each arm writes one JSON file. `--stage ablation_collect` assembles a study's JSON and plots only when every arm is `ok` and consistent.
- **Exit codes:** a failed arm exits **70** (not retried). Out-of-memory exits **75** and is retried by `scripts/run_with_retry.py`.
- **Packaging:** `scripts/package_ablation_run.py` packages a run for a remote GPU machine. It builds a code tarball and per-GPU launch scripts, and plans the arms across GPUs longest-first using measured step times.

---

## 7. Personalisation and federated simulation

### 7.1 Per-user LoRA (`models/lora.py`, `training/local_train.py`)
- **Adapters:** LoRA, **rank 8**, on the **query and value projections** of every attention layer.
  - A is initialised with N(0, 0.02²); B is zero, so a fresh adapter changes nothing.
  - Everything else is frozen: backbone, MoE experts, router, objective embeddings.
- **Local training:**
  - Each of 1,000 evaluation users trains its own adapter for 3 epochs on its **32 most recent training examples**.
  - The loss is next-item sampled softmax only.
  - AdamW, lr 1e-3, no weight decay.
- **Gain:** the user's test NDCG with the adapter minus without it. Adapters are reset between users, and a checksum confirms the frozen weights never change.

### 7.2 Federated LoRA (`training/federated.py`)
- **Setup:** 20 rounds, **100 clients per round**, sampled from the users.
- **Each round:** every client loads the global adapter, trains locally and uploads **only its LoRA weights** (80 KB). Weighted FedAvg (weighted by local example count) produces the next global adapter.
- **Global state:** the backbone and MoE are never sent; they stay frozen.
- **Monitoring:** each round is scored on a fixed 5,000-row validation subset; the last round is scored on the full validation split.

**Run12 result:** neither per-user LoRA (gain −0.0002 ± 0.047) nor federated rounds improved the model.

---

## 8. Run infrastructure

**Hardware**
- Development: Apple-silicon Mac (MPS).
- Experiments: an AWS g6.12xlarge (4× L4, SDPA), and a shared H100 (FlashAttention-3) when available.

**Commands** (run from the repository root, with `uv`):

```bash
uv run python main.py --stage preprocess --user_sample_fraction 0.1
uv run python main.py --stage ablation --study A --model t-base --seed 42 --user_sample_fraction 0.1 --dtype bf16 --mixed_precision --batch_size 256 --attention_backend sdpa --logit_temperature 0.07 --contrastive_interval 1 --epochs 10 --gradnorm_primary_objective next_item --gradnorm_aux_budget 0.3
uv run python main.py --stage ablation_collect --study A
```

**Run12** (the final 10%-user ablation) used exactly these flags for all 33 arms, plus `--sequence_length_grid 25 50 100`. It took 6.3 h on 4 L4s.

**Reproducibility**
- Every result records its resolved config, training fingerprint, code source hash and evaluation protocol version.
- The training fingerprint includes a `TRAINING_SEMANTICS_VERSION`, so backbones trained before a correctness fix are never reused.

**Tests:** 626 pytest tests (`uv run python -m pytest`). The data-dependent tests need `data/processed`.

---

## 9. Known limitations (as implemented)

**Cold items are effectively never recommended** (cold NDCG@10 ≈ 0.001). Three reasons:
- A cold item's embedding is never trained as a positive.
- The full-catalogue masked-item and multi-step softmaxes push it down as a negative.
- Metadata enters only the **input** side (the sequence encoding), not the **output** item embedding, so it can't help a cold item be scored as a target.

A metadata-based item tower on the output side would be needed.

**Other limitations**
- **Genome features lose tag identity:** they keep the 20 largest relevance *values*, not which tags they are.
- **MIM runs on a causal backbone:** a masked position sees only its left context, which makes masked-item prediction similar to next-item prediction.
- **Next-item is also trained on the masked input** whenever masked-item is enabled, since one forward pass serves all objectives. That acts as input noise for next-item.
- **One seed:** every comparison so far is single-seed (seed 42). Treat differences under about 0.01 NDCG as ties.
- **Undertrained baseline:** at 10 epochs the next-item-only baseline hasn't converged. It reaches 0.657 test NDCG at 20 epochs, against 0.594 at 10.

---

## 10. Demo for review (`scripts/demo_recommend.py`)

The demo loads the trained reference model (run12 backbone `17bcde2f65e8`: ARV-MMoE, 4 objectives, next-item-primary GradNorm with β = 0.3, best epoch 8). It then shows one user's recent likes and the model's top-10 over the **whole catalogue**, never repeating a movie the user already liked.

- **Real users:** a check mark flags each recommendation that the user **actually went on to like** in their held-out test period, which the model never saw.
- **Comparisons:** popularity is always shown; with `--compare`, the next-item-only baseline is shown too.
- **Routing:** the demo prints the router's expert mixture for that viewer.

Run from the repository root (the checkpoints are in `outputs/aws_2026-10-17/run12/outputs/checkpoints/`, which isn't in git):

```bash
uv run python scripts/demo_recommend.py --user 13408 --compare
uv run python scripts/demo_recommend.py --liked "Toy Story (1995)" "Finding Nemo" "Shrek (2001)" "Monsters, Inc." "Incredibles"
uv run python scripts/demo_recommend.py --seed 7
```

**Checks:**
- The demo uses the production scoring function. On 4,000 random test rows it reproduces the run's test NDCG@10: 0.641, against 0.649 on the full split, within sampling error.
- Over 400 random users ranked against the whole catalogue, the model averages **0.86 hits in its top-10**, against **0.29** for popularity.
- User 13408 was picked as a clear example: 6/10 hits, against 1/10 for the baseline and 0/10 for popularity. Present it as an illustration, not an average.
