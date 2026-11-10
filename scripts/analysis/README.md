# Analysis scripts

These scripts produced the numbers behind the paper's 3-seed results and the follow-up diagnostics. Each one reads finished ablation outputs and changes nothing in the run. Run them from the root of the run, where the relative data and checkpoint paths resolve (on the VM, `~/run13` for MovieLens and `~/beauty14` for Beauty).

| Script | What it does |
|---|---|
| `agg_3seed.py ROOT [OUT_MD]` | Builds a mean ± std table of test NDCG/Recall/MRR@10 over seeds 42/43/44 for every study arm. ROOT is an `outputs/ablations` folder. It also lists the source hashes behind each seed. |
| `per_user_eval.py OUT_DIR ARM_JSON...` | Re-runs an arm's full-ranking test evaluation from `best.pt` and saves per-user metrics (`seed{S}_{STUDY}_{ARM}.npz`). It checks the mean against the arm JSON. |
| `paired_tests_3seed.py NPZ_DIR...` | Runs per-user paired tests over 3 seeds, each arm against the reference plus a few contrasts within a study. Reports the per-seed difference and t p, the user-level t and Wilcoxon p on scores pooled over seeds, Holm-adjusted p, and whether the sign agrees across seeds. |
| `lora_diag.py ARM_JSON N_USERS OUT_JSONL [SETUP...]` | Shows why per-user LoRA gives no gain. For sampled users it fits an adapter in several ways and records the test target's rank and log-probability, the loss on the user's own training examples, the change in the user embedding, and the adapter norm. |
| `lora_summary.py DIAG_JSONL` | Summarises `lora_diag.py` output for each setup. |
| `grad_conflict.py ARM_JSON CKPT N_BATCHES OUT_JSON` | For each batch, takes every objective's gradient (exactly as the training losses compute it) over three parameter scopes and stores the Gram matrix. PCGrad-style conflict, the agreement between next-item and the weighted auxiliaries, and CAGrad's direction can all be computed offline from it. |
| `param_count.py CHECKPOINT` | Counts a checkpoint's parameters by module, with sizes in fp32, fp16 and int8. |

The on-device latency benchmark is the "Run benchmark" button in `demo/mobile/index.html`, served by `scripts/run_mobile_demo.sh`. Results are written to `outputs/mobile_bench/`.
