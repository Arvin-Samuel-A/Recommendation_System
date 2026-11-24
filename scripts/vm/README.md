# H100 VM helper scripts

Copied from `~/early_stop/` on the shared H100 VM on 2026-11-24, so the code that drove the protocol-v3 runs is in the repository. They ran outside the run directories (`~/run13`, `~/beauty14`, `~/beauty14_fixedw`, `~/baselines_run`) so those directories' source hashes stayed unchanged. Paths are the VM's (`/root/...`); the scripts are kept as they ran, not made portable.

| Script | What it did |
|---|---|
| `early_stop.py`, `early_stop_auto.py`, `early_stop_fixedw.py` | Early stopping for the Beauty arms: stop an arm after 50 epochs without a new best validation NDCG@10, register `best.pt` and evaluate it |
| `launch_seeds.sh`, `add_beauty_gpu*.sh`, `add_ml_lanes.sh`, `launch_fixedw.sh` | Launch the seed 43/44 runs and the Beauty fixed-weight run |
| `ml_scheduler.py`, `ml_scheduler2.py`, `start_ml_scheduler.sh`, `swap_scheduler*.sh` | Memory-gated scheduler for the ML-20M seed runs |
| `baseline_scheduler.py`, `launch_baselines.sh`, `baseline_status.py`, `time_baseline*.py` | Gated scheduler, launcher, status and timing for the study X baselines (see `docs/baselines.md`) |
| `auto_recover.sh`, `auto_recover_mlseeds.sh`, `restart_after_reboot.sh`, `bashrc_hook.txt`, `default_ssh_hook.txt` | Resume unfinished runs after the container's daily restart |
| `beauty_per_user_*.sh`, `run_ml_per_user.sh`, `beauty_arm_list.py`, `ml_arms*.txt` | Per-user scoring for the paired tests |
| `run_gc_*.sh`, `grad_conflict.py` | Gradient-conflict analysis runs (the analysis itself is `scripts/analysis/gc_analysis.py`) |
| `sampled_eval.py` | Sampled-candidate evaluation, for comparison with full ranking |
| `agg_mobile.py` | Aggregate the phone benchmark results |
| `check_fp.py`, `param_count.py`, `summ.py` | Fingerprint check, parameter counts, run summary |
| `run_tests.sh` | Run a checkout's test suite CPU-only |

Not copied:
- Scripts already in `scripts/analysis/`: `gc_analysis.py`, `results_extras.py`, `run_gc_analysis*.sh`, `train_val_curves.py`, `per_user_eval.py`, `paired_tests_3seed.py`, `agg_3seed.py`. The VM copies are identical.
- `lora_diag.py` and `lora_summary.py`, which need the LoRA code removed in `ae88201`.
- Backups (`*.bak*`) and outputs.
