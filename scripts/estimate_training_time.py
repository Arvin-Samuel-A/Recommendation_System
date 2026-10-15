"""Estimate the wall-clock time of the full experiment plan on THIS machine's GPU.

Times real training steps (`training.pretrain.train_one_step`) on the real processed
data, for every UNIQUE training config the ablation studies need, then extrapolates to
full 20-epoch runs, seeds and model sizes. Nothing is trained to completion and nothing
under `outputs/` is written.

Needs `data/processed/` from `--stage preprocess` (at minimum: dataset_stats.json,
item_metadata.npy, global_negative_pool.npy + .meta.json, train_user_history.pkl,
train_tensorised.pt; val_tensorised.pt + val_candidates.npy are used for the
validation-overhead measurement when present).

Examples:
    uv run python scripts/estimate_training_time.py
    uv run python scripts/estimate_training_time.py --models t-base --gpus 4 --price 1.15
    uv run python scripts/estimate_training_time.py --batch-sweep 256,512,1024,2048 \\
        --user-fraction 0.25 --exclude H2_seq100,H2_seq200 --out estimate.json
"""
import argparse
import copy
import inspect
import json
import math
import os
import pickle
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import training.pretrain as P  # noqa: E402
import utils  # noqa: E402
from configs.config import Config  # noqa: E402
from data.sequence_generator import (  # noqa: E402
    GlobalNegativePool, build_dataloader, load_eval_candidates, load_tensorised,
)
from training.ablation_runner import (  # noqa: E402
    STUDY_CONFIGS, _training_fingerprint, resolve_study_configs,
)

HAS_MIXED = hasattr(Config(), "mixed_precision")
STEP_TAKES_SCALER = "scaler" in inspect.signature(P.train_one_step).parameters
GRAD_DIAG_OVERHEAD = 0.03  # measured ~3% for gradient diagnostics every 500 steps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", default="t-small,t-base,t-large")
    p.add_argument("--precision", default="auto",
                   choices=["auto", "bf16", "bf16-mixed", "fp16-mixed"],
                   help="auto: bf16-mixed on GPUs with native bf16, fp16-mixed on older "
                        "CUDA GPUs (e.g. T4), pure bf16 if this branch has no mixed "
                        "precision support.")
    p.add_argument("--batch-sweep", default="256,512,1024",
                   help="Batch sizes tried on the base config to find max throughput.")
    p.add_argument("--batch", default="auto",
                   help="Batch size used for the estimate; 'auto' = fastest from the sweep.")
    p.add_argument("--steps", type=int, default=40, help="Timed steps per config (x4).")
    p.add_argument("--warmup", type=int, default=8, help="Untimed warmup steps per config.")
    p.add_argument("--epochs", type=int, default=None, help="Default: config.pretrain_epochs.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--user-fraction", type=float, default=1.0,
                   help="Scale the estimate to a user-sampled subset of the dataset.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated ablation arm labels to leave out, e.g. "
                        "H2_seq100,H2_seq200.")
    p.add_argument("--gpus", type=int, default=1, help="GPUs running runs in parallel.")
    p.add_argument("--price", type=float, default=None, help="Price per GPU-hour.")
    p.add_argument("--out", default=None, help="Also write the results as JSON here.")
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def device_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(device="cuda", gpu=props.name, gpu_mem_gb=round(props.total_memory / 1e9, 1),
                    capability=f"{props.major}.{props.minor}", gpu_count=torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        info.update(device="mps", gpu="Apple MPS")
    else:
        info.update(device="cpu", gpu="none")
    return info


def resolve_precision(choice):
    if not HAS_MIXED:
        if choice not in ("auto", "bf16"):
            sys.exit("This branch has no mixed-precision support; use --precision bf16 "
                     "or run from the fp16-mixed-precision branch.")
        return "bf16", False
    if choice == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return ("bf16", True) if major >= 8 else ("fp16", True)
        return "bf16", True
    return {"bf16": ("bf16", False), "bf16-mixed": ("bf16", True),
            "fp16-mixed": ("fp16", True)}[choice]


class Bench:
    """Holds the data loaded once and times configs against it."""

    def __init__(self, dtype, mixed):
        self.dtype, self.mixed = dtype, mixed
        self.proc = os.path.join(ROOT, "data", "processed")
        with open(os.path.join(self.proc, "dataset_stats.json")) as f:
            self.stats = json.load(f)
        self.pool = GlobalNegativePool.load(
            os.path.join(self.proc, "global_negative_pool.npy"), set()).pool
        with open(os.path.join(self.proc, "train_user_history.pkl"), "rb") as f:
            hist = pickle.load(f)
        self.hist = {int(u): np.asarray(v, dtype=np.int64) for u, v in hist.items()}
        self._datasets = {}
        self.n_train = None

    def base_config(self, model):
        cfg = Config()
        cfg.processed_data_dir = self.proc
        cfg.dtype = self.dtype
        if HAS_MIXED:
            cfg.mixed_precision = self.mixed
        cfg.apply_model_config(model)
        cfg.num_items, cfg.num_users = self.stats["num_items"], self.stats["num_users"]
        cfg.vocab_size = cfg.num_items + 2
        meta = P.load_metadata_matrix(self.proc, cfg)
        if meta is not None:
            cfg.feature_dim = meta.shape[-1]
        return cfg

    def dataset(self, cfg):
        key = (cfg.max_seq_length, cfg.use_metadata)
        if key not in self._datasets:
            meta = P.load_metadata_matrix(self.proc, cfg)
            self._datasets[key] = (load_tensorised(
                os.path.join(self.proc, "train_tensorised.pt"), meta,
                max_seq_length=cfg.max_seq_length), meta)
            if self.n_train is None:
                self.n_train = len(self._datasets[key][0])
        return self._datasets[key]

    def _amp(self, cfg):
        return utils.autocast_context(cfg) if (HAS_MIXED and cfg.mixed_precision) \
            else _Null()

    def time_config(self, cfg, model, batch, steps, warmup):
        """Mean wall ms per step (data loading + negatives + step) and peak memory."""
        cfg = copy.deepcopy(cfg)
        cfg.batch_size = batch
        device = P.get_device(cfg)
        ds, meta = self.dataset(cfg)
        if meta is not None:
            cfg.feature_dim = meta.shape[-1]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        it = iter(build_dataloader(ds, cfg, shuffle=True))
        model_, moe = P.build_model(cfg)
        loss_fns = P._build_loss_fns(cfg, device)
        opt = torch.optim.AdamW(P.collect_trainable_parameters(model_, moe, loss_fns),
                                lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                fused=device.type in ("cuda", "mps"))
        sched = P.build_scheduler(opt, cfg, 100_000)
        gn = None
        if cfg.use_gradnorm:
            gn = P.GradNormController(cfg.gradnorm_alpha, cfg.gradnorm_lr)
            gn.log_lambdas = torch.nn.Parameter(gn.log_lambdas.data.to(device=device))
            gn.optimizer = torch.optim.Adam([gn.log_lambdas], lr=cfg.gradnorm_lr)
        scaler = utils.make_grad_scaler(cfg) if STEP_TAKES_SCALER and HAS_MIXED else None
        param_dtype = utils.resolve_param_dtype(cfg) if HAS_MIXED else P.resolve_dtype(cfg)
        meta_t = meta.to(param_dtype) if meta is not None else None
        neg_rng = np.random.RandomState(0)
        buf = np.zeros((batch, cfg.num_items + 2), dtype=bool)
        mask_rng = torch.Generator(device="cpu").manual_seed(0)
        rng = random.Random(0)
        extra = {"scaler": scaler} if STEP_TAKES_SCALER else {}

        def one(i):
            b = next(it)
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            b = P.attach_training_negatives(b, self.pool, self.hist, cfg, neg_rng, device, buf)
            with self._amp(cfg):
                P.train_one_step(model_, moe, b, cfg, loss_fns, gn, opt, sched, i, rng,
                                 mask_rng=mask_rng, metadata_matrix=meta_t, **extra)

        try:
            for i in range(warmup):
                one(i)
            sync()
            t0 = time.perf_counter()
            for i in range(steps):
                one(warmup + i)
            sync()
            ms = 1000 * (time.perf_counter() - t0) / steps
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else MemoryError):
            ms = None
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            ms = None
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        val = self.validation_seconds(cfg, model_, moe, meta) if ms is not None else None
        del model_, moe, opt, loss_fns
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ms, peak, val

    def validation_seconds(self, cfg, model, moe, meta, n=5120):
        """Seconds for one full validation pass, extrapolated from `n` examples."""
        vt = os.path.join(self.proc, "val_tensorised.pt")
        vc = os.path.join(self.proc, "val_candidates.npy")
        if not (os.path.exists(vt) and os.path.exists(vc)):
            return None
        from evaluation.metrics import evaluate_model
        try:
            if not hasattr(self, "_val"):
                vds = load_tensorised(vt, meta, max_seq_length=50)
                cands = load_eval_candidates(vc, verify=False, mmap=True)
                self._val = (vds, len(vds), torch.as_tensor(np.array(cands[:n]), dtype=torch.long))
            vds, n_val, cands = self._val
            seqs = vds.sequences[:n, -cfg.max_seq_length:] if cfg.max_seq_length < 50 \
                else vds.sequences[:n]
            model.eval(); moe.eval()
            with torch.no_grad(), self._amp(cfg):
                sync(); t0 = time.perf_counter()
                evaluate_model(model, seqs, cands, cfg, moe_layer=moe,
                               metadata_matrix=meta if cfg.use_metadata else None)
                sync()
            return (time.perf_counter() - t0) * n_val / n
        except Exception as exc:  # the estimate falls back to a flat overhead
            print(f"    (validation timing skipped: {type(exc).__name__}: {exc})")
            return None


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def unique_training_configs(base, exclude):
    """Unique training fingerprints across all studies, with the arms that share each."""
    groups = {}
    for study in STUDY_CONFIGS:
        for cfg in resolve_study_configs(study, base):
            label = getattr(cfg, "_study_label", "?")
            if label in exclude:
                continue
            fp = _training_fingerprint(cfg)
            groups.setdefault(fp, {"cfg": cfg, "arms": []})["arms"].append(f"{study}:{label}")
    return list(groups.values())


def fmt_h(hours):
    return f"{hours:,.1f} h" if hours < 48 else f"{hours:,.0f} h ({hours / 24:,.1f} d)"


def main():
    args = parse_args()
    info = device_info()
    dtype, mixed = resolve_precision(args.precision)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    exclude = {e.strip() for e in args.exclude.split(",") if e.strip()}
    sweep = [int(b) for b in args.batch_sweep.split(",")]
    print(f"Device: {info.get('gpu')} ({info['device']}), torch {info['torch']}, "
          f"precision={dtype}{' mixed' if mixed else ' pure'}")

    bench = Bench(dtype, mixed)
    out = {"device": info, "precision": {"dtype": dtype, "mixed": mixed}, "models": {}}

    # 1. Batch sweep on the base config (GradNorm on) of the middle model.
    ref = "t-base" if "t-base" in models else models[0]
    print(f"\n[1/2] Batch sweep on {ref}, base config")
    sweep_rows = []
    for b in sweep:
        ms, peak, _ = bench.time_config(bench.base_config(ref), ref, b, args.steps, args.warmup)
        ex_s = b / ms * 1000 if ms else None
        sweep_rows.append({"batch": b, "ms_per_step": ms, "examples_per_s": ex_s, "peak_gb": peak})
        print(f"    batch {b:5d}: " + ("OOM" if ms is None else
              f"{ms:7.1f} ms/step  {ex_s:8.0f} ex/s" + (f"  peak {peak:.1f} GB" if peak else "")))
    out["batch_sweep"] = sweep_rows
    ok = [r for r in sweep_rows if r["ms_per_step"]]
    if not ok:
        sys.exit("Every batch size ran out of memory.")
    batch = max(ok, key=lambda r: r["examples_per_s"])["batch"] if args.batch == "auto" \
        else int(args.batch)
    out["batch"] = batch
    if batch != 256:
        print(f"    -> using batch {batch}. NOTE: a batch other than 256 changes training "
              "(InfoNCE in-batch negatives, optimizer step count, LR) -- check val NDCG.")

    # 2. Every unique training config, per model size.
    epochs = args.epochs or Config().pretrain_epochs
    print(f"\n[2/2] Timing unique training configs at batch {batch}")
    totals = {}
    for model in models:
        base = bench.base_config(model)
        groups = unique_training_configs(base, exclude)
        n_train = bench.n_train or len(bench.dataset(base)[0])
        steps_per_run = math.ceil(n_train * args.user_fraction / batch) * epochs
        rows, hours = [], 0.0
        print(f"  {model}: {len(groups)} unique configs, {steps_per_run:,} steps per run")
        for g in groups:
            ms, peak, val_s = bench.time_config(g["cfg"], model, batch, args.steps, args.warmup)
            if ms is None:
                print(f"    OOM: {', '.join(g['arms'])}")
                rows.append({"arms": g["arms"], "oom": True})
                continue
            train_s = steps_per_run * ms / 1000
            val_total = (val_s * args.user_fraction * epochs) if val_s else train_s * 0.02
            run_h = (train_s + val_total) * (1 + GRAD_DIAG_OVERHEAD) / 3600
            hours += run_h
            rows.append({"arms": g["arms"], "ms_per_step": ms, "peak_gb": peak,
                         "run_hours": run_h, "validation_s_per_epoch": val_s})
            print(f"    {ms:7.1f} ms/step  {run_h:6.1f} h/run  <- {', '.join(g['arms'][:4])}"
                  + (f" (+{len(g['arms']) - 4})" if len(g["arms"]) > 4 else ""))
        totals[model] = hours
        out["models"][model] = {"unique_configs": len(groups), "steps_per_run": steps_per_run,
                                "hours_per_seed": hours, "configs": rows}

    # 3. Summary.
    print("\n" + "=" * 72)
    print(f"ESTIMATE on {info.get('gpu')}, batch {batch}, {epochs} epochs, "
          f"user fraction {args.user_fraction}, {args.gpus} GPU(s) in parallel")
    print("=" * 72)
    scopes = []
    for model in models:
        scopes.append((f"A-H, {model}, 1 seed", totals[model]))
        scopes.append((f"A-H, {model}, {args.seeds} seeds", totals[model] * args.seeds))
    everything = sum(totals.values()) * args.seeds
    scopes.append((f"Everything ({len(models)} sizes x {args.seeds} seeds)", everything))
    for name, gpu_h in scopes:
        line = f"  {name:34s} {fmt_h(gpu_h):>18s} GPU-time   wall {fmt_h(gpu_h / args.gpus):>16s}"
        if args.price:
            line += f"   ~${gpu_h * args.price:,.0f}"
        print(line)
    print("  (wall assumes runs split evenly across GPUs; evaluation/LoRA/federated "
          "stages add a few %)")
    out["summary"] = {name: gpu_h for name, gpu_h in scopes}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
