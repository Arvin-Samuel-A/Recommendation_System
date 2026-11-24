"""Per-batch gradient geometry of the four training objectives, for one checkpoint.

For an ablation arm (its resolved config) and a checkpoint of it, draws N training
batches and, per batch, takes each objective's gradient exactly as training computes the
loss (`evaluation/gradient_diagnostics.run_gradient_diagnostics`, one batch at a time),
over three parameter scopes:
  anchor   the last Transformer block (what GradNorm balances on)
  blocks   every Transformer block
  encoder  every parameter of the Transformer, item embeddings included
and stores the per-batch Gram matrix G_ij = |g_i||g_j| cos_ij. Everything about conflict
(PCGrad's negative-cosine test, the cosine between next-item and a weighted auxiliary
sum, CAGrad's conflict-averse direction) is a function of G, so it is computed offline.

CKPT: "best" (best.pt), "last" (last.pt), "epoch_N" (epoch_NNN.pt) or "init" (the same
architecture freshly initialised from the arm's seed, i.e. the start of training).
Env: CKPT_ROOT overrides the checkpoint root (default: the config's checkpoint_dir);
FP32=1 runs fp32 weights with no autocast region (for MPS/CPU).

usage: grad_conflict.py ARM_JSON CKPT N_BATCHES OUT_JSON
"""
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch

if not torch.cuda.is_available():
    # Checkpoints saved on the VM hold CUDA tensors; load them onto the CPU first.
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "map_location": "cpu"})

from configs.config import Config
from data.sequence_generator import build_dataloader, load_train_dataset
from evaluation.gradient_diagnostics import run_gradient_diagnostics
from training.ablation_runner import resolve_dataset_fields
from training.pretrain import _build_loss_fns, build_model, load_checkpoint
from utils import get_device, set_seed

arm_json, ckpt_name, n_batches, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
arm = json.load(open(arm_json))
names = {f.name for f in dataclasses.fields(Config)}
cfg = Config(**{k: v for k, v in arm["resolved_config"].items() if k in names})
cfg.num_workers = 0
cfg.device = "auto"
if os.environ.get("FP32"):
    # fp32 master weights (mixed_precision keeps them fp32) and no autocast region.
    import contextlib
    import evaluation.gradient_diagnostics as _gd
    cfg.mixed_precision = True
    _gd.autocast_context = lambda *a, **k: contextlib.nullcontext()
set_seed(cfg.seed)
metadata_matrix = resolve_dataset_fields(cfg)
device = get_device(cfg)
model, moe = build_model(cfg)
loss_fns = _build_loss_fns(cfg, device)
root = os.environ.get("CKPT_ROOT", cfg.checkpoint_dir)
store = os.path.join(root, "ablations", "by_fingerprint", arm["training_fingerprint"], "t_base")
if ckpt_name != "init":
    fname = {"best": "best.pt", "last": "last.pt"}.get(ckpt_name)
    if fname is None:
        fname = f"epoch_{int(ckpt_name.split('_')[1]):03d}.pt"
    load_checkpoint(os.path.join(store, fname), model, moe, loss_fns=loss_fns)
model.to(device); moe.to(device)
for fn in loss_fns.values():
    if isinstance(fn, torch.nn.Module):
        fn.to(device)

train = load_train_dataset(cfg.processed_data_dir, cfg.max_seq_length, metadata_matrix)
loader = build_dataloader(train, cfg, shuffle=True)
cfg.grad_diag_num_batches = 1
scopes = {
    "anchor": [p for p in model.blocks[-1].parameters() if p.requires_grad],
    "blocks": [p for p in model.blocks.parameters() if p.requires_grad],
    "encoder": [p for p in model.parameters() if p.requires_grad],
}
mm = torch.as_tensor(metadata_matrix) if metadata_matrix is not None else None
records = []
t0 = time.time()
for bi, batch in enumerate(loader):
    if bi >= n_batches:
        break
    rec = {}
    for scope, params in scopes.items():
        r = run_gradient_diagnostics(model, moe, [batch], loss_fns, params, cfg, "t-base",
                                     metadata_matrix=mm, write_report=False)
        names_ = r["objective_names"]
        cos = r["cosine_similarity"]
        norms = [r["gradient_norms"]["per_objective"][n] for n in names_]
        rec[scope] = {"cos": cos, "norms": norms}
    records.append(rec)
    if bi % 10 == 0:
        print(f"batch {bi + 1}/{n_batches} {time.time() - t0:.0f} s", flush=True)

json.dump({"arm": f"{arm['study']}:{arm['label']}", "seed": cfg.seed, "ckpt": ckpt_name,
           "fingerprint": arm["training_fingerprint"], "objectives": names_,
           "batch_size": cfg.batch_size, "batches": records}, open(out_path, "w"))
print("DONE", out_path, flush=True)
