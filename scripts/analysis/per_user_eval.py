"""Per-user test metrics for finished ablation arms, for paired significance tests.

Re-runs exactly the arm's test evaluation (`_run_single_config`: full ranking over the
whole test split from best.pt) but keeps the per-user values that `evaluate_model`
otherwise reduces to a mean. Checks the recomputed mean against the arm JSON.
Run from the repo root (relative data/checkpoint paths); writes nothing inside it.

usage: per_user_eval.py OUT_DIR ARM_JSON [ARM_JSON ...]
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import evaluation.metrics as M
from configs.config import Config
from data.sequence_generator import load_tensorised
from training.ablation_runner import resolve_dataset_fields
from training.pretrain import _build_loss_fns, build_model, load_checkpoint
from utils import get_device, set_seed

captured = {}
_aggregate = M.aggregate_metrics


def _capture(per_example):
    if isinstance(per_example, dict):
        captured.update({k: v.detach().float().cpu().numpy() for k, v in per_example.items()})
    return _aggregate(per_example)


M.aggregate_metrics = _capture

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
names = {f.name for f in dataclasses.fields(Config)}
for arm_json in sys.argv[2:]:
    arm = json.load(open(arm_json))
    cfg = Config(**{k: v for k, v in arm["resolved_config"].items() if k in names})
    set_seed(cfg.seed)
    metadata_matrix = resolve_dataset_fields(cfg)
    store = os.path.join(cfg.checkpoint_dir, "ablations", "by_fingerprint", arm["training_fingerprint"])
    ckpt = os.path.join(store, "t_base", "best.pt")
    device = get_device(cfg)
    model, moe = build_model(cfg)
    loss_fns = _build_loss_fns(cfg, device)
    load_checkpoint(ckpt, model, moe, loss_fns=loss_fns)

    test = load_tensorised(
        os.path.join(cfg.processed_data_dir, "test_tensorised.pt"), metadata_matrix,
        max_seq_length=cfg.max_seq_length,
    )
    ranking = M.load_full_ranking(cfg.processed_data_dir, "test", test.targets, test.user_ids)
    model.eval()
    moe.eval()
    captured.clear()
    with torch.no_grad():
        metrics = M.evaluate_model(model, test.sequences, ranking, cfg, moe_layer=moe,
                                   metadata_matrix=test.metadata_matrix)
    want = arm["metrics"]["ndcg"]["mean"]
    got = metrics["ndcg"]["mean"]
    tag = f"seed{cfg.seed}_{arm['study']}_{arm['label']}"
    np.savez(os.path.join(out_dir, f"{tag}.npz"), user_ids=np.asarray(test.user_ids), **captured)
    print(f"{tag} fp={arm['training_fingerprint']} best_epoch={arm['best_epoch']} "
          f"ndcg={got:.6f} json={want:.6f} diff={got - want:+.2e} n={len(captured['ndcg'])}", flush=True)
