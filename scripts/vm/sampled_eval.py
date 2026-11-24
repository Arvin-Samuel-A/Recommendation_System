"""SASRec-style sampled evaluation of a finished arm, next to the full-ranking protocol.

For every test row: the target plus 100 negatives drawn uniformly from the items the user
never interacted with (train, validation or test), ranked by the model. Also reports the
same rows under full ranking (must reproduce the stored metric) and popularity under
both protocols. Run from the repo root of the run (relative paths).

usage: sampled_eval.py ARM_JSON [NUM_NEGATIVES]
"""
import dataclasses
import json
import os
import pickle
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

arm = json.load(open(sys.argv[1]))
K_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 100
names = {f.name for f in dataclasses.fields(Config)}
cfg = Config(**{k: v for k, v in arm["resolved_config"].items() if k in names})
set_seed(cfg.seed)
meta = resolve_dataset_fields(cfg)
ckpt = os.path.join(cfg.checkpoint_dir, "ablations", "by_fingerprint", arm["training_fingerprint"], "t_base", "best.pt")
device = get_device(cfg)
model, moe = build_model(cfg)
load_checkpoint(ckpt, model, moe, loss_fns=_build_loss_fns(cfg, device))
model.eval()
moe.eval()

test = load_tensorised(os.path.join(cfg.processed_data_dir, "test_tensorised.pt"), meta,
                       max_seq_length=cfg.max_seq_length)
splits = pickle.load(open(os.path.join(cfg.processed_data_dir, "splits.pkl"), "rb"))
targets = test.targets.long()
uids = test.user_ids.tolist()

# Negatives: uniform over items 1..N the user never interacted with.
rng = np.random.default_rng(0)
N = cfg.num_items
cands = np.empty((len(uids), K_NEG + 1), dtype=np.int64)
for r, u in enumerate(uids):
    seen = set(splits["train"].get(u, [])) | set(splits["val"].get(u, [])) | set(splits["test"].get(u, []))
    negs = []
    while len(negs) < K_NEG:
        for x in rng.integers(1, N + 1, size=2 * K_NEG):
            if x not in seen:
                seen.add(int(x))
                negs.append(int(x))
                if len(negs) == K_NEG:
                    break
    cands[r, 0] = int(targets[r])
    cands[r, 1:] = negs
cands_t = torch.from_numpy(cands)

with torch.no_grad():
    sampled = M.evaluate_model(model, test.sequences, cands_t, cfg, moe_layer=moe,
                               metadata_matrix=test.metadata_matrix)
    full = M.evaluate_model(model, test.sequences,
                            M.load_full_ranking(cfg.processed_data_dir, "test", test.targets, test.user_ids),
                            cfg, moe_layer=moe, metadata_matrix=test.metadata_matrix)

pop = M.load_item_popularity(cfg.processed_data_dir, cfg.vocab_size)
pop_sampled = M.aggregate_metrics(M.compute_metrics_batch(M.popularity_scores(cands_t, pop), target_idx=0, k=10))
row = {
    "arm": f"{arm['study']}:{arm['label']}", "seed": cfg.seed, "fingerprint": arm["training_fingerprint"],
    "dataset": getattr(cfg, "dataset_format", "movielens"), "rows": len(uids), "negatives": K_NEG, "catalogue": N,
    "model_full_ranking": {m: full[m]["mean"] for m in ("ndcg", "recall", "mrr")},
    "stored_full_ranking_ndcg": arm["metrics"]["ndcg"]["mean"],
    "model_sampled": {m: sampled[m]["mean"] for m in ("ndcg", "recall", "mrr")},
    "popularity_full_ranking_ndcg": (arm.get("popularity_reference") or {}).get("ndcg", {}).get("mean"),
    "popularity_sampled": {m: pop_sampled[m]["mean"] for m in ("ndcg", "recall", "mrr")},
}
print(json.dumps(row))
