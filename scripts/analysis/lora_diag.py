"""Why per-user LoRA personalization gives no gain: per-user diagnostics over training setups.

For a finished ablation arm (its best.pt), samples users and, for each training setup below,
fits a fresh per-user LoRA adapter (q/v projections, as `training/local_train.py`) and
records, before and after:
  - the test target's full-catalogue rank (history excluded, the eval protocol), NDCG@10
    and log-probability under a full softmax at the training temperature;
  - the full-softmax loss on the user's own local training examples (did it fit?);
  - the cosine between the before/after (normalised) user embedding on the test sequence;
  - ||B A||_F / ||W||_F of the adapted projections (how far the adapter moved).

Setups (all start from the identical frozen backbone, adapters reset per user):
  default     the production setup: last 32 train prefixes, 3 epochs, lr 1e-3, rank 8,
              sampled softmax over 100 negatives (exactly `train_user_lora`)
  lr1e-2      same, lr 1e-2
  full_sm     same as default but a full-catalogue softmax loss (the pretraining loss)
  strong      full softmax, lr 1e-2, 10 epochs (fit as hard as possible)
  recent4     full softmax, lr 1e-2, 10 epochs, only the 4 most recent train prefixes
  oracle      full softmax, lr 1e-2, 10 epochs, trained ON THE TEST EXAMPLE itself
              (capacity check only: can the adapter move the target at all?)

Run from the repo root of a run (relative data/checkpoint paths); writes only OUT_JSONL.
usage: lora_diag.py ARM_JSON N_USERS OUT_JSONL [SETUP ...]
"""
import copy
import dataclasses
import json
import math
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config
from data.sequence_generator import filter_negatives, load_examples
from evaluation.metrics import (_user_embeddings, catalogue_ranking_scores, load_full_ranking,
                                score_catalogue_batch)
from models.lora import get_lora_state_dict, inject_lora, load_lora_state_dict
from training.ablation_runner import resolve_dataset_fields
from training.local_train import build_local_training_examples
from training.losses import LogitScale, SampledSoftmaxLoss
from training.pretrain import build_model, load_checkpoint
from utils import autocast_context, get_device, set_seed

SETUPS = {
    "default": dict(lr=1e-3, epochs=3, loss="sampled", source="train", cap=32),
    "lr1e-2": dict(lr=1e-2, epochs=3, loss="sampled", source="train", cap=32),
    "full_sm": dict(lr=1e-3, epochs=3, loss="full", source="train", cap=32),
    "strong": dict(lr=1e-2, epochs=10, loss="full", source="train", cap=32),
    "recent4": dict(lr=1e-2, epochs=10, loss="full", source="train", cap=4),
    "oracle": dict(lr=1e-2, epochs=10, loss="full", source="test", cap=1),
}

arm_json, n_users, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
setups = sys.argv[4:] or list(SETUPS)
arm = json.load(open(arm_json))
names = {f.name for f in dataclasses.fields(Config)}
cfg = Config(**{k: v for k, v in arm["resolved_config"].items() if k in names})
set_seed(cfg.seed)
metadata_matrix = resolve_dataset_fields(cfg)
device = get_device(cfg)
proc = cfg.processed_data_dir
store = os.path.join(cfg.checkpoint_dir, "ablations", "by_fingerprint", arm["training_fingerprint"])

model, moe = build_model(cfg)
load_checkpoint(os.path.join(store, "t_base", "best.pt"), model, moe)
inject_lora(model, cfg.lora_rank)
model.to(device); moe.to(device)
moe.eval()
pristine = {k: v.detach().clone() for k, v in get_lora_state_dict(model).items()}
meta_dev = metadata_matrix.to(device) if metadata_matrix is not None else None

test = load_examples(os.path.join(proc, "test_examples.pkl"), cfg.max_seq_length)
ranking = load_full_ranking(proc, "test", test["targets"], test["user_ids"])
with open(os.path.join(proc, "train_user_history.pkl"), "rb") as f:
    train_hist = pickle.load(f)
neg_pool = np.load(os.path.join(proc, "global_negative_pool.npy"))
user_row = {int(u): i for i, u in enumerate(test["user_ids"])}
rng = random.Random(1234)
users = rng.sample(sorted(user_row), n_users)
scale = LogitScale(cfg.logit_temperature)
sampled_loss = SampledSoftmaxLoss(normalize_embeddings=cfg.normalize_embeddings, logit_scale=scale)
neg_rng = np.random.RandomState(cfg.seed)
mask_id = cfg.num_items + 1


def as_long(s):
    return s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long)


def meta_for(seqs):
    return None if meta_dev is None else meta_dev[seqs]


def user_emb_train(seqs):
    """Training-path user embedding (grad through LoRA), (B, D), un-normalised."""
    with autocast_context(cfg, device_type=device.type):
        h = model(seqs, metadata=meta_for(seqs), return_all_hidden=True)
        moe.clear_cache()
        z, _ = moe(h, h[:, -1, :], 0)
    return z[:, -1, :]


def full_softmax_loss(emb, targets):
    """Full-catalogue softmax at the training temperature, items 1..num_items."""
    u = F.normalize(emb.float(), dim=-1) if cfg.normalize_embeddings else emb.float()
    items = model.item_emb_weight.float()
    items = F.normalize(items, dim=-1) if cfg.normalize_embeddings else items
    logits = (u @ items.T) / cfg.logit_temperature
    logits[:, 0] = -float("inf")
    logits[:, mask_id] = -float("inf")
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def measure(test_seq, test_row, train_seqs, train_tgts):
    model.eval()
    s = test_seq.unsqueeze(0).to(device)
    raw = score_catalogue_batch(model, s, cfg, moe_layer=moe, metadata=meta_for(s))
    emb = _user_embeddings(model, s, cfg, moe, meta_for(s))
    sc = catalogue_ranking_scores(raw, ranking.select([test_row]))[0]  # col 0 = target
    tgt = sc[0]
    rank = int((sc[1:] > tgt).sum().item()) + 1
    finite = sc[torch.isfinite(sc)]
    logp = float((tgt / cfg.logit_temperature) - torch.logsumexp(finite / cfg.logit_temperature, 0))
    ndcg = 1.0 / math.log2(rank + 1) if rank <= cfg.eval_k else 0.0
    out = {"rank": rank, "ndcg": ndcg, "logp": logp, "emb": emb}
    if train_seqs is not None:
        x = torch.stack(train_seqs).to(device)
        e = user_emb_train(x)
        out["train_loss"] = float(full_softmax_loss(e, torch.tensor(train_tgts, device=device)))
    return out


def adapter_rel_norm():
    rel = []
    for blk in model.blocks:
        for proj in (blk.attn.q_proj, blk.attn.v_proj):
            d = (proj.lora_A.float() @ proj.lora_B.float()).norm() / proj.weight.float().norm()
            rel.append(float(d))
    return float(np.mean(rel))


def fit(spec, seqs, tgts, history):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=cfg.lora_weight_decay)
    for _ in range(spec["epochs"]):
        model.train()
        for seq, tgt in zip(seqs, tgts):
            x = seq.unsqueeze(0).to(device)
            t = torch.tensor([tgt], device=device)
            emb = user_emb_train(x)
            if spec["loss"] == "sampled":
                negs = filter_negatives(neg_pool, tgt, history, cfg.num_sampled_negatives,
                                        padding_id=0, mask_id=mask_id, rng=neg_rng)
                negs = torch.tensor(negs, dtype=torch.long, device=device).unsqueeze(0)
                with autocast_context(cfg, device_type=device.type):
                    loss = sampled_loss(emb, t, negs, model.item_emb_weight)
            else:
                loss = full_softmax_loss(emb, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()


with open(out_path, "a") as out:
    t0 = time.time()
    for ui, uid in enumerate(users):
        row = user_row[uid]
        test_seq, test_tgt = as_long(test["sequences"][row]), int(test["targets"][row])
        history = set(train_hist.get(uid, []))
        tr_seqs, tr_tgts = build_local_training_examples(uid, cfg, train_hist)
        if not tr_seqs:
            continue
        tr_seqs = [as_long(s) for s in tr_seqs]
        load_lora_state_dict(model, pristine)
        before = measure(test_seq, row, tr_seqs, tr_tgts)
        rec = {"user": uid, "n_train_hist": len(train_hist.get(uid, [])), "target": test_tgt,
               "before": {k: v for k, v in before.items() if k != "emb"}, "after": {}}
        for name in setups:
            spec = SETUPS[name]
            if spec["source"] == "test":
                seqs, tgts = [test_seq], [test_tgt]
            else:
                seqs, tgts = tr_seqs[-spec["cap"]:], tr_tgts[-spec["cap"]:]
            load_lora_state_dict(model, pristine)
            fit(spec, seqs, tgts, history)
            after = measure(test_seq, row, tr_seqs, tr_tgts)
            emb_cos = float(F.cosine_similarity(after["emb"], before["emb"]).item())
            rec["after"][name] = {**{k: v for k, v in after.items() if k != "emb"},
                                  "emb_cos": emb_cos, "rel_norm": adapter_rel_norm()}
        out.write(json.dumps(rec) + "\n"); out.flush()
        if ui % 10 == 0:
            print(f"{ui + 1}/{len(users)} users, {time.time() - t0:.0f} s", flush=True)
print("DONE", flush=True)
