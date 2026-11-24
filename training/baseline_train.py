"""Training and evaluation of the external baselines (models/baselines) in OUR pipeline.

What comes from the baseline's authors (models/baselines/hparams.py): architecture, loss,
hidden size, layers, heads, dropouts, initialisation, max sequence length, batch size,
the optimizer (Adam, published lr and betas, no weight decay, no LR schedule, no gradient
clipping -- the reference trainer has none).

What comes from our pipeline (docs/baselines.md):
- the leave-one-out splits and processed data;
- the training feed: each user's whole training history cut into chunks of
  `max_seq_length` predictions (`data.sequence_generator.load_train_dataset`). The
  baselines' published losses score ONE next item from the last position of a left-padded
  sequence (the reference trains on every prefix of each sequence), and BSARec's Fourier
  filter is not causal, so a whole chunk cannot be scored in one forward. Every chunk
  position with a target therefore becomes one left-padded prefix example -- the chunk's
  inputs up to that position, right-aligned to the last slot -- so every training
  interaction is a target exactly once per epoch, as in our own training;
- full-catalogue evaluation with history exclusion and tie-averaged metrics
  (`evaluation.metrics`), on validation every epoch; best epoch by validation NDCG@10;
- `pretrain_epochs` (20 on ML-20M, at most 300 on Beauty) and `early_stop_patience`;
- seeds, the fingerprinted checkpoint store, `best.pt` / `last.pt` and exact resume.

Everything trains in fp32, as the reference does; `dtype` / `mixed_precision` are ignored.
"""
import json
import logging
import math
import os
import pickle
import random
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from configs.config import Config
from models.baselines import build_baseline, count_baseline_parameters

logger = logging.getLogger(__name__)

#: How many times a same-target sample that equals its own sequence is redrawn.
SAME_TARGET_REDRAWS = 10
#: How many times a negative that is in the prefix (or is the target) is redrawn.
NEGATIVE_REDRAWS = 100


# ------------------------------------------------------------------ evaluation data


def load_eval_split(processed_dir: str, split: str, max_seq_length: int):
    """`(sequences (N, S) long, targets (N,), user_ids (N,))` of the val or test split at
    width `S = max_seq_length`.

    When the split is stored at width >= S this is `load_tensorised` narrowing. When it is
    stored narrower (Beauty is stored at 20; the baselines read 50), the sequences are
    rebuilt from the same histories preprocessing used -- val: the last S training items,
    test: the last S of training + validation -- and checked to agree with the stored rows
    on the stored width, so row order and targets are exactly the stored split's.
    """
    from data.sequence_generator import load_tensorised

    path = os.path.join(processed_dir, f"{split}_tensorised.pt")
    data = torch.load(path, weights_only=True)
    stored = data["sequences"].shape[1]
    S = int(max_seq_length)
    if S <= stored:
        ds = load_tensorised(path, None, max_seq_length=S)
        return ds.sequences.long(), ds.targets.long(), ds.user_ids.long()
    history_file = {"val": "train_user_history.pkl", "test": "eval_user_history.pkl"}[split]
    with open(os.path.join(processed_dir, history_file), "rb") as f:
        histories = pickle.load(f)
    user_ids = data["user_ids"].long()
    seqs = np.zeros((len(user_ids), S), dtype=np.int64)
    for i, u in enumerate(user_ids.tolist()):
        tail = list(histories.get(int(u), ()))[-S:]
        if tail:
            seqs[i, S - len(tail):] = tail
    seqs = torch.from_numpy(seqs)
    if not torch.equal(seqs[:, -stored:], data["sequences"].long()):
        raise ValueError(
            f"{path}: sequences rebuilt from {history_file} at width {S} do not match the "
            f"stored rows on the stored width {stored}"
        )
    return seqs, data["targets"].long(), user_ids


# ------------------------------------------------------------------ training feed


class PrefixFeed:
    """Every chunk position with a target, as a left-padded prefix example, on `device`.

    Example `e` is chunk row `rows[e]` cut after position `pos[e]` and shifted right so its
    last input sits at slot S - 1; its target is `chunk_targets[rows[e], pos[e]]`.
    """

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor, num_items: int,
                 device: torch.device, seed: int, negatives: bool = False,
                 same_target: bool = False):
        self.device = device
        self.num_items = int(num_items)
        self.S = sequences.shape[1]
        self.sequences = sequences.long().to(device)
        nz = (targets > 0).nonzero(as_tuple=False)            # row-major
        self.rows = nz[:, 0].to(device)
        self.pos = nz[:, 1].to(device)
        self.targets = targets[nz[:, 0], nz[:, 1]].long().to(device)
        self.negatives = negatives
        self.same_target = same_target
        # Epoch order on CPU (one randperm per epoch); everything else on the device.
        self.order_gen = torch.Generator().manual_seed(int(seed))
        self.gen = torch.Generator(device=device).manual_seed(int(seed) + 1)
        if same_target:
            self._build_same_target_index()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def _build_same_target_index(self) -> None:
        """Group examples by target: `group_order[group_start[t] + r]` is the r-th
        example with target t, and `rank[e]` is example e's own r."""
        order = torch.argsort(self.targets, stable=True)
        counts = torch.bincount(self.targets, minlength=self.num_items + 2)
        start = torch.zeros_like(counts)
        start[1:] = torch.cumsum(counts, 0)[:-1]
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order), device=order.device) - start[self.targets[order]]
        self.group_order, self.group_start, self.group_count, self.rank = order, start, counts, rank

    def epoch_order(self) -> torch.Tensor:
        return torch.randperm(len(self), generator=self.order_gen).to(self.device)

    def prefixes(self, ex: torch.Tensor) -> torch.Tensor:
        rows = self.sequences[self.rows[ex]]                      # (B, S)
        shift = (self.S - 1) - self.pos[ex]
        src = torch.arange(self.S, device=self.device)[None, :] - shift[:, None]
        out = rows.gather(1, src.clamp(min=0))
        return out.masked_fill(src < 0, 0)

    def sample_negatives(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """One uniform item in 1..num_items per example, not in its prefix or its target
        (the reference's `neg_sample(set(items), item_size)`)."""
        B = target.shape[0]
        neg = torch.randint(1, self.num_items + 1, (B,), generator=self.gen, device=self.device)
        for _ in range(NEGATIVE_REDRAWS):
            bad = (inputs == neg[:, None]).any(1) | (neg == target)
            if not bool(bad.any()):
                return neg
            fresh = torch.randint(1, self.num_items + 1, (B,), generator=self.gen, device=self.device)
            neg = torch.where(bad, fresh, neg)
        raise RuntimeError("could not sample a negative outside the prefix")

    def sample_same_target(self, ex: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """DuoRec's supervised view: another training example with the same target,
        uniformly, redrawn while it equals this example's own input (the reference's
        `while keep_random and sem_aug == items`). A target seen once uses itself."""
        t = self.targets[ex]
        g = self.group_count[t]
        start = self.group_start[t]
        own = self.rank[ex]
        multi = g > 1
        last = len(self.group_order) - 1
        sem_inputs = inputs
        need = multi
        for _ in range(SAME_TARGET_REDRAWS):
            # r uniform over the g - 1 OTHER members: draw in [0, g - 2], skip own rank.
            u = torch.rand(ex.shape[0], generator=self.gen, device=self.device)
            r = torch.minimum((u * (g - 1)).long(), (g - 2).clamp(min=0))
            r = r + (r >= own).long()
            cand = torch.where(multi, self.group_order[(start + r).clamp(max=last)], ex)
            sem_inputs = torch.where(need[:, None], self.prefixes(cand), sem_inputs)
            need = need & (sem_inputs == inputs).all(1)
            if not bool(need.any()):
                break
        return sem_inputs

    def batch(self, ex: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs = self.prefixes(ex)
        target = self.targets[ex]
        out = {"input": inputs, "target": target}
        if self.negatives:
            out["negative"] = self.sample_negatives(inputs, target)
        if self.same_target:
            out["same_target"] = self.sample_same_target(ex, inputs)
        return out


# ------------------------------------------------------------------ evaluation


@torch.no_grad()
def evaluate_baseline(model, sequences: torch.Tensor, ranking, k: int, batch_size: int,
                      return_per_example: bool = False):
    """Our protocol for a baseline: fp32 dot-product scores over the full catalogue minus
    the user's history (`catalogue_ranking_scores`), tie-averaged metrics."""
    from evaluation.metrics import aggregate_metrics, catalogue_ranking_scores, compute_metrics_batch

    model.eval()
    device = next(model.parameters()).device
    n = sequences.shape[0]
    collected: Dict[str, list] = {}
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        scores = model.full_scores(sequences[start:stop].to(device))
        scores = catalogue_ranking_scores(scores, ranking.rows(start, stop))
        for name, val in compute_metrics_batch(scores, target_idx=0, k=k).items():
            collected.setdefault(name, []).append(val.detach().cpu())
    per_example = {name: torch.cat(v) for name, v in collected.items()}
    metrics = aggregate_metrics(per_example)
    return (metrics, per_example) if return_per_example else metrics


# ------------------------------------------------------------------ checkpoints


def _checkpoint(model, optimizer, config: Config, epoch: int, val_metrics: Dict) -> dict:
    from utils import get_source_revision

    return {
        "epoch": epoch,
        "baseline_model": config.baseline_model,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_ndcg": val_metrics.get("ndcg", {}).get("mean", 0.0),
        "val_mrr": val_metrics.get("mrr", {}).get("mean", 0.0),
        "val_recall": val_metrics.get("recall", {}).get("mean", 0.0),
        "val_precision": val_metrics.get("precision", {}).get("mean", 0.0),
        "parameters": count_baseline_parameters(model),
        "config": asdict(config),
        "source_revision": get_source_revision(),
    }


def load_baseline_checkpoint(path: str, config: Config, device: Optional[torch.device] = None):
    """`(model, checkpoint)` from a baseline `best.pt`, in eval mode."""
    from utils import get_device

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("baseline_model") != config.baseline_model:
        raise ValueError(
            f"{path} holds baseline '{ckpt.get('baseline_model')}', not '{config.baseline_model}'"
        )
    model = build_baseline(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device or get_device(config)).eval()
    return model, ckpt


def _capture_rng(feed: PrefixFeed, device: torch.device) -> dict:
    return {
        "order_gen": feed.order_gen.get_state(),
        "gen": feed.gen.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict, feed: PrefixFeed, device: torch.device) -> None:
    feed.order_gen.set_state(state["order_gen"])
    feed.gen.set_state(state["gen"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _log_epoch(epoch_logger, epoch: int, losses: Dict[str, float], val: Dict, lr: float) -> None:
    """Same leading tokens as `training.pretrain.log_epoch` (`epoch=N | train_loss_total=`
    ... `val_ndcg@10=`), so the existing log readers work on baseline logs."""
    from utils import get_source_revision

    parts = [f"epoch={epoch}", f"train_loss_total={losses.get('total', 0.0):.6f}"]
    parts += [f"train_loss_{k}={v:.6f}" for k, v in losses.items() if k != "total"]
    for m in ("ndcg", "mrr", "recall", "precision"):
        parts.append(f"val_{m}@10={val.get(m, {}).get('mean', 0.0):.6f}")
    parts.append(f"lr={lr:.8f}")
    rev = get_source_revision()
    parts += [f"source_commit={rev.get('commit')}", f"source_hash={rev.get('source_hash')}",
              f"source_dirty={rev.get('dirty')}"]
    epoch_logger.info(" | ".join(parts))


# ------------------------------------------------------------------ training loop


def run_baseline_pretrain(config: Config, model_name: str) -> Dict:
    """Train `config.baseline_model`; same return keys as `training.pretrain.run_pretrain`,
    plus `parameters` and `early_stopped`. `model_name` only names the log and the
    checkpoint subdirectory (as for our arms), it does not size the model."""
    from data.preprocessing import check_processed_dir
    from data.sequence_generator import load_train_dataset
    from evaluation.metrics import evaluate_popularity, item_popularity, load_full_ranking
    from training.pretrain import (
        LAST_CHECKPOINT, POPULARITY_EVAL_BATCH, RESUME_FORMAT, _load_resume_checkpoint,
        _training_fingerprint_of, _truncate_epoch_log, log_popularity_header,
        setup_epoch_logger,
    )
    from utils import atomic_torch_save, get_device

    config.validate()
    if not config.baseline_model:
        raise ValueError("run_baseline_pretrain needs config.baseline_model")
    check_processed_dir(config)
    device = get_device(config)
    processed = config.processed_data_dir
    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items, config.num_users = stats["num_items"], stats["num_users"]
    config.vocab_size = config.num_items + 2
    S = config.max_seq_length

    train = load_train_dataset(processed, S, None)
    val_seq, val_targets, val_users = load_eval_split(processed, "val", S)
    val_ranking = load_full_ranking(processed, "val", val_targets, val_users)
    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_history = pickle.load(f)
    val_popularity = evaluate_popularity(
        val_ranking, item_popularity(train_history, config.vocab_size), k=config.eval_k,
        batch_size=POPULARITY_EVAL_BATCH,
    )
    del train_history

    model = build_baseline(config).to(device)
    params = config.baseline_params
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        betas=(float(params["adam_beta1"]), float(params["adam_beta2"])),
        weight_decay=config.weight_decay,
    )
    feed = PrefixFeed(
        train.sequences, train.targets, config.num_items, device, config.seed,
        negatives=model.needs_negatives, same_target=model.needs_same_target,
    )
    bs = config.batch_size
    steps_per_epoch = math.ceil(len(feed) / bs)
    if config.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)
    total_steps = steps_per_epoch * config.pretrain_epochs
    n_params = count_baseline_parameters(model)
    logger.info("%s: %d prefix examples, %d steps/epoch, %s", config.baseline_model,
                len(feed), steps_per_epoch, n_params)

    ckpt_dir = config.model_checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, LAST_CHECKPOINT)
    best_path = os.path.join(ckpt_dir, "best.pt")
    epoch_log_path = os.path.join(config.log_dir, f"pretrain_{model_name.replace('-', '_')}.log")

    start_epoch, best_val, best_epoch = 0, -1.0, -1
    epoch_mean_losses = []
    resumed = None
    if config.resume_training and os.path.exists(last_path):
        resumed = _load_resume_checkpoint(last_path, config, total_steps)
        state = resumed["resume_state"]
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        _restore_rng(state["rng"], feed, device)
        start_epoch = int(state["epoch"]) + 1
        best_val, best_epoch = float(state["best_val_ndcg"]), int(state["best_epoch"])
        epoch_mean_losses = list(state["epoch_mean_losses"])
        logger.info("Resuming %s from %s at epoch %d", config.baseline_model, last_path, start_epoch)
        _truncate_epoch_log(epoch_log_path, start_epoch)
    epoch_logger = setup_epoch_logger(config, model_name, append=resumed is not None)
    if resumed is None:
        log_popularity_header(epoch_logger, val_popularity, config.eval_k, len(val_ranking))

    patience = config.early_stop_patience
    # A run killed after writing the last.pt that recorded its early stop is finished.
    early_stopped = resumed["resume_state"].get("early_stopped") if resumed is not None else None
    for epoch in range(start_epoch, config.pretrain_epochs):
        if early_stopped is not None:
            break
        model.train()
        order = feed.epoch_order()
        sums: Dict[str, torch.Tensor] = {}
        for step in range(steps_per_epoch):
            ex = order[step * bs:(step + 1) * bs]
            loss, parts = model.training_loss(feed.batch(ex), feed.gen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in {"total": loss.detach(), **parts}.items():
                sums[name] = sums.get(name, 0.0) + value.float()
        losses = {name: float(v) / steps_per_epoch for name, v in sums.items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise AssertionError(f"{config.baseline_model}: non-finite training loss {losses}")
        epoch_mean_losses.append(losses["total"])

        val = evaluate_baseline(model, val_seq, val_ranking, config.eval_k, config.eval_batch_size)
        _log_epoch(epoch_logger, epoch, losses, val, optimizer.param_groups[0]["lr"])
        val_ndcg = val.get("ndcg", {}).get("mean", 0.0)

        interval = config.epoch_snapshot_interval
        if interval > 0 and (epoch + 1) % interval == 0:
            atomic_torch_save(_checkpoint(model, optimizer, config, epoch, val),
                              os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))
        if val_ndcg > best_val:   # strict, as run_pretrain
            best_val, best_epoch = val_ndcg, epoch
            ckpt = _checkpoint(model, optimizer, config, epoch, val)
            ckpt.update({"best_epoch": epoch, "best_val_ndcg": val_ndcg, "seed": config.seed})
            atomic_torch_save(ckpt, best_path)
        if patience > 0 and epoch - best_epoch >= patience:
            early_stopped = {"patience": patience, "last_completed_epoch": epoch}
            logger.info("%s: early stop at epoch %d (best %d, val NDCG@10 %.6f)",
                        config.baseline_model, epoch, best_epoch, best_val)

        last = _checkpoint(model, optimizer, config, epoch, val)
        last["resume_state"] = {
            "format": RESUME_FORMAT,
            "epoch": epoch,
            "global_step": (epoch + 1) * steps_per_epoch,
            "total_steps": total_steps,
            "training_fingerprint": _training_fingerprint_of(config),
            "best_val_ndcg": best_val,
            "best_epoch": best_epoch,
            "epoch_mean_losses": list(epoch_mean_losses),
            "early_stopped": early_stopped,
            "rng": _capture_rng(feed, device),
        }
        atomic_torch_save(last, last_path)
        logger.info("%s epoch %d: loss %.4f val_ndcg %.4f best %.4f (epoch %d)",
                    config.baseline_model, epoch, losses["total"], val_ndcg, best_val, best_epoch)

    return {
        "best_val_ndcg": best_val,
        "best_epoch": best_epoch,
        "epoch_mean_losses": epoch_mean_losses,
        "epoch_loss_windows": [],
        "val_popularity_reference": val_popularity,
        "parameters": n_params,
        "num_train_examples": len(feed),
        "early_stopped": early_stopped,
    }


# ------------------------------------------------------------------ ablation arm


def evaluate_baseline_arm(cfg: Config, ckpt_path: str, result: Dict) -> Dict:
    """Fill an arm `result` the way `_run_single_config` does for our models: test
    metrics on the full test split (or the seeded sample), popularity floor, protocol,
    runtime, provenance and parameter count."""
    import platform

    from evaluation.metrics import (
        EVAL_PROTOCOL, evaluate_popularity, load_full_ranking, load_item_popularity,
    )
    from training.ablation_runner import select_eval_rows
    from training.pretrain import POPULARITY_EVAL_BATCH
    from utils import artifact_source_revision, get_device

    device = get_device(cfg)
    model, ckpt = load_baseline_checkpoint(ckpt_path, cfg, device)
    result["runtime"] = {
        "attention_backend": "baseline_eager_fp32",
        "attention_fallback_reason": None,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else platform.processor() or platform.machine() or device.type),
        "dtype": "fp32",
        "mixed_precision": False,
    }
    result["source_revision"] = artifact_source_revision(ckpt)
    result["parameters"] = {k: float(v) for k, v in count_baseline_parameters(model).items()}
    result["baseline_model"] = cfg.baseline_model
    result["baseline_params"] = dict(cfg.baseline_params)

    processed = cfg.processed_data_dir
    seqs, targets, users = load_eval_split(processed, "test", cfg.max_seq_length)
    ranking = load_full_ranking(processed, "test", targets, users)
    idx, test_eval = select_eval_rows(len(targets), cfg.ablation_eval_examples, cfg.ablation_eval_seed)
    if idx is not None:
        seqs, ranking = seqs[torch.from_numpy(idx)], ranking.select(idx)
    metrics = evaluate_baseline(model, seqs, ranking, cfg.eval_k, cfg.eval_batch_size)
    result["metrics"] = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in metrics.items()}
    result["num_eval_examples"] = test_eval["num_examples"]
    result["test_eval"] = test_eval
    result["popularity_reference"] = evaluate_popularity(
        ranking, load_item_popularity(processed, cfg.vocab_size), k=cfg.eval_k,
        batch_size=POPULARITY_EVAL_BATCH,
    )
    result["eval_protocol"] = dict(EVAL_PROTOCOL)
    result["status"] = "ok"
    return result
