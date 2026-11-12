import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config
from models.transformer import l2_normalize
from utils import autocast_context, autocast_disabled

#: Stamped into every result file that reports ranking metrics (specs/eval-integrity
#: FR-1.5): fp32 scores, tie-averaged metrics. Version 1 files used bf16 scores under
#: mixed precision with ties to the target; version 2 ranked each target against 100
#: sampled negatives on an 80/10/10 split. Version 3 is leave-one-out, ranked against the
#: full catalogue minus the user's earlier interactions. No two versions are comparable.
EVAL_PROTOCOL = {
    "version": 3, "scoring": "fp32", "ties": "tie_averaged",
    "split": "leave_one_out", "candidates": "full_catalogue",
}


@dataclass(frozen=True)
class FullRanking:
    """Evaluation rows ranked against the whole catalogue (owner decision 2026-10-25).

    Row `i` ranks `targets[i]` against every item `1..num_items` except the user's
    earlier interactions, `exclude_items[exclude_ptr[i]:exclude_ptr[i + 1]]`. Padding and
    MASK are never candidates. Takes the place of a `(N, C)` candidate matrix wherever
    `evaluate_model` / `evaluate_popularity` accept one.
    """

    targets: torch.Tensor        # (N,) int64
    exclude_ptr: torch.Tensor    # (N + 1,) int64, CSR offsets into exclude_items
    exclude_items: torch.Tensor  # (E,) int64

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def rows(self, start: int, stop: int) -> "FullRanking":
        """Rows `start:stop`, with offsets rebased to 0 (a view of the item array)."""
        lo, hi = int(self.exclude_ptr[start]), int(self.exclude_ptr[stop])
        return FullRanking(
            self.targets[start:stop],
            self.exclude_ptr[start:stop + 1] - lo,
            self.exclude_items[lo:hi],
        )

    def select(self, indices) -> "FullRanking":
        """The rows at `indices`, in that order."""
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        starts = self.exclude_ptr[idx]
        counts = self.exclude_ptr[idx + 1] - starts
        ptr = torch.zeros(len(idx) + 1, dtype=torch.long)
        torch.cumsum(counts, 0, out=ptr[1:])
        offsets = torch.repeat_interleave(starts - ptr[:-1], counts) + torch.arange(
            int(ptr[-1]), dtype=torch.long,
        )
        return FullRanking(self.targets[idx], ptr, self.exclude_items[offsets])


def build_full_ranking(
    targets, user_ids, histories: Mapping[int, Sequence[int]],
) -> FullRanking:
    """One row per example: its target, excluding `histories[user]`."""
    arrays = [np.asarray(histories.get(int(u), ()), dtype=np.int64).ravel() for u in user_ids]
    counts = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    ptr = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    items = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)
    return FullRanking(
        torch.as_tensor(np.asarray(targets), dtype=torch.long).reshape(-1),
        torch.from_numpy(ptr),
        torch.from_numpy(items),
    )


#: The history each split's rows exclude: validation targets follow the training
#: history; test targets follow training + validation (`eval_user_history.pkl`).
_EXCLUDED_HISTORY = {"val": "train_user_history.pkl", "test": "eval_user_history.pkl"}


def load_full_ranking(processed_dir: str, split: str, targets, user_ids) -> FullRanking:
    """`build_full_ranking` over the stored history for `split` ("val" or "test")."""
    if split not in _EXCLUDED_HISTORY:
        raise ValueError(f"split must be one of {sorted(_EXCLUDED_HISTORY)}, got {split!r}")
    with open(os.path.join(processed_dir, _EXCLUDED_HISTORY[split]), "rb") as f:
        histories = pickle.load(f)
    return build_full_ranking(targets, user_ids, histories)


def catalogue_ranking_scores(item_scores: torch.Tensor, ranking: FullRanking) -> torch.Tensor:
    """`(B, vocab)` per-item scores -> `(B, vocab + 1)` with the target in column 0.

    Column 0 holds each row's target score; every excluded item, padding (0), MASK
    (`vocab - 1`) and the target's own column are `-inf`, so they rank below every real
    candidate and never tie the target. Ready for `compute_metrics_batch(target_idx=0)`.
    Modifies `item_scores` in place.
    """
    B, V = item_scores.shape
    device = item_scores.device
    targets = ranking.targets.to(device)
    target_scores = item_scores.gather(1, targets[:, None])
    counts = ranking.exclude_ptr[1:] - ranking.exclude_ptr[:-1]
    rows = torch.repeat_interleave(torch.arange(B), counts).to(device)
    neg_inf = float("-inf")
    item_scores[rows, ranking.exclude_items.to(device)] = neg_inf
    item_scores[:, 0] = neg_inf
    item_scores[:, V - 1] = neg_inf
    item_scores.scatter_(1, targets[:, None], neg_inf)
    return torch.cat([target_scores, item_scores], dim=1)


def compute_metrics_batch(
    scores: torch.Tensor, target_idx: int = 0, k: int = 10
) -> Dict[str, torch.Tensor]:
    """Ranking metrics for a `(B, C)` score matrix, one row per example.

    Ties never favour the target (specs/eval-integrity FR-2). Each metric is its EXPECTED
    value under a uniformly random ordering of tied scores -- the tie-averaged measure of
    McSherry & Najork (2008). With `h` candidates strictly higher than the target and `t`
    other candidates exactly equal to it, the target's rank is uniform on
    `{h+1, ..., h+t+1}` and each metric is the mean of its per-rank value over that range.
    (The former rule, `rank = 1 + h`, gave every tie to the target: bf16 cosines tied the
    target in 41-52% of examples and inflated NDCG@10 by up to 0.08.)

    REV-076 still holds: no sort and no per-example `.item()`. The mean over a rank range
    is a difference of a cumulative per-rank table, gathered on the device. With no ties
    the result is exactly the former per-rank formula (FR-2.2). A NaN score raises
    `ValueError` -- ranking a NaN is undefined (FR-2.5); one host sync per batch. `+-inf`
    is ordered and allowed (ANN masks pad/MASK with `-inf`).
    """
    if scores.dim() != 2:
        raise ValueError(f"compute_metrics_batch expects (B, C) scores, got {tuple(scores.shape)}")
    if torch.isnan(scores).any():
        raise ValueError(
            f"compute_metrics_batch: score matrix of shape {tuple(scores.shape)} contains NaN; "
            "ranking a NaN is undefined"
        )
    scores = scores.float()
    num_candidates = scores.shape[1]
    device = scores.device
    target_scores = scores[:, target_idx : target_idx + 1]
    higher = (scores > target_scores).sum(dim=1)                    # h
    ties = (scores == target_scores).sum(dim=1) - 1                 # t (target excluded)

    # Per-rank values in the former float32 arithmetic, used verbatim where t == 0 so a
    # tie-free row is bit-identical to the old metric (AC-03).
    rank = higher + 1
    rank_f = rank.float()
    hit = (rank <= k).float()
    single = {
        "ndcg": hit / torch.log2(rank_f + 1.0),
        "mrr": hit / rank_f,
        "recall": hit,
        "precision": hit / k,
    }

    # Tie-averaged values: cumulative per-rank tables in float64, `cum[r]` = sum over
    # ranks 1..r. MPS has no float64, so the table is built on CPU and moved.
    r = torch.arange(1, num_candidates + 1, dtype=torch.float64)
    in_k = (r <= k).to(torch.float64)
    tables = {
        "ndcg": in_k / torch.log2(r + 1.0),
        "mrr": in_k / r,
        "recall": in_k,
        "precision": in_k / k,
    }
    lo = higher                                   # cum index of ranks 1..h
    hi = higher + ties + 1                        # cum index of ranks 1..h+t+1
    has_tie = ties > 0
    result = {}
    for name, table in tables.items():
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(table, 0)])
        cum = cum.to(device=device, dtype=torch.float32 if device.type == "mps" else torch.float64)
        averaged = ((cum[hi] - cum[lo]) / (ties + 1).to(cum.dtype)).float()
        result[name] = torch.where(has_tie, averaged, single[name])
    return result


def compute_metrics(scores: torch.Tensor, target_idx: int = 0, k: int = 10) -> Dict[str, float]:
    """Single-example ranking metrics. Kept for toy tests and per-example callers."""
    batched = compute_metrics_batch(scores.reshape(1, -1), target_idx=target_idx, k=k)
    return {name: float(val[0].item()) for name, val in batched.items()}


def aggregate_metrics(
    per_example: Union[List[Dict[str, float]], Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """Mean/std per metric, over either a list of per-example dicts or stacked tensors."""
    if isinstance(per_example, dict):
        result = {}
        for key, t in per_example.items():
            t = t.detach().float().cpu()
            result[key] = {
                "mean": t.mean().item() if t.numel() else 0.0,
                "std": t.std().item() if t.numel() > 1 else 0.0,
            }
        return result

    if not per_example:
        return {}

    keys = per_example[0].keys()
    result = {}
    for key in keys:
        vals = [ex[key] for ex in per_example]
        t = torch.tensor(vals)
        result[key] = {
            "mean": t.mean().item(),
            "std": t.std().item() if len(vals) > 1 else 0.0,
        }
    return result


def _user_embeddings(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`(B, D)` fp32 user embeddings (L2-normalised when configured), eval mode, no grad.

    specs/eval-integrity FR-1: the forward (and MoE) run in their own autocast region --
    bf16 under mixed precision -- but the upcast and normalisation run in fp32 with
    autocast OFF, even when the caller has a region open.
    """
    model.eval()
    # The MoE layer is a separate module, so `model.eval()` doesn't reach it. A caller
    # that never switched it would score with expert dropout on, so the NDCG of the same
    # frozen model would differ run to run.
    if moe_layer is not None:
        moe_layer.eval()
    with torch.no_grad():
        # The eval datasets are held on CPU while the model lives on the accelerator, so
        # move each batch across here. CUDA rejects a cross-device index outright, and
        # MPS rejects it as "Passed CPU tensor to MPS op" -- every scoring caller
        # (validation, test, cold-start, long-tail) funnels through here.
        param_device = next(model.parameters()).device
        seqs = seqs.to(param_device)
        if metadata is not None:
            metadata = metadata.to(device=param_device, dtype=model.item_emb_weight.dtype)

        # The region's device comes from the parameters, not the config: a region for
        # the wrong device type would silently leave the forward in fp32.
        with autocast_context(config, device_type=param_device.type):
            all_hidden = model(seqs, metadata=metadata, return_all_hidden=True)
            pooled = all_hidden[:, -1, :]

            if moe_layer is not None and moe_layer.variant != "none":
                # REV-045: MoELayer invalidates its expert cache on a new input, so scoring
                # batch N no longer reuses batch 0's expert outputs.
                Z_ut, _ = moe_layer(all_hidden, pooled, 0)
                user_emb = Z_ut[:, -1, :]
            else:
                user_emb = pooled

        with autocast_disabled(param_device.type):
            # Upcast BEFORE normalising: `l2_normalize` returns its input dtype, so a bf16
            # input would come back bf16.
            user_emb = user_emb.float()
            if config.normalize_embeddings:
                user_emb = l2_normalize(user_emb)
    return user_emb


def score_candidates_batch(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Score `(B, C)` candidates for `(B, S)` sequences in ONE forward pass.

    REV-063: evaluation used to forward a batch of exactly 1 per example. Everything
    here is the same arithmetic as the single-example path, run over a batch dimension.

    specs/eval-integrity FR-1: one of the two scoring choke points (with
    `score_catalogue_batch`). The L2 normalisation, the dot products and the temperature
    run in fp32 with autocast OFF. The returned scores are always float32: bf16 cosines
    tied the target in 41-52% of examples.
    """
    user_emb = _user_embeddings(model, seqs, config, moe_layer, metadata)
    device = user_emb.device
    with torch.no_grad(), autocast_disabled(device.type):
        item_embs = model.item_emb_weight[candidate_ids.to(device)].float()  # (B, C, D)
        # REV-044: FR-8 / AC-07 -- L2-normalize both sides before the dot product so
        # scores are bounded to [-1, 1].
        if config.normalize_embeddings:
            item_embs = l2_normalize(item_embs)

        scores = torch.einsum("bd,bcd->bc", user_emb, item_embs)
        # The training logit scale (`logit_temperature` / `learnable_logit_scale`,
        # training/losses.LogitScale) is deliberately NOT applied: every metric here
        # depends only on the rank of the target among the candidates, and ranks are
        # invariant to multiplying all scores by a positive scalar (ties included).
        if temperature != 1.0:
            scores = scores / temperature

    assert scores.dtype == torch.float32, f"scores must be float32, got {scores.dtype}"
    return scores


def score_catalogue_batch(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """`(B, vocab)` fp32 scores of every embedding row for `(B, S)` sequences.

    Columns are item ids, padding (0) and MASK (last) included -- `catalogue_ranking_scores`
    takes them out. Same arithmetic as `score_candidates_batch` with every item as a
    candidate: fp32, autocast off, no logit scale.
    """
    user_emb = _user_embeddings(model, seqs, config, moe_layer, metadata)
    device = user_emb.device
    with torch.no_grad(), autocast_disabled(device.type):
        item_embs = model.item_emb_weight.float()
        if config.normalize_embeddings:
            item_embs = l2_normalize(item_embs)
        scores = user_emb @ item_embs.T
        if temperature != 1.0:
            scores = scores / temperature
    assert scores.dtype == torch.float32, f"scores must be float32, got {scores.dtype}"
    return scores


def score_candidates(
    model: torch.nn.Module,
    seq: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single-example scoring. Thin wrapper over the batched path."""
    if metadata is not None and metadata.dim() == 2:
        metadata = metadata.unsqueeze(0)
    scores = score_candidates_batch(
        model, seq.unsqueeze(0), candidate_ids.unsqueeze(0), config,
        moe_layer=moe_layer, metadata=metadata, temperature=temperature,
    )
    return scores.squeeze(0)


def _stack_sequences(eval_sequences, indices) -> torch.Tensor:
    if isinstance(eval_sequences, torch.Tensor):
        return eval_sequences[indices[0] : indices[-1] + 1]
    return torch.stack([
        s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long)
        for s in (eval_sequences[i] for i in indices)
    ]).long()


def _stack_candidates(candidate_sets, indices) -> torch.Tensor:
    if isinstance(candidate_sets, torch.Tensor):
        return candidate_sets[indices[0] : indices[-1] + 1].long()
    return torch.stack([
        c if isinstance(c, torch.Tensor) else torch.as_tensor(c, dtype=torch.long)
        for c in (candidate_sets[i] for i in indices)
    ]).long()


def evaluate_model(
    model: torch.nn.Module,
    eval_sequences: Union[torch.Tensor, Sequence[torch.Tensor]],
    candidate_sets: Union[FullRanking, torch.Tensor, Sequence[torch.Tensor]],
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata_per_example: Optional[Sequence[torch.Tensor]] = None,
    metadata_matrix: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Batched evaluation over an aligned set of sequences and candidate sets.

    `candidate_sets` is normally a `FullRanking` (the protocol: each target ranked
    against the whole catalogue minus the user's history). A stacked `(N, C)` tensor or a
    sequence of per-example tensors, target in column 0, ranks within those candidates
    instead -- the ANN retrieval path. `eval_sequences` is a stacked `(N, S)` tensor (what
    `RecDataset` holds) or a sequence of per-example tensors.

    REV-058 / REV-063: prefer `metadata_matrix`, the `(vocab, F)` matrix, over
    `metadata_per_example`. Passing the matrix lets each batch gather only its own
    `(B, S, F)` block instead of the caller materialising a dense `(N, S, F)` list.
    """
    temperature = config.eval_temperature if config.use_logit_calibration else 1.0
    n = eval_sequences.shape[0] if isinstance(eval_sequences, torch.Tensor) else len(eval_sequences)
    if n == 0:
        return {}

    bs = batch_size or max(1, config.eval_batch_size)
    collected: Dict[str, List[torch.Tensor]] = {}

    for start in range(0, n, bs):
        indices = list(range(start, min(start + bs, n)))
        seqs = _stack_sequences(eval_sequences, indices)

        meta = None
        if metadata_matrix is not None:
            meta = metadata_matrix[seqs.to(metadata_matrix.device)]
        elif metadata_per_example is not None:
            meta = torch.stack([
                torch.as_tensor(metadata_per_example[i]) for i in indices
            ])

        if isinstance(candidate_sets, FullRanking):
            scores = catalogue_ranking_scores(
                score_catalogue_batch(
                    model, seqs, config, moe_layer=moe_layer,
                    metadata=meta, temperature=temperature,
                ),
                candidate_sets.rows(indices[0], indices[-1] + 1),
            )
        else:
            scores = score_candidates_batch(
                model, seqs, _stack_candidates(candidate_sets, indices), config,
                moe_layer=moe_layer, metadata=meta, temperature=temperature,
            )
        for name, val in compute_metrics_batch(scores, target_idx=0, k=config.eval_k).items():
            collected.setdefault(name, []).append(val.detach().cpu())

    return aggregate_metrics({name: torch.cat(vals) for name, vals in collected.items()})


# ---------------------------------------------------------------------------------------
# Popularity reference (specs/eval-integrity FR-7)
# ---------------------------------------------------------------------------------------


def item_popularity(
    train_user_history: Mapping[int, Sequence[int]], vocab_size: int
) -> torch.Tensor:
    """`(vocab,)` float32 training-interaction counts; pad (0), MASK and unseen items are 0.

    The same source the long-tail buckets use (`train_user_history.pkl`). Counts are exact
    in float32: the largest is at most the number of users, far below 2^24.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2 (pad + MASK), got {vocab_size}")
    arrays = [np.asarray(items, dtype=np.int64).ravel() for items in train_user_history.values()]
    ids = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= vocab_size):
        raise ValueError(
            f"train_user_history holds item ids outside [0, {vocab_size}); "
            "the processed data does not match this vocabulary"
        )
    counts = np.bincount(ids, minlength=vocab_size).astype(np.float32)
    counts[0] = 0.0                # padding is never an item
    counts[vocab_size - 1] = 0.0   # MASK = num_items + 1 = vocab_size - 1
    return torch.from_numpy(counts)


def load_item_popularity(processed_dir: str, vocab_size: int) -> torch.Tensor:
    """`item_popularity` over `{processed_dir}/train_user_history.pkl`."""
    with open(os.path.join(processed_dir, "train_user_history.pkl"), "rb") as f:
        history = pickle.load(f)
    return item_popularity(history, vocab_size)


def popularity_scores(candidates: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    """`(N, C)` float32 popularity-only scores for a candidate matrix."""
    candidates = torch.as_tensor(candidates)
    return popularity[candidates.long().to(popularity.device)]


def popularity_ranking_scores(ranking: FullRanking, popularity: torch.Tensor) -> torch.Tensor:
    """The popularity-only ranker's `catalogue_ranking_scores` for these rows."""
    item_scores = popularity.reshape(1, -1).expand(len(ranking), -1).clone()
    return catalogue_ranking_scores(item_scores, ranking)


def evaluate_popularity(
    candidates: Union[FullRanking, torch.Tensor, np.ndarray],
    popularity: torch.Tensor,
    k: int,
    batch_size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """The popularity-only ranker's metrics on these rows: a `FullRanking`, or candidate
    rows with the target in column 0.

    Ranked with `compute_metrics_batch`, so tied counts are tie-averaged exactly as the
    model's scores are. Same `{metric: {mean, std}}` shape as `evaluate_model`.
    """
    full = isinstance(candidates, FullRanking)
    n = len(candidates) if full else int(candidates.shape[0])
    if n == 0:
        return {}
    bs = batch_size or n
    collected: Dict[str, List[torch.Tensor]] = {}
    for start in range(0, n, bs):
        if full:
            scores = popularity_ranking_scores(
                candidates.rows(start, min(start + bs, n)), popularity,
            )
        else:
            rows = candidates[start : start + bs]
            if not isinstance(rows, torch.Tensor):
                rows = torch.as_tensor(np.asarray(rows))   # e.g. a memory-mapped matrix
            scores = popularity_scores(rows, popularity)
        for name, val in compute_metrics_batch(scores, target_idx=0, k=k).items():
            collected.setdefault(name, []).append(val.detach().cpu())
    return aggregate_metrics({name: torch.cat(vals) for name, vals in collected.items()})
