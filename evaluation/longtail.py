import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from configs.config import Config
from evaluation.metrics import (
    score_candidates, score_candidates_batch, compute_metrics,
    compute_metrics_batch, aggregate_metrics,
)
from data.sequence_generator import (
    load_metadata_matrix, load_or_build_eval_candidates,
)
from utils import get_device

logger = logging.getLogger(__name__)

# FR-5 / T3.1: head 20%, mid 20%, tail 60% of items ranked by training popularity.
HEAD_FRACTION = 0.20
MID_FRACTION = 0.20
TAIL_FRACTION = 0.60


def compute_popularity_buckets(
    train_interactions: Dict[int, List[int]],
    all_item_ids: Set[int],
    save_path: Optional[str] = None,
) -> Dict[str, Set[int]]:
    item_counts: Dict[int, int] = {}
    for user_items in train_interactions.values():
        for item in user_items:
            item_counts[item] = item_counts.get(item, 0) + 1

    # Ties are broken by item id so the bucketing is deterministic.
    sorted_items = sorted(item_counts.keys(), key=lambda x: (-item_counts[x], x))
    n = len(sorted_items)

    # REV-007: `mid_cutoff` was computed as a POSITION (int(n*0.40)) but used as a
    # WIDTH, producing head 20% / mid 40% / tail 40%. FR-5 and T3.1 both specify
    # 20 / 20 / 60. Both cutoffs are now positions and the slices use them as such.
    head_end = int(n * HEAD_FRACTION)
    mid_end = int(n * (HEAD_FRACTION + MID_FRACTION))

    head = set(sorted_items[:head_end])
    mid = set(sorted_items[head_end:mid_end])
    tail = set(sorted_items[mid_end:])

    zero_items = all_item_ids - set(item_counts.keys())
    tail = tail | zero_items

    assert head.isdisjoint(mid) and head.isdisjoint(tail) and mid.isdisjoint(tail)
    assert head | mid | tail == all_item_ids

    buckets = {"head": head, "mid": mid, "tail": tail}
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(buckets, f)

    return buckets


def run_longtail_evaluation(
    model: torch.nn.Module,
    moe_layer: Optional[torch.nn.Module],
    config: Config,
    model_name: str,
) -> Dict:
    device = get_device(config)
    processed = config.processed_data_dir

    buckets_path = os.path.join(processed, "popularity_buckets.pkl")
    if os.path.exists(buckets_path):
        with open(buckets_path, "rb") as f:
            buckets = pickle.load(f)
    else:
        with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
            train_history = pickle.load(f)
        all_items = set(range(1, config.num_items + 1))
        buckets = compute_popularity_buckets(train_history, all_items, buckets_path)

    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)
    neg_pool = np.load(os.path.join(processed, "global_negative_pool.npy"))

    with open(os.path.join(processed, "test_examples.pkl"), "rb") as f:
        test_examples = pickle.load(f)

    # REV-093: score against the FIXED stored candidate sets. This used to rebuild them
    # per call with `filter_negatives` and no `rng=`, so every example's negatives were
    # a deterministic function of its TARGET ITEM -- a head item appearing thousands of
    # times in the test split met the same 100 negatives every time, correlating the
    # per-bucket errors in the one study whose entire point is comparing across
    # popularity buckets, and putting these numbers on a different candidate protocol
    # from `test_metrics_{model}.json`.
    cand_matrix = load_or_build_eval_candidates(
        os.path.join(processed, "test_candidates.npy"),
        test_examples, neg_pool, eval_user_history, config,
        expected_rows=len(test_examples["targets"]),
    )

    # REV-094: the backbone is pretrained with metadata, the best checkpoint is selected
    # on a validation pass that supplies it, and `--stage evaluate` supplies it. This
    # path supplied none, so the long-tail study ran the model with its metadata
    # projection contributing nothing -- on the items metadata is expected to help most.
    metadata_matrix = load_metadata_matrix(processed, config)

    bucket_names = ["head", "mid", "tail"]
    results: Dict[str, Dict[str, List[torch.Tensor]]] = {b: {} for b in bucket_names}

    n = len(test_examples["targets"])
    bs = max(1, config.eval_batch_size)

    # REV-063: one forward per example cost ~40 minutes over the test split. Scoring is
    # batched; candidate rows are sliced straight out of the stored matrix (REV-093).
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        seq_rows, cand_rows, bucket_of = [], [], []

        for i in range(start, stop):
            target = test_examples["targets"][i]
            seq = test_examples["sequences"][i]
            cand_rows.append(cand_matrix[i])
            seq_rows.append(seq if isinstance(seq, torch.Tensor) else torch.as_tensor(seq, dtype=torch.long))
            bucket_of.append(
                "head" if target in buckets["head"]
                else "mid" if target in buckets["mid"] else "tail"
            )

        seqs = torch.stack([r.long() for r in seq_rows])
        cands = torch.as_tensor(np.asarray(cand_rows), dtype=torch.long)

        batch_meta = None if metadata_matrix is None else metadata_matrix[seqs]
        scores = score_candidates_batch(
            model, seqs, cands, config, moe_layer=moe_layer, metadata=batch_meta,
        )
        per_metric = compute_metrics_batch(scores, target_idx=0, k=config.eval_k)

        bucket_arr = np.asarray(bucket_of)
        for bucket in bucket_names:
            sel = torch.from_numpy(bucket_arr == bucket)
            if not bool(sel.any()):
                continue
            for key, val in per_metric.items():
                results[bucket].setdefault(key, []).append(val.detach().cpu()[sel])

    report = {}
    for bucket in bucket_names:
        stacked = {k: torch.cat(v) for k, v in results[bucket].items()}
        report[bucket] = aggregate_metrics(stacked) if stacked else {}
    _save_report(config, model_name, report, buckets)
    _plot_longtail(config, model_name, report)
    return report


def _save_report(
    config: Config,
    model_name: str,
    report: Dict,
    buckets: Optional[Dict[str, Set[int]]] = None,
) -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    mn = model_name.replace("-", "_")
    path = os.path.join(config.log_dir, f"longtail_{mn}.json")

    serializable = {}
    for k, v in report.items():
        serializable[k] = {mk: {sk: float(sv) for sk, sv in mv.items()} for mk, mv in v.items()}

    # REV-129 asks a verifier to confirm, against this file, that the report carries the
    # bucket populations and that the buckets partition the catalogue. Only the per-bucket
    # metrics reached the artifact, so both had to be re-derived from the pickle by hand.
    # Recorded here so the numbers are self-describing: a tail NDCG means nothing without
    # knowing whether "tail" is 10 items or 14,000.
    # NOTE for the reader (and for the spec owner): `head_fraction_of_catalogue` is
    # deliberately reported ALONGSIDE the nominal 0.20/0.20/0.60. They differ because the
    # 20/20/60 split is applied to items that HAVE training interactions, after which
    # zero-training items are appended to tail per FR-5's second clause. See the note in
    # `specs/evaluation-diagnostics/tasks.md` T3.3.
    if buckets is not None:
        populations = {b: len(items) for b, items in buckets.items()}
        total = sum(populations.values())
        union = set().union(*buckets.values())
        serializable["bucket_populations"] = populations
        serializable["bucket_partition"] = {
            "total_items": total,
            "disjoint": all(
                buckets[a].isdisjoint(buckets[b])
                for a, b in (("head", "mid"), ("head", "tail"), ("mid", "tail"))
            ),
            "complete_union_size": len(union),
            "nominal_fractions": {
                "head": HEAD_FRACTION, "mid": MID_FRACTION, "tail": TAIL_FRACTION,
            },
            "realized_fractions_of_catalogue": {
                b: (n / total if total else 0.0) for b, n in populations.items()
            },
        }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _plot_longtail(config: Config, model_name: str, report: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    metrics = ["ndcg", "mrr", "recall", "precision"]
    bucket_names = ["head", "mid", "tail"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, bucket in enumerate(bucket_names):
        vals = [report.get(bucket, {}).get(m, {}).get("mean", 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=bucket.capitalize())

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Long-Tail Evaluation")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() + "@10" for m in metrics])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.plot_dir, f"longtail_{mn}.png"), dpi=config.plot_dpi)
    plt.close(fig)
