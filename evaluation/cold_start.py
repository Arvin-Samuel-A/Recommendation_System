import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from configs.config import Config
from evaluation.metrics import (
    EVAL_PROTOCOL, aggregate_metrics, catalogue_ranking_scores, compute_metrics_batch,
    item_popularity, load_full_ranking, popularity_ranking_scores, score_catalogue_batch,
)
from data.sequence_generator import align_metadata_matrix, load_examples
from utils import get_device

logger = logging.getLogger(__name__)


def run_cold_start_evaluation(
    model: torch.nn.Module,
    moe_layer: Optional[torch.nn.Module],
    config: Config,
    model_name: str,
    source_revision: Optional[Dict] = None,
) -> Dict:
    device = get_device(config)
    processed = config.processed_data_dir

    cold_path = os.path.join(processed, "cold_items.pkl")
    warm_path = os.path.join(processed, "warm_items.pkl")
    with open(cold_path, "rb") as f:
        cold_item_set = set(pickle.load(f))
    with open(warm_path, "rb") as f:
        warm_item_set = set(pickle.load(f))

    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)

    # REV-015: T2.1 requires "Assert cold items disjoint from training", and
    # steering/testing.md lists this among the assertions that must be active during all
    # training/evaluation. A leaked cold item would silently inflate cold-start metrics --
    # exactly what AC-03 exists to catch.
    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    assert train_user_history is not eval_user_history, (
        "train_user_history and eval_user_history must be distinct objects"
    )
    train_interacted_items: Set[int] = set()
    for items in train_user_history.values():
        train_interacted_items.update(int(i) for i in items)
    leaked = cold_item_set & train_interacted_items
    assert not leaked, (
        f"{len(leaked)} cold items appear in training interactions (e.g. "
        f"{sorted(leaked)[:10]}). The cold-start holdout is not disjoint from training; "
        f"re-run --stage preprocess."
    )

    # specs/ablation-run FR-7: splits are stored wider than the model; narrow on read.
    test_examples = load_examples(
        os.path.join(processed, "test_examples.pkl"), config.max_seq_length,
    )

    # REV-001: the metadata ON/OFF arms were the same call -- neither passed `metadata=`
    # and nothing toggled it -- so the two score tensors were identical by construction
    # and FR-4's central claim was reported from a comparison that could not differ.
    metadata_matrix = None
    meta_path = os.path.join(processed, "item_metadata.npy")
    if os.path.exists(meta_path):
        metadata_matrix = np.load(meta_path)
    if metadata_matrix is None:
        raise RuntimeError(
            f"{meta_path} not found. The cold-start metadata ON/OFF comparison (FR-4) "
            f"cannot be run without item metadata; re-run --stage preprocess."
        )
    if getattr(model, "metadata_projection", None) is None:
        raise RuntimeError(
            "The loaded model has no metadata projection (trained with "
            "use_metadata=False), so the metadata ON arm is not meaningful. Evaluate a "
            "model trained with use_metadata=True."
        )

    # REV-093: the same rows as `test_metrics_{model}.json` -- each test target ranked
    # against the full catalogue minus the user's history.
    test_ranking = load_full_ranking(
        processed, "test", test_examples["targets"], test_examples["user_ids"],
    )

    # REV-072: the matrix spans the full vocabulary (pad row 0, MASK_ID row last), so a
    # gather needs no clamp -- clamping used to give MASK_ID the last real item's genome.
    meta_matrix = align_metadata_matrix(metadata_matrix, config.vocab_size)

    results: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = {
        "cold": {"meta_on": {}, "meta_off": {}, "popularity": {}},
        "warm": {"meta_on": {}, "meta_off": {}, "popularity": {}},
    }
    # specs/eval-integrity FR-7.2: the popularity-only floor per bucket, on the same rows
    # (training-interaction counts; cold items all score 0, tie-averaged).
    popularity = item_popularity(train_user_history, config.vocab_size)

    n = len(test_examples["targets"])
    bs = max(1, config.eval_batch_size)

    # REV-063: this scored ONE example per forward, twice over (ON and OFF), so the
    # 1,059,494-example test split cost ~80 minutes. The two arms are now one batched
    # pass each; the arithmetic per example is unchanged.
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        seqs = torch.as_tensor(np.asarray(test_examples["sequences"][start:stop])).long()
        rows = test_ranking.rows(start, stop)
        is_cold = torch.tensor(
            [int(t) in cold_item_set for t in test_examples["targets"][start:stop]],
            dtype=torch.bool,
        )

        # ON: the sequence's item metadata is supplied, as at training time.
        # OFF: the same model and the same rows, with no metadata -- the projection is
        # simply skipped, which is the metadata-free serving path.
        arms = {
            "meta_on": meta_matrix[seqs],
            "meta_off": None,
        }
        for arm, meta in list(arms.items()) + [("popularity", None)]:
            if arm == "popularity":
                scores = popularity_ranking_scores(rows, popularity)
            else:
                scores = catalogue_ranking_scores(
                    score_catalogue_batch(
                        model, seqs, config, moe_layer=moe_layer, metadata=meta,
                    ),
                    rows,
                )
            per_metric = compute_metrics_batch(scores, target_idx=0, k=config.eval_k)
            for bucket, sel in (("cold", is_cold), ("warm", ~is_cold)):
                if not bool(sel.any()):
                    continue
                for key, val in per_metric.items():
                    results[bucket][arm].setdefault(key, []).append(val.detach().cpu()[sel])

    # The cold set is data-determined under the default "unseen_in_training" strategy and
    # can be EMPTY (5-core Amazon data under leave-one-out, where every item usually has
    # a training interaction). An empty bucket is reported as `{}` metrics plus these
    # counts -- never NaN, never a crash -- so study H assembles either way.
    is_cold_target = [int(t) in cold_item_set for t in test_examples["targets"]]
    cold_set = {
        "num_cold_items": len(cold_item_set),
        "num_warm_items": len(warm_item_set),
        "num_cold_test_rows": int(sum(is_cold_target)),
        "num_warm_test_rows": int(n - sum(is_cold_target)),
    }
    logger.info(
        "Cold-start evaluation: %d cold / %d warm items; %d cold / %d warm test rows",
        cold_set["num_cold_items"], cold_set["num_warm_items"],
        cold_set["num_cold_test_rows"], cold_set["num_warm_test_rows"],
    )
    if cold_set["num_cold_test_rows"] == 0:
        logger.warning(
            "Cold-start evaluation: no test target is a cold item (%d cold items) -- the "
            "cold bucket is reported empty", cold_set["num_cold_items"],
        )

    report = {}
    for bucket in ["cold", "warm"]:
        for meta in ["meta_on", "meta_off"]:
            key = f"{bucket}_{meta}"
            stacked = {k: torch.cat(v) for k, v in results[bucket][meta].items()}
            report[key] = aggregate_metrics(stacked) if stacked else {}
        stacked = {k: torch.cat(v) for k, v in results[bucket]["popularity"].items()}
        report[f"{bucket}_popularity"] = aggregate_metrics(stacked) if stacked else {}

    _save_report(config, model_name, report, source_revision=source_revision,
                 cold_set=cold_set)
    _plot_cold_start(config, model_name, report)
    report["cold_set"] = cold_set
    return report


def _save_report(
    config: Config,
    model_name: str,
    report: Dict,
    source_revision: Optional[Dict] = None,
    cold_set: Optional[Dict] = None,
) -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    mn = model_name.replace("-", "_")
    path = os.path.join(config.log_dir, f"cold_start_{mn}.json")

    serializable = {}
    for k, v in report.items():
        serializable[k] = {mk: {sk: float(sv) for sk, sv in mv.items()} for mk, mv in v.items()}

    # AC-09: echo the source_revision of the checkpoint these numbers came from.
    if source_revision is not None:
        serializable["source_revision"] = source_revision
    # specs/eval-integrity FR-1.5: fp32 scores, tie-averaged metrics.
    serializable["eval_protocol"] = dict(EVAL_PROTOCOL)
    if cold_set is not None:
        serializable["cold_set"] = dict(cold_set)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _plot_cold_start(config: Config, model_name: str, report: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    metrics = ["ndcg", "mrr", "recall", "precision"]
    groups = ["cold_meta_on", "cold_meta_off", "warm_meta_on", "warm_meta_off"]
    labels = ["Cold+Meta", "Cold-Meta", "Warm+Meta", "Warm-Meta"]

    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (group, label) in enumerate(zip(groups, labels)):
        vals = [report.get(group, {}).get(m, {}).get("mean", 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Cold vs Warm Evaluation")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([m.upper() + "@10" for m in metrics])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.plot_dir, f"cold_vs_warm_{mn}.png"), dpi=config.plot_dpi)
    plt.close(fig)
