import hashlib
import json
import logging
import os
import pickle
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from models.transformer import TransformerModel
from models.moe import MoELayer
from models.lora import (
    inject_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    count_lora_parameters,
)
from training.losses import LogitScale, SampledSoftmaxLoss
from training.pretrain import build_model, load_checkpoint
from evaluation.metrics import FullRanking, evaluate_model, load_full_ranking
from data.sequence_generator import (
    _left_pad, filter_negatives, load_metadata_matrix, load_examples,
)
from utils import (
    autocast_context, get_device, resolve_dtype, make_grad_scaler, artifact_source_revision,
    atomic_write_json,
)

logger = logging.getLogger(__name__)


def compute_frozen_checksum(model: TransformerModel, moe: MoELayer) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if "lora_" not in name:
            h.update(p.detach().cpu().float().numpy().tobytes())
    for name, p in moe.named_parameters():
        h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def _get_trainable_backbone_count(model: TransformerModel) -> int:
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" not in name:
            count += p.numel()
    return count


def build_local_training_examples(
    user_id: int,
    config: Config,
    train_user_history: Dict[int, List[int]],
    val_examples: Optional[dict] = None,
    user_val_map: Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Per-user (sequence, target) pairs drawn from `config.local_train_source`.

    REV-003/REV-004: local adaptation used to take its training pairs from the user's
    TEST examples -- in the personalization case, from the exact examples the reported
    gain was then measured on, so `gain = ndcg_with_lora - ndcg_without_lora` was
    inflated by memorisation and grew with `lora_rank` and `local_epochs` for reasons
    unrelated to personalization quality. `local_train_source` existed, validated, and
    was read nowhere; it is the switch this function implements.

    The test split is never a source. `"train"` builds one (prefix, next item) example per
    training interaction; `"val"` uses their validation examples. Either way the most
    recent `config.local_train_max_examples` are kept, since recency is what a
    personalization adapter should be fit to.
    """
    cap = max(1, config.local_train_max_examples)

    if config.local_train_source == "val":
        if val_examples is None or user_val_map is None:
            raise ValueError(
                "local_train_source='val' needs the validation examples; pass "
                "`val_examples` and `user_val_map`."
            )
        indices = user_val_map.get(user_id, [])[-cap:]
        seqs = [val_examples["sequences"][i] for i in indices]
        tgts = [int(val_examples["targets"][i]) for i in indices]
    else:
        items = list(train_user_history.get(user_id, []))
        if len(items) < 2:
            return [], []
        first = max(1, len(items) - cap)
        seqs = [_left_pad(items[:i], config.max_seq_length) for i in range(first, len(items))]
        tgts = [int(items[i]) for i in range(first, len(items))]

    tensors = [
        s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long)
        for s in seqs
    ]
    return tensors, tgts


def train_user_lora(
    model: TransformerModel,
    moe: MoELayer,
    user_sequences: List[torch.Tensor],
    user_targets: List[int],
    neg_pool: np.ndarray,
    user_history_set: Set[int],
    config: Config,
    device: torch.device,
    metadata_matrix: Optional[torch.Tensor] = None,
    rng: Optional[np.random.RandomState] = None,
) -> None:
    assert _get_trainable_backbone_count(model) == 0, "Backbone params must not be trainable"

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=config.lora_lr, weight_decay=config.lora_weight_decay)
    # fp16 mixed precision only; with it disabled every scaler call is a pass-through.
    scaler = make_grad_scaler(config)
    # REV-061: the positional argument here was the negative COUNT binding to
    # `normalize_embeddings`, a boolean flag. It happened to behave correctly only
    # because 100 is truthy and `normalize_embeddings` defaults to True; with
    # `normalize_embeddings=False` the adapter would have been trained under a different
    # scoring geometry than `score_candidates` evaluates it in.
    # The adapter trains on the backbone's next-item logit scale, fixed at
    # 1/logit_temperature (a learned backbone scale is not carried over; it is frozen
    # with the backbone in spirit, and the init is its reference value). Identity at 1.0.
    loss_fn = SampledSoftmaxLoss(
        normalize_embeddings=config.normalize_embeddings,
        logit_scale=LogitScale(config.logit_temperature),
    )
    mask_id = config.num_items + 1

    for epoch in range(config.local_epochs):
        model.train()
        for seq, target in zip(user_sequences, user_targets):
            # REV-111: without an explicit `rng`, `filter_negatives` falls back to a
            # RandomState seeded from the TARGET ITEM, so the adapter met the identical
            # 100 negatives for a given target on every one of `local_epochs` passes --
            # the sampled-softmax signal on passes 2 and 3 was a duplicate of pass 1 --
            # and every user sharing that target trained against the same set. A
            # run-scoped RandomState varies the draw per call while keeping the run
            # reproducible from `config.seed`.
            neg_ids_np = filter_negatives(
                neg_pool, target, user_history_set,
                config.num_sampled_negatives, padding_id=0, mask_id=mask_id,
                rng=rng,
            )
            neg_ids = torch.tensor(neg_ids_np, dtype=torch.long, device=device)
            target_t = torch.tensor([target], dtype=torch.long, device=device)
            seq_input = seq.unsqueeze(0).to(device)

            # REV-062b: the backbone is pretrained with `use_metadata=True` and selected
            # on a validation pass that supplies metadata. Training the adapter on the
            # metadata-free path put personalization on an off-distribution serving path
            # whose absolute NDCG was not comparable to the test metrics reported
            # elsewhere -- which the federated convergence plot then mixed on one axis.
            meta_input = None
            if metadata_matrix is not None:
                meta_input = metadata_matrix[seq_input.to(metadata_matrix.device)].to(device)

            # specs/eval-integrity FR-3.2: forward, MoE and loss own an autocast region
            # (cache off, so the adapters' in-place updates are always seen); backward
            # and the scaler steps run outside it.
            with autocast_context(config, device_type=next(model.parameters()).device.type):
                all_hidden = model(seq_input, metadata=meta_input, return_all_hidden=True)
                pooled = all_hidden[:, -1, :]

                moe.clear_cache()
                Z_next, _ = moe(all_hidden, pooled, 0)
                user_emb = Z_next[:, -1, :]

                loss = loss_fn(user_emb, target_t, neg_ids.unsqueeze(0), model.item_emb_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()


def evaluate_user(
    model: TransformerModel,
    moe: MoELayer,
    user_sequences: List[torch.Tensor],
    user_targets: List[int],
    candidate_sets: FullRanking,
    config: Config,
    metadata_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """REV-062b/REV-063: batched, and on the same metadata-carrying serving path as
    validation and test evaluation."""
    model.eval()
    if not user_sequences:
        return {}
    with torch.no_grad():
        return evaluate_model(
            model, user_sequences, candidate_sets, config,
            moe_layer=moe, metadata_matrix=metadata_matrix,
        )


def run_local_personalization(
    config: Config,
    model_name: str,
    backbone_path: Optional[str] = None,
) -> Dict:
    config.apply_model_config(model_name)
    config.validate()
    device = get_device(config)

    processed = config.processed_data_dir
    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]

    config.vocab_size = config.num_items + 2
    # REV-058/REV-072: the aligned (vocab, F) matrix -- pad row 0, MASK_ID last row.
    metadata_matrix = load_metadata_matrix(processed, config)
    if metadata_matrix is not None:
        config.feature_dim = metadata_matrix.shape[-1]

    if backbone_path is None:
        backbone_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(backbone_path):
        logger.warning(f"Checkpoint not found: {backbone_path}. Skipping personalization.")
        return {}

    neg_pool = np.load(os.path.join(processed, "global_negative_pool.npy"))

    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)

    cold_items_path = os.path.join(processed, "cold_items.pkl")
    cold_items = set()
    if os.path.exists(cold_items_path):
        with open(cold_items_path, "rb") as f:
            cold_items = set(pickle.load(f))

    # specs/ablation-run FR-7: splits are stored wider than the model; narrow on read.
    test_examples = load_examples(
        os.path.join(processed, "test_examples.pkl"), config.max_seq_length,
    )

    # REV-003: the local training source. The test split -- which the gain is measured
    # on -- is never one of the options; `Config.validate()` restricts the flag to
    # "train" and "val".
    val_examples = None
    user_val_map: Optional[Dict[int, List[int]]] = None
    if config.local_train_source == "val":
        val_examples = load_examples(
            os.path.join(processed, "val_examples.pkl"), config.max_seq_length,
        )
        user_val_map = {}
        for i, uid in enumerate(val_examples["user_ids"]):
            user_val_map.setdefault(uid, []).append(i)

    # REV-111: one run-scoped RandomState for every negative draw in this stage, so the
    # draws vary per call and the whole run still reproduces from `config.seed`.
    neg_rng = np.random.RandomState(config.seed)

    # REV-111: score the SAME rows every other stage scores -- each test target ranked
    # against the full catalogue minus the user's history. `user_test_map` indexes
    # `test_examples`, so a test-example index addresses its ranking row directly.
    test_ranking = load_full_ranking(
        processed, "test", test_examples["targets"], test_examples["user_ids"],
    )

    all_user_ids = sorted(set(test_examples["user_ids"]))
    rng = random.Random(config.seed)
    n_sample = min(config.personalization_eval_users, len(all_user_ids))
    sampled_users = rng.sample(all_user_ids, n_sample)

    user_test_map: Dict[int, List[int]] = {}
    for i, uid in enumerate(test_examples["user_ids"]):
        user_test_map.setdefault(uid, []).append(i)

    # REV-024b: the model was rebuilt and `best.pt` re-deserialized for EVERY user --
    # 1000 full builds and multi-hundred-MB checkpoint loads. The backbone is frozen
    # (asserted per user below), so it is built ONCE and only the LoRA adapters are
    # reset between users. `lora_B` is zero-initialized, so a freshly-reset adapter is
    # an exact identity: the "without LoRA" measurement is unchanged.
    model, moe = build_model(config)
    backbone_ckpt = load_checkpoint(backbone_path, model, moe)
    # AC-09: the artifacts below are derived from THIS checkpoint, so they echo ITS
    # source_revision -- not the revision of the code currently running, which may differ.
    source_revision = artifact_source_revision(backbone_ckpt)
    frozen_cksum = compute_frozen_checksum(model, moe)
    inject_lora(model, config.lora_rank)
    pristine_lora = {k: v.detach().clone() for k, v in get_lora_state_dict(model).items()}

    results = []
    num_users_skipped = 0

    for user_id in sampled_users:
        test_indices = user_test_map.get(user_id, [])
        if not test_indices:
            num_users_skipped += 1
            continue

        test_seqs = [test_examples["sequences"][i] for i in test_indices]
        test_targets = [test_examples["targets"][i] for i in test_indices]
        history_set = set(train_user_history.get(user_id, []))

        candidate_sets = test_ranking.select(test_indices)

        test_seq_tensors = [torch.tensor(s, dtype=torch.long) if not isinstance(s, torch.Tensor) else s for s in test_seqs]

        # Reset the adapters to their zero-B initialization: identity, i.e. the
        # un-personalized backbone.
        load_lora_state_dict(model, pristine_lora)

        metrics_without = evaluate_user(
            model, moe, test_seq_tensors, test_targets, candidate_sets, config,
            metadata_matrix=metadata_matrix,
        )

        # REV-003: the adapter used to be fit to the first half of the user's TEST
        # examples and then scored on ALL of them -- including the half it had just
        # memorised. `gain` is the headline RQ4 number and feeds Ablation E, Ablation A
        # arm 6, the LoRA-rank plot and the federated baseline, so the inflation
        # propagated everywhere. Training pairs now come from the user's own TRAIN (or
        # VAL) interactions; the test examples are scored, never fitted.
        train_seq_tensors, train_tgts = build_local_training_examples(
            user_id, config, train_user_history, val_examples, user_val_map,
        )
        if not train_seq_tensors:
            # Nothing to adapt on -- report the user as skipped rather than falling back
            # to their evaluation examples, which is the defect this replaces.
            num_users_skipped += 1
            continue

        train_user_lora(
            model, moe, train_seq_tensors, train_tgts, neg_pool, history_set,
            config, device, metadata_matrix=metadata_matrix, rng=neg_rng,
        )

        metrics_with = evaluate_user(
            model, moe, test_seq_tensors, test_targets, candidate_sets, config,
            metadata_matrix=metadata_matrix,
        )

        assert compute_frozen_checksum(model, moe) == frozen_cksum, "Frozen params mutated during personalization"

        ndcg_with = float(metrics_with.get("ndcg", {}).get("mean", 0.0))
        ndcg_without = float(metrics_without.get("ndcg", {}).get("mean", 0.0))
        gain = ndcg_with - ndcg_without

        is_cold = bool(any(t in cold_items for t in test_targets))

        # Plain Python types: `user_id` comes from the numpy `test_examples["user_ids"]`
        # (int32), which json cannot encode -- it discarded every LoRA ablation arm's
        # record at write time, after the arm had fully trained.
        results.append({
            "user_id": int(user_id),
            "ndcg_with_lora": ndcg_with,
            "ndcg_without_lora": ndcg_without,
            "gain": gain,
            "is_cold": is_cold,
        })

    report = _build_report(results, num_users_skipped)
    _save_report(config, model_name, report, source_revision=source_revision)
    _plot_personalization(config, model_name, results)

    return report


def _build_report(results: List[Dict], num_users_skipped: int) -> Dict:
    # REV-113: the emitted key is `num_users_skipped`, the name FR-4 gives it literally.
    # It was `num_skipped`, which `personalization_gain_{model}.json` is a declared
    # artifact of, so the rename keeps the file readable by anything written against the
    # requirement rather than against the implementation.
    if not results:
        return {"num_users_evaluated": 0, "num_users_skipped": num_users_skipped}

    with_vals = [r["ndcg_with_lora"] for r in results]
    without_vals = [r["ndcg_without_lora"] for r in results]
    gains = [r["gain"] for r in results]

    return {
        "num_users_evaluated": len(results),
        "num_users_skipped": num_users_skipped,
        "with_lora": {"mean": float(np.mean(with_vals)), "std": float(np.std(with_vals))},
        "without_lora": {
            "mean": float(np.mean(without_vals)), "std": float(np.std(without_vals)),
        },
        "gain": {
            "mean": float(np.mean(gains)), "std": float(np.std(gains)),
            "min": float(np.min(gains)), "max": float(np.max(gains)),
        },
        "per_user_gains": [{"user_id": r["user_id"], "gain": r["gain"], "is_cold": r["is_cold"]} for r in results],
        # REV-065 / FR-10: the paired NDCG values, so Ablation E can overlay an arm's
        # users as a third scatter series. `per_user_gains` carries only the difference.
        "per_user_scores": [
            {
                "user_id": r["user_id"],
                "ndcg_without_lora": r["ndcg_without_lora"],
                "ndcg_with_lora": r["ndcg_with_lora"],
            }
            for r in results
        ],
    }


def _save_report(
    config: Config,
    model_name: str,
    report: Dict,
    source_revision: Optional[Dict] = None,
) -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    path = os.path.join(config.log_dir, f"personalization_gain_{model_name.replace('-', '_')}.json")

    payload = dict(report)
    # AC-09: echo the source_revision of the checkpoint these numbers came from.
    if source_revision is not None:
        payload["source_revision"] = source_revision

    # The shared writer, not a local encoder: the one this replaced mapped every numpy
    # integer to float, so `user_id` was written as e.g. 123.0.
    atomic_write_json(path, payload)


def _plot_personalization(config: Config, model_name: str, results: List[Dict]) -> None:
    if not results:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    gains = [r["gain"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gains, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", label="zero")
    ax.axvline(x=np.mean(gains), color="green", linestyle="--", label=f"mean={np.mean(gains):.4f}")
    ax.set_xlabel("NDCG@10 Gain")
    ax.set_ylabel("Count")
    ax.set_title("Personalization Gain Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.plot_dir, f"personalization_gain_histogram_{mn}.png"), dpi=config.plot_dpi)
    plt.close(fig)

    plot_personalization_scatter(
        config, model_name, [{"label": "with LoRA", "results": results}],
        filename=f"global_vs_personalized_{mn}.png",
    )


def plot_personalization_scatter(
    config: Config,
    model_name: str,
    series: List[Dict],
    filename: str,
    title: str = "Global vs Personalized",
) -> None:
    """Scatter global vs personalized NDCG per user, for one or more series.

    REV-065 / FR-10: this used to draw exactly one series inline, so Ablation E -- whose
    entire purpose is comparing global-only, global+LoRA and global+LoRA+ARV -- had no
    way to overlay its third arm. Each entry of `series` is
    `{"label": str, "results": [{"ndcg_without_lora", "ndcg_with_lora"}, ...]}`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawable = [s for s in series if s.get("results")]
    if not drawable:
        return

    os.makedirs(config.plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim = 0.0
    markers = ["o", "^", "s", "d"]
    for i, entry in enumerate(drawable):
        without = [r["ndcg_without_lora"] for r in entry["results"]]
        with_lora = [r["ndcg_with_lora"] for r in entry["results"]]
        ax.scatter(
            without, with_lora, alpha=0.4, s=10,
            marker=markers[i % len(markers)], label=entry["label"],
        )
        lim = max([lim] + without + with_lora)

    lim = max(lim, 1.0)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"NDCG@{config.eval_k} (Without LoRA)")
    ax.set_ylabel(f"NDCG@{config.eval_k} (With LoRA)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.plot_dir, filename), dpi=config.plot_dpi)
    plt.close(fig)
