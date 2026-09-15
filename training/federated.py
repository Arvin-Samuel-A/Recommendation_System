import hashlib
import json
import logging
import os
import pickle
import random
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from configs.config import Config
from models.transformer import TransformerModel
from models.moe import MoELayer
from models.lora import (
    inject_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    count_lora_parameters,
)
from training.pretrain import build_model, load_checkpoint
from training.local_train import (
    compute_frozen_checksum,
    train_user_lora,
    evaluate_user,
    build_local_training_examples,
)
from evaluation.metrics import evaluate_model
from data.sequence_generator import (
    load_tensorised,
    build_dataloader,
    generate_eval_candidates,
    filter_negatives,
    load_metadata_matrix,
)
from utils import get_device

logger = logging.getLogger(__name__)


_MOE_KEY_PREFIXES = ("moe.router", "moe.experts", "moe.objective_embeddings", "moe.gates")


def assert_no_moe_keys(
    update: Dict[str, torch.Tensor],
    moe_layer: Optional[torch.nn.Module] = None,
    context: str = "LoRA update",
) -> None:
    """Reject any MoE parameter smuggled into a client's LoRA update.

    steering/testing.md lists MoE exclusion among the critical runtime assertions: the
    MoE layer is global and must never be communicated or federated-averaged.

    REV-026c: the previous inline check tested `k.startswith(_MOE_KEY_PREFIXES)` on a
    dict that `get_lora_state_dict` had already filtered to keys containing `lora_`, and
    `MoELayer` is never a submodule of `TransformerModel` -- so it was structurally
    unreachable and the test asserting it did not fire proved nothing. This also compares
    against the MoE's ACTUAL key space, so it fires if MoE state is ever merged in under
    any naming scheme.
    """
    offenders = {k for k in update if any(k.startswith(p) for p in _MOE_KEY_PREFIXES)}

    if moe_layer is not None:
        moe_keys = set(moe_layer.state_dict().keys())
        offenders |= set(update) & moe_keys
        offenders |= {k for k in update if k.split("moe.", 1)[-1] in moe_keys and "moe." in k}

    if offenders:
        raise AssertionError(
            f"MoE parameters present in {context}: {sorted(offenders)[:10]}. The MoE "
            f"layer is global and must never be federated-averaged or communicated."
        )


def weighted_fedavg(
    client_updates: List[Dict[str, torch.Tensor]],
    client_counts: List[int],
    use_weighted: bool = True,
) -> Dict[str, torch.Tensor]:
    if not client_updates:
        return {}

    keys = client_updates[0].keys()

    if use_weighted:
        total = sum(client_counts)
        weights = [c / total for c in client_counts]
    else:
        n = len(client_updates)
        weights = [1.0 / n] * n

    result = {}
    for key in keys:
        result[key] = sum(w * update[key].float() for w, update in zip(weights, client_updates)).to(
            client_updates[0][key].dtype
        )
    return result


def run_federated(
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
        logger.warning(f"Checkpoint not found: {backbone_path}. Skipping federated.")
        return {}

    neg_pool = np.load(os.path.join(processed, "global_negative_pool.npy"))
    mask_id = config.num_items + 1

    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)

    val_dataset = load_tensorised(os.path.join(processed, "val_tensorised.pt"), metadata_matrix)
    val_examples = {
        "targets": val_dataset.targets.numpy().tolist(),
        "user_ids": val_dataset.user_ids.numpy().tolist(),
    }
    val_candidates = generate_eval_candidates(
        val_examples, neg_pool, eval_user_history,
        config.num_sampled_negatives, padding_id=0, mask_id=mask_id,
    )
    # REV-063: stacked tensors, batched by `evaluate_model`.
    val_sequences = val_dataset.sequences
    val_candidate_tensors = torch.as_tensor(np.asarray(val_candidates), dtype=torch.long)

    with open(os.path.join(processed, "test_examples.pkl"), "rb") as f:
        test_examples = pickle.load(f)
    # The client roster is the set of users with test examples -- who participates, not
    # what they train on. REV-004: their LOCAL TRAINING data comes from
    # `config.local_train_source`, never from the test split.
    all_user_ids = sorted(set(test_examples["user_ids"]))

    val_examples_raw = None
    user_val_map: Optional[Dict[int, List[int]]] = None
    if config.local_train_source == "val":
        with open(os.path.join(processed, "val_examples.pkl"), "rb") as f:
            val_examples_raw = pickle.load(f)
        user_val_map = {}
        for i, uid in enumerate(val_examples_raw["user_ids"]):
            user_val_map.setdefault(uid, []).append(i)

    model, moe = build_model(config)
    load_checkpoint(backbone_path, model, moe)
    inject_lora(model, config.lora_rank)
    global_lora_state = get_lora_state_dict(model)

    frozen_checksum = compute_frozen_checksum(model, moe)
    lora_param_count = count_lora_parameters(model)

    # REV-024a: one reusable client model, built once. Its frozen parameters are never
    # written (only LoRA adapters are trained), and that invariant is checked per round
    # against the same checksum the global model uses.
    client_model, client_moe = build_model(config)
    load_checkpoint(backbone_path, client_model, client_moe)
    inject_lora(client_model, config.lora_rank)
    assert compute_frozen_checksum(client_model, client_moe) == frozen_checksum, (
        "The reusable client model's frozen parameters differ from the global model's"
    )

    round_logs = []
    rng = random.Random(config.seed)

    for round_idx in range(config.federated_rounds):
        assert compute_frozen_checksum(model, moe) == frozen_checksum, \
            f"Frozen params mutated before round {round_idx}"

        clients = rng.sample(all_user_ids, min(config.clients_per_round, len(all_user_ids)))
        client_updates = []
        client_counts = []

        for client_id in clients:
            # REV-004: client updates used to be fitted to the first half of that
            # client's TEST examples. The federated model is scored on validation, so
            # this was not the direct train-on-eval overlap of REV-003 -- but test-split
            # interactions still entered the global LoRA state that every later round and
            # every downstream TEST evaluation inherits, contaminating the federated
            # accuracy and communication claims (and Ablation H6 through them).
            train_seq_tensors, train_tgts = build_local_training_examples(
                client_id, config, train_user_history, val_examples_raw, user_val_map,
            )
            if not train_seq_tensors:
                continue

            # REV-024a: this used to construct a fresh model and deserialize the
            # multi-hundred-MB `best.pt` for EVERY client on EVERY round -- 20 rounds x
            # 100 clients = 2000 full builds and checkpoint loads. The backbone is frozen
            # by construction (asserted every round), so one client model is built once
            # outside the loop and only its LoRA state is reset per client.
            load_lora_state_dict(client_model, global_lora_state)

            history_set = set(train_user_history.get(client_id, []))

            train_user_lora(
                client_model, client_moe, train_seq_tensors, train_tgts,
                neg_pool, history_set, config, device,
                metadata_matrix=metadata_matrix,
            )

            update = get_lora_state_dict(client_model)
            assert_no_moe_keys(update, client_moe, context=f"client {client_id}'s update")
            client_updates.append(update)
            client_counts.append(len(train_seq_tensors))

        if client_updates:
            global_lora_state = weighted_fedavg(
                client_updates, client_counts, config.use_weighted_fedavg,
            )

        load_lora_state_dict(model, global_lora_state)

        model.eval()
        moe.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(
                model, val_sequences, val_candidate_tensors, config,
                moe_layer=moe, metadata_matrix=val_dataset.metadata_matrix,
            )

        n_clients = len(client_updates)
        upload_kb = lora_param_count * 2 / 1024
        download_kb = upload_kb
        # REV-036: AC-07/FR-7 specify the multiplier as `clients_per_round`, the
        # CONFIGURED cost. `n_clients` counts only clients that actually contributed,
        # which understates it whenever a sampled client is skipped for having no
        # examples. Both are recorded so neither reading is lost.
        total_round_kb = (upload_kb + download_kb) * config.clients_per_round
        actual_round_kb = (upload_kb + download_kb) * n_clients

        val_ndcg = val_metrics.get("ndcg", {}).get("mean", 0.0)
        round_log = {
            "round": round_idx,
            "val_ndcg": val_ndcg,
            "val_mrr": val_metrics.get("mrr", {}).get("mean", 0.0),
            "num_clients": n_clients,
            "upload_kb": upload_kb,
            "download_kb": download_kb,
            "total_round_kb": total_round_kb,
            "actual_round_kb": actual_round_kb,
            "clients_per_round": config.clients_per_round,
        }
        round_logs.append(round_log)

        logger.info(f"Round {round_idx}: val_ndcg={val_ndcg:.4f} clients={n_clients} cost={total_round_kb:.1f}KB")

    assert compute_frozen_checksum(model, moe) == frozen_checksum, \
        "Frozen params mutated after federated simulation"

    mn = model_name.replace("-", "_")
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = os.path.join(config.log_dir, f"federated_rounds_{mn}.json")
    with open(log_path, "w") as f:
        json.dump(round_logs, f, indent=2)

    baseline_ndcg, baseline_label = _resolve_personalized_baseline(config, model_name)
    _plot_convergence(config, model_name, round_logs, baseline_ndcg, baseline_label)

    return {"round_logs": round_logs, "final_val_ndcg": round_logs[-1]["val_ndcg"] if round_logs else 0.0}


def _resolve_personalized_baseline(config: Config, model_name: str):
    """The non-federated personalized NDCG that FR-8 wants drawn as the reference line.

    REV-023: `_plot_convergence` drew `ndcgs[0]` -- the first federated round's own
    score -- labelled "Round 0 baseline", so the figure compared federated progress
    against itself rather than against the baseline the research question is about.

    Falls back to the global backbone's best val NDCG when personalization has not run,
    and returns `(None, None)` when neither is available so the line is simply omitted
    rather than silently wrong.
    """
    mn = model_name.replace("-", "_")

    gain_path = os.path.join(config.log_dir, f"personalization_gain_{mn}.json")
    if os.path.exists(gain_path):
        try:
            with open(gain_path) as f:
                report = json.load(f)
            val = report.get("with_lora", {}).get("mean")
            if val is not None:
                return float(val), "Non-federated personalized (LoRA)"
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Could not read %s for the baseline line: %s", gain_path, exc)

    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            val = ckpt.get("best_val_ndcg", ckpt.get("val_ndcg"))
            if val is not None:
                return float(val), "Global backbone (no personalization)"
        except (OSError, RuntimeError) as exc:
            logger.warning("Could not read %s for the baseline line: %s", ckpt_path, exc)

    logger.warning(
        "No non-federated baseline available for %s; the convergence plot will omit the "
        "reference line. Run --stage personalize first.", model_name,
    )
    return None, None


def _plot_convergence(
    config: Config,
    model_name: str,
    round_logs: List[Dict],
    baseline_ndcg: Optional[float] = None,
    baseline_label: Optional[str] = None,
) -> None:
    if not round_logs:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    rounds = [r["round"] for r in round_logs]
    ndcgs = [r["val_ndcg"] for r in round_logs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, ndcgs, "o-", label="Federated NDCG@10")
    if baseline_ndcg is not None:
        ax.axhline(
            y=baseline_ndcg, color="gray", linestyle="--", alpha=0.7,
            label=baseline_label or "Non-federated baseline",
        )
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Val NDCG@10")
    ax.set_title("Federated Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(config.plot_dir, f"federated_convergence_{mn}.png"), dpi=config.plot_dpi)
    plt.close(fig)
