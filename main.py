import argparse
import copy
import json
import logging
import os
import random
import sys
from dataclasses import asdict

import numpy as np
import torch

from configs.config import Config
from utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_STAGES = [
    "preprocess", "pretrain", "personalize", "federated",
    "evaluate", "cold_start", "longtail", "latency",
    "routing_diagnostics", "gradient_diagnostics",
    # REV-013 / REV-014: multi-seed aggregation (FR-15) and the LoRA-rank sweep that
    # feeds personalization_gain_vs_lora_rank.png had no entry points at all.
    "lora_rank_sweep", "multi_seed",
    "ablation", "plots", "all",
]

VALID_MODELS = ["t-small", "t-base", "t-large"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Recommendation System Pipeline")
    parser.add_argument("--stage", required=True, choices=VALID_STAGES, help="Pipeline stage to run")
    parser.add_argument("--model", default="t-base", choices=VALID_MODELS, help="Model variant")
    parser.add_argument("--epochs", type=int, default=None, help="Override pretrain_epochs/federated_rounds")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--moe_variant", default=None, choices=["none", "mmoe", "arv_mmoe"])
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--use_gradnorm", action="store_true", default=None)
    parser.add_argument("--no_gradnorm", action="store_true", default=False)
    parser.add_argument("--study", default="all", help="Ablation study letter (A-H) or 'all'")
    # REV-115: `run_ablation(train=...)` existed but had no CLI surface, so evaluating
    # already-trained arms meant editing source -- which conventions.md's "Config-Driven
    # Design" forbids ("No ablation requires commenting/uncommenting code or manually
    # editing source files"). An arm with no checkpoint records
    # `status: "skipped_no_checkpoint"` and the study continues.
    parser.add_argument(
        "--no_train", action="store_true", default=False,
        help="Ablations only: evaluate existing per-arm checkpoints instead of training "
             "each arm. Arms with no checkpoint are skipped, not failed.",
    )
    parser.add_argument("--seed", type=int, default=None)
    # Owner decision (2026-09-17): Phase 17's gate is a few-step run of `run_pretrain` on
    # REAL data, not a full epoch. `max_steps_per_epoch` already existed as a Config
    # field but had no CLI surface, so running the gate meant editing source -- which
    # conventions.md's "Config-Driven Design" forbids.
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Cap training steps per epoch (0 = uncapped). The Phase 17 few-step smoke "
             "gate uses this to exercise run_pretrain end to end on real data.",
    )
    # REV-131: the owner's 2026-09-17 smoke-gate decision reduced personalization-federated
    # AC-05/AC-06 to pipeline-integrity gates at `personalization_eval_users = 10` and
    # `clients_per_round = 10`, but none of the three parameters had a CLI surface -- so
    # running that gate meant editing `configs/config.py`, which conventions.md's
    # "Config-Driven Design" forbids and which would silently change the configured values
    # every later run inherits. This is the third instance of the same gap, after
    # `--no_train` (REV-115) and `--max_steps`.
    # All three default to None so the CONFIGURED values stay authoritative when the flags
    # are absent -- Phase 14 runs the configured values; the gate is a CLI-time reduction.
    parser.add_argument(
        "--eval_users", type=int, default=None,
        help="Override personalization_eval_users (number of users scored by "
             "--stage personalize).",
    )
    parser.add_argument(
        "--clients_per_round", type=int, default=None,
        help="Override clients_per_round for --stage federated.",
    )
    parser.add_argument(
        "--local_epochs", type=int, default=None,
        help="Override local_epochs for per-user LoRA training.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config: Config, args) -> None:
    overrides = []
    if args.epochs is not None:
        # REV-031: --epochs always overwrote `pretrain_epochs` and ADDITIONALLY set
        # `federated_rounds` for the federated stage, so a federated run mutated
        # pretrain_epochs in the saved manifest. T1.2 maps the flag by stage.
        if args.stage == "federated":
            overrides.append(("federated_rounds", config.federated_rounds, args.epochs))
            config.federated_rounds = args.epochs
        else:
            overrides.append(("pretrain_epochs", config.pretrain_epochs, args.epochs))
            config.pretrain_epochs = args.epochs
    if args.batch_size is not None:
        overrides.append(("batch_size", config.batch_size, args.batch_size))
        config.batch_size = args.batch_size
    if args.moe_variant is not None:
        overrides.append(("moe_variant", config.moe_variant, args.moe_variant))
        config.moe_variant = args.moe_variant
    if args.num_experts is not None:
        overrides.append(("num_experts", config.num_experts, args.num_experts))
        config.num_experts = args.num_experts
    if args.use_gradnorm is True:
        overrides.append(("use_gradnorm", config.use_gradnorm, True))
        config.use_gradnorm = True
    elif args.no_gradnorm:
        overrides.append(("use_gradnorm", config.use_gradnorm, False))
        config.use_gradnorm = False
    if args.seed is not None:
        overrides.append(("seed", config.seed, args.seed))
        config.seed = args.seed
    if getattr(args, "max_steps", None) is not None:
        overrides.append(
            ("max_steps_per_epoch", config.max_steps_per_epoch, args.max_steps)
        )
        config.max_steps_per_epoch = args.max_steps
    # REV-131: logged original -> new like every other override (code-quality.md).
    if getattr(args, "eval_users", None) is not None:
        overrides.append((
            "personalization_eval_users",
            config.personalization_eval_users,
            args.eval_users,
        ))
        config.personalization_eval_users = args.eval_users
    if getattr(args, "clients_per_round", None) is not None:
        overrides.append(
            ("clients_per_round", config.clients_per_round, args.clients_per_round)
        )
        config.clients_per_round = args.clients_per_round
    if getattr(args, "local_epochs", None) is not None:
        overrides.append(("local_epochs", config.local_epochs, args.local_epochs))
        config.local_epochs = args.local_epochs

    for name, old, new in overrides:
        logger.info(f"CLI override: {name} {old} -> {new}")


def setup_seed(config: Config) -> None:
    """Seed every random source for the run.

    REV-083: this used to reimplement `utils.set_seed` and dropped two of its
    behaviours -- `torch.cuda.manual_seed_all` (so on a multi-GPU host every CUDA
    generator beyond the default device stayed unseeded) and
    `torch.backends.cudnn.benchmark = False` (so cuDNN autotuning could still pick
    different algorithms between runs). `utils.set_seed` is the canonical helper named
    in BUILD_ORDER step 0.2 and conventions.md, so delegate rather than duplicate.
    """
    set_seed(config.seed)


def save_config_snapshot(config: Config) -> None:
    """Write the RESOLVED run manifest.

    REV-029: this ran before any stage called `apply_model_config`, so `run_config.json`
    recorded the default t-base dimensions regardless of `--model`, and always had
    num_items/vocab_size/feature_dim = 0. code-quality.md additionally requires the
    manifest to record device, dtype and the resolved attention backend -- the backend
    that ACTUALLY ran, which may be a fallback (see models/transformer.py).
    """
    import torch as _torch

    from utils import get_device, resolve_dtype

    os.makedirs(config.log_dir, exist_ok=True)
    path = os.path.join(config.log_dir, "run_config.json")

    manifest = {"config": asdict(config)}
    try:
        device = get_device(config)
        manifest["runtime"] = {
            "torch_version": _torch.__version__,
            "device": str(device),
            "dtype": str(resolve_dtype(config)),
            "attention_backend_requested": getattr(config, "attention_backend", None),
        }
        from models.transformer import AttentionBackend

        backend = AttentionBackend(config)
        manifest["runtime"]["attention_backend_resolved"] = backend.backend_name
        manifest["runtime"]["attention_backend_fallback_reason"] = backend.fallback_reason
    except Exception as exc:  # never let manifest-writing abort a run
        logger.warning("Could not fully resolve the runtime manifest: %s", exc)
        manifest.setdefault("runtime", {})["error"] = f"{type(exc).__name__}: {exc}"

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_preprocess(config, args):
    from data.preprocessing import run_preprocessing
    run_preprocessing(config)


def run_pretrain_stage(config, args):
    from training.pretrain import run_pretrain
    return run_pretrain(config, args.model)


def run_personalize_stage(config, args):
    from training.local_train import run_local_personalization
    return run_local_personalization(config, args.model)


def run_federated_stage(config, args):
    from training.federated import run_federated
    return run_federated(config, args.model)


def run_evaluate_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.metrics import evaluate_model
    from data.sequence_generator import (
        load_tensorised, generate_eval_candidates, load_eval_candidates,
        load_metadata_matrix,
    )
    from utils import get_device
    import pickle

    config.apply_model_config(args.model)
    device = get_device(config)
    processed = config.processed_data_dir

    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}. Run: python main.py --stage pretrain --model {args.model}")
        return

    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    model, moe = build_model(config)
    load_checkpoint(ckpt_path, model, moe)

    # REV-058/REV-072: one loader, so the (vocab, F) alignment applies uniformly.
    metadata_matrix = load_metadata_matrix(processed, config)
    if metadata_matrix is not None:
        config.feature_dim = metadata_matrix.shape[-1]

    test_dataset = load_tensorised(os.path.join(processed, "test_tensorised.pt"), metadata_matrix)
    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)

    # REV-041: prefer the candidate sets stored at preprocessing time so every model is
    # scored against structurally identical sets, not incidentally identical ones.
    cand_path = os.path.join(processed, "test_candidates.npy")
    if os.path.exists(cand_path):
        matrix = load_eval_candidates(cand_path)
        if matrix.shape[0] != len(test_dataset):
            raise RuntimeError(
                f"Pre-stored test candidate sets have {matrix.shape[0]} rows but the "
                f"test split has {len(test_dataset)} examples. Re-run --stage preprocess."
            )
        candidates = torch.as_tensor(matrix, dtype=torch.long)
    else:
        logger.warning(
            "%s not found -- regenerating test candidate sets. Re-run --stage preprocess "
            "so every model evaluates against the same stored sets.", cand_path,
        )
        neg_pool = np.load(os.path.join(processed, "global_negative_pool.npy"))
        test_examples = {
            "targets": test_dataset.targets.numpy().tolist(),
            "user_ids": test_dataset.user_ids.numpy().tolist(),
        }
        candidates = torch.as_tensor(
            np.asarray(generate_eval_candidates(
                test_examples, neg_pool, eval_user_history,
                config.num_sampled_negatives, padding_id=0,
                mask_id=config.num_items + 1, seed=config.seed,
            )), dtype=torch.long,
        )
    # REV-063: the stored `(N, S)` / `(N, C)` tensors are handed to `evaluate_model`
    # whole; it batches them. Exploding them into ~1M per-example device tensors was
    # the dominant allocation cost of the evaluate stage.
    test_sequences = test_dataset.sequences
    candidate_tensors = candidates

    # FR-2 / REV-064: the retrieval-then-rerank path. `use_ann` and `ann_candidates` were
    # declared in the config and read by nothing, so this requirement had no code behind
    # it at all. Off by default, so the sampled-negatives protocol remains the headline.
    ann_report = None
    if config.use_ann:
        from models.ann_retrieval import retrieve_candidates

        logger.info(
            "use_ann=True -- retrieving %d candidates per example and force-including "
            "the target.", config.ann_candidates,
        )
        ann_report = retrieve_candidates(
            model, test_sequences, test_dataset.targets, config,
            user_ids=test_dataset.user_ids.numpy().tolist(),
            user_history=eval_user_history,
            moe_layer=moe,
            metadata_matrix=test_dataset.metadata_matrix,
        )
        candidate_tensors = ann_report["candidates"]

    # REV-017: validation (and therefore checkpoint selection) scores WITH metadata; the
    # test stage used to score without it, so headline test numbers came from a
    # different serving path than the one the model was selected under.
    # REV-058: the `(vocab, F)` matrix is passed so each batch gathers its own block.
    model.eval()
    moe.eval()
    with torch.no_grad():
        metrics = evaluate_model(
            model, test_sequences, candidate_tensors, config,
            moe_layer=moe, metadata_matrix=test_dataset.metadata_matrix,
        )

    logger.info(f"Test metrics: {metrics}")
    os.makedirs(config.log_dir, exist_ok=True)
    payload = {k: {sk: float(sv) for sk, sv in v.items()} for k, v in metrics.items()}
    if ann_report is not None:
        # FR-2 requires the force-inclusion fraction to be reported alongside the metrics
        # it conditions -- a high rate means the retrieval stage is doing the work.
        payload["ann_retrieval"] = {
            "ann_candidates": ann_report["ann_candidates"],
            "force_inclusion_rate": ann_report["force_inclusion_rate"],
            "num_force_included": ann_report["num_force_included"],
            "num_examples": ann_report["num_examples"],
        }
    with open(os.path.join(config.log_dir, f"test_metrics_{args.model.replace('-', '_')}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def run_cold_start_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.cold_start import run_cold_start_evaluation
    from utils import get_device

    config.apply_model_config(args.model)
    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}. Run: python main.py --stage pretrain --model {args.model}")
        return

    with open(os.path.join(config.processed_data_dir, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    model, moe = build_model(config)
    load_checkpoint(ckpt_path, model, moe)
    run_cold_start_evaluation(model, moe, config, args.model)


def run_longtail_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.longtail import run_longtail_evaluation

    config.apply_model_config(args.model)
    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}. Run: python main.py --stage pretrain --model {args.model}")
        return

    with open(os.path.join(config.processed_data_dir, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    model, moe = build_model(config)
    load_checkpoint(ckpt_path, model, moe)
    run_longtail_evaluation(model, moe, config, args.model)


def run_latency_stage(config, args):
    from copy import deepcopy

    from training.pretrain import build_model, load_checkpoint
    from evaluation.flops import (
        benchmark_latency, count_parameters, save_efficiency_report,
        measure_routing_overhead, estimate_flops_per_inference,
    )
    from models.moe import MoELayer
    from utils import get_device, resolve_dtype

    config.apply_model_config(args.model)
    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}. Run: python main.py --stage pretrain --model {args.model}")
        return

    processed = config.processed_data_dir
    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    meta_path = os.path.join(processed, "item_metadata.npy")
    if config.use_metadata and os.path.exists(meta_path):
        config.feature_dim = np.load(meta_path).shape[-1]

    model, moe = build_model(config)
    load_checkpoint(ckpt_path, model, moe)
    latency = benchmark_latency(model, moe, config)
    params = count_parameters(model, moe)

    # REV-021a: `measure_routing_overhead` had no call site, so `routing_overhead` was
    # never written and Ablation C lost its routing-cost number. Time the same backbone
    # against a moe_variant="none" layer to isolate the routing cost.
    overhead = None
    if config.moe_variant != "none":
        none_config = deepcopy(config)
        none_config.moe_variant = "none"
        moe_none = MoELayer(none_config).to(
            device=get_device(config), dtype=resolve_dtype(config),
        )
        overhead = measure_routing_overhead(model, moe, moe_none, config)
        logger.info(
            "Routing overhead: %.3f ms (%.3f with MoE vs %.3f without)",
            overhead["overhead_ms"], overhead["full_mean_ms"], overhead["none_mean_ms"],
        )

    # REV-021b: FR-7 names flops_per_inference; accuracy_vs_flops.png has no other source.
    params["flops_per_inference"] = estimate_flops_per_inference(config, moe)

    save_efficiency_report(config, args.model, latency, params, overhead=overhead)
    logger.info(
        "Efficiency report written for %s: %.3f ms/rec, %d params, %.1f MB bf16",
        args.model, latency["ms_per_recommendation"],
        params["total_parameters"], params["model_size_bf16_mb"],
    )


def _load_trained_model(config, args):
    """Shared setup for the standalone diagnostics stages: stats, model, best.pt."""
    from training.pretrain import build_model, load_checkpoint, _build_loss_fns
    from data.sequence_generator import load_metadata_matrix
    from utils import get_device

    config.apply_model_config(args.model)
    processed = config.processed_data_dir

    ckpt_path = os.path.join(config.model_checkpoint_dir(), "best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(
            f"Checkpoint not found: {ckpt_path}. "
            f"Run: python main.py --stage pretrain --model {args.model}"
        )
        return None

    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    # REV-058/REV-072: one loader, so the (vocab, F) alignment applies uniformly.
    metadata_matrix = load_metadata_matrix(processed, config)
    if metadata_matrix is not None:
        config.feature_dim = metadata_matrix.shape[-1]

    device = get_device(config)
    model, moe = build_model(config)
    loss_fns = _build_loss_fns(config, device)
    load_checkpoint(ckpt_path, model, moe, loss_fns=loss_fns)
    return model, moe, loss_fns, metadata_matrix, device, processed


def run_routing_diagnostics_stage(config, args):
    """Standalone comprehensive routing pass over the validation set.

    REV-008: this stage used to be a single log line telling the user to run the pretrain
    stage -- but `run_pretrain` did not call `save_routing_diagnostics` either, so the
    routing report could never be produced by any path.
    """
    from training.pretrain import collect_validation_routing
    from evaluation.routing_diagnostics import save_routing_diagnostics
    from data.sequence_generator import load_tensorised, load_metadata_matrix

    loaded = _load_trained_model(config, args)
    if loaded is None:
        return
    model, moe, _, metadata_matrix, _, processed = loaded

    val_dataset = load_tensorised(
        os.path.join(processed, "val_tensorised.pt"), metadata_matrix,
    )
    per_objective = collect_validation_routing(model, moe, val_dataset, config)
    if not per_objective:
        logger.warning(
            "No routing snapshots collected -- moe_variant=%s produces no routing "
            "weights.", config.moe_variant,
        )
        return
    report = save_routing_diagnostics(per_objective, config, args.model)
    logger.info(
        "Routing diagnostics written for %s: overall_entropy_mean=%.4f",
        args.model, report["overall_entropy_mean"],
    )
    return report


def run_gradient_diagnostics_stage(config, args):
    """Standalone gradient conflict/dominance pass on fresh training batches.

    REV-009: `run_gradient_diagnostics` previously had no call site, and required a
    `neg_ids` key that `build_dataloader` does not produce. Negatives are now attached
    through the same helper the training loop uses.
    """
    import pickle

    from training.pretrain import attach_training_negatives
    from training.gradnorm import get_anchor_layer_params
    from evaluation.gradient_diagnostics import run_gradient_diagnostics
    from data.sequence_generator import (
        load_tensorised, build_dataloader, GlobalNegativePool, load_metadata_matrix,
    )
    from utils import resolve_dtype

    loaded = _load_trained_model(config, args)
    if loaded is None:
        return
    model, moe, loss_fns, metadata_matrix, device, processed = loaded

    train_dataset = load_tensorised(
        os.path.join(processed, "train_tensorised.pt"), metadata_matrix,
    )
    loader = build_dataloader(train_dataset, config, shuffle=True)

    neg_array = GlobalNegativePool.load(
        os.path.join(processed, "global_negative_pool.npy"), set(),
    ).pool
    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    history_arrays = {
        int(u): np.asarray(items, dtype=np.int64)
        for u, items in train_user_history.items()
    }
    neg_rng = np.random.RandomState(config.seed)

    def attach_negatives(batch):
        return attach_training_negatives(
            batch, neg_array, history_arrays, config, neg_rng, device,
        )

    metadata_matrix_t = (
        metadata_matrix.to(resolve_dtype(config)) if metadata_matrix is not None else None
    )

    report = run_gradient_diagnostics(
        model, moe, loader, loss_fns,
        get_anchor_layer_params(model, config), config, args.model,
        negative_sampler=attach_negatives,
        metadata_matrix=metadata_matrix_t,
        augment_rng=random.Random(config.seed),
    )
    if not report:
        logger.warning("No diagnostic batches processed; no report written.")
        return
    logger.info(
        "Gradient diagnostics written for %s over objectives %s",
        args.model, report.get("objective_names"),
    )
    return report


def run_ablation_stage(config, args):
    from training.ablation_runner import run_ablation
    # REV-115: `--no_train` reaches the `train` parameter that already existed here.
    return run_ablation(
        config, args.model, args.study, train=not getattr(args, "no_train", False),
    )


def run_plots_stage(config, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.federated import final_val_ndcg

    os.makedirs(config.plot_dir, exist_ok=True)
    logger.info("Generating core tradeoff plots from completed evaluation results...")

    results = {}
    for model_name in ["t_small", "t_base", "t_large"]:
        metrics_path = os.path.join(config.log_dir, f"test_metrics_{model_name}.json")
        efficiency_path = os.path.join(config.log_dir, f"efficiency_{model_name}.json")
        if os.path.exists(metrics_path) and os.path.exists(efficiency_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            with open(efficiency_path) as f:
                efficiency = json.load(f)
            results[model_name] = {**metrics, **efficiency}

    if not results:
        logger.warning("No completed evaluation results found. Run evaluate and latency stages first.")
        return

    models = list(results.keys())
    ndcgs = [results[m].get("ndcg", {}).get("mean", 0.0) for m in models]
    # REV-101(a): FR-14 specifies "Size MB vs NDCG@10" for this figure, and the
    # implementation plotted `total_parameters` under an axis labelled "Total
    # Parameters" -- a different quantity from the requirement's. `model_size_bf16_mb`
    # is BF16 because that is the dtype the models are trained and served in
    # (transformer-backbone AC-08), so it is the size a reader would act on.
    sizes = [results[m].get("model_size_bf16_mb", 0.0) for m in models]

    if any(s > 0 for s in sizes):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sizes, ndcgs)
        for m, s, n in zip(models, sizes, ndcgs):
            ax.annotate(m, (s, n))
        ax.set_xlabel("Model size (MB, BF16)")
        ax.set_ylabel("NDCG@10")
        ax.set_title("Accuracy vs Model Size")
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, "accuracy_vs_model_size.png"), dpi=config.plot_dpi)
        plt.close(fig)

    # ---- FR-14 plot 2: accuracy vs communication cost ----------------------
    # REV-014: only accuracy_vs_model_size was implemented; the other three FR-14
    # figures had no code at all.
    comm_points = []
    for model_name in ["t_small", "t_base", "t_large"]:
        fed_path = os.path.join(config.log_dir, f"federated_rounds_{model_name}.json")
        if not os.path.exists(fed_path):
            continue
        with open(fed_path) as f:
            rounds = json.load(f)
        if not rounds:
            continue
        # REV-101(b): FR-14 specifies "Comm cost/round vs NDCG@10". This summed
        # `total_round_kb` across every round and labelled the axis "Cumulative
        # communication (KB)", so the ratio between two models changed with how many
        # rounds each happened to run. `total_round_kb` is already the per-round figure.
        per_round_kb = rounds[-1].get("total_round_kb", 0.0)
        # REV-117: the headline federated number is the final round's FULL-SPLIT score,
        # not the bounded per-round prefix the convergence curve plots.
        comm_points.append((model_name, per_round_kb, final_val_ndcg(rounds)))

    if comm_points:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [c for _, c, _ in comm_points]
        ys = [n for _, _, n in comm_points]
        ax.scatter(xs, ys)
        for name, c, n in comm_points:
            ax.annotate(name, (c, n))
        ax.set_xlabel("Communication per round (KB)")
        ax.set_ylabel("Final federated NDCG@10")
        ax.set_title("Accuracy vs Communication Cost")
        fig.tight_layout()
        fig.savefig(
            os.path.join(config.plot_dir, "accuracy_vs_communication.png"),
            dpi=config.plot_dpi,
        )
        plt.close(fig)
    else:
        logger.warning(
            "accuracy_vs_communication.png skipped: no federated_rounds_*.json found. "
            "Run --stage federated first."
        )

    # ---- FR-14 plot 3: accuracy vs FLOPs -----------------------------------
    flops = [results[m].get("flops_per_inference", 0) for m in models]
    if any(f > 0 for f in flops):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(flops, ndcgs)
        for m, f, n in zip(models, flops, ndcgs):
            ax.annotate(m, (f, n))
        ax.set_xlabel("FLOPs per inference")
        ax.set_ylabel("NDCG@10")
        ax.set_title("Accuracy vs FLOPs")
        fig.tight_layout()
        fig.savefig(
            os.path.join(config.plot_dir, "accuracy_vs_flops.png"), dpi=config.plot_dpi,
        )
        plt.close(fig)
    else:
        logger.warning(
            "accuracy_vs_flops.png skipped: no flops_per_inference in the efficiency "
            "reports. Re-run --stage latency."
        )

    # ---- FR-14 plot 4: personalization gain vs LoRA rank -------------------
    rank_points = []
    for rank in config.lora_rank_grid:
        gain_path = os.path.join(
            config.log_dir, f"personalization_gain_rank{rank}_{args.model.replace('-', '_')}.json",
        )
        if not os.path.exists(gain_path):
            continue
        with open(gain_path) as f:
            report = json.load(f)
        gain = report.get("gain", {})
        rank_points.append((rank, gain.get("mean", 0.0), gain.get("std", 0.0)))

    if rank_points:
        fig, ax = plt.subplots(figsize=(8, 5))
        ranks = [r for r, _, _ in rank_points]
        means = [m for _, m, _ in rank_points]
        stds = [sd for _, _, sd in rank_points]
        ax.errorbar(ranks, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel("Mean personalization gain (NDCG@10)")
        ax.set_title("Personalization Gain vs LoRA Rank")
        ax.set_xticks(ranks)
        fig.tight_layout()
        fig.savefig(
            os.path.join(config.plot_dir, "personalization_gain_vs_lora_rank.png"),
            dpi=config.plot_dpi,
        )
        plt.close(fig)
    else:
        logger.warning(
            "personalization_gain_vs_lora_rank.png skipped: no "
            "personalization_gain_rank*_%s.json found. Run --stage lora_rank_sweep.",
            args.model.replace("-", "_"),
        )

    logger.info("Plot generation complete.")


def run_lora_rank_sweep_stage(config, args):
    """Sweep `config.lora_rank_grid`, writing one gain report per rank.

    REV-014: `personalization_gain_vs_lora_rank.png` had no data source -- nothing ever
    varied `lora_rank` or wrote a per-rank report.
    """
    from training.local_train import run_local_personalization

    mn = args.model.replace("-", "_")
    for rank in config.lora_rank_grid:
        logger.info("LoRA rank sweep: rank=%d", rank)
        cfg = copy.deepcopy(config)
        cfg.lora_rank = rank
        try:
            report = run_local_personalization(cfg, args.model)
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("  rank=%d skipped: %s", rank, exc)
            continue

        path = os.path.join(config.log_dir, f"personalization_gain_rank{rank}_{mn}.json")
        os.makedirs(config.log_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "lora_rank": rank,
                    "num_users_evaluated": report["num_users_evaluated"],
                    "with_lora": {k: float(v) for k, v in report["with_lora"].items()},
                    "without_lora": {k: float(v) for k, v in report["without_lora"].items()},
                    "gain": {k: float(v) for k, v in report["gain"].items()},
                },
                f, indent=2,
            )
        logger.info("  rank=%d mean gain=%.5f", rank, report["gain"]["mean"])


def run_multi_seed_stage(config, args):
    """FR-15: run `config.final_seeds` end-to-end and aggregate mean +/- std.

    REV-013: `config.final_seeds` was never read anywhere in the repository. `--seed`
    overrode a single seed; there was no per-seed output directory, no iteration and no
    aggregation, so the only thing possible was manufacturing a std from repeated
    evaluation of ONE seed -- which FR-15 explicitly warns against.
    """
    from training.pretrain import run_pretrain

    seeds = list(config.final_seeds)
    logger.info("Multi-seed run over seeds %s", seeds)

    base_output = config.output_dir
    per_seed = {}

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        # Each seed gets its own namespace so runs never overwrite each other.
        seed_root = os.path.join(base_output, f"seed_{seed}")
        cfg.output_dir = seed_root
        cfg.checkpoint_dir = os.path.join(seed_root, "checkpoints")
        cfg.log_dir = os.path.join(seed_root, "logs")
        cfg.plot_dir = os.path.join(seed_root, "plots")
        cfg.ablation_dir = os.path.join(seed_root, "ablations")

        setup_seed(cfg)
        logger.info("=== seed %d -> %s ===", seed, seed_root)
        train_result = run_pretrain(cfg, args.model)

        seed_args = argparse.Namespace(**vars(args))
        run_evaluate_stage(cfg, seed_args)

        mn = args.model.replace("-", "_")
        metrics_path = os.path.join(cfg.log_dir, f"test_metrics_{mn}.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                per_seed[seed] = json.load(f)
        else:
            logger.warning("seed %d produced no test metrics at %s", seed, metrics_path)
            per_seed[seed] = {"best_val_ndcg": train_result["best_val_ndcg"]}

    # Aggregate across seeds -- std is ACROSS independent runs, never within one.
    metric_names = set()
    for m in per_seed.values():
        metric_names.update(k for k, v in m.items() if isinstance(v, dict) and "mean" in v)

    aggregated = {}
    for name in sorted(metric_names):
        vals = [
            per_seed[s][name]["mean"] for s in seeds
            if name in per_seed.get(s, {}) and "mean" in per_seed[s][name]
        ]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        aggregated[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_seeds": len(arr),
            "per_seed": {str(s): float(v) for s, v in zip(seeds, vals)},
        }

    out = {
        "model": args.model,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "aggregated": aggregated,
        "note": (
            "std is computed ACROSS independent seed runs (ddof=1), not from repeated "
            "evaluation of a single seed."
        ),
    }
    os.makedirs(base_output, exist_ok=True)
    path = os.path.join(base_output, f"final_results_{args.model.replace('-', '_')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    for name, agg in aggregated.items():
        logger.info("%s: %.4f +/- %.4f (n=%d)", name, agg["mean"], agg["std"], agg["n_seeds"])
    logger.info("Multi-seed results written to %s", path)
    return out


def _all_pipeline_stages():
    """The stages `--stage all` dispatches, in order.

    REV-033: exposed as data so the "all excludes ablations" test can assert against the
    RESOLVED stage list instead of grepping `inspect.getsource` for the substring
    "ablation" -- which passed only incidentally and would break on an unrelated rename
    (a log line mentioning ablations) while still missing an indirect call.
    """
    return [
        ("preprocess", run_preprocess),
        ("pretrain", run_pretrain_stage),
        ("personalize", run_personalize_stage),
        ("federated", run_federated_stage),
        ("evaluate", run_evaluate_stage),
        ("cold_start", run_cold_start_stage),
        ("longtail", run_longtail_stage),
        ("latency", run_latency_stage),
        ("routing_diagnostics", run_routing_diagnostics_stage),
        ("gradient_diagnostics", run_gradient_diagnostics_stage),
        ("plots", run_plots_stage),
    ]


def run_all_pipeline(config, args):
    for name, fn in _all_pipeline_stages():
        logger.info(f"=== Stage: {name} ===")
        try:
            fn(config, args)
        except Exception as e:
            logger.error(f"Stage {name} failed: {e}")
            raise


STAGE_MAP = {
    "preprocess": run_preprocess,
    "pretrain": run_pretrain_stage,
    "personalize": run_personalize_stage,
    "federated": run_federated_stage,
    "evaluate": run_evaluate_stage,
    "cold_start": run_cold_start_stage,
    "longtail": run_longtail_stage,
    "latency": run_latency_stage,
    "routing_diagnostics": run_routing_diagnostics_stage,
    "gradient_diagnostics": run_gradient_diagnostics_stage,
    "lora_rank_sweep": run_lora_rank_sweep_stage,
    "multi_seed": run_multi_seed_stage,
    "ablation": run_ablation_stage,
    "plots": run_plots_stage,
    "all": run_all_pipeline,
}


def _resolve_before_snapshot(config: Config, args) -> None:
    """Resolve model dims and dataset-derived fields so the manifest describes the run."""
    if args.stage != "preprocess":
        config.apply_model_config(args.model)

    stats_path = os.path.join(config.processed_data_dir, "dataset_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            config.num_items = stats["num_items"]
            config.num_users = stats["num_users"]
            config.vocab_size = config.num_items + 2
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not read %s for the manifest: %s", stats_path, exc)

    meta_path = os.path.join(config.processed_data_dir, "item_metadata.npy")
    if config.use_metadata and os.path.exists(meta_path):
        try:
            config.feature_dim = np.load(meta_path, mmap_mode="r").shape[-1]
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s for the manifest: %s", meta_path, exc)


def main(argv=None):
    args = parse_args(argv)
    config = Config()
    apply_cli_overrides(config, args)
    # REV-029: resolve the model config and dataset-derived fields BEFORE snapshotting.
    _resolve_before_snapshot(config, args)
    setup_seed(config)
    save_config_snapshot(config)
    STAGE_MAP[args.stage](config, args)


if __name__ == "__main__":
    main()
