import argparse
import copy
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from configs.config import DATASET_PRESETS, OBJECTIVE_ORDER, Config
from models.transformer import VALID_BACKENDS
from utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_STAGES = [
    "preprocess", "pretrain",
    "evaluate", "cold_start", "longtail", "latency",
    "routing_diagnostics", "gradient_diagnostics",
    # REV-013: multi-seed aggregation (FR-15) had no entry point at all.
    "multi_seed",
    "ablation", "plots", "all",
    # specs/ablation-run FR-15: assemble study JSON + plots from persisted arm files.
    "ablation_collect",
    # specs/eval-integrity FR-8: re-evaluate a finished run's checkpoints, no training.
    "ablation_rescore",
]

VALID_MODELS = ["t-small", "t-base", "t-large"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Recommendation System Pipeline")
    parser.add_argument("--stage", required=True, choices=VALID_STAGES, help="Pipeline stage to run")
    parser.add_argument("--model", default="t-base", choices=VALID_MODELS, help="Model variant")
    # Every field of a named dataset (configs/config.py DATASET_PRESETS), applied before
    # any other flag, so e.g. `--epochs` or `--sequence_length_grid` still override it.
    # Absent = the MovieLens-20M defaults.
    parser.add_argument(
        "--dataset_preset", default=None, choices=list(DATASET_PRESETS),
        help="Select a dataset's settings at once (data dirs, output roots, sequence "
             "length, epochs), e.g. amazon_beauty_2014. Explicit flags override it.",
    )
    parser.add_argument(
        "--raw_data_dir", default=None,
        help="Override raw_data_dir (where preprocessing reads the raw files).",
    )
    parser.add_argument(
        "--processed_data_dir", default=None,
        help="Override processed_data_dir. An explicit directory is never rewritten for "
             "a user sample.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help="Override max_seq_length (the reference arm's sequence width).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override pretrain_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--moe_variant", default=None, choices=["none", "mmoe", "arv_mmoe"])
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--use_gradnorm", action="store_true", default=None)
    parser.add_argument("--no_gradnorm", action="store_true", default=False)
    # Static auxiliary loss weights, used when GradNorm is off (next-item's weight is
    # always 1). For example, a GradNorm-off arm fixed at GradNorm's learned weights.
    parser.add_argument("--masked_lm_weight", type=float, default=None)
    parser.add_argument("--multi_step_weight", type=float, default=None)
    parser.add_argument("--contrastive_weight", type=float, default=None)
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
    parser.add_argument(
        "--dtype", default=None, choices=["bf16", "fp16"],
        help="Compute dtype. fp16 requires --mixed_precision.",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", default=False,
        help="fp32 master weights + autocast in --dtype (+ GradScaler for fp16). "
             "Use on GPUs without native bf16, e.g. T4.",
    )
    # Owner decision (2026-09-17): Phase 17's gate is a few-step run of `run_pretrain` on
    # REAL data, not a full epoch. `max_steps_per_epoch` already existed as a Config
    # field but had no CLI surface, so running the gate meant editing source -- which
    # conventions.md's "Config-Driven Design" forbids.
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Cap training steps per epoch (0 = uncapped). The Phase 17 few-step smoke "
             "gate uses this to exercise run_pretrain end to end on real data.",
    )
    # specs/ablation-run (owner decisions 2026-10-15). Every flag defaults to None/False
    # so the configured values stay authoritative when it is absent.
    parser.add_argument(
        "--user_sample_fraction", type=float, default=None,
        help="Keep a seeded random fraction of users (0, 1]. Below 1 the processed "
             "directory moves to data/processed_users{pct}pct unless one is set.",
    )
    parser.add_argument(
        "--user_sample_seed", type=int, default=None,
        help="Seed of the user sample (independent of --seed).",
    )
    parser.add_argument(
        "--sequence_length_grid", type=int, nargs="+", default=None,
        help="Ablation H2 sequence lengths; preprocessing stores every split at "
             "max(grid + [max_seq_length]).",
    )
    parser.add_argument(
        "--attention_backend", default=None, choices=list(VALID_BACKENDS),
        help="Attention kernel. Experiment runs always name it explicitly.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader worker processes (does not change the batch sequence).",
    )
    parser.add_argument(
        "--min_lr_ratio", type=float, default=None,
        help="Cosine LR floor as a fraction of the peak LR, in [0, 1].",
    )
    parser.add_argument(
        "--no_resume", action="store_true", default=False,
        help="Ignore last.pt and persisted ablation arms; train/evaluate from scratch.",
    )
    # specs/ablation-run FR-14: run only some arms of one study (labels as resolved for
    # this config, e.g. H2_seq200, K6). Selects runs, like --study; not a Config field.
    parser.add_argument(
        "--arms", default=None,
        help="Ablations only: comma-separated arm labels of ONE --study to run.",
    )
    # Logit scale / GradNorm options for capped experiments. Absent = the configured
    # values (whose defaults reproduce the previous training exactly).
    parser.add_argument(
        "--logit_temperature", type=float, default=None,
        help="Divide next-item, masked-item and multi-step cosine logits by this (> 0).",
    )
    parser.add_argument(
        "--learnable_logit_scale", action="store_true", default=False,
        help="Learn each objective's logit scale (log-param, init 1/logit_temperature).",
    )
    parser.add_argument(
        "--logit_scale_max", type=float, default=None,
        help="Upper clamp of a learnable logit scale (default 100).",
    )
    parser.add_argument(
        "--shared_logit_temperature", action="store_true", default=False,
        help="Contrastive uses logit_temperature (and a learnable scale with "
             "--learnable_logit_scale) instead of contrastive_temperature.",
    )
    parser.add_argument(
        "--gradnorm_renorm_scope", default=None, choices=["active", "enabled"],
        help="GradNorm renormalisation over this step's active objectives (default) or "
             "over every enabled objective.",
    )
    parser.add_argument(
        "--gradnorm_primary_objective", default=None, choices=OBJECTIVE_ORDER,
        help="Fix this objective's GradNorm lambda at 1 and balance only the others.",
    )
    parser.add_argument(
        "--gradnorm_aux_budget", type=float, default=None,
        help="With --gradnorm_primary_objective: the auxiliary lambdas sum to this "
             "(default: their count).",
    )
    parser.add_argument(
        "--contrastive_interval", type=int, default=None,
        help="Run the contrastive objective every N steps.",
    )
    parser.add_argument(
        "--allow_incomplete", action="store_true", default=False,
        help="ablation_collect / ablation_rescore: assemble a study even if arms are missing/failed/stale "
             "(placeholders in the JSON; plots of the ok arms). Never for an inconsistent study.",
    )
    return parser.parse_args(argv)


def derive_processed_dir(config: Config) -> Optional[str]:
    """The processed directory a user sample lives in, or None for the full dataset.

    specs/ablation-run FR-6: `data/processed_users{pct}pct`, with `_seed{S}` when
    `user_sample_seed` differs from its default, so a sampled dataset never overwrites
    `data/processed`. Shared with `scripts/package_ablation_run.py`.

    For another dataset's full-data directory (a preset's `processed_data_dir`, e.g.
    `data/processed_beauty`) the sample goes next to it with the same suffix
    (`data/processed_beauty_users10pct`), so it never shares a directory with MovieLens'.
    """
    fraction = config.user_sample_fraction
    if fraction >= 1.0:
        return None
    pct = format(round(fraction * 100, 6), "g")
    suffix = f"users{pct}pct"
    if config.user_sample_seed != Config.user_sample_seed:
        suffix += f"_seed{config.user_sample_seed}"
    if config.processed_data_dir != Config.processed_data_dir:
        return f"{config.processed_data_dir.rstrip('/')}_{suffix}"
    base = os.path.dirname(Config.processed_data_dir) or "."
    return f"{base}/processed_{suffix}"


def apply_cli_overrides(config: Config, args) -> None:
    overrides = []
    # The preset first, so every explicit flag below overrides it.
    preset = getattr(args, "dataset_preset", None)
    if preset is not None:
        before = asdict(config)
        config.apply_dataset_preset(preset)
        for name, new in asdict(config).items():
            if before[name] != new:
                overrides.append((f"{name} [preset {preset}]", before[name], new))
    # The selected dataset's OWN full-data directory (the preset's, else the Config
    # default); a user sample is derived only from that one, never from a directory set
    # explicitly in code or by --processed_data_dir (FR-6).
    default_processed_dir = DATASET_PRESETS.get(preset or "", {}).get(
        "processed_data_dir", Config.processed_data_dir,
    )
    if args.epochs is not None:
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
    for field in ("masked_lm_weight", "multi_step_weight", "contrastive_weight"):
        value = getattr(args, field, None)
        if value is not None:
            overrides.append((field, getattr(config, field), value))
            setattr(config, field, value)
    if args.seed is not None:
        overrides.append(("seed", config.seed, args.seed))
        config.seed = args.seed
    if getattr(args, "dtype", None) is not None:
        overrides.append(("dtype", config.dtype, args.dtype))
        config.dtype = args.dtype
    if getattr(args, "mixed_precision", False):
        overrides.append(("mixed_precision", config.mixed_precision, True))
        config.mixed_precision = True
    if getattr(args, "max_steps", None) is not None:
        overrides.append(
            ("max_steps_per_epoch", config.max_steps_per_epoch, args.max_steps)
        )
        config.max_steps_per_epoch = args.max_steps
    # specs/ablation-run T1.2.
    for flag, field_name in (
        ("raw_data_dir", "raw_data_dir"),
        ("processed_data_dir", "processed_data_dir"),
        ("max_seq_length", "max_seq_length"),
        ("user_sample_fraction", "user_sample_fraction"),
        ("user_sample_seed", "user_sample_seed"),
        ("attention_backend", "attention_backend"),
        ("num_workers", "num_workers"),
        ("min_lr_ratio", "min_lr_ratio"),
        ("logit_temperature", "logit_temperature"),
        ("logit_scale_max", "logit_scale_max"),
        ("gradnorm_renorm_scope", "gradnorm_renorm_scope"),
        ("gradnorm_primary_objective", "gradnorm_primary_objective"),
        ("gradnorm_aux_budget", "gradnorm_aux_budget"),
        ("contrastive_interval", "contrastive_interval"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides.append((field_name, getattr(config, field_name), value))
            setattr(config, field_name, value)
    if getattr(args, "sequence_length_grid", None) is not None:
        grid = list(args.sequence_length_grid)
        overrides.append(("sequence_length_grid", config.sequence_length_grid, grid))
        config.sequence_length_grid = grid
    for flag in ("learnable_logit_scale", "shared_logit_temperature"):
        if getattr(args, flag, False):
            overrides.append((flag, getattr(config, flag), True))
            setattr(config, flag, True)
    if getattr(args, "no_resume", False):
        overrides.append(("resume_training", config.resume_training, False))
        config.resume_training = False
    # FR-6: a sample gets its own directory, but an explicitly chosen one is never
    # rewritten.
    if (
        config.processed_data_dir == default_processed_dir
        and getattr(args, "processed_data_dir", None) is None
    ):
        derived = derive_processed_dir(config)
        if derived is not None:
            overrides.append(("processed_data_dir", config.processed_data_dir, derived))
            config.processed_data_dir = derived

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


def save_config_snapshot(config: Config, selection: Optional[dict] = None) -> None:
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
    # specs/ablation-run FR-17: which runs this process selected (several processes may
    # share one log dir when a study's arms are split across GPUs).
    manifest["selection"] = selection if selection is not None else getattr(
        config, "_run_selection", None,
    )
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

    # AC-09 / steering "Artifact Provenance": the run manifest records WHICH CODE this run
    # executed, alongside which hardware and kernel it ran on. `steering/code-quality.md`
    # already requires the torch version, device, dtype and resolved backend here; the
    # source revision is the same kind of fact and the one REV-129 could not recover.
    try:
        from utils import get_source_revision

        manifest["source_revision"] = get_source_revision()
    except Exception as exc:  # never let manifest-writing abort a run
        logger.warning("Could not resolve source_revision for the manifest: %s", exc)

    # FR-17 / D19: atomic, so concurrent processes never leave a torn file.
    from utils import atomic_write_json

    atomic_write_json(path, manifest)


def run_preprocess(config, args):
    from data.preprocessing import run_preprocessing
    run_preprocessing(config)


def run_pretrain_stage(config, args):
    from training.pretrain import run_pretrain
    return run_pretrain(config, args.model)


def run_evaluate_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.metrics import evaluate_model, load_full_ranking
    from data.sequence_generator import load_tensorised, load_metadata_matrix
    from utils import get_device, artifact_source_revision
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
    ckpt = load_checkpoint(ckpt_path, model, moe)
    # AC-09: the test metrics below are derived from THIS checkpoint.
    source_revision = artifact_source_revision(ckpt)

    # REV-058/REV-072: one loader, so the (vocab, F) alignment applies uniformly.
    metadata_matrix = load_metadata_matrix(processed, config)
    if metadata_matrix is not None:
        config.feature_dim = metadata_matrix.shape[-1]

    # specs/ablation-run FR-7: splits are stored wider than the model; narrow on read.
    test_dataset = load_tensorised(
        os.path.join(processed, "test_tensorised.pt"), metadata_matrix,
        max_seq_length=config.max_seq_length,
    )
    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)

    # Leave-one-out, full catalogue: each user's test item is ranked against every item
    # except their training and validation history.
    test_sequences = test_dataset.sequences
    candidate_tensors = load_full_ranking(
        processed, "test", test_dataset.targets, test_dataset.user_ids,
    )

    # FR-2 / REV-064: the retrieval-then-rerank path. `use_ann` and `ann_candidates` were
    # declared in the config and read by nothing, so this requirement had no code behind
    # it at all. Off by default, so full-catalogue ranking remains the headline.
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
    # specs/eval-integrity FR-7.2 / FR-1.5: the popularity-only floor on the candidates
    # actually scored (the ANN candidates when ANN is on), and the protocol stamp.
    from evaluation.metrics import EVAL_PROTOCOL, evaluate_popularity, item_popularity
    from training.pretrain import POPULARITY_EVAL_BATCH

    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    payload["popularity_reference"] = evaluate_popularity(
        candidate_tensors, item_popularity(train_user_history, config.vocab_size),
        k=config.eval_k, batch_size=POPULARITY_EVAL_BATCH,
    )
    payload["eval_protocol"] = dict(EVAL_PROTOCOL)
    payload["source_revision"] = source_revision
    with open(os.path.join(config.log_dir, f"test_metrics_{args.model.replace('-', '_')}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def run_cold_start_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.cold_start import run_cold_start_evaluation
    from utils import get_device, artifact_source_revision

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
    ckpt = load_checkpoint(ckpt_path, model, moe)
    run_cold_start_evaluation(
        model, moe, config, args.model,
        source_revision=artifact_source_revision(ckpt),
    )


def run_longtail_stage(config, args):
    from training.pretrain import build_model, load_checkpoint
    from evaluation.longtail import run_longtail_evaluation
    from utils import artifact_source_revision

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
    ckpt = load_checkpoint(ckpt_path, model, moe)
    run_longtail_evaluation(
        model, moe, config, args.model,
        source_revision=artifact_source_revision(ckpt),
    )


def run_latency_stage(config, args):
    from copy import deepcopy

    from training.pretrain import build_model, load_checkpoint
    from evaluation.flops import (
        benchmark_latency, count_parameters, save_efficiency_report,
        measure_routing_overhead, estimate_flops_per_inference,
    )
    from models.moe import MoELayer
    from utils import get_device, resolve_param_dtype, artifact_source_revision

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
    ckpt = load_checkpoint(ckpt_path, model, moe)
    source_revision = artifact_source_revision(ckpt)
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
            device=get_device(config), dtype=resolve_param_dtype(config),
        )
        overhead = measure_routing_overhead(model, moe, moe_none, config)
        logger.info(
            "Routing overhead: %.3f ms (%.3f with MoE vs %.3f without)",
            overhead["overhead_ms"], overhead["full_mean_ms"], overhead["none_mean_ms"],
        )

    # REV-021b: FR-7 names flops_per_inference; accuracy_vs_flops.png has no other source.
    params["flops_per_inference"] = estimate_flops_per_inference(config, moe)

    save_efficiency_report(
        config, args.model, latency, params, overhead=overhead,
        source_revision=source_revision,
    )
    logger.info(
        "Efficiency report written for %s: %.3f ms/rec, %d params, %.1f MB bf16",
        args.model, latency["ms_per_recommendation"],
        params["total_parameters"], params["model_size_bf16_mb"],
    )


def _load_trained_model(config, args):
    """Shared setup for the standalone diagnostics stages: stats, model, best.pt."""
    from training.pretrain import build_model, load_checkpoint, _build_loss_fns
    from data.sequence_generator import load_metadata_matrix
    from utils import get_device, artifact_source_revision

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
    ckpt = load_checkpoint(ckpt_path, model, moe, loss_fns=loss_fns)
    # AC-09: diagnostics derived from this checkpoint echo ITS source_revision.
    source_revision = artifact_source_revision(ckpt)
    return model, moe, loss_fns, metadata_matrix, device, processed, source_revision


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
    model, moe, _, metadata_matrix, _, processed, source_revision = loaded

    val_dataset = load_tensorised(
        os.path.join(processed, "val_tensorised.pt"), metadata_matrix,
        max_seq_length=config.max_seq_length,
    )
    per_objective = collect_validation_routing(model, moe, val_dataset, config)
    if not per_objective:
        logger.warning(
            "No routing snapshots collected -- moe_variant=%s produces no routing "
            "weights.", config.moe_variant,
        )
        return
    report = save_routing_diagnostics(
        per_objective, config, args.model, source_revision=source_revision,
    )
    logger.info(
        "Routing diagnostics written for %s: overall_entropy_mean=%.4f",
        args.model, report["overall_entropy_mean"],
    )
    return report


def run_gradient_diagnostics_stage(config, args):
    """Standalone gradient conflict/dominance pass on fresh training batches.

    REV-009: `run_gradient_diagnostics` previously had no call site. It reads the same
    chunked training split, at the same width, that `run_pretrain` trains on.
    """
    from training.gradnorm import get_anchor_layer_params
    from evaluation.gradient_diagnostics import run_gradient_diagnostics
    from data.sequence_generator import load_train_dataset, build_dataloader
    from utils import resolve_param_dtype

    loaded = _load_trained_model(config, args)
    if loaded is None:
        return
    model, moe, loss_fns, metadata_matrix, device, processed, source_revision = loaded

    train_dataset = load_train_dataset(processed, config.max_seq_length, metadata_matrix)
    loader = build_dataloader(train_dataset, config, shuffle=True)

    metadata_matrix_t = (
        metadata_matrix.to(resolve_param_dtype(config)) if metadata_matrix is not None else None
    )

    report = run_gradient_diagnostics(
        model, moe, loader, loss_fns,
        get_anchor_layer_params(model, config), config, args.model,
        metadata_matrix=metadata_matrix_t,
        augment_rng=random.Random(config.seed),
        source_revision=source_revision,
    )
    if not report:
        logger.warning("No diagnostic batches processed; no report written.")
        return
    logger.info(
        "Gradient diagnostics written for %s over objectives %s",
        args.model, report.get("objective_names"),
    )
    return report


def _exit_if_arms_failed(stage: str, failed: dict) -> None:
    """Exit `ARM_FAILED_EXIT_CODE` when any requested study holds a failed arm.

    A failed arm used to leave the process at exit 0, so the launch scripts and
    `scripts/run_with_retry.py` reported success while arms had crashed.
    """
    if not failed:
        return
    from utils import ARM_FAILED_EXIT_CODE

    logger.error(
        "%s: failed arms %s -- exiting with status %d", stage, failed, ARM_FAILED_EXIT_CODE,
    )
    raise SystemExit(ARM_FAILED_EXIT_CODE)


def run_ablation_stage(config, args):
    from training.ablation_runner import failed_arm_labels, run_ablation
    # REV-115: `--no_train` reaches the `train` parameter that already existed here.
    from training.ablation_runner import parse_arms
    results = run_ablation(
        config, args.model, args.study, train=not getattr(args, "no_train", False),
        arms=parse_arms(getattr(args, "arms", None)),
    )
    # Every arm has run and every record (and study output) is written by now; an OOM
    # never gets here -- it propagates and exits 75 in `main`.
    _exit_if_arms_failed("ablation", failed_arm_labels(results))
    return results


def run_ablation_collect_stage(config, args):
    """`--stage ablation_collect`: study JSON + plots from persisted arms (FR-15, D17).

    Trains and evaluates nothing. Exits `ARM_FAILED_EXIT_CODE` if any requested study
    holds a failed arm (assembled or not, as `--stage ablation` would), else 1 if any
    requested study could not be assembled (listing its missing / stale / inconsistent
    arms).
    """
    from training.ablation_runner import (
        STUDY_CONFIGS, _save_summary, collect_study,
    )

    allow = getattr(args, "allow_incomplete", False)
    studies = list(STUDY_CONFIGS) if args.study == "all" else [args.study]
    if any(s not in STUDY_CONFIGS for s in studies):
        raise ValueError(f"Unknown study '{args.study}'. Choose from {list(STUDY_CONFIGS)} or all")
    assembled, incomplete, failed_studies, failed_arms = {}, {}, [], {}
    for study in studies:
        assessment = collect_study(config, args.model, study, allow_incomplete=allow)
        if assessment.failed:
            failed_arms[study] = sorted(assessment.failed)
        if assessment.written:
            from training.ablation_runner import _results_root

            mn = args.model.replace("-", "_")
            with open(os.path.join(_results_root(config), f"{study}_{mn}.json")) as f:
                assembled[study] = json.load(f)
            logger.info("Collected %s", assessment.describe())
            if not assessment.complete:
                incomplete[study] = {
                    "missing": assessment.missing + assessment.stale,
                    "failed": sorted(assessment.failed),
                }
        else:
            failed_studies.append(study)
            logger.error("NOT collected -- %s", assessment.describe())
            incomplete[study] = {
                "missing": assessment.missing + assessment.stale,
                "failed": sorted(assessment.failed),
                "inconsistent": assessment.inconsistent,
            }
    if args.study == "all":
        _save_summary(
            config, args.model,
            {s: r for s, r in assembled.items() if s not in incomplete},
            incomplete=incomplete,
        )
    if failed_studies:
        logger.error("ablation_collect: studies not assembled: %s", failed_studies)
    _exit_if_arms_failed("ablation_collect", failed_arms)
    if failed_studies:
        raise SystemExit(1)


def run_ablation_rescore_stage(config, args):
    """`--stage ablation_rescore` (specs/eval-integrity FR-8): rescore a finished run's
    saved epoch checkpoints with the current evaluation protocol. Trains nothing and
    modifies no original. Exits `ARM_FAILED_EXIT_CODE` if any selected arm failed (its
    original record is `failed`, or its rescore raised), else 1 if any selected study
    could not be assembled."""
    from training.ablation_rescore import run_ablation_rescore

    report = run_ablation_rescore(
        config, args.model, args.study,
        allow_incomplete=getattr(args, "allow_incomplete", False),
    )
    for study, info in report["studies"].items():
        logger.info(
            "Rescore %s: rescored %s, already current %s, skipped %s, failed %s -- %s",
            study, info["rescored"], info["current"], sorted(info["skipped"]),
            sorted(info["failed"]), info["assessment"],
        )
    if report["not_written"]:
        logger.error("ablation_rescore: studies not assembled: %s", report["not_written"])
    _exit_if_arms_failed("ablation_rescore", report.get("failed_arms") or {})
    if report["not_written"]:
        raise SystemExit(1)
    return report


def run_plots_stage(config, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # ---- FR-14 plot 2: accuracy vs FLOPs -----------------------------------
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

    logger.info("Plot generation complete.")


def run_multi_seed_stage(config, args):
    """FR-15: run `config.final_seeds` end-to-end and aggregate mean +/- std.

    REV-013: `config.final_seeds` was never read anywhere in the repository. `--seed`
    overrode a single seed; there was no per-seed output directory, no iteration and no
    aggregation, so the only thing possible was manufacturing a std from repeated
    evaluation of ONE seed -- which FR-15 explicitly warns against.
    """
    from training.ablation_runner import _training_fingerprint, register_backbone
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
        # Register the finished per-seed backbone in the ablation store, so ablation arms
        # whose training config matches it (the base config, run with `--seed <seed>`)
        # load this model instead of retraining an identical one. The fingerprint is
        # taken AFTER `run_pretrain`, which resolves the dataset-derived fields.
        register_backbone(
            config.checkpoint_dir, _training_fingerprint(cfg), cfg,
            f"multi_seed_{seed}", train_result,
        )

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

    # specs/eval-integrity G6 / FR-1.5 (review REV-001): the headline file carries the
    # protocol stamp and the popularity floor. The test candidates are fixed, so every
    # seed's floor is the same; take it from the first seed that reports one.
    from evaluation.metrics import EVAL_PROTOCOL

    popularity_reference = next(
        (per_seed[s]["popularity_reference"] for s in seeds
         if isinstance(per_seed.get(s, {}).get("popularity_reference"), dict)),
        None,
    )
    out = {
        "model": args.model,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "aggregated": aggregated,
        "popularity_reference": popularity_reference,
        "eval_protocol": dict(EVAL_PROTOCOL),
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
    "evaluate": run_evaluate_stage,
    "cold_start": run_cold_start_stage,
    "longtail": run_longtail_stage,
    "latency": run_latency_stage,
    "routing_diagnostics": run_routing_diagnostics_stage,
    "gradient_diagnostics": run_gradient_diagnostics_stage,
    "multi_seed": run_multi_seed_stage,
    "ablation": run_ablation_stage,
    "ablation_collect": run_ablation_collect_stage,
    "ablation_rescore": run_ablation_rescore_stage,
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


def _validate_arm_selection(config: Config, args) -> None:
    """specs/ablation-run FR-14: reject a bad `--arms` before anything else happens."""
    if getattr(args, "arms", None) is None:
        return
    if args.stage != "ablation":
        raise ValueError(f"--arms is only valid with --stage ablation (got --stage {args.stage})")
    from training.ablation_runner import parse_arms, select_study_configs

    probe = copy.deepcopy(config)
    probe.apply_model_config(args.model)
    select_study_configs(args.study, probe, parse_arms(args.arms))


def main(argv=None):
    args = parse_args(argv)
    config = Config()
    apply_cli_overrides(config, args)
    _validate_arm_selection(config, args)
    # REV-029: resolve the model config and dataset-derived fields BEFORE snapshotting.
    _resolve_before_snapshot(config, args)
    setup_seed(config)
    # FR-17: the selection travels on the config (not a dataclass field, so it is in
    # neither the fingerprint nor asdict()) and is recorded by the snapshot.
    config._run_selection = {
        "stage": args.stage,
        "study": getattr(args, "study", None),
        "arms": getattr(args, "arms", None),
    }
    save_config_snapshot(config)
    # specs/eval-integrity FR-3.1: no stage-wide autocast region. Each forward site opens
    # its own (utils.autocast_context, cache off); a region spanning a whole stage froze
    # every nn.Linear at its first-step bf16 cast under mixed precision.
    from utils import OOM_EXIT_CODE, is_oom_error
    try:
        STAGE_MAP[args.stage](config, args)
    except Exception as exc:
        # specs/ablation-run FR-9: an out-of-memory error (often another user's job on a
        # shared GPU) exits with a status the retry launcher understands; completed work
        # (last.pt, persisted ablation arms) is picked up by the re-run.
        if is_oom_error(exc):
            logger.error(
                "OOM in stage %s: %s -- exiting with status %d for the retry launcher",
                args.stage, exc, OOM_EXIT_CODE,
            )
            raise SystemExit(OOM_EXIT_CODE) from exc
        raise


if __name__ == "__main__":
    main()
