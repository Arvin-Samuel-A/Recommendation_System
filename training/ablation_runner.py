import copy
import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from configs.config import Config

# REV-032: `replace` and `Optional` were imported and unused; `asdict` is now used by the
# per-result config snapshot (REV-019).

logger = logging.getLogger(__name__)

STUDY_CONFIGS = {
    "A": [
        {"label": "single_obj_baseline", "use_masked_lm": False, "use_multi_step": False, "use_contrastive": False, "moe_variant": "none", "use_gradnorm": False},
        {"label": "4obj_no_routing", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "none", "use_gradnorm": False},
        {"label": "conv_mmoe", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "mmoe", "use_gradnorm": False},
        {"label": "arv_mmoe", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "arv_mmoe", "use_gradnorm": False},
        {"label": "arv_mmoe_gradnorm", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "arv_mmoe", "use_gradnorm": True, "personalization_mode": "none"},
        # The final arm of the progression adds LoRA personalization. It used to be
        # byte-identical to the arm above it, so the two could never differ.
        {"label": "arv_mmoe_gradnorm_lora", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "arv_mmoe", "use_gradnorm": True, "personalization_mode": "lora"},
    ],
    "B": [
        {"label": "objective_conditioned", "moe_variant": "arv_mmoe", "router_conditioning": "objective_conditioned"},
        {"label": "input_only", "moe_variant": "arv_mmoe", "router_conditioning": "input_only"},
    ],
    "C": [
        {"label": "mmoe", "moe_variant": "mmoe", "match_mmoe_expert_count": True},
        {"label": "arv_mmoe", "moe_variant": "arv_mmoe", "match_mmoe_expert_count": True},
    ],
    "D": [
        {"label": "none_no_gradnorm", "moe_variant": "none", "use_gradnorm": False},
        {"label": "none_gradnorm", "moe_variant": "none", "use_gradnorm": True},
        {"label": "arv_no_gradnorm", "moe_variant": "arv_mmoe", "use_gradnorm": False},
        {"label": "arv_gradnorm", "moe_variant": "arv_mmoe", "use_gradnorm": True},
    ],
    # FR-10: global only / global+LoRA / global+LoRA+ARV. REV-019: these three used to
    # resolve to identical configs, so the arms were indistinguishable in their own
    # result JSON; `personalization_mode` makes the dimension explicit and config-driven.
    "E": [
        {"label": "global_only", "personalization_mode": "none", "moe_variant": "none"},
        {"label": "global_lora", "personalization_mode": "lora", "moe_variant": "none"},
        {"label": "global_lora_arv", "personalization_mode": "lora", "moe_variant": "arv_mmoe"},
    ],
    "F": [
        {"label": "full_reference", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True},
        {"label": "no_masked", "use_masked_lm": False, "use_multi_step": True, "use_contrastive": True},
        {"label": "no_multistep", "use_masked_lm": True, "use_multi_step": False, "use_contrastive": True},
        {"label": "no_contrastive", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": False},
    ],
    # REV-067: G's expert counts come from `config.expert_count_grid`, not from a list
    # frozen at import time. Sweeping a different set of K must be a config change, which
    # is what FR-5's "no source edits" constraint and AC-01 exist to guarantee.
    "G": [
        {"_grid": "expert_count"},
    ],
    # FR-13: each sub-study selectable without source edits. H3-H6 used to carry no
    # overrides at all, so they resolved to the base config and ran nothing specific.
    "H": [
        {"label": "H1_no_metadata", "use_metadata": False},
        # REV-067: FR-13 specifies "H2: Sequence length sweep over
        # `sequence_length_grid`" -- four points, not the single seq25 arm this was.
        {"_grid": "sequence_length"},
        {"label": "H3_cold_start", "system_eval": "cold_start"},
        {"label": "H4_longtail", "system_eval": "longtail"},
        {"label": "H5_latency", "system_eval": "latency"},
        {"label": "H6_communication", "system_eval": "communication"},
    ],
}


#: Grid marker -> (config field holding the grid, arm field to set, label template).
#: REV-067: `expert_count_grid` and `sequence_length_grid` existed in `configs/config.py`
#: and were read by nothing; the sweeps they describe were hardcoded in STUDY_CONFIGS.
_GRID_SPECS = {
    "expert_count": ("expert_count_grid", "num_experts", "K{value}"),
    "sequence_length": ("sequence_length_grid", "max_seq_length", "H2_seq{value}"),
}


def _study_entries(study: str, base_config: Config) -> List[Dict]:
    """The arm list for a study, expanding grid markers against the base config.

    REV-067: expansion happens here rather than at module import time, so the arms track
    whatever grid the resolved config carries -- including a CLI or config-file override.
    """
    entries: List[Dict] = []
    for entry in STUDY_CONFIGS[study]:
        grid_name = entry.get("_grid")
        if grid_name is None:
            entries.append(entry)
            continue
        if grid_name not in _GRID_SPECS:
            raise ValueError(
                f"Study {study}: unknown grid marker '{grid_name}'. "
                f"Choose from {sorted(_GRID_SPECS)}"
            )
        grid_field, arm_field, label_template = _GRID_SPECS[grid_name]
        grid = getattr(base_config, grid_field, None)
        if not grid:
            raise ValueError(
                f"Study {study}: config field '{grid_field}' is empty, so the "
                f"{grid_name} sweep has no arms to run."
            )
        for value in grid:
            arm = {k: v for k, v in entry.items() if k != "_grid"}
            arm["label"] = label_template.format(value=value)
            arm[arm_field] = value
            entries.append(arm)
    return entries


def resolve_study_configs(study: str, base_config: Config) -> List[Config]:
    if study not in STUDY_CONFIGS:
        raise ValueError(f"Unknown study '{study}'. Choose from {list(STUDY_CONFIGS.keys())}")

    configs = []
    for entry in _study_entries(study, base_config):
        cfg = copy.deepcopy(base_config)
        # REV-020: `overrides.pop("label")` mutated the module-level STUDY_CONFIGS in
        # place and put the key back two lines later. Any exception in between left
        # STUDY_CONFIGS permanently missing that label for the rest of the process,
        # silently degrading every later call including `--study all`. Copy first.
        overrides = dict(entry)
        label = overrides.pop("label", f"config_{len(configs)}")
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.warning(f"Study {study}: unknown config key '{key}'")
        cfg._study_label = label
        cfg._study_overrides = overrides
        configs.append(cfg)
    return configs


#: Study -> the extra diagnostics FR-5..FR-13 require beyond metrics.
STUDY_EXTRAS = {
    "A": ("personalization",),
    "B": ("routing",),
    "E": ("personalization",),
    "C": ("efficiency",),
    "D": ("gradient",),
    # REV-065: FR-12's figure is "NDCG left axis, latency right axis", so G needs the
    # efficiency numbers too -- they were not being gathered at all.
    "G": ("routing", "efficiency"),
    "H": ("efficiency",),
}


def _config_diff(cfg: Config, base: Config) -> Dict:
    """Every field where this config differs from the base.

    REV-019: the previous code took the key set of the study's FIRST override dict and
    applied it to every config in the study. Study E's first entry is `{"label": ...}`
    only, so every E result recorded an EMPTY diff; study H recorded only
    `use_metadata`. The diff is now computed per-config against the base.
    """
    base_d, cfg_d = asdict(base), asdict(cfg)
    return {k: v for k, v in cfg_d.items() if base_d.get(k) != v}


#: Config fields that are read ONLY after training finishes, so two arms differing only
#: in these produce a bit-identical backbone (REV-084). Verified against the codebase:
#: `personalization_mode` and `system_eval` are read solely by this module's post-training
#: extras dispatch, and `ablation_eval_examples` only bounds the evaluation slice.
#: Anything not listed here is treated as training-relevant -- the safe default, since a
#: missing entry only costs a redundant run, while a wrong entry would silently make two
#: genuinely different arms share one model.
POST_TRAINING_ONLY_FIELDS = frozenset({
    "personalization_mode",
    "system_eval",
    "ablation_eval_examples",
})

#: Per-arm output namespaces, which carry no training semantics.
_PATH_FIELDS = frozenset({
    "checkpoint_dir", "output_dir", "log_dir", "plot_dir", "ablation_dir",
})


def _training_fingerprint(cfg: Config) -> str:
    """Hash of everything about `cfg` that actually changes the trained model.

    REV-084: Ablation A's arms 5/6 and E's `global_only`/`global_lora` differ only in
    `personalization_mode`, which `run_pretrain` never reads. Under label-namespaced
    checkpoints each pair trained two bit-identical models -- two full pretraining runs
    wasted per pair, at `pretrain_epochs=20` over 7.8M examples -- and produced identical
    `metrics`, which reads as a null result for personalization rather than as what it
    is: the two arms sharing a backbone by construction.
    """
    fields = {
        k: v for k, v in asdict(cfg).items()
        if k not in POST_TRAINING_ONLY_FIELDS and k not in _PATH_FIELDS
    }
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _run_single_config(
    cfg: Config,
    model_name: str,
    study: str,
    base_config: Config,
    train: bool,
    trained_backbones: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Train (or load), evaluate and diagnose one ablation config.

    REV-006: the runner never executed an experiment -- it built a
    `{"label", "config_diff"}` dict and appended it. No model was constructed, nothing
    was trained or evaluated, and no plot was rendered, so all eight studies produced no
    results.
    """
    from training.pretrain import run_pretrain, build_model, load_checkpoint, _build_loss_fns
    from evaluation.metrics import evaluate_model
    from evaluation.flops import (
        benchmark_latency, count_parameters, estimate_flops_per_inference,
    )
    from data.sequence_generator import (
        load_tensorised, load_eval_candidates, load_metadata_matrix,
    )
    from utils import get_device

    label = getattr(cfg, "_study_label", "unknown")
    result = {
        "label": label,
        "study": study,
        "model_name": model_name,
        # AC-06: results must be reproducible from their own JSON. `config_diff` is
        # empty for an arm that happens to equal the base config, so the study's own
        # override set is recorded too -- an arm is always self-describing.
        "config_diff": _config_diff(cfg, base_config),
        "study_overrides": dict(getattr(cfg, "_study_overrides", {})),
        "resolved_config": asdict(cfg),
    }

    processed = cfg.processed_data_dir
    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    cfg.num_items = stats["num_items"]
    cfg.num_users = stats["num_users"]
    cfg.vocab_size = cfg.num_items + 2

    # REV-058/REV-072: one loader, so the (vocab, F) alignment applies uniformly.
    metadata_matrix = load_metadata_matrix(processed, cfg)
    if metadata_matrix is not None:
        cfg.feature_dim = metadata_matrix.shape[-1]

    # Each config gets its own checkpoint/log namespace so studies never overwrite
    # each other's artifacts.
    run_tag = f"{study}_{label}"
    cfg.checkpoint_dir = os.path.join(base_config.checkpoint_dir, "ablations", run_tag)

    # REV-084: arms that differ only in post-training fields train the SAME model. Reuse
    # the first such arm's checkpoint instead of retraining it, and say so in the result
    # so identical `metrics` between paired arms read as a shared backbone rather than as
    # a null result for the dimension the pair is supposed to isolate.
    fingerprint = _training_fingerprint(cfg)
    result["training_fingerprint"] = fingerprint
    shared = (trained_backbones or {}).get(fingerprint)
    if shared is not None:
        cfg.checkpoint_dir = shared["checkpoint_dir"]
        result["shared_backbone_with"] = shared["label"]
        for key in ("best_val_ndcg", "best_epoch"):
            if key in shared:
                result[key] = shared[key]
        logger.info(
            "  %s: training-relevant config matches '%s' -- reusing its backbone "
            "instead of retraining an identical model", label, shared["label"],
        )
        train = False

    ckpt_path = os.path.join(cfg.model_checkpoint_dir(), "best.pt")

    if train:
        train_result = run_pretrain(cfg, model_name)
        result["best_val_ndcg"] = train_result["best_val_ndcg"]
        result["best_epoch"] = train_result["best_epoch"]
    elif not os.path.exists(ckpt_path):
        result["status"] = "skipped_no_checkpoint"
        result["checkpoint_path"] = ckpt_path
        logger.warning(
            "  %s: no checkpoint at %s and train=False -- skipping", label, ckpt_path,
        )
        return result

    if trained_backbones is not None and fingerprint not in trained_backbones:
        entry = {"label": label, "checkpoint_dir": cfg.checkpoint_dir}
        for key in ("best_val_ndcg", "best_epoch"):
            if key in result:
                entry[key] = result[key]
        trained_backbones[fingerprint] = entry

    device = get_device(cfg)
    model, moe = build_model(cfg)
    loss_fns = _build_loss_fns(cfg, device)
    load_checkpoint(ckpt_path, model, moe, loss_fns=loss_fns)

    # Evaluate on the FIXED stored candidate sets, so every ablation arm is scored
    # against identical candidates (data-pipeline REV-041).
    test_dataset = load_tensorised(
        os.path.join(processed, "test_tensorised.pt"), metadata_matrix,
        max_seq_length=cfg.max_seq_length,
    )
    cand_matrix = load_eval_candidates(os.path.join(processed, "test_candidates.npy"))
    n = min(cfg.ablation_eval_examples, len(test_dataset), cand_matrix.shape[0])
    # REV-063: hand `evaluate_model` the stacked tensors and let it batch, rather than
    # building n per-example device tensors and forwarding a batch of 1 each.
    sequences = test_dataset.sequences[:n]
    candidates = torch.as_tensor(cand_matrix[:n], dtype=torch.long)

    model.eval()
    moe.eval()
    with torch.no_grad():
        metrics = evaluate_model(
            model, sequences, candidates, cfg,
            moe_layer=moe, metadata_matrix=test_dataset.metadata_matrix,
        )
    result["metrics"] = {
        k: {sk: float(sv) for sk, sv in v.items()} for k, v in metrics.items()
    }
    result["num_eval_examples"] = n

    extras = STUDY_EXTRAS.get(study, ())

    # REV-078: FR-6 requires the router/gate parameter report "for every `moe_variant`
    # run", and closes with "Never claim 'parameter-matched' without the report backing
    # it". `count_router_parameters` had no caller in the pipeline at all, so Ablation C's
    # parameter-matching claim had no artifact behind it. The run tag keeps each arm's
    # report distinct -- C runs two variants and G four expert counts at one model size.
    if cfg.moe_variant != "none":
        from models.moe import count_router_parameters

        result["router_parameters"] = count_router_parameters(cfg, run_tag=run_tag)

    if "efficiency" in extras:
        latency = benchmark_latency(model, moe, cfg)
        params = count_parameters(model, moe)
        result["latency"] = latency
        result["parameters"] = {k: float(v) for k, v in params.items()}
        result["flops_per_inference"] = estimate_flops_per_inference(cfg, moe)

    if "routing" in extras and cfg.moe_variant != "none":
        from training.pretrain import collect_validation_routing
        from evaluation.routing_diagnostics import (
            routing_entropy, expert_utilization, objective_routing_diff,
        )

        val_dataset = load_tensorised(
            os.path.join(processed, "val_tensorised.pt"), metadata_matrix,
            max_seq_length=cfg.max_seq_length,
        )
        per_obj = collect_validation_routing(model, moe, val_dataset, cfg, max_batches=16)
        if per_obj:
            all_r = torch.cat(list(per_obj.values()), dim=0)
            diff = objective_routing_diff(per_obj)
            result["routing"] = {
                "overall_entropy_mean": float(routing_entropy(all_r).mean()),
                "overall_utilization": expert_utilization(all_r).tolist(),
                "per_objective_entropy": {
                    k: float(routing_entropy(v).mean()) for k, v in per_obj.items()
                },
                "objective_cosine_similarity": diff["cosine_similarity"].tolist(),
                "objective_names": diff["objective_names"],
            }

    if "personalization" in extras and cfg.personalization_mode == "lora":
        # FR-10: the LoRA arms report the per-user personalization gain over the same
        # global backbone the "global_only" arm is scored on.
        from training.local_train import run_local_personalization

        try:
            gain_report = run_local_personalization(cfg, model_name)
            result["personalization"] = {
                "num_users_evaluated": gain_report["num_users_evaluated"],
                "with_lora": gain_report["with_lora"],
                "without_lora": gain_report["without_lora"],
                "gain": gain_report["gain"],
                # REV-065 / FR-10: the paired per-user NDCGs the Ablation E scatter
                # overlays as its third series.
                "per_user_scores": gain_report.get("per_user_scores", []),
            }
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("  %s: personalization skipped (%s)", label, exc)

    if cfg.system_eval == "cold_start":
        from evaluation.cold_start import run_cold_start_evaluation

        result["cold_start"] = run_cold_start_evaluation(model, moe, cfg, run_tag)

    elif cfg.system_eval == "longtail":
        from evaluation.longtail import run_longtail_evaluation

        result["longtail"] = run_longtail_evaluation(model, moe, cfg, run_tag)

    elif cfg.system_eval == "latency":
        latency = benchmark_latency(model, moe, cfg)
        params = count_parameters(model, moe)
        result["latency"] = latency
        result["parameters"] = {k: float(v) for k, v in params.items()}
        result["flops_per_inference"] = estimate_flops_per_inference(cfg, moe)

    elif cfg.system_eval == "communication":
        from training.federated import run_federated

        try:
            fed = run_federated(cfg, model_name)
            rounds = fed["round_logs"]
            result["communication"] = {
                "num_rounds": len(rounds),
                "final_val_ndcg": fed["final_val_ndcg"],
                "cumulative_kb": sum(r.get("total_round_kb", 0.0) for r in rounds),
                "upload_kb_per_round": rounds[0]["upload_kb"] if rounds else 0.0,
            }
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("  %s: communication study skipped (%s)", label, exc)

    if "gradient" in extras:
        from training.pretrain import attach_training_negatives
        from training.gradnorm import get_anchor_layer_params
        from evaluation.gradient_diagnostics import run_gradient_diagnostics
        from data.sequence_generator import build_dataloader, GlobalNegativePool
        import pickle
        import random as _random

        train_dataset = load_tensorised(
            os.path.join(processed, "train_tensorised.pt"), metadata_matrix,
            max_seq_length=cfg.max_seq_length,
        )
        loader = build_dataloader(train_dataset, cfg, shuffle=True)
        neg_array = GlobalNegativePool.load(
            os.path.join(processed, "global_negative_pool.npy"), set(),
        ).pool
        with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
            hist = pickle.load(f)
        hist_arrays = {int(u): np.asarray(v, dtype=np.int64) for u, v in hist.items()}
        neg_rng = np.random.RandomState(cfg.seed)

        grad_report = run_gradient_diagnostics(
            model, moe, loader, loss_fns,
            get_anchor_layer_params(model, cfg), cfg, f"{run_tag}",
            negative_sampler=lambda b: attach_training_negatives(
                b, neg_array, hist_arrays, cfg, neg_rng, device,
            ),
            metadata_matrix=metadata_matrix,
            augment_rng=_random.Random(cfg.seed),
            write_report=False,
        )
        if grad_report:
            result["gradient"] = grad_report

    result["status"] = "ok"
    return result


def run_ablation(
    config: Config,
    model_name: str,
    study: str = "all",
    train: bool = True,
) -> Dict:
    studies = list(STUDY_CONFIGS.keys()) if study == "all" else [study]
    all_results = {}

    for s in studies:
        logger.info(f"Running ablation study {s}")
        configs = resolve_study_configs(s, config)
        study_results = []
        # REV-084: fingerprint -> the arm that already trained that exact backbone.
        # Scoped to one study, so arms never reuse a model across studies.
        trained_backbones: Dict[str, Dict] = {}

        for cfg in configs:
            label = getattr(cfg, "_study_label", "unknown")
            logger.info(f"  Config: {label}")
            try:
                result = _run_single_config(
                    cfg, model_name, s, config, train,
                    trained_backbones=trained_backbones,
                )
            except Exception as exc:  # one arm failing must not lose the others
                logger.exception("  Config %s failed: %s", label, exc)
                result = {
                    "label": label,
                    "study": s,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "config_diff": _config_diff(cfg, config),
                    "resolved_config": asdict(cfg),
                }
            study_results.append(result)

        all_results[s] = study_results
        _save_study_results(config, model_name, s, study_results)
        _plot_study(config, model_name, s, study_results)
        # REV-065: four requirements name specific figures that the generic grouped bar
        # chart does not produce. They are dispatched alongside it, not instead of it.
        plot_named = _STUDY_PLOTS.get(s)
        if plot_named is not None:
            try:
                plot_named(config, model_name, study_results)
            except Exception as exc:  # a figure failing must not lose the results
                logger.exception("Study %s: named figure failed: %s", s, exc)

    if study == "all":
        _save_summary(config, model_name, all_results)

    return all_results


def _plot_study(config: Config, model_name: str, study: str, results: List[Dict]) -> None:
    """Render the per-study bar chart each FR names (REV-006)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in results if r.get("metrics")]
    if not usable:
        logger.warning("Study %s: no usable results, skipping plot", study)
        return

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    labels = [r["label"] for r in usable]
    metric_names = ["ndcg", "mrr", "recall", "precision"]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    for i, m in enumerate(metric_names):
        vals = [r["metrics"].get(m, {}).get("mean", 0.0) for r in usable]
        ax.bar(x + i * width, vals, width, label=m.upper())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"Score @{config.eval_k}")
    ax.set_title(f"Ablation Study {study} — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(config.plot_dir, f"ablation_{study}_{mn}.png"), dpi=config.plot_dpi,
    )
    plt.close(fig)


def _named_figure_path(config: Config, filename: str) -> str:
    """REV-065: the specs name these figures literally, with no model suffix."""
    os.makedirs(config.plot_dir, exist_ok=True)
    return os.path.join(config.plot_dir, filename)


def _usable(results: List[Dict]) -> List[Dict]:
    return [r for r in results if r.get("metrics")]


def _plot_architecture_progression(
    config: Config, model_name: str, results: List[Dict],
) -> None:
    """FR-6 / T2.2: `architecture_progression.png` -- Ablation A's 6-config progression.

    REV-065: the point of A is the cumulative lift as each component is added, which a
    grouped bar chart of four metrics does not show. This draws the progression as
    connected series over the ordered arms, with the baseline as a reference line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = _usable(results)
    if not usable:
        logger.warning("Ablation A: no usable results, skipping architecture_progression")
        return

    labels = [r["label"] for r in usable]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5.5))

    for metric, marker in (("ndcg", "o"), ("mrr", "s"), ("recall", "^"), ("precision", "d")):
        vals = [r["metrics"].get(metric, {}).get("mean", 0.0) for r in usable]
        ax.plot(x, vals, marker=marker, linewidth=1.8, label=metric.upper())

    baseline_ndcg = usable[0]["metrics"].get("ndcg", {}).get("mean", 0.0)
    ax.axhline(
        baseline_ndcg, color="grey", linestyle="--", alpha=0.6,
        label=f"baseline NDCG={baseline_ndcg:.4f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"Score @{config.eval_k}")
    ax.set_xlabel("Architecture progression")
    ax.set_title(f"Ablation A: Architecture Progression -- {model_name}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_named_figure_path(config, "architecture_progression.png"), dpi=config.plot_dpi)
    plt.close(fig)


def _plot_mmoe_comparison(config: Config, model_name: str, results: List[Dict]) -> None:
    """FR-8 / T2.4: `mmoe_vs_arvmmoe_comparison.png` -- params + metrics + latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = _usable(results)
    if not usable:
        logger.warning("Ablation C: no usable results, skipping mmoe_vs_arvmmoe_comparison")
        return

    labels = [r["label"] for r in usable]
    metric_names = ["ndcg", "mrr", "recall", "precision"]
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(usable), 1)

    fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(13, 5))

    for i, r in enumerate(usable):
        vals = [r["metrics"].get(m, {}).get("mean", 0.0) for m in metric_names]
        ax_m.bar(x + i * width, vals, width, label=r["label"])
    ax_m.set_xticks(x + width * (len(usable) - 1) / 2)
    ax_m.set_xticklabels([m.upper() for m in metric_names])
    ax_m.set_ylabel(f"Score @{config.eval_k}")
    ax_m.set_title("Ranking quality")
    ax_m.legend()

    # C requests "efficiency" extras, so parameters and latency are available. Both are
    # on their own axes because they differ by orders of magnitude.
    ax_p = ax_c
    ax_l = ax_c.twinx()
    cx = np.arange(len(usable))
    params = [r.get("parameters", {}).get("total", 0.0) / 1e6 for r in usable]
    latency = [r.get("latency", {}).get("mean_ms", 0.0) for r in usable]
    ax_p.bar(cx - 0.18, params, 0.36, color="tab:blue", label="Parameters (M)")
    ax_l.bar(cx + 0.18, latency, 0.36, color="tab:orange", label="Latency (ms)")
    ax_p.set_xticks(cx)
    ax_p.set_xticklabels(labels, rotation=15, ha="right")
    ax_p.set_ylabel("Parameters (millions)", color="tab:blue")
    ax_l.set_ylabel("Mean latency (ms)", color="tab:orange")
    ax_p.set_title("Cost")
    handles = ax_p.get_legend_handles_labels()[0] + ax_l.get_legend_handles_labels()[0]
    ax_p.legend(handles, ["Parameters (M)", "Latency (ms)"], loc="upper left")

    fig.suptitle(f"Ablation C: MMoE vs ARV-MMoE -- {model_name}")
    fig.tight_layout()
    fig.savefig(
        _named_figure_path(config, "mmoe_vs_arvmmoe_comparison.png"), dpi=config.plot_dpi,
    )
    plt.close(fig)


def _plot_expert_count(config: Config, model_name: str, results: List[Dict]) -> None:
    """FR-12 / T2.8: `expert_count_ablation.png` -- NDCG left axis, latency right axis.

    REV-065: G's whole point is the quality-vs-cost tradeoff across K, which the generic
    four-metric bar chart cannot show because it never plots latency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = _usable(results)
    if not usable:
        logger.warning("Ablation G: no usable results, skipping expert_count_ablation")
        return

    # Order by the expert count the arm actually ran with, not by label string order --
    # a config-driven grid (REV-067) may be any set of values.
    def _k(result: Dict) -> float:
        resolved = result.get("resolved_config", {})
        if "num_experts" in resolved:
            return float(resolved["num_experts"])
        return float(result.get("study_overrides", {}).get("num_experts", 0))

    usable = sorted(usable, key=_k)
    ks = [_k(r) for r in usable]
    ndcg = [r["metrics"].get("ndcg", {}).get("mean", 0.0) for r in usable]
    ndcg_std = [r["metrics"].get("ndcg", {}).get("std", 0.0) for r in usable]
    latency = [r.get("latency", {}).get("mean_ms", 0.0) for r in usable]

    fig, ax_n = plt.subplots(figsize=(8, 5))
    ax_l = ax_n.twinx()

    ax_n.errorbar(
        ks, ndcg, yerr=ndcg_std, marker="o", color="tab:blue", capsize=3,
        linewidth=1.8, label=f"NDCG@{config.eval_k}",
    )
    ax_l.plot(
        ks, latency, marker="s", color="tab:orange", linestyle="--",
        linewidth=1.8, label="Latency (ms)",
    )

    ax_n.set_xlabel("Number of experts (K)")
    ax_n.set_ylabel(f"NDCG@{config.eval_k}", color="tab:blue")
    ax_l.set_ylabel("Mean latency (ms)", color="tab:orange")
    ax_n.set_xticks(ks)
    ax_n.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:orange")
    ax_n.set_title(f"Ablation G: Expert Count -- {model_name}")
    ax_n.grid(alpha=0.25)

    handles = ax_n.get_legend_handles_labels()[0] + ax_l.get_legend_handles_labels()[0]
    ax_n.legend(handles, [f"NDCG@{config.eval_k}", "Latency (ms)"], loc="lower right")

    fig.tight_layout()
    fig.savefig(_named_figure_path(config, "expert_count_ablation.png"), dpi=config.plot_dpi)
    plt.close(fig)


def _plot_personalization_study(
    config: Config, model_name: str, results: List[Dict],
) -> None:
    """FR-10 / T2.6: overlay Ablation E's `global_lora_arv` arm as a third series.

    REV-065: the scatter `local_train` draws covers one arm only. E exists to compare
    global-only, global+LoRA and global+LoRA+ARV, so the ARV arm is overlaid onto the
    same axes as a distinct series.
    """
    from training.local_train import plot_personalization_scatter

    series = []
    for r in results:
        pairs = r.get("personalization", {}).get("per_user_scores")
        if pairs:
            series.append({"label": r["label"], "results": pairs})

    if not series:
        logger.warning(
            "Ablation E: no arm reported per-user personalization scores, skipping "
            "the multi-series scatter",
        )
        return

    plot_personalization_scatter(
        config, model_name, series,
        filename="global_vs_personalized_ablation_E.png",
        title=f"Ablation E: Personalization -- {model_name}",
    )


#: Study -> the specifically named figure that study's FR requires (REV-065).
_STUDY_PLOTS = {
    "A": _plot_architecture_progression,
    "C": _plot_mmoe_comparison,
    "E": _plot_personalization_study,
    "G": _plot_expert_count,
}


def _save_study_results(config: Config, model_name: str, study: str, results: List[Dict]) -> None:
    os.makedirs(config.ablation_dir, exist_ok=True)
    mn = model_name.replace("-", "_")
    path = os.path.join(config.ablation_dir, f"{study}_{mn}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _save_summary(config: Config, model_name: str, all_results: Dict) -> None:
    os.makedirs(config.ablation_dir, exist_ok=True)
    mn = model_name.replace("-", "_")
    summary = {}
    for study, results in all_results.items():
        summary[study] = {
            "num_configs": len(results),
            "num_ok": sum(1 for r in results if r.get("status") == "ok"),
            "num_failed": sum(1 for r in results if r.get("status") == "failed"),
            "best": max(
                (
                    (r["metrics"]["ndcg"]["mean"], r["label"])
                    for r in results if r.get("metrics", {}).get("ndcg")
                ),
                default=(None, None),
            )[1],
        }
    path = os.path.join(config.ablation_dir, f"summary_{mn}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
