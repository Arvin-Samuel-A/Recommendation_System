import copy
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from configs.config import Config
from utils import is_oom_error

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
        # REV-100: FR-6 ends the progression at "+federated" and T2.2 spells arm 6 out as
        # "(6) federated ARV-MMoE+GradNorm+LoRA", but the arm ran `run_local_personalization`
        # only -- nothing in study A ever called `run_federated`, so the progression's
        # final point was labelled as the federated system while measuring single-machine
        # LoRA. `system_eval="communication"` routes `_run_single_config` into
        # `run_federated` (rounds, final val NDCG, cumulative KB); `personalization_mode`
        # stays "lora" because T2.2 names both, and federated training here IS LoRA-based.
        {"label": "arv_mmoe_gradnorm_lora", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "arv_mmoe", "use_gradnorm": True, "personalization_mode": "lora", "system_eval": "communication"},
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


def parse_arms(text: Optional[str]) -> Optional[List[str]]:
    """`--arms a,b,a` -> ["a", "b"] (order kept, duplicates dropped); None -> None.

    specs/ablation-run FR-14 / D15. An empty selection is an error, not "all arms".
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        items = [str(t).strip() for t in text]
    else:
        items = [t.strip() for t in str(text).split(",")]
    arms: List[str] = []
    for item in items:
        if item and item not in arms:
            arms.append(item)
    if not arms:
        raise ValueError("--arms was given but names no arm (e.g. --arms input_only,objective_conditioned)")
    return arms


def select_study_configs(
    study: str, base_config: Config, arms: Optional[List[str]],
) -> List[Config]:
    """The resolved arm configs `run_ablation` runs: all of them, or the `--arms` subset.

    specs/ablation-run D15: validated BEFORE anything touches data, so a typo costs nothing.
    Labels are only unique within a study (e.g. `arv_mmoe` is in A and C), hence `--arms`
    needs exactly one study.
    """
    if arms is None:
        return resolve_study_configs(study, base_config)
    arms = parse_arms(arms)
    if study == "all":
        raise ValueError(
            "--arms needs exactly one --study (labels are only unique within a study); "
            "got --study all"
        )
    configs = resolve_study_configs(study, base_config)
    by_label = {cfg._study_label: cfg for cfg in configs}
    labels = [cfg._study_label for cfg in configs]
    unknown = [a for a in arms if a not in by_label]
    if unknown:
        raise ValueError(
            f"Study {study} has no arm(s) {unknown}; valid labels: {labels}"
        )
    return [by_label[a] for a in arms]


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
#: extras dispatch, and `ablation_eval_examples` / `ablation_eval_seed` only choose the
#: test rows scored after training (eval-integrity FR-6.4).
#: Anything not listed here is treated as training-relevant -- the safe default, since a
#: missing entry only costs a redundant run, while a wrong entry would silently make two
#: genuinely different arms share one model.
POST_TRAINING_ONLY_FIELDS = frozenset({
    "personalization_mode",
    "system_eval",
    "ablation_eval_examples",
    "ablation_eval_seed",
})

#: Per-arm output namespaces, which carry no training semantics.
_PATH_FIELDS = frozenset({
    "checkpoint_dir", "output_dir", "log_dir", "plot_dir", "ablation_dir",
})

#: Fields that choose WHICH runs to launch, not how one trains. `final_seeds` is read
#: only by the multi-seed loop; each run's own `seed` stays in the fingerprint. Keeping
#: it in would stop a multi-seed checkpoint from matching an ablation arm whenever the
#: seed list differs, even though the trained model is identical.
_RUN_SELECTION_FIELDS = frozenset({"final_seeds"})

#: specs/ablation-run FR-13 (design D4/D13): how a run is EXECUTED, not what it trains.
#: Worker processes and pinned memory yield the identical batch sequence (tested), and
#: `resume_training` only decides whether `last.pt` is picked up. `persistent_workers`
#: stays IN the fingerprint on purpose: a persistent iterator is reset between epochs
#: without redrawing its base seed, which consumes the loader generator differently and
#: so changes the data order from epoch 2 on.
_RUNTIME_ONLY_FIELDS = frozenset({"num_workers", "pin_memory", "resume_training"})


#: Bump when a code fix changes what training produces for an unchanged config, so
#: backbones trained before the fix are never reused (specs/eval-integrity FR-4).
#: 1 = implicit (<= c094b2c). 2 = eval-integrity FR-3: per-forward autocast regions with
#: the weight cache off (mixed-precision runs before it forwarded stale bf16 casts).
TRAINING_SEMANTICS_VERSION = 2


def _training_fields(cfg_dict: dict) -> dict:
    """The training-relevant subset of an `asdict(Config)` (the fingerprint's field set).

    Also used by the rescore identity check (eval-integrity FR-8.9).
    """
    return {
        k: v for k, v in cfg_dict.items()
        if k not in POST_TRAINING_ONLY_FIELDS and k not in _PATH_FIELDS
        and k not in _RUN_SELECTION_FIELDS and k not in _RUNTIME_ONLY_FIELDS
    }


def _training_fingerprint(cfg: Config) -> str:
    """Hash of everything about `cfg` that actually changes the trained model.

    REV-084: Ablation A's arms 5/6 and E's `global_only`/`global_lora` differ only in
    `personalization_mode`, which `run_pretrain` never reads. Under label-namespaced
    checkpoints each pair trained two bit-identical models -- two full pretraining runs
    wasted per pair, at `pretrain_epochs=20` over 7.8M examples -- and produced identical
    `metrics`, which reads as a null result for personalization rather than as what it
    is: the two arms sharing a backbone by construction.

    eval-integrity FR-4: the payload also carries `TRAINING_SEMANTICS_VERSION`, so a
    training-correctness fix changes every fingerprint and pre-fix backbones are never
    reused. Code that needs an old backbone locates it by the fingerprint recorded in its
    arm result, never by recomputing it.
    """
    fields = _training_fields(asdict(cfg))
    fields["_training_semantics"] = TRAINING_SEMANTICS_VERSION
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


#: Marker written only once a backbone has FINISHED training. `run_pretrain` writes
#: `best.pt` after its first epoch, so a crashed run leaves a checkpoint that must never
#: be mistaken for a trained model; the marker is what makes a store entry reusable.
BACKBONE_MARKER = "backbone.json"


def backbone_store_dir(checkpoint_root: str, fingerprint: str) -> str:
    """Checkpoint namespace for one training fingerprint, shared by every study.

    Keyed by fingerprint rather than by `{study}_{label}`, so any arm in any study, in
    any process, and on any re-run that trains the same model finds the same directory.
    The fingerprint covers the seed and the model dimensions, so seeds and model sizes
    never collide.
    """
    return os.path.join(checkpoint_root, "ablations", "by_fingerprint", fingerprint)


def training_log_dir(checkpoint_root: str, fingerprint: str) -> str:
    """Where one fingerprint's training logs go: inside its store directory (REV-002)."""
    return os.path.join(backbone_store_dir(checkpoint_root, fingerprint), "logs")


def find_backbone(checkpoint_root: str, fingerprint: str) -> Optional[Dict]:
    """The completed backbone registered for `fingerprint`, or None."""
    path = os.path.join(backbone_store_dir(checkpoint_root, fingerprint), BACKBONE_MARKER)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        entry = json.load(f)
    if not os.path.exists(entry["checkpoint_path"]):
        logger.warning(
            "Backbone %s is registered but %s is missing -- it will be retrained",
            fingerprint, entry["checkpoint_path"],
        )
        return None
    return entry


def register_backbone(
    checkpoint_root: str,
    fingerprint: str,
    trained_cfg: Config,
    trained_by: str,
    train_result: Optional[Dict] = None,
) -> Dict:
    """Record a finished backbone so later arms load it instead of retraining it.

    The entry records `trained_cfg.checkpoint_dir`, not just the file, because the
    personalization and federated stages re-read `model_checkpoint_dir()/best.pt` from
    the arm's config. It may live outside the store: the multi-seed stage registers its
    per-seed checkpoint, so ablation arms on the base config reuse that model.
    """
    entry = {
        "fingerprint": fingerprint,
        "checkpoint_dir": trained_cfg.checkpoint_dir,
        "checkpoint_path": os.path.join(trained_cfg.model_checkpoint_dir(), "best.pt"),
        "trained_by": trained_by,
    }
    for key in ("best_val_ndcg", "best_epoch"):
        if train_result and key in train_result:
            entry[key] = train_result[key]
    store = backbone_store_dir(checkpoint_root, fingerprint)
    os.makedirs(store, exist_ok=True)
    tmp = os.path.join(store, BACKBONE_MARKER + ".tmp")
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=2)
    os.replace(tmp, os.path.join(store, BACKBONE_MARKER))  # atomic: never half-written
    return entry


def resolve_dataset_fields(cfg: Config, stats_dir: Optional[str] = None):
    """Set `num_items`, `num_users`, `vocab_size`, `feature_dim` from a processed dir.

    specs/ablation-run D20: the ONE place these dataset-derived fields are resolved for an
    ablation arm, shared by `_run_single_config` and the packager's planner, so a plan's
    training fingerprints are exactly the ones the launched run will compute. `stats_dir`
    (default `cfg.processed_data_dir`) lets the packager read a local copy while the config
    keeps the remote's relative path. Returns the aligned metadata matrix (or None).
    """
    from data.sequence_generator import load_metadata_matrix

    directory = stats_dir or cfg.processed_data_dir
    with open(os.path.join(directory, "dataset_stats.json")) as f:
        stats = json.load(f)
    cfg.num_items = stats["num_items"]
    cfg.num_users = stats["num_users"]
    cfg.vocab_size = cfg.num_items + 2
    # REV-058/REV-072: one loader, so the (vocab, F) alignment applies uniformly.
    metadata_matrix = load_metadata_matrix(directory, cfg)
    if metadata_matrix is not None:
        cfg.feature_dim = metadata_matrix.shape[-1]
    return metadata_matrix


def select_eval_rows(total: int, n: int, seed: int) -> Tuple[Optional[np.ndarray], Dict]:
    """Which test rows an arm is scored on (specs/eval-integrity FR-6, design D7).

    `n <= 0` or `n >= total`: every row (`None`, method "full"). Otherwise a seeded
    uniform random sample of `n` rows without replacement, in sorted order -- never a
    prefix of the user-sorted split. The descriptor is recorded in the arm result.
    """
    if n <= 0 or n >= total:
        return None, {"method": "full", "num_examples": int(total), "total_rows": int(total),
                      "seed": None}
    idx = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    return idx, {
        "method": "random",
        "num_examples": int(n),
        "total_rows": int(total),
        "seed": int(seed),
        "indices_sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
    }


def _runtime_record(model, cfg: Config, device: torch.device) -> Dict:
    """Backend, device and precision an arm was evaluated (and trained) under (D2)."""
    import platform

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine() or device.type
    return {
        "attention_backend": model.backend.backend_name,
        "attention_fallback_reason": model.backend.fallback_reason,
        "device": str(device),
        "device_name": device_name,
        "dtype": cfg.dtype,
        "mixed_precision": cfg.mixed_precision,
    }


#: The per-fingerprint lock file inside a store directory (FR-10).
BACKBONE_LOCK = ".train.lock"
#: How often a waiting process re-tries the lock.
BACKBONE_LOCK_POLL_SECONDS = 5.0


@contextmanager
def file_lock(
    path: str,
    timeout: Optional[float] = None,
    poll: float = 5.0,
    on_wait=None,
):
    """Exclusive `flock` on `path` (created if needed); the holder writes "pid host time".

    specs/ablation-run D16: the loop `backbone_lock` used, extracted so the per-study lock
    shares it. `flock` locks die with their process, so a killed holder never leaves a stale
    lock. `on_wait(holder_line)` is called once, the first time the lock is busy.
    `timeout=None` waits forever; otherwise `TimeoutError` after `timeout` seconds.
    """
    import errno
    import fcntl
    import socket
    import time
    from datetime import datetime, timezone

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        start = time.monotonic()
        logged = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    raise
            elapsed = time.monotonic() - start
            if not logged:
                try:
                    with open(path) as f:
                        holder = f.read().strip() or "unknown"
                except OSError:
                    holder = "unknown"
                if on_wait is not None:
                    on_wait(holder)
                logged = True
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"Timed out after {timeout:g} s waiting for the lock {path}"
                )
            wait = poll if timeout is None else min(poll, max(0.01, timeout - elapsed))
            time.sleep(wait)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        os.write(fd, f"{os.getpid()} {socket.gethostname()} {stamp}\n".encode())
        os.fsync(fd)
        try:
            yield path
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def backbone_lock(
    checkpoint_root: str,
    fingerprint: str,
    timeout: Optional[float] = None,
    poll: Optional[float] = None,
):
    """Exclusive OS lock on one fingerprint's store directory (specs/ablation-run D10).

    `flock` locks are released by the kernel when their process dies, so an OOM-killed or
    `kill -9`-ed trainer never leaves a stale lock (unlike an O_EXCL lock file). Waiting is
    logged once, with the holder's "pid host time" line. `timeout=None` waits forever;
    otherwise `TimeoutError` after `timeout` seconds.
    """
    if poll is None:
        poll = BACKBONE_LOCK_POLL_SECONDS
    store = backbone_store_dir(checkpoint_root, fingerprint)
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, BACKBONE_LOCK)

    def log_wait(holder: str) -> None:
        logger.info(
            "Backbone %s is being trained by another process (pid host time: %s) "
            "-- waiting for its lock at %s", fingerprint, holder, path,
        )

    with file_lock(path, timeout=timeout, poll=poll, on_wait=log_wait) as locked:
        yield locked


def _run_single_config(
    cfg: Config,
    model_name: str,
    study: str,
    base_config: Config,
    train: bool,
    scoped_outputs: bool = False,
) -> Dict:
    """Train (or load), evaluate and diagnose one ablation config.

    `scoped_outputs` (specs/ablation-run FR-17, set under `--arms`): the arm's post-training
    side outputs go to `{log_dir}/arms/{study}_{label}/` and `{plot_dir}/arms/{study}_{label}/`
    so arms running in different processes never write the same file.

    REV-006: the runner never executed an experiment -- it built a
    `{"label", "config_diff"}` dict and appended it. No model was constructed, nothing
    was trained or evaluated, and no plot was rendered, so all eight studies produced no
    results.
    """
    from training.pretrain import run_pretrain, build_model, load_checkpoint, _build_loss_fns
    from evaluation.metrics import (
        EVAL_PROTOCOL, evaluate_model, evaluate_popularity, load_item_popularity,
    )
    from training.pretrain import POPULARITY_EVAL_BATCH
    from evaluation.flops import (
        benchmark_latency, count_parameters, estimate_flops_per_inference,
    )
    from data.sequence_generator import (
        load_tensorised, load_eval_candidates, load_metadata_matrix,
    )
    from utils import get_device, artifact_source_revision, set_seed

    # Every arm starts from the configured seed, so its result does not depend on which
    # arms ran earlier in this process (H2_seq4 scored 0.4154 after the earlier H arms and
    # 0.4018 alone). It runs before any model is built -- weight init draws from the global
    # generators -- and also for arms that reuse a stored backbone, whose post-training
    # steps (LoRA init, federated client choice, eval negatives) draw random numbers too.
    # A resumed `run_pretrain` restores its checkpoint's RNG state over this, as it should.
    set_seed(cfg.seed)

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
    if scoped_outputs:
        cfg.log_dir = os.path.join(cfg.log_dir, "arms", f"{study}_{label}")
        cfg.plot_dir = os.path.join(cfg.plot_dir, "arms", f"{study}_{label}")

    processed = cfg.processed_data_dir
    metadata_matrix = resolve_dataset_fields(cfg)

    # Logs and diagnostics stay namespaced per arm; the CHECKPOINT is namespaced by
    # training fingerprint instead (see `backbone_store_dir`).
    run_tag = f"{study}_{label}"

    # REV-084: arms that differ only in post-training fields train the SAME model. The
    # store is keyed by fingerprint and shared across studies, processes and re-runs, so
    # e.g. the base config trains once for B, C, D, F, G and H rather than once per
    # study, and the multi-seed stage's per-seed checkpoint is reused when it matches.
    # Reuse is recorded in the result so identical `metrics` between arms read as a
    # shared backbone rather than as a null result for the dimension they isolate.
    fingerprint = _training_fingerprint(cfg)
    result["training_fingerprint"] = fingerprint
    cfg.checkpoint_dir = backbone_store_dir(base_config.checkpoint_dir, fingerprint)
    ckpt_path = os.path.join(cfg.model_checkpoint_dir(), "best.pt")

    shared = find_backbone(base_config.checkpoint_dir, fingerprint)
    trained_here = False
    if shared is None and train:
        # specs/ablation-run FR-10 / D10: one trainer per fingerprint per machine. A
        # process that waited re-checks the store after acquiring the lock, so it reuses
        # the backbone the holder registered instead of training it again.
        with backbone_lock(base_config.checkpoint_dir, fingerprint):
            shared = find_backbone(base_config.checkpoint_dir, fingerprint)
            if shared is None:
                # Review REV-002: the training logs (per-epoch log, GradNorm JSONL, routing
                # report) live in the fingerprint's store directory, next to its checkpoints,
                # so a resume can only ever truncate/append ITS OWN records -- never splice
                # in another arm's lines from a shared `pretrain_{model}.log`.
                train_log_dir = training_log_dir(base_config.checkpoint_dir, fingerprint)
                result["training_log_dir"] = train_log_dir
                arm_log_dir = cfg.log_dir
                cfg.log_dir = train_log_dir
                try:
                    train_result = run_pretrain(cfg, model_name)
                finally:
                    cfg.log_dir = arm_log_dir
                result["best_val_ndcg"] = train_result["best_val_ndcg"]
                result["best_epoch"] = train_result["best_epoch"]
                register_backbone(
                    base_config.checkpoint_dir, fingerprint, cfg, run_tag, train_result,
                )
                trained_here = True
    if shared is not None:
        cfg.checkpoint_dir = shared["checkpoint_dir"]
        ckpt_path = shared["checkpoint_path"]
        result["shared_backbone_with"] = shared["trained_by"]
        for key in ("best_val_ndcg", "best_epoch"):
            if key in shared:
                result[key] = shared[key]
        logger.info(
            "  %s: training-relevant config matches '%s' -- reusing its backbone "
            "instead of retraining an identical model", label, shared["trained_by"],
        )
    elif trained_here:
        pass
    else:
        # `--no_train`: only a COMPLETED backbone is evaluated. A bare `best.pt` without
        # the marker is a run that did not finish.
        result["status"] = "skipped_no_checkpoint"
        result["checkpoint_path"] = ckpt_path
        logger.warning(
            "  %s: no completed backbone for fingerprint %s and train=False -- skipping",
            label, fingerprint,
        )
        return result

    device = get_device(cfg)
    model, moe = build_model(cfg)
    loss_fns = _build_loss_fns(cfg, device)
    arm_ckpt = load_checkpoint(ckpt_path, model, moe, loss_fns=loss_fns)
    # specs/ablation-run FR-2 / AC-30: the kernel and device this arm ACTUALLY ran on, so a
    # reviewer can check that every arm of a study shared one backend.
    result["runtime"] = _runtime_record(model, cfg, device)
    # AC-09: an ablation JSON compares arms, so an unstamped one cannot prove its arms
    # were produced by the same code. Each arm echoes the source_revision of the
    # checkpoint IT was evaluated from, alongside `resolved_config` and `config_diff`.
    result["source_revision"] = artifact_source_revision(arm_ckpt)

    # Evaluate on the FIXED stored candidate sets, so every ablation arm is scored
    # against identical candidates (data-pipeline REV-041).
    test_dataset = load_tensorised(
        os.path.join(processed, "test_tensorised.pt"), metadata_matrix,
        max_seq_length=cfg.max_seq_length,
    )
    cand_matrix = load_eval_candidates(
        os.path.join(processed, "test_candidates.npy"), mmap=True,
    )
    if cand_matrix.shape[0] != len(test_dataset):
        raise RuntimeError(
            f"test candidate sets have {cand_matrix.shape[0]} rows but the test split has "
            f"{len(test_dataset)}; re-run --stage preprocess"
        )
    # specs/eval-integrity FR-6: the FULL test split by default; a seeded random sample
    # only when `ablation_eval_examples > 0`. Only the selected candidate rows are read
    # from the memory-mapped matrix.
    idx, test_eval = select_eval_rows(
        len(test_dataset), cfg.ablation_eval_examples, cfg.ablation_eval_seed,
    )
    # REV-063: hand `evaluate_model` the stacked tensors and let it batch, rather than
    # building n per-example device tensors and forwarding a batch of 1 each.
    if idx is None:
        sequences = test_dataset.sequences
        candidates = torch.as_tensor(np.array(cand_matrix), dtype=torch.long)
    else:
        sequences = test_dataset.sequences[torch.from_numpy(idx)]
        candidates = torch.as_tensor(np.array(cand_matrix[idx]), dtype=torch.long)

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
    result["num_eval_examples"] = test_eval["num_examples"]
    result["test_eval"] = test_eval
    # FR-7: the popularity-only floor on exactly these rows and candidates.
    result["popularity_reference"] = evaluate_popularity(
        candidates, load_item_popularity(processed, cfg.vocab_size), k=cfg.eval_k,
        batch_size=POPULARITY_EVAL_BATCH,
    )
    result["eval_protocol"] = dict(EVAL_PROTOCOL)

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


# ---------------- per-arm persistence (specs/ablation-run FR-11, design D11) ----------------


def _results_root(config: Config) -> str:
    """Seed-namespaced results directory, so runs with different seeds never collide."""
    return os.path.join(config.ablation_dir, f"seed_{config.seed}")


def _arm_path(config: Config, study: str, model_name: str, label: str) -> str:
    mn = model_name.replace("-", "_")
    return os.path.join(_results_root(config), "arms", f"{study}_{mn}", f"{label}.json")


def _arm_key(cfg: Config, study: str, label: str, model_name: str) -> str:
    """Identity of one arm's RESULT: its resolved config (minus paths and runtime-only
    fields), the study, label, model, and the code that would produce it."""
    from utils import get_source_revision

    revision = get_source_revision()
    payload = {
        "config": {
            k: v for k, v in asdict(cfg).items()
            if k not in _PATH_FIELDS and k not in _RUNTIME_ONLY_FIELDS
        },
        "study": study,
        "label": label,
        "model": model_name,
        "source": {k: revision.get(k) for k in ("commit", "source_hash", "dirty")},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_persisted_arm(path: str, arm_key: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable persisted arm %s: %s", path, exc)
        return None
    if stored.get("status") != "ok" or stored.get("arm_key") != arm_key:
        return None
    return stored


def _write_json_atomic(path: str, payload) -> None:
    from utils import atomic_write_json

    atomic_write_json(path, payload)


def _current_revision() -> Dict:
    """This process's `source_revision` (looked up at call time, so tests can patch it)."""
    import utils

    return utils.get_source_revision()


#: How often a process waiting for a study's output lock re-tries it.
STUDY_LOCK_POLL_SECONDS = 0.5


def _study_lock_path(config: Config, study: str, model_name: str) -> str:
    """The per-study lock file (D16). Kept OUT of `arms/` so that directory holds only
    arm results."""
    mn = model_name.replace("-", "_")
    return os.path.join(_results_root(config), ".locks", f"{study}_{mn}.study.lock")


@contextmanager
def _study_lock(config: Config, study: str, model_name: str):
    path = _study_lock_path(config, study, model_name)

    def log_wait(holder: str) -> None:
        logger.info(
            "Study %s outputs are being written by another process (pid host time: %s) "
            "-- waiting for %s", study, holder, path,
        )

    with file_lock(path, poll=STUDY_LOCK_POLL_SECONDS, on_wait=log_wait) as locked:
        yield locked


def _write_study_outputs_unlocked(
    config: Config, model_name: str, study: str, results: List[Dict],
) -> None:
    _save_study_results(config, model_name, study, results)
    _plot_study(config, model_name, study, results)
    # REV-065: four requirements name specific figures that the generic grouped bar
    # chart does not produce. They are dispatched alongside it, not instead of it.
    plot_named = _STUDY_PLOTS.get(study)
    if plot_named is not None:
        try:
            plot_named(config, model_name, results)
        except Exception as exc:  # a figure failing must not lose the results
            logger.exception("Study %s: named figure failed: %s", study, exc)


# ---------------- assessment and collection (specs/ablation-run FR-15, design D17) ----------


@dataclass
class StudyAssessment:
    """What the persisted arm files of one study amount to (D17)."""

    study: str
    labels: List[str]                                   # expected, study order
    ok: Dict[str, dict] = field(default_factory=dict)
    failed: Dict[str, dict] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    inconsistent: List[str] = field(default_factory=list)
    written: bool = False                               # set by collect_study

    @property
    def complete(self) -> bool:
        return not (self.missing or self.stale or self.failed or self.inconsistent)

    def describe(self) -> str:
        return (
            f"study {self.study}: {len(self.ok)}/{len(self.labels)} arms complete; "
            f"missing {self.missing} failed {sorted(self.failed)} stale {self.stale}"
            + (f" inconsistent {self.inconsistent}" if self.inconsistent else "")
        )


def _consistency_problems(study: str, records: Dict[str, dict]) -> List[str]:
    """The one-machine and same-base-config rules over a study's persisted records."""
    problems: List[str] = []
    for key in ("attention_backend", "device_name"):
        values = {
            label: r["runtime"].get(key) for label, r in records.items()
            if isinstance(r.get("runtime"), dict)
        }
        if len(set(map(str, values.values()))) > 1:
            problems.append(f"runtime.{key} differs: {values}")
    overridden = set()
    for r in records.values():
        overridden.update((r.get("study_overrides") or {}).keys())
    excluded = overridden | _PATH_FIELDS | _RUNTIME_ONLY_FIELDS
    configs = {
        label: r["resolved_config"] for label, r in records.items()
        if isinstance(r.get("resolved_config"), dict)
    }
    keys = sorted(set().union(*[set(c) for c in configs.values()])) if configs else []
    for key in keys:
        if key in excluded:
            continue
        values = {label: c.get(key) for label, c in configs.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            problems.append(f"resolved_config.{key} differs: {values}")
    return problems


def assess_study(config: Config, model_name: str, study: str) -> StudyAssessment:
    """Classify every expected arm of `study` from the arm files on disk (D17).

    No file -> missing; a record produced by other code (`code_revision.source_hash`
    differs from this process's) -> stale; `status == "failed"` -> failed; else ok. Ok and
    failed records must agree on backend/device and on every non-overridden config field.
    """
    labels = [cfg._study_label for cfg in resolve_study_configs(study, config)]
    current_hash = _current_revision().get("source_hash")
    assessment = StudyAssessment(study=study, labels=labels)
    for label in labels:
        path = _arm_path(config, study, model_name, label)
        if not os.path.exists(path):
            assessment.missing.append(label)
            continue
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unreadable arm file %s (%s) -- counted as missing", path, exc)
            assessment.missing.append(label)
            continue
        if (record.get("code_revision") or {}).get("source_hash") != current_hash:
            assessment.stale.append(label)
        elif record.get("status") == "failed":
            assessment.failed[label] = record
        elif record.get("status") == "ok":
            assessment.ok[label] = record
        else:
            assessment.missing.append(label)
    assessment.inconsistent = _consistency_problems(
        study, {**assessment.ok, **assessment.failed},
    )
    return assessment


def collect_study(
    config: Config, model_name: str, study: str, allow_incomplete: bool = False,
) -> StudyAssessment:
    """Assemble a study's JSON and plots from its persisted arm files (D17).

    Under the study lock, so it sees every arm file persisted before the lock was granted.
    Inconsistent studies are never written; incomplete ones only with `allow_incomplete`
    (missing/stale arms become `{"status": "missing"}` placeholders, failed ones keep
    their record; plots use the ok arms).
    """
    with _study_lock(config, study, model_name):
        assessment = assess_study(config, model_name, study)
        if assessment.inconsistent:
            logger.error(
                "%s -- refusing to assemble an inconsistent study", assessment.describe(),
            )
            return assessment
        if not assessment.complete and not allow_incomplete:
            return assessment
        results = []
        for label in assessment.labels:
            if label in assessment.ok:
                record = dict(assessment.ok[label])
            elif label in assessment.failed:
                record = dict(assessment.failed[label])
            else:
                record = {"label": label, "study": study, "status": "missing"}
            record.pop("reused_from_previous_run", None)
            results.append(record)
        _write_study_outputs_unlocked(config, model_name, study, results)
        assessment.written = True
        if not assessment.complete:
            logger.warning("Assembled an INCOMPLETE %s (--allow_incomplete)", assessment.describe())
        return assessment


def write_study_outputs(
    config: Config, model_name: str, study: str, results: List[Dict],
) -> None:
    """The ONE writer of a study's JSON and plots, under the per-study lock (FR-15, D16).

    Every path that produces study-level outputs goes through here, so concurrent writers
    are serialised and the (atomic) JSON is never torn or interleaved.
    """
    with _study_lock(config, study, model_name):
        _write_study_outputs_unlocked(config, model_name, study, results)


def run_ablation(
    config: Config,
    model_name: str,
    study: str = "all",
    train: bool = True,
    arms: Optional[List[str]] = None,
) -> Dict:
    # specs/ablation-run FR-14 / D15: an `--arms` selection is validated first -- before
    # `check_processed_dir`, any data load or any training.
    if arms is not None:
        arms = parse_arms(arms)
        if study not in STUDY_CONFIGS:
            if study == "all":
                select_study_configs(study, config, arms)  # raises the one-study error
            raise ValueError(f"Unknown study '{study}'. Choose from {list(STUDY_CONFIGS.keys())}")
        selected = select_study_configs(study, config, arms)

    # specs/ablation-run FR-7 / AC-20: refuse a mismatched user sample before any arm.
    from data.preprocessing import check_processed_dir
    check_processed_dir(config)

    studies = list(STUDY_CONFIGS.keys()) if study == "all" else [study]
    all_results = {}

    for s in studies:
        logger.info(f"Running ablation study {s}")
        configs = selected if arms is not None else resolve_study_configs(s, config)
        study_results = []

        for cfg in configs:
            label = getattr(cfg, "_study_label", "unknown")
            logger.info(f"  Config: {label}")
            # FR-11: computed BEFORE `_run_single_config`, which resolves dataset fields
            # and the checkpoint namespace into `cfg`.
            arm_key = _arm_key(cfg, s, label, model_name)
            arm_path = _arm_path(config, s, model_name, label)
            if config.resume_training:
                persisted = _load_persisted_arm(arm_path, arm_key)
                if persisted is not None:
                    persisted["reused_from_previous_run"] = True
                    logger.info(
                        "  %s: reusing the persisted result %s (arm_key %s)",
                        label, arm_path, arm_key,
                    )
                    study_results.append(persisted)
                    continue
            # FR-15: the code that produced the record, for staleness checks at collection.
            code_revision = _current_revision()
            resolved_before = asdict(cfg)
            try:
                # FR-17: arm-scoped side outputs only under --arms (paths unchanged otherwise).
                extra = {"scoped_outputs": True} if arms is not None else {}
                result = _run_single_config(
                    cfg, model_name, s, config, train, **extra,
                )
                result["arm_key"] = arm_key
                result["code_revision"] = code_revision
                if result.get("status") == "ok":
                    _write_json_atomic(arm_path, result)
            except Exception as exc:  # one arm failing must not lose the others
                # specs/ablation-run FR-9: an OOM is not this arm's fault and recording it
                # as a failure would lose the arm; let the process exit 75 and retry.
                if is_oom_error(exc):
                    raise
                logger.exception("  Config %s failed: %s", label, exc)
                result = {
                    "label": label,
                    "study": s,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "config_diff": _config_diff(cfg, config),
                    "study_overrides": dict(getattr(cfg, "_study_overrides", {})),
                    "resolved_config": resolved_before,
                    "arm_key": arm_key,
                    "code_revision": code_revision,
                }
                # FR-15: failed arms are persisted too (so a collector can name them); reuse
                # still requires status "ok", so they are retried on a re-run. A failed write
                # (e.g. disk full) must not skip the remaining arms (REV-013).
                try:
                    _write_json_atomic(arm_path, result)
                except OSError as write_exc:
                    logger.error("  Could not persist failed arm %s: %s", label, write_exc)
            study_results.append(result)

        all_results[s] = study_results
        if arms is None:
            write_study_outputs(config, model_name, s, study_results)
        else:
            # FR-15 auto-collect: this process persisted its arms BEFORE asking for the
            # study lock, so whichever process persists the last missing arm sees a
            # complete study and writes the study outputs. Never fails the run.
            try:
                assessment = collect_study(config, model_name, s)
                if assessment.written:
                    logger.info("Study %s complete -- study JSON and plots written", s)
                else:
                    logger.info(
                        "%s -- study JSON and plots deferred to the process that completes "
                        "the study or to --stage ablation_collect", assessment.describe(),
                    )
            except Exception as exc:  # noqa: BLE001 - the arms themselves succeeded
                logger.error("Study %s: auto-collect failed: %s", s, exc)

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

    # eval-integrity FR-7.4: the popularity-only floor, so a bar is read against it.
    pop_ndcg = next(
        (
            r["popularity_reference"]["ndcg"]["mean"] for r in usable
            if ((r.get("popularity_reference") or {}).get("ndcg") or {}).get("mean")
            is not None
        ),
        None,
    )
    if pop_ndcg is not None:
        ax.axhline(pop_ndcg, linestyle="--", color="0.3", label=f"popularity (NDCG@{config.eval_k})")

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
    # REV-099: this read `parameters.total`, a key `evaluation/flops.py::count_parameters`
    # has never returned -- the real key is `total_parameters` -- so `.get("total", 0.0)`
    # silently evaluated to 0.0 and BOTH arms' parameter bars were plotted at zero height,
    # in the figure whose job is to substantiate the parameter-matched claim. FR-8 also
    # names expert params and router/gate params, both already in the same dict and
    # neither previously plotted.
    ax_p = ax_c
    ax_l = ax_c.twinx()
    cx = np.arange(len(usable))
    param_series = [
        ("total_parameters", "Total params (M)", "tab:blue"),
        ("expert_parameters", "Expert params (M)", "tab:cyan"),
        ("router_parameters", "Router/gate params (M)", "tab:purple"),
    ]
    latency = [r.get("latency", {}).get("mean_ms", 0.0) for r in usable]
    offsets = [-0.30, -0.10, 0.10]
    for (key, series_label, color), offset in zip(param_series, offsets):
        vals = [r.get("parameters", {}).get(key, 0.0) / 1e6 for r in usable]
        ax_p.bar(cx + offset, vals, 0.19, color=color, label=series_label)
    ax_l.bar(cx + 0.30, latency, 0.19, color="tab:orange", label="Latency (ms)")
    ax_p.set_xticks(cx)
    ax_p.set_xticklabels(labels, rotation=15, ha="right")
    ax_p.set_ylabel("Parameters (millions)", color="tab:blue")
    ax_l.set_ylabel("Mean latency (ms)", color="tab:orange")
    ax_p.set_title("Cost")
    handles = ax_p.get_legend_handles_labels()[0] + ax_l.get_legend_handles_labels()[0]
    ax_p.legend(
        handles,
        [label for _, label, _ in param_series] + ["Latency (ms)"],
        loc="upper left", fontsize="small",
    )

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
    # specs/ablation-run FR-11: under `{ablation_dir}/seed_{seed}/`.
    mn = model_name.replace("-", "_")
    path = os.path.join(_results_root(config), f"{study}_{mn}.json")
    _write_json_atomic(path, results)


def _popularity_ndcg(results: List[Dict]) -> Optional[float]:
    """The popularity-only NDCG of a study (eval-integrity FR-7.3): from the first ok arm
    that carries one -- every ok arm of a study is scored on the same test rows."""
    for r in results:
        if r.get("status") == "ok":
            ndcg = ((r.get("popularity_reference") or {}).get("ndcg") or {}).get("mean")
            if ndcg is not None:
                return float(ndcg)
    return None


def _save_summary(
    config: Config,
    model_name: str,
    all_results: Dict,
    incomplete: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(_results_root(config), exist_ok=True)
    mn = model_name.replace("-", "_")
    summary = {}
    for study, results in all_results.items():
        best_ndcg, best_label = max(
            (
                (r["metrics"]["ndcg"]["mean"], r["label"])
                for r in results if r.get("metrics", {}).get("ndcg")
            ),
            default=(None, None),
        )
        pop_ndcg = _popularity_ndcg(results)
        summary[study] = {
            "num_configs": len(results),
            "num_ok": sum(1 for r in results if r.get("status") == "ok"),
            "num_failed": sum(1 for r in results if r.get("status") == "failed"),
            "best": best_label,
            # eval-integrity FR-7.3: a gain is read against the popularity floor.
            "best_ndcg": best_ndcg,
            "popularity_reference_ndcg": pop_ndcg,
            "best_minus_popularity_ndcg": (
                best_ndcg - pop_ndcg if best_ndcg is not None and pop_ndcg is not None
                else None
            ),
        }
    # specs/ablation-run FR-15: `ablation_collect --study all` marks studies it could not
    # assemble instead of silently leaving them out.
    for study, entry in (incomplete or {}).items():
        summary[study] = {"complete": False, **entry}
    # e.g. the rescore's `rescore_caveat` (eval-integrity FR-8.7), merged at top level.
    summary.update(extra or {})
    path = os.path.join(_results_root(config), f"summary_{mn}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
