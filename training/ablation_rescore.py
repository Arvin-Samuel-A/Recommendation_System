"""`--stage ablation_rescore`: re-evaluate a finished ablation run without retraining.

specs/eval-integrity FR-8 (design D9). For every ok arm of the selected studies, the arm's
backbone is located by the training fingerprint RECORDED in its arm file (FR-4 changed how
fingerprints are computed, so it is never recomputed), every saved `epoch_XXX.pt` is scored
on the full validation split with the fixed protocol (fp32 scores, tie-averaged metrics,
per-forward autocast with the weight cache off), the best epoch is re-selected by the
`run_pretrain` rule, and that epoch is scored on the test rows of FR-6 together with the
popularity floor.

Nothing is trained and nothing original is modified. Outputs:

- `{store}/rescore.json` per fingerprint (written atomically after every epoch, so a killed
  process resumes at the first unscored epoch), serialised by `{store}/.rescore.lock`;
- rescored arm records, study JSONs and the summary under `{ablation_dir}/rescored/seed_S/`;
- plots under `{plot_dir}/rescored/`.

Rescoring gives honest numbers for old weights; it cannot repair the training defect those
weights were trained under (`RESCORE_CAVEAT`).
"""

import copy
import hashlib
import json
import logging
import os
import re
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from configs.config import Config
from evaluation.metrics import (
    EVAL_PROTOCOL, FullRanking, evaluate_model, evaluate_popularity, load_full_ranking,
    load_item_popularity,
)
from training.ablation_runner import (
    STUDY_CONFIGS,
    _arm_path,
    _current_revision,
    _results_root,
    _runtime_record,
    _save_summary,
    _training_fields,
    _write_json_atomic,
    backbone_store_dir,
    collect_study,
    file_lock,
    find_backbone,
    resolve_study_configs,
    select_eval_rows,
)
from utils import is_oom_error

logger = logging.getLogger(__name__)

#: eval-integrity FR-8.7, stated in every rescored arm record, the rescored summary and a
#: WARNING at the start and end of the stage.
RESCORE_CAVEAT = (
    "Rescoring gives honest numbers for these weights, but the weights were trained under "
    "the stale-autocast-cache bug (eval-integrity FR-3). The A–H comparison requires a "
    "re-run with the fixed training code."
)

#: Format of `rescore.json`; a record of another format is discarded and recomputed.
RESCORE_FORMAT = 1
RESCORE_RECORD = "rescore.json"
RESCORE_LOCK = ".rescore.lock"
RESCORE_LOCK_POLL_SECONDS = 5.0

#: Runtime-only fields copied from the CLI config onto a checkpoint's config (FR-8.3).
RUNTIME_FIELDS = (
    "device", "attention_backend", "eval_batch_size", "num_workers", "pin_memory",
    "processed_data_dir", "output_dir", "checkpoint_dir", "log_dir", "plot_dir",
    "ablation_dir",
)

#: Weight-dependent extras computed from the ORIGINAL best.pt with the defective pipeline;
#: moved under `rescore.pre_rescore_extras` so no plot or table reads them as current
#: (FR-8.6). Weight-independent fields (router_parameters, latency, parameters,
#: flops_per_inference) stay at the top level.
WEIGHT_DEPENDENT_EXTRAS = (
    "routing", "gradient", "cold_start", "longtail", "personalization", "communication",
)

_EPOCH_FILE = re.compile(r"^epoch_(\d{3})\.pt$")

#: Rows per batch for the popularity reference (no model); each row ranks the catalogue.
_POPULARITY_BATCH = 1024


# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------


def config_from_checkpoint(ckpt_config: Dict, runtime: Config) -> Config:
    """Rebuild the config a checkpoint was trained with; only runtime fields come from
    `runtime` (FR-8.3, DEC-9). `vocab_size`, `num_items` and `feature_dim` stay the
    checkpoint's, as the trained model needs."""
    from data.preprocessing import check_processed_dir

    cfg = Config()
    known = {f.name for f in fields(Config)}
    unknown = sorted(k for k in ckpt_config if k not in known)
    if unknown:
        logger.warning("checkpoint config keys not in Config (ignored): %s", unknown)
    for key, value in ckpt_config.items():
        if key in known:
            setattr(cfg, key, copy.deepcopy(value))
    for key in RUNTIME_FIELDS:
        setattr(cfg, key, copy.deepcopy(getattr(runtime, key)))
    check_processed_dir(cfg)
    cfg.validate()
    return cfg


def rescored_config(config: Config) -> Config:
    """The CLI config with the rescored results root (`{ablation_dir}/rescored`) and plot
    directory (`{plot_dir}/rescored`), DEC-7."""
    rescored = copy.deepcopy(config)
    rescored.ablation_dir = os.path.join(config.ablation_dir, "rescored")
    rescored.plot_dir = os.path.join(config.plot_dir, "rescored")
    return rescored


# ---------------------------------------------------------------------------------------
# Shared evaluation data
# ---------------------------------------------------------------------------------------


class EvalData:
    """Val/test data shared by every fingerprint a process rescores, loaded lazily once.

    Splits are keyed by `(max_seq_length, use_metadata)` -- study H's sequence-length arms
    use other widths. The full-ranking rows, item popularity and the popularity references
    do not depend on the model and are computed once. Test rows follow FR-6 with the CLI
    config's `ablation_eval_examples` / `ablation_eval_seed`.
    """

    def __init__(self, config: Config):
        self.config = config
        self.processed = config.processed_data_dir
        self._splits: Dict[Tuple, Dict] = {}
        self._rankings: Dict[str, FullRanking] = {}
        self._test_rows: Optional[Tuple[Optional[np.ndarray], Dict]] = None
        self._test_ranking: Optional[FullRanking] = None
        self._popularity: Dict[int, torch.Tensor] = {}
        self._val_pop: Dict[Tuple, Dict] = {}
        self._test_pop: Dict[Tuple, Dict] = {}

    # -- rankings and rows --------------------------------------------------------------

    def _path(self, name: str) -> str:
        return os.path.join(self.processed, name)

    def _ranking(self, split_name: str) -> FullRanking:
        """Every row of a split; targets and users read from the stored split, which do
        not depend on the width the model narrows it to."""
        if split_name not in self._rankings:
            data = torch.load(self._path(f"{split_name}_tensorised.pt"), weights_only=True)
            self._rankings[split_name] = load_full_ranking(
                self.processed, split_name, data["targets"], data["user_ids"],
            )
        return self._rankings[split_name]

    def val_ranking(self) -> FullRanking:
        return self._ranking("val")

    def test_rows(self) -> Tuple[Optional[np.ndarray], Dict]:
        if self._test_rows is None:
            self._test_rows = select_eval_rows(
                len(self._ranking("test")),
                self.config.ablation_eval_examples, self.config.ablation_eval_seed,
            )
        return self._test_rows

    def test_ranking(self) -> FullRanking:
        if self._test_ranking is None:
            idx, _ = self.test_rows()
            full = self._ranking("test")
            self._test_ranking = full if idx is None else full.select(idx)
        return self._test_ranking

    def protocol(self, eval_k: int) -> Dict:
        """What a rescore result depends on besides the weights and the code (D9)."""
        return {
            **EVAL_PROTOCOL,
            "eval_k": int(eval_k),
            "val_rows": len(self.val_ranking()),
            "test_eval": self.test_rows()[1],
        }

    # -- splits -------------------------------------------------------------------------

    def split(self, eval_cfg: Config, name: str):
        """`(sequences, metadata_matrix)` of the val/test split for this model's width."""
        key = (int(eval_cfg.max_seq_length), bool(eval_cfg.use_metadata), int(eval_cfg.vocab_size))
        if key not in self._splits:
            from data.sequence_generator import load_metadata_matrix, load_tensorised

            meta = load_metadata_matrix(self.processed, eval_cfg)
            entry = {}
            for split_name in ("val", "test"):
                ds = load_tensorised(
                    self._path(f"{split_name}_tensorised.pt"), meta,
                    max_seq_length=eval_cfg.max_seq_length,
                )
                if len(ds) != len(self._ranking(split_name)):
                    raise RuntimeError(
                        f"{split_name} split has {len(ds)} rows but its ranking has "
                        f"{len(self._ranking(split_name))}; re-run --stage preprocess"
                    )
                entry[split_name] = ds
            self._splits[key] = entry
        ds = self._splits[key][name]
        if name == "val":
            return ds.sequences, ds.metadata_matrix
        idx, _ = self.test_rows()
        seqs = ds.sequences if idx is None else ds.sequences[torch.from_numpy(idx)]
        return seqs, ds.metadata_matrix

    # -- popularity -------------------------------------------------------------------

    def _item_popularity(self, vocab_size: int) -> torch.Tensor:
        if vocab_size not in self._popularity:
            self._popularity[vocab_size] = load_item_popularity(self.processed, vocab_size)
        return self._popularity[vocab_size]

    def val_popularity(self, vocab_size: int, eval_k: int) -> Dict:
        key = (vocab_size, eval_k)
        if key not in self._val_pop:
            self._val_pop[key] = evaluate_popularity(
                self.val_ranking(), self._item_popularity(vocab_size), k=eval_k,
                batch_size=_POPULARITY_BATCH,
            )
        return self._val_pop[key]

    def test_popularity(self, vocab_size: int, eval_k: int) -> Dict:
        key = (vocab_size, eval_k)
        if key not in self._test_pop:
            self._test_pop[key] = evaluate_popularity(
                self.test_ranking(), self._item_popularity(vocab_size), k=eval_k,
                batch_size=_POPULARITY_BATCH,
            )
        return self._test_pop[key]


# ---------------------------------------------------------------------------------------
# Per-fingerprint engine
# ---------------------------------------------------------------------------------------


def _code_identity() -> Dict:
    revision = _current_revision()
    return {k: revision.get(k) for k in ("commit", "source_hash", "dirty")}


def _epoch_files(ckpt_dir: str) -> Dict[int, str]:
    if not os.path.isdir(ckpt_dir):
        return {}
    found = {}
    for name in os.listdir(ckpt_dir):
        match = _EPOCH_FILE.match(name)
        if match:
            found[int(match.group(1))] = os.path.join(ckpt_dir, name)
    return dict(sorted(found.items()))


def _file_list(epochs: Dict[int, str]) -> Dict[str, List[int]]:
    """Epoch files with size and mtime_ns (DEC-8): any change invalidates the record."""
    out = {}
    for _, path in epochs.items():
        st = os.stat(path)
        out[os.path.basename(path)] = [int(st.st_size), int(st.st_mtime_ns)]
    return out


def _load(path: str) -> dict:
    # map_location="cpu": checkpoints trained on CUDA are rescored on MPS/CPU too
    # (FR-8.3); load_state_dict then copies onto the model's device.
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_identity(path: str) -> dict:
    """A checkpoint opened memory-mapped, for its `config` and `source_revision` only.

    Review REV-004: the identity check runs before any evaluation (FR-8.9), and the
    evaluation loop loads every file again, so a full read here doubled the checkpoint
    I/O. With `mmap=True` the tensors (weights, optimizer state) are mapped, not read; only
    the small pickled metadata is materialised. A file torch cannot map (a legacy
    non-zip checkpoint) falls back to a full load.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        return _load(path)


def _read_record(path: str, fingerprint: str, code: Dict, protocol: Dict,
                 files: Dict) -> Optional[Dict]:
    """A valid `rescore.json` for this code, protocol and epoch-file list, or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Discarding unreadable %s (%s) -- recomputing", path, exc)
        return None
    expected = {
        "format": RESCORE_FORMAT, "fingerprint": fingerprint,
        "rescore_code": code, "protocol": protocol, "epoch_files": files,
    }
    stale = [k for k, v in expected.items() if rec.get(k) != v]
    if stale:
        logger.warning(
            "Discarding %s: %s differ from the current rescoring code/protocol/epoch files "
            "-- recomputing", path, stale,
        )
        return None
    return rec


def _verify_identity(epochs: Dict[int, str], best_ckpt: dict, best_path: str) -> None:
    """FR-8.9: every epoch checkpoint agrees with the arm's best.pt on every
    training-relevant field and on `source_revision`."""
    best_fields = _training_fields(best_ckpt.get("config") or {})
    best_rev = best_ckpt.get("source_revision")
    for _, path in epochs.items():
        ck = _load_identity(path)
        ck_fields = _training_fields(ck.get("config") or {})
        if ck_fields != best_fields:
            diff = sorted(
                k for k in set(ck_fields) | set(best_fields)
                if ck_fields.get(k) != best_fields.get(k)
            )
            raise ValueError(
                f"{path}: stored config differs from {best_path} on training-relevant "
                f"fields {diff}; this epoch file does not belong to the arm's backbone"
            )
        if ck.get("source_revision") != best_rev:
            raise ValueError(
                f"{path}: source_revision {ck.get('source_revision')} differs from "
                f"{best_path}'s {best_rev}; this epoch file was written by other code"
            )


def _earliest_argmax(epoch_vals: Dict[int, float]) -> int:
    """The `run_pretrain` rule: highest val NDCG, earliest epoch on ties."""
    best_e, best_v = None, None
    for e in sorted(epoch_vals):
        if best_v is None or epoch_vals[e] > best_v:
            best_e, best_v = e, epoch_vals[e]
    return best_e


def rescore_fingerprint(
    fingerprint: str,
    entry: Dict,
    ckpt_dir: str,
    epochs: Dict[int, str],
    config: Config,
    shared: EvalData,
) -> Dict:
    """Score every epoch of one backbone on val, re-select, score the best on test.

    Serialised per fingerprint by `.rescore.lock` in the store directory; resumable from
    `rescore.json`; a complete, valid record is returned without evaluating anything.
    """
    from training.pretrain import build_model, load_checkpoint
    from utils import get_device

    store = backbone_store_dir(config.checkpoint_dir, fingerprint)
    record_path = os.path.join(store, RESCORE_RECORD)
    lock_path = os.path.join(store, RESCORE_LOCK)

    def log_wait(holder: str) -> None:
        logger.info("Fingerprint %s is being rescored by another process (%s) -- waiting",
                    fingerprint, holder)

    best_path = os.path.join(ckpt_dir, "best.pt")
    with file_lock(lock_path, poll=RESCORE_LOCK_POLL_SECONDS, on_wait=log_wait):
        best_ckpt = _load(best_path)
        eval_cfg = config_from_checkpoint(best_ckpt["config"], runtime=config)
        code = _code_identity()
        protocol = shared.protocol(eval_cfg.eval_k)
        files = _file_list(epochs)
        rec = _read_record(record_path, fingerprint, code, protocol, files)
        if rec is not None and rec.get("complete"):
            logger.info("Fingerprint %s: rescore record is complete -- reused", fingerprint)
            return rec
        if rec is None:
            rec = {
                "format": RESCORE_FORMAT,
                "fingerprint": fingerprint,
                "rescore_code": code,
                "protocol": protocol,
                "epoch_files": files,
                "checkpoint_dir": ckpt_dir,
                "epochs": {},
                "complete": False,
            }

        todo = {e: p for e, p in epochs.items() if str(e) not in rec["epochs"]}
        _verify_identity(todo, best_ckpt, best_path)

        device = get_device(eval_cfg)
        # Built ONCE per fingerprint; each epoch's weights are loaded in place. That is
        # only correct because no autocast region caches casts any more (FR-3.3); AC-21
        # would catch a regression (every epoch would score the same).
        model, moe = build_model(eval_cfg)
        val_seqs, val_meta = shared.split(eval_cfg, "val")
        val_ranking = shared.val_ranking()
        for e, path in todo.items():
            ck = _load(path)
            load_checkpoint(path, model, moe, checkpoint=ck)
            model.eval()
            moe.eval()
            with torch.no_grad():
                val = evaluate_model(model, val_seqs, val_ranking, eval_cfg, moe_layer=moe,
                                     metadata_matrix=val_meta)
            rec["epochs"][str(e)] = {
                "val": val,
                "original_val_ndcg": float(ck.get("val_ndcg", float("nan"))),
                "checkpoint": path,
            }
            _write_json_atomic(record_path, rec)
            logger.info("Fingerprint %s epoch %d: val NDCG@%d %.6f (originally %.6f)",
                        fingerprint, e, eval_cfg.eval_k, val["ndcg"]["mean"],
                        rec["epochs"][str(e)]["original_val_ndcg"])

        scored = {int(e): v["val"]["ndcg"]["mean"] for e, v in rec["epochs"].items()}
        best_epoch = _earliest_argmax(scored)
        selected = epochs[best_epoch]
        load_checkpoint(selected, model, moe, checkpoint=_load(selected))
        model.eval()
        moe.eval()
        test_seqs, test_meta = shared.split(eval_cfg, "test")
        with torch.no_grad():
            test_metrics = evaluate_model(
                model, test_seqs, shared.test_ranking(), eval_cfg, moe_layer=moe,
                metadata_matrix=test_meta,
            )
        rec.update({
            "best_epoch": best_epoch,
            "best_val_ndcg": scored[best_epoch],
            "selected_checkpoint": selected,
            "test": {
                "metrics": test_metrics,
                "test_eval": protocol["test_eval"],
                "popularity_reference": shared.test_popularity(eval_cfg.vocab_size, eval_cfg.eval_k),
            },
            "val_popularity_reference": shared.val_popularity(eval_cfg.vocab_size, eval_cfg.eval_k),
            "eval_runtime": _runtime_record(model, eval_cfg, device),
            "complete": True,
        })
        _write_json_atomic(record_path, rec)
        return rec


# ---------------------------------------------------------------------------------------
# Arms, collection and the stage
# ---------------------------------------------------------------------------------------


def _rescore_key(orig: Dict, fingerprint: str, protocol: Dict) -> str:
    payload = {
        "arm_key": orig.get("arm_key"),
        "fingerprint": fingerprint,
        "source_hash": _current_revision().get("source_hash"),
        "protocol": protocol,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_rescored_record(orig: Dict, rec: Dict, rescore_key: str) -> Dict:
    """A copy of the original arm record with the rescored numbers (FR-8.5/8.6)."""
    from utils import get_source_revision

    out = copy.deepcopy(orig)
    out.pop("reused_from_previous_run", None)
    pre_extras = {k: out.pop(k) for k in WEIGHT_DEPENDENT_EXTRAS if k in out}
    epochs_scored = sorted(int(e) for e in rec["epochs"])
    out["metrics"] = {
        k: {sk: float(sv) for sk, sv in v.items()} for k, v in rec["test"]["metrics"].items()
    }
    out["best_epoch"] = rec["best_epoch"]
    out["best_val_ndcg"] = rec["best_val_ndcg"]
    out["popularity_reference"] = rec["test"]["popularity_reference"]
    out["val_popularity_reference"] = rec["val_popularity_reference"]
    out["test_eval"] = rec["test"]["test_eval"]
    out["eval_protocol"] = dict(EVAL_PROTOCOL)
    out["num_eval_examples"] = rec["test"]["test_eval"]["num_examples"]
    out["code_revision"] = _current_revision()
    out["rescore"] = {
        "rescored": True,
        "rescore_key": rescore_key,
        "rescored_by": get_source_revision(),
        "original_code_revision": orig.get("code_revision"),
        "original_metrics": orig.get("metrics"),
        "original_best_epoch": orig.get("best_epoch"),
        "original_best_val_ndcg": orig.get("best_val_ndcg"),
        "original_num_eval_examples": orig.get("num_eval_examples"),
        "per_epoch_val_ndcg": {
            str(e): rec["epochs"][str(e)]["val"]["ndcg"]["mean"] for e in epochs_scored
        },
        "per_epoch_original_val_ndcg": {
            str(e): rec["epochs"][str(e)]["original_val_ndcg"] for e in epochs_scored
        },
        "epochs_scored": epochs_scored,
        "selected_checkpoint": rec["selected_checkpoint"],
        "eval_runtime": rec["eval_runtime"],
        "pre_rescore_extras": pre_extras,
        "caveat": RESCORE_CAVEAT,
    }
    return out


def rescore_arm(
    orig: Dict,
    study: str,
    label: str,
    model_name: str,
    config: Config,
    rescored: Config,
    shared: EvalData,
) -> str:
    """Rescore one ok arm; returns "rescored" or "current" (skipped, already rescored)."""
    fingerprint = orig.get("training_fingerprint")
    if not fingerprint:
        raise ValueError(f"arm {study}/{label} records no training_fingerprint")
    entry = find_backbone(config.checkpoint_dir, fingerprint)
    if entry is None:
        raise FileNotFoundError(
            f"no completed backbone for fingerprint {fingerprint} at "
            f"{backbone_store_dir(config.checkpoint_dir, fingerprint)}"
        )
    # Equivalent to join(entry["checkpoint_dir"], model_dir_name); a multi-seed
    # registration's checkpoint_dir lies outside the store (Edge Cases).
    ckpt_dir = os.path.dirname(entry["checkpoint_path"])
    epochs = _epoch_files(ckpt_dir)
    if not epochs:
        raise FileNotFoundError(f"no epoch_XXX.pt checkpoints in {ckpt_dir}")

    eval_k = int((orig.get("resolved_config") or {}).get("eval_k", config.eval_k))
    key = _rescore_key(orig, fingerprint, shared.protocol(eval_k))
    out_path = _arm_path(rescored, study, model_name, label)
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if (existing.get("rescore") or {}).get("rescore_key") == key:
                logger.info("  %s/%s: rescored record is current -- skipped", study, label)
                return "current"
        except (OSError, json.JSONDecodeError):
            pass

    rec = rescore_fingerprint(fingerprint, entry, ckpt_dir, epochs, config, shared)
    _write_json_atomic(out_path, build_rescored_record(orig, rec, key))
    logger.info("  %s/%s: rescored (best epoch %d, test NDCG@%d %.4f, popularity %.4f)",
                study, label, rec["best_epoch"], eval_k,
                rec["test"]["metrics"]["ndcg"]["mean"],
                rec["test"]["popularity_reference"]["ndcg"]["mean"])
    return "rescored"


def run_ablation_rescore(
    config: Config,
    model_name: str,
    study: str = "all",
    allow_incomplete: bool = False,
) -> Dict:
    """Rescore the selected studies' arms and assemble them under the rescored root.

    Returns `{"studies": {study: {...}}, "not_written": [...], "failed_arms": {...}}`.
    `failed_arms` maps a study to its arms whose original record is `failed` or whose
    rescore raised; the CLI stage exits `ARM_FAILED_EXIT_CODE` when it is non-empty, else 1
    when `not_written` is non-empty.
    """
    from data.preprocessing import check_processed_dir

    logger.warning("ablation_rescore: %s", RESCORE_CAVEAT)
    studies = list(STUDY_CONFIGS) if study == "all" else [study]
    if any(s not in STUDY_CONFIGS for s in studies):
        raise ValueError(f"Unknown study '{study}'. Choose from {list(STUDY_CONFIGS)} or all")
    check_processed_dir(config)
    rescored = rescored_config(config)
    shared = EvalData(config)

    report: Dict[str, Dict] = {}
    assembled, incomplete, not_written, failed_arms = {}, {}, [], {}
    for s in studies:
        info = {"rescored": [], "current": [], "skipped": {}, "failed": {}}
        failed_originals = []
        for cfg in resolve_study_configs(s, config):
            label = cfg._study_label
            path = _arm_path(config, s, model_name, label)
            if not os.path.exists(path):
                info["skipped"][label] = f"original arm file missing: {path}"
                logger.warning("  %s/%s: original arm file missing (%s) -- skipped", s, label, path)
                continue
            try:
                with open(path) as f:
                    orig = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                info["skipped"][label] = f"unreadable original arm file {path}: {exc}"
                logger.warning("  %s/%s: unreadable original arm file -- skipped", s, label)
                continue
            if orig.get("status") != "ok":
                if orig.get("status") == "failed":
                    failed_originals.append(label)
                info["skipped"][label] = f"original status {orig.get('status')!r}"
                logger.warning("  %s/%s: original status %r -- skipped", s, label, orig.get("status"))
                continue
            try:
                outcome = rescore_arm(orig, s, label, model_name, config, rescored, shared)
                info[outcome].append(label)
            except Exception as exc:  # one arm failing must not lose the others
                if is_oom_error(exc):
                    raise  # exit 75; run_with_retry resumes from rescore.json
                logger.exception("  %s/%s: rescore failed: %s", s, label, exc)
                info["failed"][label] = f"{type(exc).__name__}: {exc}"

        assessment = collect_study(rescored, model_name, s, allow_incomplete=allow_incomplete)
        info["assessment"] = assessment.describe()
        info["written"] = assessment.written
        if assessment.written:
            mn = model_name.replace("-", "_")
            with open(os.path.join(_results_root(rescored), f"{s}_{mn}.json")) as f:
                assembled[s] = json.load(f)
            if not assessment.complete:
                incomplete[s] = {
                    "missing": assessment.missing + assessment.stale,
                    "failed": sorted(assessment.failed),
                }
        else:
            not_written.append(s)
            logger.error("Rescored study NOT assembled -- %s", assessment.describe())
            incomplete[s] = {
                "missing": assessment.missing + assessment.stale,
                "failed": sorted(assessment.failed),
                "inconsistent": assessment.inconsistent,
            }
        report[s] = info
        if failed_originals or info["failed"]:
            failed_arms[s] = sorted(set(failed_originals) | set(info["failed"]))

    if study == "all":
        _save_summary(
            rescored, model_name,
            {s: r for s, r in assembled.items() if s not in incomplete},
            incomplete=incomplete,
            extra={"rescore_caveat": RESCORE_CAVEAT},
        )
    logger.warning("ablation_rescore: %s", RESCORE_CAVEAT)
    return {"studies": report, "not_written": not_written, "failed_arms": failed_arms}
