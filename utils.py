import hashlib
import os
import random
import subprocess
from typing import Dict, List, Optional

import numpy as np
import torch


def get_device(config) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(config) -> torch.dtype:
    """The compute dtype. Under mixed precision this is the autocast dtype."""
    if config.dtype not in _DTYPES:
        raise ValueError(
            f"Unknown dtype '{config.dtype}'. Choose from {list(_DTYPES)}"
        )
    if config.dtype == "fp16" and not getattr(config, "mixed_precision", False):
        raise ValueError("dtype='fp16' requires mixed_precision=True")
    return _DTYPES[config.dtype]


def resolve_param_dtype(config) -> torch.dtype:
    """The storage dtype of parameters: fp32 master weights under mixed precision."""
    if getattr(config, "mixed_precision", False):
        return torch.float32
    return resolve_dtype(config)


def autocast_context(config):
    """torch.autocast in the compute dtype when mixed precision is on, else a no-op."""
    return torch.autocast(
        device_type=get_device(config).type,
        dtype=resolve_dtype(config),
        enabled=getattr(config, "mixed_precision", False),
    )


def make_grad_scaler(config) -> "torch.amp.GradScaler":
    """A GradScaler, enabled only for fp16 mixed precision (bf16 needs no loss scaling)."""
    enabled = getattr(config, "mixed_precision", False) and config.dtype == "fp16"
    return torch.amp.GradScaler(get_device(config).type, enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- Artifact provenance (steering/code-quality.md, owner decision 2026-09-18) ---
#
# Every artifact must be able to say which code produced it, from the file itself and
# without reference to prose. Commit time is not edit time: a change landing between a
# run and its write-up leaves no trace in the artifact, which is exactly how the
# repository ended up with three contradictory accounts of which objective trained one
# checkpoint (training-pipeline REV-129 / REV-136).
#
# The revision is therefore the git commit when the tree is CLEAN, and otherwise the
# commit plus a content hash over the covered sources, with `dirty: true`. A run from a
# dirty tree is recorded as such, never silently as its nearest commit.

# Coverage is defined by EXCLUSION, not enumeration -- `steering/code-quality.md`,
# "Coverage: a property, not a list" (owner, amended 2026-09-18). The governing rule is the
# PROPERTY: `source_revision` must change whenever any code that can change a published
# artifact changes. Everything below is how that property is realised.
#
# Covered: every first-party `.py` file tracked by git, EXCEPT the paths below. An allowlist
# was tried first (`models/`, `training/`, `data/`, `configs/`) and failed within one round
# by omitting `evaluation/`, which is where every derived artifact in this project is
# actually produced (REV-138). A denylist covers new code by default, so drift now requires
# someone to write an explicit exclusion rather than merely to forget an inclusion.
_SOURCE_EXCLUDED_PREFIXES = (
    "tests/",            # a test cannot change a published artifact
    ".venv/",
    "__pycache__/",
    "data/raw/",         # data, not code
    "data/processed/",   # data, not code
)
_SOURCE_EXCLUDED_SEGMENTS = frozenset({"__pycache__", ".venv", ".git"})


def _is_covered_source(rel_path: str) -> bool:
    """The single coverage rule, shared by the content hash AND the dirtiness check.

    Both halves of a revision must ask the same question, so they can never disagree
    about what "this code" means (`steering/code-quality.md`). A change to a covered file
    that moves neither the hash nor the `dirty` flag is a defect in this predicate.
    """
    rel = rel_path.replace(os.sep, "/")
    if rel.startswith("./"):
        rel = rel[2:]
    if not rel.endswith(".py"):
        return False
    if rel.startswith(_SOURCE_EXCLUDED_PREFIXES):
        return False
    return not _SOURCE_EXCLUDED_SEGMENTS.intersection(rel.split("/")[:-1])


_SOURCE_REVISION_CACHE: Optional[Dict[str, object]] = None


def _git(*args: str, cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ("git",) + args, cwd=cwd, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _walk_py_files(root: str) -> List[str]:
    """Every `.py` under `root`, repo-relative. Fallback for a non-git checkout."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SOURCE_EXCLUDED_SEGMENTS)
        for name in filenames:
            if name.endswith(".py"):
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return found


def _covered_source_files(root: str) -> List[str]:
    """The set of files the coverage rule defines, repo-relative and path-ordered.

    Enumerated from git rather than from a fixed directory list, so top-level modules
    (`main.py`, `utils.py`) and packages added later are covered without anyone
    remembering to add them.
    """
    tracked = _git("ls-files", "-z", "--", "*.py", cwd=root)
    if tracked is None:
        # Not a git checkout: fall back to walking the tree. The exclusions are applied
        # the same way; only the "tracked by git" clause cannot be honoured.
        candidates = _walk_py_files(root)
    else:
        candidates = [p for p in tracked.split("\0") if p]
    return sorted(p for p in candidates if _is_covered_source(p))


def _hash_source_tree(root: str) -> str:
    """Content hash over every covered `.py` source, path-ordered."""
    digest = hashlib.sha256()
    for rel in _covered_source_files(root):
        digest.update(rel.encode())
        try:
            with open(os.path.join(root, rel), "rb") as f:
                digest.update(f.read())
        except OSError:
            # Tracked but absent from the worktree (a deletion). Recording the absence
            # still moves the hash, which is what the property requires.
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


def _porcelain_paths(status: str) -> List[str]:
    """Every tracked path named by `git status --porcelain`.

    Untracked entries (`??`) are skipped: the coverage rule excludes anything not tracked
    by git, and the dirtiness check must not be broader than the hash. Rename entries
    (`R  old -> new`) name two paths and both are returned, since either side moving is a
    change to the covered sources.
    """
    paths: List[str] = []
    for line in status.splitlines():
        if not line.strip() or line.startswith("??"):
            continue
        entry = line[3:].strip()
        for part in entry.split(" -> "):
            part = part.strip()
            if len(part) >= 2 and part[0] == '"' and part[-1] == '"':
                part = part[1:-1]
            if part:
                paths.append(part)
    return paths


def get_source_revision(refresh: bool = False) -> Dict[str, object]:
    """Resolve this run's `source_revision`, computed once per process.

    Returns a dict carrying at least `commit`, `dirty` and `source_hash`. When the tree
    is dirty the hash is what distinguishes one run from another, since the commit
    cannot.
    """
    global _SOURCE_REVISION_CACHE
    if _SOURCE_REVISION_CACHE is not None and not refresh:
        return dict(_SOURCE_REVISION_CACHE)

    root = os.path.dirname(os.path.abspath(__file__))
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", cwd=root)

    if commit is None:
        # Not a git checkout (or git unavailable): the content hash is all there is,
        # and saying so is better than recording a commit that does not exist.
        revision: Dict[str, object] = {
            "commit": None,
            "dirty": True,
            "source_hash": _hash_source_tree(root),
            "note": "no git metadata available; identified by source_hash only",
        }
    else:
        # Dirtiness is judged by EXACTLY the rule the hash covers -- the same
        # `_is_covered_source` predicate, not a parallel string test -- so the two halves
        # of a revision can never disagree about what "this code" means.
        tracked_dirty = any(
            _is_covered_source(p) for p in _porcelain_paths(status or "")
        )
        revision = {
            "commit": commit,
            "dirty": bool(tracked_dirty),
            "source_hash": _hash_source_tree(root),
        }

    _SOURCE_REVISION_CACHE = dict(revision)
    return dict(revision)


def artifact_source_revision(checkpoint: Optional[dict]) -> Dict[str, object]:
    """The `source_revision` an artifact derived from `checkpoint` must echo (AC-09).

    Echoed VERBATIM from the checkpoint. A checkpoint written before AC-09 carries no
    revision, and the honest record of that is a note saying so -- substituting the
    currently-running code's revision would assert a provenance that does not hold,
    which is the silent-substitution failure `steering/code-quality.md` forbids and
    exactly the inference that produced REV-129.
    """
    if isinstance(checkpoint, dict):
        revision = checkpoint.get("source_revision")
        if isinstance(revision, dict) and revision:
            return dict(revision)
    return {
        "commit": None,
        "dirty": True,
        "source_hash": None,
        "note": (
            "checkpoint carries no source_revision (written before AC-09); "
            "the provenance of this artifact is NOT established"
        ),
    }
