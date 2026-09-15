import hashlib
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import Config
from utils import seed_worker

logger = logging.getLogger(__name__)


def align_metadata_matrix(
    metadata_matrix, vocab_size: Optional[int] = None
) -> torch.Tensor:
    """Return the `(vocab, feature_dim)` metadata matrix as a float32 tensor.

    REV-072: `extract_metadata` historically allocated `num_items + 1` rows, so MASK_ID
    (`num_items + 1`) had no row at all and callers clamped it onto the last real item's
    genome vector. Padding here with an all-zero row means a masked position contributes
    nothing through the metadata projection, which is what the MIM objective requires
    (training-pipeline REV-059), and lets the clamps go away. Artifacts written by an
    older preprocessing run are padded on load rather than needing regeneration.
    """
    mat = torch.as_tensor(metadata_matrix, dtype=torch.float32)
    if mat.dim() != 2:
        raise ValueError(f"metadata matrix must be 2-D (vocab, feature_dim), got {tuple(mat.shape)}")
    if vocab_size is not None and mat.shape[0] < vocab_size:
        pad = torch.zeros(vocab_size - mat.shape[0], mat.shape[1], dtype=torch.float32)
        mat = torch.cat([mat, pad], dim=0)
    return mat


def load_metadata_matrix(processed_dir: str, config: Config) -> Optional[torch.Tensor]:
    """Load `item_metadata.npy` aligned to the full embedding vocabulary, or None.

    Single loader for every consumer so the REV-072 alignment is applied uniformly:
    row 0 is padding, rows 1..num_items are real items, row `num_items + 1` is MASK_ID.
    """
    if not config.use_metadata:
        return None
    path = os.path.join(processed_dir, "item_metadata.npy")
    if not os.path.exists(path):
        return None
    vocab = config.vocab_size if config.vocab_size > 0 else (
        config.num_items + 2 if config.num_items > 0 else None
    )
    return align_metadata_matrix(np.load(path), vocab)


class RecDataset(Dataset):
    """Tensorised split. Metadata is gathered per item, never materialised densely.

    REV-058: this used to hold a dense `(num_examples, max_seq_length, feature_dim)`
    float32 block built by indexing the metadata matrix with every stored sequence. On
    MovieLens-20M that is 60.8 GB for the train split alone against a 30 GB host budget,
    so no real-scale entry point could start. The `(vocab, feature_dim)` matrix is 3.2 MB;
    `__getitem__` gathers the `(S, F)` rows for one example and the DataLoader collates
    them into exactly the `(B, S, F)` block `train_one_step` already expects. Indexing
    stays index-only, as FR-6 requires.
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        user_ids: torch.Tensor,
        lengths: torch.Tensor,
        targets_k: Optional[torch.Tensor] = None,
        metadata_matrix=None,
    ):
        self.sequences = sequences
        self.targets = targets
        self.user_ids = user_ids
        self.lengths = lengths
        self.targets_k = targets_k
        # REV-090b: align unconditionally. A caller handing in a raw `(num_items + 1, F)`
        # artifact -- which is what pre-REV-072 preprocessing runs wrote to disk -- would
        # otherwise IndexError on the first MASK_ID position. The vocabulary implied by
        # the stored sequences is the smallest safe target, so the alignment no longer
        # depends on every path going through `load_metadata_matrix`.
        self.metadata_matrix = (
            None
            if metadata_matrix is None
            else align_metadata_matrix(
                metadata_matrix, vocab_size=int(sequences.max()) + 1 if sequences.numel() else None
            )
        )

    @property
    def has_metadata(self) -> bool:
        return self.metadata_matrix is not None

    @property
    def feature_dim(self) -> int:
        return 0 if self.metadata_matrix is None else self.metadata_matrix.shape[1]

    def metadata_for(self, idx: int) -> Optional[torch.Tensor]:
        """Per-example `(S, F)` metadata, gathered on demand."""
        if self.metadata_matrix is None:
            return None
        return self.metadata_matrix[self.sequences[idx]]

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sequences": self.sequences[idx],
            "targets": self.targets[idx],
            "user_ids": self.user_ids[idx],
            "lengths": self.lengths[idx],
        }
        if self.targets_k is not None:
            item["targets_k"] = self.targets_k[idx]
        if self.metadata_matrix is not None:
            item["metadata"] = self.metadata_matrix[self.sequences[idx]]
        return item


def _left_pad(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_id] * (max_len - len(seq)) + seq


def generate_sliding_window_examples(
    items: List[int],
    max_seq_length: int,
    multi_step_k: int = 0,
    split_end_idx: Optional[int] = None,
) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    sequences = []
    targets = []
    targets_k_list = []

    for i in range(1, len(items)):
        context = items[:i]
        target = items[i]

        padded = _left_pad(context, max_seq_length)
        sequences.append(padded)
        targets.append(target)

        multi_targets = []
        if multi_step_k > 0:
            for k in range(1, multi_step_k + 1):
                future_idx = i + k
                if future_idx < len(items):
                    if split_end_idx is not None and future_idx >= split_end_idx:
                        multi_targets.append(0)
                    else:
                        multi_targets.append(items[future_idx])
                else:
                    multi_targets.append(0)
        targets_k_list.append(multi_targets)

    return sequences, targets, targets_k_list


def _fill_left_padded(row: np.ndarray, context: List[int], max_len: int) -> None:
    """Write `context` into a preallocated left-padded row, truncating to the tail."""
    row[:] = 0
    tail = context[-max_len:] if len(context) > max_len else context
    if tail:
        row[max_len - len(tail):] = tail


def _alloc_split(n: int, max_seq_length: int, multi_step_k: int) -> dict:
    """Preallocated int32 arrays for one split.

    REV-086: every example used to be a Python `list` of 50 Python `int` objects -- 482
    bytes per padded sequence against 200 for the equivalent int32 row -- with all three
    splits' dicts held simultaneously until they were pickled (4.7 + 0.6 + 0.6 = 5.9 GB
    on MovieLens-20M). int32 is sufficient: the largest id is MASK_ID = 20,721.
    """
    return {
        "sequences": np.zeros((n, max_seq_length), dtype=np.int32),
        "targets": np.zeros(n, dtype=np.int32),
        "user_ids": np.zeros(n, dtype=np.int32),
        "lengths": np.zeros(n, dtype=np.int32),
        "targets_k": np.zeros((n, multi_step_k), dtype=np.int32),
    }


def _count_examples(
    user_sequences: Dict[int, List[int]],
    train_seqs: Dict[int, List[int]],
    val_seqs: Dict[int, List[int]],
    test_seqs: Dict[int, List[int]],
) -> Tuple[int, int, int]:
    """Per-split example counts. Cheap -- each is a length arithmetic per user."""
    n_train = n_val = n_test = 0
    for user_id in user_sequences:
        train_items = train_seqs.get(user_id, [])
        if len(train_items) >= 2:
            n_train += len(train_items) - 1
        n_val += len(val_seqs.get(user_id, []))
        n_test += len(test_seqs.get(user_id, []))
    return n_train, n_val, n_test


def generate_sequences(
    user_sequences: Dict[int, List[int]],
    train_seqs: Dict[int, List[int]],
    val_seqs: Dict[int, List[int]],
    test_seqs: Dict[int, List[int]],
    config: Config,
) -> Tuple[dict, dict, dict, Dict[int, List[int]], Dict[int, List[int]]]:
    """Build the three splits' examples as preallocated int32 arrays (REV-086)."""
    n_train, n_val, n_test = _count_examples(
        user_sequences, train_seqs, val_seqs, test_seqs,
    )
    S, K = config.max_seq_length, config.multi_step_k

    train_examples = _alloc_split(n_train, S, K)
    val_examples = _alloc_split(n_val, S, K)
    test_examples = _alloc_split(n_test, S, K)

    train_user_history: Dict[int, List[int]] = {}
    eval_user_history: Dict[int, List[int]] = {}

    ti = vi = si = 0

    for user_id in user_sequences:
        train_items = train_seqs.get(user_id, [])
        val_items = val_seqs.get(user_id, [])
        test_items = test_seqs.get(user_id, [])

        train_user_history[user_id] = list(train_items)
        eval_user_history[user_id] = list(train_items) + list(val_items)

        if len(train_items) >= 2:
            seqs, tgts, tgts_k = generate_sliding_window_examples(
                train_items, S, K, split_end_idx=len(train_items),
            )
            # REV-088: the sliding window emits one example per context prefix
            # items[:i] for i in 1..len(train_items)-1, so example j carries a context of
            # j + 1 items. A single per-user length is correct only for the last example.
            for j, (seq, t, tk) in enumerate(zip(seqs, tgts, tgts_k)):
                train_examples["sequences"][ti] = seq
                train_examples["targets"][ti] = t
                train_examples["user_ids"][ti] = user_id
                train_examples["lengths"][ti] = min(j + 1, S)
                if K:
                    train_examples["targets_k"][ti, :len(tk)] = tk
                ti += 1

        if val_items:
            val_context = list(train_items)
            for item in val_items:
                _fill_left_padded(val_examples["sequences"][vi], val_context, S)
                val_examples["targets"][vi] = item
                val_examples["user_ids"][vi] = user_id
                val_examples["lengths"][vi] = min(len(val_context), S)
                vi += 1
                val_context.append(item)

        if test_items:
            test_context = list(train_items) + list(val_items)
            for item in test_items:
                _fill_left_padded(test_examples["sequences"][si], test_context, S)
                test_examples["targets"][si] = item
                test_examples["user_ids"][si] = user_id
                test_examples["lengths"][si] = min(len(test_context), S)
                si += 1
                test_context.append(item)

    assert (ti, vi, si) == (n_train, n_val, n_test), (
        f"example counts disagree with the preallocation: filled {(ti, vi, si)}, "
        f"allocated {(n_train, n_val, n_test)}"
    )

    assert train_user_history is not eval_user_history, (
        "train_user_history and eval_user_history must be distinct objects"
    )

    logger.info(
        "Generated sequences: train=%d, val=%d, test=%d",
        n_train, n_val, n_test,
    )
    return train_examples, val_examples, test_examples, train_user_history, eval_user_history


def tensorise_examples(
    examples: dict,
    metadata_matrix: Optional[np.ndarray] = None,
) -> RecDataset:
    def _as_long(value) -> torch.Tensor:
        # REV-086: the splits are int32 numpy arrays, so this is a view plus one cast
        # rather than a Python-object traversal. Lists are still accepted for the toy
        # fixtures that build examples by hand.
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).long()
        return torch.tensor(value, dtype=torch.long)

    sequences = _as_long(examples["sequences"])
    targets = _as_long(examples["targets"])
    user_ids = _as_long(examples["user_ids"])
    lengths = _as_long(examples["lengths"])

    tk = examples.get("targets_k")
    targets_k = None
    if tk is not None and len(tk) > 0:
        has_targets = tk.shape[1] > 0 if isinstance(tk, np.ndarray) else any(len(x) > 0 for x in tk)
        if has_targets:
            targets_k = _as_long(tk)

    # REV-058: the dense `(N, S, F)` block that used to be built here was never
    # persisted by `save_tensorised` -- all 76.8 GB of it across the three splits was
    # computed and discarded. `RecDataset` now keeps the `(vocab, F)` matrix and
    # gathers per item, so tensorisation only has to hand it the matrix.
    return RecDataset(sequences, targets, user_ids, lengths, targets_k, metadata_matrix)


def save_tensorised(dataset: RecDataset, path: str) -> None:
    data = {
        "sequences": dataset.sequences,
        "targets": dataset.targets,
        "user_ids": dataset.user_ids,
        "lengths": dataset.lengths,
        "targets_k": dataset.targets_k,
    }
    torch.save(data, path)


def load_tensorised(
    path: str,
    metadata_matrix: Optional[np.ndarray] = None,
    max_seq_length: Optional[int] = None,
) -> RecDataset:
    """Load a tensorised split, optionally narrowing it to `max_seq_length`.

    Sequences are stored at the preprocessing-time width. Ablation H2 (FR-13) sweeps
    `sequence_length_grid`, and a config with a SHORTER `max_seq_length` than the stored
    width previously indexed the position embedding out of bounds
    ("index 25 is out of bounds for dimension with size 25"). Under the left-padding
    convention the most recent interactions are at the END, so narrowing keeps the last
    `max_seq_length` positions -- the recent history, which is what a shorter context
    window means.
    """
    data = torch.load(path, weights_only=True)

    if max_seq_length is not None and data["sequences"].shape[1] > max_seq_length:
        data = dict(data)
        data["sequences"] = data["sequences"][:, -max_seq_length:].contiguous()
        if data.get("lengths") is not None:
            data["lengths"] = data["lengths"].clamp(max=max_seq_length)

    return RecDataset(
        sequences=data["sequences"],
        targets=data["targets"],
        user_ids=data["user_ids"],
        lengths=data["lengths"],
        targets_k=data.get("targets_k"),
        metadata_matrix=metadata_matrix,
    )


def build_dataloader(dataset: RecDataset, config: Config, shuffle: bool = True) -> DataLoader:
    worker_init = seed_worker if config.num_workers > 0 else None
    persistent = config.persistent_workers and config.num_workers > 0

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(config.seed),
    )


class GlobalNegativePool:
    def __init__(self, warm_items: Set[int], seed: int = 42):
        self.warm_items = sorted(warm_items)
        self.seed = seed
        self.pool: Optional[np.ndarray] = None
        self._fingerprint: Optional[str] = None

    def build_once(self, pool_size: int) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        n = len(self.warm_items)
        if pool_size <= n:
            indices = rng.choice(n, size=pool_size, replace=False)
        else:
            full_repeats = pool_size // n
            remainder = pool_size % n
            parts = [np.arange(n)] * full_repeats
            if remainder > 0:
                parts.append(rng.choice(n, size=remainder, replace=False))
            indices = np.concatenate(parts)
            rng.shuffle(indices)

        items = np.array(self.warm_items)
        self.pool = items[indices]
        self._fingerprint = hashlib.sha256(self.pool.tobytes()).hexdigest()[:16]
        return self.pool

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            raise RuntimeError("Pool not built yet. Call build_once() first.")
        return self._fingerprint

    def save(self, path: str) -> None:
        if self.pool is None:
            raise RuntimeError("Pool not built yet.")
        np.save(path, self.pool)
        meta = {"fingerprint": self._fingerprint, "seed": self.seed, "size": len(self.pool)}
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        """Draw `n` items uniformly at random from the pool.

        REV-040: T8.1 specifies this method. Without it the training loop reached for a
        raw contiguous slice of the pool instead (training-pipeline REV-012).
        """
        if self.pool is None:
            raise RuntimeError("Pool not built yet. Call build_once() or load() first.")
        idx = rng.randint(0, len(self.pool), size=n)
        return self.pool[idx]

    @classmethod
    def load(cls, path: str, warm_items: Set[int], verify: bool = True) -> "GlobalNegativePool":
        pool_data = np.load(path)
        meta_path = path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        obj = cls(warm_items, seed=meta["seed"])
        obj.pool = pool_data
        obj._fingerprint = meta["fingerprint"]

        if verify:
            # REV-040: the "reuse if fingerprint matches" behaviour T8.1 describes was
            # never enforced -- a pool built under a different seed or item vocabulary
            # was loaded without complaint. Recompute and compare.
            actual = hashlib.sha256(pool_data.tobytes()).hexdigest()[:16]
            if actual != meta["fingerprint"]:
                raise RuntimeError(
                    f"Negative pool fingerprint mismatch for {path}: recorded "
                    f"{meta['fingerprint']}, recomputed {actual}. The pool file and its "
                    f"meta file disagree -- regenerate with --stage preprocess."
                )
        return obj

    def matches_fingerprint(self, expected: str) -> bool:
        return self._fingerprint == expected


def filter_negatives(
    pool: np.ndarray,
    target: int,
    history_set: Set[int],
    num_negatives: int,
    padding_id: int = 0,
    mask_id: int = -1,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Draw `num_negatives` items from `pool`, excluding target / pad / mask / history.

    REV-030: this used to scan the pool from index 0 and take the first admissible
    items. Because the pool ordering is fixed, nearly every example received the SAME
    negatives, so the difficulty of the ranking task was effectively constant across
    examples and was set by whichever items sat at the head of the pool. Negatives are
    now sampled uniformly at random from the whole pool via rejection sampling.

    `rng` should be supplied by the caller so a run is reproducible. When it is omitted
    a RandomState seeded from the target is used: deterministic, and still varied across
    examples -- never the shared-prefix behaviour this finding was about.
    """
    if rng is None:
        rng = np.random.RandomState(int(target) % (2 ** 31 - 1))

    forbidden = {int(target), int(padding_id), int(mask_id)}
    forbidden.update(int(h) for h in history_set)

    n_pool = len(pool)
    if n_pool == 0:
        raise RuntimeError("Negative pool is empty.")

    selected: List[int] = []
    seen: Set[int] = set()
    # Oversample per attempt so most draws finish in one or two rounds.
    for _ in range(32):
        need = num_negatives - len(selected)
        if need <= 0:
            break
        draw = pool[rng.randint(0, n_pool, size=min(n_pool, need * 4))]
        for item in draw:
            item = int(item)
            if item in forbidden or item in seen:
                continue
            seen.add(item)
            selected.append(item)
            if len(selected) >= num_negatives:
                break

    if len(selected) < num_negatives:
        # Fall back to an exhaustive pass before declaring exhaustion, so the error is
        # only raised when the pool genuinely cannot satisfy the request.
        for item in pool:
            item = int(item)
            if item in forbidden or item in seen:
                continue
            seen.add(item)
            selected.append(item)
            if len(selected) >= num_negatives:
                break

    if len(selected) < num_negatives:
        raise RuntimeError(
            f"Negative pool exhausted: needed {num_negatives} valid negatives, "
            f"found only {len(selected)}. Increase pool_size or reduce constraints."
        )
    return np.array(selected[:num_negatives], dtype=np.int64)


def sample_training_negatives(
    pool: np.ndarray,
    targets: np.ndarray,
    user_ids: np.ndarray,
    history_arrays: Dict[int, np.ndarray],
    num_negatives: int,
    num_items: int,
    rng: np.random.RandomState,
    max_rounds: int = 8,
    _buffer: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised per-example negative sampling for a training batch.

    REV-012: the training loop used a raw contiguous slice of the global pool, so an
    example's own target could be sampled as its own negative and `np.resize` tiled
    duplicates near the end of the pool. This excludes, per example, the target, the
    padding and mask IDs, and that user's training history -- the same constraints
    `filter_negatives` applies at evaluation time -- while staying fast enough for the
    hot path by using a membership bitmap instead of Python set lookups.

    Returns an ``(B, num_negatives)`` int64 array.
    """
    B = len(targets)
    width = num_items + 2

    if _buffer is not None and _buffer.shape == (B, width):
        forbidden = _buffer
        forbidden.fill(False)
    else:
        forbidden = np.zeros((B, width), dtype=bool)

    forbidden[:, 0] = True                 # padding ID
    forbidden[:, num_items + 1] = True     # mask ID

    hists = [history_arrays.get(int(u)) for u in user_ids]
    lengths = [0 if h is None else len(h) for h in hists]
    total = sum(lengths)
    if total:
        rows = np.repeat(np.arange(B), lengths)
        cols = np.concatenate([h for h in hists if h is not None and len(h)])
        np.clip(cols, 0, width - 1, out=cols)
        forbidden[rows, cols] = True

    tgt = np.asarray(targets, dtype=np.int64)
    forbidden[np.arange(B), np.clip(tgt, 0, width - 1)] = True

    n_pool = len(pool)
    neg = pool[rng.randint(0, n_pool, size=(B, num_negatives))].astype(np.int64)

    row_idx = np.arange(B)[:, None]
    for _ in range(max_rounds):
        bad = forbidden[row_idx, neg]
        n_bad = int(bad.sum())
        if n_bad == 0:
            break
        neg[bad] = pool[rng.randint(0, n_pool, size=n_bad)]

    # Any residual collisions after max_rounds are replaced deterministically by a
    # scan, so no example ever trains against its own target.
    bad = forbidden[row_idx, neg]
    if bad.any():
        for r, c in zip(*np.nonzero(bad)):
            for item in pool:
                if not forbidden[r, int(item)]:
                    neg[r, c] = int(item)
                    break
    return neg


def generate_eval_candidates(
    examples: dict,
    pool: np.ndarray,
    user_history: Dict[int, List[int]],
    num_negatives: int,
    padding_id: int = 0,
    mask_id: int = -1,
    seed: int = 42,
    out_path: Optional[str] = None,
) -> np.ndarray:
    """Build `[target] + negatives` candidate sets, one per example.

    Deterministic given `seed` and the example order. REV-041: callers should persist
    the result with `save_eval_candidates` and load it at every evaluation site rather
    than regenerating, so cross-model identity is structural rather than incidental.

    REV-087: rows are written into ONE preallocated `(N, 1 + num_negatives)` array
    instead of being accumulated as N separate `(101,)` arrays. The list-of-arrays form
    cost ~0.9 GB for the test split in per-array overhead alone, and `save_eval_candidates`
    then copied it again into a contiguous matrix. Passing `out_path` allocates that
    destination as an on-disk `.npy` memmap, so the resident set is bounded by the OS
    page cache rather than by the split size; `save_eval_candidates` recognises such a
    memmap and fingerprints it in place without re-writing.
    """
    rng = np.random.RandomState(seed)
    n = len(examples["targets"])
    width = 1 + num_negatives
    if out_path is not None:
        out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.int64, shape=(n, width),
        )
    else:
        out = np.empty((n, width), dtype=np.int64)

    for i in range(n):
        target = examples["targets"][i]
        user_id = examples["user_ids"][i]
        history = set(user_history.get(user_id, []))
        negs = filter_negatives(
            pool, target, history, num_negatives, padding_id, mask_id, rng=rng,
        )
        out[i, 0] = target
        out[i, 1:] = negs
    return out


def _fingerprint_rows(matrix: np.ndarray, block_rows: int = 8192) -> str:
    """SHA-256 over the matrix, consumed in row blocks.

    REV-087: hashing `matrix.tobytes()` materialises a full second copy of the array --
    0.86 GB for the test split, at the same point in the pipeline as the other peaks.
    Blocks keep the transient bounded while producing the same digest, since the blocks
    concatenate to exactly the C-contiguous byte stream `tobytes()` would have built.
    """
    digest = hashlib.sha256()
    for start in range(0, matrix.shape[0], block_rows):
        digest.update(np.ascontiguousarray(matrix[start:start + block_rows]).tobytes())
    return digest.hexdigest()[:16]


def save_eval_candidates(candidates, path: str) -> str:
    """Persist candidate sets as one (N, 1 + num_negatives) matrix plus a fingerprint.

    Accepts either a `(N, C)` array -- including the memmap `generate_eval_candidates`
    writes when given `out_path`, which is already backed by `path` and so is only
    flushed rather than copied and re-written -- or the legacy list of per-example arrays.
    """
    if isinstance(candidates, np.memmap) and candidates.filename is not None and (
        os.path.realpath(candidates.filename) == os.path.realpath(path)
    ):
        candidates.flush()
        matrix = candidates
    else:
        matrix = np.asarray(candidates, dtype=np.int64)
        np.save(path, matrix)

    fingerprint = _fingerprint_rows(matrix)
    meta = {"fingerprint": fingerprint, "shape": list(matrix.shape)}
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    return fingerprint


def load_eval_candidates(path: str, verify: bool = True) -> np.ndarray:
    matrix = np.load(path)
    meta_path = path + ".meta.json"
    if verify and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        # REV-087: block-wise, so verification does not double the resident set.
        actual = _fingerprint_rows(matrix)
        if actual != meta["fingerprint"]:
            raise RuntimeError(
                f"Candidate set fingerprint mismatch for {path}: recorded "
                f"{meta['fingerprint']}, recomputed {actual}."
            )
    return matrix


def load_or_build_eval_candidates(
    path: str,
    examples: dict,
    pool: np.ndarray,
    user_history: Dict[int, List[int]],
    config: Config,
    expected_rows: Optional[int] = None,
) -> np.ndarray:
    """Return the stored candidate matrix for a split, regenerating only as a fallback.

    REV-093 / REV-111 / REV-112: the Constraints say fixed evaluation candidate sets are
    "generated once and reused -- never regenerated per model", and preprocessing writes
    `val_candidates.npy` / `test_candidates.npy` for exactly that. Several evaluation
    entry points rebuilt them per call instead, and most did so without passing `rng=`,
    so `filter_negatives` fell back to a RandomState seeded from the TARGET ITEM: two
    examples sharing a target received byte-identical negatives regardless of user or
    history, and none of them matched the stored sets. Every such site now routes through
    here, so the stored matrix is used when it exists and the fallback is seeded from
    `config.seed` rather than silently from 42.
    """
    if os.path.exists(path):
        matrix = load_eval_candidates(path)
        if expected_rows is not None and matrix.shape[0] != expected_rows:
            raise RuntimeError(
                f"Pre-stored candidate sets at {path} have {matrix.shape[0]} rows but "
                f"the split has {expected_rows} examples. Re-run --stage preprocess."
            )
        return matrix

    logger.warning(
        "%s not found -- regenerating evaluation candidate sets from seed %d. Re-run "
        "--stage preprocess so every model evaluates against the same stored sets.",
        path, config.seed,
    )
    return generate_eval_candidates(
        examples, pool, user_history, config.num_sampled_negatives,
        padding_id=0, mask_id=config.num_items + 1, seed=config.seed,
    )
