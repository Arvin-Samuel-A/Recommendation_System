import hashlib
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

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
        metadata_matrix=None,
    ):
        # `targets` is `(N,)` for an evaluation split (the held-out item) and `(N, S)` for
        # the training split (the next item at every position, 0 at padding).
        self.sequences = sequences
        self.targets = targets
        self.user_ids = user_ids
        self.lengths = lengths
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
        if self.metadata_matrix is not None:
            item["metadata"] = self.metadata_matrix[self.sequences[idx]]
        return item


def _left_pad(seq: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_id] * (max_len - len(seq)) + seq


def chunk_user_history(items: List[int], max_seq_length: int) -> List[Tuple[List[int], List[int]]]:
    """Cut one training history into LLM-style `(inputs, targets)` chunks.

    Every item after the first is a next-item target exactly once: a chunk of `m <= S`
    predictions has inputs `items[a:a+m]` and targets `items[a+1:a+m+1]`. Chunks are cut
    from the END, so the most recent `S` predictions share one chunk and only the oldest
    chunk is short. Returned oldest first. A history of fewer than 2 items yields none.
    """
    S = int(max_seq_length)
    chunks = []
    end = len(items) - 1          # index of the last target
    while end >= 1:
        start = max(0, end - S)   # first input index
        chunks.append((list(items[start:end]), list(items[start + 1:end + 1])))
        end = start
    chunks.reverse()
    return chunks


def chunk_training_histories(
    train_user_history: Dict[int, List[int]], max_seq_length: int,
) -> dict:
    """Left-padded `(N, S)` inputs and per-position targets for every training chunk.

    `targets[i, t]` is the item following `sequences[i, t]`; padded positions hold 0 in
    both. Users are visited in sorted order, so the arrays are a pure function of the
    histories and the width.
    """
    S = int(max_seq_length)
    users = sorted(train_user_history)
    n = sum(max(0, -(-(len(train_user_history[u]) - 1) // S)) for u in users)
    out = {
        "sequences": np.zeros((n, S), dtype=np.int32),
        "targets": np.zeros((n, S), dtype=np.int32),
        "user_ids": np.zeros(n, dtype=np.int32),
        "lengths": np.zeros(n, dtype=np.int32),
    }
    i = 0
    for user_id in users:
        for inputs, targets in chunk_user_history(train_user_history[user_id], S):
            m = len(inputs)
            out["sequences"][i, S - m:] = inputs
            out["targets"][i, S - m:] = targets
            out["user_ids"][i] = user_id
            out["lengths"][i] = m
            i += 1
    assert i == n, f"chunk count disagrees with the preallocation: filled {i}, allocated {n}"
    return out


def _fill_left_padded(row: np.ndarray, context: List[int], max_len: int) -> None:
    """Write `context` into a preallocated left-padded row, truncating to the tail."""
    row[:] = 0
    tail = context[-max_len:] if len(context) > max_len else context
    if tail:
        row[max_len - len(tail):] = tail


def _alloc_split(n: int, max_seq_length: int) -> dict:
    """Preallocated int32 arrays for one evaluation split.

    REV-086: every example used to be a Python `list` of Python `int` objects -- 482
    bytes per padded 50-item sequence against 200 for the equivalent int32 row. int32 is
    sufficient for any item id here.
    """
    return {
        "sequences": np.zeros((n, max_seq_length), dtype=np.int32),
        "targets": np.zeros(n, dtype=np.int32),
        "user_ids": np.zeros(n, dtype=np.int32),
        "lengths": np.zeros(n, dtype=np.int32),
    }


def generate_sequences(
    user_sequences: Dict[int, List[int]],
    train_seqs: Dict[int, List[int]],
    val_seqs: Dict[int, List[int]],
    test_seqs: Dict[int, List[int]],
    config: Config,
    width: Optional[int] = None,
) -> Tuple[dict, dict, Dict[int, List[int]], Dict[int, List[int]]]:
    """Build the validation and test examples and the two per-user histories.

    One example per held-out item: its context is everything the user did before it
    (train for val; train + val for test), left-padded to the most recent `width`
    positions. Under leave-one-out that is one val and one test example per user.
    Training examples are not built here: `load_train_dataset` chunks
    `train_user_history` at the consumer's own width.

    `width` is the stored sequence width (specs/ablation-run D7); it defaults to
    `config.max_seq_length`. Preprocessing passes `stored_seq_length(config)` so every
    Ablation H2 arm can narrow to its own `max_seq_length`.
    """
    n_val = sum(len(val_seqs.get(u, [])) for u in user_sequences)
    n_test = sum(len(test_seqs.get(u, [])) for u in user_sequences)
    S = config.max_seq_length if width is None else int(width)

    val_examples = _alloc_split(n_val, S)
    test_examples = _alloc_split(n_test, S)

    train_user_history: Dict[int, List[int]] = {}
    eval_user_history: Dict[int, List[int]] = {}

    vi = si = 0

    for user_id in user_sequences:
        train_items = train_seqs.get(user_id, [])
        val_items = val_seqs.get(user_id, [])
        test_items = test_seqs.get(user_id, [])

        train_user_history[user_id] = list(train_items)
        eval_user_history[user_id] = list(train_items) + list(val_items)

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

    assert (vi, si) == (n_val, n_test), (
        f"example counts disagree with the preallocation: filled {(vi, si)}, "
        f"allocated {(n_val, n_test)}"
    )

    assert train_user_history is not eval_user_history, (
        "train_user_history and eval_user_history must be distinct objects"
    )

    logger.info(
        "Generated sequences: %d training interactions, val=%d, test=%d",
        sum(len(v) for v in train_user_history.values()), n_val, n_test,
    )
    return val_examples, test_examples, train_user_history, eval_user_history


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

    # REV-058: the dense `(N, S, F)` block that used to be built here was never
    # persisted by `save_tensorised` -- all 76.8 GB of it across the three splits was
    # computed and discarded. `RecDataset` now keeps the `(vocab, F)` matrix and
    # gathers per item, so tensorisation only has to hand it the matrix.
    return RecDataset(sequences, targets, user_ids, lengths, metadata_matrix)


def save_tensorised(dataset: RecDataset, path: str) -> None:
    data = {
        "sequences": dataset.sequences,
        "targets": dataset.targets,
        "user_ids": dataset.user_ids,
        "lengths": dataset.lengths,
    }
    torch.save(data, path)


def stored_seq_length(config: Config) -> int:
    """The width preprocessing stores every split at (specs/ablation-run FR-7, D7).

    `max(sequence_length_grid + [max_seq_length])`, so every H2 arm can narrow to its own
    width and none silently trains on fewer positions than it asks for.
    """
    return max([int(config.max_seq_length)] + [int(w) for w in config.sequence_length_grid])


def _width_error(source: str, stored: int, requested: int) -> ValueError:
    return ValueError(
        f"{source} is stored at sequence width {stored} but {requested} positions were "
        f"requested: re-run `python main.py --stage preprocess` with "
        f"--sequence_length_grid covering {requested} (stored width = "
        f"max(sequence_length_grid + [max_seq_length]))."
    )


def narrow_examples(examples: dict, max_seq_length: int, source: str = "examples") -> dict:
    """A shallow copy of an `*_examples.pkl` dict narrowed to `max_seq_length` (D7).

    Keeps the LAST `max_seq_length` positions (the most recent history under left
    padding) and clamps `lengths`, which is exactly what generating the split at that
    width yields. Raises when more positions are requested than are stored.
    """
    sequences = examples["sequences"]
    if isinstance(sequences, np.ndarray):
        stored = sequences.shape[1] if sequences.ndim == 2 else 0
    else:
        stored = len(sequences[0]) if len(sequences) else 0
    N = int(max_seq_length)
    if len(sequences) and N > stored:
        raise _width_error(source, stored, N)
    if N >= stored:
        return dict(examples)
    out = dict(examples)
    if isinstance(sequences, np.ndarray):
        out["sequences"] = sequences[:, -N:]
    else:
        out["sequences"] = [list(row)[-N:] for row in sequences]
    lengths = examples.get("lengths")
    if lengths is not None:
        if isinstance(lengths, np.ndarray):
            out["lengths"] = np.minimum(lengths, N)
        else:
            out["lengths"] = [min(int(x), N) for x in lengths]
    return out


def load_examples(path: str, max_seq_length: int) -> dict:
    """Unpickle an `*_examples.pkl` split and narrow it to `max_seq_length` (D7).

    The ONLY sanctioned reader of these artifacts outside `data/preprocessing.py`
    (enforced by `tests/test_sequence_generator.py::TestEveryReaderNarrows`).
    """
    with open(path, "rb") as f:
        examples = pickle.load(f)
    return narrow_examples(examples, max_seq_length, source=path)


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

    # specs/ablation-run FR-7: asking for MORE positions than are stored used to return
    # the narrower data silently, so a seq200 arm trained on 50 items.
    stored = data["sequences"].shape[1]
    if max_seq_length is not None and data["sequences"].shape[0] and max_seq_length > stored:
        raise _width_error(path, stored, max_seq_length)

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
        metadata_matrix=metadata_matrix,
    )


def load_train_dataset(
    processed_dir: str,
    max_seq_length: int,
    metadata_matrix: Optional[np.ndarray] = None,
) -> RecDataset:
    """The training split, chunked at `max_seq_length` from `train_user_history.pkl`.

    Each row is one chunk of a user's chronological training history with a next-item
    target at every real position (`targets` is `(N, S)`, 0 at padding). Built at load
    time, so every reader gets chunks of exactly its own width.
    """
    with open(os.path.join(processed_dir, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    examples = chunk_training_histories(train_user_history, max_seq_length)
    return tensorise_examples(examples, metadata_matrix)


class EpochPermutationSampler(Sampler):
    """One `randperm` of the dataset per epoch, drawn from `generator`.

    `RandomSampler` draws a second (empty) permutation once its iterator is EXHAUSTED.
    An epoch cut short by `max_steps_per_epoch` exhausts it only when worker prefetch
    happens to reach the end of the epoch, so the generator -- and every later epoch's
    order -- depended on `num_workers`. This sampler consumes exactly one draw per epoch
    whether or not the epoch is read to the end. The first permutation is the one
    `RandomSampler` would have drawn.
    """

    def __init__(self, n: int, generator: torch.Generator):
        self.n = int(n)
        self.generator = generator

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        yield from torch.randperm(self.n, generator=self.generator).tolist()


def build_dataloader(dataset: RecDataset, config: Config, shuffle: bool = True) -> DataLoader:
    worker_init = seed_worker if config.num_workers > 0 else None
    persistent = config.persistent_workers and config.num_workers > 0
    generator = torch.Generator().manual_seed(config.seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=EpochPermutationSampler(len(dataset), generator) if shuffle else None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init,
        generator=generator,
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
