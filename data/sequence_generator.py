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


class RecDataset(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        user_ids: torch.Tensor,
        lengths: torch.Tensor,
        targets_k: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ):
        self.sequences = sequences
        self.targets = targets
        self.user_ids = user_ids
        self.lengths = lengths
        self.targets_k = targets_k
        self.metadata = metadata

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
        if self.metadata is not None:
            item["metadata"] = self.metadata[idx]
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


def generate_sequences(
    user_sequences: Dict[int, List[int]],
    train_seqs: Dict[int, List[int]],
    val_seqs: Dict[int, List[int]],
    test_seqs: Dict[int, List[int]],
    config: Config,
) -> Tuple[dict, dict, dict, Dict[int, List[int]], Dict[int, List[int]]]:
    train_examples = {"sequences": [], "targets": [], "user_ids": [], "lengths": [], "targets_k": []}
    val_examples = {"sequences": [], "targets": [], "user_ids": [], "lengths": [], "targets_k": []}
    test_examples = {"sequences": [], "targets": [], "user_ids": [], "lengths": [], "targets_k": []}

    train_user_history: Dict[int, List[int]] = {}
    eval_user_history: Dict[int, List[int]] = {}

    for user_id in user_sequences:
        train_items = train_seqs.get(user_id, [])
        val_items = val_seqs.get(user_id, [])
        test_items = test_seqs.get(user_id, [])

        train_user_history[user_id] = list(train_items)
        eval_user_history[user_id] = list(train_items) + list(val_items)

        if len(train_items) >= 2:
            seqs, tgts, tgts_k = generate_sliding_window_examples(
                train_items, config.max_seq_length, config.multi_step_k,
                split_end_idx=len(train_items),
            )
            for s, t, tk in zip(seqs, tgts, tgts_k):
                train_examples["sequences"].append(s)
                train_examples["targets"].append(t)
                train_examples["user_ids"].append(user_id)
                train_examples["lengths"].append(min(len(train_items) - 1, config.max_seq_length))
                train_examples["targets_k"].append(tk)

        if val_items:
            val_context = list(train_items)
            for i, item in enumerate(val_items):
                padded = _left_pad(val_context, config.max_seq_length)
                val_examples["sequences"].append(padded)
                val_examples["targets"].append(item)
                val_examples["user_ids"].append(user_id)
                val_examples["lengths"].append(min(len(val_context), config.max_seq_length))
                val_examples["targets_k"].append([0] * config.multi_step_k)
                val_context.append(item)

        if test_items:
            test_context = list(train_items) + list(val_items)
            for i, item in enumerate(test_items):
                padded = _left_pad(test_context, config.max_seq_length)
                test_examples["sequences"].append(padded)
                test_examples["targets"].append(item)
                test_examples["user_ids"].append(user_id)
                test_examples["lengths"].append(min(len(test_context), config.max_seq_length))
                test_examples["targets_k"].append([0] * config.multi_step_k)
                test_context.append(item)

    assert train_user_history is not eval_user_history, (
        "train_user_history and eval_user_history must be distinct objects"
    )

    logger.info(
        "Generated sequences: train=%d, val=%d, test=%d",
        len(train_examples["sequences"]),
        len(val_examples["sequences"]),
        len(test_examples["sequences"]),
    )
    return train_examples, val_examples, test_examples, train_user_history, eval_user_history


def tensorise_examples(
    examples: dict,
    metadata_matrix: Optional[np.ndarray] = None,
) -> RecDataset:
    sequences = torch.tensor(examples["sequences"], dtype=torch.long)
    targets = torch.tensor(examples["targets"], dtype=torch.long)
    user_ids = torch.tensor(examples["user_ids"], dtype=torch.long)
    lengths = torch.tensor(examples["lengths"], dtype=torch.long)

    targets_k = None
    if examples.get("targets_k") and any(len(tk) > 0 for tk in examples["targets_k"]):
        targets_k = torch.tensor(examples["targets_k"], dtype=torch.long)

    meta_tensor = None
    if metadata_matrix is not None:
        seq_np = np.array(examples["sequences"])
        meta_tensor = torch.tensor(
            metadata_matrix[seq_np], dtype=torch.float32
        )

    return RecDataset(sequences, targets, user_ids, lengths, targets_k, meta_tensor)


def save_tensorised(dataset: RecDataset, path: str) -> None:
    data = {
        "sequences": dataset.sequences,
        "targets": dataset.targets,
        "user_ids": dataset.user_ids,
        "lengths": dataset.lengths,
        "targets_k": dataset.targets_k,
    }
    torch.save(data, path)


def load_tensorised(path: str, metadata_matrix: Optional[np.ndarray] = None) -> RecDataset:
    data = torch.load(path, weights_only=True)
    meta_tensor = None
    if metadata_matrix is not None:
        seq_np = data["sequences"].numpy()
        meta_tensor = torch.tensor(metadata_matrix[seq_np], dtype=torch.float32)

    return RecDataset(
        sequences=data["sequences"],
        targets=data["targets"],
        user_ids=data["user_ids"],
        lengths=data["lengths"],
        targets_k=data.get("targets_k"),
        metadata=meta_tensor,
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

    @classmethod
    def load(cls, path: str, warm_items: Set[int]) -> "GlobalNegativePool":
        pool_data = np.load(path)
        meta_path = path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        obj = cls(warm_items, seed=meta["seed"])
        obj.pool = pool_data
        obj._fingerprint = meta["fingerprint"]
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
) -> np.ndarray:
    forbidden = {target, padding_id, mask_id} | history_set
    valid = []
    for item in pool:
        if int(item) not in forbidden:
            valid.append(int(item))
            if len(valid) >= num_negatives:
                break

    if len(valid) < num_negatives:
        raise RuntimeError(
            f"Negative pool exhausted: needed {num_negatives} valid negatives, "
            f"found only {len(valid)}. Increase pool_size or reduce constraints."
        )
    return np.array(valid[:num_negatives], dtype=np.int64)


def generate_eval_candidates(
    examples: dict,
    pool: np.ndarray,
    user_history: Dict[int, List[int]],
    num_negatives: int,
    padding_id: int = 0,
    mask_id: int = -1,
) -> List[np.ndarray]:
    candidates = []
    for i in range(len(examples["targets"])):
        target = examples["targets"][i]
        user_id = examples["user_ids"][i]
        history = set(user_history.get(user_id, []))
        negs = filter_negatives(pool, target, history, num_negatives, padding_id, mask_id)
        candidate_set = np.concatenate([[target], negs])
        candidates.append(candidate_set)
    return candidates
