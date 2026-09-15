"""Retrieval-then-rerank candidate generation (FR-2, step 18.1).

REV-064: `conventions.md`'s file layout named this module, `tasks.md` T1.3 was marked
complete, and `use_ann` / `ann_candidates` were declared in `configs/config.py` -- but the
module did not exist and neither flag was read anywhere, so FR-2 and AC-09 were unmet
while the task board reported otherwise.

Retrieval is brute force: the user embedding is scored against the whole item embedding
table. The project forbids external recommendation libraries, and FR-2's requirement is
the force-inclusion protocol and the reported rate, not an approximate index -- an exact
top-K is the strictest version of the same contract.
"""

import logging
from typing import Dict, List, Optional, Sequence, Set

import torch

from configs.config import Config
from models.transformer import l2_normalize

logger = logging.getLogger(__name__)


def _user_embeddings(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The `(B, D)` query embedding, built exactly as the scoring path builds it."""
    all_hidden = model(seqs, metadata=metadata, return_all_hidden=True)
    pooled = all_hidden[:, -1, :]
    if moe_layer is not None and moe_layer.variant != "none":
        Z_ut, _ = moe_layer(all_hidden, pooled, 0)
        user_emb = Z_ut[:, -1, :]
    else:
        user_emb = pooled
    return l2_normalize(user_emb) if config.normalize_embeddings else user_emb


def retrieve_candidates(
    model: torch.nn.Module,
    eval_sequences: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    user_ids: Optional[Sequence[int]] = None,
    user_history: Optional[Dict[int, Sequence[int]]] = None,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata_matrix: Optional[torch.Tensor] = None,
    num_items: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, object]:
    """Retrieve `ann_candidates` items per example, per FR-2.

    The protocol, in order:
      1. score every item in the vocabulary against the user embedding;
      2. filter invalid IDs (padding 0 and MASK) and items the user has already seen;
      3. take the top `ann_candidates`;
      4. force-include the target, replacing the LOWEST-scoring candidate when the
         retrieval missed it;
      5. report the fraction of examples that needed step 4.

    Returns `{"candidates": (N, ann_candidates) LongTensor, "force_inclusion_rate": float,
    "num_force_included": int, "num_examples": int}`. The target is placed at column 0, so
    the result drops straight into `evaluate_model`, which scores `target_idx=0`.
    """
    model.eval()
    if moe_layer is not None:
        moe_layer.eval()

    n = eval_sequences.shape[0]
    k = int(config.ann_candidates)
    if k < 1:
        raise ValueError(f"ann_candidates must be >= 1, got {k}")

    param_device = next(model.parameters()).device
    item_weight = model.item_emb_weight
    vocab_size = item_weight.shape[0]
    resolved_items = int(num_items if num_items is not None else config.num_items)
    if resolved_items <= 0:
        raise ValueError(
            "num_items is unresolved, so valid item IDs cannot be determined. "
            "Load dataset_stats.json before retrieving."
        )
    mask_id = resolved_items + 1

    # Padding (0) and MASK are not items and must never be retrieved.
    invalid_ids = torch.zeros(vocab_size, dtype=torch.bool, device=param_device)
    invalid_ids[0] = True
    if mask_id < vocab_size:
        invalid_ids[mask_id] = True
    if resolved_items + 2 < vocab_size:
        invalid_ids[resolved_items + 2:] = True

    bs = batch_size or max(1, config.eval_batch_size)
    out = torch.empty((n, k), dtype=torch.long)
    num_forced = 0

    with torch.no_grad():
        item_embs = item_weight
        if config.normalize_embeddings:
            item_embs = l2_normalize(item_embs)

        for start in range(0, n, bs):
            stop = min(start + bs, n)
            seqs = eval_sequences[start:stop].to(param_device)

            meta = None
            if metadata_matrix is not None:
                meta = metadata_matrix[seqs.to(metadata_matrix.device)].to(
                    device=param_device, dtype=item_weight.dtype,
                )

            user_emb = _user_embeddings(model, seqs, config, moe_layer, meta)
            scores = user_emb.float() @ item_embs.float().T  # (B, vocab)
            scores = scores.masked_fill(invalid_ids.unsqueeze(0), float("-inf"))

            # Filter items the user has already interacted with. The target is exempt --
            # it is force-included below and is by construction a future interaction.
            batch_targets = targets[start:stop].to(param_device).long()
            if user_history is not None and user_ids is not None:
                for row in range(stop - start):
                    seen = user_history.get(int(user_ids[start + row]))
                    if not seen:
                        continue
                    seen_t = torch.as_tensor(
                        list(seen), dtype=torch.long, device=param_device,
                    )
                    scores[row, seen_t] = float("-inf")

            top_scores, top_ids = torch.topk(scores, k, dim=1)

            # Step 4: force-include the target, replacing the lowest-scoring candidate.
            # `topk` returns descending scores, so the last column is the lowest.
            hit = (top_ids == batch_targets.unsqueeze(1)).any(dim=1)
            missed = ~hit
            num_forced += int(missed.sum().item())
            if missed.any():
                top_ids[missed, -1] = batch_targets[missed]

            # Put the target at column 0 so `evaluate_model`'s `target_idx=0` holds,
            # keeping the rest of each row in its retrieved order.
            rows = torch.arange(top_ids.shape[0], device=param_device)
            target_col = (top_ids == batch_targets.unsqueeze(1)).float().argmax(dim=1)
            swapped = top_ids.clone()
            swapped[rows, target_col] = top_ids[:, 0]
            swapped[:, 0] = batch_targets
            out[start:stop] = swapped.cpu()

    rate = float(num_forced) / n if n else 0.0
    logger.info(
        "ANN retrieval: %d/%d examples required target force-inclusion (%.2f%%)",
        num_forced, n, rate * 100,
    )
    return {
        "candidates": out,
        "force_inclusion_rate": rate,
        "num_force_included": num_forced,
        "num_examples": int(n),
        "ann_candidates": k,
    }
