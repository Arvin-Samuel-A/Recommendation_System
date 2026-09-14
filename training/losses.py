import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SampledSoftmaxLoss(nn.Module):
    def forward(
        self,
        user_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        item_emb_weight: torch.Tensor,
    ) -> torch.Tensor:
        pos_emb = item_emb_weight[pos_ids]
        neg_emb = item_emb_weight[neg_ids]

        pos_score = (user_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_score, neg_scores], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MaskedItemLoss(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_positions: torch.Tensor,
        true_item_ids: torch.Tensor,
        item_emb_weight: torch.Tensor,
    ) -> torch.Tensor:
        masked_hidden = hidden_states[mask_positions]
        if masked_hidden.shape[0] == 0:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        masked_targets = true_item_ids[mask_positions]
        logits = masked_hidden @ item_emb_weight.T
        return F.cross_entropy(logits, masked_targets)


class MultiStepLoss(nn.Module):
    def __init__(self, hidden_dim: int, multi_step_k: int):
        super().__init__()
        self.step_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(multi_step_k)
        ])
        self.multi_step_k = multi_step_k

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        targets_k: torch.Tensor,
        item_emb_weight: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = all_hidden_states.shape
        total_loss = torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        valid_steps = 0

        for step in range(self.multi_step_k):
            pos = S - 1 - step
            if pos < 0:
                continue

            hidden_at_pos = all_hidden_states[:, pos, :]
            projected = self.step_projections[step](hidden_at_pos)

            step_targets = targets_k[:, step]
            valid_mask = step_targets != 0

            if valid_mask.sum() == 0:
                continue

            valid_hidden = projected[valid_mask]
            valid_targets = step_targets[valid_mask]

            logits = valid_hidden @ item_emb_weight.T
            step_loss = F.cross_entropy(logits, valid_targets)
            total_loss = total_loss + step_loss
            valid_steps += 1

        if valid_steps == 0:
            return torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        return total_loss / valid_steps


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_orig: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
        z_orig = F.normalize(z_orig, p=2, dim=-1)
        z_aug = F.normalize(z_aug, p=2, dim=-1)

        z = torch.cat([z_orig, z_aug], dim=0)
        B2 = z.shape[0]

        sim = z @ z.T / self.temperature

        mask = torch.eye(B2, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        B = B2 // 2
        labels = torch.cat([
            torch.arange(B, B2, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


def apply_masking(
    seq: torch.Tensor,
    mask_prob: float,
    mask_id: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_seq = seq.clone()
    true_item_ids = seq.clone()

    real_items = seq != 0
    mask_candidates = real_items.float()

    rand_vals = torch.rand(seq.shape, generator=rng, device=seq.device)
    mask_positions = (rand_vals < mask_prob) & real_items

    for i in range(seq.shape[0]):
        row_real = real_items[i]
        row_mask = mask_positions[i]
        if row_real.sum() > 0 and row_mask.sum() >= row_real.sum():
            real_indices = row_real.nonzero(as_tuple=True)[0]
            keep_idx = real_indices[torch.randint(len(real_indices), (1,), generator=rng)]
            mask_positions[i, keep_idx] = False

    masked_seq[mask_positions] = mask_id
    return masked_seq, mask_positions, true_item_ids


def static_weighted_sum(active_losses: dict, config) -> torch.Tensor:
    total = active_losses.get("next_item", torch.tensor(0.0))
    if "masked_item" in active_losses:
        total = total + config.masked_lm_weight * active_losses["masked_item"]
    if "multi_step" in active_losses:
        total = total + config.multi_step_weight * active_losses["multi_step"]
    if "contrastive" in active_losses:
        total = total + config.contrastive_weight * active_losses["contrastive"]
    return total
