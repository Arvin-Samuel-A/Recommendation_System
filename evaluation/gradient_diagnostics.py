import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_cosine_similarity(
    per_objective_losses: Dict[str, torch.Tensor],
    shared_params: List[nn.Parameter],
) -> torch.Tensor:
    names = list(per_objective_losses.keys())
    T = len(names)

    # REV-128: measure in fp32. The model is bf16 everywhere, and a bf16 cosine of a
    # ~790k-element vector against ITSELF does not return 1.0 -- the real artifact read a
    # diagonal of [1.00078, 1.00391, 0.99844, 1.00078], three entries of which are outside
    # the range a cosine similarity can take. This is one temporary copy on a diagnostic
    # path that runs rarely, and it is the same fp32-at-the-measurement-site pattern
    # `routing_diagnostics` and `local_train` already use.
    grads = {}
    for name, loss in per_objective_losses.items():
        g = torch.autograd.grad(
            loss, shared_params, retain_graph=True, allow_unused=True,
        )
        flat = torch.cat([
            gi.flatten() if gi is not None else torch.zeros_like(p).flatten()
            for gi, p in zip(g, shared_params)
        ])
        grads[name] = flat.float()

    cos_sim = torch.zeros(T, T)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            cos_sim[i, j] = F.cosine_similarity(
                grads[ni].unsqueeze(0), grads[nj].unsqueeze(0),
            ).item()

    return cos_sim


def per_objective_grad_norms(
    per_objective_losses: Dict[str, torch.Tensor],
    shared_params: List[nn.Parameter],
) -> Dict[str, float]:
    norms = {}
    for name, loss in per_objective_losses.items():
        g = torch.autograd.grad(
            loss, shared_params, retain_graph=True, allow_unused=True,
        )
        flat = torch.cat([
            gi.flatten() if gi is not None else torch.zeros_like(p).flatten()
            for gi, p in zip(g, shared_params)
        ])
        # REV-128: same fp32 treatment as the cosines, for consistency of the reported
        # numbers -- a bf16 norm over ~790k elements accumulates visible error.
        norms[name] = flat.float().norm(p=2).item()

    vals = list(norms.values())
    t = torch.tensor(vals)
    mean_norm = t.mean().item()
    std_norm = t.std().item() if len(vals) > 1 else 0.0
    cv = std_norm / (mean_norm + 1e-12)

    return {
        "per_objective": norms,
        "mean": mean_norm,
        "std": std_norm,
        "coefficient_of_variation": cv,
    }


def save_gradient_diagnostics(
    cos_sim: torch.Tensor,
    grad_norms: dict,
    objective_names: List[str],
    config,
    model_name: str,
    source_revision: Optional[dict] = None,
) -> dict:
    report = {
        "cosine_similarity": cos_sim.tolist(),
        "objective_names": objective_names,
        "gradient_norms": grad_norms,
    }
    # AC-09: echo the source_revision of the checkpoint these numbers came from.
    if source_revision is not None:
        report["source_revision"] = source_revision

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # conventions.md: CLI model names are "t-small", file names are "t_small".
    # The plots already normalized; the JSON report did not, so the two halves of
    # the same report landed under different spellings.
    mn = model_name.replace("-", "_")
    path = os.path.join(log_dir, f"gradient_diagnostics_{mn}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    plot_gradient_diagnostics(report, config, model_name)
    return report


def run_gradient_diagnostics(
    model: torch.nn.Module,
    moe_layer: torch.nn.Module,
    dataloader,
    loss_fns: dict,
    shared_params: List[nn.Parameter],
    config,
    model_name: str,
    negative_sampler=None,
    metadata_matrix=None,
    augment_rng=None,
    write_report: bool = True,
    source_revision: Optional[dict] = None,
) -> dict:
    """Per-objective gradient conflict/dominance diagnostics.

    REV-009: (a) `batch["neg_ids"]` is not produced by `build_dataloader` -- it is
    attached by the training loop -- so a plain train_loader used to raise KeyError.
    `negative_sampler` is the shared attachment helper (`attach_training_negatives`),
    which keeps this path identical to the training path while drawing FRESH diagnostic
    batches rather than optimizer-step batches. (b) The contrastive objective was
    omitted, capping the cosine matrix at 3x3 where FR-9 specifies (T, T) over all
    objectives under comparison.
    """
    import random as _random

    from training.losses import apply_masking, mask_metadata
    from data.augmentation import augment_sequence
    from configs.config import OBJECTIVE_ORDER

    if augment_rng is None:
        augment_rng = _random.Random(config.seed)

    # REV-059b: the diagnostic masked from the GLOBAL torch RNG while the training loop
    # uses a run-scoped generator. That made repeated diagnostic runs irreproducible and
    # perturbed the global stream mid-training. Own generator, seeded from config.
    mask_rng = torch.Generator(device="cpu")
    mask_rng.manual_seed(config.seed)

    cos_matrices = []
    norm_records = []
    objective_names = None

    # REV-107 / REV-109: this function used to switch the model to eval mode and
    # never switch it back. The in-training caller compensated immediately
    # afterwards, so the invariant was held by the caller rather than by the
    # function that broke it -- any future site that ran the diagnostic mid-training
    # and did not know to re-enter train mode would silently train without dropout.
    # `moe_layer` was never switched at all, so the two modules disagreed for the
    # duration of the diagnostic. Save both and restore both.
    prev_model_training = model.training
    prev_moe_training = moe_layer.training
    model.eval()
    moe_layer.eval()
    try:
        batch_count = 0
        for batch in dataloader:
            if batch_count >= config.grad_diag_num_batches:
                break

            device = next(model.parameters()).device
            # specs/ablation-run D4: negatives are drawn from the loader's CPU batch
            # BEFORE the host->device copy (identical draws, no device->host round trip).
            if "neg_ids" not in batch:
                if negative_sampler is None:
                    raise ValueError(
                        "run_gradient_diagnostics needs `negative_sampler` when the "
                        "dataloader does not already provide `neg_ids`."
                    )
                batch = negative_sampler(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            seq = batch["sequences"]
            targets = batch["targets"]
            neg_ids = batch["neg_ids"]
            targets_k = batch.get("targets_k")
            metadata = batch.get("metadata")
            mask_id = config.num_items + 1

            if config.use_masked_lm:
                masked_seq, mask_positions, true_ids = apply_masking(
                    seq, config.mask_prob, mask_id, rng=mask_rng,
                )
                # REV-059/REV-059b: this reproduced the training loop's metadata leak --
                # masked positions received the true item's genome vector, so the reported
                # `masked_item` gradient norms described an easier objective than the spec's.
                metadata = mask_metadata(metadata, mask_positions)
            else:
                masked_seq = seq
                mask_positions = None
                true_ids = None

            all_hidden = model(masked_seq, metadata=metadata, return_all_hidden=True)
            pooled = all_hidden[:, -1, :]

            per_objective_losses = {}
            moe_layer.clear_cache()

            Z_next, _ = moe_layer(all_hidden, pooled, 0)
            user_emb = Z_next[:, -1, :]
            per_objective_losses["next_item"] = loss_fns["next_item"](
                user_emb, targets, neg_ids, model.item_emb_weight,
            )

            if config.use_masked_lm and mask_positions is not None and "masked_item" in loss_fns:
                Z_mim, _ = moe_layer(all_hidden, pooled, 1)
                per_objective_losses["masked_item"] = loss_fns["masked_item"](
                    Z_mim, mask_positions, true_ids, model.item_emb_weight,
                )

            if config.use_multi_step and targets_k is not None and "multi_step" in loss_fns:
                Z_ms, _ = moe_layer(all_hidden, pooled, 2)
                per_objective_losses["multi_step"] = loss_fns["multi_step"](
                    Z_ms, targets_k, model.item_emb_weight,
                )

            # REV-009b: the contrastive objective participates in the (T, T) matrix.
            if config.use_contrastive and "contrastive" in loss_fns:
                Z_cl, _ = moe_layer(all_hidden, pooled, 3)
                z_orig = torch.nn.functional.normalize(Z_cl[:, -1, :].float(), p=2, dim=-1)

                B, S = seq.shape
                aug_seqs = []
                for i in range(B):
                    real = seq[i][seq[i] != 0].tolist()
                    if len(real) == 0:
                        real = [1]
                    aug = augment_sequence(
                        real, mask_id, augment_rng, config.crop_ratio, config.mask_ratio,
                    )
                    padded = ([0] * (S - len(aug)) + aug)[-S:]
                    aug_seqs.append(padded)
                aug_seq_t = torch.tensor(aug_seqs, dtype=torch.long, device=device)

                aug_meta = None
                if metadata is not None and metadata_matrix is not None:
                    # The full (vocab, F) metadata matrix stays on CPU; index there and move
                    # only the gathered (B, S, F) block to the accelerator.
                    # REV-072: the matrix covers MASK_ID with an all-zero row, so the
                    # clamp that mapped mask tokens onto the last real item is gone.
                    aug_meta = metadata_matrix[aug_seq_t.to(metadata_matrix.device)].to(device=device)

                aug_hidden = model(aug_seq_t, metadata=aug_meta, return_all_hidden=True)
                Z_aug, _ = moe_layer(aug_hidden, aug_hidden[:, -1, :], 3)
                z_aug = torch.nn.functional.normalize(Z_aug[:, -1, :].float(), p=2, dim=-1)
                per_objective_losses["contrastive"] = loss_fns["contrastive"](z_orig, z_aug)

            if objective_names is None:
                objective_names = list(per_objective_losses.keys())

            cos_sim = gradient_cosine_similarity(per_objective_losses, shared_params)
            norms = per_objective_grad_norms(per_objective_losses, shared_params)
            cos_matrices.append(cos_sim)
            norm_records.append(norms)

            batch_count += 1
    finally:
        model.train(prev_model_training)
        moe_layer.train(prev_moe_training)

    if not cos_matrices:
        return {}

    avg_cos = torch.stack(cos_matrices).mean(dim=0)
    avg_norms = {
        "per_objective": {},
        "mean": sum(r["mean"] for r in norm_records) / len(norm_records),
        "std": sum(r["std"] for r in norm_records) / len(norm_records),
        "coefficient_of_variation": sum(r["coefficient_of_variation"] for r in norm_records) / len(norm_records),
    }
    for name in objective_names:
        avg_norms["per_objective"][name] = sum(
            r["per_objective"].get(name, 0.0) for r in norm_records
        ) / len(norm_records)

    if not write_report:
        return {
            "cosine_similarity": avg_cos.tolist(),
            "objective_names": objective_names,
            "gradient_norms": avg_norms,
        }
    return save_gradient_diagnostics(
        avg_cos, avg_norms, objective_names, config, model_name,
        source_revision=source_revision,
    )


def plot_gradient_diagnostics(report: dict, config, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    cos_sim = report.get("cosine_similarity")
    names = report.get("objective_names", [])
    if cos_sim:
        cm = np.array(cos_sim)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Gradient Cosine Similarity")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, f"gradient_cosine_similarity_{mn}.png"), dpi=config.plot_dpi)
        plt.close(fig)

    grad_norms = report.get("gradient_norms", {})
    per_obj = grad_norms.get("per_objective", {})
    cv = grad_norms.get("coefficient_of_variation", 0.0)
    if per_obj:
        obj_names = list(per_obj.keys())
        vals = [per_obj[n] for n in obj_names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(obj_names, vals)
        ax.set_ylabel("L2 Gradient Norm")
        ax.set_title(f"Per-Objective Gradient Norms (CV={cv:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, f"gradient_norms_{mn}.png"), dpi=config.plot_dpi)
        plt.close(fig)
