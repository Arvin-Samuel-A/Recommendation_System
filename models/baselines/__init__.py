"""External sequential-recommendation baselines trained and evaluated in our pipeline.

See models/baselines/models.py (architectures and losses), models/baselines/hparams.py
(published hyperparameters with sources), training/baseline_train.py (training loop) and
docs/baselines.md.
"""
from models.baselines.hparams import BASELINE_NAMES, published_params  # noqa: F401
from models.baselines.models import BASELINE_CLASSES, BaselineModel  # noqa: F401


def build_baseline(config) -> BaselineModel:
    """The baseline named by `config.baseline_model`, sized for `config.num_items`."""
    name = config.baseline_model
    if name not in BASELINE_CLASSES:
        raise ValueError(f"Unknown baseline_model '{name}'. Choose from {list(BASELINE_CLASSES)}")
    if config.num_items <= 0:
        raise ValueError("build_baseline needs config.num_items (resolve the dataset first)")
    return BASELINE_CLASSES[name](config.num_items, config.baseline_params)


def count_baseline_parameters(model) -> dict:
    """Total / embedding / non-embedding parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.item_embeddings.weight.numel()
    if hasattr(model, "position_embeddings"):
        emb += model.position_embeddings.weight.numel()
    return {
        "total_parameters": int(total),
        "embedding_parameters": int(emb),
        "non_embedding_parameters": int(total - emb),
    }
