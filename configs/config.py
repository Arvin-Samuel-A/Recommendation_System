from dataclasses import dataclass, field
from typing import List


OBJECTIVE_ORDER = ["next_item", "masked_item", "multi_step", "contrastive"]

MODEL_CONFIGS = {
    "t-small": {"num_layers": 3, "hidden_dim": 256, "num_heads": 4},
    "t-base": {"num_layers": 4, "hidden_dim": 320, "num_heads": 4},
    "t-large": {"num_layers": 5, "hidden_dim": 384, "num_heads": 6},
}


@dataclass
class Config:
    # --- Dataset ---
    dataset: str = "grouplens/movielens-20m-dataset"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    auto_download_dataset: bool = True
    force_download: bool = False

    # --- Preprocessing ---
    implicit_threshold: float = 4.0
    min_interactions: int = 5
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    cold_start_ratio: float = 0.10

    # --- Sequence ---
    max_seq_length: int = 50
    multi_step_k: int = 3

    # --- Model ---
    model_name: str = "t-base"
    hidden_dim: int = 320
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    ffn_expansion: int = 4
    normalize_embeddings: bool = True

    # --- Metadata ---
    use_metadata: bool = True
    genome_top_tags: int = 20

    # --- Attention ---
    attention_backend: str = "auto"

    # --- MoE ---
    moe_variant: str = "arv_mmoe"
    num_experts: int = 4
    expert_expansion: int = 4
    objective_embedding_dim: int = 32
    router_conditioning: str = "objective_conditioned"
    match_mmoe_expert_count: bool = True

    # --- Loss: Masked Item Modeling ---
    use_masked_lm: bool = True
    mask_prob: float = 0.15
    masked_lm_weight: float = 1.0

    # --- Loss: Multi-Step ---
    use_multi_step: bool = True
    multi_step_weight: float = 1.0

    # --- Loss: Contrastive ---
    use_contrastive: bool = True
    contrastive_interval: int = 4
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0

    # --- GradNorm ---
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-2

    # --- Training ---
    pretrain_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # --- LoRA ---
    lora_rank: int = 8
    lora_lr: float = 1e-3
    lora_weight_decay: float = 0.0
    local_epochs: int = 3

    # --- Federated ---
    federated_rounds: int = 20
    clients_per_round: int = 100
    personalization_eval_users: int = 1000
    use_weighted_fedavg: bool = True

    # --- Negative Sampling ---
    num_sampled_negatives: int = 100
    pool_size: int = 10000

    # --- Evaluation ---
    eval_k: int = 10
    use_ann: bool = False
    ann_candidates: int = 100
    use_logit_calibration: bool = False
    eval_temperature: float = 1.0

    # --- Latency ---
    warmup_runs: int = 20
    timed_runs: int = 100

    # --- Diagnostics ---
    routing_diag_interval_epochs: int = 5
    grad_diag_interval_steps: int = 500
    grad_diag_num_batches: int = 10

    # --- Augmentation ---
    crop_ratio: float = 0.6
    mask_ratio: float = 0.2

    # --- Output ---
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    plot_dir: str = "outputs/plots"
    ablation_dir: str = "outputs/ablations"
    plot_dpi: int = 150

    # --- Device / Dtype ---
    device: str = "auto"
    dtype: str = "bf16"

    # --- DataLoader ---
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    # --- Seeds ---
    seed: int = 42
    final_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # --- Ablation grids ---
    sequence_length_grid: List[int] = field(default_factory=lambda: [25, 50, 100, 200])
    expert_count_grid: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    lora_rank_grid: List[int] = field(default_factory=lambda: [2, 4, 8, 16])

    # --- Derived (set after preprocessing) ---
    num_items: int = 0
    num_users: int = 0
    feature_dim: int = 0
    vocab_size: int = 0

    def apply_model_config(self, model_name: str) -> None:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(MODEL_CONFIGS)}"
            )
        self.model_name = model_name
        mc = MODEL_CONFIGS[model_name]
        self.num_layers = mc["num_layers"]
        self.hidden_dim = mc["hidden_dim"]
        self.num_heads = mc["num_heads"]

    def model_dir_name(self) -> str:
        return self.model_name.replace("-", "_")

    def model_checkpoint_dir(self) -> str:
        return f"{self.checkpoint_dir}/{self.model_dir_name()}"

    def validate(self) -> None:
        if self.dtype != "bf16":
            raise ValueError(f"Only bf16 dtype is supported, got '{self.dtype}'")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.contrastive_interval == 0 and self.use_contrastive:
            raise ValueError(
                "contrastive_interval=0 is invalid; "
                "use use_contrastive=False to disable"
            )
        if self.train_ratio + self.val_ratio > 1.0:
            raise ValueError(
                f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) > 1.0"
            )
        if self.moe_variant not in ("none", "mmoe", "arv_mmoe"):
            raise ValueError(
                f"Unknown moe_variant '{self.moe_variant}'. "
                "Choose from 'none', 'mmoe', 'arv_mmoe'"
            )
        if self.router_conditioning not in ("objective_conditioned", "input_only"):
            raise ValueError(
                f"Unknown router_conditioning '{self.router_conditioning}'. "
                "Choose from 'objective_conditioned', 'input_only'"
            )
