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
    # REV-011: "timestamp_cutoff" selects items first seen strictly after the global
    # training cutoff -- on MovieLens-20M that set is empty, which made the whole
    # cold-start research question vacuous. "first_appearance_rank" holds out the
    # latest-appearing `cold_start_ratio` fraction of items, which is non-empty by
    # construction.
    cold_start_strategy: str = "first_appearance_rank"

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
    # REV-043: when True, "auto" may not fall back to SDPA if the hardware-preferred
    # backend cannot execute -- it raises instead. Set this for the MPS and H100 gate runs.
    require_attention_backend: bool = False

    # --- MoE ---
    moe_variant: str = "arv_mmoe"
    num_experts: int = 4
    expert_expansion: int = 4
    objective_embedding_dim: int = 32
    router_conditioning: str = "objective_conditioned"
    # REV-046: when True the conventional-MMoE arm uses `num_experts`, so MMoE and
    # ARV-MMoE differ only in the routing formulation (Ablation C's intended comparison).
    # When False the conventional arm gets `mmoe_num_experts` experts of its own.
    match_mmoe_expert_count: bool = True
    mmoe_num_experts: int = 4

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
    # REV-055: T3.3 specifies a config-driven anchor selection.
    gradnorm_anchor: str = "last_shared_transformer"

    # --- Training ---
    pretrain_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # --- LoRA / personalization ---
    # REV-070: `use_lora` and `use_federated` were added for REV-018 but REV-018 was
    # solved through `personalization_mode`, which IS read by the runner and by
    # `_run_single_config`. Two flags that validated and changed nothing were recorded
    # in every ablation's `resolved_config`, where a reader could reasonably think they
    # distinguished the arms. Removed rather than left dangling.
    # REV-003/REV-004: which split a client's LoRA is trained on. "train" uses the
    # user's training interactions; the evaluation set is never trained on. Read by
    # `training/local_train.py` and `training/federated.py`.
    local_train_source: str = "train"
    # REV-003: cap on the per-user local training examples drawn from that split, taking
    # the MOST RECENT interactions. Unbounded, a heavy user contributes thousands of
    # examples per local epoch across `personalization_eval_users` users.
    local_train_max_examples: int = 32
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
    # REV-025: `pool_size` is the configured value; `resolve_pool_size()` applies the
    # `max(pool_size, num_items * pool_size_items_multiplier)` floor so the manifest and
    # the built artifact always agree.
    pool_size: int = 10000
    pool_size_items_multiplier: int = 10

    # --- Evaluation ---
    eval_k: int = 10
    # REV-063: evaluation forwards a batch at a time, not one example per forward.
    eval_batch_size: int = 256
    use_ann: bool = False
    ann_candidates: int = 100
    use_logit_calibration: bool = False
    eval_temperature: float = 1.0

    # --- Latency ---
    warmup_runs: int = 20
    timed_runs: int = 100

    # Cap on optimizer steps per epoch. 0 = uncapped (the production setting). The
    # few-step dev-host smoke gate (training-pipeline T6.3) sets this so `run_pretrain`
    # itself is exercised end-to-end -- scheduler, validation, checkpointing, logging --
    # without a full pass over MovieLens-20M. The one-epoch gate runs uncapped on the
    # training host (BUILD_ORDER Phase 18).
    max_steps_per_epoch: int = 0

    # Ablation H (FR-13) selects a system-level evaluation per arm. H3-H6 previously
    # carried NO overrides at all -- `{"label": "H3_cold_start"}` and friends resolved to
    # the base config, so the four arms were identical to each other and ran nothing
    # specific to their names. "Each selectable without source edits" is FR-13's wording,
    # so this is a config flag.
    system_eval: str = "none"

    # Ablation E (FR-10) compares "global only" against "global + LoRA". That is a real
    # experimental dimension with no config field, so every E arm resolved to an
    # identical config and the arms were indistinguishable in their own result JSON
    # (REV-019). "none" = evaluate the global backbone as-is; "lora" = additionally run
    # per-user LoRA personalization and report the gain.
    personalization_mode: str = "none"

    # Examples scored per ablation arm. The full test split is ~1M examples and each
    # study has up to 6 arms; capping keeps a study tractable while every arm is scored
    # on the SAME prefix of the same stored candidate sets, so arms stay comparable.
    ablation_eval_examples: int = 5000

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

    def resolve_pool_size(self) -> int:
        """Actual negative-pool size, derived from config rather than hardcoded."""
        if self.num_items <= 0:
            return self.pool_size
        return max(self.pool_size, self.num_items * self.pool_size_items_multiplier)

    def model_dir_name(self) -> str:
        return self.model_name.replace("-", "_")

    def model_checkpoint_dir(self) -> str:
        return f"{self.checkpoint_dir}/{self.model_dir_name()}"

    PERSONALIZATION_MODES = ("none", "lora")
    SYSTEM_EVALS = ("none", "cold_start", "longtail", "latency", "communication")

    def validate(self) -> None:
        if self.system_eval not in self.SYSTEM_EVALS:
            raise ValueError(
                f"Unknown system_eval '{self.system_eval}'. "
                f"Choose from {list(self.SYSTEM_EVALS)}"
            )
        if self.personalization_mode not in self.PERSONALIZATION_MODES:
            raise ValueError(
                f"Unknown personalization_mode '{self.personalization_mode}'. "
                f"Choose from {list(self.PERSONALIZATION_MODES)}"
            )
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
        if self.cold_start_strategy not in ("first_appearance_rank", "timestamp_cutoff"):
            raise ValueError(
                f"Unknown cold_start_strategy '{self.cold_start_strategy}'. "
                "Choose from 'first_appearance_rank', 'timestamp_cutoff'"
            )
        if self.local_train_source not in ("train", "val"):
            raise ValueError(
                f"Unknown local_train_source '{self.local_train_source}'. "
                "Choose from 'train', 'val'. The test split is never a training source."
            )
        if self.attention_backend not in ("auto", "flex_attention", "flash_attention_4", "sdpa"):
            raise ValueError(
                f"Unknown attention_backend '{self.attention_backend}'. "
                "Choose from 'auto', 'flex_attention', 'flash_attention_4', 'sdpa'"
            )
        if not 0.0 < self.cold_start_ratio < 1.0:
            raise ValueError(
                f"cold_start_ratio must be in (0, 1), got {self.cold_start_ratio}"
            )
