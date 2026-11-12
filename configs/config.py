import copy
from dataclasses import dataclass, field
from typing import List, Optional


OBJECTIVE_ORDER = ["next_item", "masked_item", "multi_step", "contrastive"]

MODEL_CONFIGS = {
    "t-small": {"num_layers": 3, "hidden_dim": 256, "num_heads": 4},
    "t-base": {"num_layers": 4, "hidden_dim": 320, "num_heads": 4},
    "t-large": {"num_layers": 5, "hidden_dim": 384, "num_heads": 6},
}


@dataclass
class Config:
    # --- Dataset ---
    # Which raw format `data/preprocessing.py` reads. "movielens" (default): the
    # MovieLens-20M Kaggle CSVs, whose Kaggle slug is `dataset`. "amazon2014": one
    # category of the Amazon Reviews 2014 5-core release (`amazon_category`), read by
    # data/amazon.py. Every stage after preprocessing sees the same artifacts either way.
    # Select the Amazon settings together with `apply_dataset_preset("amazon_beauty_2014")`
    # (CLI `--dataset_preset`), not field by field.
    dataset_format: str = "movielens"
    # Amazon 2014 category: the raw files are `reviews_{category}_5.json.gz` and
    # `meta_{category}.json.gz`. Read only when dataset_format="amazon2014".
    amazon_category: str = "Beauty"
    dataset: str = "grouplens/movielens-20m-dataset"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    auto_download_dataset: bool = True
    force_download: bool = False

    # --- Preprocessing ---
    # Ratings below this are dropped before anything else. MovieLens ratings run from
    # 0.5 to 5.0, so the default 0.0 keeps EVERY rating as an interaction (owner decision
    # 2026-10-25; it was 4.0, which kept ~half of ML-20M).
    implicit_threshold: float = 0.0
    min_interactions: int = 5
    # specs/ablation-run FR-6 (owner decision 5): keep a seeded random fraction of the
    # USERS that survive the rating/min-interactions filters (full histories). 1.0 = no
    # sampling. `user_sample_seed` is independent of the training `seed`, so re-seeding
    # training never changes which users are in the dataset.
    user_sample_fraction: float = 1.0
    user_sample_seed: int = 42
    # Splits are leave-one-out (owner decision 2026-10-25): per user, the last
    # interaction is the test target, the second-to-last the validation target, and
    # everything before is training. This replaced an 80/10/10 per-user ratio split.
    # NOTE: `cold_start_ratio` applies ONLY to "first_appearance_rank" and
    # "timestamp_cutoff". Under the default "unseen_in_training" strategy the size of
    # the cold set is data-determined and this field is not consulted (FR-3).
    cold_start_ratio: float = 0.10
    # REV-137 (owner decision 2026-09-18): the DEFAULT is "unseen_in_training" -- an item
    # is cold iff it has zero interactions in any user's training portion. Nothing is
    # deleted to produce it, so the disjointness guarantee holds trivially rather than by
    # construction-after-deletion.
    #
    # The two prior strategies remain selectable. "timestamp_cutoff" selects items first
    # seen strictly after `global_training_cutoff`; that set is empty on MovieLens-20M --
    # not because items appear early (43.5% of the catalogue is first seen after a p80
    # global date) but because the cutoff is the MAXIMUM over every user's own training
    # end, which lands 0.1 days before the dataset's final interaction.
    # "first_appearance_rank" holds out the latest-appearing `cold_start_ratio` fraction
    # and is non-empty by construction, but it is INDUCED: measured, 927 of its 2,072
    # cold items had 6,634 real training interactions deleted to make them cold.
    cold_start_strategy: str = "unseen_in_training"
    # Optional first-appearance threshold for "unseen_in_training" (FR-3): when set to a
    # POSIX timestamp, an item is cold only if it also has no interaction before it, so a
    # strictly temporal set can be requested. `None` applies no threshold.
    cold_start_first_seen_after: Optional[float] = None

    # --- Sequence ---
    # Training is LLM-style (owner decision 2026-10-25): each user's training history is
    # fed whole, with a next-item target at every position. Histories longer than this
    # are cut into consecutive chunks of `max_seq_length` predictions, so every training
    # interaction is a target exactly once per epoch. 200 keeps 81% of ML-20M users in a
    # single chunk; evaluation reads the most recent `max_seq_length` items.
    max_seq_length: int = 200
    # Multi-step horizons: step s predicts the item s + 2 ahead, so k = 3 covers
    # horizons 2..4. Each horizon is a full-catalogue softmax at every position, so the
    # objective costs ~k next-item losses (owner decision 2026-10-31: back to 3 from 9,
    # which made multi-step 9 of the step's 11 softmaxes).
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
    # amazon2014 item features, replacing MovieLens' genome+genres block: [title TF-IDF
    # -> SVD (L2-normalised) | top-K brand one-hot | price quantile-bin one-hot | top-K
    # category multi-hot (second-level and leaf of every `categories` path)]. A missing
    # meta field leaves its block zero. Read only when dataset_format="amazon2014".
    amazon_title_dim: int = 32
    amazon_top_brands: int = 32
    amazon_price_bins: int = 8
    amazon_top_categories: int = 64

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

    # --- Loss: logit scale (next_item, masked_item, multi_step) ---
    # Those three objectives score L2-normalised cosines over the full catalogue, so
    # their logits lie in [-1, 1] and their cross-entropy cannot fall below
    # log(1 + (C-1)e^-2): ~8.2 on ML-20M with every rating (C ~ 26.7k, chance ~10.2).
    # Logits are multiplied by 1/logit_temperature. 1.0 is the previous behaviour exactly
    # (no op is applied).
    logit_temperature: float = 1.0
    # CLIP-style learnable scale, stored as log(scale), initialised to log(1/
    # logit_temperature), and clamped to <= logit_scale_max (after every optimizer step
    # and inside the forward). ONE SCALE PER OBJECTIVE, not one shared: the objectives'
    # CE-optimal scales differ (the contrastive one ranks 2B-1 in-batch views, the
    # others the full catalogue), and a shared scalar would be pulled between them.
    # Each scale lives in its loss module (so `loss_fns_state_dict` checkpoints it), is
    # excluded from weight decay, and is logged per epoch.
    learnable_logit_scale: bool = False
    logit_scale_max: float = 100.0
    # True: contrastive uses logit_temperature (and, with learnable_logit_scale, its own
    # learnable scale with the same init and clamp) instead of contrastive_temperature,
    # so all four objectives start on one logit scale. False: contrastive keeps its fixed
    # 1/contrastive_temperature (the previous behaviour).
    shared_logit_temperature: bool = False

    # --- GradNorm ---
    use_gradnorm: bool = True
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 1e-2
    # REV-055: T3.3 specifies a config-driven anchor selection.
    gradnorm_anchor: str = "last_shared_transformer"
    # Which lambdas are rescaled after each GradNorm step. "active" (the previous
    # behaviour): only this step's active objectives, to sum to their count. An
    # intermittent objective (contrastive, active every `contrastive_interval` steps) is
    # then pinned near 1: the other lambdas are reset to sum to 3 on every step without
    # it, so on its steps it can only take the remainder. "enabled": every enabled
    # objective, active this step or not, to sum to the number of enabled objectives.
    gradnorm_renorm_scope: str = "active"
    # Primary-objective GradNorm. "": GradNorm balances every active objective. An
    # objective name (e.g. "next_item", the one ranking is evaluated on): that
    # objective's lambda is fixed at 1 and left out of GradNorm, which balances only the
    # auxiliary objectives; their lambdas are renormalised to sum to `gradnorm_aux_budget`
    # (beta; None = their count). Defaults (owner decision 2026-10-25): next-item primary
    # with beta = 3. Run12 passed beta = 0.3 on the command line.
    gradnorm_primary_objective: str = "next_item"
    gradnorm_aux_budget: Optional[float] = 3.0
    # Gradient projection instead of GradNorm. "" (default): off. "next_item": each
    # auxiliary gradient that conflicts with next-item's (negative dot product) loses its
    # component along it; next-item's gradient is never changed. Uses the static weights
    # (masked_lm_weight, ...), so it requires use_gradnorm=False. One backward pass per
    # objective. See training/grad_projection.py.
    grad_projection: str = ""

    # --- Training ---
    pretrain_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    # specs/ablation-run FR-5 (owner decision 4): after warmup the cosine decays to this
    # fraction of the peak LR instead of 0. 0.0 reproduces the previous schedule exactly.
    min_lr_ratio: float = 0.1
    # specs/ablation-run FR-8: `run_pretrain` resumes from `last.pt` when it exists.
    # `--no_resume` turns it off (also disables reuse of persisted ablation arms, FR-11).
    resume_training: bool = True
    # Which per-epoch snapshots (`epoch_XXX.pt`) `run_pretrain` keeps. 1 (default, the
    # previous behaviour): every epoch. N > 1: every N-th epoch. 0: none. `best.pt` and
    # `last.pt` (the resume state) are written every epoch regardless. On ML-20M, 20
    # epochs x 14 arms kept 280 snapshots (55 GB); the 300-epoch Beauty preset sets 0.
    # `--stage ablation_rescore` re-scores the kept snapshots, so it needs them.
    epoch_snapshot_interval: int = 1

    # --- Negative Sampling ---
    # Pretraining and evaluation use the full catalogue. Preprocessing still builds the
    # global negative pool and records these values in its manifest; no training reads them.
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

    # Cap on optimizer steps per epoch. 0 = uncapped (the production setting, used by
    # full training in BUILD_ORDER Phase 14). The few-step smoke gate
    # (training-pipeline T6.3, settable from the CLI as `--max_steps`) sets this so
    # `run_pretrain` itself is exercised end-to-end -- scheduler, validation,
    # checkpointing, logging -- without a full pass over MovieLens-20M. Owner decision
    # (2026-09-17): a capped run on real data IS the Phase 11/17 gate; the uncapped full
    # epoch is no longer required to authorise training.
    max_steps_per_epoch: int = 0

    # Ablation H (FR-13) selects a system-level evaluation per arm. H3-H6 previously
    # carried NO overrides at all -- `{"label": "H3_cold_start"}` and friends resolved to
    # the base config, so the four arms were identical to each other and ran nothing
    # specific to their names. "Each selectable without source edits" is FR-13's wording,
    # so this is a config flag.
    system_eval: str = "none"

    # Test rows scored per ablation arm (specs/eval-integrity FR-6). 0 = every row of the
    # stored test split (one per user) -- the default. N > 0 = a seeded uniform random
    # sample of N rows (`ablation_eval_seed`), for tests and quick checks only.
    ablation_eval_examples: int = 0
    # Seed of the random test-row sample; independent of the training `seed`.
    ablation_eval_seed: int = 0

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
    # Compute dtype: "bf16" or "fp16". "fp16" requires `mixed_precision=True`.
    dtype: str = "bf16"
    # False: parameters are stored in `dtype` (pure bf16, the original design).
    # True: fp32 master weights, forward/backward under torch.autocast(dtype), and a
    # GradScaler when dtype is fp16. Needed for GPUs without native bf16 (e.g. T4).
    mixed_precision: bool = False

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

    DATASET_FORMATS = ("movielens", "amazon2014")
    SYSTEM_EVALS = ("none", "cold_start", "longtail", "latency")

    def apply_dataset_preset(self, name: str) -> None:
        """Set every field of a named dataset preset (see DATASET_PRESETS).

        Applied before any other override, so an explicit CLI flag still wins.
        """
        if name not in DATASET_PRESETS:
            raise ValueError(
                f"Unknown dataset preset '{name}'. Choose from {list(DATASET_PRESETS)}"
            )
        for key, value in DATASET_PRESETS[name].items():
            if not hasattr(self, key):
                raise AttributeError(f"dataset preset '{name}': unknown Config field '{key}'")
            setattr(self, key, copy.deepcopy(value))

    def validate(self) -> None:
        if self.dataset_format not in self.DATASET_FORMATS:
            raise ValueError(
                f"Unknown dataset_format '{self.dataset_format}'. "
                f"Choose from {list(self.DATASET_FORMATS)}"
            )
        if self.epoch_snapshot_interval < 0:
            raise ValueError(
                f"epoch_snapshot_interval must be >= 0 (0 = keep no per-epoch snapshot), "
                f"got {self.epoch_snapshot_interval}"
            )
        if self.system_eval not in self.SYSTEM_EVALS:
            raise ValueError(
                f"Unknown system_eval '{self.system_eval}'. "
                f"Choose from {list(self.SYSTEM_EVALS)}"
            )
        if self.dtype not in ("bf16", "fp16"):
            raise ValueError(f"Unknown dtype '{self.dtype}'. Choose from 'bf16', 'fp16'")
        if self.dtype == "fp16" and not self.mixed_precision:
            raise ValueError(
                "dtype='fp16' requires mixed_precision=True: pure fp16 weights underflow "
                "small updates and gradients without fp32 master weights and loss scaling."
            )
        if self.ablation_eval_examples < 0:
            raise ValueError(
                f"ablation_eval_examples must be >= 0 (0 = all test rows), "
                f"got {self.ablation_eval_examples}"
            )
        if self.ablation_eval_seed < 0:
            raise ValueError(f"ablation_eval_seed must be >= 0, got {self.ablation_eval_seed}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not self.logit_temperature > 0:
            raise ValueError(f"logit_temperature must be > 0, got {self.logit_temperature}")
        if self.learnable_logit_scale and 1.0 / self.logit_temperature > self.logit_scale_max:
            raise ValueError(
                f"learnable_logit_scale: initial scale 1/logit_temperature = "
                f"{1.0 / self.logit_temperature:g} exceeds logit_scale_max "
                f"{self.logit_scale_max:g}"
            )
        if self.gradnorm_renorm_scope not in ("active", "enabled"):
            raise ValueError(
                f"Unknown gradnorm_renorm_scope '{self.gradnorm_renorm_scope}'. "
                "Choose from 'active', 'enabled'"
            )
        if self.gradnorm_primary_objective not in ("",) + tuple(OBJECTIVE_ORDER):
            raise ValueError(
                f"Unknown gradnorm_primary_objective '{self.gradnorm_primary_objective}'. "
                f"Choose from {OBJECTIVE_ORDER} or '' (off)"
            )
        if self.gradnorm_aux_budget is not None:
            if not self.gradnorm_primary_objective:
                raise ValueError("gradnorm_aux_budget requires gradnorm_primary_objective")
            if not self.gradnorm_aux_budget > 0:
                raise ValueError(
                    f"gradnorm_aux_budget must be > 0, got {self.gradnorm_aux_budget}"
                )
        if self.grad_projection not in ("", "next_item"):
            raise ValueError(
                f"Unknown grad_projection '{self.grad_projection}'. Choose from '', 'next_item'"
            )
        if self.grad_projection and self.use_gradnorm:
            raise ValueError("grad_projection replaces GradNorm: set use_gradnorm=False")
        if self.grad_projection and self.dtype == "fp16":
            raise ValueError("grad_projection does not support fp16 loss scaling")
        if self.contrastive_interval == 0 and self.use_contrastive:
            raise ValueError(
                "contrastive_interval=0 is invalid; "
                "use use_contrastive=False to disable"
            )
        if self.implicit_threshold < 0:
            raise ValueError(
                f"implicit_threshold must be >= 0 (0 keeps every rating), "
                f"got {self.implicit_threshold}"
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
        if self.cold_start_strategy not in (
            "unseen_in_training",
            "first_appearance_rank",
            "timestamp_cutoff",
        ):
            raise ValueError(
                f"Unknown cold_start_strategy '{self.cold_start_strategy}'. "
                "Choose from 'unseen_in_training', 'first_appearance_rank', "
                "'timestamp_cutoff'"
            )
        # T1.3 / AC-09: `flex_attention` is NOT a sanctioned value. It has no backward
        # pass on MPS in the pinned PyTorch build, so it can never satisfy the training
        # gates on the dev host; a selectable backend that provably cannot train is a
        # trap (FR-4 rationale, design D2b). specs/ablation-run FR-2: FlashAttention-3
        # replaces the misnamed FA4 backend (which called the FA2 API). The message names
        # only the supported choices.
        if self.attention_backend not in ("auto", "flash_attention_3", "sdpa"):
            raise ValueError(
                f"Unknown attention_backend '{self.attention_backend}'. "
                "Choose from 'auto', 'flash_attention_3', 'sdpa'"
            )
        # specs/ablation-run FR-5 / FR-6.
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(
                f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}"
            )
        if not 0.0 < self.user_sample_fraction <= 1.0:
            raise ValueError(
                f"user_sample_fraction must be in (0, 1], got {self.user_sample_fraction}"
            )
        if not 0.0 < self.cold_start_ratio < 1.0:
            raise ValueError(
                f"cold_start_ratio must be in (0, 1), got {self.cold_start_ratio}"
            )


#: Named dataset presets: every field that differs from the MovieLens-20M defaults, set
#: together by `Config.apply_dataset_preset` (CLI `--dataset_preset`). Everything not
#: listed -- model size, batch 256, losses, multi_step_k and the
#: leave-one-out full-ranking protocol -- stays at the defaults.
DATASET_PRESETS = {
    # The defaults themselves, so a command can name its dataset explicitly.
    "movielens_20m": {},
    "amazon_beauty_2014": {
        "dataset_format": "amazon2014",
        "amazon_category": "Beauty",
        # Own raw/processed directories and output roots, so nothing collides with the
        # MovieLens data, checkpoints (the by-fingerprint store) or ablation results.
        "raw_data_dir": "data/raw_beauty",
        "processed_data_dir": "data/processed_beauty",
        "output_dir": "outputs/amazon_beauty_2014",
        "checkpoint_dir": "outputs/amazon_beauty_2014/checkpoints",
        "log_dir": "outputs/amazon_beauty_2014/logs",
        "plot_dir": "outputs/amazon_beauty_2014/plots",
        "ablation_dir": "outputs/amazon_beauty_2014/ablations",
        # 5-core Beauty histories are short (mean ~8.9 interactions per user), so the
        # reference arm reads 20 items and study H2 sweeps 5/10/15/20 -- the reference
        # coincides with H2_seq20 as max_seq_length=200 does with H2_seq200 on ML-20M.
        "max_seq_length": 20,
        "sequence_length_grid": [5, 10, 15, 20],
        # The best epoch is still chosen on validation NDCG (`run_pretrain`).
        "pretrain_epochs": 300,
        # 300 snapshots per arm would dwarf ML's 55 GB; best.pt and last.pt (resume) stay.
        "epoch_snapshot_interval": 0,
    },
}
