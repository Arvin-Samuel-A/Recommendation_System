"""Published hyperparameters of the five baselines, with the source of every value.

Owner rule (2026-11-13): each baseline runs with its authors' published settings, no
tuning. All five come from the BSARec benchmark (Shin et al., AAAI 2024, arXiv 2312.10325;
code https://github.com/yehjin-shin/BSARec at commit c80bdc0), which reports every one of
them on Beauty and ML-1M under full ranking. MovieLens-20M uses the ML-1M settings.

Sources, cited per value below:
  [args]   `parse_args()` in `src/utils.py` -- the defaults; the README says "For the
           hyperparameters for the baselines, check the parse_args() function".
  [paper]  Section "Experimental Setup": D = 64, N = 50, L = 2 blocks, Adam, batch 256,
           lr in {5e-4, 1e-3}, heads in {1, 2, 4}.
  [table6] Appendix Table 6, BSARec's best hyperparameters per dataset.
  [readme] The README's example commands (BSARec Beauty: lr 5e-4, alpha 0.7, c 5, 1 head;
           SASRec Beauty: 1 head). The released `src/output/BSARec_Beauty_best.log` echoes
           the full BSARec Beauty argument namespace and agrees with [table6].
The paper says baselines were searched "based on their recommended hyperparameters" but
publishes no per-dataset baseline values; where none is published the [args] default is
used. Training-loop settings (epochs, early stopping, data feed, evaluation) are OURS, not
the reference's (docs/baselines.md).
"""
import copy
from typing import Dict

#: Shared by every baseline on every dataset.
COMMON = {
    "hidden_size": 64,          # [args] --hidden_size, [paper] D = 64
    "num_layers": 2,            # [args] --num_hidden_layers, [paper] L = 2
    "num_heads": 2,             # [args] --num_attention_heads (overridden per model below)
    "hidden_act": "gelu",       # [args] --hidden_act
    "attn_dropout": 0.5,        # [args] --attention_probs_dropout_prob
    "hidden_dropout": 0.5,      # [args] --hidden_dropout_prob
    "initializer_range": 0.02,  # [args] --initializer_range
    "max_seq_length": 50,       # [args] --max_seq_length, [paper] N = 50
    "batch_size": 256,          # [args] --batch_size, [paper]
    "learning_rate": 1e-3,      # [args] --lr
    "weight_decay": 0.0,        # [args] --weight_decay (Adam, not AdamW)
    "adam_beta1": 0.9,          # [args] --adam_beta1
    "adam_beta2": 0.999,        # [args] --adam_beta2
}

#: Model-specific values. Keys: model -> dataset ("ml-1m" | "beauty") -> overrides.
PER_MODEL = {
    "sasrec": {
        "ml-1m": {},                                   # [args] 2 heads
        "beauty": {"num_heads": 1},                    # [readme] SASRec Beauty example
    },
    "bert4rec": {
        "ml-1m": {"mask_ratio": 0.2},                  # [args] --mask_ratio
        "beauty": {"mask_ratio": 0.2},
    },
    "gru4rec": {
        "ml-1m": {"gru_hidden_size": 64},              # [args] --gru_hidden_size
        "beauty": {"gru_hidden_size": 64},
    },
    "duorec": {
        # [args] --tau 1.0 --lmd 0.1 --lmd_sem 0.1 --ssl us_x --sim dot. Under us_x only
        # lmd_sem weights a loss term; lmd is kept for the record.
        "ml-1m": {"tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "beauty": {"tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
    },
    "bsarec": {
        "ml-1m": {"alpha": 0.3, "c": 9, "num_heads": 4, "learning_rate": 5e-4},   # [table6]
        "beauty": {"alpha": 0.7, "c": 5, "num_heads": 1, "learning_rate": 5e-4},  # [table6] [readme]
    },
}

#: Our dataset -> the reference dataset whose settings it uses.
DATASET_KEYS = {"movielens": "ml-1m", "amazon2014:Beauty": "beauty"}

#: Published full-ranking test results (BSARec paper, Table 2), for the port check.
PUBLISHED_RESULTS = {
    "beauty": {
        "gru4rec": {"hr@10": 0.0304, "ndcg@10": 0.0147},
        "sasrec": {"hr@10": 0.0531, "ndcg@10": 0.0283},
        "bert4rec": {"hr@10": 0.0705, "ndcg@10": 0.0387},
        "duorec": {"hr@10": 0.0965, "ndcg@10": 0.0584},
        "bsarec": {"hr@10": 0.1008, "ndcg@10": 0.0611},
    },
    "ml-1m": {
        "gru4rec": {"hr@10": 0.1657, "ndcg@10": 0.0828},
        "sasrec": {"hr@10": 0.2137, "ndcg@10": 0.1116},
        "bert4rec": {"hr@10": 0.2346, "ndcg@10": 0.1289},
        "duorec": {"hr@10": 0.2704, "ndcg@10": 0.1530},
        "bsarec": {"hr@10": 0.2757, "ndcg@10": 0.1568},
    },
}

BASELINE_NAMES = tuple(PER_MODEL)


def dataset_key(dataset_format: str, amazon_category: str = "Beauty") -> str:
    key = dataset_format if dataset_format == "movielens" else f"{dataset_format}:{amazon_category}"
    if key not in DATASET_KEYS:
        raise ValueError(
            f"No published baseline settings for dataset '{key}'. "
            f"Known: {sorted(DATASET_KEYS)}"
        )
    return DATASET_KEYS[key]


def published_params(model: str, dataset_format: str, amazon_category: str = "Beauty") -> Dict:
    """The full published hyperparameter dict of `model` for our dataset."""
    if model not in PER_MODEL:
        raise ValueError(f"Unknown baseline '{model}'. Choose from {list(PER_MODEL)}")
    params = copy.deepcopy(COMMON)
    params.update(copy.deepcopy(PER_MODEL[model][dataset_key(dataset_format, amazon_category)]))
    return params
