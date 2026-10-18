"""Recommendation demo: show a trained ARV-MMoE model recommending movies.

Picks a MovieLens user from the 10% sample (or takes a list of movies you name), shows
what they liked most recently, and lists the model's top-N recommendations from the whole
catalogue. Movies the user has already rated are never recommended. For a real user, a
check mark shows the recommendations the user actually went on to like in their held-out
test period, which the model never saw. The popularity ranking, and optionally a second
model (the next-item-only baseline), are shown for comparison.

Run from the repository root:

    uv run python scripts/demo_recommend.py                       # a random user
    uv run python scripts/demo_recommend.py --user 42             # internal user id
    uv run python scripts/demo_recommend.py --movielens-user 1234
    uv run python scripts/demo_recommend.py --liked "Toy Story" "Finding Nemo" "Shrek"
    uv run python scripts/demo_recommend.py --compare             # also show the baseline

Scoring is the production path (`evaluation.metrics.score_candidates_batch`): the
next-item user representation (ARV routing with the next-item objective) is compared by
cosine with every item embedding, in fp32.
"""
import argparse
import csv
import os
import pickle
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config  # noqa: E402
from data.sequence_generator import load_metadata_matrix  # noqa: E402
from evaluation.metrics import load_item_popularity, score_candidates_batch  # noqa: E402
from training.pretrain import build_model, load_checkpoint  # noqa: E402
from utils import autocast_context  # noqa: E402

RUN12 = "outputs/aws_2026-10-17/run12/outputs/checkpoints/ablations/by_fingerprint"
DEFAULT_MODEL = f"{RUN12}/17bcde2f65e8/t_base/best.pt"      # A: arv_mmoe_gradnorm (reference)
DEFAULT_BASELINE = f"{RUN12}/aa8ea802dd80/t_base/best.pt"   # A: single_obj_baseline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group()
    who.add_argument("--user", type=int, help="internal user id (0 .. num_users-1)")
    who.add_argument("--movielens-user", type=int, help="original MovieLens userId")
    who.add_argument("--liked", nargs="+", metavar="TITLE",
                     help="build a viewing history from these titles (case-insensitive "
                          "substring match), oldest first")
    p.add_argument("--checkpoint", default=DEFAULT_MODEL)
    p.add_argument("--compare", action="store_true",
                   help="also show the next-item-only baseline's recommendations")
    p.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    p.add_argument("--processed-dir", default="data/processed_users10pct")
    p.add_argument("--movies-csv", default="data/raw/movie.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--history", type=int, default=10, help="recent liked movies to show")
    p.add_argument("--seed", type=int, default=None, help="seed for the random user pick")
    p.add_argument("--device", default="cpu", help="cpu (default), mps, cuda or auto")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- loading

def load_model(path: str, processed_dir: str, device: str):
    """Rebuild the model a checkpoint was trained as, and load its weights."""
    if not os.path.exists(path):
        raise SystemExit(
            f"checkpoint not found: {path}\nRun from the repository root, or pass --checkpoint."
        )
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = Config()
    for key, value in ckpt["config"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.processed_data_dir = processed_dir
    cfg.device = device
    cfg.attention_backend = "sdpa"
    cfg.require_attention_backend = False
    model, moe = build_model(cfg)
    load_checkpoint(path, model, moe, checkpoint=ckpt)
    model.eval()
    moe.eval()
    return cfg, model, moe, ckpt


def load_titles(movies_csv: str) -> Dict[int, Tuple[str, str]]:
    """movieId -> (title, genres)."""
    out = {}
    with open(movies_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["movieId"])] = (row["title"], row["genres"].replace("|", ", "))
    return out


def load_pickle(processed_dir: str, name: str):
    with open(os.path.join(processed_dir, name), "rb") as f:
        return pickle.load(f)


def describe_model(cfg: Config, ckpt: dict) -> str:
    if cfg.moe_variant == "arv_mmoe":
        arch = f"ARV-MMoE ({cfg.num_experts} experts)"
    elif cfg.moe_variant == "mmoe":
        arch = f"MMoE ({cfg.num_experts} experts)"
    else:
        arch = "no routing"
    objectives = ["next-item"] + [
        name for name, on in (("masked-item", cfg.use_masked_lm), ("multi-step", cfg.use_multi_step),
                              ("contrastive", cfg.use_contrastive)) if on
    ]
    weighting = ""
    if len(objectives) > 1:
        weighting = ", GradNorm" if cfg.use_gradnorm else ", equal weights"
        if cfg.use_gradnorm and cfg.gradnorm_primary_objective:
            weighting = (f", next-item-primary GradNorm (aux budget "
                         f"{cfg.gradnorm_aux_budget:g})")
    return (f"{cfg.model_name} Transformer, {arch}, objectives: {' + '.join(objectives)}"
            f"{weighting}; best epoch {ckpt.get('best_epoch')}, "
            f"val NDCG@10 {ckpt.get('best_val_ndcg', float('nan')):.3f}")


# --------------------------------------------------------------------------- scoring

def encode(items: Sequence[int], cfg: Config, metadata: Optional[torch.Tensor]):
    """The model input for one history: the most recent `max_seq_length` items,
    left-padded with 0, and the matching metadata rows."""
    S = cfg.max_seq_length
    recent = list(items)[-S:]
    seq = torch.tensor([[0] * (S - len(recent)) + recent], dtype=torch.long)
    meta = metadata[seq] if metadata is not None else None
    return seq, meta


def routing_weights(model, moe, cfg: Config, seq, meta) -> Optional[List[float]]:
    """The next-item objective's expert mixture for this history (None without routing)."""
    if moe.variant == "none":
        return None
    device = next(model.parameters()).device
    with torch.no_grad(), autocast_context(cfg, device_type=device.type):
        meta_d = meta.to(device=device, dtype=model.item_emb_weight.dtype) if meta is not None else None
        hidden = model(seq.to(device), metadata=meta_d, return_all_hidden=True)
        moe.clear_cache()
        _, r = moe(hidden, hidden[:, -1, :], 0)
        moe.clear_cache()
    return [float(x) for x in r[0].float().cpu()]


def recommend(model, moe, cfg, metadata, history: Sequence[int], exclude: set, k: int):
    """Top-k (item, score) over the whole catalogue, excluding `exclude`."""
    seq, meta = encode(history, cfg, metadata)
    candidates = torch.arange(1, cfg.num_items + 1, dtype=torch.long).unsqueeze(0)
    scores = score_candidates_batch(model, seq, candidates, cfg, moe_layer=moe, metadata=meta)[0].cpu()
    moe.clear_cache()
    if exclude:
        idx = torch.tensor(sorted(i - 1 for i in exclude if 1 <= i <= cfg.num_items))
        scores[idx] = float("-inf")
    top = torch.topk(scores, k)
    return [(int(i) + 1, float(s)) for s, i in zip(top.values, top.indices)], (seq, meta)


def popular(popularity: torch.Tensor, exclude: set, k: int) -> List[Tuple[int, float]]:
    pop = popularity.clone()
    pop[0] = float("-inf")
    pop[-1] = float("-inf")
    for i in exclude:
        pop[i] = float("-inf")
    top = torch.topk(pop, k)
    return [(int(i), float(s)) for s, i in zip(top.values, top.indices)]


# --------------------------------------------------------------------------- display

def title_of(item: int, rev_map, titles) -> Tuple[str, str]:
    return titles.get(int(rev_map.get(item, -1)), (f"<item {item}>", ""))


def clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_list(header: str, recs, rev_map, titles, future: set, score_name: str):
    print(f"\n{header}")
    for rank, (item, score) in enumerate(recs, 1):
        title, genres = title_of(item, rev_map, titles)
        mark = "✓" if item in future else " "
        value = f"score {score:.3f}" if score_name == "score" else f"{int(score):>5d} likes"
        print(f"  {rank:2d}. {mark} {clip(title, 56):56s} {clip(genres, 36):36s} {value}")
    if future:
        hits = sum(item in future for item, _ in recs)
        print(f"      → hits: {hits}/{len(recs)} recommendations were movies this user liked "
              f"in the held-out test period ({len(future)} movies)")


def resolve_liked(queries: Sequence[str], titles, item_map, popularity) -> List[int]:
    """Each query -> the most popular catalogue item whose title contains it."""
    items = []
    for q in queries:
        ql = q.lower()
        matches = [
            item_map[mid] for mid, (t, _) in titles.items()
            if ql in t.lower() and mid in item_map
        ]
        if not matches:
            print(f"  (no catalogue movie matches '{q}', skipped)")
            continue
        items.append(max(matches, key=lambda i: float(popularity[i])))
    if not items:
        raise SystemExit("none of the --liked titles matched a movie in the catalogue")
    return items


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)
    cfg, model, moe, ckpt = load_model(args.checkpoint, args.processed_dir, args.device)
    metadata = load_metadata_matrix(args.processed_dir, cfg)
    titles = load_titles(args.movies_csv)
    item_map = load_pickle(args.processed_dir, "item_mapping.pkl")         # movieId -> item
    rev_map = load_pickle(args.processed_dir, "reverse_item_mapping.pkl")  # item -> movieId
    popularity = load_item_popularity(args.processed_dir, cfg.num_items + 2)

    print("=" * 100)
    print("ARV-MMoE movie recommendation demo (MovieLens-20M, 10% user sample)")
    print(f"Model: {describe_model(cfg, ckpt)}")
    print("=" * 100)

    future: set = set()
    if args.liked:
        history = resolve_liked(args.liked, titles, item_map, popularity)
        print(f"\nA new viewer who liked these {len(history)} movies (oldest first):")
        exclude = set(history)
    else:
        splits = load_pickle(args.processed_dir, "splits.pkl")
        user_map = {int(k): v for k, v in load_pickle(args.processed_dir, "user_mapping.pkl").items()}
        if args.movielens_user is not None:
            if args.movielens_user not in user_map:
                raise SystemExit(f"MovieLens user {args.movielens_user} is not in the 10% sample")
            user = user_map[args.movielens_user]
        elif args.user is not None:
            user = args.user
        else:
            pool = [u for u, t in splits["test"].items() if len(t) >= 5]
            user = random.Random(args.seed).choice(pool)
        orig = {v: k for k, v in user_map.items()}.get(user)
        # The model sees everything up to the test period (train + validation), exactly as
        # in evaluation; the test-period items are what we check the recommendations against.
        history = list(splits["train"].get(user, [])) + list(splits["val"].get(user, []))
        future = set(splits["test"].get(user, []))
        exclude = set(history)
        print(f"\nUser {user} (MovieLens userId {orig}): liked {len(history)} movies before "
              f"the test period, then {len(future)} more in it (hidden from the model).")
        print(f"Their {min(args.history, len(history))} most recent liked movies (oldest first):")

    for item in history[-args.history:]:
        title, genres = title_of(item, rev_map, titles)
        print(f"      {clip(title, 56):56s} {clip(genres, 36)}")

    recs, (seq, meta) = recommend(model, moe, cfg, metadata, history, exclude, args.top_k)
    print_list(f"Top-{args.top_k} recommendations from the model"
               + (" (✓ = the user really liked it later)" if future else ""),
               recs, rev_map, titles, future, "score")

    weights = routing_weights(model, moe, cfg, seq, meta)
    if weights is not None:
        mix = "  ".join(f"expert {i}: {w:.2f}" for i, w in enumerate(weights))
        print(f"\nHow the router mixed the experts for this viewer (next-item objective): {mix}")

    if args.compare:
        b_cfg, b_model, b_moe, b_ckpt = load_model(args.baseline_checkpoint, args.processed_dir, args.device)
        b_meta = load_metadata_matrix(args.processed_dir, b_cfg)
        b_recs, _ = recommend(b_model, b_moe, b_cfg, b_meta, history, exclude, args.top_k)
        print_list(f"Top-{args.top_k} from the comparison model: {describe_model(b_cfg, b_ckpt)}",
                   b_recs, rev_map, titles, future, "score")

    print_list(f"Top-{args.top_k} most popular movies (what a non-personalised system would show)",
               popular(popularity, exclude, args.top_k), rev_map, titles, future, "likes")
    print()


if __name__ == "__main__":
    main()
