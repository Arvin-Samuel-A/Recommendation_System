"""Export the demo model to ONNX, plus the data the on-device demo (demo/mobile/) needs.

Writes demo/mobile/assets/:
    model.onnx   seq (1, S) int64 item ids, left-padded with 0 (S = the model's max_seq_length)
                 -> scores (1, num_items) cosine per item (item i at index i-1)
                    routing (1, num_experts) next-item expert mixture
    items.json   titles, genres and like counts, indexed like `scores`
    users.json   a sample of real test users: history (train + val) and test-period items
    meta.json    seq_len, num_items, parameter count and size, read by the page

The graph is the production scoring path of scripts/demo_recommend.py `recommend()`:
Transformer -> ARV-MMoE (next-item objective) -> cosine with every item embedding. The
item-metadata table is baked in, so the phone only sends item ids. The script checks the
exported model against PyTorch with onnxruntime before it finishes.

    uv run python scripts/export_mobile.py
    uv run python scripts/export_mobile.py --checkpoint <path> --num-users 500
    uv run python scripts/export_mobile.py --dataset beauty      # -> demo/mobile/assets_beauty

The page loads another export with ?assets=DIR (e.g. ?assets=assets_beauty).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from data.sequence_generator import load_metadata_matrix  # noqa: E402
from demo_recommend import (  # noqa: E402
    DEFAULT_MODEL, load_model, load_pickle, load_titles, recommend,
)
from evaluation.metrics import load_item_popularity  # noqa: E402

EXAMPLE_USERS = [13408]  # the prepared example in scripts/run_demo.sh (MovieLens)
#: Two top-10 items whose scores differ by less than this may appear in either order.
TIE_TOL = 1e-3

#: --dataset beauty: the Amazon Beauty 2014 reference model (A:arv_mmoe_gradnorm, seed 42).
BEAUTY = {
    "checkpoint": "outputs/h100_beauty_2026-11-07/outputs/checkpoints/ablations/by_fingerprint/"
                  "7802dd4b7636/t_base/best.pt",
    "processed_dir": "data/processed_beauty",
    "out": "demo/mobile/assets_beauty",
}


def load_amazon_titles(cfg, rev_map, num_items):
    """item id -> (title, category path) from the Amazon metadata file."""
    import html

    from data.amazon import read_metadata

    meta = read_metadata(cfg, [rev_map[i] for i in range(1, num_items + 1) if i in rev_map])
    out = {}
    for i in range(1, num_items + 1):
        r = meta.get(rev_map.get(i), {})
        cats = r.get("categories") or [[]]
        out[i] = (html.unescape(str(r.get("title") or f"<item {i}>")),
                  " > ".join(str(c) for c in (cats[0] if cats else [])))
    return out


class OnDeviceRecommender(nn.Module):
    """seq (1, S) -> (scores over all items, next-item routing weights)."""

    def __init__(self, model, moe, metadata: torch.Tensor, num_items: int, seq_len: int):
        super().__init__()
        self.model = model
        self.moe = moe
        self.register_buffer("metadata", metadata.float())
        self.num_items = num_items
        # Cosine = (normalised user . raw item) / |item|, against the WHOLE embedding table:
        # normalising or slicing the table inside the graph makes constant folding store a
        # second 17 MB copy of it.
        table = model.item_embedding.weight.detach().float()
        self.register_buffer("item_norms", table.norm(dim=-1).clamp_min(1e-12))
        # The attention builds its causal mask with torch.triu at the input's (traced) length,
        # which exports as a Trilu node -- an op onnxruntime-web 1.18's single-threaded build
        # (what the phone page runs) lacks. Give it the same mask as a constant instead.
        self.register_buffer("causal_blocked", torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1))

        def sdpa_mask(q, padding_mask, is_causal):
            if padding_mask is None:  # unpadded: SDPA's own is_causal path, as before
                return None
            blocked = ~padding_mask[:, None, None, :]
            if is_causal:
                blocked = blocked | self.causal_blocked
            return ~blocked

        model.backend._sdpa_additive_mask = sdpa_mask

    def forward(self, seq: torch.Tensor):
        m = self.model
        S = seq.shape[1]
        x = m.item_embedding(seq) + m.position_embedding(torch.arange(S))
        if m.metadata_projection is not None:
            x = x + m.metadata_projection(self.metadata[seq])
        # TransformerModel.forward skips the mask when nothing is padded; always passing
        # it keeps the graph static and gives the same result (only the causal part acts).
        padding_mask = seq != 0
        for block in m.blocks:
            x = block(x, padding_mask)
        pooled = m.final_norm(x)[:, -1, :]
        # The routing weights are broadcast over positions and the score uses only the
        # last one, so running the experts on the last position alone is exact.
        experts = torch.stack([e(pooled) for e in self.moe.experts.experts], dim=1)  # (1, K, D)
        e_t = self.moe.objective_embeddings.weight[0]                                # next-item
        routing = self.moe.router(pooled, e_t)                                       # (1, K)
        user = (routing[:, :, None] * experts).sum(dim=1)
        # table @ user^T, not user @ table^T: the exporter would store table^T as a new constant.
        scores = (m.item_embedding.weight @ F.normalize(user, dim=-1).T).T / self.item_norms
        scores = scores[:, 1:self.num_items + 1]  # drop padding (0) and the mask token
        return scores, routing


def main():
    p = argparse.ArgumentParser(description="Export the on-device demo assets")
    p.add_argument("--dataset", choices=["movielens", "beauty"], default="movielens",
                   help="beauty: Amazon Beauty 2014 defaults for the next three paths")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--movies-csv", default="data/raw/movie.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--num-users", type=int, default=300, help="sample users to ship")
    args = p.parse_args()
    beauty = args.dataset == "beauty"
    args.checkpoint = args.checkpoint or (BEAUTY["checkpoint"] if beauty else DEFAULT_MODEL)
    args.processed_dir = args.processed_dir or (BEAUTY["processed_dir"] if beauty else "data/processed_users10pct")
    args.out = args.out or (BEAUTY["out"] if beauty else "demo/mobile/assets")
    os.chdir(ROOT)
    os.makedirs(args.out, exist_ok=True)

    cfg, model, moe, _ = load_model(args.checkpoint, args.processed_dir, "cpu")
    metadata = load_metadata_matrix(args.processed_dir, cfg)
    if moe.variant != "arv_mmoe":
        raise SystemExit("export_mobile.py supports the ARV-MMoE models only")
    S, N = cfg.max_seq_length, cfg.num_items
    wrapper = OnDeviceRecommender(model, moe, metadata, N, S).eval()

    # ---------------------------------------------------------------- model
    onnx_path = os.path.join(args.out, "model.onnx")
    example = torch.randint(1, N + 1, (1, S))
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (example,), onnx_path, input_names=["seq"],
            output_names=["scores", "routing"], opset_version=17, dynamo=False,
        )
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    import onnx

    ops = sorted({n.op_type for n in onnx.load(onnx_path).graph.node})
    print(f"  ops: {', '.join(ops)}")
    # The phone page runs onnxruntime-web 1.18's single-threaded build (see
    # demo/mobile/index.html), which has no Trilu kernel. Folding turns the causal mask
    # into a constant; fail here rather than on the phone if it ever stops doing so.
    if "Trilu" in ops:
        raise SystemExit("the graph still contains Trilu, which the phone runtime lacks")
    # What the page needs to build the input (seq_len) and what its benchmark reports.
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in moe.parameters())
    meta = {"dataset": args.dataset, "seq_len": S, "num_items": N, "params": n_params, "num_experts": cfg.num_experts,
            "hidden_dim": cfg.hidden_dim, "onnx_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
            "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), ROOT)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote meta.json: {meta}")

    # ---------------------------------------------------------------- data
    rev_map = load_pickle(args.processed_dir, "reverse_item_mapping.pkl")
    if beauty:
        by_item = load_amazon_titles(cfg, rev_map, N)
    else:
        titles = load_titles(args.movies_csv)
        by_item = {i: titles.get(int(rev_map.get(i, -1)), (f"<item {i}>", "")) for i in range(1, N + 1)}
    popularity = load_item_popularity(args.processed_dir, N + 2)
    items = {"title": [], "genres": [], "likes": []}
    for i in range(1, N + 1):
        t, g = by_item[i]
        items["title"].append(t)
        items["genres"].append(g)
        items["likes"].append(int(popularity[i]))
    with open(os.path.join(args.out, "items.json"), "w") as f:
        json.dump(items, f, separators=(",", ":"))

    splits = load_pickle(args.processed_dir, "splits.pkl")
    # MovieLens user ids are numbers; Amazon reviewer ids are strings.
    user_map = {(k if beauty else int(k)): v
                for k, v in load_pickle(args.processed_dir, "user_mapping.pkl").items()}
    orig = {v: k for k, v in user_map.items()}
    # Leave-one-out (protocol v3) gives every user exactly one test item.
    pool = sorted(u for u, t in splits["test"].items() if len(t) >= 1)
    examples = [] if beauty else EXAMPLE_USERS
    chosen = examples + [u for u in random.Random(0).sample(pool, args.num_users)
                         if u not in examples]
    users = []
    for u in chosen:
        history = list(splits["train"].get(u, [])) + list(splits["val"].get(u, []))
        if history:
            users.append({"id": u, "ml": orig.get(u), "n": len(history),
                          # the model reads the last S; the rest is only excluded from recs
                          "history": history, "test": list(splits["test"].get(u, []))})
    with open(os.path.join(args.out, "users.json"), "w") as f:
        json.dump(users, f, separators=(",", ":"))
    print(f"wrote items.json ({N} items) and users.json ({len(users)} users)")

    # ---------------------------------------------------------------- check
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    checks = [users[0]["history"], users[1]["history"], users[2]["history"][:3], [1, 5735, 4474]]
    for hist in checks:
        exclude = set(hist)
        ref, _ = recommend(model, moe, cfg, metadata, hist, exclude, 10)
        recent = hist[-S:]
        seq = np.array([[0] * (S - len(recent)) + recent], dtype=np.int64)
        scores, routing = sess.run(None, {"seq": seq})
        scores = scores[0].copy()
        scores[[i - 1 for i in exclude]] = -np.inf
        top = (np.argsort(-scores)[:10] + 1).tolist()
        ref_items = [i for i, _ in ref]
        diff = max(abs(float(scores[i - 1]) - s) for i, s in ref)
        # Near-tied items may swap places under fp32 round-off. Accept a swap only when, at
        # every position, the two items' scores differ by less than TIE_TOL.
        gaps = [abs(float(scores[a - 1]) - float(scores[b - 1])) for a, b in zip(top, ref_items)]
        swaps = sum(a != b for a, b in zip(top, ref_items))
        if top == ref_items:
            status = "OK"
        elif max(gaps) < TIE_TOL:
            status = f"OK (near-tied items differ at {swaps} of 10 positions, max gap {max(gaps):.1e})"
        else:
            status = "MISMATCH"
        print(f"  check len={len(hist):4d}: top-10 {status}, max |score diff| {diff:.2e}, "
              f"routing {np.round(routing[0], 3).tolist()}")
        if status == "MISMATCH":
            raise SystemExit("ONNX and PyTorch disagree")


if __name__ == "__main__":
    main()
