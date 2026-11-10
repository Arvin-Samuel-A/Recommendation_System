"""Web demo: a FastAPI server around scripts/demo_recommend.py, plus a one-page frontend.

Loads the run12 ARV-MMoE model (and the next-item-only baseline) once at startup and serves
recommendations as JSON; demo/index.html is the frontend. Scoring is the same production
path the CLI demo uses (`evaluation.metrics.score_candidates_batch`, fp32).

Run from the repository root:

    uv run python demo/server.py                  # http://127.0.0.1:8000
    uv run python demo/server.py --port 8080 --host 0.0.0.0

or with scripts/run_demo_server.sh, which works from any worktree.
"""
import argparse
import os
import random
import sys
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from data.sequence_generator import load_metadata_matrix  # noqa: E402
from demo_recommend import (  # noqa: E402
    DEFAULT_BASELINE, DEFAULT_MODEL, load_model, load_pickle, load_titles,
    popular, recommend, routing_weights, title_of,
)
from evaluation.metrics import load_item_popularity  # noqa: E402

INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


def short_name(cfg) -> str:
    """Model name and type only, e.g. "t-base Transformer, ARV-MMoE (4 experts)"."""
    if cfg.moe_variant == "arv_mmoe":
        arch = f"ARV-MMoE ({cfg.num_experts} experts)"
    elif cfg.moe_variant == "mmoe":
        arch = f"MMoE ({cfg.num_experts} experts)"
    else:
        arch = "no expert routing"
    return f"{cfg.model_name} Transformer, {arch}"


class Settings:
    checkpoint = os.environ.get("DEMO_CHECKPOINT", DEFAULT_MODEL)
    baseline_checkpoint = os.environ.get("DEMO_BASELINE_CHECKPOINT", DEFAULT_BASELINE)
    processed_dir = os.environ.get("DEMO_PROCESSED_DIR", "data/processed_users10pct")
    movies_csv = os.environ.get("DEMO_MOVIES_CSV", "data/raw/movie.csv")
    device = os.environ.get("DEMO_DEVICE", "cpu")


class Recommender:
    """Everything the demo needs, loaded once. The MoE layer keeps a per-call cache, so
    requests are served one at a time (`lock`)."""

    def __init__(self, s: Settings):
        torch.manual_seed(0)
        self.lock = threading.Lock()
        self.cfg, self.model, self.moe, _ = load_model(s.checkpoint, s.processed_dir, s.device)
        self.metadata = load_metadata_matrix(s.processed_dir, self.cfg)
        self.description = short_name(self.cfg)
        self.baseline = None
        if os.path.exists(s.baseline_checkpoint):
            b_cfg, b_model, b_moe, _ = load_model(s.baseline_checkpoint, s.processed_dir, s.device)
            self.baseline = (b_cfg, b_model, b_moe, load_metadata_matrix(s.processed_dir, b_cfg),
                             short_name(b_cfg))
        self.titles = load_titles(s.movies_csv)
        self.item_map = load_pickle(s.processed_dir, "item_mapping.pkl")         # movieId -> item
        self.rev_map = load_pickle(s.processed_dir, "reverse_item_mapping.pkl")  # item -> movieId
        self.popularity = load_item_popularity(s.processed_dir, self.cfg.num_items + 2)
        self.splits = load_pickle(s.processed_dir, "splits.pkl")
        self.user_map = {int(k): v for k, v in load_pickle(s.processed_dir, "user_mapping.pkl").items()}
        self.orig_user = {v: k for k, v in self.user_map.items()}
        self.random_pool = sorted(u for u, t in self.splits["test"].items() if len(t) >= 5)
        # (item, lowercase title) for search, most popular first.
        catalogue = [(item, self.titles[mid][0]) for mid, item in self.item_map.items() if mid in self.titles]
        catalogue.sort(key=lambda it: -float(self.popularity[it[0]]))
        self.search_index = [(item, title.lower()) for item, title in catalogue]

    def movie(self, item: int, score: Optional[float] = None, future: frozenset = frozenset()):
        title, genres = title_of(item, self.rev_map, self.titles)
        out = {"item": item, "title": title, "genres": genres}
        if score is not None:
            out["score"] = score
        if future:
            out["hit"] = item in future
        return out

    def search(self, q: str, limit: int):
        q = q.lower().strip()
        if not q:
            return []
        hits = [item for item, t in self.search_index if q in t][:limit]
        return [self.movie(i) for i in hits]

    def resolve(self, liked: List[Union[int, str]]):
        """Item ids pass through; a title becomes the most popular item whose title contains it."""
        items, unmatched = [], []
        for x in liked:
            if isinstance(x, int):
                if 1 <= x <= self.cfg.num_items:
                    items.append(x)
                else:
                    unmatched.append(str(x))
                continue
            q = x.lower().strip()
            match = next((item for item, t in self.search_index if q and q in t), None)
            if match is not None:
                items.append(match)
            else:
                unmatched.append(x)
        return items, unmatched


class RecommendRequest(BaseModel):
    user: Optional[int] = Field(None, description="internal user id")
    movielens_user: Optional[int] = Field(None, description="original MovieLens userId")
    liked: Optional[List[Union[int, str]]] = Field(
        None, description="a new viewer's liked movies, oldest first: item ids or title substrings")
    seed: Optional[int] = Field(None, description="seed for a random user (used when no one is named)")
    top_k: int = Field(10, ge=1, le=50)
    history: int = Field(10, ge=0, le=100, description="recent liked movies to return")
    compare: bool = False


rec: Optional[Recommender] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global rec
    print("Loading the model and data ...", flush=True)
    rec = Recommender(Settings())
    print(f"Ready: {rec.description}", flush=True)
    yield


app = FastAPI(title="ARV-MMoE recommendation demo", lifespan=lifespan)
# Lets demo/index.html work when opened straight from disk (origin "null"), not only via "/".
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/info")
def info():
    return {
        "model": rec.description,
        "baseline": rec.baseline[4] if rec.baseline else None,
        "num_items": rec.cfg.num_items,
        "num_users": len(rec.orig_user),
        "num_test_users": len(rec.random_pool),
    }


@app.get("/api/search")
def search(q: str, limit: int = 10):
    return rec.search(q, max(1, min(limit, 50)))


@app.post("/api/recommend")
def recommend_endpoint(req: RecommendRequest):
    future: frozenset = frozenset()
    unmatched: List[str] = []
    viewer: dict
    if req.liked:
        history, unmatched = rec.resolve(req.liked)
        if not history:
            raise HTTPException(400, "none of the liked movies matched the catalogue")
        viewer = {"kind": "new", "num_history": len(history)}
    else:
        if req.movielens_user is not None:
            if req.movielens_user not in rec.user_map:
                raise HTTPException(404, f"MovieLens user {req.movielens_user} is not in the 10% sample")
            user = rec.user_map[req.movielens_user]
        elif req.user is not None:
            if req.user not in rec.orig_user:
                raise HTTPException(404, f"no user {req.user} (ids are 0 .. {len(rec.orig_user) - 1})")
            user = req.user
        else:
            user = random.Random(req.seed).choice(rec.random_pool)
        # As in evaluation: the model sees train + validation; the test period is held out.
        history = list(rec.splits["train"].get(user, [])) + list(rec.splits["val"].get(user, []))
        if not history:
            raise HTTPException(400, f"user {user} has no history before the test period")
        future = frozenset(rec.splits["test"].get(user, []))
        viewer = {"kind": "user", "user": user, "movielens_user": rec.orig_user.get(user),
                  "num_history": len(history), "num_future": len(future)}
    exclude = set(history)

    with rec.lock:
        recs, (seq, meta) = recommend(rec.model, rec.moe, rec.cfg, rec.metadata, history, exclude, req.top_k)
        weights = routing_weights(rec.model, rec.moe, rec.cfg, seq, meta)
        baseline = None
        if req.compare and rec.baseline:
            b_cfg, b_model, b_moe, b_meta, b_desc = rec.baseline
            b_recs, _ = recommend(b_model, b_moe, b_cfg, b_meta, history, exclude, req.top_k)
            baseline = {"model": b_desc, "items": [rec.movie(i, s, future) for i, s in b_recs]}

    def lst(items):
        out = {"items": items}
        if future:
            out["hits"] = sum(m["hit"] for m in items)
        return out

    return {
        "viewer": viewer,
        "unmatched": unmatched,
        "history": [rec.movie(i) for i in history[-req.history:]] if req.history else [],
        "model": {"model": rec.description, **lst([rec.movie(i, s, future) for i, s in recs])},
        "routing": weights,
        "baseline": {"model": baseline["model"], **lst(baseline["items"])} if baseline else None,
        "popular": lst([rec.movie(i, s, future) for i, s in popular(rec.popularity, exclude, req.top_k)]),
    }


def main():
    import uvicorn

    p = argparse.ArgumentParser(description="ARV-MMoE recommendation demo server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="cpu (default), mps, cuda or auto")
    args = p.parse_args()
    if args.device:
        Settings.device = args.device
    os.chdir(ROOT)  # the default checkpoint and data paths are relative to the repo root
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
