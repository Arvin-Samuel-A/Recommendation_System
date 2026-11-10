"""Amazon Reviews 2014 (McAuley et al.): one category's 5-core reviews and item metadata.

The reviews are mapped onto the frame `data.preprocessing` already filters, splits and
remaps -- `userId`, `movieId`, `rating`, `timestamp` (epoch seconds) -- with `userId` the
`reviewerID` and `movieId` the `asin`, both kept as strings. Item features replace
MovieLens' genome+genres block. Ported from the 2023-dataset fork
(`amazon_beauty/data/amazon.py`); the 2014 release differs in file names, field names,
compression (gzip), timestamp unit (seconds) and the metadata encoding (one Python dict
literal per line, not JSON).

Raw files, in `raw_data_dir`:

- `reviews_{category}_5.json.gz`: one strict JSON object per line; `reviewerID`, `asin`,
  `overall`, `unixReviewTime` are read, everything else is ignored.
- `meta_{category}.json.gz`: one Python dict literal per line (parsed with
  `ast.literal_eval`); `asin`, `title`, `price`, `brand`, `categories` are read and any of
  them may be missing.
"""
import ast
import gzip
import hashlib
import html
import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from configs.config import Config

logger = logging.getLogger(__name__)

REVIEW_FIELDS = ("reviewerID", "asin", "overall", "unixReviewTime")

#: Published statistics of the full 5-core files (the SASRec paper's Beauty row; verified
#: on the downloaded files 2026-11-07). A mismatch is logged as a warning, not raised: it
#: is expected for a user sample or a different `min_interactions`.
EXPECTED_STATS = {
    "Beauty": {"num_users": 22_363, "num_items": 12_101, "total_interactions": 198_502},
}

#: Seed of the title SVD's random range finder. Fixed rather than `config.seed`: the
#: processed directory is shared by every training seed, so its features must not depend
#: on which seed happened to run preprocessing.
TITLE_SVD_SEED = 0


def raw_files(category: str) -> Dict[str, str]:
    return {"reviews": f"reviews_{category}_5.json.gz", "meta": f"meta_{category}.json.gz"}


def _raw_path(config: Config, key: str) -> str:
    return os.path.join(config.raw_data_dir, raw_files(config.amazon_category)[key])


def _require(path: str, config: Config) -> None:
    if not os.path.exists(path):
        # No automatic download for the 2014 release (it is served from the UCSD
        # datarepo behind a request form). A download step would go here, guarded by
        # `config.auto_download_dataset`, as `acquire_dataset` does for MovieLens.
        raise FileNotFoundError(
            f"Amazon 2014 file '{os.path.basename(path)}' not found in "
            f"'{config.raw_data_dir}'. Expected {sorted(raw_files(config.amazon_category).values())} "
            f"there (category '{config.amazon_category}'); download them from the Amazon "
            f"Reviews 2014 release (5-core reviews + metadata) and re-run --stage preprocess."
        )


def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8")


def _first_line(path: str) -> str:
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                return line
    raise ValueError(f"{path} is empty")


def acquire(config: Config) -> str:
    """Check the category's two files are in `raw_data_dir`; return a fingerprint.

    The fingerprint is name + size of each file, as `_compute_fingerprint` does for the
    MovieLens CSVs.
    """
    paths = {key: _raw_path(config, key) for key in ("reviews", "meta")}
    for path in paths.values():
        _require(path, config)

    first = json.loads(_first_line(paths["reviews"]))
    missing = [k for k in REVIEW_FIELDS if k not in first]
    if missing:
        raise ValueError(
            f"{paths['reviews']} records lack {missing}. Found: {sorted(first)}"
        )
    first_meta = ast.literal_eval(_first_line(paths["meta"]))
    if not isinstance(first_meta, dict) or "asin" not in first_meta:
        raise ValueError(f"{paths['meta']}: the first record has no 'asin'")

    h = hashlib.sha256()
    for path in sorted(paths.values(), key=os.path.basename):
        h.update(os.path.basename(path).encode())
        h.update(str(os.path.getsize(path)).encode())
    fingerprint = h.hexdigest()[:16]
    logger.info(
        "Amazon 2014 '%s' raw files present in %s; dataset fingerprint: %s",
        config.amazon_category, config.raw_data_dir, fingerprint,
    )
    return fingerprint


#: Column of `read_reviews`' frame holding each review's line number in the file. The
#: chronological sort in `_read_ratings` breaks timestamp ties by it.
REVIEW_ORDER_COLUMN = "review_order"


def read_reviews(config: Config) -> pd.DataFrame:
    """Every review, in file order, in the shared schema plus `review_order`.

    `unixReviewTime` is day-granular, so many of a user's reviews share a timestamp. The
    standard SASRec preprocessing appends reviews in file order and then sorts each
    user's list by time with Python's stable sort, i.e. ties keep file order;
    `review_order` lets `_read_ratings` reproduce exactly that. Repeat (user, item)
    reviews are kept as separate interactions, as SASRec does (the 5-core Beauty file
    has none); their count is logged.
    """
    path = _raw_path(config, "reviews")
    _require(path, config)
    logger.info("Loading %s...", path)
    users: List[str] = []
    items: List[str] = []
    ratings: List[float] = []
    stamps: List[int] = []
    with _open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            users.append(r["reviewerID"])
            items.append(r["asin"])
            ratings.append(r["overall"])
            stamps.append(r["unixReviewTime"])
    df = pd.DataFrame({
        "userId": users,
        "movieId": items,
        "rating": np.asarray(ratings, dtype=np.float32),
        # Amazon 2014 `unixReviewTime` is already epoch seconds (midnight of the day).
        "timestamp": np.asarray(stamps, dtype=np.int64),
        REVIEW_ORDER_COLUMN: np.arange(len(users), dtype=np.int64),
    })
    repeats = int(df.duplicated(["userId", "movieId"]).sum())
    logger.info(
        "Amazon reviews: %d rows, %d users, %d items, %d repeat (user, item) reviews",
        len(df), df["userId"].nunique(), df["movieId"].nunique(), repeats,
    )
    return df


def check_expected_stats(config: Config, stats: Dict) -> None:
    """Log the dataset size, and WARN when it differs from the published 5-core numbers."""
    logger.info(
        "Amazon 2014 '%s' after preprocessing: %d users, %d items, %d interactions",
        config.amazon_category, stats["num_users"], stats["num_items"],
        stats["total_interactions"],
    )
    expected = EXPECTED_STATS.get(config.amazon_category)
    if expected is None:
        return
    got = {k: int(stats[k]) for k in expected}
    if got != expected:
        logger.warning(
            "Amazon 2014 '%s' statistics %s differ from the expected full 5-core %s "
            "(expected only for user_sample_fraction=1, min_interactions<=5, "
            "implicit_threshold=0 and the unmodified raw file)",
            config.amazon_category, got, expected,
        )


# ---------------------------------------------------------------------------------------
# Item metadata
# ---------------------------------------------------------------------------------------

#: The asin of a meta line, found without parsing the whole literal, so only the lines
#: of catalogue items pay for `ast.literal_eval`.
_ASIN = re.compile(r"""['"]asin['"]\s*:\s*['"]([^'"]+)['"]""")


def read_metadata(config: Config, asins) -> Dict[str, dict]:
    """asin -> its meta record, for the asins in `asins` (first record wins)."""
    path = _raw_path(config, "meta")
    _require(path, config)
    wanted = set(asins)
    meta: Dict[str, dict] = {}
    malformed = 0
    with _open_text(path) as f:
        for line in f:
            match = _ASIN.search(line)
            if match is not None and (match.group(1) not in wanted or match.group(1) in meta):
                continue
            if not line.strip():
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            asin = record.get("asin")
            if asin in wanted and asin not in meta:
                meta[asin] = record
    if malformed:
        logger.warning("%s: %d unparseable metadata lines skipped", path, malformed)
    return meta


def _parse_price(value) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 0 else None


def _category_tokens(categories) -> List[str]:
    """The second-level and leaf name of every `categories` path, deduplicated, in order.

    The top level is the category itself (nearly every Beauty item starts with "Beauty"),
    so it carries no information and is skipped; a one-level path contributes nothing.
    """
    tokens: List[str] = []
    if not isinstance(categories, (list, tuple)):
        return tokens
    for path in categories:
        if not isinstance(path, (list, tuple)) or len(path) < 2:
            continue
        for name in (path[1], path[-1]):
            name = html.unescape(str(name)).strip()
            if name and name not in tokens:
                tokens.append(name)
    return tokens


_TOKEN = re.compile(r"[a-z0-9]{2,}")


def _title_features(
    titles: List[str], dims: int, seed: int, max_terms: int = 5_000,
) -> np.ndarray:
    """TF-IDF over the titles -> rank-`dims` randomized SVD, each row L2-normalised.

    NumPy only: SciPy's compiled extensions (and so scikit-learn) fail to load in this
    project's venv on macOS 27. Terms in fewer than two titles are dropped and the
    `max_terms` most frequent kept, so the dense matrix stays catalogue x 5k. Returns
    (len(titles), d) with d <= dims; d is 0 when fewer than two terms survive.
    """
    docs = [_TOKEN.findall(t.lower()) for t in titles]
    df_counts: Dict[str, int] = {}
    for tokens in docs:
        for term in set(tokens):
            df_counts[term] = df_counts.get(term, 0) + 1
    kept = sorted(
        (t for t, c in df_counts.items() if c >= 2), key=lambda t: (-df_counts[t], t),
    )[:max_terms]
    dims = min(dims, len(kept) - 1, len(titles) - 1)
    if dims < 1:
        return np.zeros((len(titles), 0), dtype=np.float32)
    vocab = {t: j for j, t in enumerate(kept)}

    x = np.zeros((len(titles), len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(docs):
        for term in tokens:
            j = vocab.get(term)
            if j is not None:
                x[i, j] += 1.0
    np.log1p(x, out=x)  # sublinear term frequency
    idf = np.log((1 + len(titles)) / (1 + np.array([df_counts[t] for t in kept]))) + 1
    x *= idf.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    # Randomized range finder (Halko et al.) with power iterations, seeded. NumPy 2.2's
    # Accelerate-backed float32 matmul on macOS raises spurious divide/overflow/invalid
    # FP flags on finite inputs (the products match float64 to ~1e-6), so the flags are
    # silenced here and finiteness is asserted instead.
    rng = np.random.RandomState(seed)
    with np.errstate(all="ignore"):
        y = x @ rng.standard_normal((x.shape[1], dims + 10)).astype(np.float32)
        for _ in range(5):
            y, _ = np.linalg.qr(y)
            y = x @ (x.T @ y)
        q, _ = np.linalg.qr(y)
        u, s, _ = np.linalg.svd(q.T @ x, full_matrices=False)
        block = (q @ u[:, :dims]) * s[:dims]
    assert np.isfinite(block).all(), "title SVD produced non-finite values"
    # A title with no kept term projects to round-off (~1e-17), not exactly zero, and the
    # normalisation below would blow that up to a unit vector of noise.
    block[~x.any(axis=1)] = 0.0
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    return np.divide(block, norms, out=np.zeros_like(block), where=norms > 0).astype(
        np.float32
    )


def _top_k(counts: Dict[str, int], k: int) -> Dict[str, int]:
    """The `k` most frequent names -> column index (ties broken by name)."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {name: i for i, (name, _) in enumerate(ranked[:k])}


def extract_metadata(config: Config, item_map: Dict[str, int], num_items: int) -> np.ndarray:
    """(num_items + 2, D) item features; the pad row 0 and MASK row num_items+1 stay zero.

    D = title SVD dims + `amazon_top_brands` + `amazon_price_bins` +
    `amazon_top_categories`. The vocabularies -- TF-IDF terms, brands, price quantiles,
    categories -- are fitted on the catalogue's own metadata, never on interactions, so
    no split's activity reaches them. `salesRank` and `related` (also-bought / also-viewed)
    are deliberately unused: they summarise activity that includes the test period.
    """
    meta = read_metadata(config, item_map)

    asins = sorted(item_map, key=item_map.get)
    records = [meta.get(a, {}) for a in asins]
    # The 2014 metadata keeps HTML entities in text fields ("Nature&#39;s Plus").
    titles = [html.unescape(str(r.get("title") or "")) for r in records]
    brands = [html.unescape(str(r.get("brand") or "")).strip() for r in records]
    prices = [_parse_price(r.get("price")) for r in records]
    categories = [_category_tokens(r.get("categories")) for r in records]

    title_block = _title_features(titles, config.amazon_title_dim, TITLE_SVD_SEED)
    title_dim = title_block.shape[1]

    brand_counts: Dict[str, int] = {}
    for b in brands:
        if b:
            brand_counts[b] = brand_counts.get(b, 0) + 1
    brand_to_idx = _top_k(brand_counts, config.amazon_top_brands)

    category_counts: Dict[str, int] = {}
    for tokens in categories:
        for c in tokens:
            category_counts[c] = category_counts.get(c, 0) + 1
    category_to_idx = _top_k(category_counts, config.amazon_top_categories)

    # Price: quantile bins over the catalogue's known prices.
    known = np.asarray([p for p in prices if p is not None], dtype=np.float64)
    n_bins = config.amazon_price_bins
    edges = (
        np.quantile(known, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(known) else np.array([])
    )

    brand_off = title_dim
    price_off = brand_off + config.amazon_top_brands
    cat_off = price_off + n_bins
    feature_dim = cat_off + config.amazon_top_categories
    metadata_matrix = np.zeros((num_items + 2, feature_dim), dtype=np.float32)
    zero_rows = 0
    for row, asin in enumerate(asins):
        features = np.zeros(feature_dim, dtype=np.float32)
        features[:title_dim] = title_block[row]
        brand = brand_to_idx.get(brands[row])
        if brand is not None:
            features[brand_off + brand] = 1.0
        if prices[row] is not None and n_bins:
            bucket = int(np.searchsorted(edges, prices[row], side="right"))
            features[price_off + bucket] = 1.0
        for c in categories[row]:
            col = category_to_idx.get(c)
            if col is not None:
                features[cat_off + col] = 1.0
        if not features.any():
            zero_rows += 1
        metadata_matrix[item_map[asin]] = features

    assert np.all(metadata_matrix[0] == 0), "Row 0 (padding) must be all zeros"
    assert np.all(metadata_matrix[num_items + 1] == 0), (
        f"Row {num_items + 1} (MASK_ID) must be all zeros"
    )
    os.makedirs(config.processed_data_dir, exist_ok=True)
    np.save(os.path.join(config.processed_data_dir, "item_metadata.npy"), metadata_matrix)
    logger.info(
        "Amazon metadata: shape=%s (title=%d, brands=%d of %d, price bins=%d, "
        "categories=%d of %d); %d of %d items have metadata records, %d have known "
        "prices, %d all-zero rows",
        metadata_matrix.shape, title_dim, len(brand_to_idx), len(brand_counts), n_bins,
        len(category_to_idx), len(category_counts), len(meta), len(asins), len(known),
        zero_rows,
    )
    return metadata_matrix
