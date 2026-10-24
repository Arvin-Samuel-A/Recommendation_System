"""specs/eval-integrity: fp32 scoring, honest ties, autocast scoping, popularity floor, rescore.

Tiny in-memory models and the shared tiny fixture (`tests/conftest.py`), CPU only.
"""
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from utils import autocast_context


def _mixed_cfg(**overrides) -> Config:
    cfg = Config()
    cfg.device = "cpu"
    cfg.dtype = "bf16"
    cfg.mixed_precision = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------------------
# T1.2 -- autocast helpers with the weight cache disabled (FR-3.3, part of AC-09)
# ---------------------------------------------------------------------------------------


def _linear_and_input(seed: int = 0):
    torch.manual_seed(seed)
    lin = nn.Linear(16, 16)
    x = torch.randn(4, 16)
    opt = torch.optim.AdamW(lin.parameters(), lr=0.5)
    return lin, x, opt


def _step(lin, x, opt, out):
    opt.zero_grad(set_to_none=True)
    out.float().pow(2).mean().backward()
    opt.step()


def test_raw_autocast_region_forwards_stale_weights():
    """Characterisation of the torch behaviour behind the bug: inside ONE raw region with
    the cache on, a Linear stepped in place keeps forwarding its first-step bf16 cast."""
    lin, x, opt = _linear_and_input()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out1 = lin(x)
        _step(lin, x, opt, out1)
        out2 = lin(x)
        # A fresh region with the post-step weights gives a different output.
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            fresh = F.linear(x, lin.weight, lin.bias)
    assert torch.equal(out1, out2), "torch no longer caches casts inside one region"
    assert not torch.equal(out2, fresh)


def test_autocast_context_never_reads_a_stale_cast():
    """Each forward in its own `autocast_context` (cache off), nested inside the same raw
    region, forwards the updated weights."""
    cfg = _mixed_cfg()
    lin, x, opt = _linear_and_input()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with autocast_context(cfg, "cpu"):
            out1 = lin(x)
        _step(lin, x, opt, out1)
        with autocast_context(cfg, "cpu"):
            out2 = lin(x)
    with autocast_context(cfg, "cpu"):
        fresh = F.linear(x, lin.weight.detach(), lin.bias.detach())
    assert out2.dtype == torch.bfloat16
    assert torch.equal(out2, fresh)
    assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------------------
# In-memory tiny model (the setup tests/test_training.py::TestMoECacheLifetime uses)
# ---------------------------------------------------------------------------------------


def _tiny_cfg(mixed_precision: bool = True, moe_variant: str = "arv_mmoe", **overrides) -> Config:
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 200
    cfg.vocab_size = 202
    cfg.use_metadata = False
    cfg.feature_dim = 0
    cfg.device = "cpu"
    cfg.moe_variant = moe_variant
    cfg.batch_size = 4
    cfg.pin_memory = False
    cfg.dtype = "bf16"
    cfg.mixed_precision = mixed_precision
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def _tiny_model(cfg: Config, seed: int = 0):
    from training.pretrain import build_model

    torch.manual_seed(seed)
    return build_model(cfg)


def _eval_inputs(cfg: Config, n: int = 6, c: int = 21, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    seqs = torch.randint(1, cfg.num_items + 1, (n, cfg.max_seq_length), generator=gen)
    cands = torch.randint(1, cfg.num_items + 1, (n, c), generator=gen)
    return seqs, cands


def _fp32_reference(user_emb: torch.Tensor, item_weight: torch.Tensor, cands: torch.Tensor):
    from models.transformer import l2_normalize

    u = l2_normalize(user_emb.detach().float())
    items = l2_normalize(item_weight.detach()[cands].float())
    return torch.einsum("bd,bcd->bc", u, items)


# ---------------------------------------------------------------------------------------
# T1.3 -- fp32 scoring at the choke point (FR-1.1, AC-05)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mixed_precision,enclosing",
    [(True, False), (True, True), (False, False)],
    ids=["mixed-no-region", "mixed-enclosing-region", "pure-bf16"],
)
@pytest.mark.parametrize("moe_variant", ["arv_mmoe", "none"])
def test_score_candidates_batch_is_fp32(mixed_precision, enclosing, moe_variant):
    import contextlib

    from evaluation.metrics import score_candidates_batch

    cfg = _tiny_cfg(mixed_precision=mixed_precision, moe_variant=moe_variant)
    model, moe = _tiny_model(cfg)
    seqs, cands = _eval_inputs(cfg)

    captured = {}
    if moe_variant == "none":
        handle = model.register_forward_hook(
            lambda m, a, out: captured.__setitem__("user", out[:, -1, :])
        )
    else:
        handle = moe.register_forward_hook(
            lambda m, a, out: captured.__setitem__("user", out[0][:, -1, :])
        )
    region = (
        torch.autocast("cpu", dtype=torch.bfloat16) if enclosing else contextlib.nullcontext()
    )
    try:
        with region:
            scores = score_candidates_batch(model, seqs, cands, cfg, moe_layer=moe)
    finally:
        handle.remove()

    assert scores.dtype == torch.float32
    ref = _fp32_reference(captured["user"], model.item_emb_weight, cands)
    assert torch.allclose(scores, ref, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------------------
# T1.4 -- ANN retrieval scores in fp32 (FR-1.3, AC-06)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("enclosing", [False, True], ids=["no-region", "enclosing-region"])
def test_ann_retrieval_ranks_by_fp32_scores(monkeypatch, enclosing):
    import contextlib

    from models.ann_retrieval import retrieve_candidates
    from models.transformer import l2_normalize

    cfg = _tiny_cfg(mixed_precision=True, ann_candidates=10)
    model, moe = _tiny_model(cfg)
    seqs, _ = _eval_inputs(cfg, n=5)

    captured = {}
    handle = moe.register_forward_hook(
        lambda m, a, out: captured.__setitem__("user", out[0][:, -1, :])
    )
    seen_scores = []
    real_topk = torch.topk

    def spy_topk(t, *args, **kwargs):
        seen_scores.append(t.detach().clone())
        return real_topk(t, *args, **kwargs)

    # First pass: the reference top-1 becomes the target, so no force-inclusion or column
    # swap disturbs the retrieved order.
    with torch.no_grad():
        moe.eval()
        model.eval()
        with autocast_context(cfg, "cpu"):
            hidden = model(seqs, return_all_hidden=True)
            z, _ = moe(hidden, hidden[:, -1, :], 0)
    u = l2_normalize(z[:, -1, :].float())
    items = l2_normalize(model.item_emb_weight.detach().float())
    ref = u @ items.T
    ref[:, 0] = float("-inf")
    ref[:, cfg.num_items + 1] = float("-inf")
    ref_top = torch.topk(ref, cfg.ann_candidates, dim=1).indices
    targets = ref_top[:, 0].clone()

    monkeypatch.setattr(torch, "topk", spy_topk)
    region = (
        torch.autocast("cpu", dtype=torch.bfloat16) if enclosing else contextlib.nullcontext()
    )
    try:
        with region:
            result = retrieve_candidates(model, seqs, targets, cfg, moe_layer=moe)
    finally:
        handle.remove()

    assert len(seen_scores) == 1
    scores = seen_scores[0]
    assert scores.dtype == torch.float32
    ref_from_hook = l2_normalize(captured["user"].float()) @ items.T
    finite = torch.isfinite(scores)
    assert torch.allclose(scores[finite], ref_from_hook[finite], atol=1e-6, rtol=0)
    assert torch.equal(result["candidates"], ref_top)
    assert result["num_force_included"] == 0


# ---------------------------------------------------------------------------------------
# T1.5 -- scoring-site guard (FR-1.4, design D10, AC-07)
# ---------------------------------------------------------------------------------------

#: Every audited `einsum` / `@` / `matmul` site under `evaluation/` and in
#: `models/ann_retrieval.py`, as (file, enclosing function). A new site fails the test until
#: it is audited: a scoring site must score in fp32 outside autocast (eval-integrity FR-1).
AUDITED_SCORING_SITES = {
    # The two scoring choke points: fp32 normalise + einsum / matmul under
    # autocast_disabled, over candidate rows or over the whole catalogue.
    ("evaluation/metrics.py", "score_candidates_batch"),
    ("evaluation/metrics.py", "score_catalogue_batch"),
    # Full-vocabulary ANN scoring: fp32 normalise + matmul under autocast_disabled.
    ("models/ann_retrieval.py", "retrieve_candidates"),
}

_MATMUL_CALLS = {"einsum", "matmul", "mm", "bmm"}


def _scoring_sites(root: str):
    import ast
    import glob
    import os

    files = sorted(glob.glob(os.path.join(root, "evaluation", "*.py")))
    files.append(os.path.join(root, "models", "ann_retrieval.py"))
    sites = set()
    for path in files:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        tree = ast.parse(open(path).read(), filename=path)

        def visit(node, func):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = node.name
            hit = (
                (isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult))
                or (isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult))
                or (
                    isinstance(node, ast.Call)
                    and (
                        (isinstance(node.func, ast.Attribute) and node.func.attr in _MATMUL_CALLS)
                        or (isinstance(node.func, ast.Name) and node.func.id in _MATMUL_CALLS)
                    )
                )
            )
            if hit:
                sites.add((rel, func or "<module>"))
            for child in ast.iter_child_nodes(node):
                visit(child, func)

        visit(tree, None)
    return sites


def test_no_unaudited_scoring_sites():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sites = _scoring_sites(root)
    unaudited = sites - AUDITED_SCORING_SITES
    assert not unaudited, (
        f"unaudited einsum/@/matmul sites {sorted(unaudited)}: score through "
        "evaluation/metrics.py::score_candidates_batch (fp32, outside autocast), or audit "
        "the site and add it to AUDITED_SCORING_SITES with a justification"
    )
    stale = AUDITED_SCORING_SITES - sites
    assert not stale, f"allow-list entries no longer present: {sorted(stale)}"


# ---------------------------------------------------------------------------------------
# T1.6 -- popularity ranker and protocol stamp (FR-7.1, FR-1.5, AC-18)
# ---------------------------------------------------------------------------------------


def test_item_popularity_counts_are_exact():
    from evaluation.metrics import item_popularity

    # vocab 7: pad 0, items 1..5, MASK 6. Items 2 and 4 share a count; 5 is unseen.
    history = {10: [1, 2, 3], 11: [2, 4, 1], 12: [1, 4, 3], 13: [3]}
    pop = item_popularity(history, vocab_size=7)
    assert pop.dtype == torch.float32
    assert pop.tolist() == [0.0, 3.0, 2.0, 3.0, 2.0, 0.0, 0.0]


def test_item_popularity_zeroes_pad_and_mask_even_if_present():
    from evaluation.metrics import item_popularity

    pop = item_popularity({1: [0, 0, 6, 2]}, vocab_size=7)
    assert pop[0].item() == 0.0 and pop[6].item() == 0.0 and pop[2].item() == 1.0


def test_popularity_metrics_are_tie_averaged():
    import math

    from evaluation.metrics import evaluate_popularity, item_popularity

    history = {10: [1, 2, 3], 11: [2, 4, 1], 12: [1, 4, 3], 13: [3]}
    pop = item_popularity(history, vocab_size=7)   # counts 1:3 2:2 3:3 4:2 5:0
    cands = torch.tensor([
        [2, 1, 4, 5],   # target count 2: one higher (1), tied with 4 -> rank uniform {2,3}
        [1, 3, 2, 5],   # target count 3: tied with 3 -> rank uniform {1,2}
        [5, 1, 2, 3],   # target count 0: three higher, none tied -> rank 4
    ])
    m = evaluate_popularity(cands, pop, k=10, batch_size=2)
    row = [
        (1 / math.log2(3) + 1 / math.log2(4)) / 2,
        (1 + 1 / math.log2(3)) / 2,
        1 / math.log2(5),
    ]
    assert abs(m["ndcg"]["mean"] - sum(row) / 3) < 1e-6
    mrr = [(1 / 2 + 1 / 3) / 2, (1 + 1 / 2) / 2, 1 / 4]
    assert abs(m["mrr"]["mean"] - sum(mrr) / 3) < 1e-6
    assert set(m) == {"ndcg", "mrr", "recall", "precision"}
    assert set(m["ndcg"]) == {"mean", "std"}


def test_load_item_popularity_reads_train_history(tmp_path):
    import pickle

    from evaluation.metrics import load_item_popularity

    with open(tmp_path / "train_user_history.pkl", "wb") as f:
        pickle.dump({1: [1, 1, 2]}, f)
    assert load_item_popularity(str(tmp_path), 4).tolist() == [0.0, 2.0, 1.0, 0.0]


def test_eval_protocol_constant():
    from evaluation.metrics import EVAL_PROTOCOL

    assert EVAL_PROTOCOL == {
        "version": 3, "scoring": "fp32", "ties": "tie_averaged",
        "split": "leave_one_out", "candidates": "full_catalogue",
    }


# ---------------------------------------------------------------------------------------
# T2.1 -- train_one_step owns its region; stale-cache regression (FR-3.2/3.4, AC-08/09)
# ---------------------------------------------------------------------------------------


def _train_setup(lr: float = 0.05, **cfg_overrides):
    import random

    from torch.optim.lr_scheduler import LambdaLR

    from training.pretrain import _build_loss_fns

    cfg = _tiny_cfg(mixed_precision=True, contrastive_interval=1, **cfg_overrides)
    model, moe = _tiny_model(cfg)
    loss_fns = _build_loss_fns(cfg, torch.device("cpu"))
    params = list(model.parameters()) + list(moe.parameters())
    for fn in loss_fns.values():
        params += list(fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    return cfg, model, moe, loss_fns, optimizer, scheduler, random.Random(0)


def _train_batch(cfg: Config, seed: int):
    gen = torch.Generator().manual_seed(seed)
    b, n = cfg.batch_size, cfg.num_items
    return {
        "sequences": torch.randint(1, n + 1, (b, cfg.max_seq_length), generator=gen),
        "targets": torch.randint(1, n + 1, (b, cfg.max_seq_length), generator=gen),
    }


def _all_linears(model, moe, loss_fns):
    named = []
    for prefix, module in [("model", model), ("moe", moe)] + [
        (f"loss.{k}", v) for k, v in loss_fns.items()
    ]:
        for name, sub in module.named_modules():
            if isinstance(sub, nn.Linear):
                named.append((f"{prefix}.{name}", sub))
    return named


def _snapshot(linears):
    return {
        name: (
            lin.weight.detach().clone(),
            None if lin.bias is None else lin.bias.detach().clone(),
        )
        for name, lin in linears
    }


def _assert_step2_uses_current_weights(enclosing: bool):
    import contextlib

    from training.pretrain import train_one_step

    cfg, model, moe, loss_fns, optimizer, scheduler, rng = _train_setup()
    linears = _all_linears(model, moe, loss_fns)
    assert any(n.startswith("loss.multi_step") for n, _ in linears)

    region = (
        torch.autocast("cpu", dtype=torch.bfloat16) if enclosing else contextlib.nullcontext()
    )
    calls = []
    with region:
        before = _snapshot(linears)
        train_one_step(model, moe, _train_batch(cfg, 1), cfg, loss_fns, None,
                       optimizer, scheduler, 0, rng)
        after = _snapshot(linears)
        handles = [
            lin.register_forward_hook(
                lambda m, a, out, name=name: calls.append((name, a[0].detach(), out.detach()))
            )
            for name, lin in linears
        ]
        try:
            train_one_step(model, moe, _train_batch(cfg, 2), cfg, loss_fns, None,
                           optimizer, scheduler, 1, rng)
        finally:
            for h in handles:
                h.remove()

    ran = {name for name, _, _ in calls}
    # Encoder, experts, router and the multi-step projections all ran at step 2.
    assert any(n.startswith("model.") for n in ran)
    assert any(n.startswith("moe.") for n in ran)
    assert any(n.startswith("loss.multi_step") for n in ran)
    for name, inp, out in calls:
        assert out.dtype == torch.bfloat16, name
        with autocast_context(cfg, "cpu"):
            fresh = F.linear(inp, *after[name])
            stale = F.linear(inp, *before[name])
        assert torch.equal(out, fresh), f"{name} forwarded weights other than its current ones"
        assert not torch.equal(out, stale), f"{name} forwarded its pre-step-1 weights"


def test_train_one_step_forwards_current_weights():
    """AC-08: no caller region; every Linear at step 2 uses its post-step-1 weights."""
    _assert_step2_uses_current_weights(enclosing=False)


def test_train_one_step_forwards_current_weights_inside_cached_region():
    """AC-09: the same inside one enclosing raw torch.autocast with the cache ON."""
    _assert_step2_uses_current_weights(enclosing=True)


# ---------------------------------------------------------------------------------------
# T2.2 -- every forward site owns its region (FR-3.2, AC-10, AC-12)
# ---------------------------------------------------------------------------------------

FORWARD_SITES = [
    "train_one_step",
    "collect_validation_routing",
    "score_candidates_batch",
    "score_catalogue_batch",
    "run_gradient_diagnostics",
    "benchmark_latency",
    "retrieve_candidates",
    "train_user_lora",
]


def _run_forward_site(site: str):
    """Set up a tiny mixed-bf16 model, hook an encoder Linear, call `site` with no caller
    region. Returns the dtypes the hooked Linear produced."""
    import numpy as np

    cfg, model, moe, loss_fns, optimizer, scheduler, rng = _train_setup(
        warmup_runs=1, timed_runs=1, grad_diag_num_batches=1, ann_candidates=10,
        num_sampled_negatives=8,
    )
    batch = _train_batch(cfg, 3)
    seqs, cands = _eval_inputs(cfg)

    if site == "train_user_lora":
        from models.lora import inject_lora

        inject_lora(model, cfg.lora_rank)

    seen = []
    # k_proj stays a plain nn.Linear even after LoRA injection (q/v become LoRALinear).
    handle = model.blocks[0].attn.k_proj.register_forward_hook(
        lambda m, a, out: seen.append(out.dtype)
    )
    try:
        assert not torch.is_autocast_enabled("cpu")
        if site == "train_one_step":
            from training.pretrain import train_one_step

            train_one_step(model, moe, batch, cfg, loss_fns, None, optimizer, scheduler, 0, rng)
        elif site == "collect_validation_routing":
            from data.sequence_generator import RecDataset
            from training.pretrain import collect_validation_routing

            n = seqs.shape[0]
            ds = RecDataset(seqs, torch.ones(n, dtype=torch.long), torch.arange(n),
                            torch.full((n,), cfg.max_seq_length))
            collect_validation_routing(model, moe, ds, cfg)
        elif site == "score_candidates_batch":
            from evaluation.metrics import score_candidates_batch

            score_candidates_batch(model, seqs, cands, cfg, moe_layer=moe)
        elif site == "score_catalogue_batch":
            from evaluation.metrics import score_catalogue_batch

            scores = score_catalogue_batch(model, seqs, cfg, moe_layer=moe)
            assert scores.dtype == torch.float32
            assert scores.shape == (seqs.shape[0], cfg.num_items + 2)
        elif site == "run_gradient_diagnostics":
            from evaluation.gradient_diagnostics import run_gradient_diagnostics

            run_gradient_diagnostics(
                model, moe, [batch], loss_fns, list(model.blocks[-1].parameters()), cfg,
                "t-small", write_report=False,
            )
        elif site == "benchmark_latency":
            from evaluation.flops import benchmark_latency

            benchmark_latency(model, moe, cfg)
        elif site == "retrieve_candidates":
            from models.ann_retrieval import retrieve_candidates

            retrieve_candidates(model, seqs, cands[:, 0], cfg, moe_layer=moe)
        elif site == "train_user_lora":
            from training.local_train import train_user_lora

            train_user_lora(
                model, moe, [seqs[0], seqs[1]], [int(cands[0, 0]), int(cands[1, 0])],
                np.arange(1, cfg.num_items + 1), set(), cfg, torch.device("cpu"),
                rng=np.random.RandomState(0),
            )
        else:
            raise AssertionError(site)
    finally:
        handle.remove()
    assert not torch.is_autocast_enabled("cpu"), f"{site} leaked an autocast region"
    return seen


@pytest.mark.parametrize("site", FORWARD_SITES)
def test_forward_site_owns_its_autocast_region(site):
    """AC-10: under mixed precision with no caller region, an encoder Linear computes in
    bf16 at every forward site, and no region is left open afterwards."""
    seen = _run_forward_site(site)
    assert seen, f"{site} never ran the hooked encoder Linear"
    assert all(dt == torch.bfloat16 for dt in seen), f"{site}: {seen}"


@pytest.mark.parametrize("mixed_precision", [True, False], ids=["mixed", "pure-bf16"])
def test_evaluate_user_sees_loaded_lora_adapters(monkeypatch, mixed_precision):
    """AC-12: evaluate_user -> load_lora_state_dict(non-zero B) -> evaluate_user, all inside
    one enclosing raw autocast region; the second evaluation reflects the new adapters."""
    import evaluation.metrics as metrics_mod
    from models.lora import get_lora_state_dict, inject_lora, load_lora_state_dict
    from training.local_train import evaluate_user

    cfg = _tiny_cfg(mixed_precision=mixed_precision)
    model, moe = _tiny_model(cfg)
    inject_lora(model, cfg.lora_rank)
    # `score_candidates_batch` puts the model in eval mode but not the MoE layer; without
    # this the experts' dropout alone would make the two evaluations differ.
    moe.eval()
    seqs, cands = _eval_inputs(cfg, n=4)

    recorded = []
    real = metrics_mod.score_candidates_batch

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        recorded.append(out.detach().clone())
        return out

    monkeypatch.setattr(metrics_mod, "score_candidates_batch", spy)
    new_sd = {
        k: (torch.randn_like(v) * 0.5 if "lora_B" in k else v)
        for k, v in get_lora_state_dict(model).items()
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        evaluate_user(model, moe, list(seqs), [0] * 4, list(cands), cfg)
        load_lora_state_dict(model, new_sd)
        evaluate_user(model, moe, list(seqs), [0] * 4, list(cands), cfg)
    assert len(recorded) == 2
    assert not torch.equal(recorded[0], recorded[1]), "the loaded adapters were not used"
    # Control: re-evaluating without a change reproduces the second scores exactly.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        evaluate_user(model, moe, list(seqs), [0] * 4, list(cands), cfg)
    assert torch.equal(recorded[1], recorded[2])


# ---------------------------------------------------------------------------------------
# T2.3 -- no stage-wide autocast region (FR-3.1, AC-11)
# ---------------------------------------------------------------------------------------


def test_main_runs_stages_without_an_autocast_region(monkeypatch):
    import main as main_mod
    from utils import get_device

    seen = {}

    def stub_stage(config, args):
        device_type = get_device(config).type
        seen["mixed_precision"] = config.mixed_precision
        seen["enabled"] = {
            dt: torch.is_autocast_enabled(dt) for dt in {device_type, "cpu"}
        }

    monkeypatch.setitem(main_mod.STAGE_MAP, "_eval_integrity_stub", stub_stage)
    monkeypatch.setattr(main_mod, "VALID_STAGES", main_mod.VALID_STAGES + ["_eval_integrity_stub"])
    monkeypatch.setattr(main_mod, "save_config_snapshot", lambda *a, **k: None)
    main_mod.main([
        "--stage", "_eval_integrity_stub", "--model", "t-small",
        "--dtype", "bf16", "--mixed_precision",
    ])
    assert seen["mixed_precision"] is True
    assert seen["enabled"] and not any(seen["enabled"].values()), seen


# ---------------------------------------------------------------------------------------
# T2.4 -- training-semantics version in the fingerprint (FR-4, AC-14)
# ---------------------------------------------------------------------------------------


def test_fingerprint_changes_with_semantics_version(monkeypatch):
    import training.ablation_runner as ar

    cfg = Config()
    current = ar._training_fingerprint(cfg)
    assert ar.TRAINING_SEMANTICS_VERSION == 4
    monkeypatch.setattr(ar, "TRAINING_SEMANTICS_VERSION", 5)
    assert ar._training_fingerprint(cfg) != current


def _legacy_fingerprint(cfg: Config) -> str:
    """The pre-eval-integrity fingerprint: the same payload without the version key."""
    import hashlib
    import json
    from dataclasses import asdict

    from training.ablation_runner import _training_fields

    payload = json.dumps(_training_fields(asdict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def test_legacy_backbone_is_not_reused(tiny_processed_dir, tmp_path, monkeypatch):
    import copy

    import training.pretrain as pretrain
    from tests.conftest import tiny_config
    from training.ablation_runner import (
        _run_single_config, _training_fingerprint, backbone_store_dir, register_backbone,
        resolve_dataset_fields, resolve_study_configs,
    )

    base = tiny_config(tiny_processed_dir(), str(tmp_path))
    arm = resolve_study_configs("B", base)[0]
    resolved = copy.deepcopy(arm)
    resolve_dataset_fields(resolved)
    legacy = _legacy_fingerprint(resolved)
    assert legacy != _training_fingerprint(resolved)

    # A completed pre-fix backbone registered under the legacy fingerprint.
    legacy_cfg = copy.deepcopy(resolved)
    legacy_cfg.checkpoint_dir = backbone_store_dir(base.checkpoint_dir, legacy)
    os.makedirs(legacy_cfg.model_checkpoint_dir(), exist_ok=True)
    open(os.path.join(legacy_cfg.model_checkpoint_dir(), "best.pt"), "wb").close()
    register_backbone(base.checkpoint_dir, legacy, legacy_cfg, "pre-fix")

    class _Trained(Exception):
        pass

    calls = []

    def spy_run_pretrain(cfg, model_name):
        calls.append(cfg.checkpoint_dir)
        raise _Trained

    monkeypatch.setattr(pretrain, "run_pretrain", spy_run_pretrain)
    with pytest.raises(_Trained):
        _run_single_config(arm, "t-small", "B", base, True)
    assert len(calls) == 1
    assert legacy not in calls[0]


# ---------------------------------------------------------------------------------------
# T2.5 -- selection on the fixed metric; val popularity reference (FR-5, AC-15)
# ---------------------------------------------------------------------------------------


def _post_load_val_ndcg(cfg: Config, ckpt_path: str) -> float:
    """NDCG@k of a checkpoint loaded into a freshly built model, on the full val split."""
    from data.sequence_generator import load_tensorised
    from evaluation.metrics import evaluate_model, load_full_ranking
    from training.ablation_runner import resolve_dataset_fields
    from training.pretrain import build_model, load_checkpoint

    meta = resolve_dataset_fields(cfg)
    val = load_tensorised(
        os.path.join(cfg.processed_data_dir, "val_tensorised.pt"), meta,
        max_seq_length=cfg.max_seq_length,
    )
    cands = load_full_ranking(cfg.processed_data_dir, "val", val.targets, val.user_ids)
    model, moe = build_model(cfg)
    load_checkpoint(ckpt_path, model, moe)
    model.eval()
    moe.eval()
    m = evaluate_model(model, val.sequences, cands, cfg, moe_layer=moe,
                       metadata_matrix=val.metadata_matrix)
    return m["ndcg"]["mean"]


def test_logged_val_ndcg_equals_post_load_evaluation(tiny_processed_dir, tmp_path):
    from tests.conftest import tiny_config
    from training.pretrain import run_pretrain

    cfg = tiny_config(
        tiny_processed_dir(), str(tmp_path), mixed_precision=True, dtype="bf16",
        pretrain_epochs=3,
    )
    result = run_pretrain(cfg, "t-small")
    ckpt_dir = cfg.model_checkpoint_dir()

    stored, reloaded = [], []
    for e in range(3):
        path = os.path.join(ckpt_dir, f"epoch_{e:03d}.pt")
        stored.append(torch.load(path, weights_only=False)["val_ndcg"])
        reloaded.append(_post_load_val_ndcg(cfg, path))
    for e in range(3):
        assert abs(stored[e] - reloaded[e]) < 1e-6, (e, stored, reloaded)

    best = torch.load(os.path.join(ckpt_dir, "best.pt"), weights_only=False)
    assert best["best_epoch"] == max(range(3), key=lambda e: (reloaded[e], -e))
    assert result["best_epoch"] == best["best_epoch"]

    pop = result["val_popularity_reference"]
    assert set(pop) == {"ndcg", "mrr", "recall", "precision"}
    assert all(set(v) == {"mean", "std"} for v in pop.values())
    log = open(os.path.join(cfg.log_dir, "pretrain_t_small.log")).read()
    header = [l for l in log.splitlines() if "val_popularity_reference" in l]
    assert len(header) == 1 and "epoch=" not in header[0]
    assert f"popularity_ndcg@{cfg.eval_k}={pop['ndcg']['mean']:.6f}" in header[0]


# ---------------------------------------------------------------------------------------
# T3.1 -- ablation test on the full split; seeded sample (FR-6, AC-16, AC-17)
# ---------------------------------------------------------------------------------------


def _arm_base(tiny_processed_dir, out_dir, **overrides):
    from tests.conftest import tiny_config

    return tiny_config(
        tiny_processed_dir(), str(out_dir), pretrain_epochs=1, max_steps_per_epoch=2,
        **overrides,
    )


def _full_test_metrics(cfg: Config, ckpt_path: str):
    from data.sequence_generator import load_tensorised
    from evaluation.metrics import evaluate_model, load_full_ranking
    from training.ablation_runner import resolve_dataset_fields
    from training.pretrain import build_model, load_checkpoint

    import copy

    cfg = copy.deepcopy(cfg)
    meta = resolve_dataset_fields(cfg)
    test = load_tensorised(os.path.join(cfg.processed_data_dir, "test_tensorised.pt"), meta,
                           max_seq_length=cfg.max_seq_length)
    cands = load_full_ranking(cfg.processed_data_dir, "test", test.targets, test.user_ids)
    model, moe = build_model(cfg)
    load_checkpoint(ckpt_path, model, moe)
    model.eval()
    moe.eval()
    return evaluate_model(model, test.sequences, cands, cfg, moe_layer=moe,
                          metadata_matrix=test.metadata_matrix), len(test)


def test_arm_scores_the_full_test_split(tiny_processed_dir, tmp_path):
    """AC-16 (and the arm-result part of AC-19)."""
    import copy

    from evaluation.metrics import EVAL_PROTOCOL
    from training.ablation_runner import _run_single_config, resolve_study_configs

    base = _arm_base(tiny_processed_dir, tmp_path, ablation_eval_examples=0)
    arm = resolve_study_configs("B", base)[0]
    result = _run_single_config(copy.deepcopy(arm), "t-small", "B", base, True)
    assert result["status"] == "ok"

    from training.ablation_runner import find_backbone

    entry = find_backbone(base.checkpoint_dir, result["training_fingerprint"])
    expected, n_rows = _full_test_metrics(arm, entry["checkpoint_path"])
    assert result["num_eval_examples"] == n_rows
    assert result["test_eval"] == {
        "method": "full", "num_examples": n_rows, "total_rows": n_rows, "seed": None,
    }
    for metric, stats in expected.items():
        assert abs(result["metrics"][metric]["mean"] - stats["mean"]) < 1e-7, metric
    pop = result["popularity_reference"]
    assert set(pop) == {"ndcg", "mrr", "recall", "precision"}
    assert all(set(v) == {"mean", "std"} for v in pop.values())
    assert result["eval_protocol"] == EVAL_PROTOCOL


def test_select_eval_rows_is_seeded_and_not_a_prefix():
    import numpy as np

    from training.ablation_runner import select_eval_rows

    idx, desc = select_eval_rows(1000, 50, seed=0)
    idx2, desc2 = select_eval_rows(1000, 50, seed=0)
    idx3, desc3 = select_eval_rows(1000, 50, seed=1)
    assert np.array_equal(idx, idx2) and desc == desc2
    assert not np.array_equal(idx, idx3) and desc["indices_sha256"] != desc3["indices_sha256"]
    assert not np.array_equal(idx, np.arange(50))
    assert np.all(np.diff(idx) > 0) and len(set(idx.tolist())) == 50
    assert desc["method"] == "random" and desc["seed"] == 0 and desc["num_examples"] == 50
    assert desc["total_rows"] == 1000
    for n in (0, 1000, 5000):
        none_idx, full = select_eval_rows(1000, n, seed=0)
        assert none_idx is None and full["method"] == "full" and full["num_examples"] == 1000


def test_arm_records_a_seeded_random_sample(tiny_processed_dir, tmp_path):
    """AC-17: the arm records the sample; reruns agree; another seed differs."""
    import copy

    from training.ablation_runner import _run_single_config, resolve_study_configs

    base = _arm_base(tiny_processed_dir, tmp_path, ablation_eval_examples=16)
    arm = resolve_study_configs("B", base)[0]
    first = _run_single_config(copy.deepcopy(arm), "t-small", "B", base, True)
    again = _run_single_config(copy.deepcopy(arm), "t-small", "B", base, True)
    other = copy.deepcopy(arm)
    other.ablation_eval_seed = 5
    third = _run_single_config(other, "t-small", "B", base, True)

    te = first["test_eval"]
    assert te["method"] == "random" and te["seed"] == 0 and te["num_examples"] == 16
    assert first["num_eval_examples"] == 16
    assert te["indices_sha256"] == again["test_eval"]["indices_sha256"]
    assert first["metrics"] == again["metrics"]
    assert third["test_eval"]["seed"] == 5
    assert third["test_eval"]["indices_sha256"] != te["indices_sha256"]
    # Not the first 16 rows.
    import hashlib

    import numpy as np

    prefix = hashlib.sha256(np.arange(16, dtype=np.int64).tobytes()).hexdigest()
    assert te["indices_sha256"] != prefix
    # The same backbone served all three (post-training fields only).
    assert third["training_fingerprint"] == first["training_fingerprint"]


def test_config_rejects_negative_eval_fields():
    for field in ("ablation_eval_examples", "ablation_eval_seed"):
        cfg = Config()
        setattr(cfg, field, -1)
        with pytest.raises(ValueError, match=field):
            cfg.validate()
    assert Config().ablation_eval_examples == 0 and Config().ablation_eval_seed == 0


# ---------------------------------------------------------------------------------------
# A tiny study B trained ONCE per module, mixed bf16 (T3.2 and the rescore tests)
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_b(tiny_processed_dir, tmp_path_factory):
    """`run_ablation(study="B")`: two arms, 3 epochs, mixed bf16, CPU, full test split."""
    from tests.conftest import tiny_config
    from training.ablation_runner import run_ablation

    out = tmp_path_factory.mktemp("study_b")
    cfg = tiny_config(
        tiny_processed_dir(), str(out), mixed_precision=True, dtype="bf16",
        pretrain_epochs=3, max_steps_per_epoch=2, ablation_eval_examples=0,
    )
    results = run_ablation(cfg, "t-small", "B")
    assert all(r["status"] == "ok" for r in results["B"])
    return cfg, results["B"]


# ---------------------------------------------------------------------------------------
# T3.2 -- popularity in the study JSON, summary and plot (FR-7.3, FR-7.4, part of AC-19)
# ---------------------------------------------------------------------------------------


def _assert_metric_shape(block):
    assert set(block) == {"ndcg", "mrr", "recall", "precision"}, block
    assert all(set(v) == {"mean", "std"} for v in block.values()), block


def test_study_json_carries_popularity_per_arm(study_b):
    import json

    from evaluation.metrics import EVAL_PROTOCOL
    from training.ablation_runner import _arm_path, _results_root

    cfg, results = study_b
    with open(os.path.join(_results_root(cfg), "B_t_small.json")) as f:
        study = json.load(f)
    assert [r["label"] for r in study] == ["objective_conditioned", "input_only"]
    for entry in study:
        _assert_metric_shape(entry["popularity_reference"])
        assert entry["eval_protocol"] == EVAL_PROTOCOL
        assert entry["test_eval"]["method"] == "full"
        with open(_arm_path(cfg, "B", "t-small", entry["label"])) as f:
            arm = json.load(f)
        _assert_metric_shape(arm["popularity_reference"])
        assert arm["eval_protocol"] == EVAL_PROTOCOL
    # Every arm of a study is scored on the same rows, so they share one floor.
    assert study[0]["popularity_reference"] == study[1]["popularity_reference"]


def test_summary_carries_popularity_and_extra(study_b, tmp_path):
    import copy
    import json

    from training.ablation_runner import _results_root, _save_summary

    cfg, results = study_b
    cfg = copy.deepcopy(cfg)
    cfg.ablation_dir = str(tmp_path / "ablations")
    _save_summary(cfg, "t-small", {"B": results}, extra={"rescore_caveat": "text"})
    with open(os.path.join(_results_root(cfg), "summary_t_small.json")) as f:
        summary = json.load(f)
    b = summary["B"]
    best = max(r["metrics"]["ndcg"]["mean"] for r in results)
    pop = results[0]["popularity_reference"]["ndcg"]["mean"]
    assert b["best_ndcg"] == best
    assert b["popularity_reference_ndcg"] == pop
    assert abs(b["best_minus_popularity_ndcg"] - (best - pop)) < 1e-12
    assert b["best"] in {"objective_conditioned", "input_only"}
    assert summary["rescore_caveat"] == "text"


def test_study_plot_draws_the_popularity_line(study_b, tmp_path, monkeypatch):
    import copy

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    from training.ablation_runner import _plot_study

    cfg, results = study_b
    cfg = copy.deepcopy(cfg)
    cfg.plot_dir = str(tmp_path / "plots")
    calls = []
    real = Axes.axhline

    def spy(self, y=0, *args, **kwargs):
        calls.append((y, kwargs))
        return real(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", spy)
    _plot_study(cfg, "t-small", "B", results)
    pop = results[0]["popularity_reference"]["ndcg"]["mean"]
    assert any(
        y == pop and kw.get("linestyle") == "--" and kw.get("label") == "popularity (NDCG@10)"
        for y, kw in calls
    ), calls
    assert os.path.exists(os.path.join(cfg.plot_dir, "ablation_B_t_small.png"))


# ---------------------------------------------------------------------------------------
# T3.3 -- popularity + protocol in test_metrics / cold_start / longtail (rest of AC-19)
# ---------------------------------------------------------------------------------------


def test_evaluate_stage_writes_popularity_and_protocol(tiny_processed_dir, tmp_path):
    import json
    from types import SimpleNamespace

    import main as main_mod
    from evaluation.metrics import EVAL_PROTOCOL
    from tests.conftest import tiny_config
    from training.pretrain import run_pretrain

    cfg = tiny_config(tiny_processed_dir(), str(tmp_path), mixed_precision=True,
                      pretrain_epochs=1, max_steps_per_epoch=2)
    run_pretrain(cfg, "t-small")
    main_mod.run_evaluate_stage(cfg, SimpleNamespace(model="t-small"))
    with open(os.path.join(cfg.log_dir, "test_metrics_t_small.json")) as f:
        payload = json.load(f)
    _assert_metric_shape(payload["popularity_reference"])
    _assert_metric_shape({k: payload[k] for k in ("ndcg", "mrr", "recall", "precision")})
    assert payload["eval_protocol"] == EVAL_PROTOCOL
    assert "source_revision" in payload


def _fresh_tiny_model(tiny_processed_dir, out_dir, **overrides):
    from tests.conftest import tiny_config
    from training.ablation_runner import resolve_dataset_fields

    cfg = tiny_config(tiny_processed_dir(), str(out_dir), **overrides)
    resolve_dataset_fields(cfg)
    model, moe = _tiny_model(cfg)
    model.eval()
    moe.eval()
    return cfg, model, moe


def test_cold_start_report_carries_popularity(tiny_processed_dir, tmp_path):
    import json

    from evaluation.cold_start import run_cold_start_evaluation
    from evaluation.metrics import EVAL_PROTOCOL

    cfg, model, moe = _fresh_tiny_model(tiny_processed_dir, tmp_path, mixed_precision=True)
    report = run_cold_start_evaluation(model, moe, cfg, "t-small")
    with open(os.path.join(cfg.log_dir, "cold_start_t_small.json")) as f:
        payload = json.load(f)
    for key in ("cold_popularity", "warm_popularity"):
        _assert_metric_shape(report[key])
        _assert_metric_shape(payload[key])
    assert payload["eval_protocol"] == EVAL_PROTOCOL
    # Cold items have no training interactions: every cold candidate list ties at 0
    # except for warm negatives, so the floor is far from the target-favouring 1.0.
    assert payload["cold_popularity"]["ndcg"]["mean"] < 1.0
    assert os.path.exists(os.path.join(cfg.plot_dir, "cold_vs_warm_t_small.png"))


def test_longtail_report_carries_popularity(tiny_processed_dir, tmp_path):
    import json

    from evaluation.longtail import run_longtail_evaluation
    from evaluation.metrics import EVAL_PROTOCOL

    cfg, model, moe = _fresh_tiny_model(tiny_processed_dir, tmp_path, mixed_precision=True)
    report = run_longtail_evaluation(model, moe, cfg, "t-small")
    with open(os.path.join(cfg.log_dir, "longtail_t_small.json")) as f:
        payload = json.load(f)
    for key in ("popularity_head", "popularity_mid", "popularity_tail"):
        _assert_metric_shape(report[key])
        _assert_metric_shape(payload[key])
    assert payload["eval_protocol"] == EVAL_PROTOCOL
    assert os.path.exists(os.path.join(cfg.plot_dir, "longtail_t_small.png"))


# ---------------------------------------------------------------------------------------
# T4.1 -- per-fingerprint rescore engine (FR-8.1/8.3/8.4/8.9; AC-21, AC-23, AC-24, AC-25)
# ---------------------------------------------------------------------------------------


def _copy_run(study_b, dest):
    """A private copy of the trained study-B outputs, with every backbone.json relinked to
    the copied checkpoints, and the config pointed at the copy."""
    import copy
    import glob
    import shutil

    cfg, _ = study_b
    src = cfg.output_dir
    shutil.copytree(src, str(dest))
    for path in glob.glob(os.path.join(str(dest), "checkpoints", "ablations", "by_fingerprint",
                                       "*", "backbone.json")):
        text = open(path).read().replace(src, str(dest))
        open(path, "w").write(text)
    new = copy.deepcopy(cfg)
    for field in ("output_dir", "checkpoint_dir", "log_dir", "plot_dir", "ablation_dir"):
        setattr(new, field, getattr(cfg, field).replace(src, str(dest)))
    return new


def _spy_rescore_evals(monkeypatch, fail_on_call=None):
    """Count `evaluate_model` calls made by the rescore (optionally raising on call N)."""
    import training.ablation_rescore as rs

    calls = []
    real = rs.evaluate_model

    def spy(*args, **kwargs):
        calls.append(len(args[1]))
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("injected evaluation failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(rs, "evaluate_model", spy)
    return calls


def _arm_record(cfg, label, rescored=False):
    import json

    from training.ablation_rescore import rescored_config
    from training.ablation_runner import _arm_path

    root = rescored_config(cfg) if rescored else cfg
    with open(_arm_path(root, "B", "t-small", label)) as f:
        return json.load(f)


def _epoch_ckpt(cfg, label, e):
    from training.ablation_runner import find_backbone

    entry = find_backbone(cfg.checkpoint_dir, _arm_record(cfg, label)["training_fingerprint"])
    return os.path.join(os.path.dirname(entry["checkpoint_path"]), f"epoch_{e:03d}.pt")


B_LABELS = ("objective_conditioned", "input_only")


def test_rescore_of_a_post_fix_run_reproduces_logged_values(study_b, tmp_path, monkeypatch):
    """AC-21: on a run trained with the fixed code, the rescore is a no-op in value."""
    from training.ablation_rescore import run_ablation_rescore

    cfg = _copy_run(study_b, tmp_path / "run")
    calls = _spy_rescore_evals(monkeypatch)
    report = run_ablation_rescore(cfg, "t-small", "B")
    assert report["not_written"] == []
    assert len(calls) == 2 * (3 + 1)  # two fingerprints x (3 val epochs + 1 test pass)
    for label in B_LABELS:
        orig = _arm_record(cfg, label)
        rec = _arm_record(cfg, label, rescored=True)
        for e in range(3):
            stored = torch.load(_epoch_ckpt(cfg, label, e), weights_only=False)["val_ndcg"]
            assert abs(rec["rescore"]["per_epoch_val_ndcg"][str(e)] - stored) < 1e-6
            assert rec["rescore"]["per_epoch_original_val_ndcg"][str(e)] == stored
        assert rec["best_epoch"] == orig["best_epoch"]


def test_rescore_resumes_after_a_failed_epoch(study_b, tmp_path, monkeypatch):
    """AC-23: an exception in a fingerprint's second epoch evaluation fails the arm; a
    re-run evaluates only the remaining epochs plus the test pass."""
    import json

    from training.ablation_rescore import RESCORE_RECORD, run_ablation_rescore
    from training.ablation_runner import backbone_store_dir

    cfg = _copy_run(study_b, tmp_path / "run")
    calls = _spy_rescore_evals(monkeypatch, fail_on_call=2)
    first = run_ablation_rescore(cfg, "t-small", "B")
    failed = first["studies"]["B"]["failed"]
    assert list(failed) == ["objective_conditioned"] and "injected" in failed["objective_conditioned"]
    assert first["studies"]["B"]["rescored"] == ["input_only"]
    assert first["not_written"] == ["B"]
    assert len(calls) == 2 + 4

    fp = _arm_record(cfg, "objective_conditioned")["training_fingerprint"]
    rec_path = os.path.join(backbone_store_dir(cfg.checkpoint_dir, fp), RESCORE_RECORD)
    partial = json.load(open(rec_path))
    assert sorted(partial["epochs"]) == ["0"] and not partial["complete"]

    import training.ablation_rescore as rs
    import training.pretrain as pretrain

    loaded = []
    real_load = pretrain.load_checkpoint
    monkeypatch.setattr(pretrain, "load_checkpoint",
                        lambda path, *a, **k: loaded.append(path) or real_load(path, *a, **k))
    calls2 = _spy_rescore_evals(monkeypatch)
    second = run_ablation_rescore(cfg, "t-small", "B")
    assert second["not_written"] == []
    assert second["studies"]["B"]["current"] == ["input_only"]
    assert len(calls2) == 2 + 1  # epochs 1 and 2, then the test pass; never epoch 0 again
    assert not any(p.endswith("epoch_000.pt") for p in loaded[:2]), loaded
    assert json.load(open(rec_path))["complete"]


def test_arms_sharing_a_fingerprint_share_one_evaluation(study_b, tmp_path, monkeypatch):
    """AC-24: two original arm records referencing one trained fingerprint trigger one
    per-epoch evaluation sequence."""
    import json

    from training.ablation_rescore import run_ablation_rescore
    from training.ablation_runner import _arm_path

    cfg = _copy_run(study_b, tmp_path / "run")
    first = _arm_record(cfg, "objective_conditioned")
    path = _arm_path(cfg, "B", "t-small", "input_only")
    second = json.load(open(path))
    second["training_fingerprint"] = first["training_fingerprint"]
    json.dump(second, open(path, "w"), indent=2)

    calls = _spy_rescore_evals(monkeypatch)
    report = run_ablation_rescore(cfg, "t-small", "B")
    assert report["studies"]["B"]["rescored"] == list(B_LABELS)
    assert len(calls) == 3 + 1, calls
    a = _arm_record(cfg, "objective_conditioned", rescored=True)
    b = _arm_record(cfg, "input_only", rescored=True)
    assert a["metrics"] == b["metrics"]
    assert a["rescore"]["per_epoch_val_ndcg"] == b["rescore"]["per_epoch_val_ndcg"]


def test_epoch_with_foreign_config_fails_only_its_arm(study_b, tmp_path):
    """AC-25: a training-relevant field changed in one epoch checkpoint fails that arm
    with a message naming the file; the other arm is still rescored."""
    from training.ablation_rescore import run_ablation_rescore

    cfg = _copy_run(study_b, tmp_path / "run")
    bad = _epoch_ckpt(cfg, "input_only", 1)
    ck = torch.load(bad, weights_only=False)
    ck["config"]["num_experts"] = ck["config"]["num_experts"] + 1
    torch.save(ck, bad)

    report = run_ablation_rescore(cfg, "t-small", "B")
    info = report["studies"]["B"]
    assert info["rescored"] == ["objective_conditioned"]
    assert bad in info["failed"]["input_only"] and "num_experts" in info["failed"]["input_only"]
    assert report["not_written"] == ["B"]
    assert _arm_record(cfg, "objective_conditioned", rescored=True)["rescore"]["rescored"]


# ---------------------------------------------------------------------------------------
# T4.2 -- rescored arm records, collection and the CLI stage (FR-8.1/8.2/8.5-8.8;
#         AC-20, AC-22, AC-26)
# ---------------------------------------------------------------------------------------


def _sha256_tree(paths):
    import hashlib

    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sorted(paths)}


def _original_files(cfg):
    import glob

    from training.ablation_runner import _results_root

    root = _results_root(cfg)
    files = glob.glob(os.path.join(root, "arms", "B_t_small", "*.json"))
    files.append(os.path.join(root, "B_t_small.json"))
    files += glob.glob(os.path.join(cfg.checkpoint_dir, "**", "*.pt"), recursive=True)
    return files


def _rescored_files(cfg):
    import glob

    from training.ablation_rescore import rescored_config

    r = rescored_config(cfg)
    files = glob.glob(os.path.join(r.ablation_dir, "**", "*.json"), recursive=True)
    files += glob.glob(os.path.join(r.plot_dir, "**", "*.png"), recursive=True)
    files += glob.glob(os.path.join(cfg.checkpoint_dir, "**", "rescore.json"), recursive=True)
    return files


def test_rescore_stage_end_to_end(study_b, tmp_path, monkeypatch, caplog):
    """AC-20 and AC-22 through the CLI stage function."""
    import json
    import logging
    from types import SimpleNamespace

    import main as main_mod
    from evaluation.metrics import EVAL_PROTOCOL
    from training.ablation_rescore import RESCORE_CAVEAT, rescored_config
    from training.ablation_runner import _results_root

    cfg = _copy_run(study_b, tmp_path / "run")
    originals = _original_files(cfg)
    assert len([p for p in originals if p.endswith(".pt")]) >= 2 * 3
    before = _sha256_tree(originals)

    args = SimpleNamespace(model="t-small", study="B", allow_incomplete=False)
    with caplog.at_level(logging.WARNING, logger="training.ablation_rescore"):
        main_mod.run_ablation_rescore_stage(cfg, args)
    assert caplog.text.count(RESCORE_CAVEAT) == 2  # start and end

    for label in B_LABELS:
        orig = _arm_record(cfg, label)
        rec = _arm_record(cfg, label, rescored=True)
        rs = rec["rescore"]
        assert rs["rescored"] is True and rs["caveat"] == RESCORE_CAVEAT
        assert rs["epochs_scored"] == [0, 1, 2]
        per = {int(e): v for e, v in rs["per_epoch_val_ndcg"].items()}
        assert rec["best_epoch"] == max(sorted(per), key=lambda e: (per[e], -e))
        expected, n_rows = _full_test_metrics(
            _arm_cfg_from_record(orig), _epoch_ckpt(cfg, label, rec["best_epoch"]),
        )
        for metric, stats in expected.items():
            assert abs(rec["metrics"][metric]["mean"] - stats["mean"]) < 1e-7, metric
        assert rec["test_eval"]["method"] == "full" and rec["num_eval_examples"] == n_rows
        _assert_metric_shape(rec["popularity_reference"])
        _assert_metric_shape(rec["val_popularity_reference"])
        assert rec["eval_protocol"] == EVAL_PROTOCOL
        assert rs["original_metrics"] == orig["metrics"]
        assert rs["original_best_epoch"] == orig["best_epoch"]
        assert rs["selected_checkpoint"].endswith(f"epoch_{rec['best_epoch']:03d}.pt")
        # FR-8.5: provenance of the training run is kept; the record is fresh code.
        assert rec["source_revision"] == orig["source_revision"]
        assert rec["runtime"] == orig["runtime"]
        assert rec["arm_key"] == orig["arm_key"]
        assert "routing" not in rec and "routing" in rs["pre_rescore_extras"]

    r = rescored_config(cfg)
    study = json.load(open(os.path.join(_results_root(r), "B_t_small.json")))
    assert [e["label"] for e in study] == list(B_LABELS)
    assert all(e["rescore"]["rescored"] for e in study)
    assert os.path.exists(os.path.join(r.plot_dir, "ablation_B_t_small.png"))
    assert _sha256_tree(originals) == before, "an original file or checkpoint was modified"

    # AC-22: a second run evaluates nothing, rewrites nothing in substance, exits 0.
    produced = _rescored_files(cfg)
    snapshot = _sha256_tree(produced)
    calls = _spy_rescore_evals(monkeypatch)
    main_mod.run_ablation_rescore_stage(cfg, args)
    assert calls == []
    assert sorted(_rescored_files(cfg)) == sorted(produced)
    assert _sha256_tree(produced) == snapshot
    assert _sha256_tree(originals) == before


def _arm_cfg_from_record(record):
    """The arm's evaluation config, rebuilt from its recorded resolved_config."""
    cfg = Config()
    for key, value in record["resolved_config"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize("case", ["original_failed", "rescore_raised"])
def test_rescore_failed_arm_exits_70(study_b, tmp_path, monkeypatch, case):
    """A failed arm makes `ablation_rescore` exit ARM_FAILED_EXIT_CODE (70), as
    `--stage ablation` and `ablation_collect` do; the other arm is still rescored."""
    import json
    from types import SimpleNamespace

    import main as main_mod
    from training.ablation_runner import _arm_path

    cfg = _copy_run(study_b, tmp_path / "run")
    if case == "original_failed":
        path = _arm_path(cfg, "B", "t-small", "input_only")
        record = json.load(open(path))
        record.update(status="failed", error="KeyError: x")
        json.dump(record, open(path, "w"))
    else:
        _spy_rescore_evals(monkeypatch, fail_on_call=1)  # the first arm's first eval
    with pytest.raises(SystemExit) as exc:
        main_mod.run_ablation_rescore_stage(
            cfg, SimpleNamespace(model="t-small", study="B", allow_incomplete=False),
        )
    assert exc.value.code == 70
    survivor = "objective_conditioned" if case == "original_failed" else "input_only"
    assert _arm_record(cfg, survivor, rescored=True)["rescore"]["rescored"] is True


def test_rescore_all_writes_summary_with_caveat(study_b, tmp_path, monkeypatch):
    """`--study all` assembles what it can, writes the rescored summary with the caveat,
    and exits 1 because the other studies were never run."""
    import json
    from types import SimpleNamespace

    import main as main_mod
    from training.ablation_rescore import RESCORE_CAVEAT, rescored_config
    from training.ablation_runner import _results_root

    cfg = _copy_run(study_b, tmp_path / "run")
    with pytest.raises(SystemExit) as exc:
        main_mod.run_ablation_rescore_stage(
            cfg, SimpleNamespace(model="t-small", study="all", allow_incomplete=False),
        )
    assert exc.value.code == 1
    summary = json.load(open(os.path.join(_results_root(rescored_config(cfg)), "summary_t_small.json")))
    assert summary["rescore_caveat"] == RESCORE_CAVEAT
    assert summary["B"]["popularity_reference_ndcg"] is not None
    assert summary["B"]["best_ndcg"] is not None
    assert summary["A"]["complete"] is False


def test_rescore_quarantines_weight_dependent_extras(study_b, tmp_path):
    """AC-26: `latency` stays at the top level; `routing` moves to pre_rescore_extras."""
    import json

    from training.ablation_rescore import run_ablation_rescore
    from training.ablation_runner import _arm_path

    cfg = _copy_run(study_b, tmp_path / "run")
    path = _arm_path(cfg, "B", "t-small", "objective_conditioned")
    orig = json.load(open(path))
    assert "routing" in orig
    orig["latency"] = {"mean_ms": 1.25}
    json.dump(orig, open(path, "w"), indent=2)

    run_ablation_rescore(cfg, "t-small", "B")
    rec = _arm_record(cfg, "objective_conditioned", rescored=True)
    assert rec["latency"] == {"mean_ms": 1.25}
    assert "routing" not in rec
    assert rec["rescore"]["pre_rescore_extras"]["routing"] == orig["routing"]
    assert "router_parameters" in rec


def test_rescore_stage_is_registered_on_the_cli():
    import main as main_mod

    assert "ablation_rescore" in main_mod.VALID_STAGES
    assert main_mod.STAGE_MAP["ablation_rescore"] is main_mod.run_ablation_rescore_stage
    args = main_mod.parse_args([
        "--stage", "ablation_rescore", "--study", "B", "--model", "t-base",
        "--allow_incomplete",
    ])
    assert args.stage == "ablation_rescore" and args.allow_incomplete


# ---------------------------------------------------------------------------------------
# Review REV-001 -- multi-seed final_results carries the protocol stamp and the floor
# ---------------------------------------------------------------------------------------


def test_multi_seed_final_results_carry_protocol_and_popularity(tmp_path, monkeypatch):
    import argparse
    import json

    import main as main_mod
    import training.pretrain as pretrain
    from evaluation.metrics import EVAL_PROTOCOL

    base = Config()
    base.apply_model_config("t-base")
    base.final_seeds = [42, 123]
    base.checkpoint_dir = str(tmp_path / "ckpt")
    base.output_dir = str(tmp_path / "out")
    base.processed_data_dir = str(tmp_path / "processed")
    os.makedirs(base.processed_data_dir)
    with open(os.path.join(base.processed_data_dir, "dataset_stats.json"), "w") as f:
        json.dump({"num_items": 100, "num_users": 10}, f)

    def fake_pretrain(cfg, model_name):
        cfg.apply_model_config(model_name)
        cfg.num_items, cfg.num_users, cfg.vocab_size = 100, 10, 102
        os.makedirs(cfg.model_checkpoint_dir(), exist_ok=True)
        open(os.path.join(cfg.model_checkpoint_dir(), "best.pt"), "wb").close()
        return {"best_val_ndcg": 0.4, "best_epoch": 1}

    pop = {m: {"mean": 0.37, "std": 0.1} for m in ("ndcg", "mrr", "recall", "precision")}

    def fake_evaluate(cfg, args):
        os.makedirs(cfg.log_dir, exist_ok=True)
        payload = {m: {"mean": 0.3 + cfg.seed * 1e-4, "std": 0.1}
                   for m in ("ndcg", "mrr", "recall", "precision")}
        payload.update(popularity_reference=pop, eval_protocol=dict(EVAL_PROTOCOL))
        with open(os.path.join(cfg.log_dir, "test_metrics_t_base.json"), "w") as f:
            json.dump(payload, f)

    monkeypatch.setattr(pretrain, "run_pretrain", fake_pretrain)
    monkeypatch.setattr(main_mod, "run_evaluate_stage", fake_evaluate)
    main_mod.run_multi_seed_stage(base, argparse.Namespace(model="t-base"))

    with open(os.path.join(base.output_dir, "final_results_t_base.json")) as f:
        out = json.load(f)
    assert out["eval_protocol"] == EVAL_PROTOCOL
    assert out["popularity_reference"] == pop
    assert set(out["aggregated"]) == {"ndcg", "mrr", "recall", "precision"}
    assert out["aggregated"]["ndcg"]["n_seeds"] == 2


# ---------------------------------------------------------------------------------------
# Review REV-002 -- rescoring a LEGACY run (pre-fix fingerprint, foreign code)
# ---------------------------------------------------------------------------------------


def test_rescore_of_a_legacy_run(study_b, tmp_path, caplog):
    """The production scenario: the arm records a pre-fix (version-less) fingerprint and
    a foreign source_hash, and its checkpoints were written by other code with an older
    Config. The rescore must locate the backbone by the RECORDED fingerprint, accept the
    old checkpoint configs, and produce current-code records the collector accepts."""
    import glob
    import hashlib
    import json
    import logging
    import shutil

    from training.ablation_rescore import RESCORE_RECORD, run_ablation_rescore
    from training.ablation_runner import (
        _arm_path, _current_revision, _training_fingerprint, backbone_store_dir,
    )

    cfg = _copy_run(study_b, tmp_path / "run")
    label = "objective_conditioned"
    arm_file = _arm_path(cfg, "B", "t-small", label)
    orig = json.load(open(arm_file))
    current_fp = orig["training_fingerprint"]
    arm_cfg = _arm_cfg_from_record(orig)
    arm_cfg.num_items = orig["resolved_config"]["num_items"]
    legacy_fp = _legacy_fingerprint(arm_cfg)
    assert legacy_fp != current_fp and legacy_fp != _training_fingerprint(arm_cfg)

    # Move the store entry to the legacy fingerprint and relink backbone.json.
    old_store = backbone_store_dir(cfg.checkpoint_dir, current_fp)
    new_store = backbone_store_dir(cfg.checkpoint_dir, legacy_fp)
    shutil.move(old_store, new_store)
    marker = os.path.join(new_store, "backbone.json")
    text = open(marker).read().replace(current_fp, legacy_fp)
    open(marker, "w").write(text)

    # Checkpoints written by other code: a foreign source_revision (consistent across the
    # backbone) and an older Config with a field this code no longer has.
    foreign = {"commit": "c094b2c", "source_hash": "foreign000000000", "dirty": False}
    for pt in glob.glob(os.path.join(new_store, "**", "*.pt"), recursive=True):
        ck = torch.load(pt, weights_only=False)
        ck["source_revision"] = dict(foreign)
        ck["config"].pop("ablation_eval_seed", None)
        ck["config"]["legacy_only_field"] = 1
        torch.save(ck, pt)

    # The arm record as the pre-fix run wrote it.
    orig["training_fingerprint"] = legacy_fp
    orig["source_revision"] = dict(foreign)
    orig["code_revision"] = dict(orig["code_revision"], source_hash="foreign000000000")
    json.dump(orig, open(arm_file, "w"), indent=2)
    before = hashlib.sha256(open(arm_file, "rb").read()).hexdigest()

    with caplog.at_level(logging.WARNING, logger="training.ablation_rescore"):
        report = run_ablation_rescore(cfg, "t-small", "B")
    assert report["not_written"] == [], report
    assert sorted(report["studies"]["B"]["rescored"]) == sorted(B_LABELS)
    assert "legacy_only_field" in caplog.text  # unknown checkpoint keys are reported

    rec = _arm_record(cfg, label, rescored=True)
    assert rec["training_fingerprint"] == legacy_fp
    assert rec["code_revision"] == _current_revision()
    assert rec["source_revision"] == foreign
    assert rec["rescore"]["original_code_revision"]["source_hash"] == "foreign000000000"
    assert rec["rescore"]["epochs_scored"] == [0, 1, 2]
    assert rec["rescore"]["selected_checkpoint"].startswith(new_store)
    assert os.path.exists(os.path.join(new_store, RESCORE_RECORD))
    assert not os.path.exists(old_store)
    assert hashlib.sha256(open(arm_file, "rb").read()).hexdigest() == before


# ---------------------------------------------------------------------------------------
# Review REV-004 -- the identity check maps checkpoints instead of reading them in full
# ---------------------------------------------------------------------------------------


def test_identity_check_does_not_read_checkpoints_twice(study_b, tmp_path, monkeypatch):
    import collections

    import training.ablation_rescore as rs

    cfg = _copy_run(study_b, tmp_path / "run")
    loads = []
    real = torch.load

    def spy(path, *args, **kwargs):
        loads.append((os.path.basename(os.path.dirname(os.path.dirname(str(path)))),
                      os.path.basename(str(path)), bool(kwargs.get("mmap"))))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(rs.torch, "load", spy)
    report = rs.run_ablation_rescore(cfg, "t-small", "B")
    assert report["not_written"] == []

    epoch_loads = [(fp, name, mm) for fp, name, mm in loads if name.startswith("epoch_")]
    mapped = collections.Counter((fp, name) for fp, name, mm in epoch_loads if mm)
    full = collections.Counter((fp, name) for fp, name, mm in epoch_loads if not mm)
    assert len(mapped) == 2 * 3 and set(mapped.values()) == {1}  # identity check, mapped
    # Each epoch is read in full once for its val pass; only the selected epoch again
    # for the test pass.
    assert sum(full.values()) == 2 * 3 + 2
    assert all(n in (1, 2) for n in full.values())
