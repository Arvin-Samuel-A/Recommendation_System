import torch
import pytest
import math

from configs.config import Config, OBJECTIVE_ORDER
from models.transformer import TransformerModel
from training.gradnorm import get_anchor_layer_params
from evaluation.routing_diagnostics import (
    routing_entropy,
    expert_utilization,
    objective_expert_heatmap,
    collapse_tests,
)
from evaluation.gradient_diagnostics import (
    gradient_cosine_similarity,
    per_objective_grad_norms,
)
from utils import get_device


class TestRoutingEntropy:
    def test_uniform_max_entropy(self):
        K = 4
        r = torch.ones(10, K) / K
        ent = routing_entropy(r)
        expected = math.log(K)
        assert (ent - expected).abs().max().item() < 1e-5

    def test_delta_zero_entropy(self):
        r = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ent = routing_entropy(r)
        assert ent.item() < 1e-5


class TestExpertUtilization:
    def test_uniform(self):
        K = 4
        r = torch.ones(100, K) / K
        util = expert_utilization(r)
        assert (util - 0.25).abs().max().item() < 1e-5


class TestHeatmapShape:
    def test_shape_matches_objectives_experts(self):
        K = 4
        per_obj = {
            "next_item": torch.randn(20, K).softmax(dim=-1),
            "masked_item": torch.randn(20, K).softmax(dim=-1),
            "multi_step": torch.randn(20, K).softmax(dim=-1),
            "contrastive": torch.randn(20, K).softmax(dim=-1),
        }
        heatmap = objective_expert_heatmap(per_obj)
        assert heatmap.shape == (len(OBJECTIVE_ORDER), K)


class TestCollapseTests:
    def test_uniform_detected(self):
        K = 4
        r = torch.ones(100, K) / K
        result = collapse_tests(r, cv_threshold=0.1, domination_threshold=0.9)
        assert result["fraction_near_uniform"] == 1.0
        assert result["fraction_single_expert_dominated"] == 0.0

    def test_dominated_detected(self):
        r = torch.tensor([[0.95, 0.02, 0.02, 0.01]] * 50)
        result = collapse_tests(r, cv_threshold=0.1, domination_threshold=0.9)
        assert result["fraction_single_expert_dominated"] == 1.0


class TestGradientCosineSimilarity:
    def test_symmetric_with_ones_diagonal(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.dropout = 0.0
        device = get_device(cfg)

        model = TransformerModel(cfg)
        model.train()
        anchor = get_anchor_layer_params(model)

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        losses = {}
        for name in ["next_item", "masked_item"]:
            logits = pooled @ model.item_emb_weight.T
            targets = torch.randint(1, 50, (B,), device=device)
            losses[name] = torch.nn.functional.cross_entropy(logits, targets)

        cos_sim = gradient_cosine_similarity(losses, anchor)
        T = len(losses)
        assert cos_sim.shape == (T, T)
        # REV-128: tightened from `< 0.01`, which was wide enough to accept the bf16
        # diagonal of 1.0039 that reached the real artifact. fp32 measurement puts a
        # vector's similarity with itself at 1.0 to well within 1e-4.
        for i in range(T):
            assert abs(cos_sim[i, i].item() - 1.0) < 1e-4, (
                f"diagonal entry {i} is {cos_sim[i, i].item():.6f}; a gradient's cosine "
                f"with itself must be 1.0 -- measure in fp32, not bf16"
            )
        # A cosine similarity cannot exceed 1. The bf16 diagonal did, at three of four
        # entries (max 1.00391), and a heatmap reading 1.004 is a paper figure inviting
        # the wrong question. Assert the RANGE so the violation cannot come back silently.
        # FP32_ULP absorbs fp32 round-off (measured 5e-7 here) while staying four orders
        # of magnitude tighter than the bf16 excursion this guards against.
        FP32_ULP = 1e-5
        assert cos_sim.max().item() <= 1.0 + FP32_ULP, (
            f"cosine similarity matrix contains {cos_sim.max().item():.6f}, which exceeds "
            f"1.0 by more than fp32 round-off -- it is being measured in a lower precision"
        )
        assert cos_sim.min().item() >= -1.0 - FP32_ULP
        assert abs(cos_sim[0, 1].item() - cos_sim[1, 0].item()) < 1e-6


class TestGradNorms:
    def test_norms_positive(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.dropout = 0.0
        device = get_device(cfg)

        model = TransformerModel(cfg)
        model.train()
        anchor = get_anchor_layer_params(model)

        B, S = 4, cfg.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        all_hidden = model(seq, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        losses = {}
        for name in ["next_item", "masked_item"]:
            logits = pooled @ model.item_emb_weight.T
            targets = torch.randint(1, 50, (B,), device=device)
            losses[name] = torch.nn.functional.cross_entropy(logits, targets)

        result = per_objective_grad_norms(losses, anchor)
        for name, norm in result["per_objective"].items():
            assert norm > 0
        assert result["mean"] > 0


class TestLongtailBucketPartition:
    def test_disjoint_complete_correct_percentiles(self):
        from evaluation.longtail import compute_popularity_buckets

        train_interactions = {
            1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            2: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
            3: [1, 2, 3, 4, 5, 13, 14, 15, 16, 17],
        }
        all_items = set(range(1, 21))

        buckets = compute_popularity_buckets(train_interactions, all_items)

        assert buckets["head"].isdisjoint(buckets["mid"])
        assert buckets["head"].isdisjoint(buckets["tail"])
        assert buckets["mid"].isdisjoint(buckets["tail"])
        assert buckets["head"] | buckets["mid"] | buckets["tail"] == all_items

        items_with_counts = set()
        for items in train_interactions.values():
            items_with_counts.update(items)
        zero_items = all_items - items_with_counts
        assert zero_items.issubset(buckets["tail"])

        # Owner decision (2026-09-18): the 20/20/60 split is over the WHOLE CATALOGUE,
        # not over the interacted subset. 20 items, 17 of them interacted -> 4 / 4 / 12.
        # Under the superseded scheme this fixture gave 3 / 4 / 13 (int(17*.2)=3), so
        # these assertions fail on it.
        n = len(all_items)
        assert len(buckets["head"]) == int(n * 0.20) == 4, (
            f"head={len(buckets['head'])}, expected 4 of {n} catalogue items"
        )
        assert len(buckets["mid"]) == 4, f"mid={len(buckets['mid'])}, expected 4"
        assert len(buckets["tail"]) == 12, f"tail={len(buckets['tail'])}, expected 12"

    def test_exact_20_20_60_proportions(self):
        """REV-028: T7.4's text is "correct percentiles", but the partition test only
        checked disjointness/completeness -- the proportions, which is precisely what
        REV-007 found broken (head 20 / mid 40 / tail 40), went unasserted.

        100 distinctly-ranked items so integer truncation is exact.
        """
        from evaluation.longtail import compute_popularity_buckets

        # item i gets (101 - i) interactions => a strict, unambiguous popularity order
        train_interactions = {}
        for i in range(1, 101):
            train_interactions[i] = [i] * (101 - i)
        all_items = set(range(1, 101))

        buckets = compute_popularity_buckets(train_interactions, all_items)

        assert len(buckets["head"]) == 20, f"head={len(buckets['head'])}, expected 20"
        assert len(buckets["mid"]) == 20, f"mid={len(buckets['mid'])}, expected 20"
        assert len(buckets["tail"]) == 60, f"tail={len(buckets['tail'])}, expected 60"

        # and they must be the RIGHT items: head is the most popular fifth
        assert buckets["head"] == set(range(1, 21))
        assert buckets["mid"] == set(range(21, 41))
        assert buckets["tail"] == set(range(41, 101))


class TestColdStartMetadataArms:
    """REV-001: the metadata ON and OFF arms must be able to differ.

    They were the same call -- neither passed `metadata=` and nothing toggled
    `use_metadata` -- so `meta_on` and `meta_off` were equal by construction and FR-4's
    central claim was reported from a comparison that could not differ.
    """

    def test_arms_produce_different_scores(self):
        import numpy as np

        from models.transformer import TransformerModel
        from models.moe import MoELayer
        from evaluation.metrics import score_candidates
        from utils import get_device, resolve_dtype

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 200
        cfg.vocab_size = cfg.num_items + 2
        cfg.feature_dim = 16
        cfg.use_metadata = True
        cfg.dropout = 0.0
        cfg.moe_variant = "arv_mmoe"

        device = get_device(cfg)
        dtype = resolve_dtype(cfg)
        torch.manual_seed(0)
        model = TransformerModel(cfg)
        model.eval()
        moe = MoELayer(cfg).to(device=device, dtype=dtype)
        moe.eval()
        assert model.metadata_projection is not None

        rng = np.random.RandomState(0)
        meta_matrix = rng.rand(cfg.vocab_size, cfg.feature_dim).astype(np.float32)
        meta_matrix[0] = 0.0  # padding row

        differing = 0
        for _ in range(5):
            seq = torch.tensor(
                rng.randint(1, cfg.num_items, size=cfg.max_seq_length), dtype=torch.long,
            )
            cand = torch.tensor(rng.randint(1, cfg.num_items, size=21), dtype=torch.long)
            meta_t = torch.as_tensor(meta_matrix[seq.numpy()], dtype=torch.float32)

            on = score_candidates(model, seq, cand, cfg, moe_layer=moe, metadata=meta_t)
            off = score_candidates(model, seq, cand, cfg, moe_layer=moe, metadata=None)
            if not torch.allclose(on.float(), off.float(), atol=1e-4):
                differing += 1

        assert differing > 0, (
            "metadata ON and OFF produced identical scores on every example -- the two "
            "arms are not actually different serving paths"
        )

    def test_off_arm_skips_projection_rather_than_erroring(self):
        """The OFF arm passes metadata=None to a model that HAS a projection; that must
        be an ordinary metadata-free forward, not an error."""
        from models.transformer import TransformerModel
        from utils import get_device

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 50
        cfg.vocab_size = 52
        cfg.feature_dim = 8
        cfg.use_metadata = True
        cfg.dropout = 0.0

        model = TransformerModel(cfg)
        model.eval()
        seq = torch.randint(1, 50, (1, cfg.max_seq_length), device=get_device(cfg))
        with torch.no_grad():
            out = model(seq, metadata=None, return_all_hidden=True)
        assert torch.isfinite(out.float()).all()


class TestCandidateSetConsistency:
    """REV-095 / AC-01: every evaluation entry point must score the SAME candidate rows.

    The test this replaces constructed no model and ran no evaluation -- it called
    `generate_eval_candidates` twice with identical arguments and asserted the results
    matched, i.e. that a seeded generator is deterministic. That is a different property
    (already covered by `TestEvalCandidates::test_identical_across_configs`) and it could
    not detect the real risk AC-01 exists to catch: two evaluation entry points scoring
    the same example against DIFFERENT candidate sets, which is exactly what REV-093
    found in `longtail.py` and `cold_start.py`.

    This drives the property at the evaluation boundary instead: build a small processed
    directory, run both entry points, and assert the candidate rows each one hands to
    `score_candidates_batch` are the stored `test_candidates.npy` rows.
    """

    N_EXAMPLES = 12
    NUM_ITEMS = 60

    @classmethod
    def _config(cls, tmp_path):
        from configs.config import Config

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = cls.NUM_ITEMS
        cfg.vocab_size = cls.NUM_ITEMS + 2
        cfg.feature_dim = 8
        cfg.use_metadata = True
        cfg.device = "cpu"
        cfg.num_sampled_negatives = 9
        cfg.eval_batch_size = 5
        cfg.processed_data_dir = str(tmp_path / "processed")
        cfg.log_dir = str(tmp_path / "logs")
        cfg.plot_dir = str(tmp_path / "plots")
        return cfg

    @classmethod
    def _write_processed(cls, cfg):
        import os
        import pickle

        import numpy as np

        from data.sequence_generator import generate_eval_candidates, save_eval_candidates

        processed = cfg.processed_data_dir
        os.makedirs(processed, exist_ok=True)
        rng = np.random.RandomState(0)
        n, S = cls.N_EXAMPLES, cfg.max_seq_length

        # Half the examples share a target: the case whose negatives were identical
        # under the target-seeded RandomState fallback.
        targets = [1 + (i % 3) for i in range(n)]
        user_ids = [i % 4 for i in range(n)]
        test_examples = {
            "targets": targets,
            "user_ids": user_ids,
            "sequences": rng.randint(1, cls.NUM_ITEMS + 1, size=(n, S)).tolist(),
        }
        with open(os.path.join(processed, "test_examples.pkl"), "wb") as f:
            pickle.dump(test_examples, f)

        train_history = {u: [10 + u, 11 + u] for u in set(user_ids)}
        eval_history = {u: v + [20 + u] for u, v in train_history.items()}
        for name, obj in (
            ("train_user_history.pkl", train_history),
            ("eval_user_history.pkl", eval_history),
            ("cold_items.pkl", {50, 51, 52}),
            ("warm_items.pkl", set(range(1, 50))),
            ("popularity_buckets.pkl", {
                "head": {1}, "mid": {2},
                "tail": set(range(3, cls.NUM_ITEMS + 1)),
            }),
        ):
            with open(os.path.join(processed, name), "wb") as f:
                pickle.dump(obj, f)

        pool = np.arange(1, 50)
        np.save(os.path.join(processed, "global_negative_pool.npy"), pool)
        np.save(
            os.path.join(processed, "item_metadata.npy"),
            np.vstack([
                np.zeros((1, cfg.feature_dim), np.float32),
                rng.rand(cls.NUM_ITEMS, cfg.feature_dim).astype(np.float32),
                np.zeros((1, cfg.feature_dim), np.float32),
            ]),
        )

        candidates = generate_eval_candidates(
            test_examples, pool, eval_history, cfg.num_sampled_negatives,
            padding_id=0, mask_id=cls.NUM_ITEMS + 1, seed=cfg.seed,
        )
        path = os.path.join(processed, "test_candidates.npy")
        save_eval_candidates(candidates, path)
        return np.load(path)

    @staticmethod
    def _model(cfg):
        import torch

        from training.pretrain import build_model

        torch.manual_seed(0)
        model, moe = build_model(cfg)
        model.eval()
        moe.eval()
        return model, moe

    @staticmethod
    def _record_candidates(monkeypatch, module_name):
        """Capture the candidate rows the entry point actually scores."""
        import importlib

        import evaluation.metrics as metrics_mod

        module = importlib.import_module(module_name)
        seen = []
        real = metrics_mod.score_candidates_batch

        def spy(model, seqs, candidate_ids, config, **kwargs):
            seen.append((candidate_ids.detach().cpu().clone(), kwargs.get("metadata")))
            return real(model, seqs, candidate_ids, config, **kwargs)

        monkeypatch.setattr(module, "score_candidates_batch", spy)
        return seen

    def test_longtail_and_cold_start_score_the_stored_candidate_rows(
        self, tmp_path, monkeypatch
    ):
        import numpy as np
        import torch

        from evaluation.cold_start import run_cold_start_evaluation
        from evaluation.longtail import run_longtail_evaluation

        cfg = self._config(tmp_path)
        stored = self._write_processed(cfg)
        model, moe = self._model(cfg)

        lt_seen = self._record_candidates(monkeypatch, "evaluation.longtail")
        run_longtail_evaluation(model, moe, cfg, "t-small")
        cs_seen = self._record_candidates(monkeypatch, "evaluation.cold_start")
        run_cold_start_evaluation(model, moe, cfg, "t-small")

        lt_rows = torch.cat([rows for rows, _ in lt_seen]).numpy()

        # Cold-start scores each batch twice -- metadata ON then OFF -- so the recorded
        # calls interleave per batch (b0,b0,b1,b1,...), not as two whole passes back to
        # back. Split them by the arm the spy captured; truncating the concatenation to
        # len(stored) would only line up if every example fitted in one batch.
        assert len(lt_seen) > 1, (
            "the fixture must span more than one batch for this to test anything -- "
            "raise N_EXAMPLES or lower eval_batch_size"
        )
        cs_on = torch.cat([rows for rows, meta in cs_seen if meta is not None]).numpy()
        cs_off = torch.cat([rows for rows, meta in cs_seen if meta is None]).numpy()

        assert np.array_equal(lt_rows, stored), (
            "run_longtail_evaluation scored candidates that are not the stored "
            "test_candidates.npy rows -- the long-tail numbers are on a different "
            "candidate protocol from the headline test metrics"
        )
        for arm, cs_rows in (("meta_on", cs_on), ("meta_off", cs_off)):
            assert np.array_equal(cs_rows, stored), (
                f"run_cold_start_evaluation's {arm} arm scored candidates that are not "
                f"the stored test_candidates.npy rows"
            )
        assert np.array_equal(lt_rows, cs_on), (
            "the two entry points disagree on the candidate set for the same examples"
        )

    def test_longtail_supplies_metadata(self, tmp_path, monkeypatch):
        """REV-094: the long-tail study must run the same serving path as every other
        reported number, metadata projection included."""
        from evaluation.longtail import run_longtail_evaluation

        cfg = self._config(tmp_path)
        self._write_processed(cfg)
        model, moe = self._model(cfg)

        seen = self._record_candidates(monkeypatch, "evaluation.longtail")
        run_longtail_evaluation(model, moe, cfg, "t-small")

        assert seen, "no scoring call was made"
        for rows, metadata in seen:
            assert metadata is not None, (
                "run_longtail_evaluation scored without metadata; the metadata "
                "projection contributed nothing on exactly the items it should help most"
            )
            assert metadata.shape[:2] == rows.shape[:1] + (cfg.max_seq_length,)


class TestLatencyBenchmarkPath:
    """REV-035b: FR-6 requires the benchmark to exercise the full SERVING path.

    `benchmark_latency` built `torch.zeros(1, max_seq_length)` -- an all-padding
    sequence. `TransformerModel.forward` computes `padding_mask = (seq != 0)`, so an
    all-zero input makes `has_padding` True and every block takes the explicit-mask
    branch, building an (B, H, S, S) float mask. Real traffic is valid item IDs and
    takes the `is_causal=True` fast path, so the reported numbers described a path
    production never runs.
    """

    @staticmethod
    def _config():
        from configs.config import Config

        config = Config()
        config.apply_model_config("t-small")
        config.num_items = 500
        config.vocab_size = 502
        config.use_metadata = False
        config.feature_dim = 0
        config.warmup_runs = 1
        config.timed_runs = 2
        config.device = "cpu"
        return config

    def test_timed_input_has_no_padding(self):
        from evaluation.flops import benchmark_latency
        from training.pretrain import build_model

        config = self._config()
        model, moe = build_model(config)
        report = benchmark_latency(model, moe, config)

        assert report["timed_padding_positions"] == 0, (
            "the benchmark is timing a padded sequence, not a serving request"
        )
        assert report["timed_sequence_length"] == config.max_seq_length

    def test_blocks_take_the_causal_fast_path(self):
        import models.transformer as transformer_module
        from evaluation.flops import benchmark_latency
        from training.pretrain import build_model

        config = self._config()
        model, moe = build_model(config)

        observed = []
        original = transformer_module.TransformerBlock.forward

        def spy(self, x, padding_mask=None, *args, **kwargs):
            observed.append(padding_mask)
            return original(self, x, padding_mask, *args, **kwargs)

        transformer_module.TransformerBlock.forward = spy
        try:
            benchmark_latency(model, moe, config)
        finally:
            transformer_module.TransformerBlock.forward = original

        assert observed, "no transformer block ran during the benchmark"
        assert all(mask is None for mask in observed), (
            "a padding mask reached a block, so the masked branch is still being timed"
        )

    def test_works_before_the_vocabulary_is_resolved(self):
        """`num_items` is 0 until stats are loaded; the input must still be valid."""
        from evaluation.flops import benchmark_latency
        from training.pretrain import build_model

        config = self._config()
        model, moe = build_model(config)
        config.num_items = 0
        report = benchmark_latency(model, moe, config)

        assert report["timed_padding_positions"] == 0
        assert report["mean_ms"] > 0


class TestLatencyBenchmarkMetadata:
    """REV-110: the timed request must carry real metadata, never an all-zero block.

    `dummy_meta` was `torch.zeros(1, S, feature_dim)`. A real request gathers
    `metadata_matrix[seq]`, whose rows are dense genome-relevance floats, and an all-zero
    block is precisely what a metadata-FREE model produces -- so the timed input could
    not be told apart from the `use_metadata=False` configuration on the one path FR-6
    calls the "full serving path".
    """

    @staticmethod
    def _config(tmp_path):
        from configs.config import Config

        config = Config()
        config.apply_model_config("t-small")
        config.num_items = 60
        config.vocab_size = 62
        config.use_metadata = True
        config.feature_dim = 8
        config.warmup_runs = 1
        config.timed_runs = 2
        config.device = "cpu"
        config.processed_data_dir = str(tmp_path / "processed")
        return config

    def test_gathers_real_metadata_rows(self, tmp_path):
        import torch

        from evaluation.flops import benchmark_latency
        from training.pretrain import build_model

        config = self._config(tmp_path)
        model, moe = build_model(config)

        torch.manual_seed(0)
        matrix = torch.randn(config.vocab_size, config.feature_dim)

        seen = []
        original = type(model).forward

        def spy(self, seq, metadata=None, **kwargs):
            seen.append(None if metadata is None else metadata.detach().cpu().clone())
            return original(self, seq, metadata=metadata, **kwargs)

        type(model).forward = spy
        try:
            report = benchmark_latency(model, moe, config, metadata_matrix=matrix)
        finally:
            type(model).forward = original

        assert report["timed_with_metadata"] is True
        assert report["timed_metadata_source"] == "item_metadata"
        assert seen and seen[0] is not None, "no metadata reached the model"
        block = seen[0].float()
        assert block.abs().sum().item() > 0, (
            "the timed metadata block is all zeros -- that is the metadata-free input, "
            "not the serving input"
        )
        # Every timed row must be an actual row of the matrix, i.e. a real gather. The
        # serving path casts to bfloat16, so compare against the matrix under the same
        # cast rather than against full float32 precision.
        cast = matrix.to(torch.bfloat16).float()
        rows = {tuple(r) for r in cast.tolist()}
        for row in block[0].tolist():
            assert tuple(row) in rows, (
                "a timed metadata row is not a row of the metadata matrix"
            )

    def test_falls_back_to_noise_not_zeros(self, tmp_path):
        """With no `item_metadata.npy` on disk the block must still be non-degenerate."""
        from evaluation.flops import benchmark_latency
        from training.pretrain import build_model

        config = self._config(tmp_path)
        model, moe = build_model(config)

        report = benchmark_latency(model, moe, config)
        assert report["timed_with_metadata"] is True
        assert report["timed_metadata_source"] == "randn"


class TestGradientDiagnosticsRestoresTrainingMode:
    """REV-107 / REV-109: the diagnostic must leave train/eval mode as it found it.

    `run_gradient_diagnostics` called `model.eval()` and never restored it; the
    training loop compensated by calling `model.train()` right afterwards, so the
    invariant was held by the caller rather than by the function that broke it. The
    second caller (`main.py::run_gradient_diagnostics_stage`) does not restore it.
    `moe_layer` was never switched at all, so the two disagreed mid-diagnostic.
    """

    @staticmethod
    def _run(train_mode: bool, dataloader=None):
        import torch
        from configs.config import Config
        from evaluation.gradient_diagnostics import run_gradient_diagnostics
        from training.gradnorm import get_anchor_layer_params
        from training.pretrain import build_model, _build_loss_fns

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 60
        cfg.vocab_size = 62
        cfg.use_metadata = False
        cfg.feature_dim = 0
        cfg.device = "cpu"
        cfg.grad_diag_num_batches = 1
        cfg.use_contrastive = False

        torch.manual_seed(0)
        model, moe = build_model(cfg)
        model.train(train_mode)
        moe.train(train_mode)
        loss_fns = _build_loss_fns(cfg, torch.device("cpu"))

        b, S = 2, cfg.max_seq_length
        batches = dataloader if dataloader is not None else [{
            "sequences": torch.randint(1, cfg.num_items + 1, (b, S)),
            "targets": torch.randint(1, cfg.num_items + 1, (b,)),
            "neg_ids": torch.randint(1, cfg.num_items + 1, (b, cfg.num_sampled_negatives)),
            "targets_k": torch.randint(1, cfg.num_items + 1, (b, cfg.multi_step_k)),
        }]
        run_gradient_diagnostics(
            model, moe, batches, loss_fns, get_anchor_layer_params(model, cfg),
            cfg, "t-small", write_report=False,
        )
        return model, moe

    def test_training_mode_is_restored(self):
        model, moe = self._run(train_mode=True)
        assert model.training, "model was left in eval mode after the diagnostic"
        assert moe.training, "moe_layer was left in eval mode after the diagnostic"

    def test_eval_mode_is_preserved(self):
        model, moe = self._run(train_mode=False)
        assert not model.training
        assert not moe.training

    def test_mode_is_restored_even_when_the_diagnostic_raises(self):
        """The restore lives in a `finally`, so a failing batch cannot strand the mode."""
        import pytest
        import torch
        from configs.config import Config
        from evaluation.gradient_diagnostics import run_gradient_diagnostics
        from training.gradnorm import get_anchor_layer_params
        from training.pretrain import build_model, _build_loss_fns

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 60
        cfg.vocab_size = 62
        cfg.use_metadata = False
        cfg.feature_dim = 0
        cfg.device = "cpu"
        cfg.grad_diag_num_batches = 1
        cfg.use_contrastive = False

        torch.manual_seed(0)
        model, moe = build_model(cfg)
        model.train()
        moe.train()

        class _Exploding:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            run_gradient_diagnostics(
                model, moe, _Exploding(), _build_loss_fns(cfg, torch.device("cpu")),
                get_anchor_layer_params(model, cfg), cfg, "t-small", write_report=False,
            )
        assert model.training, "an exception stranded the model in eval mode"
        assert moe.training, "an exception stranded the moe layer in eval mode"
