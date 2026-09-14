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
        for i in range(T):
            assert abs(cos_sim[i, i].item() - 1.0) < 0.01
        assert abs(cos_sim[0, 1].item() - cos_sim[1, 0].item()) < 0.01


class TestGradNorms:
    def test_norms_positive(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
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
