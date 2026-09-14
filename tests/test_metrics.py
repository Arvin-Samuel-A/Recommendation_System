import math
import torch
import pytest

from evaluation.metrics import compute_metrics, aggregate_metrics


class TestComputeMetrics:
    def test_target_at_rank_1(self):
        scores = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.0])
        m = compute_metrics(scores, target_idx=0, k=10)
        assert abs(m["ndcg"] - 1.0 / math.log2(2)) < 1e-6
        assert abs(m["mrr"] - 1.0) < 1e-6
        assert m["recall"] == 1.0
        assert abs(m["precision"] - 0.1) < 1e-6

    def test_target_at_rank_10(self):
        scores = torch.zeros(20)
        scores[0] = 0.5
        for i in range(1, 10):
            scores[i] = 10.0 - i
        m = compute_metrics(scores, target_idx=0, k=10)
        assert m["ndcg"] > 0
        assert m["mrr"] > 0
        assert m["recall"] == 1.0

    def test_target_at_rank_11_miss(self):
        scores = torch.zeros(15)
        for i in range(1, 15):
            scores[i] = 15 - i
        scores[0] = -1.0
        m = compute_metrics(scores, target_idx=0, k=10)
        assert m["ndcg"] == 0.0
        assert m["mrr"] == 0.0
        assert m["recall"] == 0.0
        assert m["precision"] == 0.0

    def test_exact_ndcg_value(self):
        scores = torch.tensor([5.0, 10.0, 3.0])
        m = compute_metrics(scores, target_idx=0, k=10)
        assert abs(m["ndcg"] - 1.0 / math.log2(3)) < 1e-6
        assert abs(m["mrr"] - 0.5) < 1e-6

    def test_single_candidate(self):
        scores = torch.tensor([1.0])
        m = compute_metrics(scores, target_idx=0, k=10)
        assert abs(m["ndcg"] - 1.0 / math.log2(2)) < 1e-6
        assert m["recall"] == 1.0


class TestAggregateMetrics:
    def test_mean_std(self):
        examples = [
            {"ndcg": 1.0, "mrr": 1.0, "recall": 1.0, "precision": 0.1},
            {"ndcg": 0.0, "mrr": 0.0, "recall": 0.0, "precision": 0.0},
        ]
        agg = aggregate_metrics(examples)
        assert abs(agg["ndcg"]["mean"] - 0.5) < 1e-6
        assert agg["ndcg"]["std"] > 0

    def test_empty_list(self):
        assert aggregate_metrics([]) == {}
