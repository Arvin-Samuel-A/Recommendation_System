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


from evaluation.metrics import compute_metrics_batch  # noqa: E402


class TestTieAveraged:
    """specs/eval-integrity FR-2 (AC-01 - AC-04): ties never favour the target."""

    C = 101
    K = 10

    @staticmethod
    def _dcg_sum(k):
        return sum(1.0 / math.log2(r + 1) for r in range(1, k + 1))

    def test_all_scores_equal(self):
        # AC-01(a): the target ties all 100 others, so its rank is uniform on 1..101.
        m = compute_metrics(torch.zeros(self.C), target_idx=0, k=self.K)
        assert abs(m["ndcg"] - self._dcg_sum(self.K) / self.C) < 1e-6
        assert abs(m["mrr"] - sum(1.0 / r for r in range(1, self.K + 1)) / self.C) < 1e-6
        assert abs(m["recall"] - self.K / self.C) < 1e-6
        assert abs(m["precision"] - 1.0 / self.C) < 1e-6
        # AC-01(e): the former rule (`1 + #higher`) gave NDCG 1.0 here.
        assert m["ndcg"] < 0.05

    def test_target_tied_with_one_at_top(self):
        # AC-01(b): rank uniform on {1, 2}.
        scores = torch.linspace(0.0, -1.0, self.C)
        scores[1] = scores[0]
        m = compute_metrics(scores, target_idx=0, k=self.K)
        assert abs(m["ndcg"] - (1.0 + 1.0 / math.log2(3)) / 2) < 1e-6
        assert abs(m["mrr"] - 0.75) < 1e-6

    def test_ties_among_others_do_not_matter(self):
        # AC-01(c) / FR-2.3: the target is strictly highest; 50 others tie each other.
        scores = torch.full((self.C,), -1.0)
        scores[0] = 1.0
        scores[51:] = -2.0
        m = compute_metrics(scores, target_idx=0, k=self.K)
        assert m["ndcg"] == 1.0
        assert m["mrr"] == 1.0

    def test_batched_equals_single(self):
        # AC-01(d).
        torch.manual_seed(0)
        scores = torch.randint(0, 5, (32, self.C)).float()
        batched = compute_metrics_batch(scores, target_idx=0, k=self.K)
        for i in range(scores.shape[0]):
            single = compute_metrics(scores[i], target_idx=0, k=self.K)
            for name, val in single.items():
                assert abs(batched[name][i].item() - val) < 1e-7

    def test_monte_carlo_random_tie_breaking(self):
        # AC-02: the expectation over 2,000 random tie-breaks of the strict-rank metric.
        gen = torch.Generator().manual_seed(123)
        scores = torch.randint(0, 6, (20, 30), generator=gen).float()
        expected = compute_metrics_batch(scores, target_idx=0, k=self.K)
        acc = {name: torch.zeros(scores.shape[0], dtype=torch.float64) for name in expected}
        n_perm = 2000
        for _ in range(n_perm):
            # A random tie-break: integer scores plus jitter far below one score unit.
            jitter = torch.rand(scores.shape, generator=gen) * 1e-3
            broken = scores.double() + jitter.double()
            rank = (broken > broken[:, :1]).sum(1) + 1
            hit = (rank <= self.K).double()
            acc["ndcg"] += hit / torch.log2(rank.double() + 1)
            acc["mrr"] += hit / rank.double()
            acc["recall"] += hit
            acc["precision"] += hit / self.K
        for name in expected:
            mc = acc[name] / n_perm
            assert torch.allclose(expected[name].double(), mc, atol=0.01), name

    def test_no_ties_identical_to_former_rule(self):
        # AC-03: on a tie-free fp32 matrix the new metric equals the old rule exactly.
        torch.manual_seed(7)
        scores = torch.randn(256, self.C)
        assert all(len(set(row.tolist())) == self.C for row in scores)
        rank = (scores > scores[:, :1]).sum(1) + 1
        rank_f = rank.float()
        hit = (rank <= self.K).float()
        old = {
            "ndcg": hit / torch.log2(rank_f + 1.0),
            "mrr": hit / rank_f,
            "recall": hit,
            "precision": hit / self.K,
        }
        new = compute_metrics_batch(scores, target_idx=0, k=self.K)
        for name in old:
            assert torch.equal(new[name], old[name]), name

    def test_nan_raises(self):
        # AC-04.
        scores = torch.zeros(4, self.C)
        scores[2, 5] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            compute_metrics_batch(scores, target_idx=0, k=self.K)
