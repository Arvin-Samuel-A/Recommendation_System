"""FR-2 / AC-09: retrieval-then-rerank with target force-inclusion.

REV-064: `models/ann_retrieval.py` did not exist, `use_ann` and `ann_candidates` were
read nowhere, and T1.3 was marked complete regardless, so `steering/testing.md` Phase 15
("ANN target force-inclusion toy test") had no gate behind it.
"""

import torch

from configs.config import Config
from models.ann_retrieval import retrieve_candidates
from training.pretrain import build_model


def _setup(num_items=60, ann_candidates=10, seed=0):
    config = Config()
    config.apply_model_config("t-small")
    config.num_items = num_items
    config.vocab_size = num_items + 2
    config.use_metadata = False
    config.feature_dim = 0
    config.device = "cpu"
    config.ann_candidates = ann_candidates
    config.eval_batch_size = 4
    torch.manual_seed(seed)
    model, moe = build_model(config)
    return config, model, moe


class TestForceInclusion:
    def test_target_is_always_present_at_column_zero(self):
        config, model, moe = _setup()
        n = 12
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))

        report = retrieve_candidates(model, seqs, targets, config, moe_layer=moe)
        candidates = report["candidates"]

        assert candidates.shape == (n, config.ann_candidates)
        assert torch.equal(candidates[:, 0], targets), (
            "evaluate_model scores target_idx=0, so the target must sit at column 0"
        )

    def test_force_inclusion_replaces_the_lowest_scoring_candidate(self):
        """FR-2: 'Replace lowest-scoring candidate if target not in initial retrieval'."""
        config, model, moe = _setup(ann_candidates=5)
        n = 6
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))

        # Retrieve once with an arbitrary target to learn the natural top-K ordering.
        probe = torch.ones(n, dtype=torch.long)
        natural = retrieve_candidates(
            model, seqs, probe, config, moe_layer=moe,
        )["candidates"]

        # Choose a target that is NOT in the natural retrieval for each row.
        targets = []
        for row in range(n):
            retrieved = set(natural[row].tolist())
            targets.append(next(
                i for i in range(1, config.num_items + 1) if i not in retrieved
            ))
        targets = torch.tensor(targets, dtype=torch.long)

        report = retrieve_candidates(model, seqs, targets, config, moe_layer=moe)
        candidates = report["candidates"]

        assert report["num_force_included"] == n, "every row should have missed"
        assert report["force_inclusion_rate"] == 1.0
        # The forced target replaced exactly one slot; the rest of the retrieval
        # survives, minus the single lowest-scoring entry that was displaced.
        for row in range(n):
            got = set(candidates[row].tolist())
            assert int(targets[row]) in got
            # natural[row, 0] is the probe target at column 0, not a retrieved item.
            retrieved_only = set(natural[row, 1:].tolist())
            assert len(got & retrieved_only) >= config.ann_candidates - 2

    def test_no_force_inclusion_when_target_was_retrieved(self):
        config, model, moe = _setup(ann_candidates=8)
        n = 5
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))

        probe = torch.ones(n, dtype=torch.long)
        natural = retrieve_candidates(
            model, seqs, probe, config, moe_layer=moe,
        )["candidates"]
        # Target each row's own second-best retrieved item: it is genuinely in the top-K.
        targets = natural[:, 1].clone()

        report = retrieve_candidates(model, seqs, targets, config, moe_layer=moe)

        assert report["num_force_included"] == 0, (
            "a target already in the retrieval must not count as force-included"
        )
        assert report["force_inclusion_rate"] == 0.0
        assert torch.equal(report["candidates"][:, 0], targets)


class TestRetrievalFiltering:
    def test_invalid_ids_are_never_retrieved(self):
        config, model, moe = _setup()
        n = 10
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))

        candidates = retrieve_candidates(
            model, seqs, targets, config, moe_layer=moe,
        )["candidates"]

        assert (candidates != 0).all(), "padding ID 0 was retrieved"
        assert (candidates != config.num_items + 1).all(), "MASK was retrieved"
        assert (candidates >= 1).all() and (candidates <= config.num_items).all()

    def test_seen_items_are_filtered_but_the_target_is_exempt(self):
        config, model, moe = _setup()
        n = 8
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))
        user_ids = list(range(n))
        history = {u: [int(x) for x in seqs[u][:5]] for u in range(n)}

        candidates = retrieve_candidates(
            model, seqs, targets, config, user_ids=user_ids,
            user_history=history, moe_layer=moe,
        )["candidates"]

        for row in range(n):
            seen = set(history[user_ids[row]]) - {int(targets[row])}
            assert not (seen & set(candidates[row, 1:].tolist())), (
                f"row {row} retrieved an item the user had already seen"
            )
            # The target is a future interaction and must survive the filter.
            assert candidates[row, 0] == targets[row]

    def test_no_duplicates_within_a_row(self):
        config, model, moe = _setup()
        n = 10
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))

        candidates = retrieve_candidates(
            model, seqs, targets, config, moe_layer=moe,
        )["candidates"]

        for row in range(n):
            ids = candidates[row].tolist()
            assert len(set(ids)) == len(ids), f"row {row} has duplicates: {ids}"


class TestRetrievalContract:
    def test_result_is_consumable_by_evaluate_model(self):
        from evaluation.metrics import evaluate_model

        config, model, moe = _setup()
        n = 6
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))

        candidates = retrieve_candidates(
            model, seqs, targets, config, moe_layer=moe,
        )["candidates"]
        metrics = evaluate_model(model, seqs, candidates, config, moe_layer=moe)

        for name in ("ndcg", "mrr", "recall", "precision"):
            assert name in metrics
            assert 0.0 <= metrics[name]["mean"] <= 1.0

    def test_unresolved_vocabulary_raises(self):
        config, model, moe = _setup()
        config.num_items = 0
        seqs = torch.randint(1, 10, (2, config.max_seq_length))
        targets = torch.ones(2, dtype=torch.long)

        try:
            retrieve_candidates(model, seqs, targets, config, moe_layer=moe)
        except ValueError as exc:
            assert "num_items" in str(exc)
        else:
            raise AssertionError("an unresolved vocabulary must raise, not guess")

    def test_batching_does_not_change_the_result(self):
        config, model, moe = _setup()
        n = 9
        seqs = torch.randint(1, config.num_items + 1, (n, config.max_seq_length))
        targets = torch.randint(1, config.num_items + 1, (n,))

        one = retrieve_candidates(
            model, seqs, targets, config, moe_layer=moe, batch_size=1,
        )
        many = retrieve_candidates(
            model, seqs, targets, config, moe_layer=moe, batch_size=n,
        )

        assert torch.equal(one["candidates"], many["candidates"])
        assert one["force_inclusion_rate"] == many["force_inclusion_rate"]
