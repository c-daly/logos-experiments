"""Unit tests for W2 scoring + null rankings (logos-experiments#27)."""

import pytest

from score import (
    auc_sampled,
    precision_at_k,
    random_ranking,
    recency_ranking,
)

P = frozenset


class TestPrecisionAtK:
    def test_counts_hits_in_prefix(self):
        ranked = [(P(("a", "b")), 3.0), (P(("c", "d")), 2.0), (P(("e", "f")), 1.0)]
        test = {P(("a", "b")), P(("e", "f"))}
        assert precision_at_k(ranked, test, (1, 2, 3)) == {
            1: 1.0,
            2: 0.5,
            3: pytest.approx(2 / 3),
        }

    def test_k_beyond_ranking_uses_available_prefix(self):
        ranked = [(P(("a", "b")), 1.0)]
        assert precision_at_k(ranked, {P(("a", "b"))}, (5,)) == {5: 1.0}


class TestAUC:
    def test_perfect_predictor_scores_one(self):
        test = {P(("a", "b"))}
        negatives = [P(("c", "d")), P(("e", "f"))]
        scores = {P(("a", "b")): 10.0, P(("c", "d")): 1.0, P(("e", "f")): 0.0}
        auc = auc_sampled(scores.__getitem__, test, negatives, seed=0)
        assert auc == 1.0

    def test_constant_predictor_scores_half(self):
        test = {P(("a", "b"))}
        negatives = [P(("c", "d"))]
        auc = auc_sampled(lambda p: 1.0, test, negatives, seed=0)
        assert auc == 0.5


class TestNulls:
    def test_random_ranking_is_seed_deterministic(self):
        cands = [P((str(i), str(i + 1))) for i in range(0, 20, 2)]
        assert random_ranking(cands, seed=7) == random_ranking(cands, seed=7)
        assert random_ranking(cands, seed=7) != random_ranking(cands, seed=8)

    def test_recency_ranks_recently_active_endpoints_first(self):
        last_seen = {"a": 100, "b": 90, "x": 1, "y": 2}
        cands = [P(("x", "y")), P(("a", "b"))]
        ranked = recency_ranking(cands, last_seen)
        assert ranked[0][0] == P(("a", "b"))
