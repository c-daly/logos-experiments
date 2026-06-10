"""Unit tests for W2 predictors (logos-experiments#27) -- synthetic graphs only."""

import math

import pytest

from predict import (
    adamic_adar,
    backtest_split,
    build_adjacency,
    candidate_pairs,
    common_neighbors,
    preferential_attachment,
    rank_candidates,
    resource_allocation,
)


def _adj():
    # hub h connects a,b,c,d; rare neighbor r connects a,b only.
    return build_adjacency(
        [("h", "a"), ("h", "b"), ("h", "c"), ("h", "d"), ("r", "a"), ("r", "b")]
    )


class TestAdjacency:
    def test_undirected(self):
        adj = build_adjacency([("u", "v")])
        assert adj["u"] == {"v"} and adj["v"] == {"u"}

    def test_self_loops_dropped(self):
        adj = build_adjacency([("u", "u"), ("u", "v")])
        assert adj["u"] == {"v"}


class TestCandidates:
    def test_two_hop_non_edges_only(self):
        adj = _adj()
        cands = candidate_pairs(adj)
        assert frozenset(("a", "b")) in cands  # share h and r
        assert frozenset(("h", "a")) not in cands  # existing edge
        # a-d share h -> candidate; r-d share nothing... r's neighbors {a,b}, d's {h}: no overlap
        assert frozenset(("a", "d")) in cands
        assert frozenset(("r", "d")) not in cands


class TestScorers:
    def test_common_neighbors(self):
        adj = _adj()
        assert common_neighbors(adj, "a", "b") == 2  # h and r
        assert common_neighbors(adj, "a", "c") == 1  # h only

    def test_adamic_adar_weights_rare_neighbor_higher(self):
        adj = _adj()
        # a-b via hub h (deg 4) and rare r (deg 2): r contributes 1/log(2) > h's 1/log(4)
        score = adamic_adar(adj, "a", "b")
        assert score == pytest.approx(1 / math.log(4) + 1 / math.log(2))

    def test_resource_allocation(self):
        adj = _adj()
        assert resource_allocation(adj, "a", "b") == pytest.approx(1 / 4 + 1 / 2)

    def test_preferential_attachment(self):
        adj = _adj()
        assert preferential_attachment(adj, "a", "c") == 2 * 1  # deg(a)=2, deg(c)=1


class TestRanking:
    def test_descending_with_deterministic_ties(self):
        adj = _adj()
        ranked = rank_candidates(adj, common_neighbors)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)
        # run twice -> identical order (tie-break on sorted pair tuple)
        assert ranked == rank_candidates(adj, common_neighbors)


class TestBacktestSplit:
    def _edges(self):
        return [
            ("a", "b", "2026-06-08T01:00:00"),
            ("b", "c", "2026-06-08T02:00:00"),
            ("c", "d", "2026-06-08T03:00:00"),
            ("a", "c", "2026-06-08T04:00:00"),
            ("b", "d", "2026-06-08T05:00:00"),
        ]

    def test_temporal_cut(self):
        train, test = backtest_split(self._edges(), train_fraction=0.6)
        assert [(u, v) for u, v, _ in train] == [("a", "b"), ("b", "c"), ("c", "d")]
        assert test == {frozenset(("a", "c")), frozenset(("b", "d"))}

    def test_pairs_already_in_train_are_not_test_pairs(self):
        edges = self._edges() + [("a", "b", "2026-06-08T06:00:00")]  # repeat
        train, test = backtest_split(edges, train_fraction=0.5)
        assert frozenset(("a", "b")) not in test

    def test_order_is_timestamp_not_input_order(self):
        shuffled = list(reversed(self._edges()))
        train, test = backtest_split(shuffled, train_fraction=0.6)
        assert [(u, v) for u, v, _ in train] == [("a", "b"), ("b", "c"), ("c", "d")]
