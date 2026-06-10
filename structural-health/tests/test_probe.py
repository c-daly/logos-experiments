"""Unit tests for the W0.1 structural-health probe (logos-experiments#25).

All tests run against the pure functions on synthetic data -- no Neo4j.
The frame-check lesson (2026-06-06) applies: these tests verify the math,
not the model; the BASELINE.md comparison against the live graph is the
actual measurement.
"""

import numpy as np
import pytest

from probe import (
    SemanticEdge,
    NodeRecord,
    asserted_type_map,
    build_matrix,
    predicate_stats,
    variance_curve,
)


def _nodes():
    return [
        NodeRecord("n1", "entity", "narwhal"),
        NodeRecord("n2", "entity", "tusk"),
        NodeRecord("n3", "entity", "ocean"),
        NodeRecord("t1", "type_definition", "marine mammal"),
        NodeRecord("t2", "type_definition", "body part"),
    ]


def _edges():
    return [
        SemanticEdge("IS_A", "n1", "t1"),
        SemanticEdge("IS_A", "n2", "t2"),
        SemanticEdge("PART_OF", "n2", "n1"),
        SemanticEdge("LIVES_IN", "n1", "n3"),
    ]


class TestAssertedTypeMap:
    def test_is_a_target_name_wins(self):
        m = asserted_type_map(_nodes(), _edges())
        assert m["n1"] == "marine mammal"
        assert m["n2"] == "body part"

    def test_untyped_node_falls_back_to_kind(self):
        m = asserted_type_map(_nodes(), _edges())
        assert m["n3"] == "entity"

    def test_type_definition_nodes_map_to_their_kind(self):
        m = asserted_type_map(_nodes(), _edges())
        assert m["t1"] == "type_definition"

    def test_multiple_is_a_edges_break_ties_deterministically(self):
        # review #33 (greptile): Neo4j returns edges in internal order, so a
        # multi-IS_A node must resolve identically regardless of edge order.
        nodes = _nodes() + [NodeRecord("t3", "type_definition", "arctic animal")]
        extra = SemanticEdge("IS_A", "n1", "t3")
        forward = asserted_type_map(nodes, _edges() + [extra])
        backward = asserted_type_map(nodes, [extra] + _edges())
        assert forward["n1"] == backward["n1"] == "arctic animal"  # lexicographic min

    def test_empty_type_name_keeps_realm_fallback(self):
        # review #33 (gemini): an unnamed type_definition must not overwrite
        # the realm fallback with "".
        nodes = _nodes() + [NodeRecord("t4", "type_definition", "")]
        m = asserted_type_map(nodes, [SemanticEdge("IS_A", "n3", "t4")])
        assert m["n3"] == "entity"


class TestBuildMatrix:
    def test_rows_are_data_nodes_only(self):
        mat, row_ids, feat_names = build_matrix(_nodes(), _edges())
        assert set(row_ids) == {"n1", "n2", "n3"}  # no edge/type_definition rows

    def test_features_are_rel_dir_neighbortype(self):
        mat, row_ids, feat_names = build_matrix(_nodes(), _edges())
        # n2 -[PART_OF]-> n1 gives n2 the outgoing feature against n1's type
        assert ("PART_OF", "out", "marine mammal") in feat_names
        # and n1 the incoming feature against n2's type
        assert ("PART_OF", "in", "body part") in feat_names

    def test_counts_land_in_the_right_cells(self):
        mat, row_ids, feat_names = build_matrix(_nodes(), _edges())
        r = {u: i for i, u in enumerate(row_ids)}
        f = {name: j for j, name in enumerate(feat_names)}
        dense = mat.toarray()
        assert dense[r["n2"], f[("PART_OF", "out", "marine mammal")]] == 1
        assert dense[r["n1"], f[("PART_OF", "in", "body part")]] == 1
        assert dense[r["n3"], f[("LIVES_IN", "in", "marine mammal")]] == 1

    def test_exclude_relations_filter(self):
        mat, row_ids, feat_names = build_matrix(
            _nodes(), _edges(), exclude_relations={"IS_A"}
        )
        assert not any(rel == "IS_A" for rel, _, _ in feat_names)
        # non-typing relations survive
        assert any(rel == "PART_OF" for rel, _, _ in feat_names)


class TestPredicateStats:
    def test_edge_df_counts(self):
        stats = predicate_stats(_nodes(), _edges())
        assert stats["distinct_relations"] == 3
        assert stats["edge_df"]["IS_A"] == 2
        assert stats["edge_df"]["PART_OF"] == 1

    def test_df1_fraction(self):
        # PART_OF and LIVES_IN each appear on exactly one edge -> 2/3
        stats = predicate_stats(_nodes(), _edges())
        assert stats["df1_fraction"] == pytest.approx(2 / 3)

    def test_edges_per_data_node(self):
        # 4 semantic edges, 3 data nodes
        stats = predicate_stats(_nodes(), _edges())
        assert stats["edges_per_data_node"] == pytest.approx(4 / 3)


class TestVarianceCurve:
    def test_low_rank_matrix_saturates_at_its_rank(self):
        # rank-2 matrix: 200 rows built from 2 orthogonal patterns + no noise
        rng = np.random.default_rng(42)
        basis = np.zeros((2, 50))
        basis[0, :25] = 1.0
        basis[1, 25:] = 1.0
        weights = rng.uniform(1, 5, size=(200, 2))
        from scipy.sparse import csr_matrix

        mat = csr_matrix(weights @ basis)
        curve = variance_curve(mat, ks=(1, 2, 5))
        assert curve[2] == pytest.approx(1.0, abs=1e-6)
        assert curve[5] == pytest.approx(1.0, abs=1e-6)
        assert curve[1] < 1.0

    def test_ks_larger_than_rank_are_clamped(self):
        from scipy.sparse import csr_matrix

        mat = csr_matrix(np.eye(4))
        curve = variance_curve(mat, ks=(16,))
        # k clamps to matrix rank limit instead of raising
        assert 16 in curve

    def test_degenerate_matrix_returns_zeros_without_fitting(self):
        # review #33 (gemini): 0/1-row matrices must early-return, not hit
        # the TF-IDF transformer.
        from scipy.sparse import csr_matrix

        mat = csr_matrix(np.ones((1, 5)))
        assert variance_curve(mat, ks=(16, 128)) == {16: 0.0, 128: 0.0}
