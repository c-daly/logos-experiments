"""Unit tests for the consolidation apply step's pure logic (no Neo4j)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apply_mapping import project_after_folds, select_folds  # noqa: E402


def _row(pred, target, tier, review=""):
    return {
        "predicate": pred, "proposed_target": target,
        "tier": tier, "review": review,
    }


def _snap(**counts):
    return {p: {"edge_count": c} for p, c in counts.items()}


class TestSelectFolds:
    def test_default_tiers_only(self):
        rows = [
            _row("A", "X", "high"),
            _row("B", "Y", "embed"),
            _row("C", "Z", "medium"),  # not in default tiers
            _row("D", "", "keep"),  # no target
        ]
        assert select_folds(rows) == [("A", "X"), ("B", "Y")]

    def test_review_accept_overrides_tier(self):
        rows = [_row("C", "Z", "medium", review="accept")]
        assert select_folds(rows) == [("C", "Z")]

    def test_review_reject_vetoes_in_scope_row(self):
        rows = [_row("A", "X", "high", review="keep")]
        assert select_folds(rows) == []

    def test_widening_tiers(self):
        rows = [_row("C", "Z", "medium")]
        assert select_folds(rows, tiers=("high", "embed", "medium")) == [("C", "Z")]


class TestProjectAfterFolds:
    def test_folds_reduce_df1(self):
        snap = _snap(SURV=10, A=1, B=1, C=1)
        # fold A,B into SURV; leave C
        proj = project_after_folds([("A", "SURV"), ("B", "SURV")], snap)
        assert proj["distinct_before"] == 4
        assert proj["distinct_after"] == 2  # SURV, C
        assert proj["df1_before"] == round(3 / 4, 3)
        assert proj["df1_after"] == round(1 / 2, 3)  # only C
        assert proj["edges_moved"] == 2

    def test_chained_folds_resolve_transitively(self):
        # A -> B -> C : A's edge should land on C, not B
        snap = _snap(C=5, B=1, A=1)
        proj = project_after_folds([("A", "B"), ("B", "C")], snap)
        # B and A both gone; C absorbs both
        assert proj["distinct_after"] == 1
        assert proj["df1_after"] == 0.0
