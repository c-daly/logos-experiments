"""W1 ledger sanity gates (logos-experiments#26) -- the plan's calibration
tests, on synthetic snapshots with known-good answers.

The two load-bearing gates from the plan:
  * an obviously-good merge (duplicate types) must have dL < 0
  * deleting (evicting) a well-populated, edge-coherent type must have
    dL > 0
Encoding v1-as-built makes edge targets type-aware precisely so the second
gate can hold at all; see ledger.py docstring.
"""

import math

import pytest

from delta import evict_type, graft, merge_types, mint_type
from ledger import Snapshot, compute_ledger, delta


def _coherent_snapshot():
    """100 'entity' blob members + 10 'disease' members; 30 TREATED_BY
    edges all pointing at disease members -- a type that earns its bits."""
    membership = {f"e{i}": "entity" for i in range(100)}
    membership.update({f"d{i}": "disease" for i in range(10)})
    type_parents = {"entity": None, "disease": "entity"}
    edges = tuple(
        ("TREATED_BY", f"e{i}", f"d{i % 10}") for i in range(30)
    )
    return Snapshot(membership=membership, type_parents=type_parents, edges=edges)


def _duplicate_types_snapshot():
    """Two types with identical edge behaviour -- merge should win."""
    membership = {f"a{i}": "person" for i in range(20)}
    membership.update({f"b{i}": "human" for i in range(20)})
    membership.update({f"x{i}": "entity" for i in range(40)})
    type_parents = {"entity": None, "person": "entity", "human": "entity"}
    edges = tuple(("KNOWS", f"x{i}", f"a{i % 20}") for i in range(20)) + tuple(
        ("KNOWS", f"x{i + 20}", f"b{i % 20}") for i in range(20)
    )
    return Snapshot(membership=membership, type_parents=type_parents, edges=edges)


class TestLedgerShape:
    def test_total_decomposes(self):
        rep = compute_ledger(_coherent_snapshot())
        assert rep["L_total"] == pytest.approx(
            rep["L_model"] + rep["L_data_nodes"] + rep["L_data_edges"]
        )

    def test_all_terms_positive(self):
        rep = compute_ledger(_coherent_snapshot())
        assert rep["L_model"] > 0 and rep["L_data_nodes"] > 0
        assert rep["L_data_edges"] > 0

    def test_singleton_pays_the_most(self):
        s = _coherent_snapshot()
        membership = dict(s.membership)
        membership["lonely"] = "one_off_type"
        type_parents = dict(s.type_parents)
        type_parents["one_off_type"] = "entity"
        s2 = Snapshot(membership, type_parents, s.edges)
        rep = compute_ledger(s2)
        most_expensive = rep["top_expensive_nodes"][0]
        assert most_expensive[0] == "lonely"


class TestSanityGates:
    def test_duplicate_merge_pays(self):
        s = _duplicate_types_snapshot()
        assert delta(s, merge_types("human", "person")) < 0

    def test_evicting_a_coherent_populated_type_costs(self):
        s = _coherent_snapshot()
        assert delta(s, evict_type("disease")) > 0

    def test_minting_a_target_coherent_type_pays(self):
        # inverse of eviction: start from the blob, mint the coherent type
        s = _coherent_snapshot()
        blob = evict_type("disease")(s)
        d = delta(
            blob,
            mint_type("disease", [f"d{i}" for i in range(10)], parent="entity"),
        )
        assert d < 0

    def test_graft_is_position_blind_in_v1(self):
        s = _coherent_snapshot()
        membership = dict(s.membership)
        membership["c0"] = "condition"
        tp = dict(s.type_parents)
        tp["condition"] = "entity"
        s2 = Snapshot(membership, tp, s.edges)
        # moving 'disease' under 'condition' keeps hierarchy-edge COUNT equal
        assert delta(s2, graft("disease", "condition")) == pytest.approx(0.0)


class TestDeltaConsistency:
    def test_delta_equals_recompute(self):
        s = _duplicate_types_snapshot()
        op = merge_types("human", "person")
        before = compute_ledger(s)["L_total"]
        after = compute_ledger(op(s))["L_total"]
        assert delta(s, op) == pytest.approx(after - before)

    def test_ops_are_pure(self):
        s = _coherent_snapshot()
        before = compute_ledger(s)["L_total"]
        evict_type("disease")(s)
        assert compute_ledger(s)["L_total"] == pytest.approx(before)
