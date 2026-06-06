"""Cascade tests -- 2026-06-06 contract (simulate_cluster_placement).

One named cluster -> one placement. The namer returns {name, parent,
residual_ids}; the cascade RE-PARENTS the kept subgraph: reuse attaches it
under an existing type (no new node), mint creates one node under an existing
parent. Minting/attaching under a domain root is always legal. No chain is
built -- ancestry lives in the graph; root/depth are reads of the catalog
(the frozen mirror of that structure).
"""

from __future__ import annotations

from harness.cascade import (
    ceiling_violation,
    simulate_cluster_placement,
)

M = ["m1", "m2"]


def _place(name, parent, *, members=None, residual_ids=None, cbu, bn, **kw):
    return simulate_cluster_placement(
        cluster_id="c",
        member_ids=members if members is not None else M,
        name=name,
        parent=parent,
        residual_ids=residual_ids or [],
        catalog_by_uuid=cbu,
        by_norm=bn,
        **kw,
    )


# --- ceiling (still used by the placement) -------------------------------


def test_ceiling_violation_word_count():
    assert ceiling_violation("a b c d") is True
    assert ceiling_violation("red sports car") is False  # 3 words


def test_ceiling_violation_conjunction_whole_word():
    assert ceiling_violation("cats and dogs") is True
    assert ceiling_violation("android") is False  # 'and' as a substring is fine


# --- reuse (parent is None) ----------------------------------------------


def test_reuse_existing_type(catalog_by_uuid, by_norm):
    rec = _place("car", None, cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G1_REUSE"
    assert rec.assign_to == "u-car"
    assert rec.resolved_parent_uuid == "u-car"  # attach under the reused type
    assert rec.minted is False
    assert rec.member_ids == ("m1", "m2")
    assert rec.chain == ()  # no chain, ever


def test_reuse_multi_match_picks_tiebreak_winner(catalog_by_uuid, by_norm):
    # 'vehicle' has two uuids; the deterministic tiebreak favors more members.
    rec = _place("vehicle", None, cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G1_REUSE"
    assert rec.assign_to == "u-vehicle-A"  # member_count 9 > 3


def test_reuse_of_protected_root_coerces_to_mint(catalog_by_uuid, by_norm):
    rec = _place("entity", None, cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G3_ROOT"
    assert rec.minted is True
    assert "REUSE_ROOT_COERCE_MINT" in rec.events


def test_reuse_unresolved_name_mints_at_root(catalog_by_uuid, by_norm):
    rec = _place("nonexistent-kind", None, cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G3_ROOT"
    assert rec.minted is True
    assert "REUSE_UNRESOLVED_MINT_ROOT" in rec.events


# --- mint (parent is an existing type) -----------------------------------


def test_mint_grafts_under_nonroot_parent(catalog_by_uuid, by_norm):
    rec = _place("sedan", "car", cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G2_GRAFT"
    assert rec.assign_to == "NEW"
    assert rec.resolved_parent_uuid == "u-car"
    assert rec.minted is True
    assert rec.name == "sedan"
    assert rec.chain == ()


def test_mint_under_domain_root_is_g3_and_legal(catalog_by_uuid, by_norm):
    rec = _place("widget", "entity", cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G3_ROOT"
    assert rec.minted is True
    assert rec.floor_ok is True  # under a root is always legal


def test_mint_unresolved_parent_falls_back_to_root(catalog_by_uuid, by_norm):
    rec = _place("widget", "no-such-parent", cbu=catalog_by_uuid, bn=by_norm)
    assert rec.branch == "G3_ROOT"
    assert any(e.startswith("PARENT_UNRESOLVED") for e in rec.events)


# --- re-parent + residuals + reads ---------------------------------------


def test_reparent_records_removed_edge(catalog_by_uuid, by_norm):
    rec = _place(
        "sedan", "car", cbu=catalog_by_uuid, bn=by_norm,
        current_parent_uuid="u-entity",
    )
    # the kept subgraph drops its old IS_A (to entity) and gains one to car
    assert rec.removed_parent_uuid == "u-entity"
    assert rec.resolved_parent_uuid == "u-car"


def test_outliers_become_residual_and_drop_from_kept(catalog_by_uuid, by_norm):
    rec = _place(
        "car", None, members=["m1", "m2", "m3"], residual_ids=["m3"],
        cbu=catalog_by_uuid, bn=by_norm,
    )
    assert set(rec.member_ids) == {"m1", "m2"}
    assert rec.residual_ids == ("m3",)


def test_all_members_residual_yields_residual_branch(catalog_by_uuid, by_norm):
    rec = _place(
        "car", None, members=["m1", "m2"], residual_ids=["m1", "m2"],
        cbu=catalog_by_uuid, bn=by_norm,
    )
    assert rec.branch == "RESIDUAL"
    assert rec.member_ids == ()
    assert set(rec.residual_ids) == {"m1", "m2"}
    assert "ALL_MEMBERS_RESIDUAL" in rec.events


def test_covering_depth_is_a_catalog_read(catalog_by_uuid, by_norm):
    # depth = parent's hops to top + 1; car sits under vehicle under entity.
    root_rec = _place("widget", "entity", cbu=catalog_by_uuid, bn=by_norm)
    graft_rec = _place("sedan", "car", cbu=catalog_by_uuid, bn=by_norm)
    assert graft_rec.covering_depth > root_rec.covering_depth
