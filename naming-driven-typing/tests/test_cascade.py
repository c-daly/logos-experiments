"""Tests for harness.cascade -- simulate-only placement cascade (SPEC §5)."""

from __future__ import annotations

import dataclasses

import pytest

from harness.cascade import (
    EVICT_LEVEL_DROP,
    MAX_WORDS,
    MIN_DEPTH,
    PROTECTED_ROOT_NAMES,
    PlacementRecord,
    ceiling_violation,
    floor_ok,
    resolve_deepest_ancestor,
    simulate_cascade,
    simulate_group,
)


# ---- gates --------------------------------------------------------------

def test_floor_ok_uses_covering_depth():
    # covering_depth = len(chain)-1; MIN_DEPTH default 2
    assert floor_ok(["sedan", "car", "vehicle", "entity"]) is True  # depth 3
    assert floor_ok(["car", "vehicle", "entity"]) is True           # depth 2
    assert floor_ok(["thing", "entity"]) is False                   # depth 1
    assert floor_ok(["entity"]) is False                            # depth 0


def test_floor_min_depth_param_is_surfaced():
    assert MIN_DEPTH == 2
    # depth 1 passes only if caller lowers the floor to 1
    assert floor_ok(["thing", "entity"], min_depth=1) is True


def test_ceiling_violation_word_count():
    assert MAX_WORDS == 3
    assert ceiling_violation("car") is False
    assert ceiling_violation("sports utility vehicle") is False  # 3 words == MAX
    assert ceiling_violation("large sports utility vehicle") is True  # 4 words


def test_ceiling_violation_conjunction_whole_word():
    assert ceiling_violation("cars and boats") is True
    assert ceiling_violation("watercraft or aircraft") is True
    assert ceiling_violation("food/drink") is True
    assert ceiling_violation("feature-of car") is True
    # 'sandwich' contains 'and' as a substring but NOT as a whole word
    assert ceiling_violation("sandwich") is False
    assert ceiling_violation("brand") is False


# ---- deepest-ancestor resolver -----------------------------------------

def test_resolver_returns_deepest_published_ancestor(catalog_by_uuid, by_norm):
    # mint 'sedan'; chain[1:] = car, vehicle, entity -> deepest published = car
    parent_uuid, parent_name, events = resolve_deepest_ancestor(
        ["sedan", "car", "vehicle", "entity"], catalog_by_uuid, by_norm
    )
    assert parent_uuid == "u-car"
    assert parent_name == "car"


def test_resolver_multi_match_tiebreak_most_member_count(catalog_by_uuid, by_norm):
    # mint 'truck'; chain[1:] = vehicle, entity. 'vehicle' has two uuids.
    # tiebreak: most member_count -> u-vehicle-A (9 > 3).
    parent_uuid, parent_name, events = resolve_deepest_ancestor(
        ["truck", "vehicle", "entity"], catalog_by_uuid, by_norm
    )
    assert parent_uuid == "u-vehicle-A"
    assert parent_name == "vehicle"


def test_resolver_root_only_returns_root(catalog_by_uuid, by_norm):
    # 'gizmo' unpublished, only 'entity' (root) matches -> returns the root.
    parent_uuid, parent_name, events = resolve_deepest_ancestor(
        ["gizmo", "entity"], catalog_by_uuid, by_norm
    )
    assert parent_uuid == "u-entity"
    assert parent_name == "entity"


def test_resolver_no_root_defaults_entity_and_logs(catalog_by_uuid, by_norm):
    # malformed chain with no root token -> default entity + CHAIN_NO_ROOT.
    parent_uuid, parent_name, events = resolve_deepest_ancestor(
        ["gizmo", "gadget"], catalog_by_uuid, by_norm
    )
    assert parent_name == "entity"
    assert "CHAIN_NO_ROOT" in events


# ---- G1 reuse -----------------------------------------------------------

def test_g1_reuse_published_uuid(catalog_by_uuid, by_norm):
    group = {
        "assign_to": "u-car",
        "name": "car",
        "chain": ["car", "vehicle", "entity"],
        "member_ids": ["m1", "m2"],
    }
    rec = simulate_group(group, {"m1", "m2"}, catalog_by_uuid, by_norm)
    assert rec.branch == "G1_REUSE"
    assert rec.resolved_parent_uuid == "u-car"
    assert rec.residual_ids == ()


def test_g1_assign_to_protected_root_coerced_to_graft(catalog_by_uuid, by_norm):
    # assign_to resolves to a protected root -> NEVER reuse; coerce NEW + graft.
    group = {
        "assign_to": "u-entity",
        "name": "widget",
        "chain": ["widget", "vehicle", "entity"],
        "member_ids": ["m1"],
    }
    rec = simulate_group(group, {"m1"}, catalog_by_uuid, by_norm)
    assert rec.branch != "G1_REUSE"
    assert rec.branch == "G2_GRAFT"
    assert rec.resolved_parent_uuid == "u-vehicle-A"
    assert any("PROTECTED" in e for e in rec.events)


def test_g1_unresolvable_assign_to_falls_to_new(catalog_by_uuid, by_norm):
    group = {
        "assign_to": "u-does-not-exist",
        "name": "widget",
        "chain": ["widget", "car", "vehicle", "entity"],
        "member_ids": ["m1"],
    }
    rec = simulate_group(group, {"m1"}, catalog_by_uuid, by_norm)
    assert rec.branch == "G2_GRAFT"
    assert rec.resolved_parent_uuid == "u-car"
    assert any("UNRESOLVED_ASSIGN_TO" in e for e in rec.events)


# ---- G2 graft -----------------------------------------------------------

def test_g2_graft_under_deepest(catalog_by_uuid, by_norm):
    group = {
        "assign_to": "NEW",
        "name": "sedan",
        "chain": ["sedan", "car", "vehicle", "entity"],
        "member_ids": ["m1", "m2", "m3"],
    }
    rec = simulate_group(group, {"m1", "m2", "m3"}, catalog_by_uuid, by_norm)
    assert rec.branch == "G2_GRAFT"
    assert rec.resolved_parent_uuid == "u-car"
    assert rec.resolved_parent_name == "car"


# ---- G3 root fallback ---------------------------------------------------

def test_g3_new_branch_at_root(catalog_by_uuid, by_norm):
    # only the root matches -> mint under terminal root (entity).
    group = {
        "assign_to": "NEW",
        "name": "phenomenon",
        "chain": ["phenomenon", "abstraction", "entity"],
        "member_ids": ["m1", "m2"],
    }
    rec = simulate_group(group, {"m1", "m2"}, catalog_by_uuid, by_norm)
    assert rec.branch == "G3_ROOT"
    assert rec.resolved_parent_uuid == "u-entity"


def test_g3_bad_terminal_root_coerced_entity(catalog_by_uuid, by_norm):
    # terminal 'thing' not in whitelist -> map entity + log BAD_ROOT.
    group = {
        "assign_to": "NEW",
        "name": "gizmo",
        "chain": ["gizmo", "thing"],
        "member_ids": ["m1"],
    }
    # Depth-1 chain: lower the floor to reach the G3 terminal check (the
    # FLOOR gate runs first per SPEC §5.5 and would otherwise route this
    # shallow chain to RESIDUAL before the BAD_ROOT coercion).
    rec = simulate_group(group, {"m1"}, catalog_by_uuid, by_norm, min_depth=1)
    assert rec.branch == "G3_ROOT"
    assert rec.resolved_parent_name == "entity"
    assert any("BAD_ROOT" in e for e in rec.events)


# ---- floor gate -> residual --------------------------------------------

def test_floor_violation_routes_to_residual(catalog_by_uuid, by_norm):
    # covering_depth 1 < MIN_DEPTH 2 -> residual, no branch placement.
    group = {
        "assign_to": "NEW",
        "name": "shallow",
        "chain": ["shallow", "entity"],
        "member_ids": ["m1", "m2"],
    }
    rec = simulate_group(group, {"m1", "m2"}, catalog_by_uuid, by_norm)
    assert rec.branch == "RESIDUAL"
    assert rec.floor_ok is False
    assert set(rec.residual_ids) == {"m1", "m2"}
    assert rec.resolved_parent_uuid is None


# ---- ceiling gate -> residual (offline: split not re-issued) ------------

def test_ceiling_violation_routes_to_residual(catalog_by_uuid, by_norm):
    group = {
        "assign_to": "NEW",
        "name": "cars and boats",
        "chain": ["cars and boats", "vehicle", "entity"],
        "member_ids": ["m1", "m2"],
    }
    rec = simulate_group(group, {"m1", "m2"}, catalog_by_uuid, by_norm)
    assert rec.branch == "RESIDUAL"
    assert rec.ceiling_ok is False
    assert set(rec.residual_ids) == {"m1", "m2"}


# ---- freeze-snapshot guard (§5.12) -------------------------------------

def test_minted_this_pass_not_a_graft_target(catalog_by_uuid, by_norm):
    # Group A mints 'minivan'. Group B's chain[1:] would graft under 'minivan'
    # (unpublished but minted THIS pass) -> must NOT use it; fall through to
    # the deepest PUBLISHED ancestor (car).
    groups = [
        {
            "assign_to": "NEW",
            "name": "minivan",
            "chain": ["minivan", "car", "vehicle", "entity"],
            "member_ids": ["a1", "a2"],
        },
        {
            "assign_to": "NEW",
            "name": "people-mover",
            "chain": ["people-mover", "minivan", "car", "vehicle", "entity"],
            "member_ids": ["b1"],
        },
    ]
    records = simulate_cascade(groups, catalog_by_uuid, by_norm)
    b = [r for r in records if r.name == "people-mover"][0]
    assert b.resolved_parent_name != "minivan"
    assert b.resolved_parent_uuid == "u-car"


def test_gated_out_group_does_not_block_sibling_graft_target(
    catalog_by_uuid, by_norm
):
    # SPEC 5.12 regression: a FLOOR-gated group never mints, so its name must
    # NOT enter the minted-this-pass set and freeze-skip the PUBLISHED graft
    # target of a sibling. Group A (FLOOR violation -> RESIDUAL) is named
    # "car" -- also a published catalog node (u-car). Group B grafts under
    # "car"; gated-out A must leave the resolution of B unaffected.
    groups = [
        {
            "assign_to": "NEW",
            "name": "car",
            "chain": ["car", "entity"],  # covering_depth 1 < MIN_DEPTH
            "member_ids": ["a1"],
        },
        {
            "assign_to": "NEW",
            "name": "sedan",
            "chain": ["sedan", "car", "vehicle", "entity"],
            "member_ids": ["b1"],
        },
    ]
    records = simulate_cascade(groups, catalog_by_uuid, by_norm)
    a = next(r for r in records if r.name == "car")
    b = next(r for r in records if r.name == "sedan")
    assert a.branch == "RESIDUAL"
    assert b.branch == "G2_GRAFT"
    assert b.resolved_parent_uuid == "u-car"
    assert not any(e.startswith("FREEZE_SKIP:car") for e in b.events)


# ---- eviction proxy (§5.14) --------------------------------------------

def test_eviction_proxy_conservative_keeps_all(catalog_by_uuid, by_norm):
    # Offline has no per-member depth signal -> evict nothing (conservative).
    assert EVICT_LEVEL_DROP == 2
    group = {
        "assign_to": "NEW",
        "name": "sedan",
        "chain": ["sedan", "car", "vehicle", "entity"],
        "member_ids": ["m1", "m2", "m3"],
    }
    rec = simulate_group(group, {"m1", "m2", "m3"}, catalog_by_uuid, by_norm)
    assert rec.evicted_ids == ()


# ---- total partition ----------------------------------------------------

def test_cascade_total_partition_over_input_ids(catalog_by_uuid, by_norm):
    groups = [
        {
            "assign_to": "u-car",
            "name": "car",
            "chain": ["car", "vehicle", "entity"],
            "member_ids": ["m1", "m2"],
        },
        {
            "assign_to": "NEW",
            "name": "shallow",
            "chain": ["shallow", "entity"],
            "member_ids": ["m3"],
        },
    ]
    records = simulate_cascade(groups, catalog_by_uuid, by_norm)
    placed = {mid for r in records for mid in r.member_ids if r.branch != "RESIDUAL"}
    residual = {mid for r in records for mid in r.residual_ids}
    assert placed.isdisjoint(residual)
    assert placed | residual == {"m1", "m2", "m3"}


# ---- homonym guard via protected roots constant -------------------------

def test_protected_root_names_constant():
    assert "entity" in PROTECTED_ROOT_NAMES
    assert "concept" in PROTECTED_ROOT_NAMES
    assert "process" in PROTECTED_ROOT_NAMES
    assert "root" in PROTECTED_ROOT_NAMES
    assert "node" in PROTECTED_ROOT_NAMES
    assert "cognition" in PROTECTED_ROOT_NAMES


# ---- determinism proxy (§5.11 twice-run) --------------------------------

def test_determinism_proxy_twice_run_identical(catalog_by_uuid, by_norm):
    groups = [
        {
            "assign_to": "NEW",
            "name": "truck",
            "chain": ["truck", "vehicle", "entity"],
            "member_ids": ["m1"],
        },
        {
            "assign_to": "u-car",
            "name": "car",
            "chain": ["car", "vehicle", "entity"],
            "member_ids": ["m2", "m3"],
        },
        {
            "assign_to": "NEW",
            "name": "shallow",
            "chain": ["shallow", "entity"],
            "member_ids": ["m4"],
        },
    ]
    run1 = simulate_cascade(groups, catalog_by_uuid, by_norm)
    run2 = simulate_cascade(groups, catalog_by_uuid, by_norm)
    def key(recs):
        return [
            (r.cluster_id, r.branch, r.resolved_parent_uuid, tuple(r.residual_ids))
            for r in recs
        ]

    assert key(run1) == key(run2)


def test_placement_record_is_frozen(catalog_by_uuid, by_norm):
    group = {
        "assign_to": "u-car",
        "name": "car",
        "chain": ["car", "vehicle", "entity"],
        "member_ids": ["m1"],
    }
    rec = simulate_group(group, {"m1"}, catalog_by_uuid, by_norm)
    assert isinstance(rec, PlacementRecord)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.branch = "G3_ROOT"  # type: ignore[misc]
    # deep immutability: collection fields are tuples, not lists
    assert isinstance(rec.chain, tuple)
    assert isinstance(rec.member_ids, tuple)
    assert isinstance(rec.evicted_ids, tuple)
    assert isinstance(rec.residual_ids, tuple)
    assert isinstance(rec.events, tuple)
# ---- gate toggles (A5/no-gate ablation seam) -----------------------------

def test_ceiling_toggle_off_mints_instead_of_residual(catalog_by_uuid, by_norm):
    # Ceiling-violating name (whole-word conjunction) over a floor-passing
    # chain: gated under defaults, minted when enforce_ceiling=False.
    group = {
        "cluster_id": "cg-ceil",
        "assign_to": "NEW",
        "name": "gadget and gizmo",
        "chain": ["gadget and gizmo", "car", "vehicle", "entity"],
        "member_ids": ["m1"],
    }
    gated = simulate_cascade([group], catalog_by_uuid, by_norm)[0]
    assert gated.branch == "RESIDUAL"
    assert "CEILING_VIOLATION_SPLIT_DEFERRED" in gated.events

    ungated = simulate_cascade(
        [group], catalog_by_uuid, by_norm, enforce_ceiling=False
    )[0]
    assert ungated.branch == "G2_GRAFT"  # chain resolves through car
    assert ungated.ceiling_ok is True  # vacuous: toggle off
    assert not any("VIOLATION" in e for e in ungated.events)


def test_floor_disabled_via_min_depth_zero(catalog_by_uuid, by_norm):
    # covering_depth >= 0 always holds, so min_depth=0 IS the floor-off seam.
    group = {
        "cluster_id": "cg-floor",
        "assign_to": "NEW",
        "name": "widget",
        "chain": ["widget", "entity"],
        "member_ids": ["m1"],
    }
    gated = simulate_cascade([group], catalog_by_uuid, by_norm)[0]
    assert gated.branch == "RESIDUAL"
    assert "FLOOR_VIOLATION" in gated.events

    ungated = simulate_cascade([group], catalog_by_uuid, by_norm, min_depth=0)[0]
    assert ungated.branch == "G3_ROOT"
    assert ungated.floor_ok is True
    assert not any("VIOLATION" in e for e in ungated.events)


def test_minted_set_respects_ceiling_toggle(catalog_by_uuid, by_norm):
    # With the ceiling ON the gated group never mints, so a sibling chain
    # naming it is NOT freeze-skipped; with the ceiling OFF it mints and the
    # freeze-snapshot guard (SPEC 5.12) kicks in for the sibling.
    gated_group = {
        "cluster_id": "cg-a",
        "assign_to": "NEW",
        "name": "gadget and gizmo",
        "chain": ["gadget and gizmo", "car", "vehicle", "entity"],
        "member_ids": ["m1"],
    }
    sibling = {
        "cluster_id": "cg-b",
        "assign_to": "NEW",
        "name": "widget",
        "chain": ["widget", "gadget and gizmo", "entity"],
        "member_ids": ["m2"],
    }
    on = simulate_cascade([gated_group, sibling], catalog_by_uuid, by_norm)
    assert on[0].branch == "RESIDUAL"
    assert not any(e.startswith("FREEZE_SKIP") for e in on[1].events)

    off = simulate_cascade(
        [gated_group, sibling], catalog_by_uuid, by_norm, enforce_ceiling=False
    )
    assert off[0].branch == "G2_GRAFT"
    assert any(e.startswith("FREEZE_SKIP") for e in off[1].events)
    assert off[1].branch == "G3_ROOT"  # minted name is not a graft target
