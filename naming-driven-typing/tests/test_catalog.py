"""Unit tests for the enriched catalog builder (SPEC \u00a74.2\u20134.4, corrected).

Fully offline: no Neo4j, no Milvus, no hermes. ``build_catalog`` takes raw
type-def dicts (the shape ``HCGClient.get_all_type_definitions`` returns),
reified IS_A edge rows, and an injected member-counter callable, so the
partition/normalization/roots logic is tested without the live stack.
``build_catalog_from_client`` is exercised against a hand-built read-only
client fake.
"""

from __future__ import annotations

import pytest

from harness.catalog import (
    REALM_ROOTS,
    NonCanonicalNameError,
    RootMissingError,
    build_catalog,
    build_catalog_from_client,
)


def _fake_canonicalize(name: str) -> str:
    """Deterministic stand-in for hermes.canonical.canonicalize.

    Lowercase, strip, and collapse the one plural the fixtures exercise
    (vehicles -> vehicle). Real canonicalize (T1) is exercised in its own
    golden-table CI test; here we only need a stable, injectable function.
    """
    n = " ".join(name.strip().lower().split())
    return "vehicle" if n == "vehicles" else n


# Two same-name "vehicle" twins (distinct uuids \u2014 mint_type mints distinct
# type_<slug>_<hex8> for same-label clusters) + a deeper type + the three
# realm roots + the chain head "node". Names are already canonical
# (canonical-at-the-boundary, 2026-06-05 decisions); roots carry NO
# is_type_definition flag, exactly as the seeder leaves them (SPEC \u00a74.3).
# The stale "ancestors" prop on sedan must be IGNORED (SPEC \u00a74.4 corrected:
# parent_uuid comes from the reified IS_A edge walk, not stored ancestors).
TYPE_DEFS = [
    {"uuid": "type_node", "name": "node", "properties": {}},
    {"uuid": "type_entity", "name": "entity", "properties": {}},
    {"uuid": "type_concept", "name": "concept", "properties": {}},
    {"uuid": "type_process", "name": "process", "properties": {}},
    {
        "uuid": "type_vehicle_aa11",
        "name": "vehicle",
        "properties": {"is_type_definition": True},
    },
    {
        "uuid": "type_vehicle_bb22",  # twin: different uuid, same name
        "name": "vehicle",
        "properties": {"is_type_definition": True},
    },
    {
        "uuid": "type_sedan_cc33",
        "name": "sedan",
        "properties": {
            "ancestors": ["root", "node", "entity", "stale_garbage"],
            "is_type_definition": True,
        },
    },
]

# Reified IS_A edge rows as the live edge pull returns them
# ((t)<-[:FROM]-(e {relation:\u0027IS_A\u0027})-[:TO]->(parent), flattened to
# source -> target). The instance->type IS_A edge and the non-IS_A edge
# must both be ignored.
IS_A_EDGES = [
    {"source": "type_entity", "target": "type_node", "relation": "IS_A"},
    {"source": "type_concept", "target": "type_node", "relation": "IS_A"},
    {"source": "type_process", "target": "type_node", "relation": "IS_A"},
    {"source": "type_vehicle_aa11", "target": "type_entity", "relation": "IS_A"},
    {"source": "type_vehicle_bb22", "target": "type_entity", "relation": "IS_A"},
    {"source": "type_sedan_cc33", "target": "type_vehicle_aa11", "relation": "IS_A"},
    # instance membership edge \u2014 source is not a catalog uuid: ignored.
    {"source": "ent_car_1", "target": "type_vehicle_aa11", "relation": "IS_A"},
    # non-IS_A relation between catalog uuids: ignored.
    {"source": "type_sedan_cc33", "target": "type_entity", "relation": "PART_OF"},
]

MEMBER_COUNTS = {
    "type_vehicle_aa11": 5,
    "type_vehicle_bb22": 2,
    "type_sedan_cc33": 3,
    "type_entity": 40,
    "type_concept": 0,
    "type_process": 0,
    "type_node": 0,
}


def _counter(uuid: str) -> int:
    return MEMBER_COUNTS.get(uuid, 0)


@pytest.fixture()
def result():
    return build_catalog(
        TYPE_DEFS, _counter, IS_A_EDGES, canonicalize=_fake_canonicalize
    )


def test_catalog_keyed_by_uuid_not_name(result):
    # Both vehicle twins survive because the catalog is uuid-keyed, NOT
    # name-keyed (production blocker fixed, SPEC \u00a74.1/\u00a74.2).
    assert "type_vehicle_aa11" in result.catalog_by_uuid
    assert "type_vehicle_bb22" in result.catalog_by_uuid
    assert len(result.catalog_by_uuid) == len(TYPE_DEFS)


def test_entry_schema_fields(result):
    e = result.catalog_by_uuid["type_sedan_cc33"]
    assert e["uuid"] == "type_sedan_cc33"
    assert e["name"] == "sedan"
    assert e["norm_name"] == "sedan"
    assert e["member_count"] == 3
    # immediate IS_A parent uuid from the reified edge walk (\u00a74.4 corrected).
    assert e["parent_uuid"] == "type_vehicle_aa11"
    assert e["is_root"] is False


def test_no_stored_ancestors_chain_depth_fields(result):
    # 2026-06-05 correction: ancestors/chain/depth are NOT stored; chains
    # are reconstructed by following parent_uuid upward. The stale
    # "ancestors" node prop is ignored entirely.
    expected_keys = {"uuid", "name", "norm_name", "member_count", "parent_uuid", "is_root"}
    for entry in result.catalog_by_uuid.values():
        assert set(entry) == expected_keys


def test_parent_uuid_from_edge_walk_not_ancestors_prop(result):
    # sedan\u0027s stale ancestors prop says "stale_garbage"; the edge walk says
    # type_vehicle_aa11. Edge walk wins; instance and non-IS_A edges ignored.
    assert result.catalog_by_uuid["type_sedan_cc33"]["parent_uuid"] == "type_vehicle_aa11"
    assert result.catalog_by_uuid["type_vehicle_bb22"]["parent_uuid"] == "type_entity"
    # chain heads top out: node has no IS_A edge upward in the fixture.
    assert result.catalog_by_uuid["type_node"]["parent_uuid"] is None


def test_by_norm_is_multivalued_for_twins(result):
    # The known fragments case: by_norm[\u0027vehicle\u0027] MUST have >1 uuid, else the
    # disambiguation criterion (SPEC \u00a74.2) is silently un-exercised.
    assert len(result.by_norm["vehicle"]) > 1
    assert set(result.by_norm["vehicle"]) == {"type_vehicle_aa11", "type_vehicle_bb22"}
    # by_norm values are deterministically ordered (smallest uuid first).
    assert result.by_norm["vehicle"] == ["type_vehicle_aa11", "type_vehicle_bb22"]
    assert result.by_norm["sedan"] == ["type_sedan_cc33"]


def test_member_count_from_counter_not_props(result):
    # props value is uniformly 0; member_count comes from the injected pull.
    assert result.catalog_by_uuid["type_vehicle_aa11"]["member_count"] == 5
    assert result.catalog_by_uuid["type_entity"]["member_count"] == 40


def test_roots_are_root_by_name_membership_regardless_of_flags(result):
    # entity/concept/process are is_root purely by name, even though they
    # carry no is_type_definition flag (SPEC \u00a74.3). Their own IS_A parent is
    # the chain head node (\u00a74.2 corrected); node itself is not a realm root.
    for ruuid in ("type_entity", "type_concept", "type_process"):
        e = result.catalog_by_uuid[ruuid]
        assert e["is_root"] is True
        assert e["parent_uuid"] == "type_node"
    assert result.catalog_by_uuid["type_node"]["is_root"] is False
    assert REALM_ROOTS == frozenset({"entity", "concept", "process"})


def test_roots_present_flag_and_root_uuid_index(result):
    assert result.roots_present_in_live_catalog is True
    assert result.root_uuids == {
        "concept": "type_concept",
        "entity": "type_entity",
        "process": "type_process",
    }


def test_missing_root_fails_loudly():
    # Drop the process root -> builder must raise, not silently patch.
    defs_no_process = [d for d in TYPE_DEFS if d["name"] != "process"]
    with pytest.raises(RootMissingError) as exc:
        build_catalog(
            defs_no_process, _counter, IS_A_EDGES, canonicalize=_fake_canonicalize
        )
    assert "process" in str(exc.value)


def test_non_canonical_name_fails_loudly():
    # A stored plural violates canonical-at-the-boundary: the builder must
    # raise (never silently repair) and emit non_canonical_names_in_catalog
    # (2026-06-05 decisions, R6-style loud failure).
    defs_with_plural = TYPE_DEFS + [
        {"uuid": "type_vehicles_dd44", "name": "vehicles", "properties": {}}
    ]
    with pytest.raises(NonCanonicalNameError) as exc:
        build_catalog(
            defs_with_plural, _counter, IS_A_EDGES, canonicalize=_fake_canonicalize
        )
    msg = str(exc.value)
    assert "non_canonical_names_in_catalog" in msg
    assert "vehicles" in msg


def test_build_is_deterministic_twice_run():
    r1 = build_catalog(TYPE_DEFS, _counter, IS_A_EDGES, canonicalize=_fake_canonicalize)
    r2 = build_catalog(TYPE_DEFS, _counter, IS_A_EDGES, canonicalize=_fake_canonicalize)
    assert r1.catalog_by_uuid == r2.catalog_by_uuid
    assert r1.by_norm == r2.by_norm


class _FakeClient:
    """Hand-built read-only HCGClient fake (no Neo4j, no network)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._nodes_by_type_uuid = {
            "type_vehicle_aa11": [{"uuid": f"ent_{i}"} for i in range(5)],
            # one malformed row without uuid \u2014 must not be counted.
            "type_vehicle_bb22": [{"uuid": "ent_a"}, {"uuid": "ent_b"}, {"name": "junk"}],
            "type_sedan_cc33": [{"uuid": f"ent_s{i}"} for i in range(3)],
        }

    def get_all_type_definitions(self):
        self.calls.append("get_all_type_definitions")
        return [dict(d) for d in TYPE_DEFS]

    def list_all_edges(self, relation_type=None):
        self.calls.append(f"list_all_edges:{relation_type}")
        return [
            dict(e) for e in IS_A_EDGES if relation_type in (None, e["relation"])
        ]

    def get_nodes_by_type_uuid(self, type_uuid):
        self.calls.append(f"get_nodes_by_type_uuid:{type_uuid}")
        return self._nodes_by_type_uuid.get(type_uuid, [])

    def list_all_nodes(self, node_type=None):
        self.calls.append(f"list_all_nodes:{node_type}")
        assert node_type == "entity"
        return [{"uuid": f"ent_{i}"} for i in range(40)]


def test_build_catalog_from_client_wires_reads_only():
    client = _FakeClient()
    result = build_catalog_from_client(client, canonicalize=_fake_canonicalize)
    # member_count: base entity junk-drawer via node-type scan; minted types
    # via the type_uuid pull; rows without uuid are not counted.
    assert result.catalog_by_uuid["type_entity"]["member_count"] == 40
    assert result.catalog_by_uuid["type_vehicle_aa11"]["member_count"] == 5
    assert result.catalog_by_uuid["type_vehicle_bb22"]["member_count"] == 2
    # parent_uuid wired from the IS_A edge pull.
    assert result.catalog_by_uuid["type_sedan_cc33"]["parent_uuid"] == "type_vehicle_aa11"
    # known-fragments invariant on the client path (SPEC \u00a74.2).
    assert len(result.by_norm["vehicle"]) > 1
    # zero graph writes: only read methods were ever called.
    read_prefixes = (
        "get_all_type_definitions",
        "list_all_edges",
        "get_nodes_by_type_uuid",
        "list_all_nodes",
    )
    assert all(c.startswith(read_prefixes) for c in client.calls)
