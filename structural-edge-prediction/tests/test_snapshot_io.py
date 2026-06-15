"""snapshot_io: validation catches bad snapshots; chain/members walks are right."""

from __future__ import annotations

import pytest

from harness.snapshot_io import (
    SnapshotValidationError,
    from_dict,
    members_of,
    node_type,
    type_chain,
)


def test_toy_fixture_loads_and_validates(toy_snapshot):
    assert len(toy_snapshot.nodes) == 17
    assert toy_snapshot.has_embeddings()
    assert node_type(toy_snapshot, "narwhal#1") == "whale"
    assert node_type(toy_snapshot, "blowhole#1") == "whale_anatomy"


def test_type_chain_walks_to_root(toy_snapshot):
    assert type_chain(toy_snapshot, "whale") == ["whale", "animal"]
    assert type_chain(toy_snapshot, "whale_anatomy") == ["whale_anatomy", "anatomy"]
    assert type_chain(toy_snapshot, "animal") == ["animal"]
    assert type_chain(toy_snapshot, None) == []


def test_members_of_pools_subtypes(toy_snapshot):
    whales = members_of(toy_snapshot, "whale")
    assert "narwhal#1" in whales and "salmon#1" not in whales
    # animal pools both whales and fish via the IS_A chain.
    animals = members_of(toy_snapshot, "animal", include_subtypes=True)
    assert "narwhal#1" in animals and "salmon#1" in animals
    # Immediate-only excludes subtypes (no node has immediate type animal).
    assert members_of(toy_snapshot, "animal", include_subtypes=False) == []
    # anatomy pools both whale_anatomy and fish_anatomy members.
    anat = members_of(toy_snapshot, "anatomy", include_subtypes=True)
    assert "blowhole#1" in anat and "gills#1" in anat


def test_validation_catches_dangling_edge():
    bad = {
        "nodes": [{"id": "a", "type": None, "label": "a"}],
        "type_parents": {},
        "edges": [{"src": "a", "rel": "has", "dst": "ghost"}],
    }
    with pytest.raises(SnapshotValidationError, match="unknown node"):
        from_dict(bad)


def test_validation_catches_type_cycle():
    bad = {
        "nodes": [],
        "type_parents": {"x": "y", "y": "x"},
        "edges": [],
    }
    with pytest.raises(SnapshotValidationError, match="cycle"):
        from_dict(bad)


def test_validation_catches_unknown_node_type():
    bad = {
        "nodes": [{"id": "a", "type": "nope", "label": "a"}],
        "type_parents": {"whale": None},
        "edges": [],
    }
    with pytest.raises(SnapshotValidationError, match="unknown type"):
        from_dict(bad)
