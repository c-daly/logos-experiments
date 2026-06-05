"""Offline unit tests for the naming-driven-typing fixtures layer.

Label-free (2026-06-05 override): NO labels.json, and the validators must
reject any label/labels key. No live stack, no network — pure functions only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.fixtures_io import (
    FIXTURE_VERSION,
    CatalogFixtureError,
    ClusterFixtureError,
    freeze_catalog,
    freeze_clusters,
    load_catalog,
    load_clusters,
    validate_catalog,
    validate_clusters,
)
from harness.reseed import (
    ReseedInputError,
    clusters_from_node_members,
    validate_corpus_items,
)


def test_clusters_from_node_members_maps_uuid_to_id() -> None:
    node_clusters = [
        {
            "label": 0,
            "current_name": "entity",
            "members": [
                {"uuid": "u1", "name": "cheetah"},
                {"uuid": "u2", "name": "narwhal"},
            ],
        },
        {
            "label": 1,
            "current_name": "entity",
            "members": [{"uuid": "u3", "name": "sedan"}],
        },
    ]
    out = clusters_from_node_members(node_clusters)
    assert [c["cluster_id"] for c in out] == ["0", "1"]
    assert out[0]["current_name"] == "entity"
    # uuid -> id, name preserved, nothing else leaks
    assert out[0]["members"] == [
        {"id": "u1", "name": "cheetah"},
        {"id": "u2", "name": "narwhal"},
    ]
    # whole cluster sent => coverage 1.0 (SPEC §6.1: no down-sampling)
    assert out[0]["sample_coverage"] == 1.0
    assert out[1]["sample_coverage"] == 1.0
    # mapper never emits a label/labels field (label-free)
    assert "label" not in out[0]
    assert "labels" not in out[0]


def test_clusters_from_node_members_rejects_missing_label() -> None:
    # live emergence output is an external input boundary: no coercion
    bad = [{"members": [{"uuid": "u1", "name": "cheetah"}]}]
    with pytest.raises(
        ReseedInputError, match=r"node_clusters\[0\] is missing required key: label"
    ):
        clusters_from_node_members(bad)


def test_clusters_from_node_members_rejects_member_missing_uuid() -> None:
    bad = [{"label": 0, "members": [{"name": "cheetah"}]}]
    with pytest.raises(
        ReseedInputError,
        match=r"node_clusters\[0\].members\[0\] is missing required key: uuid",
    ):
        clusters_from_node_members(bad)


def test_clusters_from_node_members_rejects_member_missing_name() -> None:
    bad = [{"label": 0, "members": [{"uuid": "u1"}]}]
    with pytest.raises(
        ReseedInputError,
        match=r"node_clusters\[0\].members\[0\] is missing required key: name",
    ):
        clusters_from_node_members(bad)


# --- cluster validation ---------------------------------------------------

def _good_clusters() -> list[dict[str, object]]:
    return [
        {
            "cluster_id": "0",
            "current_name": "entity",
            "members": [
                {"id": "u1", "name": "cheetah"},
                {"id": "u2", "name": "narwhal"},
            ],
            "sample_coverage": 1.0,
        },
        {
            "cluster_id": "1",
            "current_name": "entity",
            "members": [{"id": "u3", "name": "sedan"}],
            "sample_coverage": 1.0,
        },
    ]


def test_validate_clusters_accepts_good() -> None:
    validate_clusters(_good_clusters())  # no raise


def test_validate_clusters_rejects_label_key() -> None:
    bad = _good_clusters()
    bad[0]["label"] = "mammal"  # label-free: forbidden
    with pytest.raises(ClusterFixtureError, match="label"):
        validate_clusters(bad)


def test_validate_clusters_rejects_duplicate_member_id() -> None:
    bad = _good_clusters()
    bad[1]["members"] = [{"id": "u1", "name": "dup"}]  # u1 already in cluster 0
    with pytest.raises(ClusterFixtureError, match="unique"):
        validate_clusters(bad)


def test_validate_clusters_rejects_coverage_out_of_range() -> None:
    bad = _good_clusters()
    bad[0]["sample_coverage"] = 0.0
    with pytest.raises(ClusterFixtureError, match="sample_coverage"):
        validate_clusters(bad)


def test_validate_clusters_rejects_missing_current_name() -> None:
    bad = _good_clusters()
    del bad[0]["current_name"]  # required by the documented cluster schema
    with pytest.raises(ClusterFixtureError, match="current_name"):
        validate_clusters(bad)


def test_clusters_from_node_members_rejects_missing_members() -> None:
    node_clusters = [{"label": "mammal"}]  # no members key at all
    with pytest.raises(
        ReseedInputError, match=r"node_clusters\[0\] is missing required non-empty"
    ):
        clusters_from_node_members(node_clusters)


def test_validate_clusters_rejects_non_dict_cluster() -> None:
    bad = _good_clusters()
    bad[1] = "not-a-cluster"  # element is not an object
    with pytest.raises(ClusterFixtureError, match=r"clusters\[1\] is not an object"):
        validate_clusters(bad)


def test_validate_clusters_rejects_non_dict_member() -> None:
    bad = _good_clusters()
    bad[1]["members"] = ["u3"]  # member element is not an object
    with pytest.raises(ClusterFixtureError, match=r"members\[0\] is not an object"):
        validate_clusters(bad)


def test_clusters_freeze_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    freeze_clusters(_good_clusters(), path)
    loaded = load_clusters(path)
    assert loaded == _good_clusters()
    # on-disk has the version envelope and is deterministically sorted
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == FIXTURE_VERSION
    assert [c["cluster_id"] for c in raw["clusters"]] == ["0", "1"]


def test_clusters_freeze_is_byte_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    # input order shuffled — output must be identical bytes
    freeze_clusters(_good_clusters(), a)
    freeze_clusters(list(reversed(_good_clusters())), b)
    assert a.read_bytes() == b.read_bytes()


def test_clusters_freeze_member_order_is_byte_deterministic(tmp_path: Path) -> None:
    # member ELEMENT order shuffled -- sort_keys=True alone would NOT catch
    # this (it sorts dict keys only); the writer must sort members by id.
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    freeze_clusters(_good_clusters(), a)
    shuffled = _good_clusters()
    for c in shuffled:
        c["members"] = list(reversed(c["members"]))
    freeze_clusters(shuffled, b)
    assert a.read_bytes() == b.read_bytes()


def test_load_clusters_rejects_non_object_json(tmp_path: Path) -> None:
    # a list/scalar top level must raise the module error, not AttributeError
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")
    with pytest.raises(ClusterFixtureError, match="must be a JSON object"):
        load_clusters(path)


def test_load_clusters_rejects_scalar_json(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(42) + "\n", encoding="utf-8")
    with pytest.raises(ClusterFixtureError, match="must be a JSON object"):
        load_clusters(path)


# --- catalog validation ---------------------------------------------------

def _good_catalog() -> dict[str, object]:
    return {
        "catalog_by_uuid": {
            "r-ent": {"uuid": "r-ent", "name": "entity", "norm_name": "entity",
                      "member_count": 0, "ancestors": [], "chain": ["entity"],
                      "depth": 1, "parent_uuid": None, "is_root": True},
            "r-con": {"uuid": "r-con", "name": "concept", "norm_name": "concept",
                      "member_count": 0, "ancestors": [], "chain": ["concept"],
                      "depth": 1, "parent_uuid": None, "is_root": True},
            "r-pro": {"uuid": "r-pro", "name": "process", "norm_name": "process",
                      "member_count": 0, "ancestors": [], "chain": ["process"],
                      "depth": 1, "parent_uuid": None, "is_root": True},
            "t-veh1": {"uuid": "t-veh1", "name": "vehicle", "norm_name": "vehicle",
                       "member_count": 3, "ancestors": ["entity"],
                       "chain": ["entity", "vehicle"], "depth": 2,
                       "parent_uuid": "r-ent", "is_root": False},
            "t-veh2": {"uuid": "t-veh2", "name": "Vehicles", "norm_name": "vehicle",
                       "member_count": 2, "ancestors": ["entity"],
                       "chain": ["entity", "vehicle"], "depth": 2,
                       "parent_uuid": "r-ent", "is_root": False},
        },
        "by_norm": {
            "entity": ["r-ent"], "concept": ["r-con"], "process": ["r-pro"],
            "vehicle": ["t-veh1", "t-veh2"],
        },
        "roots_present_in_live_catalog": True,
    }


def test_validate_catalog_accepts_good() -> None:
    validate_catalog(_good_catalog())  # no raise


def test_validate_catalog_rejects_missing_root() -> None:
    bad = _good_catalog()
    del bad["catalog_by_uuid"]["r-pro"]
    bad["by_norm"].pop("process")
    with pytest.raises(CatalogFixtureError, match="process"):
        validate_catalog(bad)


def test_validate_catalog_rejects_scalar_by_norm() -> None:
    bad = _good_catalog()
    bad["by_norm"]["vehicle"] = "t-veh1"  # must be a LIST
    with pytest.raises(CatalogFixtureError, match="list"):
        validate_catalog(bad)


def test_validate_catalog_rejects_non_dict_catalog() -> None:
    with pytest.raises(CatalogFixtureError, match="catalog must be an object"):
        validate_catalog(["not", "a", "catalog"])


def test_validate_catalog_rejects_non_dict_record() -> None:
    bad = _good_catalog()
    bad["catalog_by_uuid"]["t-veh1"] = "not-a-record"
    with pytest.raises(CatalogFixtureError, match=r"catalog_by_uuid\[.t-veh1.\] must be an object"):
        validate_catalog(bad)


def test_validate_catalog_rejects_non_str_uuid_in_by_norm() -> None:
    bad = _good_catalog()
    bad["by_norm"]["vehicle"] = ["t-veh1", 42]
    with pytest.raises(CatalogFixtureError, match=r"by_norm\[.vehicle.\]\[1\] must be a str uuid"):
        validate_catalog(bad)


def test_validate_catalog_rejects_by_norm_uuid_absent_from_catalog() -> None:
    bad = _good_catalog()
    bad["by_norm"]["vehicle"] = ["t-veh1", "t-ghost"]  # t-ghost has no record
    with pytest.raises(
        CatalogFixtureError,
        match=r"by_norm\[.vehicle.\] references uuid .t-ghost. absent from catalog_by_uuid",
    ):
        validate_catalog(bad)


def test_catalog_freeze_load_asserts_vehicle_fragments(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    freeze_catalog(_good_catalog(), path)
    loaded = load_catalog(path)
    # load_catalog enforces the fragments invariant (SPEC §4.2)
    assert len(loaded["by_norm"]["vehicle"]) > 1


def test_catalog_freeze_by_norm_order_is_byte_deterministic(tmp_path: Path) -> None:
    # by_norm uuid-list ELEMENT order shuffled -- sort_keys=True alone would
    # NOT catch this (it sorts dict keys only); the writer must sort the lists.
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    freeze_catalog(_good_catalog(), a)
    shuffled = _good_catalog()
    for norm, uuids in shuffled["by_norm"].items():
        shuffled["by_norm"][norm] = list(reversed(uuids))
    freeze_catalog(shuffled, b)
    assert a.read_bytes() == b.read_bytes()


def test_load_catalog_raises_when_vehicle_not_fragmented(tmp_path: Path) -> None:
    cat = _good_catalog()
    del cat["catalog_by_uuid"]["t-veh2"]
    cat["by_norm"]["vehicle"] = ["t-veh1"]  # only one => invariant broken
    path = tmp_path / "catalog.json"
    freeze_catalog(cat, path)
    with pytest.raises(CatalogFixtureError, match="vehicle"):
        load_catalog(path)


def test_load_catalog_rejects_non_object_json(tmp_path: Path) -> None:
    # a list/scalar top level must raise the module error, not AttributeError
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(["not", "a", "catalog"]) + "\n", encoding="utf-8")
    with pytest.raises(CatalogFixtureError, match="must be a JSON object"):
        load_catalog(path)


def test_load_catalog_rejects_scalar_json(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps("nope") + "\n", encoding="utf-8")
    with pytest.raises(CatalogFixtureError, match="must be a JSON object"):
        load_catalog(path)


import os

EXP_DIR = Path(__file__).resolve().parents[1]


def test_corpus_jsonl_is_wellformed() -> None:
    corpus = EXP_DIR / "corpus" / "corpus.jsonl"
    lines = [ln for ln in corpus.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "corpus must be non-empty"
    domains = set()
    for ln in lines:
        rec = json.loads(ln)
        assert set(rec) == {"round", "domain", "text"}
        assert isinstance(rec["round"], int) and rec["round"] >= 1
        assert rec["domain"] and rec["text"]
        domains.add(rec["domain"])
    # the vehicles domain must exist so the by_norm['vehicle']>1 case is seedable
    assert "vehicles" in domains


def test_validate_corpus_items_accepts_good() -> None:
    validate_corpus_items(
        [{"round": 1, "domain": "animals", "text": "a cheetah sprints"}]
    )  # no raise


def test_validate_corpus_items_rejects_missing_text() -> None:
    with pytest.raises(
        ReseedInputError, match=r"corpus\[0\] is missing required key: text"
    ):
        validate_corpus_items([{"round": 1, "domain": "animals"}])


def test_validate_corpus_items_rejects_missing_domain() -> None:
    with pytest.raises(
        ReseedInputError, match=r"corpus\[1\] is missing required key: domain"
    ):
        validate_corpus_items(
            [{"domain": "animals", "text": "ok"}, {"text": "no domain"}]
        )


def test_validate_corpus_items_rejects_non_object_item() -> None:
    with pytest.raises(ReseedInputError, match=r"corpus\[0\] is not an object"):
        validate_corpus_items(["just a string"])


@pytest.mark.skipif(
    os.environ.get("RESEED_LIVE") != "1",
    reason="live reseed needs Neo4j+Milvus+Hermes; set RESEED_LIVE=1 to run",
)
def test_reseed_and_build_smoke() -> None:
    # Imported lazily: logos_hcg / pymilvus only needed on the live path.
    from logos_config.settings import MilvusConfig, Neo4jConfig
    from logos_hcg.client import HCGClient
    from logos_hcg.sync import HCGMilvusSync

    from harness.reseed import reseed_and_build

    cfg = Neo4jConfig(password=os.environ.get("NEO4J_PASSWORD", "logosdev"))
    client = HCGClient(uri=cfg.uri, user=cfg.user, password=cfg.password)
    sync = HCGMilvusSync(client, MilvusConfig())
    out = reseed_and_build(
        client,
        sync,
        corpus_path=EXP_DIR / "corpus" / "corpus.jsonl",
        hermes_url=os.environ.get("HERMES_URL", "http://localhost:17000"),
    )
    assert out["clusters"], "reseed produced no clusters"
    validate_clusters(out["clusters"])
    validate_catalog(out["catalog"])
    assert len(out["catalog"]["by_norm"].get("vehicle", [])) > 1
