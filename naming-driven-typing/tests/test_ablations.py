"""Ablation-arm tests -- 2026-06-06 contract.

The arms re-scoped onto {name, parent, outliers}: A4 no_chain is RETIRED (no
LLM chain to ablate). The rest are parent-choice / catalog-view coercions:
  naive_llm -- no catalog, name+root only (-> all roots)
  no_reuse  -- a would-be reuse (parent None) is coerced to mint at the root
  no_graft  -- parent forced to a root + roots-only view (-> all G3_ROOT)
  no_gate   -- ceiling off (floor is always-legal under the re-parent model)

Logic tests run offline. Tests that validate the FROZEN llm_responses_*.json
files are xfail-pending the new-contract re-freeze (the committed fixtures are
the old {groups, chain} shape).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import ablations as ab
from harness import run_experiment as rx

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
REPEATS = 5
ARMS = ("naive_llm", "no_reuse", "no_graft", "no_gate")  # A4 no_chain retired
ROOT_UUIDS = {"u-entity", "u-concept", "u-process"}


def _members(*pairs):
    return [{"id": i, "name": n} for i, n in pairs]


# ---- arm registry -------------------------------------------------------


def test_responses_filename_mapping():
    assert ab.responses_filename("full") == "llm_responses.json"
    assert ab.responses_filename("naive_llm") == "llm_responses_naive_llm.json"
    assert ab.responses_filename("no_graft") == "llm_responses_no_graft.json"
    # no_reuse / no_gate share the full fixture (prompt-identical)
    assert ab.responses_filename("no_reuse") == "llm_responses.json"
    assert ab.responses_filename("no_gate") == "llm_responses.json"


def test_no_chain_is_retired():
    with pytest.raises(Exception):
        ab.responses_filename("no_chain")


def test_unknown_arm_fails_closed():
    with pytest.raises(Exception):
        ab.simulate_arm_cascade({}, {}, cluster_id="c", member_ids=[], ablation="bogus")


def test_simulate_arm_cascade_rejects_full():
    with pytest.raises(ValueError):
        ab.simulate_arm_cascade({}, {}, cluster_id="c", member_ids=[], ablation="full")


def test_arm_seam_factories_default_to_full_wiring():
    for arm in ("full", "no_reuse", "no_gate"):
        assert ab.arm_registry_factory(arm) is None
    for arm in ("full", "no_reuse", "no_graft", "no_gate"):
        assert ab.arm_client_factory(arm, replayer=None) is None


def test_arm_message_transform_is_identity_now():
    # No arm rewrites the prompt under the new contract.
    for arm in ("full",) + ARMS:
        assert ab.arm_message_transform(arm) is None


# ---- simulate_arm_cascade unit semantics (conftest catalog) -------------


def test_no_reuse_coerces_would_be_reuse_to_root(catalog_by_uuid, by_norm):
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    # parent None + name matches an existing type would REUSE in the full arm;
    # no_reuse forbids that -> mint at the entity root.
    out = ab.simulate_arm_cascade(
        {"name": "car", "parent": None, "residual_ids": []},
        catalog, cluster_id="c", member_ids=["m1"], ablation="no_reuse",
    )
    br = out["branches"][0]
    assert br["branch"] == "G3_ROOT"
    assert br["assign_to"] == "NEW"  # never reuse


def test_no_reuse_leaves_real_grafts_alone(catalog_by_uuid, by_norm):
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    out = ab.simulate_arm_cascade(
        {"name": "sedan", "parent": "car", "residual_ids": []},
        catalog, cluster_id="c", member_ids=["m1"], ablation="no_reuse",
    )
    assert out["branches"][0]["branch"] == "G2_GRAFT"  # graft untouched


def test_no_graft_forces_root(catalog_by_uuid, by_norm):
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    out = ab.simulate_arm_cascade(
        {"name": "sedan", "parent": "car", "residual_ids": []},
        catalog, cluster_id="c", member_ids=["m1"], ablation="no_graft",
    )
    assert out["branches"][0]["branch"] == "G3_ROOT"  # no grafting under non-root


def test_naive_llm_lands_at_root(catalog_by_uuid, by_norm):
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    out = ab.simulate_arm_cascade(
        {"name": "newkind", "parent": "entity", "residual_ids": []},
        catalog, cluster_id="c", member_ids=["m1"], ablation="naive_llm",
    )
    assert out["branches"][0]["branch"] == "G3_ROOT"


def test_roots_only_catalog_is_a_view_of_the_same_fixture():
    from harness.fixtures_io import load_catalog

    catalog = load_catalog(FIXTURES_DIR / "catalog.json")
    view = ab.roots_only_catalog(catalog)
    assert set(view["by_norm"]) == {"entity", "concept", "process"}
    assert "t-veh1" in catalog["catalog_by_uuid"]  # source untouched


# ---- NaiveLLMClient (A1 prompt path), new {name, parent} body -----------


def test_naive_llm_client_emits_name_parent_root_body():
    content = json.dumps({"name": "Mammal", "root": "entity"})
    frozen = {"c1::0": {"choices": [{"message": {"content": content}}]}}
    replayer = rx.FrozenLLMReplayer(frozen)
    replayer.for_cluster("c1")
    replayer.set_repeat(0)
    client = ab.NaiveLLMClient(replayer)
    resp = client.post(
        "/type-cluster",
        json={"members": _members(("u1", "cat")), "request_id": "c1::0"},
    )
    body = resp.json()
    assert body["name"] == "mammal"  # canonicalized
    assert body["parent"] == "entity"  # no catalog -> root parent
    assert "groups" not in body  # one type per cluster, no partition


def test_naive_llm_client_minimal_prompt_has_no_catalog_or_chain():
    messages = ab.NaiveLLMClient.build_messages([{"id": "u1", "name": "cheetah"}])
    blob = " ".join(m["content"] for m in messages)
    for forbidden in ("CATALOG", "assign_to", "chain", "subgroup"):
        assert forbidden not in blob


def test_naive_llm_client_rejects_unknown_path():
    client = ab.NaiveLLMClient(rx.FrozenLLMReplayer({}))
    with pytest.raises(ValueError):
        client.post("/type_cluster", json={"members": []})


def test_naive_llm_client_rejects_non_object_completion():
    content = json.dumps(["not", "an", "object"])
    frozen = {"c1::0": {"choices": [{"message": {"content": content}}]}}
    replayer = rx.FrozenLLMReplayer(frozen)
    replayer.for_cluster("c1")
    replayer.set_repeat(0)
    client = ab.NaiveLLMClient(replayer)
    with pytest.raises(ValueError):
        client.post(
            "/type-cluster",
            json={"members": _members(("u1", "x")), "request_id": "c1::0"},
        )


# ---- frozen-fixture coverage: gated on the new-contract re-freeze --------


@pytest.mark.xfail(reason="frozen fixtures are old {groups} shape; need re-freeze (#18)", strict=False)
@pytest.mark.parametrize("ablation", ("full",) + ARMS)
def test_arm_fixture_covers_k5_per_cluster(ablation):
    clusters = json.loads(
        (FIXTURES_DIR / "clusters.json").read_text(encoding="utf-8")
    )["clusters"]
    responses = json.loads(
        (FIXTURES_DIR / ab.responses_filename(ablation)).read_text("utf-8")
    )
    for cluster in clusters:
        for k in range(REPEATS):
            key = f"{cluster['cluster_id']}::{k}"
            assert key in responses
            body = json.loads(responses[key]["choices"][0]["message"]["content"])
            assert "name" in body  # new contract shape
