"""Tests for harness.ablations -- arms A1-A5 over the frozen seams (#11).

Every arm: coverage-guarded frozen fixture, end-to-end --replay run over the
REAL frozen clusters/catalog, deterministic (run twice, byte-identical
snapshot), arm recorded in the snapshot, and the arm-specific semantics
asserted (A2 never reuses; A3 grafts only to roots; A5 never gates; A1/A4
emit no chain beyond [name, root]).
"""

from __future__ import annotations

import json

import pytest

import harness.ablations as ab
import harness.run_experiment as rx

FIXTURES_DIR = rx.FIXTURES
REPEATS = 5
ARMS = ("naive_llm", "no_reuse", "no_graft", "no_chain", "no_gate")
ROOT_UUIDS = {"r-ent", "r-con", "r-pro"}  # frozen catalog realm roots


def _probe():
    return {
        "type_def_count_before": 0,
        "type_def_count_after": 0,
        "redis_key_before": "",
        "redis_key_after": "",
        "prod_hermes_targeted": False,
    }


def _run_arm(ablation, tmp_path, monkeypatch, run_ts="20260605T040404Z"):
    """Drive run() over the REAL frozen fixtures with the per-arm seams."""
    import hermes.main as m

    from harness.fixtures_io import load_catalog

    paths = rx.HarnessPaths(fixtures_dir=FIXTURES_DIR, workspace_dir=tmp_path / "ws")
    fname = ab.responses_filename(ablation)
    responses = json.loads((FIXTURES_DIR / fname).read_text(encoding="utf-8"))
    replayer = rx.FrozenLLMReplayer(responses)
    monkeypatch.setattr(m, "generate_completion", replayer)

    return rx.run(
        paths=paths,
        catalog_loader=lambda pp: load_catalog(pp.fixtures_dir / "catalog.json"),
        cascade_fn=rx.simulate_cascade_response,
        llm_replayer=replayer,
        nonmutation_probe=_probe,
        ablation=ablation,
        repeats=REPEATS,
        run_ts=run_ts,
        registry_factory=ab.arm_registry_factory(ablation),
        client_factory=ab.arm_client_factory(ablation, replayer),
    )


def _snapshot(ablation, tmp_path, monkeypatch):
    out = _run_arm(ablation, tmp_path, monkeypatch)
    return json.loads(out.read_text(encoding="utf-8"))


def _all_branches(snap):
    return [
        b
        for c in snap["clusters"]
        for r in c["repeats"]
        for b in r["cascade"]["branches"]
    ]


# ---- arm wiring tables ----------------------------------------------------

def test_responses_filename_mapping():
    # Arms whose prompt is byte-identical to full SHARE the full fixture.
    assert ab.responses_filename("full") == "llm_responses.json"
    assert ab.responses_filename("no_reuse") == "llm_responses.json"
    assert ab.responses_filename("no_gate") == "llm_responses.json"
    # Arms whose prompt differs carry their own frozen file.
    assert ab.responses_filename("naive_llm") == "llm_responses_naive_llm.json"
    assert ab.responses_filename("no_graft") == "llm_responses_no_graft.json"
    assert ab.responses_filename("no_chain") == "llm_responses_no_chain.json"


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="unknown ablation arm"):
        ab.responses_filename("bogus")
    with pytest.raises(ValueError, match="unknown ablation arm"):
        ab.arm_registry_factory("bogus")
    with pytest.raises(ValueError, match="unknown ablation arm"):
        ab.arm_client_factory("bogus", replayer=None)
    with pytest.raises(ValueError, match="unknown ablation arm"):
        ab.simulate_arm_cascade({"groups": []}, {}, ablation="bogus")


def test_simulate_arm_cascade_rejects_full():
    with pytest.raises(ValueError, match="A1-A5 only"):
        ab.simulate_arm_cascade({"groups": []}, {}, ablation="full")


def test_arm_seam_factories_default_to_full_wiring():
    for arm in ("full", "no_reuse", "no_chain", "no_gate"):
        assert ab.arm_registry_factory(arm) is None
    for arm in ("full", "no_reuse", "no_graft", "no_chain", "no_gate"):
        assert ab.arm_client_factory(arm, replayer=None) is None


def test_roots_only_catalog_is_a_view_of_the_same_fixture():
    from harness.fixtures_io import load_catalog

    catalog = load_catalog(FIXTURES_DIR / "catalog.json")
    view = ab.roots_only_catalog(catalog)
    assert set(view["catalog_by_uuid"]) == ROOT_UUIDS
    assert set(view["by_norm"]) == {"entity", "concept", "process"}
    # a VIEW: the loaded fixture is untouched
    assert "t-veh1" in catalog["catalog_by_uuid"]
    reg = ab.arm_registry_factory("no_graft")(catalog)
    assert reg.get_type_names() == ["concept", "entity", "process"]


# ---- per-arm frozen fixture coverage guards -------------------------------

@pytest.mark.parametrize("ablation", ("full",) + ARMS)
def test_arm_fixture_covers_k5_per_cluster(ablation):
    clusters = json.loads(
        (FIXTURES_DIR / "clusters.json").read_text(encoding="utf-8")
    )["clusters"]
    fname = ab.responses_filename(ablation)
    responses = json.loads((FIXTURES_DIR / fname).read_text(encoding="utf-8"))
    for cluster in clusters:
        cid = cluster["cluster_id"]
        for k in range(REPEATS):
            key = f"{cid}::{k}"
            assert key in responses, f"{fname} missing frozen response {key}"
            content = responses[key]["choices"][0]["message"]["content"]
            json.loads(content)  # frozen content parses


def test_naive_llm_fixture_is_minimal_name_root_shape():
    responses = json.loads(
        (FIXTURES_DIR / "llm_responses_naive_llm.json").read_text(encoding="utf-8")
    )
    roots_seen = set()
    for key, env in responses.items():
        data = json.loads(env["choices"][0]["message"]["content"])
        assert set(data) <= {"name", "root", "confidence"}, key
        assert isinstance(data["name"], str) and data["name"], key
        assert data["root"] in {"entity", "concept", "process"}, key
        roots_seen.add(data["root"])
    assert len(roots_seen) > 1  # fixture exercises root variety


def test_no_chain_fixture_emits_no_chain():
    responses = json.loads(
        (FIXTURES_DIR / "llm_responses_no_chain.json").read_text(encoding="utf-8")
    )
    for key, env in responses.items():
        data = json.loads(env["choices"][0]["message"]["content"])
        for group in data["groups"]:
            assert len(group["chain"]) <= 2, key  # [name, root] only
            assert group["chain"][-1] in {"entity", "concept", "process"}, key


def test_no_graft_fixture_never_references_the_catalog():
    responses = json.loads(
        (FIXTURES_DIR / "llm_responses_no_graft.json").read_text(encoding="utf-8")
    )
    for key, env in responses.items():
        data = json.loads(env["choices"][0]["message"]["content"])
        for group in data["groups"]:
            assert group["assign_to"] == "NEW", key  # catalog was hidden


# ---- NaiveLLMClient (A1 prompt path) --------------------------------------

def test_naive_llm_client_synthesizes_whole_cluster_group():
    content = json.dumps({"name": "Mammals", "root": "bogus"})
    frozen = {"c1::0": {"choices": [{"message": {"content": content}}]}}
    replayer = rx.FrozenLLMReplayer(frozen)
    replayer.for_cluster("c1")
    replayer.set_repeat(0)
    client = ab.NaiveLLMClient(replayer)

    resp = client.post(
        "/type-cluster",
        json={
            "members": [
                {"id": "u1", "name": "cheetah"},
                {"id": "u2", "name": "narwhal"},
            ],
            "request_id": "c1::0",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "c1::0"
    assert body["raw_partition_ok"] is True
    assert body["residual_ids"] == []
    assert len(body["groups"]) == 1  # single call per WHOLE cluster
    group = body["groups"][0]
    assert group["assign_to"] == "NEW"  # no catalog: nothing to reuse
    assert group["name"] == "mammal"  # canonicalized
    assert group["chain"] == ["mammal", "entity"]  # bad root -> entity
    assert group["member_ids"] == ["u1", "u2"]  # covers every input id


def test_naive_llm_client_minimal_prompt_has_no_catalog_or_rules():
    messages = ab.NaiveLLMClient.build_messages([{"id": "u1", "name": "cheetah"}])
    joined = " ".join(m["content"] for m in messages)
    assert "cheetah" in joined
    for forbidden in ("CATALOG", "assign_to", "chain", "subgroup"):
        assert forbidden not in joined


def test_naive_llm_client_fails_closed():
    replayer = rx.FrozenLLMReplayer({})
    replayer.for_cluster("c1")
    client = ab.NaiveLLMClient(replayer)
    with pytest.raises(KeyError):
        client.post("/type-cluster", json={"members": [{"id": "u1", "name": "x"}]})


def test_naive_llm_client_rejects_unknown_path():
    """Duck-typing the TestClient surface stays fail-closed on a typo path."""
    replayer = rx.FrozenLLMReplayer({})
    client = ab.NaiveLLMClient(replayer)
    with pytest.raises(ValueError, match="only serves /type-cluster"):
        client.post("/type_cluster", json={"members": []})


def test_naive_llm_client_rejects_non_object_completion():
    """A completion that parses to a non-object fails closed with a clear
    error (the arm's analog of the /type-cluster 502 convention, SPEC 3.4)."""
    content = json.dumps(["not", "an", "object"])
    frozen = {"c1::0": {"choices": [{"message": {"content": content}}]}}
    replayer = rx.FrozenLLMReplayer(frozen)
    replayer.for_cluster("c1")
    replayer.set_repeat(0)
    client = ab.NaiveLLMClient(replayer)
    with pytest.raises(ValueError, match="expected a JSON object"):
        client.post(
            "/type-cluster",
            json={"members": [{"id": "u1", "name": "x"}], "request_id": "c1::0"},
        )

    content = json.dumps({"root": "entity"})  # name missing
    frozen = {"c2::0": {"choices": [{"message": {"content": content}}]}}
    replayer2 = rx.FrozenLLMReplayer(frozen)
    replayer2.for_cluster("c2")
    client2 = ab.NaiveLLMClient(replayer2)
    with pytest.raises(ValueError, match="no name"):
        client2.post("/type-cluster", json={"members": [{"id": "u1", "name": "x"}]})


# ---- simulate_arm_cascade unit semantics (conftest catalog) ----------------

def test_arm_cascade_no_reuse_forces_new_without_mutating_input(
    catalog_by_uuid, by_norm
):
    response = {
        "groups": [
            {
                "assign_to": "u-car",
                "name": "car",
                "chain": ["car", "vehicle", "entity"],
                "member_ids": ["m1"],
            }
        ],
        "residual_ids": [],
    }
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    full = rx.simulate_cascade_response(response, catalog, ablation="full")
    assert full["branches"][0]["branch"] == "G1_REUSE"

    armed = ab.simulate_arm_cascade(response, catalog, ablation="no_reuse")
    rec = armed["branches"][0]
    assert rec["branch"] == "G2_GRAFT"  # graft stays alive, reuse is dead
    assert rec["assign_to"] == "NEW"
    # the input response was copied, never mutated
    assert response["groups"][0]["assign_to"] == "u-car"


def test_arm_cascade_no_graft_resolves_only_roots(catalog_by_uuid, by_norm):
    response = {
        "groups": [
            {
                "assign_to": "NEW",
                "name": "sedan",
                "chain": ["sedan", "car", "vehicle", "entity"],
                "member_ids": ["m1"],
            }
        ],
        "residual_ids": [],
    }
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    full = rx.simulate_cascade_response(response, catalog, ablation="full")
    assert full["branches"][0]["branch"] == "G2_GRAFT"  # grafts under car

    armed = ab.simulate_arm_cascade(response, catalog, ablation="no_graft")
    rec = armed["branches"][0]
    assert rec["branch"] == "G3_ROOT"
    assert rec["resolved_parent_uuid"] == "u-entity"


def test_arm_cascade_no_chain_floor_off_reuse_alive(catalog_by_uuid, by_norm):
    response = {
        "groups": [
            {
                "assign_to": "u-car",
                "name": "car",
                "chain": ["car", "entity"],  # degenerate: name + root only
                "member_ids": ["m1"],
            },
            {
                "assign_to": "NEW",
                "name": "sedan",
                "chain": ["sedan", "entity"],
                "member_ids": ["m2"],
            },
        ],
        "residual_ids": [],
    }
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    # under the full arm the degenerate chains floor-gate to RESIDUAL
    full = rx.simulate_cascade_response(response, catalog, ablation="full")
    assert all(b["branch"] == "RESIDUAL" for b in full["branches"])

    armed = ab.simulate_arm_cascade(response, catalog, ablation="no_chain")
    branches = {b["name"]: b for b in armed["branches"]}
    assert branches["car"]["branch"] == "G1_REUSE"  # reuse survives A4
    assert branches["sedan"]["branch"] == "G3_ROOT"  # no chain -> root
    assert not any(
        "FLOOR_VIOLATION" in e for b in armed["branches"] for e in b["events"]
    )


def test_arm_cascade_no_gate_never_gates(catalog_by_uuid, by_norm):
    response = {
        "groups": [
            {
                "assign_to": "NEW",
                "name": "gadget and gizmo",  # ceiling violation
                "chain": ["gadget and gizmo", "entity"],  # floor violation
                "member_ids": ["m1"],
            }
        ],
        "residual_ids": [],
    }
    catalog = {"catalog_by_uuid": catalog_by_uuid, "by_norm": by_norm}
    full = rx.simulate_cascade_response(response, catalog, ablation="full")
    assert full["branches"][0]["branch"] == "RESIDUAL"

    armed = ab.simulate_arm_cascade(response, catalog, ablation="no_gate")
    rec = armed["branches"][0]
    assert rec["branch"] == "G3_ROOT"  # mints instead of gating
    assert not any("VIOLATION" in e for e in rec["events"])


# ---- end-to-end --replay per arm (REAL frozen fixtures) --------------------

@pytest.mark.parametrize("ablation", ARMS)
def test_arm_end_to_end_replay_is_deterministic(ablation, tmp_path, monkeypatch):
    out = _run_arm(ablation, tmp_path, monkeypatch)
    first = out.read_text(encoding="utf-8")
    snap = json.loads(first)
    assert snap["ablation"] == ablation  # snapshot records the arm
    assert snap["label_free"] is True
    assert len(snap["clusters"]) == 3
    assert all(len(c["repeats"]) == REPEATS for c in snap["clusters"])
    # same frozen clusters/catalog as the full arm (the fixture, unchanged)
    assert snap["catalog_size"] == 5
    # deterministic: same fixtures + run_ts -> byte-identical snapshot
    _run_arm(ablation, tmp_path, monkeypatch)
    assert out.read_text(encoding="utf-8") == first


@pytest.mark.parametrize("ablation", ARMS)
def test_arm_snapshot_is_consumable_by_metrics(ablation, tmp_path, monkeypatch):
    from eval.metrics import compute_metrics

    snap = _snapshot(ablation, tmp_path, monkeypatch)
    metrics = compute_metrics(snap)
    assert isinstance(metrics, dict)
    assert "graft_depth_fraction" in metrics
    assert "residual_fraction" in metrics


def test_e2e_naive_llm_one_group_name_root_only(tmp_path, monkeypatch):
    snap = _snapshot("naive_llm", tmp_path, monkeypatch)
    member_ids_by_cluster = {
        "0": {"u1", "u2", "u3"},
        "1": {"u4", "u5", "u6"},
        "2": {"u7", "u8"},
    }
    for cluster in snap["clusters"]:
        want = member_ids_by_cluster[cluster["cluster_id"]]
        for rep in cluster["repeats"]:
            assert rep["raw_partition_ok"] is True
            groups = rep["response"]["groups"]
            assert len(groups) == 1  # single call per whole cluster
            assert set(groups[0]["member_ids"]) == want
            branches = rep["cascade"]["branches"]
            assert len(branches) == 1
            rec = branches[0]
            assert rec["branch"] == "G3_ROOT"  # no catalog, no graft
            assert rec["assign_to"] == "NEW"  # no reuse
            assert len(rec["chain"]) == 2  # name + root, no chain
    # the 2::4 frozen response routes to the concept root
    parents = {b["resolved_parent_uuid"] for b in _all_branches(snap)}
    assert parents <= ROOT_UUIDS
    assert "r-con" in parents


def test_e2e_no_reuse_never_reuses_but_still_grafts(tmp_path, monkeypatch):
    snap = _snapshot("no_reuse", tmp_path, monkeypatch)
    branches = _all_branches(snap)
    assert branches
    assert all(b["branch"] != "G1_REUSE" for b in branches)  # A2: never reuses
    assert all(b["assign_to"] == "NEW" for b in branches)
    # the graft rule stays alive: car/motor vehicle graft under vehicle
    assert any(b["branch"] == "G2_GRAFT" for b in branches)


def test_e2e_no_graft_grafts_only_to_roots(tmp_path, monkeypatch):
    snap = _snapshot("no_graft", tmp_path, monkeypatch)
    branches = _all_branches(snap)
    assert branches
    assert all(b["branch"] not in {"G1_REUSE", "G2_GRAFT"} for b in branches)
    placed = [b for b in branches if b["branch"] == "G3_ROOT"]
    assert placed
    assert all(b["resolved_parent_uuid"] in ROOT_UUIDS for b in placed)


def test_e2e_no_chain_emits_no_chain_but_keeps_reuse(tmp_path, monkeypatch):
    snap = _snapshot("no_chain", tmp_path, monkeypatch)
    branches = _all_branches(snap)
    assert branches
    assert all(len(b["chain"]) <= 2 for b in branches)  # A4: no chain
    assert all(b["branch"] != "G2_GRAFT" for b in branches)  # nothing to walk
    assert any(b["branch"] == "G1_REUSE" for b in branches)  # reuse survives
    assert not any(
        "FLOOR_VIOLATION" in e for b in branches for e in b["events"]
    )


def test_e2e_no_gate_never_gates(tmp_path, monkeypatch):
    snap = _snapshot("no_gate", tmp_path, monkeypatch)
    branches = _all_branches(snap)
    assert branches
    assert all(b["branch"] != "RESIDUAL" for b in branches)
    assert not any("VIOLATION" in e for b in branches for e in b["events"])


# ---- main() CLI wiring ------------------------------------------------------

def test_main_cli_wires_arm_seams(monkeypatch, tmp_path):
    import hermes.main as m

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run_x.json"

    monkeypatch.setattr(rx, "run", fake_run)
    # register a restore for the module-global replay seam main() installs
    monkeypatch.setattr(m, "generate_completion", m.generate_completion)

    rc = rx.main(["--replay", "--ablation", "naive_llm"])
    assert rc == 0
    assert captured["ablation"] == "naive_llm"
    assert isinstance(captured["client_factory"](None), ab.NaiveLLMClient)
    reg = captured["registry_factory"](
        {
            "catalog_by_uuid": {
                "r1": {
                    "uuid": "r1",
                    "name": "entity",
                    "chain": ["entity"],
                    "is_root": True,
                },
                "t1": {
                    "uuid": "t1",
                    "name": "vehicle",
                    "chain": ["entity", "vehicle"],
                    "is_root": False,
                },
            },
            "by_norm": {"entity": ["r1"], "vehicle": ["t1"]},
        }
    )
    assert reg.get_type_names() == ["entity"]  # roots-only registry view

    captured.clear()
    rc = rx.main(["--replay", "--ablation", "full"])
    assert rc == 0
    assert captured["registry_factory"] is None  # full arm: default wiring
    assert captured["client_factory"] is None
