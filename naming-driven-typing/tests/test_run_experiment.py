"""Tests for the naming-driven-typing offline harness (T6).

Label-free, replay-by-default, in-process. No Neo4j/Milvus/Redis/network here:
every dependency is injected as a stub and fixtures live in tmp_path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from harness import run_experiment as rx


def _content(d: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(d)}}]}


def test_replayer_returns_frozen_completion_per_repeat():
    responses = {
        "c1::0": _content({"groups": [{"assign_to": "NEW", "name": "vehicle"}]}),
        "c1::1": _content({"groups": [{"assign_to": "NEW", "name": "conveyance"}]}),
    }
    replayer = rx.FrozenLLMReplayer(responses).for_cluster("c1")
    # repeat 0
    replayer.set_repeat(0)
    out0 = asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
    assert (
        json.loads(out0["choices"][0]["message"]["content"])["groups"][0]["name"]
        == "vehicle"
    )
    # repeat 1
    replayer.set_repeat(1)
    out1 = asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))
    assert (
        json.loads(out1["choices"][0]["message"]["content"])["groups"][0]["name"]
        == "conveyance"
    )


def test_replayer_raises_on_missing_key():
    replayer = rx.FrozenLLMReplayer({}).for_cluster("c1")
    replayer.set_repeat(0)
    with pytest.raises(KeyError):
        asyncio.run(replayer(messages=[{"role": "user", "content": "x"}]))


def test_build_snapshot_shape_and_coverage_flag():
    cluster_results = [
        {
            "cluster_id": "cX",
            "current_name": "vehicle",
            "member_count": 2,
            "repeats": [
                {
                    "repeat": 0,
                    "request_id": "cX::0",
                    "response": {"groups": []},
                    "raw_partition_ok": True,
                    "sample_coverage": 1.0,
                    "cascade": {},
                },
                {
                    "repeat": 1,
                    "request_id": "cX::1",
                    "response": {"groups": []},
                    "raw_partition_ok": True,
                    "sample_coverage": 0.5,
                    "cascade": {},
                },
            ],
        }
    ]
    snap = rx.build_snapshot(
        cluster_results=cluster_results,
        catalog={
            "catalog_by_uuid": {"u1": {"name": "vehicle"}},
            "by_norm": {"vehicle": ["u1"]},
        },
        ablation="full",
        model="gpt-4.1",
        repeats=2,
        catalog_mode="in_process",
        roots_present=True,
        run_ts="20260605T000000Z",
    )
    assert snap["experiment"] == "naming-driven-typing"
    assert snap["ablation"] == "full"
    assert snap["model"] == "gpt-4.1"
    assert snap["repeats"] == 2
    assert snap["catalog_mode"] == "in_process"
    assert snap["roots_present"] is True
    assert snap["label_free"] is True
    assert snap["catalog_size"] == 1
    assert snap["clusters"][0]["cluster_id"] == "cX"
    # the 0.5-coverage repeat must be surfaced so partition==0 isn't misread
    assert "cX" in snap["coverage_flags"]


def test_assert_non_mutation_passes_when_unchanged():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": False,
    }
    rx.assert_non_mutation(probe)  # no raise


def test_assert_non_mutation_raises_on_type_def_drift():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 43,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": False,
    }
    with pytest.raises(rx.NonMutationViolation, match="type-def count"):
        rx.assert_non_mutation(probe)


def test_assert_non_mutation_raises_on_redis_drift():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "MUTATED",
        "prod_hermes_targeted": False,
    }
    with pytest.raises(rx.NonMutationViolation, match="redis"):
        rx.assert_non_mutation(probe)


def test_assert_non_mutation_raises_when_prod_hermes_targeted():
    probe = {
        "type_def_count_before": 42,
        "type_def_count_after": 42,
        "redis_key_before": "abc",
        "redis_key_after": "abc",
        "prod_hermes_targeted": True,
    }
    with pytest.raises(rx.NonMutationViolation, match="prod Hermes"):
        rx.assert_non_mutation(probe)


def _members(*pairs):
    return [{"id": i, "name": n} for i, n in pairs]


def test_call_type_cluster_drives_inprocess_endpoint(monkeypatch):
    import hermes.main as m
    from fastapi.testclient import TestClient

    responses = {
        "cA::0": _content(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "conveyance", "entity"],
                        "member_ids": ["m1", "m2"],
                    }
                ],
                "residual_ids": [],
            }
        )
    }
    replayer = rx.FrozenLLMReplayer(responses).for_cluster("cA")
    replayer.set_repeat(0)
    monkeypatch.setattr(m, "generate_completion", replayer)

    client = TestClient(m.app)
    body = rx.call_type_cluster(
        client, _members(("m1", "boat"), ("m2", "car")), request_id="cA::0"
    )
    assert body["groups"][0]["name"] == "vehicle"
    assert body["raw_partition_ok"] is True


def test_call_type_cluster_raises_on_non_200(monkeypatch):
    import hermes.main as m
    from fastapi.testclient import TestClient

    async def bad_completion(**kwargs):
        # non-dict JSON -> server fails closed with 502 (SPEC 3.4.1)
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(m, "generate_completion", bad_completion)
    client = TestClient(m.app, raise_server_exceptions=False)
    with pytest.raises(rx.HarnessEndpointError):
        rx.call_type_cluster(client, _members(("m1", "boat")), request_id="cB::0")


def _fake_cascade(response, catalog, *, ablation="full"):
    # Minimal T5-shaped record: one branch per group.
    return {
        "branches": [
            {"group": g["name"], "branch": "G2_graft", "parent": "entity"}
            for g in response.get("groups", [])
        ],
        "residual_ids": list(response.get("residual_ids", [])),
    }


def test_run_cluster_repeats_runs_k_times(monkeypatch):
    import hermes.main as m
    from fastapi.testclient import TestClient

    def resp(name):
        return _content(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": name,
                        "chain": [name, "entity"],
                        "member_ids": ["m1", "m2"],
                    }
                ],
                "residual_ids": [],
            }
        )

    responses = {
        "cX::0": resp("vehicle"),
        "cX::1": resp("vehicle"),
        "cX::2": resp("conveyance"),
    }
    replayer = rx.FrozenLLMReplayer(responses)
    monkeypatch.setattr(m, "generate_completion", replayer)
    client = TestClient(m.app)

    cluster = {
        "cluster_id": "cX",
        "current_name": "vehicle",
        "members": [{"id": "m1", "name": "boat"}, {"id": "m2", "name": "car"}],
    }
    catalog = {"catalog_by_uuid": {}, "by_norm": {}}

    results = rx.run_cluster_repeats(
        cluster,
        catalog,
        client=client,
        replayer=replayer,
        cascade_fn=_fake_cascade,
        repeats=3,
    )
    assert len(results) == 3
    assert [r["repeat"] for r in results] == [0, 1, 2]
    assert all(r["sample_coverage"] == 1.0 for r in results)
    assert all(r["raw_partition_ok"] is True for r in results)
    # cascade ran each repeat and recorded a branch
    assert results[0]["cascade"]["branches"][0]["group"] == "vehicle"
    assert results[2]["cascade"]["branches"][0]["group"] == "conveyance"


def test_run_end_to_end_replay_writes_snapshot(tmp_path, monkeypatch):
    import hermes.main as m

    # ---- frozen fixtures in tmp_path (T3-shaped, label-free) ----
    fixtures = tmp_path / "fixtures"
    workspace = tmp_path / "workspace"
    fixtures.mkdir()
    clusters = [
        {
            "cluster_id": "cX",
            "current_name": "vehicle",
            "members": [{"id": "m1", "name": "boat"}, {"id": "m2", "name": "car"}],
        },
    ]
    (fixtures / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    (fixtures / "catalog.json").write_text(
        json.dumps(
            {
                "catalog_by_uuid": {"u1": {"name": "vehicle"}},
                "by_norm": {"vehicle": ["u1"]},
                "roots_present": True,
            }
        ),
        encoding="utf-8",
    )

    def grp(name):
        return {
            "groups": [
                {
                    "assign_to": "NEW",
                    "name": name,
                    "chain": [name, "entity"],
                    "member_ids": ["m1", "m2"],
                }
            ],
            "residual_ids": [],
        }

    llm = {
        "cX::0": _content(grp("vehicle")),
        "cX::1": _content(grp("vehicle")),
    }
    (fixtures / "llm_responses.json").write_text(json.dumps(llm), encoding="utf-8")

    # ---- injected seams ----
    paths = rx.HarnessPaths(fixtures_dir=fixtures, workspace_dir=workspace)

    def catalog_loader(p):
        data = json.loads((p.fixtures_dir / "catalog.json").read_text(encoding="utf-8"))
        return data

    def nonmutation_probe():
        return {
            "type_def_count_before": 7,
            "type_def_count_after": 7,
            "redis_key_before": "x",
            "redis_key_after": "x",
            "prod_hermes_targeted": False,
        }

    # in-process replayer wired onto the real /type-cluster route
    replayer = rx.FrozenLLMReplayer(llm)
    monkeypatch.setattr(m, "generate_completion", replayer)

    out_path = rx.run(
        paths=paths,
        catalog_loader=catalog_loader,
        cascade_fn=_fake_cascade,
        llm_replayer=replayer,
        nonmutation_probe=nonmutation_probe,
        ablation="full",
        repeats=2,
        catalog_mode="in_process",
        model="gpt-4.1",
        run_ts="20260605T010101Z",
    )

    assert out_path.exists()
    snap = json.loads(out_path.read_text(encoding="utf-8"))
    assert snap["experiment"] == "naming-driven-typing"
    assert snap["repeats"] == 2
    assert snap["roots_present"] is True
    assert len(snap["clusters"]) == 1
    assert len(snap["clusters"][0]["repeats"]) == 2
    assert (
        snap["clusters"][0]["repeats"][0]["response"]["groups"][0]["name"] == "vehicle"
    )
    assert snap["coverage_flags"] == []  # whole cluster sent every repeat


def test_run_violation_leaves_no_snapshot_on_disk(tmp_path, monkeypatch):
    """NonMutationViolation fires BEFORE persistence: no run_*.json on disk."""
    import hermes.main as m

    fixtures = tmp_path / "fixtures"
    workspace = tmp_path / "workspace"
    fixtures.mkdir()
    clusters = [
        {
            "cluster_id": "cX",
            "current_name": "vehicle",
            "members": [{"id": "m1", "name": "boat"}, {"id": "m2", "name": "car"}],
        },
    ]
    (fixtures / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    (fixtures / "catalog.json").write_text(
        json.dumps(
            {
                "catalog_by_uuid": {"u1": {"name": "vehicle"}},
                "by_norm": {"vehicle": ["u1"]},
                "roots_present": True,
            }
        ),
        encoding="utf-8",
    )
    llm = {
        "cX::0": _content(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "entity"],
                        "member_ids": ["m1", "m2"],
                    }
                ],
                "residual_ids": [],
            }
        ),
    }
    (fixtures / "llm_responses.json").write_text(json.dumps(llm), encoding="utf-8")

    paths = rx.HarnessPaths(fixtures_dir=fixtures, workspace_dir=workspace)

    def catalog_loader(p):
        return json.loads(
            (p.fixtures_dir / "catalog.json").read_text(encoding="utf-8")
        )

    # Probe reports a graph mutation -- the run must fail loudly.
    def tripping_probe():
        return {
            "type_def_count_before": 7,
            "type_def_count_after": 8,
            "redis_key_before": "x",
            "redis_key_after": "x",
            "prod_hermes_targeted": False,
        }

    replayer = rx.FrozenLLMReplayer(llm)
    monkeypatch.setattr(m, "generate_completion", replayer)

    with pytest.raises(rx.NonMutationViolation, match="type-def count"):
        rx.run(
            paths=paths,
            catalog_loader=catalog_loader,
            cascade_fn=_fake_cascade,
            llm_replayer=replayer,
            nonmutation_probe=tripping_probe,
            ablation="full",
            repeats=1,
            catalog_mode="in_process",
            model="gpt-4.1",
            run_ts="20260605T020202Z",
        )

    # The violation fired before the write: no poisoned snapshot for T7.
    assert not list(workspace.glob("run_*.json"))


def test_run_limit_truncates_clusters(tmp_path, monkeypatch):
    import hermes.main as m

    fixtures = tmp_path / "fixtures"
    workspace = tmp_path / "workspace"
    fixtures.mkdir()
    clusters = [
        {
            "cluster_id": f"c{i}",
            "current_name": "x",
            "members": [{"id": f"{i}a", "name": "boat"}],
        }
        for i in range(3)
    ]
    (fixtures / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    (fixtures / "catalog.json").write_text(
        json.dumps({"catalog_by_uuid": {}, "by_norm": {}, "roots_present": True}),
        encoding="utf-8",
    )
    llm = {
        f"c{i}::0": _content(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "thing",
                        "chain": ["thing", "entity"],
                        "member_ids": [f"{i}a"],
                    }
                ],
                "residual_ids": [],
            }
        )
        for i in range(3)
    }
    (fixtures / "llm_responses.json").write_text(json.dumps(llm), encoding="utf-8")

    paths = rx.HarnessPaths(fixtures_dir=fixtures, workspace_dir=workspace)
    replayer = rx.FrozenLLMReplayer(llm)
    monkeypatch.setattr(m, "generate_completion", replayer)

    out_path = rx.run(
        paths=paths,
        catalog_loader=lambda p: json.loads(
            (p.fixtures_dir / "catalog.json").read_text(encoding="utf-8")
        ),
        cascade_fn=_fake_cascade,
        llm_replayer=replayer,
        nonmutation_probe=lambda: {
            "type_def_count_before": 0,
            "type_def_count_after": 0,
            "redis_key_before": "",
            "redis_key_after": "",
            "prod_hermes_targeted": False,
        },
        repeats=1,
        limit=1,
        run_ts="20260605T020202Z",
    )
    snap = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(snap["clusters"]) == 1  # --limit 1


def test_main_cli_defaults_and_invokes_run(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run_x.json"

    monkeypatch.setattr(rx, "run", fake_run)
    # avoid touching live deps: stub the live wiring builders
    monkeypatch.setattr(
        rx,
        "_build_replay_wiring",
        lambda paths, model: ({}, lambda: {"prod_hermes_targeted": False}),
    )

    rc = rx.main(["--replay", "--repeats", "5", "--ablation", "full", "--limit", "2"])
    assert rc == 0
    assert captured["repeats"] == 5
    assert captured["ablation"] == "full"
    assert captured["limit"] == 2


def test_main_cli_rejects_unknown_ablation():
    with pytest.raises(SystemExit):
        rx.main(["--replay", "--ablation", "bogus"])


# ---- T6 extras: stub registry, cascade adapter, frozen replay fixture ----

FIXTURES_DIR = rx.FIXTURES


def test_stub_type_registry_duck_type_from_catalog():
    catalog = {
        "catalog_by_uuid": {
            "r-ent": {"name": "entity", "chain": ["entity"], "is_root": True},
            "t-veh": {
                "uuid": "t-veh",
                "name": "vehicle",
                "chain": ["entity", "vehicle"],
                "is_root": False,
            },
        },
        "by_norm": {"entity": ["r-ent"], "vehicle": ["t-veh"]},
    }
    reg = rx.StubTypeRegistry.from_catalog(catalog)
    assert reg.get_type_names() == ["entity", "vehicle"]
    veh = reg.get_type("vehicle")
    assert veh == {
        "uuid": "t-veh",
        "root": "entity",
        "chain": ["entity", "vehicle"],
        "is_root": False,
    }
    # uuid falls back to the catalog key; root is chain[0]
    ent = reg.get_type("entity")
    assert ent["uuid"] == "r-ent"
    assert ent["is_root"] is True
    # copies, not aliases -- callers cannot mutate the registry
    veh["uuid"] = "MUTATED"
    assert reg.get_type("vehicle")["uuid"] == "t-veh"
    assert reg.get_type("missing") is None


def test_cascade_adapter_wraps_t5_simulator():
    response = {
        "groups": [
            {
                "assign_to": "NEW",
                "name": "mammal",
                "chain": ["mammal", "animal", "entity"],
                "member_ids": ["u1", "u2"],
            }
        ],
        "residual_ids": ["u3"],
    }
    catalog = {
        "catalog_by_uuid": {
            "r-ent": {
                "uuid": "r-ent",
                "name": "entity",
                "norm_name": "entity",
                "chain": ["entity"],
                "ancestors": [],
                "depth": 1,
                "member_count": 10,
                "parent_uuid": None,
                "is_root": True,
            }
        },
        "by_norm": {"entity": ["r-ent"]},
    }
    out = rx.simulate_cascade_response(response, catalog, ablation="full")
    assert out["residual_ids"] == ["u3"]
    assert len(out["branches"]) == 1
    rec = out["branches"][0]
    assert rec["branch"] in {"G1_REUSE", "G2_GRAFT", "G3_ROOT", "RESIDUAL"}
    assert rec["member_ids"] == ["u1", "u2"]  # tuples converted for JSON
    json.dumps(out)  # snapshot-serializable


def test_cascade_adapter_dispatches_ablation_arms():
    # A1-A5 are wired (issue #11): the seam dispatches to
    # harness.ablations.simulate_arm_cascade instead of raising.
    out = rx.simulate_cascade_response(
        {"groups": [], "residual_ids": ["u9"]},
        {"catalog_by_uuid": {}, "by_norm": {}},
        ablation="no_gate",
    )
    assert out == {"branches": [], "residual_ids": ["u9"]}
    with pytest.raises(ValueError, match="unknown ablation arm"):
        rx.simulate_cascade_response(
            {"groups": []},
            {"catalog_by_uuid": {}, "by_norm": {}},
            ablation="bogus",
        )


def test_frozen_llm_responses_cover_k5_per_cluster():
    """Replay fixture invariant: K>=5 frozen completions per frozen cluster."""
    clusters = json.loads(
        (FIXTURES_DIR / "clusters.json").read_text(encoding="utf-8")
    )["clusters"]
    responses = json.loads(
        (FIXTURES_DIR / "llm_responses.json").read_text(encoding="utf-8")
    )
    assert clusters, "frozen clusters fixture is empty"
    for cluster in clusters:
        cid = cluster["cluster_id"]
        member_ids = {m["id"] for m in cluster["members"]}
        for k in range(5):
            key = rx.ReplayKey(cid, k).token()
            assert key in responses, f"missing frozen response {key!r}"
            content = responses[key]["choices"][0]["message"]["content"]
            data = json.loads(content)
            claimed = [
                gid for g in data["groups"] for gid in g["member_ids"]
            ]
            residual = data.get("residual_ids", [])
            # frozen partitions are valid: disjoint + total over the cluster
            assert len(claimed) == len(set(claimed))
            assert set(claimed) | set(residual) == member_ids


def test_run_replays_real_frozen_fixtures_deterministically(tmp_path, monkeypatch):
    """End-to-end --replay over the REAL frozen fixtures: 3 clusters x K=5,
    real /type-cluster route, real T5 cascade, stub registry from the frozen
    catalog, snapshot deterministic across runs. Workspace stays in tmp_path."""
    import hermes.main as m

    from harness.fixtures_io import load_catalog

    paths = rx.HarnessPaths(fixtures_dir=FIXTURES_DIR, workspace_dir=tmp_path / "ws")
    responses, probe = rx._build_replay_wiring(paths, "gpt-4.1")
    replayer = rx.FrozenLLMReplayer(responses)
    monkeypatch.setattr(m, "generate_completion", replayer)

    registry_seen = {}

    def spying_cascade(response, catalog, *, ablation="full"):
        registry_seen["during"] = m._type_registry
        return rx.simulate_cascade_response(response, catalog, ablation=ablation)

    before_registry = m._type_registry
    out_path = rx.run(
        paths=paths,
        catalog_loader=lambda pp: load_catalog(pp.fixtures_dir / "catalog.json"),
        cascade_fn=spying_cascade,
        llm_replayer=replayer,
        nonmutation_probe=probe,
        ablation="full",
        repeats=5,
        run_ts="20260605T030303Z",
    )
    first = out_path.read_text(encoding="utf-8")

    # the in-process stub registry served the catalog and was restored after
    assert isinstance(registry_seen["during"], rx.StubTypeRegistry)
    assert m._type_registry is before_registry

    snap = json.loads(first)
    assert len(snap["clusters"]) == 3
    assert all(len(c["repeats"]) == 5 for c in snap["clusters"])
    assert snap["coverage_flags"] == []
    assert snap["catalog_size"] == 5
    assert snap["roots_present"] is True
    repeats_flat = [r for c in snap["clusters"] for r in c["repeats"]]
    assert all(r["raw_partition_ok"] is True for r in repeats_flat)
    # the reuse arm is exercised: cluster '1' reuses published uuid t-veh1
    cluster1 = next(c for c in snap["clusters"] if c["cluster_id"] == "1")
    branches0 = cluster1["repeats"][0]["cascade"]["branches"]
    assert any(b["branch"] == "G1_REUSE" for b in branches0)

    # deterministic: same fixtures + run_ts -> byte-identical snapshot
    rx.run(
        paths=paths,
        catalog_loader=lambda pp: load_catalog(pp.fixtures_dir / "catalog.json"),
        cascade_fn=spying_cascade,
        llm_replayer=replayer,
        nonmutation_probe=probe,
        ablation="full",
        repeats=5,
        run_ts="20260605T030303Z",
    )
    assert out_path.read_text(encoding="utf-8") == first
