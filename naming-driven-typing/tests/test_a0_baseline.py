"""Offline tests for the A0 measured rollup baseline (harness/a0_baseline.py).

Pure-function coverage only: geometry, the published seed plan, the graph
outcome -> T6 cascade mapping, snapshot assembly, the metrics dialect
bridge, and deltas-path compatibility (a synthetic A0 snapshot plus a
full-arm snapshot through ablation_deltas / ablation_criterion_metrics).
No live Neo4j: the live driver path is env-gated (A0_LIVE=1) and exercised
via the CLI, never from this suite.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import fields
from pathlib import Path

import pytest

from eval.metrics import (
    ablation_criterion_metrics,
    ablation_deltas,
    compute_metrics,
)
from harness import a0_baseline as a0
from harness.cascade import PlacementRecord
from harness.fixtures_io import load_catalog, load_clusters
from harness.run_experiment import build_snapshot

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

_NS = "tok"
_SEED = "type_tokdrawer"
_PUB = "type_tokpub0"


@pytest.fixture(scope="module")
def clusters():
    return load_clusters(_FIXTURES / "clusters.json")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(_FIXTURES / "catalog.json")


@pytest.fixture(scope="module")
def plan(clusters, catalog):
    return a0.geometry_plan(clusters, catalog)


@pytest.fixture(scope="module")
def realm_roots(catalog):
    return a0.realm_roots_from_catalog(catalog)


def _outcome_kwargs(**overrides):
    kwargs = {
        "ns": _NS,
        "seed_uuid": _SEED,
        "member_type": {},
        "type_name_of": {},
        "parent_of": {},
        "published": {_PUB: {"uuid": "t-veh1", "name": "vehicle"}},
    }
    kwargs.update(overrides)
    return kwargs


# ---- geometry -------------------------------------------------------------


def test_sidon_pairwise_differences_distinct():
    eps = [a0.sidon_epsilon(i, scale=0.01) for i in range(11)]
    gaps = [abs(x - y) for x, y in itertools.combinations(eps, 2)]
    assert len(gaps) == len(set(gaps))


def test_sidon_extends_on_demand_at_graded_scale():
    """142 clusters + ~220 published norms need ~365 axes (#18)."""
    eps = [a0.sidon_epsilon(i, scale=1.0) for i in range(400)]
    gaps = {round(abs(x - y), 9) for x, y in itertools.combinations(eps, 2)}
    assert len(gaps) == 400 * 399 // 2  # all pairwise differences distinct


def test_geometry_plan_eps_stay_below_unit_axis_at_graded_scale():
    clusters = [
        {"cluster_id": str(i), "members": [{"id": f"m{i}", "name": f"m{i}"}]}
        for i in range(142)
    ]
    catalog = {
        "catalog_by_uuid": {
            f"t{i}": {"name": f"type {i}", "is_root": False} for i in range(220)
        }
    }
    plan = a0.geometry_plan(clusters, catalog)
    n_main = len(plan["cluster_axis"]) + len(plan["publish_axis"])
    eps = [a0.sidon_epsilon(i, scale=plan["eps_scale"]) for i in range(n_main)]
    assert max(eps) <= a0._EPS_CEILING
    assert len(set(eps)) == n_main


def test_geometry_plan_axes(plan):
    assert plan["cluster_axis"] == {"0": 0, "1": 1, "2": 2}
    # Both published vehicle fragments share ONE normalized-name axis.
    assert plan["publish_axis"] == {"vehicle": 3}
    assert plan["dim"] == plan["eps_axis"] + 1


def test_inter_group_center_distances_all_distinct(plan):
    n_main = len(plan["cluster_axis"]) + len(plan["publish_axis"])
    centers = [a0.axis_vector(plan, i) for i in range(n_main)]

    def d2(u, v):
        return sum((x - y) ** 2 for x, y in zip(u, v))

    dists = [round(d2(u, v), 12) for u, v in itertools.combinations(centers, 2)]
    assert len(dists) == len(set(dists))


def test_member_vectors_jitter_apart_on_their_axis(plan):
    vecs = [a0.member_vector(plan, "0", p, 3) for p in range(3)]
    main = plan["cluster_axis"]["0"]
    jit = plan["jitter_axis"]
    assert all(v[main] == 1.0 for v in vecs)
    assert vecs[0][jit] < vecs[1][jit] < vecs[2][jit]
    assert vecs[1][jit] == 0.0


def test_published_vectors_offset_apart_within_group(plan):
    va = a0.published_vector(plan, "vehicle", 0, 2)
    vb = a0.published_vector(plan, "vehicle", 1, 2)
    axis = plan["publish_axis"]["vehicle"]
    assert va[axis] == vb[axis] == 1.0
    assert va[plan["off_axis"]] != vb[plan["off_axis"]]


# ---- published seed plan ---------------------------------------------------


def test_published_rows_seed_plan(catalog):
    rows = a0.published_rows(catalog, _NS)
    assert [r["catalog_uuid"] for r in rows] == ["t-veh1", "t-veh2"]
    for row in rows:
        assert _NS in row["live_uuid"]
        assert row["parent_uuid"] == "type_entity"
        assert row["ancestors"] == ["root", "node", "entity"]
        assert row["norm"] == "vehicle"
        assert row["group_size"] == 2
    assert [r["group_position"] for r in rows] == [0, 1]


def test_realm_roots_from_catalog(realm_roots):
    assert set(realm_roots) == {"entity", "concept", "process"}
    assert realm_roots["entity"]["uuid"] == "r-ent"


# ---- branch records --------------------------------------------------------


def _g3_branch(**overrides):
    values = {
        "cluster_id": "0",
        "branch": "G3_ROOT",
        "assign_to": "NEW",
        "name": "tok c0",
        "chain": ["tok c0", "entity"],
        "member_ids": ["u1"],
        "resolved_parent_uuid": "r-ent",
        "resolved_parent_name": "entity",
        "covering_depth": 1,
    }
    values.update(overrides)
    return a0._branch_record(**values)


def test_branch_record_exact_placement_record_parity():
    rec = _g3_branch()
    assert set(rec) == {f.name for f in fields(PlacementRecord)}
    assert rec["self_reported"] is False  # measured from the graph


def test_branch_record_fails_closed_on_schema_drift():
    with pytest.raises(ValueError, match="branch schema drift"):
        _g3_branch(bogus_key=1)


# ---- map_outcomes ----------------------------------------------------------


def _scenario_a_kwargs(realm_roots):
    """G3 (cluster 0) + G2 with residual (cluster 1) + G1 reuse (cluster 2)."""
    return dict(
        realm_roots=realm_roots,
        **_outcome_kwargs(
            member_type={
                "tok-u1": "type_tok_c0_aaaa",
                "tok-u2": "type_tok_c0_aaaa",
                "tok-u3": "type_tok_c0_aaaa",
                "tok-u4": "type_tok_c1_bbbb",
                "tok-u5": "type_tok_c1_bbbb",
                "tok-u6": _SEED,
                "tok-u7": _PUB,
                "tok-u8": _PUB,
            },
            type_name_of={
                "type_tok_c0_aaaa": "tok c0",
                "type_tok_c1_bbbb": "tok c1",
                _PUB: "vehicle",
            },
            parent_of={
                "type_tok_c0_aaaa": "type_entity",
                "type_tok_c1_bbbb": _PUB,
                _PUB: "type_entity",
            },
        ),
    )


@pytest.fixture
def scenario_a(clusters, realm_roots):
    return a0.map_outcomes(clusters, **_scenario_a_kwargs(realm_roots))


def test_map_outcomes_g3_root(scenario_a):
    (branch,) = scenario_a["0"]["branches"]
    assert branch["branch"] == "G3_ROOT"
    assert branch["assign_to"] == "NEW"
    assert branch["chain"] == ["tok c0", "entity"]
    assert branch["covering_depth"] == 1
    # Realm-root placement maps onto the FROZEN catalog root record.
    assert branch["resolved_parent_uuid"] == "r-ent"
    assert branch["resolved_parent_name"] == "entity"
    assert sorted(branch["member_ids"]) == ["u1", "u2", "u3"]
    assert scenario_a["0"]["residual_ids"] == []


def test_map_outcomes_g2_graft_and_residual(scenario_a):
    (branch,) = scenario_a["1"]["branches"]
    assert branch["branch"] == "G2_GRAFT"
    assert branch["assign_to"] == "NEW"
    assert branch["chain"] == ["tok c1", "vehicle", "entity"]
    assert branch["covering_depth"] == 2
    # Nearest published ancestor maps onto the FROZEN catalog uuid.
    assert branch["resolved_parent_uuid"] == "t-veh1"
    assert branch["resolved_parent_name"] == "vehicle"
    assert sorted(branch["member_ids"]) == ["u4", "u5"]
    assert ("A0_GRAFT_VIA_LIVE:" + _PUB) in branch["events"]
    # The never-typed member is the fragmentation signal, not a branch.
    assert scenario_a["1"]["residual_ids"] == ["u6"]


def test_map_outcomes_g1_reuse(scenario_a):
    (branch,) = scenario_a["2"]["branches"]
    assert branch["branch"] == "G1_REUSE"
    assert branch["assign_to"] == "t-veh1"
    assert branch["resolved_parent_uuid"] == "t-veh1"
    assert branch["name"] == "vehicle"
    assert sorted(branch["member_ids"]) == ["u7", "u8"]
    assert ("A0_REUSE_LIVE:" + _PUB) in branch["events"]


def test_map_outcomes_deterministic(clusters, realm_roots):
    kwargs = _scenario_a_kwargs(realm_roots)
    first = a0.map_outcomes(clusters, **kwargs)
    second = a0.map_outcomes(clusters, **kwargs)
    assert first == second


def test_map_outcomes_type_spanning_clusters_splits_branches(
    clusters, realm_roots
):
    shared = "type_tok_cx_cccc"
    member_type = {f"tok-u{i}": shared for i in range(1, 7)}
    member_type["tok-u7"] = _SEED
    member_type["tok-u8"] = _SEED
    out = a0.map_outcomes(
        clusters,
        realm_roots=realm_roots,
        **_outcome_kwargs(
            member_type=member_type,
            type_name_of={shared: "tok cx"},
            parent_of={shared: "type_entity"},
        ),
    )
    (b0,) = out["0"]["branches"]
    (b1,) = out["1"]["branches"]
    # One branch per cluster, each carrying ONLY its own member ids.
    assert sorted(b0["member_ids"]) == ["u1", "u2", "u3"]
    assert sorted(b1["member_ids"]) == ["u4", "u5", "u6"]
    assert "A0_TYPE_SPANS_CLUSTERS" in b0["events"]
    assert "A0_TYPE_SPANS_CLUSTERS" in b1["events"]
    assert sorted(out["2"]["residual_ids"]) == ["u7", "u8"]


def test_map_outcomes_drawer_parent_reads_as_entity(clusters, realm_roots):
    minted = "type_tok_c0_dddd"
    out = a0.map_outcomes(
        clusters,
        realm_roots=realm_roots,
        **_outcome_kwargs(
            member_type={f"tok-u{i}": minted for i in (1, 2, 3)},
            type_name_of={minted: "tok c0"},
            parent_of={minted: _SEED},
        ),
    )
    (branch,) = out["0"]["branches"]
    # The disposable junk-drawer seed reads back as the entity drawer.
    assert branch["branch"] == "G3_ROOT"
    assert branch["chain"] == ["tok c0", "entity"]
    assert branch["resolved_parent_uuid"] == "r-ent"


def test_map_outcomes_cycle_guarded(clusters, realm_roots):
    out = a0.map_outcomes(
        clusters,
        realm_roots=realm_roots,
        **_outcome_kwargs(
            member_type={"tok-u7": "type_tok_a", "tok-u8": "type_tok_a"},
            type_name_of={"type_tok_a": "na", "type_tok_b": "nb"},
            parent_of={
                "type_tok_a": "type_tok_b",
                "type_tok_b": "type_tok_a",
            },
        ),
    )
    (branch,) = out["2"]["branches"]
    # A corrupt adjacency terminates instead of hanging; no realm root is
    # reached, so the parent stays unresolved.
    assert branch["branch"] == "G3_ROOT"
    assert branch["chain"] == ["na", "nb"]
    assert branch["resolved_parent_uuid"] is None
    assert branch["resolved_parent_name"] == "nb"


# ---- T6 snapshot -----------------------------------------------------------

_T6_TOP_KEYS = {
    "experiment",
    "label_free",
    "run_ts",
    "model",
    "ablation",
    "repeats",
    "catalog_mode",
    "roots_present",
    "catalog_size",
    "coverage_flags",
    "clusters",
}


def test_build_a0_snapshot_exact_t6_schema(clusters, catalog, scenario_a):
    snap = a0.build_a0_snapshot(
        clusters, scenario_a, catalog, run_ts="20260101T000000Z"
    )
    assert set(snap) == _T6_TOP_KEYS
    assert snap["ablation"] == "rollup_baseline"
    assert snap["repeats"] == 1  # deterministic pipeline: K = 1
    assert snap["roots_present"] is True
    assert snap["coverage_flags"] == []
    assert snap["run_ts"] == "20260101T000000Z"
    for cluster in snap["clusters"]:
        cid = cluster["cluster_id"]
        (rep,) = cluster["repeats"]
        assert rep["repeat"] == 0
        assert rep["request_id"] == cid + "::0"
        # Vacuous for the baseline (module docstring): production consumes
        # whole clusters and has no LLM partition contract.
        assert rep["raw_partition_ok"] is True
        assert rep["sample_coverage"] == 1.0
        assert rep["cascade"] == scenario_a[cid]


# ---- metrics dialect bridge ------------------------------------------------


def test_as_metrics_snapshot_bridges_t6_dialect(clusters, catalog, scenario_a):
    snap = a0.build_a0_snapshot(
        clusters, scenario_a, catalog, run_ts="20260101T000000Z"
    )
    bridged = a0.as_metrics_snapshot(snap)
    assert bridged["roots_present_in_live_catalog"] is True
    by_id = {c["cluster_id"]: c for c in bridged["clusters"]}
    assert {cid: c["total_members"] for cid, c in by_id.items()} == {
        "0": 3,
        "1": 3,
        "2": 2,
    }
    (rep1,) = by_id["1"]["repeats"]
    assert rep1["residual_ids"] == ["u6"]
    (group,) = rep1["groups"]
    assert group["branch"] == "G2_GRAFT"
    assert group["is_grafted"] is True
    assert group["graft_parent_uuid"] == "t-veh1"
    (rep2,) = by_id["2"]["repeats"]
    (reuse,) = rep2["groups"]
    assert reuse["is_reuse"] is True
    assert reuse["reuse_target_uuid"] == "t-veh1"


def test_compute_metrics_consumes_bridged_a0_snapshot(
    clusters, catalog, scenario_a
):
    snap = a0.build_a0_snapshot(
        clusters, scenario_a, catalog, run_ts="20260101T000000Z"
    )
    metrics = compute_metrics(a0.as_metrics_snapshot(snap))
    # Two NEW groups (G3 + G2), one grafted at depth 2; one reuse; one
    # residual member of eight; K = 1 aggregates cleanly (n == 1).
    assert metrics["graft_depth_fraction"]["mean"] == 0.5
    assert metrics["graft_depth_fraction"]["n"] == 1
    assert metrics["mean_graft_depth"]["mean"] == 2.0
    assert metrics["new_floated_at_root"]["mean"] == 1.0
    assert metrics["reuse_collapses"]["mean"] == 1.0
    assert metrics["residual_fraction"]["mean"] == pytest.approx(1 / 8)
    assert metrics["raw_partition_violation_rate"]["mean"] == 0.0
    assert metrics["hallucinated_target_rate"]["mean"] == 0.0
    assert metrics["root_distribution"] == {"entity": 3.0}


def test_as_metrics_snapshot_handles_residual_branch_dialect(catalog):
    residual_branch = a0._branch_record(
        cluster_id="x",
        branch="RESIDUAL",
        assign_to="RESIDUAL",
        name="",
        chain=[],
        member_ids=["x1", "x2"],
        resolved_parent_uuid=None,
        resolved_parent_name=None,
        covering_depth=0,
    )
    reuse_branch = a0._branch_record(
        cluster_id="x",
        branch="G1_REUSE",
        assign_to="t-veh1",
        name="vehicle",
        chain=["vehicle", "entity"],
        member_ids=["x5"],
        resolved_parent_uuid="t-veh1",
        resolved_parent_name="vehicle",
        covering_depth=1,
        residual_ids=["x3"],
        evicted_ids=["x4"],
    )
    snap = build_snapshot(
        cluster_results=[
            {
                "cluster_id": "x",
                "current_name": "entity",
                "member_count": 5,
                "repeats": [
                    {
                        "repeat": 0,
                        "request_id": "x::0",
                        "raw_partition_ok": False,
                        "sample_coverage": 1.0,
                        "cascade": {
                            "branches": [residual_branch, reuse_branch],
                            "residual_ids": [],
                        },
                    }
                ],
            }
        ],
        catalog=catalog,
        ablation="no_reuse",
        model="m",
        repeats=1,
        catalog_mode="frozen",
        roots_present=True,
        run_ts="20260101T000000Z",
    )
    bridged = a0.as_metrics_snapshot(snap)
    (rep,) = bridged["clusters"][0]["repeats"]
    assert rep["raw_partition_ok"] is False
    # RESIDUAL branches park members in residual_ids and emit no group.
    assert rep["residual_ids"] == ["x1", "x2", "x3"]
    assert rep["evicted_ids"] == ["x4"]
    assert [g["branch"] for g in rep["groups"]] == ["G1_REUSE"]


# ---- eval compatibility: the deltas path -----------------------------------


def _full_arm_snapshot(clusters, catalog):
    """Synthetic full-v2 arm: K = 2, every cluster grafted (varying depth)."""
    cluster_results = []
    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        member_ids = [str(m["id"]) for m in cluster["members"]]
        repeats = []
        for k, depth in enumerate((2, 3)):
            branch = a0._branch_record(
                cluster_id=cid,
                branch="G2_GRAFT",
                assign_to="NEW",
                name="full " + cid,
                chain=["full " + cid] + ["mid"] * (depth - 1) + ["entity"],
                member_ids=member_ids,
                resolved_parent_uuid="t-veh1",
                resolved_parent_name="vehicle",
                covering_depth=depth,
            )
            repeats.append(
                {
                    "repeat": k,
                    "request_id": cid + "::" + str(k),
                    "raw_partition_ok": True,
                    "sample_coverage": 1.0,
                    "cascade": {"branches": [branch], "residual_ids": []},
                }
            )
        cluster_results.append(
            {
                "cluster_id": cid,
                "current_name": cluster.get("current_name", ""),
                "member_count": len(member_ids),
                "repeats": repeats,
            }
        )
    return build_snapshot(
        cluster_results=cluster_results,
        catalog=catalog,
        ablation="full",
        model="m",
        repeats=2,
        catalog_mode="frozen",
        roots_present=True,
        run_ts="20260101T000001Z",
    )


def test_deltas_path_accepts_a0_alongside_full_arm(
    clusters, catalog, scenario_a
):
    a0_snap = a0.build_a0_snapshot(
        clusters, scenario_a, catalog, run_ts="20260101T000000Z"
    )
    full_snap = _full_arm_snapshot(clusters, catalog)
    by_arm = dict(
        [a0.deltas_arm_entry(full_snap), a0.deltas_arm_entry(a0_snap)]
    )
    assert set(by_arm) == {"full", "rollup_baseline"}
    # Every entry is aggregate-shaped: the exact by_arm input contract.
    for arm_metrics in by_arm.values():
        for agg in arm_metrics.values():
            assert "mean" in agg and "stdev" in agg
    deltas = ablation_deltas(by_arm)
    gd = deltas["graft_depth_fraction"]["full_vs_rollup_baseline"]
    # full grafts every NEW group (1.0); the A0 scenario grafts one of two.
    assert gd["delta"] == pytest.approx(0.5)
    assert set(gd) == {"delta", "noise_band", "passes"}
    flat = ablation_criterion_metrics(by_arm)
    assert "ablation_A6_beats_A0_graft_depth_fraction" in flat
    assert "ablation_A6_beats_A0_residual_fraction" in flat
    assert flat["ablation_A6_beats_A0_graft_depth_fraction"] > 0
    assert all(math.isfinite(v) for v in flat.values())


# ---- namer + vector store seams ---------------------------------------------


class _Cluster:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def test_axis_labels_embed_namespace_token(plan):
    labels = a0.axis_labels(plan, _NS)
    n_main = len(plan["cluster_axis"]) + len(plan["publish_axis"])
    assert set(labels) == set(range(n_main))
    assert all(_NS in label for label in labels.values())


def test_make_axis_namer_dominant_axis(plan):
    labels = a0.axis_labels(plan, _NS)
    namer = a0.make_axis_namer(labels)
    members = [a0.member_vector(plan, "1", p, 3) for p in range(3)]
    result = namer(_Cluster(members), [], "http://stub.invalid", "tok")
    assert result.label == labels[plan["cluster_axis"]["1"]]
    assert result.confidence == 0.9
    assert namer.unexpected_axes == []


def test_make_axis_namer_surfaces_unlabeled_axis(plan):
    namer = a0.make_axis_namer(a0.axis_labels(plan, _NS))
    weird = [0.0] * plan["dim"]
    weird[plan["jitter_axis"]] = 1.0
    result = namer(_Cluster([weird]), [], "http://stub.invalid", "tok")
    # Recorded and surfaced AFTER the run, never silently defaulted.
    assert result.confidence == 0.0
    assert result.label == ""
    assert namer.unexpected_axes == [plan["jitter_axis"]]


def test_fake_milvus_contract():
    milvus = a0.FakeMilvus()
    milvus.update_centroid(type_uuid="type_x", centroid=[1.0, 0.0], model="m")
    milvus.update_centroid(type_uuid="type_y", centroid=[0.0, 1.0], model="m")
    row = milvus.get_embedding("TypeCentroid", "type_x")
    assert row["embedding"] == [1.0, 0.0]
    assert milvus.get_embedding("entity", "type_x") is None
    assert milvus.find_nearest_types([0.9, 0.1], top_k=1) == [{"uuid": "type_x"}]
    ranked = milvus.find_nearest_types([0.1, 0.9], top_k=2)
    assert [r["uuid"] for r in ranked] == ["type_y", "type_x"]


# ---- env gate ----------------------------------------------------------------


def test_run_live_refuses_without_gate(monkeypatch):
    monkeypatch.delenv("A0_LIVE", raising=False)
    with pytest.raises(a0.A0LiveGateError, match="A0_LIVE=1"):
        a0.run_live()


def test_cli_refuses_without_gate(monkeypatch, capsys):
    monkeypatch.delenv("A0_LIVE", raising=False)
    assert a0.main([]) == 2
    assert "A0_LIVE=1" in capsys.readouterr().err


class _ResidueHCG:
    """Fake client whose residue count never reaches zero; records queries."""

    def __init__(self):
        self.queries = []

    def _execute_query(self, query, params):
        self.queries.append((query, params))
        if "RETURN count(n) AS c" in query:
            return [{"c": 2}]
        return []


def test_teardown_cleans_shared_root_even_on_residue_violation():
    """The shared root does not carry the namespace token; its cleanup must
    run even when the zero-residue check raises (PR #17 review)."""
    hcg = _ResidueHCG()
    with pytest.raises(a0.ZeroResidueViolation, match="left 2"):
        a0._teardown(hcg, "ns-token", created_root=True)
    root_deletes = [
        (q, p) for q, p in hcg.queries if "uuid: $u" in q and "DETACH DELETE" in q
    ]
    assert len(root_deletes) == 1
    assert root_deletes[0][1]["u"] == "type_entity"


def test_teardown_skips_root_cleanup_when_root_preexisted():
    hcg = _ResidueHCG()
    with pytest.raises(a0.ZeroResidueViolation):
        a0._teardown(hcg, "ns-token", created_root=False)
    assert not [q for q, _ in hcg.queries if "uuid: $u" in q]


def test_outside_type_defs_skips_uuidless_records():
    """A malformed type-def without a uuid must be skipped consistently on
    both the before and after capture, not crash the run (PR #17 review)."""

    class _Defs:
        def get_all_type_definitions(self):
            return [
                {"uuid": "keep-1", "properties": {"ancestors": ["root"]}},
                {"properties": {"ancestors": ["root"]}},
                {"uuid": None, "properties": {}},
                {"uuid": "ns-token-x", "properties": {}},
            ]

    out = a0._outside_type_defs(_Defs(), "ns-token")
    assert out == {"keep-1": ["root"]}


class _RootProbeHCG:
    def __init__(self, existing):
        self._existing = existing
        self.added = []

    def get_node(self, uuid):
        return self._existing

    def add_node(self, **kwargs):
        self.added.append(kwargs)
        return kwargs.get("uuid")


def test_ensure_root_creates_when_missing():
    hcg = _RootProbeHCG(existing=None)
    assert a0._ensure_root(hcg) is True
    assert [n["uuid"] for n in hcg.added] == ["type_entity"]


def test_ensure_root_noops_when_present():
    hcg = _RootProbeHCG(existing={"uuid": "type_entity"})
    assert a0._ensure_root(hcg) is False
    assert hcg.added == []


def test_close_quietly_suppresses_close_errors(capsys):
    """A close() failure must not mask the teardown diagnostic (PR #17)."""

    class _BadClose:
        def close(self):
            raise ConnectionError("socket gone")

    a0._close_quietly(_BadClose())
    assert "close() failed" in capsys.readouterr().err


def test_connect_requires_explicit_password(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(a0.A0LiveGateError, match="NEO4J_PASSWORD"):
        a0._connect()
