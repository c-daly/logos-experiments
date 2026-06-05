"""Tests for the label-free structural metrics of the naming-driven-typing experiment."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from eval.metrics import (
    ablation_criterion_metrics,
    ablation_deltas,
    aggregate_repeats,
    compute_metrics,
    emit,
    eyeball_dump,
)

_FIX = Path(__file__).resolve().parent.parent / "eval" / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIX / name).read_text(encoding="utf-8"))


# ----- aggregate_repeats -------------------------------------------------

def test_aggregate_repeats_basic():
    agg = aggregate_repeats([0.6, 0.8])
    assert agg["mean"] == 0.7
    assert round(agg["stdev"], 4) == 0.1  # population stdev of {0.6, 0.8}
    assert agg["n"] == 2
    assert round(agg["cv"], 4) == round(0.1 / 0.7, 4)
    assert round(agg["ci_lo"], 4) == 0.6
    assert round(agg["ci_hi"], 4) == 0.8


def test_aggregate_repeats_zero_mean_cv_is_zero():
    agg = aggregate_repeats([0.0, 0.0])
    assert agg["mean"] == 0.0
    assert agg["stdev"] == 0.0
    assert agg["cv"] == 0.0  # guard div-by-zero


def test_aggregate_repeats_empty():
    agg = aggregate_repeats([])
    assert agg["mean"] == 0.0
    assert agg["n"] == 0
    assert agg["cv"] == 0.0


# ----- compute_metrics: structural, label-free ---------------------------

def test_compute_metrics_keys_present():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    for key in (
        "graft_depth_fraction",
        "mean_graft_depth",
        "new_floated_at_root",
        "reuse_collapses",
        "canonical_merge_collapses",
        "residual_fraction",
        "raw_partition_violation_rate",
        "hallucinated_target_rate",
        "placement_conflict_rate",
        "root_distribution",
        "residual_bloat",
        "roots_present_in_live_catalog",
        "live_redis_catalog_staleness",
        "sample_coverage_min",
        "stability_cv_max",
    ):
        assert key in m, f"missing metric {key}"


def test_graft_depth_fraction_aggregated_over_repeats():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # Per repeat: exactly one NEW group (c_aircraft), grafted under non-root in
    # BOTH repeats => grafted/total_new = 1/1 = 1.0 each repeat => mean 1.0.
    assert m["graft_depth_fraction"]["mean"] == 1.0
    assert m["graft_depth_fraction"]["stdev"] == 0.0


def test_mean_graft_depth():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # Only grafted group has covering_depth 2 in both repeats.
    assert m["mean_graft_depth"]["mean"] == 2.0


def test_reuse_collapses_semantic_only():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # c_boat reuses uuid-vehicle (is_reuse) in both repeats => 1 per repeat.
    # canonical_merged_into is null everywhere => canonical_merge_collapses == 0.
    assert m["reuse_collapses"]["mean"] == 1.0
    assert m["canonical_merge_collapses"]["mean"] == 0.0


def test_residual_fraction_and_bloat():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # total_members = 4 + 3 + 2 = 9.
    # repeat 0 residual: c_shallow s1,s2 => 2/9.
    # repeat 1 residual: c_aircraft a3 + c_shallow s1,s2 => 3/9.
    # mean residual_fraction = (2/9 + 3/9) / 2 = 2.5/9.
    assert round(m["residual_fraction"]["mean"], 4) == round(2.5 / 9, 4)
    assert m["residual_bloat"] is False  # 0.27 < 0.4


def test_raw_partition_violation_rate():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # Per repeat over 3 clusters: repeat 0 => 0/3 violations, repeat 1 => 1/3
    # (c_aircraft) => rates [0, 1/3], mean 1/6. Matches the old globally
    # pooled value because every repeat has the same cell count.
    assert round(m["raw_partition_violation_rate"]["mean"], 4) == round(1 / 6, 4)


def test_raw_partition_violation_rate_aggregated_per_repeat():
    # Regression: this PRIMARY metric used to be pooled globally, reaching
    # aggregate_repeats as a single value (n=1, stdev=0) and defeating the
    # K-repeat CI-lower-bound gate. Differing per-repeat rates must surface
    # as n == K with real variance.
    snap = {
        "repeats": 2,
        "clusters": [
            {
                "cluster_id": "c_a",
                "total_members": 2,
                "sample_coverage": 1.0,
                "repeats": [
                    {"raw_partition_ok": True, "residual_ids": [], "groups": []},
                    {"raw_partition_ok": False, "residual_ids": [], "groups": []},
                ],
            },
            {
                "cluster_id": "c_b",
                "total_members": 2,
                "sample_coverage": 1.0,
                "repeats": [
                    {"raw_partition_ok": True, "residual_ids": [], "groups": []},
                    {"raw_partition_ok": True, "residual_ids": [], "groups": []},
                ],
            },
        ],
    }
    agg = compute_metrics(snap)["raw_partition_violation_rate"]
    # repeat 0: 0/2 violations => 0.0; repeat 1: 1/2 => 0.5.
    assert agg["n"] == 2
    assert agg["mean"] == 0.25
    assert agg["stdev"] == 0.25  # pstdev([0.0, 0.5]) -- real variance, not 0


def test_root_distribution_descriptive():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # Per repeat, chain[-1] over emitted groups: c_boat->entity, c_aircraft->entity.
    # root_distribution is a descriptive dict of mean counts, NOT an accuracy.
    rd = m["root_distribution"]
    assert rd["entity"] == 2.0
    assert "concept" not in rd or rd.get("concept", 0.0) == 0.0


def test_hallucinated_target_rate_zero_in_fixture():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # No group has a reuse_target/graft_parent uuid that is null while branch
    # claims a target => 0.
    assert m["hallucinated_target_rate"]["mean"] == 0.0


def test_roots_and_staleness_passthrough():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    assert m["roots_present_in_live_catalog"] is True
    assert m["live_redis_catalog_staleness"] == 3


# ----- new_floated_at_root (SPEC \u00a77.3(d)) ---------------------------------

def test_new_floated_at_root_zero_in_fixture():
    snap = _load("run_synthetic.json")
    m = compute_metrics(snap)
    # The only NEW group (c_aircraft) is grafted in both repeats => no NEW
    # group floats flat at a root.
    assert m["new_floated_at_root"]["mean"] == 0.0


def test_new_floated_at_root_counts_ungrafted_new_groups():
    # SPEC SS7.3(d): grafted iff the resolved graft parent is NOT a root;
    # new_floated_at_root counts NEW groups whose resolved parent IS a root
    # (the complement: total_new - grafted).
    snap = {
        "repeats": 1,
        "clusters": [
            {
                "cluster_id": "c_mixed",
                "total_members": 4,
                "sample_coverage": 1.0,
                "repeats": [
                    {
                        "raw_partition_ok": True,
                        "residual_ids": [],
                        "groups": [
                            {
                                "assign_to": "NEW",
                                "is_grafted": True,
                                "graft_parent_uuid": "uuid-vehicle",
                                "covering_depth": 2,
                                "chain": ["car", "vehicle", "entity"],
                            },
                            {
                                "assign_to": "NEW",
                                "is_grafted": False,
                                "graft_parent_uuid": None,
                                "chain": ["widget", "entity"],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    m = compute_metrics(snap)
    assert m["new_floated_at_root"]["mean"] == 1.0
    assert m["graft_depth_fraction"]["mean"] == 0.5


# ----- ablation_deltas ---------------------------------------------------

def test_ablation_deltas_noise_band_gate():
    arms = _load("ablation_arms.json")
    out = ablation_deltas(arms)
    # graft_depth_fraction: A6 0.70 vs A1 0.10 => delta 0.60, band max(0.03,0.02)=0.03 => passes.
    gd = out["graft_depth_fraction"]["full_vs_naive_llm"]
    assert round(gd["delta"], 4) == 0.60
    assert gd["passes"] is True
    # residual_fraction: A6 0.28 vs A1 0.30 => delta -0.02, band max(0.04,0.05)=0.05 => does NOT pass.
    rf = out["residual_fraction"]["full_vs_naive_llm"]
    assert round(rf["delta"], 4) == -0.02
    assert rf["passes"] is False


def test_ablation_criterion_metrics_flatten_to_gateable_keys():
    arms = _load("ablation_arms.json")
    flat = ablation_criterion_metrics(arms)
    # graft_depth_fraction: delta 0.60, band 0.03 => margin 0.57 > 0
    # (A6 beats A1 beyond the noise band; gates with comparator gt 0).
    assert round(flat["ablation_A6_beats_A1_graft_depth_fraction"], 4) == 0.57
    assert flat["ablation_A6_beats_A1_graft_depth_fraction"] > 0
    # residual_fraction: delta -0.02, band 0.05 => margin -0.07 (no beat).
    assert round(flat["ablation_A6_beats_A1_residual_fraction"], 4) == -0.07
    assert flat["ablation_A6_beats_A1_residual_fraction"] <= 0


# ----- goal.yaml <-> emitted metrics coupling ----------------------------

def test_goal_yaml_criteria_resolve_against_emitted_metrics():
    # Kill the class of dangling criteria: every success_criteria metric name
    # in goal.yaml must resolve against the union of compute_metrics output
    # and the flattened ablation criterion keys, both produced from the
    # synthetic fixtures. A criterion naming a key no code path emits (the
    # old ablation_A6_beats_A1_graft_depth) fails here instead of being
    # silently skipped by a name-keyed gate.
    goal_path = Path(__file__).resolve().parent.parent / "goal.yaml"
    goal = yaml.safe_load(goal_path.read_text(encoding="utf-8"))
    criteria = goal["success_criteria"]
    assert criteria, "goal.yaml defines no success_criteria"
    produced = set(compute_metrics(_load("run_synthetic.json"))) | set(
        ablation_criterion_metrics(_load("ablation_arms.json"))
    )
    missing = [c["metric"] for c in criteria if c["metric"] not in produced]
    assert not missing, f"goal.yaml criteria with no emitted metric key: {missing}"


# ----- emit + eyeball dump -----------------------------------------------

def test_emit_metric_lines(capsys):
    snap = _load("run_synthetic.json")
    emit(compute_metrics(snap))
    out = capsys.readouterr().out
    assert "[METRIC] graft_depth_fraction.mean=1.0" in out
    assert "[METRIC] reuse_collapses.mean=1.0" in out
    assert "[METRIC] canonical_merge_collapses.mean=0.0" in out
    assert "[METRIC] residual_fraction.mean=0.2778" in out
    assert "[METRIC] raw_partition_violation_rate.mean=0.1667" in out
    assert "[METRIC] residual_bloat=False" in out
    assert "[METRIC] roots_present_in_live_catalog=True" in out
    assert "[METRIC] live_redis_catalog_staleness=3" in out


def test_eyeball_dump_human_readable():
    snap = _load("run_synthetic.json")
    dump = eyeball_dump(snap)
    # Every cluster appears with its per-repeat decisions; residual/eviction
    # parking is visible.
    assert "c_boat" in dump
    assert "c_aircraft" in dump
    assert "c_shallow" in dump
    assert "G1_reuse" in dump
    assert "G2_graft" in dump
    assert "residual" in dump
    assert "evicted" in dump
