"""Label-free structural metrics for the naming-driven-typing (v2) experiment.

Reads ONE run snapshot written by ``harness/run_experiment.py`` (containing all
K repeats per cluster) and emits intrinsic STRUCTURAL metrics as
``[METRIC] key=value`` lines. Mirrors the edge-embeddings-worth-it precedent
(``compute_metrics(snapshot)`` -> ``emit``) but is **label-free**
(SHARED context / SPEC SS0, Chris 2026-06-05): NO per-cluster coherence labels,
NO root ground-truth, NO external hypernymy oracle, NO label-derived
precision/recall/accuracy. Supersedes SPEC SS7.1/SS7.2 and every
label-dependent metric in SS7.3/SS7.5.

Metrics (all aggregated mean +/- stdev over the K repeats; cv emitted):
  graft_depth_fraction        -- per repeat, grafted NEW groups / total NEW groups.
  mean_graft_depth            -- mean covering_depth of grafted NEW groups.
  new_floated_at_root         -- count of NEW groups NOT grafted (floated flat at
                                 a root). SPEC SS7.3(d).
  reuse_collapses             -- count of SEMANTIC reuse groups (is_reuse, assign_to
                                 a non-name-identical published type). The thesis signal.
  canonical_merge_collapses   -- count of string-equality intra-response merges
                                 (canonical_merged_into set). Reported SEPARATELY,
                                 NOT a semantic claim.
  residual_fraction           -- residual member ids / total members.
  raw_partition_violation_rate-- per repeat, fraction of clusters with
                                 raw_partition_ok False. The RAW LLM fidelity signal.
  hallucinated_target_rate    -- fraction of reuse/graft groups whose claimed target
                                 uuid is null/unresolved.
  placement_conflict_rate     -- fraction of grafted groups whose v2 parent differs
                                 from what rollup would pick (recorded by harness; 0 if
                                 the snapshot did not measure it).
  root_distribution           -- DESCRIPTIVE mean count of groups per terminal root.
  residual_bloat              -- bool: residual_fraction mean > 0.4.

Scalar passthroughs (validity guards): roots_present_in_live_catalog,
live_redis_catalog_staleness, sample_coverage_min, stability_cv_max.

Ablation gating (SPEC SS7.4): ``ablation_deltas`` computes full-v2-vs-arm
deltas with a K-repeat noise band; ``ablation_criterion_metrics`` flattens
them into goal.yaml-gateable scalar keys
(``ablation_A6_beats_<arm>_<metric>`` = delta - noise_band) so EVERY
goal.yaml success criterion resolves by metric name against emitted keys.

Metrics consume the snapshot dict ONLY -- no harness or hermes imports
(reuse/merge flags are pre-computed by the harness into the snapshot).

Usable as a library (``compute_metrics(snapshot)``) or a CLI
(``python eval/metrics.py workspace/run_<ts>.json [--eyeball]``).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_RESIDUAL_BLOAT_THRESHOLD = 0.4


def aggregate_repeats(values: list[float]) -> dict[str, float]:
    """Aggregate per-repeat scalars to mean/stdev/cv/ci (population stdev).

    cv = stdev / |mean|, defined as 0.0 when mean == 0 (avoids div-by-zero and
    a spurious "noisy" verdict on an all-zero metric). ci is the simple
    mean +/- stdev band used as the K-repeat noise band.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "stdev": 0.0, "cv": 0.0, "n": 0, "ci_lo": 0.0, "ci_hi": 0.0}
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if n > 1 else 0.0
    cv = (stdev / abs(mean)) if mean != 0 else 0.0
    return {
        "mean": mean,
        "stdev": stdev,
        "cv": cv,
        "n": n,
        "ci_lo": mean - stdev,
        "ci_hi": mean + stdev,
    }


def ablation_deltas(
    by_arm: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Compute full-v2-vs-each-arm deltas with a K-repeat noise band gate.

    ``by_arm`` maps arm_name -> {metric -> aggregate dict (mean/stdev/...)}.
    The full-v2 arm is the one keyed ``"full"`` (A6). For every other arm and
    every metric shared with ``full``, returns delta = mean(full) - mean(arm),
    noise_band = max(stdev(full), stdev(arm)), and passes = |delta| > band.
    A comparison only counts as a real lift when it clears the noise band.
    """
    if "full" not in by_arm:
        return {}
    full = by_arm["full"]
    out: dict[str, Any] = {}
    for metric, full_agg in full.items():
        out[metric] = {}
        for arm, arm_metrics in by_arm.items():
            if arm == "full" or metric not in arm_metrics:
                continue
            arm_agg = arm_metrics[metric]
            delta = full_agg["mean"] - arm_agg["mean"]
            band = max(full_agg.get("stdev", 0.0), arm_agg.get("stdev", 0.0))
            out[metric][f"full_vs_{arm}"] = {
                "delta": delta,
                "noise_band": band,
                "passes": abs(delta) > band,
            }
    return out


# SPEC SS7.4 ablation arm identifiers, keyed by harness arm name. ``full`` is
# A6 (full v2); every delta is full-vs-arm (see ``ablation_deltas``).
_ABLATION_ARM_IDS = {
    "clustering_baseline": "A0",
    "naive_llm": "A1",
    "no_reuse": "A2",
    "no_graft": "A3",
    "no_chain": "A4",
    "no_gate": "A5",
    "full": "A6",
}


def ablation_criterion_metrics(
    by_arm: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Flatten ``ablation_deltas`` output into goal.yaml-gateable scalar keys.

    For every (metric, arm) comparison emits
    ``ablation_A6_beats_<armId>_<metric>`` = delta - noise_band: positive iff
    full-v2 (A6) beats that arm on that metric by MORE than the K-repeat
    noise band (SPEC SS7.4). goal.yaml criteria (threshold 0, comparator gt)
    reference these keys by name, so a name-keyed gate resolves them directly
    instead of digging into the nested ``ablation_deltas`` structure. The key
    is derived mechanically from the emitted metric name (no alias table for
    metrics), so criterion keys cannot drift from real output.

    Arm names without a SPEC id fall back to the raw arm name in the key.
    """
    out: dict[str, float] = {}
    for metric, comparisons in ablation_deltas(by_arm).items():
        for compare_key, stats in comparisons.items():
            arm = compare_key.removeprefix("full_vs_")
            arm_id = _ABLATION_ARM_IDS.get(arm, arm)
            out[f"ablation_A6_beats_{arm_id}_{metric}"] = (
                stats["delta"] - stats["noise_band"]
            )
    return out


def _root_of(chain: list[str]) -> str:
    """Terminal root of a chain; default ``entity`` on empty/malformed chains."""
    return chain[-1] if chain else "entity"


def compute_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compute label-free structural metrics from ONE run snapshot.

    The snapshot (written by harness/run_experiment.py) holds K repeats per
    cluster. Each per-repeat structural scalar is computed across all clusters,
    then aggregated across the K repeats via ``aggregate_repeats``. No labels,
    no ground truth, no oracle.
    """
    clusters: list[dict[str, Any]] = snapshot.get("clusters", [])
    repeats_k = int(snapshot.get("repeats", 0))
    total_members = sum(int(c.get("total_members", 0)) for c in clusters) or 1

    # Per-repeat accumulators (index r over the K repeats).
    graft_frac_per_r: list[float] = []
    graft_depth_per_r: list[float] = []
    floated_per_r: list[float] = []
    reuse_per_r: list[float] = []
    canon_per_r: list[float] = []
    residual_frac_per_r: list[float] = []
    halluc_per_r: list[float] = []
    conflict_per_r: list[float] = []
    root_counts_per_r: list[Counter] = []

    # raw-partition violations are collected per repeat so the violation rate
    # aggregates over the K repeats like every other metric.
    raw_ok_cells: list[list[bool]] = [[] for _ in range(repeats_k)]
    sample_coverages: list[float] = [
        float(c.get("sample_coverage", 1.0)) for c in clusters
    ]

    for r in range(repeats_k):
        new_groups = 0
        grafted = 0
        graft_depths: list[int] = []
        reuse_count = 0
        canon_count = 0
        residual_ids: set[str] = set()
        target_claims = 0
        hallucinated = 0
        conflict_groups = 0
        conflict_total = 0
        roots: Counter = Counter()

        for cl in clusters:
            rep_list = cl.get("repeats", [])
            if r >= len(rep_list):
                continue
            rep = rep_list[r]
            raw_ok_cells[r].append(bool(rep.get("raw_partition_ok", True)))
            residual_ids.update(rep.get("residual_ids", []))

            for g in rep.get("groups", []):
                roots[_root_of(g.get("chain", []))] += 1

                if g.get("canonical_merged_into"):
                    canon_count += 1

                if g.get("is_reuse"):
                    reuse_count += 1
                    target_claims += 1
                    if not g.get("reuse_target_uuid"):
                        hallucinated += 1

                if g.get("assign_to") == "NEW":
                    new_groups += 1
                    if g.get("is_grafted"):
                        grafted += 1
                        graft_depths.append(int(g.get("covering_depth", 0)))
                        target_claims += 1
                        if not g.get("graft_parent_uuid"):
                            hallucinated += 1
                        conflict_total += 1
                        if g.get("placement_conflict"):
                            conflict_groups += 1

        graft_frac_per_r.append(grafted / new_groups if new_groups else 0.0)
        graft_depth_per_r.append(
            statistics.fmean(graft_depths) if graft_depths else 0.0
        )
        floated_per_r.append(float(new_groups - grafted))
        reuse_per_r.append(float(reuse_count))
        canon_per_r.append(float(canon_count))
        residual_frac_per_r.append(len(residual_ids) / total_members)
        halluc_per_r.append(hallucinated / target_claims if target_claims else 0.0)
        conflict_per_r.append(
            conflict_groups / conflict_total if conflict_total else 0.0
        )
        root_counts_per_r.append(roots)

    raw_violation_per_r = [
        sum(1 for ok in cells if not ok) / len(cells) if cells else 0.0
        for cells in raw_ok_cells
    ]

    # Descriptive root distribution: mean count per terminal root over repeats.
    root_sums: dict[str, float] = defaultdict(float)
    for roots in root_counts_per_r:
        for root, count in roots.items():
            root_sums[root] += count
    denom = len(root_counts_per_r) or 1
    root_distribution = {root: total / denom for root, total in root_sums.items()}

    residual_frac_agg = aggregate_repeats(residual_frac_per_r)

    metrics: dict[str, Any] = {
        "graft_depth_fraction": aggregate_repeats(graft_frac_per_r),
        "mean_graft_depth": aggregate_repeats(graft_depth_per_r),
        "new_floated_at_root": aggregate_repeats(floated_per_r),
        "reuse_collapses": aggregate_repeats(reuse_per_r),
        "canonical_merge_collapses": aggregate_repeats(canon_per_r),
        "residual_fraction": residual_frac_agg,
        "raw_partition_violation_rate": aggregate_repeats(raw_violation_per_r),
        "hallucinated_target_rate": aggregate_repeats(halluc_per_r),
        "placement_conflict_rate": aggregate_repeats(conflict_per_r),
        "root_distribution": root_distribution,
        "residual_bloat": residual_frac_agg["mean"] > _RESIDUAL_BLOAT_THRESHOLD,
        "roots_present_in_live_catalog": bool(
            snapshot.get("roots_present_in_live_catalog", False)
        ),
        "live_redis_catalog_staleness": int(
            snapshot.get("live_redis_catalog_staleness", 0)
        ),
        "sample_coverage_min": min(sample_coverages) if sample_coverages else 1.0,
    }

    # stability_cv_max: worst cv across the aggregated (dict-valued) metrics.
    cvs = [v["cv"] for v in metrics.values() if isinstance(v, dict) and "cv" in v]
    metrics["stability_cv_max"] = max(cvs) if cvs else 0.0
    return metrics


def eyeball_dump(snapshot: dict[str, Any]) -> str:
    """Render a human-readable per-cluster decision dump for eyeballing.

    One block per cluster; one line per (repeat, group) decision plus the
    residual / evicted ids and the raw-partition flag. No judgment is applied
    -- this is the label-free look-at-what-it-actually-did artifact (SPEC SS0).
    """
    rid = snapshot.get("request_id", "?")
    model = snapshot.get("model", "?")
    ablation = snapshot.get("ablation", "?")
    repeats = snapshot.get("repeats", "?")
    lines: list[str] = [
        f"run={rid} model={model} ablation={ablation} repeats={repeats}"
    ]
    for cl in snapshot.get("clusters", []):
        cid = cl.get("cluster_id", "?")
        cname = cl.get("current_name", "?")
        total = cl.get("total_members", "?")
        coverage = cl.get("sample_coverage", "?")
        lines.append("")
        lines.append(
            f"cluster {cid} (current_name={cname!r}, members={total}, "
            f"coverage={coverage})"
        )
        for r, rep in enumerate(cl.get("repeats", [])):
            ok = "ok" if rep.get("raw_partition_ok", True) else "VIOLATION"
            lines.append(f"  repeat {r} [raw_partition={ok}]")
            for g in rep.get("groups", []):
                chain = " -> ".join(g.get("chain", []))
                branch = g.get("branch", "?")
                name = g.get("name", "?")
                assign_to = g.get("assign_to", "?")
                target = (
                    g.get("reuse_target_uuid")
                    or g.get("graft_parent_name")
                    or "-"
                )
                depth = g.get("covering_depth", "?")
                members = g.get("member_ids", [])
                lines.append(
                    f"    {branch} name={name!r} assign_to={assign_to} "
                    f"target={target} depth={depth} chain[{chain}] "
                    f"members={members}"
                )
            residual = rep.get("residual_ids", [])
            evicted = rep.get("evicted_ids", [])
            if residual:
                lines.append(f"    residual: {residual}")
            if evicted:
                lines.append(f"    evicted:  {evicted}")
    return "\n".join(lines)


def emit(metrics: dict[str, Any]) -> None:
    """Print metrics as ``[METRIC] key=value`` lines.

    Aggregate (dict) metrics emit one line per sub-key (mean/stdev/cv);
    scalar metrics emit a single line. Mirrors the edge-embeddings precedent.
    """
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value:
            for sub in ("mean", "stdev", "cv"):
                print(f"[METRIC] {key}.{sub}={round(value[sub], 4)}")
        elif isinstance(value, dict):
            for sub, sub_v in value.items():
                print(f"[METRIC] {key}.{sub}={round(sub_v, 4)}")
        else:
            print(f"[METRIC] {key}={value}")


def main() -> None:
    argv = sys.argv[1:]
    show_eyeball = "--eyeball" in argv
    args = [a for a in argv if a != "--eyeball"]
    if args:
        path = Path(args[0])
    else:
        ws = Path(__file__).resolve().parent.parent / "workspace"
        snaps = sorted(ws.glob("run_*.json"))
        if not snaps:
            print("[METRIC] error=no_snapshots_found")
            return
        path = snaps[-1]
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    emit(compute_metrics(snapshot))
    if show_eyeball:
        print(eyeball_dump(snapshot))


if __name__ == "__main__":
    main()
