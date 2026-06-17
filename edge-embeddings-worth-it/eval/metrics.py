"""Metric computation for the edge-embeddings-worth-it experiment.

Reads the per-round JSON snapshots written by the harness into ``workspace/``
and emits the four success-criterion metrics as ``[METRIC] key=value`` lines.

Metrics:
  node_type_ari            -- adjusted Rand index between the top-level HIERARCHY-ROOT
                              assignment and the provenance domain labels (final round).
                              Scored at hierarchy granularity (not the over-fragmented
                              flat clusters). node_flat_clusters is emitted for reference.
  classification_coverage  -- fraction of entity-kind nodes that landed in SOME
                              emergent cluster (vs. ungrouped junk-drawer residue).
  edge_types_formed        -- number of edge clusters with size >= min_cluster_size.
  edge_type_merge_ratio    -- (# distinct semantic relation labels) / (# resulting
                              edge clusters). >1 => clustering MERGED synonymous
                              relations (the real "worth it" signal); ~1 => it just
                              recovered the labels. THE headline edge metric.
  edge_cluster_coherence   -- DEMOTED. Mean within-cluster relation-label purity.
                              Near-tautological: same-label edges embed to the
                              IDENTICAL "RELATIONSHIP:<label>" vector, so they
                              cluster together by construction. Kept for continuity
                              but NOT a "worth it" signal -- prefer edge_type_merge_ratio.

Usable as a library (``compute_metrics(snapshot)``) or a CLI
(``python eval/metrics.py workspace/round_4.json``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _comb2(x: float) -> float:
    return x * (x - 1) / 2.0


def adjusted_rand_index(true: list, pred: list) -> float:
    """ARI between two labelings (pure-Python; mirrors run_cluster_sweep)."""
    if not true:
        return 0.0
    classes = sorted(set(true))
    clusters = sorted(set(pred))
    ci = {c: i for i, c in enumerate(classes)}
    ki = {c: i for i, c in enumerate(clusters)}
    cont = [[0 for _ in clusters] for _ in classes]
    for t, p in zip(true, pred):
        cont[ci[t]][ki[p]] += 1
    sum_comb = sum(_comb2(v) for row in cont for v in row)
    a = [sum(row) for row in cont]
    b = [sum(cont[r][c] for r in range(len(classes))) for c in range(len(clusters))]
    sa = sum(_comb2(v) for v in a)
    sb = sum(_comb2(v) for v in b)
    total = _comb2(len(true))
    exp = sa * sb / total if total else 0.0
    mx = (sa + sb) / 2
    return 0.0 if mx == exp else (sum_comb - exp) / (mx - exp)


def purity(true: list, pred: list) -> float:
    """Fraction of items in the majority class of their predicted cluster."""
    if not true:
        return 0.0
    groups: dict = {}
    for t, p in zip(true, pred):
        groups.setdefault(p, []).append(t)
    return sum(Counter(v).most_common(1)[0][1] for v in groups.values()) / len(true)


def compute_metrics(snapshot: dict[str, Any]) -> dict[str, float]:
    """Compute the four success metrics from a single round snapshot.

    The snapshot schema (written by the harness):
      entity_domains: {uuid: domain}                  -- provenance ground truth
      node_clusters:  [{"label": int, "members": [{uuid,name}]}]
      min_cluster_size: int
      edge_clusters:  [{"label": int, "edges": [{uuid,relation,...}]}]
    """
    ms = int(snapshot.get("min_cluster_size", 2))
    entity_domains: dict[str, str] = snapshot.get("entity_domains", {})

    # --- node_type_ari (HIERARCHY granularity) + classification_coverage ---
    # node_type_ari is scored against the top-level HIERARCHY-ROOT assignment,
    # NOT the over-fragmented flat clusters: the flat clustering splits each
    # domain into many tiny groups, which deflates ARI against the coarse
    # provenance domain labels. The harness emits {uuid: root_idx} in
    # ``node_hierarchy_assignment`` (every leaf under a root maps to that root).
    # Entities not under ANY root are treated as their own singleton cluster.
    node_clusters = snapshot.get("node_clusters", [])
    flat_clusters = len(node_clusters)
    clustered_uuids: set[str] = set()
    for cl in node_clusters:
        for m in cl["members"]:
            clustered_uuids.add(m["uuid"])

    hierarchy_assignment: dict[str, int] = snapshot.get(
        "node_hierarchy_assignment", {}
    )
    true_labels: list[str] = []
    pred_labels: list[int] = []
    _singleton = -1
    for u, domain in entity_domains.items():
        true_labels.append(domain)
        if u in hierarchy_assignment:
            pred_labels.append(hierarchy_assignment[u])
        else:
            pred_labels.append(_singleton)
            _singleton -= 1
    node_type_ari = adjusted_rand_index(true_labels, pred_labels)

    total_entities = len(entity_domains) or 1
    covered = len([u for u in entity_domains if u in clustered_uuids])
    classification_coverage = covered / total_entities

    # --- edge_types_formed + edge_cluster_coherence ---
    edge_clusters = snapshot.get("edge_clusters", [])
    sized = [c for c in edge_clusters if len(c.get("edges", [])) >= ms]
    edge_types_formed = len(sized)

    e_true: list[str] = []
    e_pred: list[int] = []
    for cl in sized:
        for e in cl["edges"]:
            e_true.append(e.get("relation", ""))
            e_pred.append(cl["label"])
    # DEMOTED metric: near-tautological. Same-label edges embed to the identical
    # "RELATIONSHIP:<label>" vector, so within-cluster relation purity is high by
    # construction. Reported for continuity only; not the "worth it" signal.
    edge_cluster_coherence = purity(e_true, e_pred) if e_true else 0.0

    # HEADLINE edge metric. distinct semantic relation labels / resulting edge
    # clusters. >1 => synonymous relations were MERGED into one edge-type (what
    # embedding edges would actually buy us); ~1 => clustering merely recovered
    # the existing labels (no merge => embedding edges adds nothing).
    distinct_relation_labels = len(
        {e.get("relation", "") for cl in sized for e in cl["edges"]}
    )
    n_edge_clusters = len(sized)
    edge_type_merge_ratio = (
        distinct_relation_labels / n_edge_clusters if n_edge_clusters else 0.0
    )

    return {
        "node_type_ari": round(node_type_ari, 4),
        # flat-cluster count emitted for reference only (NOT the ARI basis).
        "node_flat_clusters": flat_clusters,
        "classification_coverage": round(classification_coverage, 4),
        "edge_types_formed": edge_types_formed,
        "edge_type_merge_ratio": round(edge_type_merge_ratio, 4),
        "edge_cluster_coherence": round(edge_cluster_coherence, 4),
    }


def emit(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        print(f"[METRIC] {key}={value}")


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        ws = Path(__file__).resolve().parent.parent / "workspace"
        snaps = sorted(ws.glob("round_*.json"))
        if not snaps:
            print("[METRIC] error=no_snapshots_found")
            return
        path = snaps[-1]
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    emit(compute_metrics(snapshot))


if __name__ == "__main__":
    main()
