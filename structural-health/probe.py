"""W0.1 structural-health probe (logos-experiments#25, epic logos#557).

Re-runs the 2026-06-04 latent-structure probe on the current (post-reset)
graph: build the node x (relation, direction, neighbor-type) matrix over the
HCG's semantic edges, TF-IDF it, and measure how much variance truncated SVD
captures. Plus the source-noise indicators the 2026-06-04 diagnosis used:
edges per data node and the df=1 predicate fraction.

Baselines to beat (2026-06-04, pre-reset graph): top-128 = 5.7% variance
(top-16: 1.2%), ~2.6 edges/node, rarest-predicate df=1 junk dominant.
See BASELINE.md; gate thresholds live there too.

The probe is read-only. Connection follows the harness convention
(a0_baseline.py): NEO4J_URI/NEO4J_USER default, NEO4J_PASSWORD required
explicitly -- fail loud, no default credential.

Current graph model (verified 2026-06-10): semantic edges are reified as
Node{type:'edge'} with relation/source/target properties; a data node's
asserted type is the NAME of its IS_A edge's target (a type_definition
node), falling back to its realm kind (n.type) when untyped.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix

# Node kinds that are typing/plumbing infrastructure, not data rows.
NON_DATA_KINDS = {"edge", "type_definition"}
# Relations that assert typing; excluded in the "non-typing" matrix variant
# so the signature health number isn't propped up by the typing edges
# themselves.
TYPING_RELATIONS = {"IS_A", "INSTANCE_OF", "SUBTYPE_OF"}


@dataclass(frozen=True)
class NodeRecord:
    uuid: str
    kind: str  # n.type: entity / concept / process / type_definition / edge ...
    name: str


@dataclass(frozen=True)
class SemanticEdge:
    relation: str
    source: str  # uuid of source node
    target: str  # uuid of target node


def asserted_type_map(
    nodes: list[NodeRecord], edges: list[SemanticEdge]
) -> dict[str, str]:
    """uuid -> asserted type name.

    Data node with an IS_A edge -> the target type_definition's name;
    otherwise the node's own kind (realm fallback). type_definition and
    edge nodes map to their kind.
    """
    by_uuid = {n.uuid: n for n in nodes}
    out = {n.uuid: n.kind for n in nodes}
    candidates: dict[str, list[str]] = {}
    for e in edges:
        if e.relation != "IS_A":
            continue
        src = by_uuid.get(e.source)
        tgt = by_uuid.get(e.target)
        if src is None or tgt is None or src.kind in NON_DATA_KINDS:
            continue
        if tgt.kind == "type_definition" and tgt.name:
            candidates.setdefault(e.source, []).append(tgt.name)
    # A node can carry multiple IS_A edges; Neo4j returns them in internal
    # order, so pick deterministically (lexicographic min) to keep runs
    # reproducible.
    for uuid, names in candidates.items():
        out[uuid] = min(names)
    return out


def build_matrix(
    nodes: list[NodeRecord],
    edges: list[SemanticEdge],
    exclude_relations: set[str] | None = None,
) -> tuple[csr_matrix, list[str], list[tuple[str, str, str]]]:
    """node x (relation, direction, neighbor-type) count matrix.

    Rows: data nodes only (kinds outside NON_DATA_KINDS). Each semantic
    edge (rel, src, tgt) contributes (rel, 'out', type(tgt)) to src and
    (rel, 'in', type(src)) to tgt. Returns (matrix, row uuids, feature
    names).
    """
    exclude = exclude_relations or set()
    types = asserted_type_map(nodes, edges)
    data_ids = [n.uuid for n in nodes if n.kind not in NON_DATA_KINDS]
    row_index = {u: i for i, u in enumerate(data_ids)}

    cells: Counter[tuple[int, tuple[str, str, str]]] = Counter()
    for e in edges:
        if e.relation in exclude:
            continue
        if e.source in row_index and e.target in types:
            cells[(row_index[e.source], (e.relation, "out", types[e.target]))] += 1
        if e.target in row_index and e.source in types:
            cells[(row_index[e.target], (e.relation, "in", types[e.source]))] += 1

    feat_names = sorted({feat for (_, feat) in cells})
    feat_index = {f: j for j, f in enumerate(feat_names)}
    rows, cols, vals = [], [], []
    for (i, feat), count in cells.items():
        rows.append(i)
        cols.append(feat_index[feat])
        vals.append(count)
    mat = csr_matrix(
        (vals, (rows, cols)), shape=(len(data_ids), len(feat_names)), dtype=np.float64
    )
    return mat, data_ids, feat_names


def predicate_stats(
    nodes: list[NodeRecord], edges: list[SemanticEdge]
) -> dict:
    """Source-noise indicators: relation document frequencies + density."""
    edge_df = Counter(e.relation for e in edges)
    distinct = len(edge_df)
    df1 = sum(1 for c in edge_df.values() if c == 1)
    data_nodes = sum(1 for n in nodes if n.kind not in NON_DATA_KINDS)
    return {
        "distinct_relations": distinct,
        "edge_df": dict(edge_df),
        "df1_fraction": (df1 / distinct) if distinct else 0.0,
        "edges_per_data_node": (len(edges) / data_nodes) if data_nodes else 0.0,
        "data_nodes": data_nodes,
        "semantic_edges": len(edges),
    }


def variance_curve(mat: csr_matrix, ks: tuple[int, ...] = (16, 64, 128)) -> dict:
    """Cumulative explained-variance ratio of TF-IDF'd matrix at each k.

    k is clamped to min(shape) - 1 (TruncatedSVD limit); one decomposition
    at max k, partial sums for the smaller ks.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    if min(mat.shape) <= 1:
        return {k: 0.0 for k in ks}
    tfidf = TfidfTransformer().fit_transform(mat)
    max_rank = min(tfidf.shape) - 1
    k_fit = min(max(ks), max_rank)
    svd = TruncatedSVD(n_components=k_fit, random_state=0)
    svd.fit(tfidf)
    ratios = svd.explained_variance_ratio_
    return {k: float(ratios[: min(k, k_fit)].sum()) for k in ks}


# ---------------------------------------------------------------- live fetch


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit(
            "NEO4J_PASSWORD must be set explicitly for the structural-health "
            "probe (no default credential; harness convention)."
        )
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_graph(driver) -> tuple[list[NodeRecord], list[SemanticEdge]]:
    def work(tx):
        nodes = [
            NodeRecord(r["uuid"], r["kind"], r["name"] or "")
            for r in tx.run(
                "MATCH (n:Node) WHERE n.type <> 'edge' "
                "RETURN n.uuid AS uuid, n.type AS kind, n.name AS name"
            )
            if r["uuid"] and r["kind"]
        ]
        raw_edges = [
            (r["rel"], r["src"], r["tgt"])
            for r in tx.run(
                "MATCH (e:Node {type:'edge'}) "
                "RETURN e.relation AS rel, e.source AS src, e.target AS tgt"
            )
        ]
        return nodes, raw_edges

    with driver.session() as s:
        nodes, raw_edges = s.execute_read(work)
    edges = [SemanticEdge(*t) for t in raw_edges if all(t)]
    dropped = len(raw_edges) - len(edges)
    if dropped:
        # Malformed edges are themselves a data-quality signal for this probe.
        print(
            f"warn: dropped {dropped} edge node(s) with missing "
            "relation/source/target",
            file=sys.stderr,
        )
    return nodes, edges


def report(nodes: list[NodeRecord], edges: list[SemanticEdge]) -> str:
    stats = predicate_stats(nodes, edges)
    mat_all, _, feats_all = build_matrix(nodes, edges)
    mat_nt, _, feats_nt = build_matrix(nodes, edges, exclude_relations=TYPING_RELATIONS)
    curve_all = variance_curve(mat_all)
    curve_nt = variance_curve(mat_nt)

    df_sorted = sorted(stats["edge_df"].items(), key=lambda kv: -kv[1])
    kinds = Counter(n.kind for n in nodes)
    typed = asserted_type_map(nodes, edges)
    n_typed = sum(
        1
        for n in nodes
        if n.kind not in NON_DATA_KINDS and typed[n.uuid] != n.kind
    )
    is_a_targets: dict[str, set[str]] = {}
    for e in edges:
        if e.relation == "IS_A":
            is_a_targets.setdefault(e.source, set()).add(e.target)
    n_multi = sum(1 for tgts in is_a_targets.values() if len(tgts) > 1)

    lines = [
        f"structural-health probe -- {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"nodes (non-edge): {sum(kinds.values())}  by kind: {dict(kinds)}",
        f"data nodes: {stats['data_nodes']}  (IS_A-typed: {n_typed},"
        f" multi-IS_A: {n_multi})",
        f"semantic edges: {stats['semantic_edges']}"
        f"  edges/data-node: {stats['edges_per_data_node']:.2f}",
        f"distinct relations: {stats['distinct_relations']}"
        f"  df=1 fraction: {stats['df1_fraction']:.3f}",
        f"top relations: {df_sorted[:10]}",
        f"bottom relations: {df_sorted[-5:]}",
        "",
        f"matrix (all relations): {mat_all.shape[0]} x {len(feats_all)}"
        f"  nnz={mat_all.nnz}",
        f"  explained variance: k=16: {curve_all[16]:.3f}"
        f"  k=64: {curve_all[64]:.3f}  k=128: {curve_all[128]:.3f}",
        f"matrix (non-typing relations): {mat_nt.shape[0]} x {len(feats_nt)}"
        f"  nnz={mat_nt.nnz}",
        f"  explained variance: k=16: {curve_nt[16]:.3f}"
        f"  k=64: {curve_nt[64]:.3f}  k=128: {curve_nt[128]:.3f}",
        "",
        "baseline row (append to BASELINE.md):",
        f"| {datetime.now(timezone.utc).date().isoformat()} "
        f"| {stats['data_nodes']} | {stats['semantic_edges']} "
        f"| {stats['edges_per_data_node']:.2f} "
        f"| {stats['distinct_relations']} | {stats['df1_fraction']:.3f} "
        f"| {curve_all[16]:.3f} / {curve_all[128]:.3f} "
        f"| {curve_nt[16]:.3f} / {curve_nt[128]:.3f} |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    driver = _driver()
    try:
        nodes, edges = fetch_graph(driver)
    finally:
        driver.close()
    print(report(nodes, edges))


if __name__ == "__main__":
    main()
