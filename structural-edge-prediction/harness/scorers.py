"""Candidate-edge scorers \u2014 one per experiment arm.

Each scorer takes a candidate relational edge ``(src, rel, dst)`` plus a
``TrainGraph`` (the training-only view of the snapshot) and returns a
real-valued plausibility score \u2014 higher = more plausible. A ``None`` return
means the arm is unavailable for this snapshot (e.g. A2 with no embeddings);
the runner skips it.

Arms:
  * A0 ``score_marginal`` \u2014 type-BLIND global frequency of (rel, dst_type).
    The null: knows base rates, nothing about the source\u0027s type.
  * A1 ``score_signature`` \u2014 P(rel, dst_type | type(src)) from the
    most-specific supported type signature, backing off up the IS_A chain.
    The hypothesis: a type IS its predictive signature.
  * A2 ``score_embedding_knn`` \u2014 cosine-kNN over node embeddings; score =
    fraction of src\u0027s nearest neighbours that carry a matching edge.
    "Embeddings point."
  * A3 ``score_structural_knn`` (optional) \u2014 Jaccard-kNN over nodes\u0027
    (rel, dst_type) edge-sets; the analogy form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from harness.signature import Signature
from harness.snapshot_io import Snapshot, node_type, type_chain


@dataclass
class TrainGraph:
    """Training-only view: the held-out positives are NOT in here.

    Bundles the snapshot (for the always-known IS_A backbone + embeddings),
    the surviving train edges, the per-type signatures built from them, and a
    few precomputed indices the scorers reuse.
    """

    snapshot: Snapshot
    train_edges: list[dict[str, str]]
    signatures: dict[str, Signature]
    min_support: int = 1
    # Derived (filled by build_train_graph).
    marginal_counts: dict[tuple[str, str | None], int] = field(
        default_factory=dict
    )
    marginal_total: int = 0
    edge_set_by_src: dict[str, set[tuple[str, str | None]]] = field(
        default_factory=dict
    )
    has_edge_by_src: dict[str, set[tuple[str, str | None]]] = field(
        default_factory=dict
    )


def build_train_graph(
    snapshot: Snapshot,
    train_edges: list[dict[str, str]],
    signatures: dict[str, Signature],
    *,
    min_support: int = 1,
) -> TrainGraph:
    """Assemble a :class:`TrainGraph` with all derived indices precomputed."""
    marginal_counts: dict[tuple[str, str | None], int] = {}
    edge_set_by_src: dict[str, set[tuple[str, str | None]]] = {}
    for e in train_edges:
        pat = (e["rel"], node_type(snapshot, e["dst"]))
        marginal_counts[pat] = marginal_counts.get(pat, 0) + 1
        edge_set_by_src.setdefault(e["src"], set()).add(pat)
    return TrainGraph(
        snapshot=snapshot,
        train_edges=list(train_edges),
        signatures=signatures,
        min_support=min_support,
        marginal_counts=marginal_counts,
        marginal_total=sum(marginal_counts.values()),
        edge_set_by_src=edge_set_by_src,
        has_edge_by_src=edge_set_by_src,
    )


def score_marginal(graph: TrainGraph, src: str, rel: str, dst: str) -> float:
    """A0: global frequency of (rel, type(dst)) over all train edges.

    Type-blind \u2014 ignores type(src) entirely. The base-rate null.
    """
    dst_type = node_type(graph.snapshot, dst)
    if graph.marginal_total == 0:
        return 0.0
    return graph.marginal_counts.get((rel, dst_type), 0) / graph.marginal_total


def score_signature(graph: TrainGraph, src: str, rel: str, dst: str) -> float:
    """A1: P(rel, type(dst) | type(src)) from the most-specific supported sig.

    Looks up the source\u0027s most-specific type signature with support, then the
    probability it assigns the candidate (rel, dst_type) pattern. Backs off up
    the IS_A chain: if the most-specific supported signature assigns the
    pattern zero mass, we try the next supertype that DOES place mass on it,
    discounted by a per-level backoff factor so a specific hit always beats a
    backed-off hit. Returns 0.0 if no supported signature on the chain places
    any mass on the pattern.
    """
    dst_type = node_type(graph.snapshot, dst)
    pattern = (rel, dst_type)
    src_type = node_type(graph.snapshot, src)

    backoff = 1.0
    for t in type_chain(graph.snapshot, src_type):
        sig = graph.signatures.get(t)
        if sig is None or sig.support() < graph.min_support:
            continue
        p = sig.prob(pattern)
        if p > 0.0:
            return backoff * p
        # Pattern unseen at this (supported) level: back off to the supertype,
        # discounting so a shallower hit never outranks a more-specific one.
        backoff *= 0.5
    return 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def score_embedding_knn(
    graph: TrainGraph, src: str, rel: str, dst: str, *, k: int = 10
) -> float | None:
    """A2: cosine-kNN over embeddings; fraction of neighbours with the pattern.

    For ``src``, finds its ``k`` nearest OTHER nodes by cosine in embedding
    space, then scores the candidate as the fraction of those neighbours that
    have a train edge matching ``(rel, type(dst))``. Returns ``None`` if the
    snapshot has no embeddings (the runner then skips A2) or ``src`` lacks one.
    """
    emb = graph.snapshot.embeddings
    if not emb or src not in emb:
        return None
    dst_type = node_type(graph.snapshot, dst)
    pattern = (rel, dst_type)
    src_vec = emb[src]
    # Rank other embedded nodes by cosine; ties broken by id for determinism.
    sims: list[tuple[float, str]] = [
        (_cosine(src_vec, vec), nid)
        for nid, vec in emb.items()
        if nid != src
    ]
    sims.sort(key=lambda t: (-t[0], t[1]))
    neighbours = [nid for _, nid in sims[:k]]
    if not neighbours:
        return 0.0
    hits = sum(
        1
        for nid in neighbours
        if pattern in graph.edge_set_by_src.get(nid, set())
    )
    return hits / len(neighbours)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def score_structural_knn(
    graph: TrainGraph, src: str, rel: str, dst: str, *, k: int = 10
) -> float | None:
    """A3 (optional): Jaccard-kNN over (rel, dst_type) edge-sets; the analogy.

    Finds the ``k`` nodes whose train edge-set is most Jaccard-similar to
    ``src``\u0027s, then scores the candidate as the fraction of those structural
    neighbours carrying the same ``(rel, type(dst))`` pattern. Returns ``None``
    if ``src`` has no train edges (no structure to compare).
    """
    src_set = graph.edge_set_by_src.get(src)
    if not src_set:
        return None
    dst_type = node_type(graph.snapshot, dst)
    pattern = (rel, dst_type)
    sims: list[tuple[float, str]] = [
        (_jaccard(src_set, other_set), nid)
        for nid, other_set in graph.edge_set_by_src.items()
        if nid != src
    ]
    sims.sort(key=lambda t: (-t[0], t[1]))
    neighbours = [nid for _, nid in sims[:k]]
    if not neighbours:
        return 0.0
    hits = sum(
        1
        for nid in neighbours
        if pattern in graph.edge_set_by_src.get(nid, set())
    )
    return hits / len(neighbours)
