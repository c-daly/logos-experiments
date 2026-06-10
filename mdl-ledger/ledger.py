"""W1 MDL ledger (logos-experiments#26, epic logos#557).

Two-part code length over the HCG: L_total = L(model) + L(data | model).
The forward objective for the world-model half of the program: bits are
arithmetic, and incompressible data cannot be flattered into compressing.

Encoding v1-as-built (one deliberate correction to the plan's draft):

  L(model)        = 16 * |T|                       # types exist
                  + H * log2(max(|T|, 2))          # type-hierarchy edges
                  + 16 * |R|                       # relation vocabulary
  L(data | model) = sum_nodes  -log2 P(type(n))                 # membership
                  + sum_edges  -log2 P(rel)                     # which relation
                              + -log2 P(type(tgt) | rel)        # target's type
                              + log2(max(members(type(tgt)),1)) # which member

All probabilities Laplace-smoothed. The plan's draft coded an edge as a
flat exception (log2|R| + log2|N|), independent of types -- under that
encoding the type term is pure entropy and deleting ANY type lowers
L_total (degenerate optimum; it fails the plan's own sanity gate
"evicting a well-populated type must cost bits"). v1-as-built instead
lets types EARN their bits: a type compresses every edge that points
into it by narrowing the target candidate set from "any node" to "any
member". A coherent type pays for itself; a catch-all blob doesn't.
Singletons pay maximal per-node cost, the relation vocabulary's 16*|R|
makes the df=1 predicate problem (see structural-health/BASELINE.md)
legible in bits. Refinements must beat this encoding on held-out
prediction (W2), not on taste.

Rules (W3) are not yet part of the model; when they land they enter as a
third model term plus cheaper codes for rule-predicted edges.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    """A typed-graph summary the ledger prices.

    membership:  data-node uuid -> type name (realm kind if untyped)
    type_parents: type name -> parent type name (None for roots);
                  every type in use must have an entry or it is treated
                  as a parentless root
    edges:       non-typing semantic edges (relation, src uuid, tgt uuid)
    """

    membership: dict[str, str]
    type_parents: dict[str, str | None]
    edges: tuple[tuple[str, str, str], ...]


def _log2(x: float) -> float:
    return math.log2(x)


def compute_ledger(s: Snapshot) -> dict:
    types = set(s.type_parents) | set(s.membership.values())
    n_types = len(types)
    members = Counter(s.membership.values())
    n_nodes = len(s.membership)

    hierarchy_edges = sum(1 for p in s.type_parents.values() if p is not None)
    rel_counts = Counter(rel for rel, _, _ in s.edges)
    n_rels = len(rel_counts)
    n_edges = len(s.edges)

    L_model = (
        16.0 * n_types
        + hierarchy_edges * _log2(max(n_types, 2))
        + 16.0 * n_rels
    )

    # membership term, Laplace over types
    def node_cost(t: str) -> float:
        return -_log2((members[t] + 1) / (n_nodes + n_types))

    L_data_nodes = sum(node_cost(t) for t in s.membership.values())

    # edge terms
    tgt_type_by_rel: dict[str, Counter] = defaultdict(Counter)
    for rel, _, tgt in s.edges:
        tgt_type_by_rel[rel][s.membership.get(tgt, "?")] += 1

    def edge_cost(rel: str, tgt: str) -> float:
        t_tgt = s.membership.get(tgt, "?")
        p_rel = (rel_counts[rel] + 1) / (n_edges + n_rels) if n_edges else 1.0
        c = tgt_type_by_rel[rel]
        p_t = (c[t_tgt] + 1) / (sum(c.values()) + n_types)
        return -_log2(p_rel) - _log2(p_t) + _log2(max(members[t_tgt], 1))

    L_data_edges = sum(edge_cost(rel, tgt) for rel, _, tgt in s.edges)

    # attribution for the report: node cost + its outgoing edge costs
    per_node = {u: node_cost(t) for u, t in s.membership.items()}
    for rel, src, tgt in s.edges:
        if src in per_node:
            per_node[src] += edge_cost(rel, tgt)
    top_nodes = sorted(per_node.items(), key=lambda kv: -kv[1])[:20]

    per_type = defaultdict(float)
    for u, t in s.membership.items():
        per_type[t] += node_cost(t)
    top_types = sorted(per_type.items(), key=lambda kv: -kv[1])[:10]

    return {
        "nodes": n_nodes,
        "types": n_types,
        "relations": n_rels,
        "edges": n_edges,
        "hierarchy_edges": hierarchy_edges,
        "L_model": L_model,
        "L_data_nodes": L_data_nodes,
        "L_data_edges": L_data_edges,
        "L_total": L_model + L_data_nodes + L_data_edges,
        "relation_vocab_bits": 16.0 * n_rels,
        "top_expensive_nodes": top_nodes,
        "top_expensive_types": top_types,
    }


def delta(s: Snapshot, op) -> float:
    """dL for an ontology operation: recompute on the transformed copy.

    Exact by construction (no incremental approximation); ops are pure
    Snapshot -> Snapshot functions from delta.py.
    """
    return compute_ledger(op(s))["L_total"] - compute_ledger(s)["L_total"]
