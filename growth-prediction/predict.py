"""W2 growth-prediction predictors (logos-experiments#27, epic logos#557).

Pure functions: graph in, ranked candidate edges out. The standing question
they answer: how much does the HCG's *shape* know about what arrives next?

Predictors are the classical topology heuristics (common-neighbors,
Adamic-Adar, resource-allocation, preferential-attachment). Candidate space
is the 2-hop non-edges of the training graph -- the only pairs the first
three can score at all; preferential-attachment is restricted to the same
set for comparability (noted in the report). PA doubles as the degree-only
null: it is the closed-form expectation of a configuration-model link, so
lift *over PA* is structure beyond degree.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable

Pair = frozenset


def build_adjacency(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Undirected adjacency over node ids; self-loops dropped."""
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return dict(adj)


def candidate_pairs(adj: dict[str, set[str]]) -> set[Pair]:
    """All 2-hop pairs (share >= 1 neighbor) that are not existing edges."""
    out: set[Pair] = set()
    for w, neighbors in adj.items():
        ns = sorted(neighbors)
        for i, u in enumerate(ns):
            for v in ns[i + 1 :]:
                if v not in adj[u]:
                    out.add(Pair((u, v)))
    return out


def common_neighbors(adj: dict[str, set[str]], u: str, v: str) -> float:
    return float(len(adj.get(u, set()) & adj.get(v, set())))


def adamic_adar(adj: dict[str, set[str]], u: str, v: str) -> float:
    # a shared neighbor w is adjacent to both u and v, so deg(w) >= 2 and
    # log(deg) is safe; guard anyway.
    return sum(
        1.0 / math.log(len(adj[w]))
        for w in adj.get(u, set()) & adj.get(v, set())
        if len(adj[w]) > 1
    )


def resource_allocation(adj: dict[str, set[str]], u: str, v: str) -> float:
    return sum(1.0 / len(adj[w]) for w in adj.get(u, set()) & adj.get(v, set()))


def preferential_attachment(adj: dict[str, set[str]], u: str, v: str) -> float:
    return float(len(adj.get(u, set())) * len(adj.get(v, set())))


PREDICTORS: dict[str, Callable[[dict[str, set[str]], str, str], float]] = {
    "common_neighbors": common_neighbors,
    "adamic_adar": adamic_adar,
    "resource_allocation": resource_allocation,
    "preferential_attachment": preferential_attachment,
}


def rank_candidates(
    adj: dict[str, set[str]],
    scorer: Callable[[dict[str, set[str]], str, str], float],
    candidates: set[Pair] | None = None,
) -> list[tuple[Pair, float]]:
    """Candidates scored and sorted descending; ties broken on the sorted
    pair tuple so runs are reproducible."""
    cands = candidates if candidates is not None else candidate_pairs(adj)
    scored = [(p, scorer(adj, *sorted(p))) for p in cands]
    return sorted(scored, key=lambda ps: (-ps[1], tuple(sorted(ps[0]))))


def backtest_split(
    edges: Iterable[tuple[str, str, str]], train_fraction: float = 0.8
) -> tuple[list[tuple[str, str, str]], set[Pair]]:
    """Temporal split by timestamp (ISO strings sort correctly).

    Returns (train edges in temporal order, test PAIRS). A pair already
    present in training is not a test pair -- the harness predicts NEW
    structure, not repeats.
    """
    ordered = sorted(edges, key=lambda e: (e[2], e[0], e[1]))
    cut = int(len(ordered) * train_fraction)
    train = ordered[:cut]
    train_pairs = {Pair((u, v)) for u, v, _ in train}
    test_pairs = {
        Pair((u, v)) for u, v, _ in ordered[cut:] if u != v
    } - train_pairs
    return train, test_pairs
