"""W2 backtest report (logos-experiments#27, epic logos#557).

Temporal backtest on the live graph: train the adjacency on the first
80% of non-typing semantic edges between data nodes (by created_at),
predict the remaining 20%'s NEW pairs, score every predictor against the
nulls. Emits prediction_report.json and appends a row to trend.jsonl.

Read-only. Credentials follow the harness convention: NEO4J_PASSWORD
required explicitly, no default.

Coverage is reported, never silently capped: test pairs whose endpoints
the training graph has not seen (or that fall outside the 2-hop candidate
set) cannot be scored by these predictors -- their count is in the report
as `test_pairs_unreachable`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from predict import (
    PREDICTORS,
    backtest_split,
    build_adjacency,
    candidate_pairs,
    rank_candidates,
)
from score import auc_sampled, precision_at_k, random_ranking, recency_ranking

TYPING_RELATIONS = {"IS_A", "INSTANCE_OF", "SUBTYPE_OF"}
NON_DATA_KINDS = {"edge", "type_definition"}
KS = (50, 100, 500)
AUC_NEGATIVES = 5_000


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit(
            "NEO4J_PASSWORD must be set explicitly for the growth-prediction "
            "harness (no default credential; harness convention)."
        )
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_timestamped_edges(driver) -> list[tuple[str, str, str]]:
    """Non-typing semantic edges between data nodes, as (src, tgt, ts)."""

    def work(tx):
        kinds = {
            r["uuid"]: r["kind"]
            for r in tx.run(
                "MATCH (n:Node) WHERE n.type <> 'edge' "
                "RETURN n.uuid AS uuid, n.type AS kind"
            )
            if r["uuid"] and r["kind"]
        }
        raw = [
            (r["src"], r["tgt"], r["ts"])
            for r in tx.run(
                "MATCH (e:Node {type:'edge'}) "
                "WHERE NOT e.relation IN $typing "
                "RETURN e.source AS src, e.target AS tgt, e.created_at AS ts",
                typing=sorted(TYPING_RELATIONS),
            )
        ]
        return kinds, raw

    with driver.session() as s:
        kinds, raw = s.execute_read(work)
    edges = [
        (src, tgt, ts)
        for src, tgt, ts in raw
        if src and tgt and ts
        and kinds.get(src) not in (None, *NON_DATA_KINDS)
        and kinds.get(tgt) not in (None, *NON_DATA_KINDS)
    ]
    dropped = len(raw) - len(edges)
    if dropped:
        print(
            f"warn: dropped {dropped} edge(s) (missing fields or non-data "
            "endpoints)",
            file=sys.stderr,
        )
    return edges


def run_backtest(edges: list[tuple[str, str, str]], train_fraction: float = 0.8) -> dict:
    train, test = backtest_split(edges, train_fraction)
    adj = build_adjacency([(u, v) for u, v, _ in train])
    cands = candidate_pairs(adj)
    reachable = test & cands
    last_seen = {}
    for i, (u, v, _) in enumerate(train):
        last_seen[u] = i
        last_seen[v] = i

    negatives = sorted(cands - test, key=lambda p: tuple(sorted(p)))[:AUC_NEGATIVES]

    def evaluate(ranked, score_fn):
        return {
            "precision_at_k": precision_at_k(ranked, reachable, KS),
            "hits_at_test_size": precision_at_k(ranked, reachable, (len(reachable) or 1,))[
                len(reachable) or 1
            ],
            "auc": auc_sampled(score_fn, reachable, negatives, seed=0),
        }

    results = {}
    for name, fn in PREDICTORS.items():
        ranked = rank_candidates(adj, fn, cands)
        scores = dict(ranked)
        results[name] = evaluate(ranked, lambda p, s=scores: s.get(p, 0.0))

    rnd = random_ranking(cands, seed=0)
    results["null_random"] = evaluate(rnd, lambda p: 0.0)
    rec = recency_ranking(cands, last_seen)
    rec_scores = dict(rec)
    results["null_recency"] = evaluate(rec, lambda p: rec_scores.get(p, -1.0))

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_fraction": train_fraction,
        "edges_total": len(edges),
        "train_edges": len(train),
        "test_pairs_new": len(test),
        "test_pairs_reachable": len(reachable),
        "test_pairs_unreachable": len(test) - len(reachable),
        "candidates": len(cands),
        "results": results,
    }


def main() -> None:
    driver = _driver()
    try:
        edges = fetch_timestamped_edges(driver)
    finally:
        driver.close()
    report = run_backtest(edges)
    out = Path(__file__).parent / "prediction_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    with (Path(__file__).parent / "trend.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
