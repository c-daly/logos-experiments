"""W0.3b live runner: propose the consolidation mapping (logos-experiments#34).

Reads the live graph (read-only; NEO4J_PASSWORD required, no default),
derives each semantic edge's (source-type, target-type) via IS_A
membership (lexicographic tie-break, realm-kind fallback), runs the
proposer over the df=1 predicates, and writes mapping.csv for review.

The csv is the deliverable; nothing is applied to the graph. The review
column is filled by hand; an approved table is applied by a separate,
reviewed maintenance step.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from propose import Edge, propose_mappings

TYPING_RELATIONS = {"IS_A", "INSTANCE_OF", "SUBTYPE_OF"}
NON_DATA_KINDS = {"edge", "type_definition"}


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit(
            "NEO4J_PASSWORD must be set explicitly for the relation-vocab "
            "proposer (no default credential; harness convention)."
        )
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_typed_edges(driver) -> list[Edge]:
    def work(tx):
        nodes = {
            r["uuid"]: (r["kind"], r["name"] or "")
            for r in tx.run(
                "MATCH (n:Node) WHERE n.type <> 'edge' "
                "RETURN n.uuid AS uuid, n.type AS kind, n.name AS name"
            )
            if r["uuid"] and r["kind"]
        }
        raw = [
            (r["rel"], r["src"], r["tgt"])
            for r in tx.run(
                "MATCH (e:Node {type:'edge'}) "
                "RETURN e.relation AS rel, e.source AS src, e.target AS tgt"
            )
        ]
        return nodes, raw

    with driver.session() as s:
        nodes, raw = s.execute_read(work)
    edges = [t for t in raw if all(t)]

    candidates: dict[str, list[str]] = {}
    for rel, src, tgt in edges:
        if rel != "IS_A":
            continue
        src_rec, tgt_rec = nodes.get(src), nodes.get(tgt)
        if (
            src_rec is not None
            and tgt_rec is not None
            and src_rec[0] not in NON_DATA_KINDS
            and tgt_rec[0] == "type_definition"
            and tgt_rec[1]
        ):
            candidates.setdefault(src, []).append(tgt_rec[1])

    def type_of(uuid: str) -> str:
        if uuid in candidates:
            return min(candidates[uuid])
        rec = nodes.get(uuid)
        return rec[0] if rec else "?"

    return [
        Edge(rel, type_of(src), type_of(tgt))
        for rel, src, tgt in edges
        if rel not in TYPING_RELATIONS
    ]


def main() -> None:
    driver = _driver()
    try:
        edges = fetch_typed_edges(driver)
    finally:
        driver.close()

    rows = propose_mappings(edges)
    out = Path(__file__).parent / "mapping.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicate", "df", "proposed_target", "tier", "evidence", "review"])
        for r in rows:
            w.writerow([r.predicate, r.df, r.target, r.tier, r.evidence, r.review])

    tiers = Counter(r.tier for r in rows)
    covered = sum(1 for r in rows if r.tier != "keep")
    total = len(rows)
    print(f"relation-vocab proposer -- {datetime.now(timezone.utc).date().isoformat()}")
    print(f"semantic edges: {len(edges)}  df=1 predicates: {total}")
    print(f"tiers: {dict(tiers)}")
    print(
        f"coverage (proposal with evidence): {covered}/{total} "
        f"= {covered / total:.1%}  (ticket gate: >=80%)"
    )
    print(f"written: {out}")


if __name__ == "__main__":
    main()
