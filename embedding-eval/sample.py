"""Pull the labeled node sample for the embedding bake-off (lx39, #39).

Ground truth = the fine type a node IS_A (a ``type_definition``), restricted to
types with >= MIN_MEMBERS members so kNN/silhouette are meaningful. One type
per node (lexicographic-first, matching the proposer convention). Writes
sample.json: ``[{"name": ..., "type": ...}, ...]``.

    NEO4J_PASSWORD=... uv run python sample.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_MEMBERS = 5


def fetch() -> list[dict]:
    from neo4j import GraphDatabase

    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise SystemExit("NEO4J_PASSWORD must be set")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    def work(tx):
        return tx.run(
            "MATCH (e:Node {type:'edge', relation:'IS_A'}) "
            "MATCH (n:Node {uuid: e.source}) "
            "WHERE n.type <> 'edge' AND n.type <> 'type_definition' "
            "MATCH (td:Node {uuid: e.target, type:'type_definition'}) "
            "RETURN n.uuid AS uuid, n.name AS name, td.name AS type"
        ).data()

    try:
        with driver.session() as s:
            return s.execute_read(work)
    finally:
        driver.close()


def build() -> list[dict]:
    by_node: dict[str, tuple[str, list[str]]] = {}
    for r in fetch():
        by_node.setdefault(r["uuid"], (r["name"], []))[1].append(r["type"])
    labeled = [(nm, sorted(ts)[0]) for nm, ts in by_node.values() if nm]
    counts = Counter(t for _, t in labeled)
    keep = {t for t, c in counts.items() if c >= MIN_MEMBERS}
    sample = [{"name": nm, "type": t} for nm, t in labeled if t in keep]
    (HERE / "sample.json").write_text(json.dumps(sample, indent=2))
    print(f"{len(sample)} nodes across {len(keep)} types (>= {MIN_MEMBERS} members)")
    return sample


if __name__ == "__main__":
    build()
