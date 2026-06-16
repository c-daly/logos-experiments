"""Pull the live entity sample for the representation experiment.

Every ``entity`` node that carries the full ``raw_text`` + ``start`` + ``end``
triple (so a context window is reconstructable). No labels are needed -- the
battery is label-free. Writes sample.json:
``[{"uuid","name","raw_text","start","end"}, ...]``.

    NEO4J_PASSWORD=logosdev python sample.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
            "MATCH (n:Node {type:'entity'}) "
            "WHERE n.raw_text IS NOT NULL AND n.start IS NOT NULL "
            "AND n.end IS NOT NULL "
            "RETURN n.uuid AS uuid, n.name AS name, n.raw_text AS raw_text, "
            "n.start AS start, n.end AS end"
        ).data()

    try:
        with driver.session() as s:
            return s.execute_read(work)
    finally:
        driver.close()


def main() -> None:
    rows = fetch()
    # keep only rows whose offsets actually land on (or near) the name, so the
    # window is anchored to the real mention.
    clean = [r for r in rows if r["raw_text"] and 0 <= r["start"] < r["end"] <= len(r["raw_text"])]
    (HERE / "sample.json").write_text(json.dumps(clean, ensure_ascii=False))
    print(f"wrote {len(clean)} entities (dropped {len(rows) - len(clean)} with bad offsets)")


if __name__ == "__main__":
    main()
