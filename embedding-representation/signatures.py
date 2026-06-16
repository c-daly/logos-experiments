"""RELATIONS facet: each entity's (relation_type, neighbor_type) signature,
pulled from the live reified-edge graph and scored by weighted Jaccard.

Mirrors production sophia.hcg_client.HCGClient.query_edges_from -- FROM-edges
only (the node as edge source). Reified edges are :Node carrying a `relation`
property, linked (edge)-[:FROM]->(source) and (edge)-[:TO]->(target). The
signature is chunk-blind by construction: no passage text touches it.

    NEO4J_PASSWORD=... .venv/bin/python signatures.py   # pull -> signatures.json
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from sophia.maintenance.structural_signature import build_signature, signature_similarity

HERE = Path(__file__).resolve().parent
SIG_PATH = HERE / "signatures.json"

# FROM-edges of each sampled node + the neighbour's type, in one batched read.
_CYPHER = """
MATCH (edge:Node)-[:FROM]->(src:Node)
WHERE src.uuid IN $uuids AND edge.relation IS NOT NULL
OPTIONAL MATCH (edge)-[:TO]->(tgt:Node)
RETURN src.uuid AS uuid, edge.relation AS relation, tgt.type AS neighbor_type
"""


def fetch_signatures(uuids: list[str]) -> dict[str, list[list[str]]]:
    """Return {uuid: [[relation, neighbor_type], ...]} for every uuid (empty
    list if the node has no typed FROM-edges)."""
    from neo4j import GraphDatabase

    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise SystemExit("NEO4J_PASSWORD must be set")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    out: dict[str, list[list[str]]] = {u: [] for u in uuids}

    def work(tx):
        rows: list[dict] = []
        for i in range(0, len(uuids), 1000):
            rows += tx.run(_CYPHER, uuids=uuids[i : i + 1000]).data()
        return rows

    try:
        with driver.session() as s:
            for rec in s.execute_read(work):
                if rec["relation"] and rec["neighbor_type"]:
                    out[rec["uuid"]].append([rec["relation"], rec["neighbor_type"]])
    finally:
        driver.close()
    return out


def load_signatures() -> dict[str, list[list[str]]]:
    return json.loads(SIG_PATH.read_text()) if SIG_PATH.exists() else {}


def to_counter(pairs: list[list[str]]) -> Counter:
    """Reuse production build_signature on stored [relation, neighbor_type] pairs."""
    return build_signature(
        [{"relation": r, "neighbor_type": t} for r, t in pairs]
    )


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    uuids = [r["uuid"] for r in sample]
    sigs = fetch_signatures(uuids)
    SIG_PATH.write_text(json.dumps(sigs, ensure_ascii=False))
    nonempty = sum(1 for u in uuids if sigs.get(u))
    print(f"wrote {SIG_PATH}")
    print(f"signature coverage: {nonempty}/{len(uuids)} = {nonempty / len(uuids):.3f}")


if __name__ == "__main__":
    main()
