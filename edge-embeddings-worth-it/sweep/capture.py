"""CAPTURE: dump a static clustering fixture from the LIVE LOGOS graph.

This is the *expensive* half of the capture+sweep split. It hits Neo4j, Milvus
and Hermes ONCE to materialise everything ``sweep.py`` needs, so we can then try
many clustering configs offline without re-ingesting.

WHAT IT WRITES (``fixture.json``):
  entities: [{uuid, name, domain, type, embedding}]   -- Milvus "Entity" vector
  edges:    [{uuid, relation, src_uuid, tgt_uuid, src_name, tgt_name,
             src_type, tgt_type,
             embeddings: {relationship_label, triple, name}}]
             -- the THREE edge-embedding schemes, each via Hermes /embed_text:
               relationship_label : "RELATIONSHIP: <relation>"
               triple             : "<src> <relation> <tgt>"
               name               : edge.name
             Hermes is hit once per DISTINCT text (cached).
  meta: {n_entities, n_edges, n_domains, domains, dim, embedding_model, ...}

Provenance domains come from the run round_N.json (key ``entity_domains``, a
``{uuid: domain}`` map) -- pass it via ``--round-json``. Entities not in the map
get domain ``"unknown"``.

IS_A / COMPONENT_OF edges are EXCLUDED (structural, not semantic).

Usage (MUST run inside the sophia poetry env, against the live stack):
  poetry run python capture.py --round-json ../workspace/round_3.json --out fixture.json

Stack defaults (overridable via env), matching the harness:
  NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=logosdev
  MILVUS_HOST=localhost            MILVUS_PORT=19530
  HERMES_URL=http://localhost:17000

NOTE: this script is WRITE-then-park scaffolding -- it is intentionally NOT run
against the live graph while a multi-round ingestion is in progress.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "logosdev")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")

# Structural relations excluded from the SEMANTIC edge population.
_STRUCTURAL_RELATIONS = {"IS_A", "COMPONENT_OF"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Hermes -- /embed_text, cached per distinct text
# ---------------------------------------------------------------------------


class HermesEmbedder:
    """Caches /embed_text calls so each distinct string is embedded once."""

    def __init__(self, url: str = HERMES_URL) -> None:
        self.url = url
        self._cache: dict[str, list[float]] = {}
        self.model: str = "unknown"
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        import time

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = requests.post(
                    f"{self.url}/embed_text", json={"text": text}, timeout=120
                )
                r.raise_for_status()
                d = r.json()
                emb = list(d["embedding"])
                self.model = d.get("model", self.model)
                self._cache[text] = emb
                self.calls += 1
                return emb
            except Exception as e:  # transient upstream (e.g. OpenAI 503)
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"embed failed after 5 retries for {text!r}: {last_err}")


# ---------------------------------------------------------------------------
# Neo4j -- entities + semantic edges
# ---------------------------------------------------------------------------


def fetch_entities(driver: Any) -> list[dict[str, Any]]:
    """Return entity-kind nodes: {uuid, name, type}. Type/reserved nodes excluded."""
    cypher = (
        "MATCH (n:Node) "
        "WHERE n.type IS NOT NULL "
        "AND n.type <> \"edge\" "
        "AND NOT n.type STARTS WITH \"reserved_\" "
        "AND NOT n.type STARTS WITH \"type_\" "
        "AND NOT n.type STARTS WITH \"edge_type\" "
        "RETURN n.uuid AS uuid, n.name AS name, n.type AS type"
    )
    with driver.session() as s:
        return [dict(rec) for rec in s.run(cypher)]


def fetch_semantic_edges(driver: Any) -> list[dict[str, Any]]:
    """Return reified edge nodes, excluding structural (IS_A/COMPONENT_OF).

    Each row carries the edge relation + name and both endpoints
    uuid / name / type, resolved through the reified-edge source/target props.
    """
    cypher = (
        "MATCH (e:Node {type: \"edge\"}) "
        "WHERE NOT e.relation IN $structural "
        "OPTIONAL MATCH (s:Node {uuid: e.source}) "
        "OPTIONAL MATCH (t:Node {uuid: e.target}) "
        "RETURN e.uuid AS uuid, e.relation AS relation, e.name AS name, "
        "e.source AS src_uuid, e.target AS tgt_uuid, "
        "s.name AS src_name, t.name AS tgt_name, "
        "s.type AS src_type, t.type AS tgt_type"
    )
    with driver.session() as s:
        return [dict(rec) for rec in s.run(cypher, structural=list(_STRUCTURAL_RELATIONS))]


# ---------------------------------------------------------------------------
# Milvus -- Entity vectors
# ---------------------------------------------------------------------------


def build_sync() -> Any:
    """Construct an ``HCGMilvusSync`` bound to the configured Milvus."""
    from logos_hcg.sync import HCGMilvusSync

    sync = HCGMilvusSync(milvus_host=MILVUS_HOST, milvus_port=str(MILVUS_PORT))
    sync.connect()
    return sync


def entity_embedding(sync: Any, uuid: str) -> list[float] | None:
    rec = sync.get_embedding("Entity", uuid)
    if not rec:
        return None
    emb = rec.get("embedding")
    return list(emb) if emb is not None else None


# ---------------------------------------------------------------------------
# Edge embedding texts (the three schemes)
# ---------------------------------------------------------------------------


def _scheme_texts(edge: dict[str, Any]) -> dict[str, str]:
    """The three scheme texts for one edge. ``name`` may be empty -> skipped."""
    relation = edge.get("relation") or "RELATED_TO"
    src = edge.get("src_name") or edge.get("src_uuid") or "?"
    tgt = edge.get("tgt_name") or edge.get("tgt_uuid") or "?"
    name = edge.get("name") or ""
    return {
        "relationship_label": f"RELATIONSHIP: {relation}",
        "triple": f"{src} {relation} {tgt}",
        "name": name,
    }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def load_entity_domains(round_json: Path) -> dict[str, str]:
    d = json.loads(round_json.read_text())
    ed = d.get("entity_domains")
    if not isinstance(ed, dict):
        raise SystemExit(f"--round-json {round_json} has no entity_domains map")
    return {str(k): str(v) for k, v in ed.items()}


def capture(round_json: Path, out: Path) -> dict[str, Any]:
    from neo4j import GraphDatabase

    entity_domains = load_entity_domains(round_json)
    _round = json.loads(round_json.read_text())
    entity_sentences = _round.get("entity_sentences", {}) or {}
    edge_sentences = _round.get("edge_sentences", {}) or {}
    log(f"[capture] {len(entity_domains)} provenance domains from {round_json.name}")
    log(
        f"[capture] {len(entity_sentences)} entity / {len(edge_sentences)} edge "
        f"source-sentence joins"
    )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    sync = build_sync()
    embedder = HermesEmbedder()

    try:
        raw_entities = fetch_entities(driver)
        log(f"[capture] {len(raw_entities)} entity nodes from Neo4j")
        entities: list[dict[str, Any]] = []
        missing_vec = 0
        for ent in raw_entities:
            emb = entity_embedding(sync, ent["uuid"])
            if emb is None:
                missing_vec += 1
                continue
            ent_out = {
                "uuid": ent["uuid"],
                "name": ent["name"],
                "domain": entity_domains.get(ent["uuid"], "unknown"),
                "type": ent["type"],
                "embedding": emb,
            }
            # CONTEXT vector: the source sentence the entity was extracted from.
            # The sweep fuses this with the name embedding (name+context schemes).
            sent = entity_sentences.get(ent["uuid"])
            if sent:
                ent_out["context_embedding"] = embedder.embed(sent)
            entities.append(ent_out)
        log(
            f"[capture] {len(entities)} entities with Entity vectors "
            f"({missing_vec} skipped -- no Milvus vector)"
        )

        raw_edges = fetch_semantic_edges(driver)
        log(f"[capture] {len(raw_edges)} semantic edges (IS_A/COMPONENT_OF excluded)")
        edges: list[dict[str, Any]] = []
        for e in raw_edges:
            texts = _scheme_texts(e)
            embeddings: dict[str, list[float]] = {}
            for scheme, text in texts.items():
                if not text:
                    continue
                embeddings[scheme] = embedder.embed(text)
            # CONTEXT vector: the source sentence the edge was extracted from.
            sent = edge_sentences.get(e["uuid"])
            if sent:
                embeddings["context"] = embedder.embed(sent)
            edges.append(
                {
                    "uuid": e["uuid"],
                    "relation": e.get("relation"),
                    "name": e.get("name"),
                    "src_uuid": e.get("src_uuid"),
                    "tgt_uuid": e.get("tgt_uuid"),
                    "src_name": e.get("src_name"),
                    "tgt_name": e.get("tgt_name"),
                    "src_type": e.get("src_type"),
                    "tgt_type": e.get("tgt_type"),
                    "embeddings": embeddings,
                }
            )
        log(
            f"[capture] embedded {len(edges)} edges via {embedder.calls} "
            f"distinct Hermes /embed_text calls"
        )
    finally:
        driver.close()

    domains = sorted({en["domain"] for en in entities})
    dim = len(entities[0]["embedding"]) if entities else 0
    fixture = {
        "meta": {
            "source": "live-graph-capture",
            "round_json": str(round_json),
            "n_entities": len(entities),
            "n_edges": len(edges),
            "n_domains": len(domains),
            "domains": domains,
            "dim": dim,
            "embedding_model": embedder.model,
            "hermes_calls": embedder.calls,
        },
        "entities": entities,
        "edges": edges,
    }
    out.write_text(json.dumps(fixture))
    log(f"[capture] wrote {out} ({len(entities)} entities, {len(edges)} edges)")
    return fixture


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--round-json",
        required=True,
        type=Path,
        help="Path to the run round_N.json (provides entity_domains map).",
    )
    ap.add_argument(
        "--out",
        default=Path("fixture.json"),
        type=Path,
        help="Output fixture path (default: fixture.json).",
    )
    args = ap.parse_args(argv)
    capture(args.round_json, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
