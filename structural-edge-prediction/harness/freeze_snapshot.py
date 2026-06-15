"""Export the live HCG (Neo4j) into the offline snapshot schema.

This is the ONLY file that touches the live stack. It uses Sophia's
``HCGClient`` (``sophia.hcg_client.client``), which extends the shared LOGOS
client and adds the generic graph-read accessors this exporter needs
(``list_all_nodes`` / ``list_all_edges``) on top of the inherited
``get_all_type_definitions``. Pull type definitions, the reified IS_A edges,
and the relational edges, then flatten them into::

    {"nodes": [...], "type_parents": {...}, "edges": [...], "embeddings": {...}}

Gated behind ``FREEZE_LIVE=1`` so it never runs by accident. Performs NO writes.
Everything else in this experiment runs offline on the toy fixture; A2 simply
skips when embeddings are absent.

Run (live):
    FREEZE_LIVE=1 NEO4J_PASSWORD=... \\
        python -m harness.freeze_snapshot --n 350
(run inside the sophia env so ``sophia.hcg_client`` imports.)

Stack defaults (overridable via env):
    NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=(required)
    MILVUS_HOST=localhost  MILVUS_PORT=19530  (only needed for the A2 embeddings)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIXTURES = EXP / "fixtures"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

# IS_A is the type backbone; everything else is a relational edge we predict.
IS_A = "IS_A"

# Generous read cap so a freshly reseeded graph is never silently truncated.
READ_LIMIT = 1_000_000


def _type_id_of(row: dict[str, Any]) -> str:
    """Stable type id for a type-def row -- prefer name, fall back to uuid."""
    return str(row.get("name") or row["uuid"])


def build_snapshot_from_client(
    client: Any, *, with_embeddings: bool = True
) -> dict[str, Any]:
    """Flatten a read-only ``HCGClient`` into the snapshot schema.

    Pulls type defs and the reified IS_A edges, derives ``type_parents`` from
    IS_A edges BETWEEN type uuids, assigns each entity node its immediate type
    via IS_A, and collects the non-IS_A relational edges among entity nodes.
    Embeddings are pulled best-effort and omitted on any failure (A2 then skips).
    """
    type_defs = client.get_all_type_definitions() or []
    type_uuids = {row["uuid"] for row in type_defs}
    type_id_by_uuid = {row["uuid"]: _type_id_of(row) for row in type_defs}

    is_a_edges = client.list_all_edges(relation_type=IS_A, limit=READ_LIMIT) or []

    # type_parents: IS_A edges whose BOTH endpoints are type uuids.
    type_parents: dict[str, str | None] = {
        type_id_by_uuid[u]: None for u in type_uuids
    }
    for e in is_a_edges:
        src, dst = e.get("source"), e.get("target")
        if src in type_uuids and dst in type_uuids:
            type_parents[type_id_by_uuid[src]] = type_id_by_uuid[dst]

    # node.type: IS_A edges from an ENTITY uuid to a type uuid (membership).
    node_type_by_uuid: dict[str, str] = {}
    for e in is_a_edges:
        src, dst = e.get("source"), e.get("target")
        if src not in type_uuids and dst in type_uuids:
            node_type_by_uuid[src] = type_id_by_uuid[dst]

    all_nodes = client.list_all_nodes(limit=READ_LIMIT) or []
    nodes: list[dict[str, Any]] = []
    node_uuids: set[str] = set()
    for n in all_nodes:
        uuid = n.get("uuid")
        if uuid is None or uuid in type_uuids:
            continue  # skip type-definition nodes; we want entity instances
        node_uuids.add(uuid)
        nodes.append(
            {
                "id": uuid,
                "type": node_type_by_uuid.get(uuid),
                "label": n.get("name", uuid),
            }
        )

    # Relational edges: every non-IS_A edge among kept entity nodes.
    all_edges = client.list_all_edges(limit=READ_LIMIT) or []
    edges: list[dict[str, str]] = []
    for e in all_edges:
        rel = e.get("relation")
        src, dst = e.get("source"), e.get("target")
        if rel == IS_A:
            continue
        if src in node_uuids and dst in node_uuids:
            edges.append({"src": src, "rel": rel, "dst": dst})

    snapshot: dict[str, Any] = {
        "nodes": nodes,
        "type_parents": type_parents,
        "edges": edges,
    }

    if with_embeddings:
        embeddings = _try_fetch_embeddings(client, node_uuids, node_type_by_uuid)
        if embeddings:
            snapshot["embeddings"] = embeddings

    return snapshot


def _try_fetch_embeddings(
    client: Any, node_uuids: set[str], node_type_by_uuid: dict[str, str]
) -> dict[str, list[float]]:
    """Best-effort per-node embedding pull; empty dict if unavailable.

    Sophia's client exposes ``get_embedding(node_type, uuid)`` (vectors live in
    Milvus, partitioned by node type). Any failure means A2 skips, which is
    fine -- the structural arms (A0/A1/A3) do not need embeddings.
    """
    getter = getattr(client, "get_embedding", None)
    if getter is None:
        return {}
    out: dict[str, list[float]] = {}
    for uuid in sorted(node_uuids):
        ntype = node_type_by_uuid.get(uuid)
        if ntype is None:
            continue
        try:
            vec = getter(ntype, uuid)
        except Exception:  # noqa: BLE001 -- embeddings are optional
            continue
        if vec:
            out[uuid] = [float(x) for x in vec]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=350, help="label only -- names the output file"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip the embedding pull (A2 will be unavailable)",
    )
    args = parser.parse_args(argv)

    if os.environ.get("FREEZE_LIVE") != "1":
        print(
            "[freeze] refusing to touch the live stack: set FREEZE_LIVE=1 to "
            "enable the read-only export",
            file=sys.stderr,
        )
        return 2

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print(
            "[freeze] NEO4J_PASSWORD must be set explicitly (refusing a default "
            "credential)",
            file=sys.stderr,
        )
        return 2

    # Lazy: only the live path needs the stack client (pyproject deps = []).
    from sophia.hcg_client.client import HCGClient

    client = HCGClient(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=password,
        milvus_host=os.environ.get("MILVUS_HOST", "localhost"),
        milvus_port=int(os.environ.get("MILVUS_PORT", "19530")),
    )
    snapshot = build_snapshot_from_client(
        client, with_embeddings=not args.no_embeddings
    )

    FIXTURES.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES / f"graph_{args.n}.json"
    out_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[freeze] wrote {p}: {nn} nodes, {nt} types, {ne} relational edges, "
        "{em} embeddings".format(
            p=out_path,
            nn=len(snapshot["nodes"]),
            nt=len(snapshot["type_parents"]),
            ne=len(snapshot["edges"]),
            em=len(snapshot.get("embeddings", {})),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
