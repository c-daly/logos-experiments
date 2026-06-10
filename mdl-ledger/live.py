"""W1 live runner: price the current HCG (logos-experiments#26).

Read-only. Builds a Snapshot from the live graph (membership from IS_A
edges with the deterministic lexicographic tie-break, realm-kind fallback;
type hierarchy from type_definition -> type_definition IS_A edges;
non-typing data-node edges) and prints the ledger report. Credentials per
the harness convention: NEO4J_PASSWORD required, no default.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from ledger import Snapshot, compute_ledger

TYPING_RELATIONS = {"IS_A", "INSTANCE_OF", "SUBTYPE_OF"}
NON_DATA_KINDS = {"edge", "type_definition"}


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit(
            "NEO4J_PASSWORD must be set explicitly for the MDL ledger "
            "(no default credential; harness convention)."
        )
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_snapshot(driver) -> Snapshot:
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
    dropped = len(raw) - len(edges)
    if dropped:
        print(f"warn: dropped {dropped} malformed edge node(s)", file=sys.stderr)

    # membership: IS_A target name (lexicographic min on ties), kind fallback.
    # Typing is IS_A-only by design; INSTANCE_OF/SUBTYPE_OF are excluded from
    # data edges as typing-shaped extraction artifacts (B6 territory) but are
    # NOT membership sources. Verified on the live graph 2026-06-10 (zero
    # such edges target a type_definition); warn if that assumption breaks.
    shadow_typing = sum(
        1
        for rel, _, tgt in edges
        if rel in ("INSTANCE_OF", "SUBTYPE_OF")
        and (rec := nodes.get(tgt)) is not None
        and rec[0] == "type_definition"
    )
    if shadow_typing:
        print(
            f"warn: {shadow_typing} INSTANCE_OF/SUBTYPE_OF edge(s) target a "
            "type_definition -- membership only follows IS_A, so these are "
            "not counted; review whether the typing model changed",
            file=sys.stderr,
        )
    candidates: dict[str, list[str]] = {}
    type_parent_candidates: dict[str, list[str]] = {}
    for rel, src, tgt in edges:
        if rel != "IS_A":
            continue
        src_rec, tgt_rec = nodes.get(src), nodes.get(tgt)
        if src_rec is None or tgt_rec is None:
            continue
        if tgt_rec[0] == "type_definition" and tgt_rec[1]:
            if src_rec[0] not in NON_DATA_KINDS:
                candidates.setdefault(src, []).append(tgt_rec[1])
            elif src_rec[0] == "type_definition" and src_rec[1]:
                type_parent_candidates.setdefault(src_rec[1], []).append(tgt_rec[1])

    membership = {
        u: (kind if u not in candidates else min(candidates[u]))
        for u, (kind, _) in nodes.items()
        if kind not in NON_DATA_KINDS
    }
    type_parents: dict[str, str | None] = {
        name: None for kind, name in nodes.values() if kind == "type_definition" and name
    }
    for t, parents in type_parent_candidates.items():
        type_parents[t] = min(parents)

    data_edges = tuple(
        (rel, src, tgt)
        for rel, src, tgt in edges
        if rel not in TYPING_RELATIONS
        and src in membership
        and tgt in membership
    )
    return Snapshot(membership, type_parents, data_edges)


def main() -> None:
    driver = _driver()
    try:
        snap = fetch_snapshot(driver)
    finally:
        driver.close()
    rep = compute_ledger(snap)
    rep["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
