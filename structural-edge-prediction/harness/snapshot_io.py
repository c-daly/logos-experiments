"""Load + validate the frozen-graph snapshot schema, and walk its IS_A backbone.

The snapshot is the single interchange format shared by the toy fixture, the
live exporter (``freeze_snapshot.py``), and every offline arm. Schema::

    {
      "nodes":        [{"id": str, "type": str | None, "label": str}],
      "type_parents": {type_id: parent_type_id | None},
      "edges":        [{"src": str, "rel": str, "dst": str}],
      "embeddings":   {node_id: [float, ...]}   # OPTIONAL
    }

``nodes[].type`` is the node\u0027s immediate IS_A parent type (may be ``None``).
``type_parents`` is the type->parent hierarchy used to POOL members up the
IS_A chain. ``edges`` are the RELATIONAL edges among nodes \u2014 the only thing
ever held out; the IS_A backbone (node.type + type_parents) is always known.

Validation is loud: a dangling edge, an unknown node type, or a type-parent
cycle raises rather than degrading silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SnapshotValidationError(ValueError):
    """The snapshot violates the schema invariants (fail loudly, never repair)."""


@dataclass
class Snapshot:
    """A validated frozen graph: nodes, the IS_A backbone, relational edges."""

    nodes: list[dict[str, Any]]
    type_parents: dict[str, str | None]
    edges: list[dict[str, str]]
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    # Derived indices (built by load_snapshot/from_dict after validation).
    _node_type: dict[str, str | None] = field(default_factory=dict)
    _node_ids: set[str] = field(default_factory=set)

    def has_embeddings(self) -> bool:
        """True iff at least one node carries an embedding vector."""
        return bool(self.embeddings)


def _validate(snap: Snapshot) -> None:
    """Raise :class:`SnapshotValidationError` on any schema violation."""
    node_ids = {n["id"] for n in snap.nodes}
    if len(node_ids) != len(snap.nodes):
        raise SnapshotValidationError("duplicate node id(s) in snapshot.nodes")

    type_ids = set(snap.type_parents)

    # Every node.type is a known type id (or None).
    for n in snap.nodes:
        t = n.get("type")
        if t is not None and t not in type_ids:
            node_id = n["id"]
            raise SnapshotValidationError(
                f"node {node_id!r} has unknown type {t!r} "
                "(not in type_parents)"
            )

    # Every parent reference resolves to a known type id (or None).
    for t, parent in snap.type_parents.items():
        if parent is not None and parent not in type_ids:
            raise SnapshotValidationError(
                f"type {t!r} has unknown parent {parent!r}"
            )

    # No cycles in the type hierarchy.
    for start in type_ids:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise SnapshotValidationError(
                    f"cycle in type_parents reachable from {start!r}"
                )
            seen.add(cur)
            cur = snap.type_parents.get(cur)

    # Every edge src/dst is a known node.
    for e in snap.edges:
        for endpoint in ("src", "dst"):
            if e[endpoint] not in node_ids:
                raise SnapshotValidationError(
                    f"edge {e!r} references unknown node {e[endpoint]!r}"
                )


def load_snapshot(path: str | Path) -> Snapshot:
    """Load and validate a snapshot JSON file.

    Raises:
        SnapshotValidationError: on any schema violation (dangling edge,
            unknown node type, or a cycle in the type hierarchy).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> Snapshot:
    """Build + validate a :class:`Snapshot` from an in-memory dict."""
    snap = Snapshot(
        nodes=list(raw.get("nodes", [])),
        type_parents=dict(raw.get("type_parents", {})),
        edges=list(raw.get("edges", [])),
        embeddings={
            k: list(v) for k, v in (raw.get("embeddings") or {}).items()
        },
    )
    _validate(snap)
    snap._node_type = {n["id"]: n.get("type") for n in snap.nodes}
    snap._node_ids = set(snap._node_type)
    return snap


def node_type(snapshot: Snapshot, node_id: str) -> str | None:
    """The immediate IS_A type of a node (or ``None`` if untyped/unknown)."""
    return snapshot._node_type.get(node_id)


def type_chain(snapshot: Snapshot, type_id: str | None) -> list[str]:
    """Walk type_parents from ``type_id`` up to the root, most-specific first.

    ``[type_id, parent, ..., root]``. An empty list for ``None``. Validation
    has already proven the hierarchy acyclic, so the walk always terminates.
    """
    chain: list[str] = []
    cur = type_id
    while cur is not None:
        chain.append(cur)
        cur = snapshot.type_parents.get(cur)
    return chain


def members_of(
    snapshot: Snapshot, type_id: str, *, include_subtypes: bool = True
) -> list[str]:
    """Node ids whose IS_A chain passes through ``type_id``.

    With ``include_subtypes`` (default), pools every node whose chain contains
    ``type_id`` \u2014 i.e. members of the type AND of all its descendants. With
    ``include_subtypes=False``, only nodes whose IMMEDIATE type is ``type_id``.
    Result is sorted for determinism.
    """
    out: list[str] = []
    for n in snapshot.nodes:
        t = n.get("type")
        if t is None:
            continue
        if include_subtypes:
            if type_id in type_chain(snapshot, t):
                out.append(n["id"])
        elif t == type_id:
            out.append(n["id"])
    return sorted(out)
