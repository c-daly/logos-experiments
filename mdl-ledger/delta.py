"""W1 ontology operations as pure Snapshot transforms (logos-experiments#26).

Each op returns a function Snapshot -> Snapshot; ledger.delta(s, op) prices
it. Nothing here writes anywhere -- pricing an operation and applying it to
the live graph are different acts, and only the first one lives in this
project.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ledger import Snapshot

Op = Callable[[Snapshot], Snapshot]


def merge_types(loser: str, winner: str) -> Op:
    """Retype every member of `loser` to `winner`; drop `loser`.

    If `loser` was `winner`'s parent, `winner` inherits `loser`'s parent
    rather than becoming its own parent (review #36)."""

    def apply(s: Snapshot) -> Snapshot:
        membership = {
            u: (winner if t == loser else t) for u, t in s.membership.items()
        }
        loser_parent = s.type_parents.get(loser)
        type_parents = {}
        for t, p in s.type_parents.items():
            if t == loser:
                continue
            if p == loser:
                p = loser_parent if t == winner else winner
            type_parents[t] = p
        return Snapshot(membership, type_parents, s.edges)

    return apply


def evict_type(t: str) -> Op:
    """Drop type `t`; members fall to its parent (or to their realm root
    if `t` is parentless)."""

    def apply(s: Snapshot) -> Snapshot:
        parent = s.type_parents.get(t)
        fallback = parent if parent is not None else "entity"
        if fallback == t:
            raise ValueError(
                f"cannot evict root type {t!r}: members would fall back to "
                "the evicted type itself (review #36)"
            )
        membership = {
            u: (fallback if tt == t else tt) for u, tt in s.membership.items()
        }
        type_parents = {
            tt: (fallback if p == t else p)
            for tt, p in s.type_parents.items()
            if tt != t
        }
        return Snapshot(membership, type_parents, s.edges)

    return apply


def mint_type(name: str, member_uuids: Iterable[str], parent: str) -> Op:
    """Create type `name` under `parent` and move the given members into it."""
    moved = set(member_uuids)

    def apply(s: Snapshot) -> Snapshot:
        membership = {
            u: (name if u in moved else t) for u, t in s.membership.items()
        }
        type_parents = dict(s.type_parents)
        type_parents[name] = parent
        return Snapshot(membership, type_parents, s.edges)

    return apply


def graft(t: str, new_parent: str) -> Op:
    """Re-parent type `t`. v1 prices this at dL = 0 (the encoding counts
    hierarchy edges, not positions) -- a documented v1 blindness; a
    position-aware refinement must beat v1 on W2 to replace it."""

    def apply(s: Snapshot) -> Snapshot:
        type_parents = dict(s.type_parents)
        type_parents[t] = new_parent
        return Snapshot(s.membership, type_parents, s.edges)

    return apply
