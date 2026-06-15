"""Type signatures: ``P(rel, dst_type | src_type)`` over the training edges.

A type is the reified IS_A edge; its SIGNATURE is the distribution of relational
edges its members tend to have. We build one signature per type by pooling
edges whose source is a member of that type (members pooled UP the IS_A chain
via ``members_of``), then reading off the observed frequency of each
``(rel, dst_type)`` pattern.

Backoff: a node gets a signature at EACH level of its IS_A chain (its own
type, its supertype, ..., the root). The scorer picks the most-specific type
with enough support and backs off upward when a pattern is unseen there
(see ``harness/scorers.py``). Sharpness = negative Shannon entropy of the
signature distribution \u2014 a sharp (low-entropy) signature concentrates its
mass on a few patterns.
"""

from __future__ import annotations

import math
from typing import Any

from harness.snapshot_io import Snapshot, members_of, node_type

# A signature pattern key is (rel, dst_type). dst_type may be None (untyped dst).
Pattern = tuple[str, str | None]


class Signature:
    """One type\u0027s relational fingerprint: P(pattern) plus pooled edge count."""

    __slots__ = ("type_id", "probs", "counts", "total")

    def __init__(
        self,
        type_id: str,
        counts: dict[Pattern, int],
    ) -> None:
        self.type_id = type_id
        self.counts: dict[Pattern, int] = dict(counts)
        self.total: int = sum(counts.values())
        self.probs: dict[Pattern, float] = (
            {p: c / self.total for p, c in counts.items()}
            if self.total
            else {}
        )

    def prob(self, pattern: Pattern) -> float:
        """Observed probability of ``pattern`` for this type (0.0 if unseen)."""
        return self.probs.get(pattern, 0.0)

    def support(self) -> int:
        """Number of pooled training edges backing this signature."""
        return self.total


def build_signatures(
    train_edges: list[dict[str, str]], snapshot: Snapshot
) -> dict[str, Signature]:
    """Compute ``P(rel, dst_type | src_type==T)`` for every type T.

    For each type T, pools all TRAIN edges whose source is a member of T
    (members include subtypes, via ``members_of``), then counts each
    ``(rel, dst_type)`` pattern. This is observed-frequency only \u2014 no
    smoothing, no training. A type with no member-sourced train edges gets an
    empty (zero-support) signature.
    """
    # Index train edges by their source node for a single pass per type.
    edges_by_src: dict[str, list[dict[str, str]]] = {}
    for e in train_edges:
        edges_by_src.setdefault(e["src"], []).append(e)

    signatures: dict[str, Signature] = {}
    for type_id in snapshot.type_parents:
        counts: dict[Pattern, int] = {}
        for member in members_of(snapshot, type_id, include_subtypes=True):
            for e in edges_by_src.get(member, []):
                pattern: Pattern = (e["rel"], node_type(snapshot, e["dst"]))
                counts[pattern] = counts.get(pattern, 0) + 1
        signatures[type_id] = Signature(type_id, counts)
    return signatures


def signature_entropy(sig: Signature) -> float:
    """Shannon entropy (bits) of a signature\u0027s (rel, dst_type) distribution.

    A zero-support signature has entropy 0.0 by convention (no spread because
    there is nothing). Lower entropy = sharper.
    """
    if not sig.probs:
        return 0.0
    return -sum(p * math.log2(p) for p in sig.probs.values() if p > 0.0)


def signature_sharpness(sig: Signature) -> float:
    """Sharpness = negative entropy. Sharper signatures score higher.

    Bounded above by 0.0 (a degenerate one-pattern signature). The more spread
    the signature, the more negative. A zero-support signature is treated as
    minimally sharp (very negative) so it never wins a sharpness ranking; it
    carries no predictive evidence.
    """
    if not sig.probs:
        return float("-inf")
    return -signature_entropy(sig)


def most_specific_signature(
    snapshot: Snapshot,
    signatures: dict[str, Signature],
    type_id: str | None,
    *,
    min_support: int = 1,
) -> Signature | None:
    """Walk the IS_A chain from ``type_id`` and return the first signature
    meeting ``min_support``. ``None`` if the whole chain is unsupported.

    This is the backoff entry point: most-specific type first, then up.
    """
    from harness.snapshot_io import type_chain

    for t in type_chain(snapshot, type_id):
        sig = signatures.get(t)
        if sig is not None and sig.support() >= min_support:
            return sig
    return None


def as_dict(sig: Signature) -> dict[str, Any]:
    """Serialize a signature for snapshot output (string pattern keys)."""
    return {
        "type_id": sig.type_id,
        "support": sig.total,
        "entropy": signature_entropy(sig),
        "sharpness": signature_sharpness(sig),
        "probs": {f"{rel}|{dst}": p for (rel, dst), p in sig.probs.items()},
    }
