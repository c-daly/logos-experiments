"""Reducers: map an entity's CONTEXT SUBSTRATE (a list of sentence embeddings) to a
usable entity vector.

The per-entity ``{sentence: embedding}`` set is the substrate; each reducer is a
swappable VIEW over it. New reducers slot into ``REGISTRY`` and become sweep
schemes automatically. The point: the entity's "embedding" is whatever reduction
you run over its contexts -- so a representation becomes a *choice* (and, later, a
*learned* / Sophia-selected one).

Each reducer takes ``ctx`` = list of raw context vectors and returns one vector.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _l2(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def centroid(ctx: list[Any]) -> np.ndarray:
    """Mean of unit-normalised contexts -> denoised topical centroid (default)."""
    return _l2(np.mean([_l2(c) for c in ctx], axis=0))


def medoid(ctx: list[Any]) -> np.ndarray:
    """The single REAL context closest to the centroid -- no blending, keeps a
    true sense vector rather than an average (robust to outlier mentions)."""
    U = np.asarray([_l2(c) for c in ctx])
    c = _l2(U.mean(axis=0))
    return U[int(np.argmax(U @ c))]


def maxpool(ctx: list[Any]) -> np.ndarray:
    """Dimension-wise max over unit contexts -- union of salient features."""
    return _l2(np.max([_l2(c) for c in ctx], axis=0))


def first(ctx: list[Any]) -> np.ndarray:
    """First/only context -- the single-context baseline (what we had before)."""
    return _l2(ctx[0])


def transform(ctx: list[Any], M: Any = None) -> np.ndarray:
    """Apply a transform M to each context, then centroid. M=None -> identity.

    This is the HOOK for a learned / RL transformation matrix: swap in an M
    fitted to maximise downstream fitness (silhouette) or task reward, and this
    reducer becomes the learned representation. See the rl-learned-representation
    memory.
    """
    U = np.asarray([_l2(c) for c in ctx])
    if M is not None:
        U = U @ np.asarray(M, dtype=float)
    return _l2(U.mean(axis=0))


# Reducers that need only the context set (no query / no fitted matrix) -- these
# become sweep schemes directly. transform() and a future sense_select(query)
# need extra args, so they're invoked explicitly rather than auto-swept.
REGISTRY = {
    "centroid": centroid,
    "medoid": medoid,
    "maxpool": maxpool,
    "first": first,
}
