"""Geometry + downstream metrics for the embedding bake-off (lx39, #39).

All pure NumPy/sklearn on an (N, D) matrix X and N type labels. Two families:

- **Geometry** -- describe the vector space itself: how many dimensions carry
  the variance (``variance_dims`` / ``effective_rank``), and how anisotropic it
  is (``anisotropy`` -- the mean cosine of random pairs; high means the vectors
  sit in a narrow cone and discriminate poorly).
- **Downstream** -- does the space do the embeddings' actual job, type
  classification: ``knn_loo_accuracy`` (leave-one-out kNN, cosine) and
  ``silhouette`` (type-cluster separation). ``whiten`` decorrelates the space;
  re-scoring after it tells you whether a model's raw anisotropy was hiding
  usable structure.

The decisive evidence for "is 1536 enough?" is ``variance_dims``: if a
3072-dim model reaches 95% variance well below 1536, the extra dims are noise.
"""

from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def variance_dims(
    X: np.ndarray, thresholds: tuple[float, ...] = (0.90, 0.95, 0.99)
) -> dict[float, int]:
    """Number of principal components needed to reach each variance threshold."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    return {t: int(np.searchsorted(cum, t) + 1) for t in thresholds}


def effective_rank(X: np.ndarray) -> float:
    """Spectral entropy effective rank: exp(-sum p_i log p_i) over the
    normalized singular-value spectrum. A scale-free 'how many dimensions are
    really in use' number (<= D)."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def anisotropy(X: np.ndarray, n_pairs: int = 100_000, seed: int = 0) -> float:
    """Mean cosine similarity of random (i != j) pairs. ~0 is isotropic;
    high (0.2-0.5) means a narrow cone -- poor discrimination."""
    Xn = _normalize(X)
    n = len(Xn)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    return float(np.sum(Xn[i[keep]] * Xn[j[keep]], axis=1).mean())


def whiten(X: np.ndarray) -> np.ndarray:
    """PCA-whiten: centre, rotate to the principal axes, scale each to unit
    variance. Removes the anisotropic cone so cosine kNN sees the structure."""
    Xc = X - X.mean(axis=0)
    u, s, _ = np.linalg.svd(Xc, full_matrices=False)
    keep = s > 1e-9
    return (u[:, keep] * np.sqrt(len(X))).astype("float32")


def knn_loo_accuracy(X: np.ndarray, labels, k: int = 10) -> float:
    """Leave-one-out kNN top-1 accuracy under cosine similarity: predict each
    point's type by majority vote of its k nearest *other* points."""
    labels = np.asarray(labels)
    Xn = _normalize(X)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    k = min(k, len(X) - 1)
    nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    correct = 0
    for idx in range(len(X)):
        vals, counts = np.unique(labels[nn[idx]], return_counts=True)
        correct += vals[counts.argmax()] == labels[idx]
    return float(correct / len(X))


def silhouette(X: np.ndarray, labels) -> float:
    """Cosine silhouette of the type clustering (-1..1; higher = better
    separated). Lazily imports sklearn so the geometry metrics stay dep-light."""
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(_normalize(X), np.asarray(labels), metric="cosine"))
