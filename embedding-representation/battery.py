"""Label-free intrinsic coherence battery for the representation experiment.

Pure NumPy on an (N, D) matrix X of entity vectors. No labels: we do not score
against a "correct" type (most entities have several legitimate types). Instead
we measure whether the vectors have the internal + relative structure that
information-bearing embeddings should have.

Internal  -- is the space using its capacity?
  anisotropy_centroid   mean cosine of each vector to the global mean (cone
                        tightness). LOWER is better.
  anisotropy_pairs      mean cosine of random i!=j pairs (lx39 definition).
  effective_rank        exp(spectral entropy) of the singular spectrum. HIGHER.
  participation_ratio   (sum l)^2 / sum l^2 over squared singulars. HIGHER.
  intrinsic_dim_twonn   TwoNN manifold dim, N / sum log(r2/r1). pathological-low
                        => collapsed.
Relative  -- do neighbourhoods mean anything?
  pairwise_cos_mean/spread   spread of the cloud. more spread (std) is better.
  nn_margin             median (dist to 2nd NN)/(dist to 1st NN), cosine. >1;
                        HIGHER = crisper neighbourhoods.
  hopkins               clustering tendency; 0.5 random, ->1 clusterable.

PERF: the three spectral metrics and the whitening control all need the SAME
SVD of the centred matrix, so ``score()`` computes one thin SVD per arm (with U)
and shares it -- a 5330x3072 SVD is expensive on WSL BLAS, so doing it once
instead of four times matters.
"""

from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _top2_sims(Xn: np.ndarray, idx: np.ndarray):
    sims = Xn[idx] @ Xn.T
    sims[np.arange(len(idx)), idx] = -np.inf  # exclude self
    part = np.partition(sims, -2, axis=1)[:, -2:]
    return part.max(axis=1), part.min(axis=1)  # nearest, 2nd nearest


def anisotropy_centroid(X: np.ndarray) -> float:
    Xn = _normalize(X)
    c = Xn.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return float((Xn @ c).mean())


def pairwise_cos_stats(X: np.ndarray, n_pairs: int = 100_000, seed: int = 1):
    Xn = _normalize(X)
    n = len(Xn)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    c = (Xn[i[keep]] * Xn[j[keep]]).sum(axis=1)
    return float(c.mean()), float(c.std())


def effective_rank_from_s(s: np.ndarray) -> float:
    total = s.sum()
    if total == 0.0:
        return 1.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio_from_s(s: np.ndarray) -> float:
    lam = s**2
    if lam.sum() == 0.0:
        return 1.0
    return float(lam.sum() ** 2 / (lam**2).sum())


def variance_dims_from_s(s: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> dict:
    total = float(np.sum(s**2))
    if total == 0.0:
        return {str(t): 1 for t in thresholds}
    cum = np.cumsum(s**2) / total
    return {str(t): int(np.searchsorted(cum, t) + 1) for t in thresholds}


def nn_margin(X: np.ndarray, sample: int = 3000, seed: int = 2) -> float:
    Xn = _normalize(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xn), min(sample, len(Xn)), replace=False)
    s1, s2 = _top2_sims(Xn, idx)
    d1 = np.clip(1 - s1, 1e-9, None)
    d2 = 1 - s2
    return float(np.median(d2 / d1))


def intrinsic_dim_twonn(X: np.ndarray, sample: int = 3000, seed: int = 3) -> float:
    Xn = _normalize(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xn), min(sample, len(Xn)), replace=False)
    s1, s2 = _top2_sims(Xn, idx)
    r1 = np.sqrt(np.clip(2 - 2 * s1, 1e-12, None))
    r2 = np.sqrt(np.clip(2 - 2 * s2, 1e-12, None))
    mu = r2 / np.clip(r1, 1e-12, None)
    mu = mu[mu > 1 + 1e-9]
    return float(len(mu) / np.sum(np.log(mu))) if len(mu) else float("nan")


def hopkins(X: np.ndarray, sample: int = 200, seed: int = 4) -> float:
    Xn = _normalize(X)
    n, d = Xn.shape
    m = min(sample, n // 2)
    rng = np.random.default_rng(seed)
    ridx = rng.choice(n, m, replace=False)
    simsR = Xn[ridx] @ Xn.T
    simsR[np.arange(m), ridx] = -np.inf
    w = np.sqrt(np.clip(2 - 2 * simsR.max(axis=1), 0, None))
    lo, hi = Xn.min(axis=0), Xn.max(axis=0)
    Un = _normalize(rng.uniform(lo, hi, size=(m, d)))
    u = np.sqrt(np.clip(2 - 2 * (Un @ Xn.T).max(axis=1), 0, None))
    denom = u.sum() + w.sum()
    return float(u.sum() / denom) if denom > 0 else 0.5


def score(X: np.ndarray) -> dict:
    """Full battery + whitened control, with a single shared SVD per arm."""
    X = np.asarray(X, dtype="float32")
    if len(X) < 2:
        raise ValueError(f"score() requires at least 2 vectors; got {len(X)}")
    Xc = X - X.mean(axis=0)
    u, s, _ = np.linalg.svd(Xc, full_matrices=False)  # the one expensive op
    pm, ps = pairwise_cos_stats(X)
    raw = {
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "anisotropy_centroid": round(anisotropy_centroid(X), 4),
        "pairwise_cos_mean": round(pm, 4),
        "pairwise_cos_spread": round(ps, 4),
        "nn_margin": round(nn_margin(X), 4),
        "effective_rank": round(effective_rank_from_s(s), 1),
        "participation_ratio": round(participation_ratio_from_s(s), 1),
        "intrinsic_dim_twonn": round(intrinsic_dim_twonn(X), 2),
        "hopkins": round(hopkins(X), 3),
        "variance_dims": variance_dims_from_s(s),
    }
    # whitened control: reuse U, scale to unit variance, re-score the relatives.
    keep = s > 1e-9
    W = (u[:, keep] * np.sqrt(len(X))).astype("float32")
    wpm, wps = pairwise_cos_stats(W)
    whitened = {
        "anisotropy_centroid": round(anisotropy_centroid(W), 4),
        "pairwise_cos_spread": round(wps, 4),
        "nn_margin": round(nn_margin(W), 4),
        "intrinsic_dim_twonn": round(intrinsic_dim_twonn(W), 2),
    }
    return {"raw": raw, "whitened": whitened}
