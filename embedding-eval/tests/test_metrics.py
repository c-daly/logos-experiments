"""Unit tests for the embedding metrics on synthetic data with known geometry."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import (  # noqa: E402
    anisotropy,
    effective_rank,
    knn_loo_accuracy,
    variance_dims,
    whiten,
)


def _two_clusters(n=100, d=20, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d)) + sep  # shifted cluster
    X = np.vstack([a, b]).astype("float32")
    y = np.array([0] * n + [1] * n)
    return X, y


def test_variance_dims_concentrated():
    # variance only in the first 2 of 10 dims -> 95% var in ~2 dims
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 10)).astype("float32")
    X[:, 0] *= 50
    X[:, 1] *= 30
    X[:, 2:] *= 0.01
    vd = variance_dims(X)
    assert vd[0.95] <= 3


def test_effective_rank_low_for_concentrated():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((500, 50)).astype("float32")
    X[:, 0] *= 100  # one dominant direction
    X[:, 1:] *= 0.01  # the rest near-flat -> effective rank ~1
    assert effective_rank(X) < 3


def test_anisotropy_high_for_shared_direction():
    base = np.ones((200, 16), dtype="float32")
    noise = np.random.default_rng(3).standard_normal((200, 16)) * 0.01
    assert anisotropy(base + noise) > 0.9  # nearly collinear -> ~1


def test_anisotropy_low_for_isotropic():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((500, 64)).astype("float32")
    assert abs(anisotropy(X)) < 0.1  # high-dim gaussian -> ~0


def test_knn_separates_clusters():
    X, y = _two_clusters()
    assert knn_loo_accuracy(X, y, k=5) > 0.95


def test_knn_chance_on_random_labels():
    X, _ = _two_clusters()
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, len(X))  # random labels -> ~chance
    assert knn_loo_accuracy(X, y, k=5) < 0.7


def test_degenerate_inputs_dont_crash():
    # identical vectors / single sample must not divide-by-zero or NaN
    same = np.ones((5, 8), dtype="float32")
    assert variance_dims(same)[0.95] == 1
    assert effective_rank(same) == 1.0
    assert anisotropy(same) in (0.0, 1.0) or np.isfinite(anisotropy(same))
    one = np.ones((1, 8), dtype="float32")
    assert anisotropy(one) == 0.0
    assert knn_loo_accuracy(one, [0]) == 0.0


def test_whiten_decorrelates():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((300, 8))
    X = A @ rng.standard_normal((8, 8))  # correlated dims
    W = whiten(X)
    cov = np.cov(W, rowvar=False)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 0.1  # near-identity covariance
