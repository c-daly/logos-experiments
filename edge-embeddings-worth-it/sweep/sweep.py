"""SWEEP: offline clustering-config grid over a captured fixture.

The cheap half of the capture+sweep split. Reads ``fixture.json`` (entity +
edge embeddings already materialised by ``capture.py``) and runs a grid of
clustering configs PURELY in-process -- no Hermes, no Neo4j, no Milvus -- so
many configs can be tried without re-ingesting.

NODE grid (clustering the entity embeddings vs ``domain`` ground truth):
  algorithm    : agglomerative_avg | agglomerative_complete | agglomerative_ward
                 | kmeans | cosine_kmeans | hdbscan (skipped if not installed)
                 | hierarchy_rollup (reuses
                   sophia.maintenance.emergence_clustering.find_emergent_hierarchy)
  preprocessing: raw | l2norm | pca50
  min_cluster_size : 2 | 3 | 5
  k            : n_domains | silhouette-best (where the algo needs a k)
  score        : adjusted_rand_score + purity vs domain; n_clusters; combined
                 score penalises |n_clusters - n_domains|.

EDGE grid (clustering edge embeddings vs relation label):
  scheme : relationship_label | triple | name
  algo   : agglomerative_avg | kmeans | hdbscan
  preproc: raw | l2norm
  score  : relation-label purity; merge_ratio = distinct_relation_labels /
           n_clusters; endpoint-type homogeneity.

scikit-learn is used WHEN AVAILABLE; otherwise pure-numpy fallbacks (cosine
k-means + a Lance-Williams agglomeration that matches the cosine/Ward geometry
the shipped emergence code uses) keep the sweep runnable. Optional deps probed
at import: sklearn, hdbscan, umap.

Usage:
  poetry run python sweep.py --fixture fixture.json --out results.json
  poetry run python sweep.py --selftest        # synthetic end-to-end check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent

# --- optional dependency probes ------------------------------------------
HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAVE_HDBSCAN = importlib.util.find_spec("hdbscan") is not None
HAVE_UMAP = importlib.util.find_spec("umap") is not None

_SKIP_LOG: list[str] = []


def _skip(msg: str) -> None:
    if msg not in _SKIP_LOG:
        _SKIP_LOG.append(msg)


# =========================================================================
# Scoring metrics (pure numpy -- no sklearn dependency)
# =========================================================================


def _comb2(x: float) -> float:
    return x * (x - 1) / 2.0


def adjusted_rand_score(true: list, pred: list) -> float:
    """ARI -- numpy reimplementation (matches sklearn.adjusted_rand_score)."""
    classes = sorted(set(true))
    clusters = sorted(set(pred))
    ci = {c: i for i, c in enumerate(classes)}
    ki = {c: i for i, c in enumerate(clusters)}
    cont = np.zeros((len(classes), len(clusters)))
    for t, p in zip(true, pred):
        cont[ci[t], ki[p]] += 1
    sum_comb = sum(_comb2(v) for v in cont.flatten())
    a, b = cont.sum(1), cont.sum(0)
    sa = sum(_comb2(v) for v in a)
    sb = sum(_comb2(v) for v in b)
    total = _comb2(len(true))
    exp = sa * sb / total if total else 0.0
    mx = (sa + sb) / 2
    return 0.0 if mx == exp else (sum_comb - exp) / (mx - exp)


def purity(true: list, pred: list) -> float:
    """Fraction of points in the majority class of their assigned cluster."""
    groups: dict = {}
    for t, p in zip(true, pred):
        groups.setdefault(p, []).append(t)
    if not groups:
        return 0.0
    return sum(Counter(v).most_common(1)[0][1] for v in groups.values()) / len(true)


def silhouette_cosine(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine silhouette; -1.0 if fewer than 2 populated clusters.

    Fully vectorised: per-point a/b come from point->cluster distance sums
    (``dist @ masks.T``), never a Python loop over points.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    xn = _l2norm(x)
    dist = 1.0 - (xn @ xn.T)
    np.fill_diagonal(dist, 0.0)
    masks = (labels[None, :] == uniq[:, None]).astype(float)  # (k, n)
    sizes = masks.sum(axis=1)  # (k,)
    sum_to = dist @ masks.T  # (n, k): sum dist from point i to all of cluster c
    own = labels[:, None] == uniq[None, :]  # (n, k) bool
    own_size = own.astype(float) @ sizes  # (n,) size of each point's own cluster
    own_sum = (sum_to * own).sum(axis=1)  # (n,) sum dist to own cluster (self=0)
    a = np.where(own_size > 1, own_sum / np.maximum(own_size - 1.0, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_to = sum_to / sizes[None, :]  # (n, k) mean dist to each cluster
    b = np.where(own, np.inf, mean_to).min(axis=1)  # nearest OTHER cluster
    b = np.where(np.isfinite(b), b, 0.0)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)
    return float(s.mean())


# =========================================================================
# Preprocessing
# =========================================================================


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    k_eff = min(k, x.shape[0], x.shape[1])
    if HAVE_SKLEARN:
        from sklearn.decomposition import PCA

        return np.asarray(PCA(n_components=k_eff, random_state=0).fit_transform(xc))
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return u[:, :k_eff] * s[:k_eff]


def preprocess(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x
    if mode == "l2norm":
        return _l2norm(x)
    if mode == "pca50":
        return _pca(x, 50)
    raise ValueError(f"unknown preprocessing {mode}")


# =========================================================================
# Component fusion (label/name + context) -- the "combine" experiment
# =========================================================================

# Weight on the FIRST component (label / name); (1 - alpha) on context.
COMBINE_ALPHAS = [0.3, 0.5, 0.7]
COMBINE_METHODS = ["weighted", "concat"]


def _combine(a: np.ndarray, b: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """Fuse two embedding blocks, each L2-normalised first so neither dominates.

    weighted: alpha*a_hat + (1-alpha)*b_hat   -> stays d-dim
    concat  : [alpha*a_hat | (1-alpha)*b_hat] -> 2d-dim (clusterer weights via distance)
    """
    an, bn = _l2norm(a), _l2norm(b)
    if method == "concat":
        return np.concatenate([alpha * an, (1.0 - alpha) * bn], axis=1)
    return alpha * an + (1.0 - alpha) * bn


# =========================================================================
# Clustering primitives (sklearn when present, numpy fallback otherwise)
# =========================================================================


def _cosine_kmeans_np(x: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    xn = _l2norm(x)
    rng = np.random.default_rng(seed)
    centroids = xn[rng.choice(len(xn), k, replace=False)].copy()
    labels = np.zeros(len(xn), dtype=int)
    for _ in range(iters):
        new = np.argmax(xn @ centroids.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = xn[labels == c]
            if len(pts):
                centroids[c] = pts.mean(0)
                centroids[c] /= np.linalg.norm(centroids[c]) + 1e-9
    return labels


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if HAVE_SKLEARN:
        from sklearn.cluster import KMeans

        return np.asarray(
            KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
        )
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(50):
        d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = x[labels == c]
            if len(pts):
                centroids[c] = pts.mean(0)
    return labels


def _agglom_linkage_np(
    x: np.ndarray, linkage: str
) -> list[tuple[int, int]]:
    """Lance-Williams agglomeration -> ordered merge list. O(n^2) overall.

    Cosine distance for average/complete; squared-euclidean (Ward) otherwise.
    Each merge is a pair of *original-point* representative ids (lower kept);
    replaying the first ``n - k`` merges via union-find yields a k-cut.
    """
    n = len(x)
    if linkage == "ward":
        d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1).astype(float)
    else:
        xn = _l2norm(x)
        d = (1.0 - (xn @ xn.T)).astype(float)
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    active = list(range(n))
    merges: list[tuple[int, int]] = []

    while len(active) > 1:
        sub = d[np.ix_(active, active)]
        flat = int(np.argmin(sub))
        ai, bi = divmod(flat, len(active))
        i, j = active[ai], active[bi]
        if i > j:
            i, j = j, i
        merges.append((i, j))
        si, sj = size[i], size[j]
        for m in active:
            if m == i or m == j:
                continue
            if linkage == "complete":
                nd = max(d[i, m], d[j, m])
            elif linkage == "ward":
                sm = size[m]
                nd = (
                    (si + sm) * d[i, m] + (sj + sm) * d[j, m] - sm * d[i, j]
                ) / (si + sj + sm)
            else:  # average
                nd = (si * d[i, m] + sj * d[j, m]) / (si + sj)
            d[i, m] = nd
            d[m, i] = nd
        size[i] = si + sj
        active.remove(j)

    return merges


def _labels_from_merges(
    n: int, merges: list[tuple[int, int]], k: int
) -> np.ndarray:
    """Replay the first ``n - k`` merges via union-find -> k-cut labels."""
    k = max(1, min(k, n))
    par = list(range(n))

    def find(a: int) -> int:
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for step in range(max(0, n - k)):
        i, j = merges[step]
        ri, rj = find(i), find(j)
        if ri != rj:
            par[max(ri, rj)] = min(ri, rj)
    roots: dict[int, int] = {}
    labels = np.zeros(n, dtype=int)
    for idx in range(n):
        r = find(idx)
        if r not in roots:
            roots[r] = len(roots)
        labels[idx] = roots[r]
    return labels


def _agglomerative(x: np.ndarray, k: int, linkage: str) -> np.ndarray:
    """Agglomerative clustering; linkage in {average, complete, ward}."""
    if HAVE_SKLEARN:
        from sklearn.cluster import AgglomerativeClustering

        metric = "euclidean" if linkage == "ward" else "cosine"
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
        return np.asarray(model.fit_predict(x))
    merges = _agglom_linkage_np(x, linkage)
    return _labels_from_merges(len(x), merges, k)


def _hdbscan(x: np.ndarray, min_cluster_size: int) -> np.ndarray | None:
    if not HAVE_HDBSCAN:
        _skip("hdbscan: not installed -- skipped")
        return None
    import hdbscan  # type: ignore

    model = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size))
    return np.asarray(model.fit_predict(x))


def _best_k_by_silhouette(
    x: np.ndarray, fn: Callable[[int], np.ndarray], lo: int, hi: int
) -> tuple[int, np.ndarray]:
    best: tuple[float, int, np.ndarray] | None = None
    for k in range(lo, hi + 1):
        labels = fn(k)
        score = silhouette_cosine(x, labels)
        if best is None or score > best[0]:
            best = (score, k, labels)
    assert best is not None
    return best[1], best[2]


# =========================================================================
# hierarchy_rollup -- reuse the SHIPPED emergence hierarchy
# =========================================================================


def _hierarchy_rollup_labels(
    embeddings: list[list[float]], min_cluster_size: int
) -> np.ndarray | None:
    """Run find_emergent_hierarchy and flatten its LEAF clusters into labels.

    Points not placed into any leaf cluster get label -1 (noise/unclustered).
    Returns None if sophia is not importable (e.g. running outside its env).
    """
    try:
        from sophia.maintenance.emergence_clustering import find_emergent_hierarchy
        from sophia.maintenance.emergence_types import Member
    except Exception as exc:  # pragma: no cover - env dependent
        _skip(f"hierarchy_rollup: sophia import failed ({exc}) -- skipped")
        return None

    members = [
        Member(
            uuid=str(i),
            name=str(i),
            embedding=list(e),
            signature=Counter(),
            current_type="entity",
            hermes_type_hint=None,
            neighbors=[],
            model=None,
        )
        for i, e in enumerate(embeddings)
    ]
    roots = find_emergent_hierarchy(
        members,
        min_cluster_size=min_cluster_size,
        variance_threshold=0.0,
    )

    leaves: list[Any] = []

    def _collect(node: Any) -> None:
        if not node.children:
            leaves.append(node)
        else:
            for c in node.children:
                _collect(c)

    for r in roots:
        _collect(r)

    labels = np.full(len(embeddings), -1, dtype=int)
    for lab, leaf in enumerate(leaves):
        for m in leaf.members:
            labels[int(m.name)] = lab
    return labels


# =========================================================================
# Node-clustering grid
# =========================================================================


NODE_ALGOS = [
    "agglomerative_avg",
    "agglomerative_complete",
    "agglomerative_ward",
    "kmeans",
    "cosine_kmeans",
    "hdbscan",
    "hierarchy_rollup",
]
PREPROCS_NODE = ["raw", "l2norm", "pca50"]
MIN_SIZES = [2, 3, 5]
K_MODES = ["n_domains", "silhouette"]

_LINKAGE = {
    "agglomerative_avg": "average",
    "agglomerative_complete": "complete",
    "agglomerative_ward": "ward",
}


def _node_labels(
    algo: str,
    x: np.ndarray,
    raw_embeddings: list[list[float]],
    n_domains: int,
    min_size: int,
    k_mode: str,
    merge_cache: dict[str, list[tuple[int, int]]],
    seed: int = 0,
) -> np.ndarray | None:
    """Return per-point integer labels for one node config, or None if skipped."""
    n = len(x)
    # Silhouette k-sweep upper bound -- capped so the grid stays fast: real
    # cluster counts sit near n_domains, so a window of ~2x n_domains brackets
    # the optimum without paying for high-k silhouettes on raw 1536-dim data.
    k_ceiling = max(12, 2 * n_domains + 3)
    hi = max(2, min(n - 1, n // max(1, min_size), k_ceiling))
    k_fixed = max(2, min(n_domains, n - 1))

    if algo == "hdbscan":
        return _hdbscan(x, min_size)
    if algo == "hierarchy_rollup":
        return _hierarchy_rollup_labels(raw_embeddings, min_size)

    if algo == "kmeans":
        if k_mode == "silhouette":
            _, labels = _best_k_by_silhouette(x, lambda k: _kmeans(x, k, seed), 2, hi)
            return labels
        return _kmeans(x, k_fixed, seed)
    if algo == "cosine_kmeans":
        if k_mode == "silhouette":
            _, labels = _best_k_by_silhouette(
                x, lambda k: _cosine_kmeans_np(x, k, seed), 2, hi
            )
            return labels
        return _cosine_kmeans_np(x, k_fixed, seed)

    linkage = _LINKAGE[algo]
    if HAVE_SKLEARN:
        cut = lambda k: _agglomerative(x, k, linkage)  # noqa: E731
    else:
        if linkage not in merge_cache:
            merge_cache[linkage] = _agglom_linkage_np(x, linkage)
        merges = merge_cache[linkage]
        cut = lambda k: _labels_from_merges(n, merges, k)  # noqa: E731
    if k_mode == "silhouette":
        _, labels = _best_k_by_silhouette(x, cut, 2, hi)
        return labels
    return cut(k_fixed)


def _apply_min_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Relabel clusters smaller than ``min_size`` to noise (-1)."""
    out = labels.copy()
    counts = Counter(int(v) for v in labels)
    for i, v in enumerate(labels):
        if int(v) != -1 and counts[int(v)] < min_size:
            out[i] = -1
    return out


def score_node_config(
    labels: np.ndarray, domains: list[str], n_domains: int
) -> dict[str, Any]:
    clustered = [i for i, lab in enumerate(labels) if int(lab) != -1]
    n = len(labels)
    coverage = len(clustered) / n if n else 0.0
    if len(clustered) < 2:
        return {
            "n_clusters": 0,
            "coverage": coverage,
            "ari": float("nan"),
            "purity": float("nan"),
            "combined": float("-inf"),
        }
    pred = [int(labels[i]) for i in clustered]
    true = [domains[i] for i in clustered]
    n_clusters = len(set(pred))
    ari = adjusted_rand_score(true, pred)
    pur = purity(true, pred)
    penalty = abs(n_clusters - n_domains) / max(1, n_domains)
    combined = ari + 0.25 * pur - 0.3 * penalty
    return {
        "n_clusters": n_clusters,
        "coverage": coverage,
        "ari": ari,
        "purity": pur,
        "combined": combined,
    }


def _node_feature_schemes(
    entities: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Feature variants for the node sweep: name-only, context-only, and fused.

    Each variant rewrites every entity's ``embedding`` to the scheme's vector so
    the existing node grid runs unchanged per scheme. Fused schemes (name+ctx)
    only appear when entities carry a ``context_embedding``.
    """
    schemes: list[tuple[str, list[dict[str, Any]]]] = [("name", entities)]
    ctx = [e for e in entities if e.get("context_embedding") is not None]
    if len(ctx) >= 4:
        schemes.append(
            ("context", [{**e, "embedding": e["context_embedding"]} for e in ctx])
        )
        name_mat = np.asarray([e["embedding"] for e in ctx], dtype=float)
        ctx_mat = np.asarray([e["context_embedding"] for e in ctx], dtype=float)
        for method in COMBINE_METHODS:
            for alpha in COMBINE_ALPHAS:
                fused = _combine(name_mat, ctx_mat, method, alpha)
                variant = [
                    {**e, "embedding": fused[i].tolist()} for i, e in enumerate(ctx)
                ]
                schemes.append((f"name+ctx:{method}:a{alpha}", variant))
    return schemes


def run_node_sweep_schemes(
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Run the node grid for every feature scheme; tag each row with ``scheme``."""
    all_rows: list[dict[str, Any]] = []
    n_domains = 0
    for label, ents in _node_feature_schemes(entities):
        rows, n_domains = run_node_sweep(ents)
        for r in rows:
            r["scheme"] = label
        all_rows.extend(rows)
    all_rows.sort(
        key=lambda r: (
            r["ari"] if r["ari"] == r["ari"] else -1e9,
            r["combined"] if r["combined"] == r["combined"] else -1e9,
        ),
        reverse=True,
    )
    return all_rows, n_domains


def run_node_sweep(
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    embeddings = [e["embedding"] for e in entities]
    domains = [e.get("domain", "unknown") for e in entities]
    n_domains = len({d for d in domains if d != "unknown"}) or len(set(domains))
    x0 = np.asarray(embeddings, dtype=float)

    rows: list[dict[str, Any]] = []
    for preproc in PREPROCS_NODE:
        x = preprocess(x0, preproc)
        merge_cache: dict[str, list[tuple[int, int]]] = {}
        for algo in NODE_ALGOS:
            k_modes = (
                ["n_domains"]
                if algo in ("hdbscan", "hierarchy_rollup")
                else K_MODES
            )
            for min_size in MIN_SIZES:
                for k_mode in k_modes:
                    try:
                        labels = _node_labels(
                            algo,
                            x,
                            embeddings,
                            n_domains,
                            min_size,
                            k_mode,
                            merge_cache,
                        )
                    except Exception as exc:
                        _skip(f"{algo}/{preproc}: error {exc}")
                        labels = None
                    if labels is None:
                        continue
                    labels = _apply_min_size(labels, min_size)
                    score = score_node_config(labels, domains, n_domains)
                    rows.append(
                        {
                            "algorithm": algo,
                            "preprocessing": preproc,
                            "min_cluster_size": min_size,
                            "k_mode": k_mode,
                            **score,
                        }
                    )
    rows.sort(
        key=lambda r: (
            r["ari"] if r["ari"] == r["ari"] else -1e9,
            r["combined"] if r["combined"] == r["combined"] else -1e9,
        ),
        reverse=True,
    )
    return rows, n_domains


# =========================================================================
# Edge-clustering grid
# =========================================================================


EDGE_SCHEMES = ["relationship_label", "triple", "name"]
EDGE_ALGOS = ["agglomerative_avg", "kmeans", "hdbscan"]
EDGE_PREPROCS = ["raw", "l2norm"]


def _endpoint_homogeneity(labels: np.ndarray, edges: list[dict[str, Any]]) -> float:
    """Mean per-cluster purity of the (src_type, tgt_type) endpoint signature."""
    groups: dict[int, list[tuple[str, str]]] = {}
    for lab, e in zip(labels, edges):
        if int(lab) == -1:
            continue
        sig = (str(e.get("src_type")), str(e.get("tgt_type")))
        groups.setdefault(int(lab), []).append(sig)
    if not groups:
        return float("nan")
    scores = [Counter(v).most_common(1)[0][1] / len(v) for v in groups.values() if v]
    return float(np.mean(scores)) if scores else float("nan")


def _inject_edge_combine_schemes(edges: list[dict[str, Any]]) -> None:
    """Add fused RELATIONSHIP-label + context edge embeddings in-place.

    New scheme keys ``label+ctx:<method>:a<alpha>`` join the existing
    relationship_label / triple / name schemes in the edge sweep.
    """
    have = [
        e
        for e in edges
        if (e.get("embeddings") or {}).get("relationship_label") is not None
        and (e.get("embeddings") or {}).get("context") is not None
    ]
    if len(have) < 4:
        return
    for method in COMBINE_METHODS:
        for alpha in COMBINE_ALPHAS:
            key = f"label+ctx:{method}:a{alpha}"
            for e in have:
                emb = e["embeddings"]
                a = np.asarray([emb["relationship_label"]], dtype=float)
                b = np.asarray([emb["context"]], dtype=float)
                emb[key] = _combine(a, b, method, alpha)[0].tolist()


def _edge_scheme_list(edges: list[dict[str, Any]]) -> list[str]:
    extra = sorted(
        {
            k
            for e in edges
            for k in (e.get("embeddings") or {})
            if k.startswith("label+ctx")
        }
    )
    return list(EDGE_SCHEMES) + extra


def run_edge_sweep(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not edges:
        return rows
    n_relation_labels = len({str(e.get("relation")) for e in edges})

    for scheme in _edge_scheme_list(edges):
        idx = [i for i, e in enumerate(edges) if scheme in (e.get("embeddings") or {})]
        if len(idx) < 4:
            _skip(f"edge scheme {scheme}: <4 edges with this embedding -- skipped")
            continue
        sub_edges = [edges[i] for i in idx]
        vecs = np.asarray([edges[i]["embeddings"][scheme] for i in idx], dtype=float)
        relations = [str(edges[i].get("relation")) for i in idx]
        n_rel = len(set(relations))
        k = max(2, min(n_rel, len(idx) - 1))

        for preproc in EDGE_PREPROCS:
            x = preprocess(vecs, preproc)
            for algo in EDGE_ALGOS:
                try:
                    if algo == "hdbscan":
                        labels = _hdbscan(x, 2)
                    elif algo == "kmeans":
                        labels = _kmeans(x, k, 0)
                    else:
                        labels = _agglomerative(x, k, "average")
                except Exception as exc:
                    _skip(f"edge {algo}/{scheme}/{preproc}: error {exc}")
                    labels = None
                if labels is None:
                    continue
                clustered = [i for i, lab in enumerate(labels) if int(lab) != -1]
                if len(clustered) < 2:
                    continue
                pred = [int(labels[i]) for i in clustered]
                true = [relations[i] for i in clustered]
                n_clusters = len(set(pred))
                rel_purity = purity(true, pred)
                merge_ratio = n_relation_labels / n_clusters if n_clusters else 0.0
                homog = _endpoint_homogeneity(labels, sub_edges)
                rows.append(
                    {
                        "scheme": scheme,
                        "algorithm": algo,
                        "preprocessing": preproc,
                        "n_clusters": n_clusters,
                        "relation_purity": rel_purity,
                        "merge_ratio": merge_ratio,
                        "endpoint_homogeneity": homog,
                    }
                )
    rows.sort(
        key=lambda r: (
            r["relation_purity"],
            r["endpoint_homogeneity"]
            if r["endpoint_homogeneity"] == r["endpoint_homogeneity"]
            else -1e9,
        ),
        reverse=True,
    )
    return rows


# =========================================================================
# Report rendering
# =========================================================================


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.3f}"
    return str(v)


def render_report(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    n_domains: int,
) -> str:
    lines: list[str] = []
    lines.append("# Clustering parameter-sweep report")
    lines.append("")
    src = meta.get("source", "unknown")
    n_ent = meta.get("n_entities")
    n_edg = meta.get("n_edges")
    lines.append(f"- source: {src}")
    lines.append(f"- entities: {n_ent}  edges: {n_edg}")
    lines.append(f"- n_domains (node ground truth): {n_domains}")
    lines.append(
        f"- deps: sklearn={HAVE_SKLEARN} hdbscan={HAVE_HDBSCAN} umap={HAVE_UMAP}"
    )
    lines.append("")

    lines.append("## Node configs (ranked by ARI vs domain)")
    lines.append("")
    hdr = [
        "scheme", "algorithm", "preproc", "min", "k_mode", "n_cl",
        "cover", "ARI", "purity", "combined",
    ]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in node_rows[:25]:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("scheme", "name"),
                    r["algorithm"],
                    r["preprocessing"],
                    str(r["min_cluster_size"]),
                    r["k_mode"],
                    str(r["n_clusters"]),
                    _fmt(r["coverage"]),
                    _fmt(r["ari"]),
                    _fmt(r["purity"]),
                    _fmt(r["combined"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Edge configs (ranked by relation-label purity)")
    lines.append("")
    ehdr = [
        "scheme", "algorithm", "preproc", "n_cl",
        "rel_purity", "merge_ratio", "endpoint_homog",
    ]
    lines.append("| " + " | ".join(ehdr) + " |")
    lines.append("|" + "|".join(["---"] * len(ehdr)) + "|")
    for r in edge_rows[:25]:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["scheme"],
                    r["algorithm"],
                    r["preprocessing"],
                    str(r["n_clusters"]),
                    _fmt(r["relation_purity"]),
                    _fmt(r["merge_ratio"]),
                    _fmt(r["endpoint_homogeneity"]),
                ]
            )
            + " |"
        )
    lines.append("")

    if _SKIP_LOG:
        lines.append("## Skipped / notes")
        lines.append("")
        for s in _SKIP_LOG:
            lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines)


# =========================================================================
# Orchestration
# =========================================================================


def run_sweep(fixture: dict[str, Any], out: Path) -> dict[str, Any]:
    entities = fixture.get("entities", [])
    edges = fixture.get("edges", [])
    meta = fixture.get("meta", {})

    node_rows, n_domains = run_node_sweep_schemes(entities) if entities else ([], 0)
    _inject_edge_combine_schemes(edges)
    edge_rows = run_edge_sweep(edges)

    results = {
        "meta": meta,
        "n_domains": n_domains,
        "deps": {
            "sklearn": HAVE_SKLEARN,
            "hdbscan": HAVE_HDBSCAN,
            "umap": HAVE_UMAP,
        },
        "node_results": node_rows,
        "edge_results": edge_rows,
        "skipped": _SKIP_LOG,
    }
    out.write_text(json.dumps(results, indent=2))
    report = render_report(node_rows, edge_rows, meta, n_domains)
    (out.parent / "REPORT.md").write_text(report)
    return results


# =========================================================================
# Synthetic self-test
# =========================================================================


def _build_synthetic_fixture(
    n_blobs: int = 6, per_blob: int = 20, dim: int = 1536, seed: int = 0
) -> dict[str, Any]:
    """6 unit-normalised gaussian blobs (~20 pts each) + a little noise.

    Blob means are well separated on random directions; small intra-blob
    spread, so a sound clustering should recover ~n_blobs with ARI > 0.9.
    A few edge groups are synthesised for the edge sweep.
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_blobs, dim))
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9

    entities: list[dict[str, Any]] = []
    for b in range(n_blobs):
        for j in range(per_blob):
            v = means[b] + 0.04 * rng.normal(size=dim)
            v /= np.linalg.norm(v) + 1e-9
            entities.append(
                {
                    "uuid": f"e{b}_{j}",
                    "name": f"blob{b}_pt{j}",
                    "domain": f"domain_{b}",
                    "type": "entity",
                    "embedding": v.tolist(),
                }
            )
    for j in range(6):
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v) + 1e-9
        entities.append(
            {
                "uuid": f"noise_{j}",
                "name": f"noise{j}",
                "domain": "noise",
                "type": "entity",
                "embedding": v.tolist(),
            }
        )

    rel_means = rng.normal(size=(3, dim))
    rel_means /= np.linalg.norm(rel_means, axis=1, keepdims=True) + 1e-9
    relations = ["CAUSES", "PART_OF", "LOCATED_IN"]
    types = [("animal", "place"), ("part", "whole"), ("thing", "place")]
    edges: list[dict[str, Any]] = []
    for ri, rel in enumerate(relations):
        for j in range(8):
            base = rel_means[ri] + 0.05 * rng.normal(size=dim)
            base /= np.linalg.norm(base) + 1e-9
            edges.append(
                {
                    "uuid": f"edge_{rel}_{j}",
                    "relation": rel,
                    "name": f"{rel} edge {j}",
                    "src_uuid": f"s{j}",
                    "tgt_uuid": f"t{j}",
                    "src_name": f"src{j}",
                    "tgt_name": f"tgt{j}",
                    "src_type": types[ri][0],
                    "tgt_type": types[ri][1],
                    "embeddings": {
                        "relationship_label": base.tolist(),
                        "triple": base.tolist(),
                        "name": base.tolist(),
                    },
                }
            )

    domains = sorted({e["domain"] for e in entities})
    return {
        "meta": {
            "source": "synthetic-selftest",
            "n_entities": len(entities),
            "n_edges": len(edges),
            "n_domains": len(domains),
            "domains": domains,
            "dim": dim,
        },
        "entities": entities,
        "edges": edges,
    }


def selftest(out_dir: Path) -> int:
    fixture = _build_synthetic_fixture()
    (out_dir / "selftest_fixture.json").write_text(json.dumps(fixture))
    results = run_sweep(fixture, out_dir / "selftest_results.json")

    node_rows = results["node_results"]
    if not node_rows:
        print("SELFTEST FAIL: no node configs produced", file=sys.stderr)
        return 1
    winner = node_rows[0]
    w_algo = winner["algorithm"]
    w_pp = winner["preprocessing"]
    w_min = winner["min_cluster_size"]
    w_km = winner["k_mode"]
    w_ncl = winner["n_clusters"]
    w_ari = winner["ari"]
    w_pur = winner["purity"]
    w_comb = winner["combined"]
    print("=== synthetic self-test ===")
    print(
        f"winner: algo={w_algo} preproc={w_pp} min={w_min} k_mode={w_km}"
    )
    print(
        f"        n_clusters={w_ncl} ARI={w_ari:.4f} "
        f"purity={w_pur:.4f} combined={w_comb:.4f}"
    )
    report_path = out_dir / "REPORT.md"
    print(f"report written: {report_path} (exists={report_path.exists()})")

    ok_ari = winner["ari"] > 0.9
    ok_nclust = 5 <= winner["n_clusters"] <= 7
    ok_report = report_path.exists()
    if ok_ari and ok_nclust and ok_report:
        print("SELFTEST PASS")
        return 0
    print(
        f"SELFTEST FAIL: ari>0.9={ok_ari} n_clusters~6={ok_nclust} report={ok_report}",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline clustering parameter sweep.")
    ap.add_argument("--fixture", type=Path, help="Path to fixture.json")
    ap.add_argument(
        "--out", type=Path, default=HERE / "results.json", help="Output results.json"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="Run the synthetic end-to-end self-test instead of a real fixture.",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(HERE)

    if not args.fixture:
        ap.error("--fixture is required unless --selftest is given")
    fixture = json.loads(args.fixture.read_text())
    results = run_sweep(fixture, args.out)
    n_node = len(results["node_results"])
    n_edge = len(results["edge_results"])
    print(f"node configs: {n_node}  edge configs: {n_edge}  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
