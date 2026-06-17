"""Positive control: do bare-name embeddings work AT ALL on known structure?

We ruled out the input representation and chunk size; the intrinsic battery looks
"weak" on our ~5000 all-topics entities (anisotropy 0.41, nn_margin 1.11). Either
embeddings can't do this, or we were measuring the wrong thing. This pushes
known-category bare names through the SAME path (text-embedding-3-large/3072) and
asks two things:

1. Do they SEPARATE? (cosine silhouette by category, kNN purity, within vs
   between). If yes -> bare-name embedding works; our flatness is global
   diversity, not failure.
2. Does the GLOBAL intrinsic battery on this KNOWN-clustered set look just as
   "weak" as our entities? If yes -> the global battery (anisotropy/nn_margin/
   hopkins) does NOT diagnose clusterability; only labels do, and our whole
   intrinsic-coherence frame was measuring diversity, not dysfunction.

    OPENAI_API_KEY=... python control.py    # embedding-eval venv (httpx+numpy)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import battery as B
from run import embed_cached

HERE = Path(__file__).resolve().parent

CATS = {
    "animals": "lion tiger elephant wolf zebra giraffe leopard rhinoceros kangaroo dolphin".split(),
    "countries": "France Japan Brazil Egypt Canada Norway Kenya Thailand Mexico Australia".split(),
    "math": "integral derivative matrix theorem eigenvalue topology polynomial logarithm vector geometry".split(),
    "fruits": "apple banana mango pineapple apricot strawberry blueberry watermelon papaya cherry".split(),
    "emotions": "anger joy fear sadness jealousy gratitude anxiety pride disgust hope".split(),
    "metals": "iron copper gold silver zinc nickel aluminum titanium platinum mercury".split(),
    "instruments": "violin trumpet piano guitar flute cello drums saxophone clarinet harp".split(),
    "weather": "rain snow thunder hurricane drought blizzard fog hail breeze monsoon".split(),
}


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def silhouette_cosine(Xn, labels):
    n = len(Xn)
    D = 1 - Xn @ Xn.T  # cosine distance
    labels = np.asarray(labels)
    sils = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in set(labels) if c != labels[i])
        sils.append((b - a) / max(a, b))
    return float(np.mean(sils))


def knn_purity(Xn, labels, k=5):
    labels = np.asarray(labels)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return float((labels[nn] == labels[:, None]).mean())


def main():
    names, labels = [], []
    for cat, items in CATS.items():
        names += items
        labels += [cat] * len(items)
    print(f"control: {len(names)} bare names across {len(CATS)} categories; embedding ...")
    X = embed_cached(names)
    Xn = _norm(X)

    # within vs between category cosine
    L = np.asarray(labels)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    same = L[iu[0]] == L[iu[1]]
    within = float(S[iu][same].mean())
    between = float(S[iu][~same].mean())

    sil = silhouette_cosine(Xn, labels)
    pur = knn_purity(Xn, labels)
    glob = B.score(X)["raw"]

    print("\n=== does the control SEPARATE? ===")
    print(f"  within-category cosine : {within:.3f}")
    print(f"  between-category cosine: {between:.3f}")
    print(f"  separation (within-between): {within - between:.3f}")
    print(f"  cosine silhouette       : {sil:.3f}   (>0.1 = real clusters)")
    print(f"  5-NN category purity    : {pur:.3f}   (1.0 = every neighbour same category)")

    print("\n=== GLOBAL intrinsic battery on this KNOWN-clustered set vs our entities ===")
    print(f"  {'metric':<22}{'control':>10}{'our entities':>14}")
    ours = {"anisotropy_centroid": 0.406, "nn_margin": 1.110, "pairwise_cos_spread": 0.067,
            "intrinsic_dim_twonn": 1.47, "hopkins": 0.619}
    for k in ("anisotropy_centroid", "nn_margin", "pairwise_cos_spread", "intrinsic_dim_twonn", "hopkins"):
        print(f"  {k:<22}{glob[k]:>10}{ours[k]:>14}")

    out = {"within": within, "between": between, "silhouette": sil, "knn_purity": pur,
           "global_battery": glob, "our_entities": ours}
    (HERE / "control.json").write_text(json.dumps(out, indent=2))
    print("\nwrote control.json")


if __name__ == "__main__":
    main()
