"""Run the embedding bake-off (lx39, #39).

Embeds the labeled sample (sample.json) with each candidate model, then scores
geometry (effective rank, 95%-variance dims, anisotropy) and the downstream
type-classification task (cosine kNN leave-one-out, raw and whitened) plus
type-cluster silhouette. Writes results.json and prints a table.

    NEO4J_PASSWORD=... uv run python sample.py     # once, to build sample.json
    uv run python run.py                           # embed (cached) + score
"""

from __future__ import annotations

import json
from pathlib import Path

from embedders import MODELS, embed
from metrics import (
    anisotropy,
    effective_rank,
    knn_loo_accuracy,
    silhouette,
    variance_dims,
    whiten,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    texts = [s["name"] for s in sample]
    labels = [s["type"] for s in sample]
    print(f"sample: {len(texts)} nodes, {len(set(labels))} types")

    results = []
    for name, kind, model, dim, cost in MODELS:
        print(f"embedding {name} ...", flush=True)
        X = embed(name, kind, model, dim, texts)
        row = {
            "model": name,
            "dim": int(X.shape[1]),
            "eff_rank": round(effective_rank(X), 1),
            "dims_95pct_var": variance_dims(X)[0.95],
            "anisotropy": round(anisotropy(X), 3),
            "silhouette": round(silhouette(X, labels), 3),
            "knn_acc_raw": round(knn_loo_accuracy(X, labels), 3),
            "knn_acc_whitened": round(knn_loo_accuracy(whiten(X), labels), 3),
            "cost": cost,
        }
        results.append(row)
        print("  ", {k: row[k] for k in ("dim", "eff_rank", "knn_acc_raw", "knn_acc_whitened")}, flush=True)

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    cols = [
        "model", "dim", "eff_rank", "dims_95pct_var", "anisotropy",
        "silhouette", "knn_acc_raw", "knn_acc_whitened", "cost",
    ]
    width = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}
    print("\n" + " | ".join(c.ljust(width[c]) for c in cols))
    print("-+-".join("-" * width[c] for c in cols))
    for r in results:
        print(" | ".join(str(r[c]).ljust(width[c]) for c in cols))


if __name__ == "__main__":
    main()
