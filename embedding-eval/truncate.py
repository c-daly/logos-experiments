"""Matryoshka truncation of 3-large: re-score at smaller dims (lx39, #39).

text-embedding-3 is Matryoshka-trained -- the first N dims are a valid N-dim
embedding (equivalent to requesting ``dimensions=N`` from the API; cosine kNN is
scale-invariant, so the documented post-truncation renormalization is implicit
here). Re-scores the *cached* 3-large vectors truncated across a range of dims
to find how small the strongest model can go before it loses its edge over
3-small / the free models. Free: no new embedding, no graph, no API.

    uv run python truncate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import knn_loo_accuracy, silhouette, variance_dims

HERE = Path(__file__).resolve().parent
DIMS = [3072, 2048, 1536, 1024, 768, 512, 256, 128, 64]
# baselines from the full bake-off (results.json) for context
BASELINES = {"3-small@1536": 0.439, "mpnet@768": 0.397, "minilm@384": 0.380}


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    labels = [s["type"] for s in sample]
    full = np.load(HERE / ".cache" / "openai-3-large.npy")
    print(f"3-large cached: {full.shape}; 95%-variance at "
          f"{variance_dims(full)[0.95]} dims\n")

    rows = []
    for d in DIMS:
        Xt = full[:, :d]
        row = {
            "dim": d,
            "knn_acc": round(knn_loo_accuracy(Xt, labels), 3),
            "silhouette": round(silhouette(Xt, labels), 3),
        }
        rows.append(row)
        print(row, flush=True)

    (HERE / "truncation.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n  dim | knn_acc | silhouette")
    print("  ----+---------+-----------")
    for r in rows:
        print(f"  {r['dim']:>4} | {r['knn_acc']:>7} | {r['silhouette']}")
    print("\n  baselines: " + ", ".join(f"{k}={v}" for k, v in BASELINES.items()))


if __name__ == "__main__":
    main()
