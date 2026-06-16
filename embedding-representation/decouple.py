"""Run 2 -- decouple entity signal from the shared-chunk signal.

Run 1 showed context arms cluster by source chunk (~58% chunk-mate neighbours).
Cross-mention averaging is a no-op here (names don't recur: 1.04 mentions/name),
so we remove the chunk component *directly* from the name_sentence vectors:

  chunk_centered  -- subtract each chunk's mean vector (remove the shared-chunk
                     centroid; what's left is the entity's deviation from its
                     chunk). Free, reuses the name_sentence cache.
  chunk_residual  -- project out the raw_text (whole-passage) embedding from each
                     entity vector. Removes the chunk's semantic direction.

If a decoupled arm keeps nn_margin up while chunk_ratio falls toward the name
floor (37x), there is entity signal under the chunk. If margin collapses with
chunk_ratio, the context arm's apparent coherence WAS the chunk.

    OPENAI_API_KEY=... python decouple.py   # embedding-eval venv (httpx+numpy)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import battery as B
from represent import REPS
from run import embed_cached, nn_chunk_rate

HERE = Path(__file__).resolve().parent


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def chunk_centered(X, chunk_ids):
    cids = np.asarray(chunk_ids)
    out = X.astype("float32").copy()
    for c in np.unique(cids):
        m = cids == c
        out[m] -= X[m].mean(axis=0)
    return out


def chunk_residual(X, chunk_vecs):
    """Remove, per entity, the component of X along its chunk vector."""
    cv = _norm(chunk_vecs)
    proj = (X * cv).sum(axis=1, keepdims=True) * cv
    return (X - proj).astype("float32")


def main():
    sample = json.loads((HERE / "sample.json").read_text())
    chunk_ids = [r["raw_text"] for r in sample]

    print("loading name_sentence vectors (cache) ...")
    ns = embed_cached([REPS["name_sentence"](r) for r in sample])
    print("embedding raw_text (chunk) vectors (cache; 344 unique) ...")
    chunk_vecs = embed_cached([r["raw_text"] for r in sample])

    arms = {
        "name_sentence": ns,
        "chunk_centered": chunk_centered(ns, chunk_ids),
        "chunk_residual": chunk_residual(ns, chunk_vecs),
    }
    results = {"model": "text-embedding-3-large", "dim": 3072, "n": len(sample), "arms": {}}
    for arm, X in arms.items():
        print(f"[{arm}] scoring ...")
        scored = B.score(X)
        scored["chunk_collapse"] = nn_chunk_rate(X, chunk_ids)
        results["arms"][arm] = scored

    (HERE / "results_decouple.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nwrote results_decouple.json\n")
    cols = ["anisotropy_centroid", "nn_margin", "pairwise_cos_spread", "effective_rank", "intrinsic_dim_twonn"]
    print(f"{'arm':<16}" + "".join(f"{c[:11]:>13}" for c in cols) + f"{'chunk_ratio':>13}")
    for arm, d in results["arms"].items():
        r = d["raw"]
        print(f"{arm:<16}" + "".join(f"{r[c]:>13}" for c in cols) + f"{d['chunk_collapse']['ratio']!s:>13}")


if __name__ == "__main__":
    main()
