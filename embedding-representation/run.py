"""Score each representation arm with the label-free battery + a chunk-collapse
diagnostic, reusing embeddings we already have.

Arms:
  name           -- pulled FREE from the live hcg_entity_embeddings (the actual
                    production vectors, by uuid). No re-embed.
  sentence       -- mention's sentence (new strings, embedded via API, cached)
  name_sentence  -- "{name} — {sentence}"
  marked         -- sentence with the mention wrapped in «»

THE confound (see represent.py): ~15 entities per chunk, 100% chunk-shared. So
besides the battery we report nn_chunk_rate -- the fraction of each entity's k
nearest neighbours that sit in its *same raw_text chunk*, vs the rate expected
if neighbours were random. A representation whose neighbourhoods are just
chunk-mates is measuring chunk geometry, not entity geometry -- a FALSE
coherence we must rule out.

    OPENAI_API_KEY=... python run.py     # run under a venv with pymilvus
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import battery as B
from represent import REPS

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)
MODEL = "text-embedding-3-large"
DIM = 3072


def load_name_vectors(uuids: list[str]) -> dict[str, list[float]]:
    """Production name vectors, free, from the live collection."""
    from pymilvus import Collection, connections

    connections.connect("default", host="localhost", port="19530")
    col = Collection("hcg_entity_embeddings")
    col.load()
    out: dict[str, list[float]] = {}
    for i in range(0, len(uuids), 500):
        expr = "uuid in [%s]" % ",".join(f'"{u}"' for u in uuids[i : i + 500])
        for r in col.query(expr=expr, output_fields=["uuid", "embedding"]):
            out[r["uuid"]] = r["embedding"]
    return out


def _openai(texts: list[str], chunk: int = 1000) -> np.ndarray:
    import httpx

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    out: list[list[float]] = []
    for i in range(0, len(texts), chunk):
        batch = texts[i : i + chunk]
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            json={"model": MODEL, "input": batch, "dimensions": DIM},
            headers={"Authorization": f"Bearer {key}"},
            timeout=180,
        )
        resp.raise_for_status()
        out.extend(it["embedding"] for it in sorted(resp.json()["data"], key=lambda d: d["index"]))
        print(f"    embedded {min(i + chunk, len(texts))}/{len(texts)}")
    return np.asarray(out, dtype="float32")


def embed_cached(texts: list[str]) -> np.ndarray:
    uniq = sorted(set(texts))
    key = hashlib.sha1("\n".join(uniq).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"{key}.npy"
    if path.exists():
        vecs = np.load(path)
    else:
        vecs = _openai(uniq)
        np.save(path, vecs)
    index = {t: r for r, t in enumerate(uniq)}
    return vecs[[index[t] for t in texts]]


def nn_chunk_rate(X, chunk_ids, k: int = 10, sample: int = 3000, seed: int = 5):
    """Fraction of each point's k-NN that share its chunk, vs the random-neighbour
    expectation. ratio >> 1 means neighbourhoods are chunk-driven (the confound)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = len(Xn)
    cids = np.asarray(chunk_ids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sample, n), replace=False)
    sims = Xn[idx] @ Xn.T
    sims[np.arange(len(idx)), idx] = -np.inf
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    observed = float((cids[nn] == cids[idx][:, None]).mean())
    cnt = Counter(chunk_ids)
    expected = sum(c * (c - 1) for c in cnt.values()) / (n * (n - 1))
    return {
        "nn_same_chunk": round(observed, 3),
        "expected_random": round(expected, 3),
        "ratio": round(observed / expected, 1) if expected else None,
    }


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    uuids = [r["uuid"] for r in sample]
    chunk_ids = [r["raw_text"] for r in sample]
    print(f"sample: {len(sample)} entities  |  model: {MODEL}/{DIM}")

    results = {"model": MODEL, "dim": DIM, "n": len(sample), "arms": {}}

    print("[name] loading production vectors from hcg_entity_embeddings (free) ...")
    name_vecs = load_name_vectors(uuids)
    arm_vecs = {"name": np.asarray([name_vecs[u] for u in uuids], dtype="float32")}

    for arm in ("sentence", "name_sentence", "marked"):
        texts = [REPS[arm](r) for r in sample]
        uniq = len(set(texts))
        print(f"[{arm}] {uniq} unique / {len(texts)} nodes; embedding (cached) ...")
        arm_vecs[arm] = embed_cached(texts)

    for arm, X in arm_vecs.items():
        print(f"[{arm}] scoring battery (1 SVD) ...")
        scored = B.score(X)
        scored["chunk_collapse"] = nn_chunk_rate(X, chunk_ids)
        results["arms"][arm] = scored

    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nwrote results.json")
    _print_table(results)


def _print_table(results: dict) -> None:
    cols = ["anisotropy_centroid", "nn_margin", "pairwise_cos_spread", "effective_rank", "intrinsic_dim_twonn", "hopkins"]
    print("\n=== battery (raw) — vs the name baseline ===")
    print(f"{'arm':<14}" + "".join(f"{c[:11]:>13}" for c in cols) + f"{'chunk_ratio':>13}")
    for arm, d in results["arms"].items():
        r = d["raw"]
        cc = d["chunk_collapse"]["ratio"]
        print(f"{arm:<14}" + "".join(f"{r[c]:>13}" for c in cols) + f"{cc!s:>13}")
    print("\n=== whitened control (nn_margin flat => no hidden structure) ===")
    for arm, d in results["arms"].items():
        w = d["whitened"]
        print(f"{arm:<14} nn_margin={w['nn_margin']:<8} anisotropy={w['anisotropy_centroid']:<8} spread={w['pairwise_cos_spread']}")
    print("\n=== chunk-collapse (nn_same_chunk vs random; high ratio = measuring chunks not entities) ===")
    for arm, d in results["arms"].items():
        cc = d["chunk_collapse"]
        print(f"{arm:<14} same_chunk={cc['nn_same_chunk']:<7} random={cc['expected_random']:<7} ratio={cc['ratio']}")


if __name__ == "__main__":
    main()
