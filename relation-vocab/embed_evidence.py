"""Name-embedding evidence for the consolidation proposer (logos-experiments#34).

A complementary evidence source to the canon/token/signature passes in
``propose.py``: for each df=1 predicate, the nearest *surviving* predicate by
embedding cosine (OpenAI ``text-embedding-3``). It catches synonyms with no
shared tokens (``AFFILIATED_WITH`` -> ``ASSOCIATED_WITH``) that the token pass
misses, and gives the otherwise evidence-less ``keep`` rows a nearest
neighbour so the table clears the >=80%-with-evidence gate.

Self-contained: embeds via the OpenAI HTTP API (``OPENAI_API_KEY``), caching
vectors under ``.cache/vectors.json`` so re-runs don't re-embed. Kept out of
``propose.py`` so the core proposer stays network-free and unit-testable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache" / "vectors.json"
MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
DIM = int(os.environ.get("EMBED_DIM", "3072"))


def load_vectors(
    preds: list[str], cache: Path = CACHE, chunk: int = 1000
) -> dict[str, list[float]]:
    """Embed predicate surfaces (cached). Only labels missing from the cache
    hit the API; the cache is keyed by surface and re-used across runs."""
    vectors: dict[str, list[float]] = {}
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        # The cache is keyed by surface only, so a model/dimension change
        # strands vectors of the old size; mixing dimensions breaks the
        # cosine matrix downstream (inhomogeneous numpy stack). Treat any
        # entry whose dimension differs from DIM as a miss and re-embed it
        # under the current config (the next batch write drops the stale
        # entries from the file).
        vectors = {
            k: v for k, v in raw.items() if isinstance(v, list) and len(v) == DIM
        }
        stale = len(raw) - len(vectors)
        if stale:
            print(
                f"  [embed_evidence] ignoring {stale} cached vector(s) with "
                f"dim != {DIM}; re-embedding the ones this run needs",
                file=sys.stderr,
            )
    missing = [p for p in preds if p not in vectors]
    if missing:
        import httpx

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                f"{len(missing)} predicate(s) are not in the cache and "
                "OPENAI_API_KEY is not set, so they cannot be embedded. Set the "
                f"key, or provide a populated cache at {cache}."
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(missing), chunk):
            batch = missing[i : i + chunk]
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                json={"model": MODEL, "input": batch, "dimensions": DIM},
                headers={"Authorization": f"Bearer {key}"},
                timeout=120,
            )
            resp.raise_for_status()
            for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                vectors[batch[item["index"]]] = item["embedding"]
            # Persist after each batch: a later batch failing on a multi-chunk
            # run must not discard vectors already paid for.
            cache.write_text(json.dumps(vectors), encoding="utf-8")
    return vectors


def nearest_survivors(
    one_offs: list[str],
    survivors: set[str],
    vectors: dict[str, list[float]],
) -> dict[str, tuple[str, float]]:
    """For each one-off with a vector, the nearest surviving predicate and its
    cosine. Survivors without a vector are skipped; a one-off with no vector or
    no survivors yields no entry (the proposer then leaves it a bare keep)."""
    import numpy as np

    surv = [s for s in sorted(survivors) if s in vectors]
    if not surv:
        return {}
    mat = np.asarray([vectors[s] for s in surv], dtype="float32")
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)

    surv_idx = {s: idx for idx, s in enumerate(surv)}
    out: dict[str, tuple[str, float]] = {}
    for p in one_offs:
        v = vectors.get(p)
        if v is None:
            continue
        vec = np.asarray(v, dtype="float32")
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            continue
        sims = mat @ (vec / norm)
        if p in surv_idx:  # a predicate is never its own consolidation target
            if len(surv) == 1:
                continue  # the lone survivor is p itself -> no target
            sims[surv_idx[p]] = -1.0
        i = int(sims.argmax())
        out[p] = (surv[i], float(sims[i]))
    return out
