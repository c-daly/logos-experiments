"""Re-score the representations we ALREADY embedded with the metric that works.

Runs 1-2 scored representations with the intrinsic battery, which the positive
control proved non-diagnostic. The metric that actually discriminates is the one
that separated control (silhouette 0.213) from the junk-drawer (0.06): the
best-cut silhouette of the SYSTEM'S OWN agglomerative clustering. So re-score
every cached arm with (silhouette, chunk_ratio) jointly -- does any representation
we already have lift cluster separation toward 0.213 WITHOUT just re-importing
chunk identity? No new embeddings.

    python rescore.py    # sophia venv (pymilvus + numpy); arms are cached
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from sophia.maintenance.emergence_clustering import (
    _agglomerative_partitions,
    _distance_matrix,
    _silhouette,
)
from run import embed_cached, load_name_vectors, nn_chunk_rate
from represent import REPS
from decouple import chunk_centered, chunk_residual
from gloss import attach_glosses, load_glosses

HERE = Path(__file__).resolve().parent
SAMPLE = 800  # match _MAX_CLUSTER_INPUT


def best_cut_silhouette(X, seed=0):
    work = X if len(X) <= SAMPLE else X[random.Random(seed).sample(range(len(X)), SAMPLE)]
    dm = _distance_matrix([list(map(float, v)) for v in work])
    parts = _agglomerative_partitions(dm, 2, max(2, len(work) // 3))
    if not parts:
        return None, None
    k, lab = max(parts.items(), key=lambda kv: _silhouette(dm, kv[1]))
    return int(k), round(_silhouette(dm, lab), 4)


def main():
    sample = json.loads((HERE / "sample.json").read_text())
    uuids = [r["uuid"] for r in sample]
    chunk_ids = [r["raw_text"] for r in sample]
    attach_glosses(sample, load_glosses())

    name_vecs = load_name_vectors(uuids)
    name = np.asarray([name_vecs[u] for u in uuids], dtype="float32")
    ns = embed_cached([REPS["name_sentence"](r) for r in sample])
    chunk_vecs = embed_cached([r["raw_text"] for r in sample])

    arms = {
        "name": name,
        "sentence": embed_cached([REPS["sentence"](r) for r in sample]),
        "name_sentence": ns,
        "marked": embed_cached([REPS["marked"](r) for r in sample]),
        "chunk_centered": chunk_centered(ns, chunk_ids),
        "chunk_residual": chunk_residual(ns, chunk_vecs),
        "gloss": embed_cached([REPS["gloss"](r) for r in sample]),
        "name_gloss": embed_cached([REPS["name_gloss"](r) for r in sample]),
    }

    print(f"{'arm':<16}{'best_k':>8}{'silhouette':>12}{'chunk_ratio':>13}")
    print(f"{'(control)':<16}{'':>8}{'0.213':>12}{'':>13}")
    print(f"{'(junk-drawer)':<16}{'':>8}{'0.060':>12}{'':>13}")
    out = {}
    for arm, X in arms.items():
        k, sil = best_cut_silhouette(X)
        cr = nn_chunk_rate(X, chunk_ids)["ratio"]
        out[arm] = {"best_k": k, "silhouette": sil, "chunk_ratio": cr}
        print(f"{arm:<16}{k!s:>8}{sil!s:>12}{cr!s:>13}")
    (HERE / "rescore.json").write_text(json.dumps(out, indent=2))
    print("\nwrote rescore.json")


if __name__ == "__main__":
    main()
