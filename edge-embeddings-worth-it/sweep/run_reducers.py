"""Compare context REDUCERS over the per-entity context substrate.

Substrate comes from rejoin_from_corpus.py: a top-level ``sentence_embeddings``
dict + per-entity ``context_sentences``. For each reducer in reducers.REGISTRY we
build entity vectors, cluster them, and report INTRINSIC fitness (silhouette, no
labels) -- and eyeball one reducer's clusters. This is the "try lots of things
with the dict" harness; new reducers (incl. learned transforms) drop into the
registry and show up here.

Usage:
  poetry run python run_reducers.py --fixture fixture_rejoined.json [--eyeball centroid]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


sw = _load("sweep", HERE / "sweep.py")
R = _load("reducers", HERE / "reducers.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--preproc", default="pca50")
    ap.add_argument("--algo", default="agglomerative_avg")
    ap.add_argument("--eyeball", default="centroid")
    a = ap.parse_args(argv)

    fx = json.loads(a.fixture.read_text())
    SE = fx.get("sentence_embeddings") or {}
    if not SE:
        raise SystemExit("fixture has no sentence_embeddings substrate -- re-run rejoin")
    ents = fx["entities"]
    multi = sum(1 for e in ents if len(e.get("context_sentences", [])) > 1)
    print(f"substrate: {len(SE)} distinct sentence vectors; "
          f"{len(ents)} entities ({multi} multi-context)")
    print(f"\n{'reducer':10} {'silhouette':>10} {'n_clusters':>11} {'n_ent':>6}")
    print("-" * 40)

    cached = {}
    for name, fn in R.REGISTRY.items():
        X, names = [], []
        for e in ents:
            ctx = [SE[s] for s in e.get("context_sentences", []) if s in SE]
            if not ctx:
                continue
            X.append(fn(ctx))
            names.append(e["name"])
        Xa = np.asarray(X, dtype=float)
        x = sw.preprocess(Xa, a.preproc)
        labels = np.asarray(
            sw._apply_min_size(
                np.asarray(sw._node_labels(a.algo, x, X, 5, 2, "silhouette", {})), 2
            )
        )
        sil = sw.silhouette_cosine(x, labels)
        nclust = len({int(v) for v in labels if v != -1})
        print(f"{name:10} {sil:10.3f} {nclust:11d} {len(names):6d}")
        cached[name] = (labels, names)

    if a.eyeball in cached:
        labels, names = cached[a.eyeball]
        byc: dict[int, list[str]] = defaultdict(list)
        for n, lab in zip(names, labels):
            byc[int(lab)].append(n)
        print(f"\n=== eyeball: {a.eyeball} ({len(byc)} clusters) ===")
        for c in sorted(byc, key=lambda c: -len(byc[c])):
            tag = "noise" if c == -1 else f"c{c}"
            print(f"  [{tag}] n={len(byc[c])}: {', '.join(sorted(byc[c]))[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
