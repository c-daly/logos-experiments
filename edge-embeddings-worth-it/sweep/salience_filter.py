"""Rank + filter substrate entities by TF-IDF salience, to cut extraction noise
(one-off dates/numbers, generic cross-topic terms) while KEEPING the recurring,
topic-distinctive multi-context entities.

salience(entity) = tf * idf
  tf  = number of context sentences (recurrence)
  idf = log(N_articles / #articles the entity appears in)   <-- TOPIC-level docs,
        NOT sentences: sentence-level idf would wrongly penalise recurrence.
Pure-numeric / no-letter surface forms (dates, "10 km") are dropped outright.

Post-processing step on a substrate (no re-extraction). Needs the corpus to map
each context sentence back to its source article.

Usage:
  poetry run python salience_filter.py --substrate substrate.json --corpus corpus.jsonl \
      --out filtered.json [--keep-frac 0.5 | --min-salience X]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_HAS_LETTER = re.compile(r"[A-Za-z]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--min-salience", type=float, default=None)
    a = ap.parse_args(argv)

    sub = json.loads(a.substrate.read_text())
    corpus = [json.loads(ln) for ln in a.corpus.read_text().splitlines() if ln.strip()]
    src = {c["text"]: c.get("source", "_") for c in corpus}
    n_art = len(set(src.values())) or 1

    scored = []
    dropped_numeric = 0
    for e in sub["entities"]:
        name = e.get("name", "")
        if not _HAS_LETTER.search(name):  # pure number / date / symbol
            dropped_numeric += 1
            continue
        ctx = e.get("context_sentences", [])
        if not ctx:
            continue
        arts = {src.get(s, "_") for s in ctx}
        idf = math.log(n_art / len(arts)) if arts else 0.0
        e["salience"] = round(len(ctx) * idf, 4)
        scored.append(e)

    scored.sort(key=lambda e: -e["salience"])
    if a.min_salience is not None:
        kept = [e for e in scored if e["salience"] >= a.min_salience]
    else:
        kept = scored[: max(1, int(len(scored) * a.keep_frac))]

    out = {
        "meta": {**sub.get("meta", {}), "source": "salience-filtered",
                 "n_entities": len(kept)},
        "sentence_embeddings": sub["sentence_embeddings"],
        "entities": kept,
    }
    a.out.write_text(json.dumps(out))
    print(f"input {len(sub['entities'])} entities -> dropped {dropped_numeric} numeric, "
          f"kept {len(kept)} after salience")
    print("top salient:")
    for e in kept[:18]:
        print(f"  {e['salience']:6.2f}  {e['name']:32} ctx={e['n_contexts']}")
    print("bottom of kept:")
    for e in kept[-6:]:
        print(f"  {e['salience']:6.2f}  {e['name']:32} ctx={e['n_contexts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
