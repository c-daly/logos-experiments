"""Re-join experiment provenance from the CORPUS by CONTENT, fixing the async-timing
join bug (run_experiment.py attributes domain+sentence by polling order, ~76% wrong).

For each fixture entity, find the corpus line(s) whose text contains the entity name
(word-boundary match) -> the TRUE domain (majority vote) + the actual source
sentence(s) -> averaged context vector (multi-context). The name embedding is kept
unchanged; only `domain` and `context_embedding` are re-derived. No re-ingest.

Output is a corrected fixture; run sweep.py on it for an honest result.

Usage:
  poetry run python rejoin_from_corpus.py --fixture <in> --corpus ../corpus/corpus.jsonl --out <out>
Env: HERMES_URL (default http://localhost:17000)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

HERMES = os.environ.get("HERMES_URL", "http://localhost:17000")
_cache: dict[str, list[float]] = {}


def embed(text: str) -> list[float]:
    if text in _cache:
        return _cache[text]
    last: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(f"{HERMES}/embed_text", json={"text": text}, timeout=120)
            r.raise_for_status()
            v = list(r.json()["embedding"])
            _cache[text] = v
            return v
        except Exception as e:  # transient upstream
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"embed failed after retries: {last}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    fx = json.loads(a.fixture.read_text())
    corpus = [json.loads(ln) for ln in a.corpus.read_text().splitlines() if ln.strip()]
    texts = [c["text"] for c in corpus]
    domains = [c["domain"] for c in corpus]
    pats = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in texts]  # unused placeholder
    line_emb = [embed(t) for t in texts]  # cached; ~corpus distinct lines
    print(f"[rejoin] embedded {len(_cache)} distinct corpus lines")

    def lines_for(name: str) -> list[int]:
        nl = (name or "").lower().strip()
        if not nl:
            return []
        # word-boundary match avoids 'u'/'x' matching everywhere; fall back to
        # substring for names with no word chars (symbols like ∇f, ∑).
        if re.search(r"\w", nl):
            rx = re.compile(r"\b" + re.escape(nl) + r"\b")
            return [i for i, t in enumerate(texts) if rx.search(t.lower())]
        return [i for i, t in enumerate(texts) if nl in t.lower()]

    ents = []
    dropped = 0
    multi = 0
    for e in fx["entities"]:
        idx = lines_for(e.get("name", ""))
        if not idx:
            dropped += 1
            continue
        doms = [domains[i] for i in idx]
        dom = Counter(doms).most_common(1)[0][0]
        ctx_avg = list(np.mean(np.asarray([line_emb[i] for i in idx], dtype=float), axis=0))
        if len(idx) > 1:
            multi += 1
        ents.append(
            {
                **e,
                "domain": dom,
                "context_embedding": ctx_avg,  # back-compat: centroid of contexts
                "context_sentences": [texts[i] for i in idx],  # SUBSTRATE keys
                "n_contexts": len(idx),
                "context_domains": sorted(set(doms)),
            }
        )

    out = {
        "meta": {
            **fx.get("meta", {}),
            "source": "corpus-rejoined",
            "n_entities": len(ents),
            "n_edges": len(fx.get("edges", [])),
            "rejoin_dropped_no_namematch": dropped,
            "rejoin_multi_context": multi,
        },
        # SUBSTRATE: the {sentence: embedding} dict. Each entity references its
        # contexts by sentence text (context_sentences); reducers map that set
        # of vectors to a usable entity embedding (centroid / transform / sense).
        "sentence_embeddings": {texts[i]: line_emb[i] for i in range(len(texts))},
        "entities": ents,
        "edges": fx.get("edges", []),
    }
    a.out.write_text(json.dumps(out))
    print(
        f"[rejoin] wrote {a.out}: {len(ents)} entities "
        f"({dropped} dropped no-match, {multi} multi-context)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
