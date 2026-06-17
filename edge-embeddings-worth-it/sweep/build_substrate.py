"""Build the multi-context SUBSTRATE from a sentence corpus, using HERMES's OWN
extractor + embedder (thin OpenAI wrappers) -- no new prompt, no proposal queue.

Extraction: hermes.combined_extractor (in-process) -> entities (with spans).
Embedding:  hermes /embed_text (the running server).
Corpus is sentence-level, so an entity's contexts = the sentences that contain it.
No labels.

MUST run inside the hermes poetry env (imports hermes.*); hermes server up for /embed_text.
Output -> run_reducers.py:
  {meta, sentence_embeddings:{sentence:[emb]}, entities:[{name,name_embedding,context_sentences,n_contexts}]}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from hermes.combined_extractor import get_combined_instance

HERMES = os.environ.get("HERMES_URL", "http://localhost:17000")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def embed_one(text: str):
    for _ in range(4):
        try:
            r = requests.post(f"{HERMES}/embed_text", json={"text": text}, timeout=60)
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception:
            time.sleep(2)
    return None


def embed_many(texts: list[str], workers: int = 8) -> list:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(embed_one, texts))


async def extract_all(sentences: list[str], workers: int) -> list[list[str]]:
    extractor = get_combined_instance()
    sem = asyncio.Semaphore(workers)

    async def one(s: str) -> list[str]:
        async with sem:
            try:
                ents = await extractor.extract_entities(s)
                return [e for e in ents if e.get("name")]
            except Exception as exc:
                print("  [extract err]", str(exc)[:80])
                return []

    return await asyncio.gather(*[one(s) for s in sentences])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-contexts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    corpus = [json.loads(ln) for ln in a.corpus.read_text().splitlines() if ln.strip()]
    sentences = list(dict.fromkeys(c["text"] for c in corpus))
    print(f"extracting from {len(sentences)} sentences via hermes combined_extractor ...")
    per = asyncio.run(extract_all(sentences, a.workers))

    ent_ctx: dict[str, set] = defaultdict(set)
    surface: dict[str, str] = {}
    etype: dict[str, object] = {}
    evalue: dict[str, object] = {}
    eunit: dict[str, object] = {}
    for s, ents in zip(sentences, per):
        for e in ents:
            nm = (e.get("name") or "").strip()
            n = norm(nm)
            if len(n) < 2:
                continue
            ent_ctx[n].add(s)
            surface.setdefault(n, nm)
            etype.setdefault(n, e.get("type"))
            evalue.setdefault(n, e.get("value"))
            eunit.setdefault(n, e.get("unit"))
    ent_ctx = {n: c for n, c in ent_ctx.items() if len(c) >= a.min_contexts}
    names = list(ent_ctx)
    print(f"{len(names)} distinct entities; embedding sentences + names via /embed_text ...")

    sent_vecs = embed_many(sentences, a.workers)
    name_vecs = embed_many([surface[n] for n in names], a.workers)
    sent_emb = {s: v for s, v in zip(sentences, sent_vecs) if v is not None}
    name_emb = {n: v for n, v in zip(names, name_vecs) if v is not None}

    entities = [
        {
            "name": surface[n],
            "type": etype.get(n),
            "value": evalue.get(n),
            "unit": eunit.get(n),
            "name_embedding": name_emb.get(n),
            "context_sentences": sorted(c for c in ent_ctx[n] if c in sent_emb),
            "n_contexts": len(ent_ctx[n]),
        }
        for n in names
        if n in name_emb
    ]
    entities = [e for e in entities if e["context_sentences"]]
    multi = sum(1 for e in entities if len(e["context_sentences"]) > 1)
    out = {
        "meta": {
            "source": "hermes-extracted",
            "n_sentences": len(sent_emb),
            "n_entities": len(entities),
            "multi_context": multi,
        },
        "sentence_embeddings": sent_emb,
        "entities": entities,
    }
    a.out.write_text(json.dumps(out))
    print(f"wrote {a.out}: {len(entities)} entities ({multi} multi-context), "
          f"{len(sent_emb)} sentence vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
