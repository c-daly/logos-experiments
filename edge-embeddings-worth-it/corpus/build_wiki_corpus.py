"""Build a sentence corpus from Wikipedia for the context-reducer experiments.

Parameterised + reusable (scales to a huge corpus later with the same script):
  - topics:  explicit --titles, a curated science set (--topics N), or --random N
  - volume:  --sentences-per-topic
  - quality: --min-words (drop fragments); boilerplate sections are skipped

No labels needed -- the source article title is kept only as metadata for eyeballing.

Uses `wikipedia-api` (pip install wikipedia-api) -- API-based and requires a
User-Agent, which the unmaintained classic `wikipedia` lib can't set (Wikipedia
now blocks UA-less requests). Random titles come from the MediaWiki random list.

Usage:
  poetry run python build_wiki_corpus.py --topics 6  --sentences-per-topic 40 --out corpus_wiki.jsonl
  poetry run python build_wiki_corpus.py --titles "Mitochondrion,Neutron star" --out small.jsonl
  poetry run python build_wiki_corpus.py --random 200 --sentences-per-topic 30 --out big.jsonl   # huge
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
import wikipediaapi

UA = "logos-experiment/0.1 (https://github.com/c-daly/logos; research)"
WIKI = wikipediaapi.Wikipedia(user_agent=UA, language="en")

DEFAULT_TOPICS = [
    "Mitochondrion", "Neutron star", "Plate tectonics", "Immune system",
    "Artificial neural network", "Benzene", "Photosynthesis", "Black hole",
    "DNA replication", "Volcano", "Antibody", "Gradient descent",
    "Enzyme", "Galaxy", "Earthquake", "Transformer (deep learning architecture)",
]

_SKIP_SECTIONS = {
    "references", "see also", "external links", "further reading", "notes",
    "bibliography", "citations", "sources", "footnotes", "gallery",
    "explanatory notes", "general and cited references",
}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")


def collect_text(page) -> str:
    """Summary + all non-boilerplate section text, recursively."""
    parts = [page.summary]

    def rec(sections) -> None:
        for s in sections:
            if s.title.strip().lower() in _SKIP_SECTIONS:
                continue
            if s.text:
                parts.append(s.text)
            rec(s.sections)

    rec(page.sections)
    return "\n".join(parts)


def article_sentences(title: str, max_sentences: int, min_words: int):
    page = WIKI.page(title)
    if not page.exists():
        print(f"  [skip] {title}: page does not exist")
        return None, []
    sents: list[str] = []
    for para in collect_text(page).split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT.split(para):
            s = s.strip()
            if len(s.split()) < min_words or s.endswith(":"):
                continue
            sents.append(s)
            if len(sents) >= max_sentences:
                return page.title, sents
    return page.title, sents


def random_titles(n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "random", "rnnamespace": 0,
                    "rnlimit": min(50, n - len(out)), "format": "json"},
            headers={"User-Agent": UA}, timeout=20,
        )
        out += [x["title"] for x in r.json()["query"]["random"]]
    return out[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--titles", help="comma-separated explicit article titles")
    g.add_argument("--random", type=int, help="fetch N random articles instead")
    ap.add_argument("--topics", type=int, default=6,
                    help="how many curated topics to use (if no --titles/--random)")
    ap.add_argument("--sentences-per-topic", type=int, default=40)
    ap.add_argument("--min-words", type=int, default=6,
                    help="drop sentences shorter than this many words")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="politeness delay between article fetches")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.titles:
        titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    elif args.random:
        titles = random_titles(args.random)
    else:
        titles = DEFAULT_TOPICS[: args.topics]

    rows: list[dict] = []
    ok = 0
    for title in titles:
        used, sents = article_sentences(title, args.sentences_per_topic, args.min_words)
        if not sents:
            continue
        ok += 1
        rows.extend({"text": s, "source": used} for s in sents)
        print(f"  {used}: {len(sents)} sentences")
        time.sleep(args.sleep)

    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}: {len(rows)} sentences from {ok}/{len(titles)} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
