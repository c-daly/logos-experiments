#!/usr/bin/env python3
"""Interleaved paragraph-block corpus for the 24-topic huge run.

6 domains x 4 related articles = 24 topics. ~N ~target-word blocks per topic
(coherent prose passages where entities co-occur WITH relationships, unlike the
isolated-sentence corpora). Round-robin interleaved across topics so every domain
accumulates in the entity drawer together. Reuses wikipedia-api.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import requests, wikipediaapi

UA = "logos-experiment/0.1 (https://github.com/c-daly/logos; research)"
WIKI = wikipediaapi.Wikipedia(user_agent=UA, language="en")

DOMAINS = {
    "animals":       ["Mammal", "Fish", "Bird", "Insect"],
    "astronomy":     ["Star", "Planet", "Galaxy", "Comet"],
    "chemistry":     ["Chemical element", "Acid", "Polymer", "Catalysis"],
    "earth_science": ["Volcano", "Earthquake", "Mineral", "Glacier"],
    "technology":    ["Computer", "Internet", "Robot", "Algorithm"],
    "medicine":      ["Heart", "Brain", "Immune system", "Virus"],
}
SKIP = {"references","see also","external links","further reading","notes",
        "bibliography","citations","sources","explanatory notes",
        "general and cited references"}

def paragraphs(page):
    parts = [page.summary]
    def rec(sections):
        for s in sections:
            if s.title.strip().lower() in SKIP: continue
            if s.text: parts.append(s.text)
            rec(s.sections)
    rec(page.sections)
    out = []
    for chunk in parts:
        for p in chunk.split("\n"):
            p = p.strip()
            if p and len(p.split()) >= 8 and not p.endswith(":"):
                out.append(p)
    return out

def blocks_for(title, n, target, max_words):
    page = WIKI.page(title)
    if not page.exists():
        print(f"  [skip] {title}: not found"); return title, []
    blocks, buf = [], ""
    for p in paragraphs(page):
        buf = (buf + " " + p).strip() if buf else p
        if len(buf.split()) >= target:
            block = buf
            if len(block.split()) > max_words:
                block = " ".join(block.split()[:max_words])
            blocks.append(block); buf = ""
            if len(blocks) >= n: break
    return page.title, blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-topic", type=int, default=10)
    ap.add_argument("--target-words", type=int, default=120)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--titles", help="comma-separated Wikipedia titles; each becomes its own "
                    "topic (domain = slugified title). Overrides the built-in 24-topic set.")
    ap.add_argument("--domains-json", help='JSON {"domain": ["Title A", "Title B"]} for grouped '
                    "topics (richer domain metadata). Overrides --titles and the built-in set.")
    a = ap.parse_args()

    if a.domains_json:
        domain_groups = json.loads(a.domains_json)
    elif a.titles:
        domain_groups = {t.strip().lower().replace(" ", "_"): [t.strip()]
                         for t in a.titles.split(",") if t.strip()}
    else:
        domain_groups = DOMAINS

    per_topic = []
    for domain, titles in domain_groups.items():
        for t in titles:
            rt, bl = blocks_for(t, a.blocks_per_topic, a.target_words, a.max_words)
            print(f"  {domain}/{rt}: {len(bl)} blocks")
            per_topic.append((domain, rt, bl)); time.sleep(a.sleep)
    rows, maxb = [], max((len(b) for _,_,b in per_topic), default=0)
    for bi in range(maxb):
        for domain, title, bl in per_topic:
            if bi < len(bl):
                rows.append({"domain":domain,"topic":title,"block":bi,"text":bl[bi]})
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wc = sorted(len(r["text"].split()) for r in rows)
    print(f"wrote {len(rows)} blocks -> {a.out}; words/block min/med/max = "
          f"{wc[0]}/{wc[len(wc)//2]}/{wc[-1]}")

if __name__ == "__main__":
    main()
