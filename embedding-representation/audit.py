"""Step 0 -- extraction audit (genuine errors) + the generic-concept question.

Genericity is NOT noise: "gene", "energy", "cell", "protein" are legitimate
concepts -- the backbone of a concept graph, not extraction failures. Only
measurements and fragments ("0.1 to 5.0 μm", "1 cm") are genuine errors. So this
splits two distinct questions:

1. GENUINE ERROR RATE -- what fraction of `entity` nodes are non-entities
   (measurements, bare numbers, too-short/long fragments). These are removable.

2. THE GENERIC-CONCEPT HYPOTHESIS -- generic common-noun concepts are exactly
   the entities whose BARE NAME is most ambiguous (a lone "energy"/"cell" is an
   average over senses), so their name-vector geometry should be WEAKER than
   proper-named entities'. If so, genericity isn't a corpus defect -- it is *why
   the bare-name representation hurts*, and the lever is context, not selection.

Detectors (label-free):
  measurement/year_number/has_digit/too_short/too_long -- genuine errors.
  proper -- DESCRIPTIVE (not a quality filter): the mention is capitalised
            mid-sentence in raw_text (a named entity) vs a common-noun concept.

Battery on the live production NAME vectors, by group (all legitimate except the
errors):
  raw          -- all entities
  minus_errors -- drop only genuine errors
  common_noun  -- legitimate common-noun concepts (generic), errors removed
  proper       -- named entities, errors removed
Compare common_noun vs proper: weaker margin / higher anisotropy for common_noun
is evidence that general concepts need context most.

    python audit.py    # sophia venv (neo4j + pymilvus + numpy)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

import battery as B

HERE = Path(__file__).resolve().parent

UNIT = r"(?:nm|µm|μm|um|mm|cm|dm|m|km|kg|mg|µg|g|ml|l|s|ms|min|h|hz|khz|mhz|ghz|°c|°f|k|%|bp|kb|mb|da|kda|mol|ev|kev|mev|v|w|j)"
MEASURE = re.compile(r"^\s*[\d.,]+\s*(?:(?:to|[-–±x×])\s*[\d.,]+\s*)?" + UNIT + r"?\s*$", re.I)
YEAR = re.compile(r"^\s*\d{1,4}\s*(?:bc|ad|bce|ce)?\s*$", re.I)
DIGIT = re.compile(r"\d")


def classify(name: str) -> list[str]:
    n = name.strip()
    flags = []
    if len(n) <= 2:
        flags.append("too_short")
    if len(n) > 60:
        flags.append("too_long")
    if YEAR.match(n):
        flags.append("year_number")
    if MEASURE.match(n):
        flags.append("measurement")
    elif DIGIT.search(n):
        flags.append("has_digit")
    return flags


def proper_in_context(name: str, raw_text: str, start: int) -> bool:
    """The mention is capitalised mid-sentence (a named entity), not generic."""
    if not raw_text:
        return False
    best, bestd = None, 1 << 30
    for m in re.finditer(re.escape(name), raw_text, re.I):
        d = abs(m.start() - start)
        if d < bestd:
            bestd, best = d, m
    if best is None:
        return False
    j = best.start() - 1
    while j >= 0 and raw_text[j].isspace():
        j -= 1
    sentence_initial = j < 0 or raw_text[j] in ".!?:;"
    first_alpha = next((c for c in best.group() if c.isalpha()), "")
    return first_alpha.isupper() and not sentence_initial


def fetch() -> list[dict]:
    from neo4j import GraphDatabase

    pw = os.environ.get("NEO4J_PASSWORD", "logosdev")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pw))
    q = (
        "MATCH (n:Node {type:'entity'}) "
        "RETURN n.uuid AS uuid, n.name AS name, n.raw_text AS raw_text, "
        "n.start AS start, n.end AS end, n.confidence AS confidence, "
        "n.hermes_type_hint AS hint"
    )
    try:
        with driver.session() as s:
            return s.execute_read(lambda tx: tx.run(q).data())
    finally:
        driver.close()


def name_vectors(uuids: list[str]) -> dict:
    from pymilvus import Collection, connections

    connections.connect("default", host="localhost", port="19530")
    col = Collection("hcg_entity_embeddings")
    col.load()
    out = {}
    for i in range(0, len(uuids), 500):
        expr = "uuid in [%s]" % ",".join(f'"{u}"' for u in uuids[i : i + 500])
        for r in col.query(expr=expr, output_fields=["uuid", "embedding"]):
            out[r["uuid"]] = r["embedding"]
    return out


def main():
    ents = fetch()
    print(f"live entities: {len(ents)}")
    flagged = Counter()
    proper = 0
    examples: dict[str, list] = {}
    for e in ents:
        fl = classify(e["name"] or "")
        for f in fl:
            flagged[f] += 1
            examples.setdefault(f, [])
            if len(examples[f]) < 6:
                examples[f].append(e["name"])
        e["_flags"] = fl
        e["_proper"] = proper_in_context(e["name"] or "", e.get("raw_text") or "", e.get("start") or 0)
        proper += e["_proper"]

    print("\n=== genuine extraction errors (fraction of live entities) ===")
    for f, c in flagged.most_common():
        print(f"  {f:14} {c:5} ({c/len(ents)*100:4.1f}%)   e.g. {examples[f][:5]}")
    any_noise = sum(1 for e in ents if e["_flags"])
    print(f"  {'ANY error':14} {any_noise:5} ({any_noise/len(ents)*100:4.1f}%)   <- removable")
    print(f"  {'proper (named)':14} {proper:5} ({proper/len(ents)*100:4.1f}%)   "
          f"vs {len(ents)-proper} common-noun concepts (both legitimate)")

    # confidence by quality
    def mean_conf(rows):
        cs = [r["confidence"] for r in rows if r.get("confidence") is not None]
        return round(sum(cs) / len(cs), 3) if cs else None

    print(f"\nmean confidence: all={mean_conf(ents)}  "
          f"noise={mean_conf([e for e in ents if e['_flags']])}  "
          f"proper={mean_conf([e for e in ents if e['_proper']])}")

    # battery on selection subsets (production name vectors)
    vecs = name_vectors([e["uuid"] for e in ents])
    def mat(rows):
        v = [vecs[r["uuid"]] for r in rows if r["uuid"] in vecs]
        return np.asarray(v, dtype="float32")

    subsets = {
        "raw": ents,
        "minus_errors": [e for e in ents if not e["_flags"]],
        "common_noun": [e for e in ents if not e["_flags"] and not e["_proper"]],
        "proper": [e for e in ents if not e["_flags"] and e["_proper"]],
    }
    print("\n=== battery on NAME vectors by group (common-noun vs proper: who needs context?) ===")
    cols = ["n", "anisotropy_centroid", "nn_margin", "pairwise_cos_spread", "effective_rank", "intrinsic_dim_twonn"]
    print(f"{'subset':<13}" + "".join(f"{c[:11]:>13}" for c in cols))
    out = {"n_entities": len(ents), "detectors": dict(flagged), "proper": proper, "subsets": {}}
    for name, rows in subsets.items():
        X = mat(rows)
        r = B.score(X)["raw"]
        out["subsets"][name] = {"n": int(len(X)), **{k: r[k] for k in cols[1:]}}
        print(f"{name:<13}" + "".join(f"{r[c] if c in r else len(X):>13}" for c in cols))

    (HERE / "audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote audit.json")


if __name__ == "__main__":
    main()
