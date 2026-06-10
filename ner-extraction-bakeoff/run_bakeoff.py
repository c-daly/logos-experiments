"""Run the NER/RE bake-off over the gold set and emit a results table.

Usage (from the hermes venv):
  cd hermes && poetry run python \
    ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py [arm ...]
Default: all arms. Writes results.json next to this file.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms as arms_mod  # noqa: E402
from metrics import (  # noqa: E402
    compactness,
    score_entities,
    score_relation_labels,
    score_relation_links,
)


def load_gold() -> list[dict]:
    return [json.loads(line) for line in (HERE / "gold.jsonl").open()]


async def run_arm(name: str, fn, gold: list[dict]) -> dict:
    ent_p, ent_g, rel_p, rel_g = [], [], [], []
    all_pred_rel: list[dict] = []
    failures = 0
    t0 = time.time()
    for row in gold:
        try:
            entities, relations = await fn(row["text"])
        except Exception as exc:  # an arm that can't run is reported, not hidden
            failures += 1
            entities, relations = [], []
            print(f"  [{name}] failed on a sentence: {str(exc)[:80]}", file=sys.stderr)
        ent_p.append(entities)
        ent_g.append(row["entities"])
        rel_p.append(relations)
        rel_g.append(row["relations"])
        all_pred_rel.extend(relations)
    elapsed = time.time() - t0

    def agg(scorer, preds, golds, key):
        # Skip sentences with nothing predicted AND nothing in gold: prf()
        # scores empty-vs-empty as a perfect 1.0, which would inflate the
        # relation averages for silent arms on any relation-free sentence.
        vals = [scorer(p, g)[key] for p, g in zip(preds, golds) if p or g]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "arm": name,
        "failures": failures,
        "elapsed_s": round(elapsed, 1),
        "entity_f1": round(agg(score_entities, ent_p, ent_g, "f1"), 3),
        "entity_recall": round(agg(score_entities, ent_p, ent_g, "recall"), 3),
        "type_accuracy": round(agg(score_entities, ent_p, ent_g, "type_accuracy"), 3),
        "link_f1": round(agg(score_relation_links, rel_p, rel_g, "f1"), 3),
        "link_recall": round(agg(score_relation_links, rel_p, rel_g, "recall"), 3),
        "label_f1": round(agg(score_relation_labels, rel_p, rel_g, "f1"), 3),
        **{
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in compactness(all_pred_rel).items()
        },
    }


async def main() -> None:
    gold = load_gold()
    names = sys.argv[1:] or list(arms_mod.ARMS)
    results = []
    for name in names:
        print(f"running arm: {name} ...", file=sys.stderr)
        results.append(await run_arm(name, arms_mod.ARMS[name], gold))

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    cols = [
        "arm", "entity_f1", "type_accuracy", "link_recall", "link_f1",
        "label_f1", "distinct_predicates", "df1_fraction", "failures",
        "elapsed_s",
    ]
    print("\n" + " | ".join(f"{c:>16}" for c in cols))
    for r in results:
        print(" | ".join(f"{str(r.get(c, '')):>16}" for c in cols))


if __name__ == "__main__":
    asyncio.run(main())
