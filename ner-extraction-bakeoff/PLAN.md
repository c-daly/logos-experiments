# NER/RE Extraction Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline harness that runs 4 extraction arms (baseline OpenAI, spaCy-local, closed-vocab OpenAI, cheap-model OpenAI) over a 40-sentence hand-labeled gold set and scores each on entity F1, relation-link F1, relation-label F1, and relation-vocabulary compactness — to find the source-side lever that cuts relation over-generation without losing recall.

**Architecture:** Pure scoring functions (`metrics.py`, TDD-unit-tested) + a thin arms layer that normalizes each extractor's output to a common shape (`arms.py`) + an orchestrator that runs arms over the gold set and emits a results table (`run_bakeoff.py`). Runs in the **hermes venv** so it can import the Hermes extractors.

**Tech Stack:** Python (hermes venv), pytest; `hermes.combined_extractor`, `hermes.ner_provider`, `hermes.relation_extractor`, `hermes.llm.generate_completion`, `hermes.canonical`.

**Spec:** `logos-experiments/ner-extraction-bakeoff/DESIGN.md`

**Run everything from the hermes venv:**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run pip install --quiet pytest   # if missing
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests -q
```
Paths below are relative to `logos-experiments/ner-extraction-bakeoff/` unless absolute.

---

### Task 1: Gold set (already authored)

**Files:**
- Verify: `logos-experiments/ner-extraction-bakeoff/gold.jsonl` (40 records, committed during planning)

The gold set is already written — 40 curated sentences (the 16 `corpus.jsonl` originals + 24 authored in the same clean style), hand-labeled with entities `{name,type}` and relations `{source,relation,target}` using a compact 20-relation canonical vocabulary.

- [ ] **Step 1: Verify it parses, is the right size, and has no dangling relations** — Run:
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff
python3 -c "
import json
rows=[json.loads(l) for l in open('gold.jsonl')]
assert len(rows)==40, len(rows)
assert all('entities' in r and 'relations' in r for r in rows)
for r in rows:
    names={e['name'] for e in r['entities']}
    for x in r['relations']:
        assert x['source'] in names and x['target'] in names, x
print('ok:', len(rows), 'records,',
      sum(len(r['relations']) for r in rows), 'relations')
"
```
Expected: `ok: 40 records, 64 relations`.

- [ ] **Step 2: Commit** (if not already committed during planning)
```bash
git add gold.jsonl
git commit -m "feat(lx38): hand-labeled 40-sentence gold set (entities + relations)" || true
```

---

### Task 2: Metrics — generic PRF + entity scoring

**Files:**
- Create: `logos-experiments/ner-extraction-bakeoff/metrics.py`
- Test: `logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_metrics.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import prf, score_entities


def test_prf_basic():
    p, r, f = prf(pred={"a", "b", "c"}, gold={"b", "c", "d"})
    assert p == 2 / 3  # 2 of 3 predicted are correct
    assert r == 2 / 3  # 2 of 3 gold found
    assert round(f, 3) == 0.667


def test_prf_empty_pred_is_zero():
    assert prf(pred=set(), gold={"a"}) == (0.0, 0.0, 0.0)


def test_prf_empty_gold_pred_empty_is_perfect():
    assert prf(pred=set(), gold=set()) == (1.0, 1.0, 1.0)


def test_score_entities_matches_on_canonical_name():
    pred = [{"name": "Cheetahs", "type": "animal"}]  # plural/case differ
    gold = [{"name": "cheetah", "type": "animal"}]
    out = score_entities(pred, gold)
    assert out["recall"] == 1.0  # canonicalize folds Cheetahs -> cheetah
    assert out["type_accuracy"] == 1.0


def test_score_entities_type_accuracy_independent_of_name_f1():
    pred = [{"name": "cheetah", "type": "vehicle"}]  # right name, wrong type
    gold = [{"name": "cheetah", "type": "animal"}]
    out = score_entities(pred, gold)
    assert out["f1"] == 1.0  # name matched
    assert out["type_accuracy"] == 0.0  # but type wrong
```

- [ ] **Step 2: Run to verify it fails**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 3: Write `metrics.py`**:

```python
"""Pure scoring for the NER/RE bake-off (logos-experiments#38).

Entity and relation matching is on CANONICAL forms (hermes.canonical) so
surface variation (case/plural/morphology) is not penalized. No network.
"""

from __future__ import annotations

from hermes.canonical import canonicalize, canonicalize_predicate


def prf(*, pred: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1 over two sets. Empty/empty == perfect."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def _cname(name: str) -> str:
    return canonicalize(name or "")


def score_entities(pred: list[dict], gold: list[dict]) -> dict:
    """Entity precision/recall/F1 over canonical names + type-accuracy on
    the names that matched."""
    pred_names = {_cname(e["name"]) for e in pred if e.get("name")}
    gold_names = {_cname(e["name"]) for e in gold if e.get("name")}
    p, r, f = prf(pred=pred_names, gold=gold_names)

    gold_type = {_cname(e["name"]): e.get("type", "") for e in gold if e.get("name")}
    pred_type = {_cname(e["name"]): e.get("type", "") for e in pred if e.get("name")}
    matched = pred_names & gold_names
    type_ok = sum(1 for n in matched if pred_type.get(n) == gold_type.get(n))
    type_accuracy = type_ok / len(matched) if matched else (1.0 if not gold else 0.0)

    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "type_accuracy": type_accuracy,
        "pred_count": len(pred_names),
        "gold_count": len(gold_names),
    }
```

- [ ] **Step 4: Run to verify it passes**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/metrics.py ner-extraction-bakeoff/tests/test_metrics.py
git commit -m "feat(lx38): metrics — prf + entity scoring (canonical-name + type-accuracy)"
```

---

### Task 3: Metrics — relation link & label scoring

**Files:**
- Modify: `logos-experiments/ner-extraction-bakeoff/metrics.py`
- Modify: `logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py`

- [ ] **Step 1: Add failing tests** — append to `tests/test_metrics.py`:

```python
from metrics import score_relation_labels, score_relation_links


def test_relation_links_ignore_label():
    # same (src->tgt) link, different relation label -> link still matches
    pred = [{"source": "tusk", "relation": "GROWS_FROM", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_links(pred, gold)
    assert out["recall"] == 1.0  # the link was found
    assert out["f1"] == 1.0


def test_relation_links_directional():
    pred = [{"source": "narwhal", "relation": "PART_OF", "target": "tusk"}]  # reversed
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_links(pred, gold)
    assert out["recall"] == 0.0  # direction matters


def test_relation_labels_require_canonical_relation_match():
    pred = [{"source": "tusk", "relation": "GROWS_FROM", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_labels(pred, gold)
    assert out["f1"] == 0.0  # link right, label wrong


def test_relation_labels_match_on_canonical_predicate():
    # PART_OF vs "part of" / "parts of" canonicalize to the same key
    pred = [{"source": "tusk", "relation": "parts of", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_labels(pred, gold)
    assert out["recall"] == 1.0
```

- [ ] **Step 2: Run to verify they fail**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: FAIL — `ImportError: cannot import name 'score_relation_links'`.

- [ ] **Step 3: Add the functions to `metrics.py`**:

```python
def _link(rel: dict) -> tuple[str, str]:
    return (_cname(rel.get("source", "")), _cname(rel.get("target", "")))


def _triple(rel: dict) -> tuple[str, str, str]:
    return (
        _cname(rel.get("source", "")),
        canonicalize_predicate(rel.get("relation", "")),
        _cname(rel.get("target", "")),
    )


def score_relation_links(pred: list[dict], gold: list[dict]) -> dict:
    """P/R/F1 over directional (source, target) pairs, ignoring the label."""
    p, r, f = prf(pred={_link(x) for x in pred}, gold={_link(x) for x in gold})
    return {"precision": p, "recall": r, "f1": f}


def score_relation_labels(pred: list[dict], gold: list[dict]) -> dict:
    """P/R/F1 over (source, canonical-relation, target) triples."""
    p, r, f = prf(pred={_triple(x) for x in pred}, gold={_triple(x) for x in gold})
    return {"precision": p, "recall": r, "f1": f}
```

- [ ] **Step 4: Run to verify they pass**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/metrics.py ner-extraction-bakeoff/tests/test_metrics.py
git commit -m "feat(lx38): metrics — relation link-F1 (label-blind) + label-F1 (triple)"
```

---

### Task 4: Metrics — relation-vocabulary compactness

**Files:**
- Modify: `logos-experiments/ner-extraction-bakeoff/metrics.py`
- Modify: `logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py`

- [ ] **Step 1: Add failing test** — append to `tests/test_metrics.py`:

```python
from metrics import compactness


def test_compactness_counts_distinct_canonical_predicates():
    # CARRIES / carries / carried -> one canonical key; PART_OF -> another
    rels = [
        {"source": "a", "relation": "CARRIES", "target": "b"},
        {"source": "c", "relation": "carried", "target": "d"},
        {"source": "e", "relation": "PART_OF", "target": "f"},
    ]
    out = compactness(rels)
    assert out["distinct_predicates"] == 2  # CARR* folds together
    assert out["total_relations"] == 3
    # df=1: PART_OF appears once (1 of 2 distinct) -> 0.5
    assert out["df1_fraction"] == 0.5


def test_compactness_empty():
    out = compactness([])
    assert out["distinct_predicates"] == 0 and out["df1_fraction"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: FAIL — `ImportError: cannot import name 'compactness'`.

- [ ] **Step 3: Add to `metrics.py`** (add `from collections import Counter` at the top):

```python
def compactness(relations: list[dict]) -> dict:
    """Relation-vocabulary compactness over an arm's output."""
    keys = [canonicalize_predicate(r.get("relation", "")) for r in relations]
    keys = [k for k in keys if k]
    counts = Counter(keys)
    distinct = len(counts)
    df1 = sum(1 for c in counts.values() if c == 1)
    return {
        "distinct_predicates": distinct,
        "total_relations": len(keys),
        "predicates_per_relation": (distinct / len(keys)) if keys else 0.0,
        "df1_fraction": (df1 / distinct) if distinct else 0.0,
    }
```

- [ ] **Step 4: Run to verify it passes**
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -m pytest /home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff/tests/test_metrics.py -q
```
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/metrics.py ner-extraction-bakeoff/tests/test_metrics.py
git commit -m "feat(lx38): metrics — relation-vocabulary compactness (distinct preds, df=1)"
```

---

### Task 5: Arms

**Files:**
- Create: `logos-experiments/ner-extraction-bakeoff/arms.py`

Each arm is an async function `run(text: str) -> tuple[list[dict], list[dict]]` returning `(entities, relations)` normalized to: entity `{"name","type"}`, relation `{"source","relation","target"}`. `arms.py` also exposes `ARMS: dict[str, callable]` and a `relation_vocabulary()` helper for the closed-vocab arm.

- [ ] **Step 1: Write `arms.py`**:

```python
"""The 4 extraction arms for the bake-off (logos-experiments#38).

Each arm: async run(text) -> (entities, relations), normalized to
entity {name,type} and relation {source,relation,target}. Runs in the
hermes venv. The OpenAI arms call the real extractors / generate_completion;
the spaCy arm uses the local providers. closed_vocab is built here (a
vocab-injected prompt) so production code is untouched.
"""

from __future__ import annotations

import json
import os

# --- normalization -------------------------------------------------------

def _norm_entities(raw: list[dict]) -> list[dict]:
    out = []
    for e in raw or []:
        name = e.get("name") or e.get("text")
        if name:
            out.append({"name": name, "type": e.get("type", "entity")})
    return out


def _norm_relations(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw or []:
        src = r.get("source") or r.get("source_name")
        tgt = r.get("target") or r.get("target_name")
        rel = r.get("relation")
        if src and tgt and rel:
            out.append({"source": src, "relation": rel, "target": tgt})
    return out


# --- OpenAI combined arms -----------------------------------------------

async def _combined(text: str, model: str | None) -> tuple[list[dict], list[dict]]:
    from hermes.combined_extractor import OpenAICombinedExtractor

    ex = OpenAICombinedExtractor()
    if model:
        # OpenAICombinedExtractor reads the configured model; override via env
        os.environ["HERMES_LLM_MODEL"] = model
    entities, relations = await ex.extract_entities_and_relations(text)
    return _norm_entities(entities), _norm_relations(relations)


async def baseline(text: str):
    return await _combined(text, model=None)


async def cheap_model(text: str):
    # set via env so a swap is one place; default to a mini tier
    model = os.environ.get("BAKEOFF_CHEAP_MODEL", "gpt-4o-mini")
    return await _combined(text, model=model)


# --- spaCy local arm -----------------------------------------------------

async def spacy(text: str):
    from hermes.ner_provider import SpacyNERProvider
    from hermes.relation_extractor import SpacyRelationExtractor

    ner = SpacyNERProvider()
    re = SpacyRelationExtractor()
    entities = await ner.extract_entities(text)
    relations = await re.extract(text, entities)
    return _norm_entities(entities), _norm_relations(relations)


# --- closed-vocab OpenAI arm --------------------------------------------

_CLOSED_SYSTEM = (
    "You extract entities and relations from text for a knowledge graph.\n"
    "Return ONLY JSON: {\"entities\":[{\"name\":..,\"type\":..}],"
    "\"relations\":[{\"source\":..,\"relation\":..,\"target\":..}]}.\n"
    "For each relation, REUSE a relation from this known vocabulary when one "
    "fits; only coin a NEW relation label if none of these fit:\n{vocab}\n"
    "source and target must be names from your entities list."
)


def relation_vocabulary(limit: int = 120) -> list[str]:
    """Known descriptive relations to inject. Prefer the Redis snapshot
    (logos:ontology:relations); fall back to a small seed."""
    try:
        import redis

        raw = redis.Redis(decode_responses=True).get("logos:ontology:relations")
        if raw:
            vocab = sorted(json.loads(raw).keys())
            return vocab[:limit]
    except Exception:
        pass
    return [
        "IS_A", "PART_OF", "LOCATED_IN", "PRODUCES", "USED_FOR", "EATS",
        "CATCHES", "AFFECTS", "MEMBER_OF", "PLAYS", "TOWS", "FASTER_THAN",
    ]


async def closed_vocab(text: str):
    from hermes.llm import generate_completion

    vocab = ", ".join(relation_vocabulary())
    result = await generate_completion(
        messages=[
            {"role": "system", "content": _CLOSED_SYSTEM.format(vocab=vocab)},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=1024,
        metadata={"scenario": "bakeoff_closed_vocab"},
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re as _re

        m = _re.search(r"```(?:json)?\s*(.*?)```", content, _re.DOTALL)
        data = json.loads(m.group(1)) if m else {"entities": [], "relations": []}
    return _norm_entities(data.get("entities")), _norm_relations(data.get("relations"))


ARMS = {
    "baseline": baseline,
    "spacy": spacy,
    "closed_vocab": closed_vocab,
    "cheap_model": cheap_model,
}
```

- [ ] **Step 2: Smoke-test imports + normalization** (no network) — Run:
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -c "
import sys; sys.path.insert(0,'/home/fearsidhe/projects/logos-workspace/logos-experiments/ner-extraction-bakeoff')
import arms
assert set(arms.ARMS)=={'baseline','spacy','closed_vocab','cheap_model'}
assert arms._norm_relations([{'source_name':'a','target_name':'b','relation':'R'}])==[{'source':'a','relation':'R','target':'b'}]
assert arms._norm_entities([{'name':'x','type':'animal'}])==[{'name':'x','type':'animal'}]
print('arms ok; vocab sample:', arms.relation_vocabulary()[:5])
"
```
Expected: `arms ok; vocab sample: [...]`.

- [ ] **Step 3: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/arms.py
git commit -m "feat(lx38): the 4 extraction arms (baseline/spacy/closed_vocab/cheap_model) + normalization"
```

---

### Task 6: Orchestrator

**Files:**
- Create: `logos-experiments/ner-extraction-bakeoff/run_bakeoff.py`

- [ ] **Step 1: Write `run_bakeoff.py`**:

```python
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
        vals = [scorer(p, g)[key] for p, g in zip(preds, golds)]
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
        **{k: round(v, 3) if isinstance(v, float) else v
           for k, v in compactness(all_pred_rel).items()},
    }


async def main() -> None:
    gold = load_gold()
    names = sys.argv[1:] or list(arms_mod.ARMS)
    results = []
    for name in names:
        print(f"running arm: {name} ...", file=sys.stderr)
        results.append(await run_arm(name, arms_mod.ARMS[name], gold))

    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    cols = ["arm", "entity_f1", "type_accuracy", "link_recall", "link_f1",
            "label_f1", "distinct_predicates", "df1_fraction", "failures",
            "elapsed_s"]
    print("\n" + " | ".join(f"{c:>16}" for c in cols))
    for r in results:
        print(" | ".join(f"{str(r.get(c, '')):>16}" for c in cols))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Dry-run on the spaCy arm only** (free, no API) — first ensure spaCy is available:
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python -c "import spacy" 2>/dev/null || poetry run pip install --quiet spacy
poetry run python -m spacy download en_core_web_sm 2>&1 | tail -1
poetry run python ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py spacy
```
Expected: a one-row table for `spacy` (likely low entity recall — that is the finding, not a bug). If spaCy can't load, the arm reports `failures: 16` rather than crashing.

- [ ] **Step 3: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/run_bakeoff.py
git commit -m "feat(lx38): bake-off orchestrator + results table"
```

---

### Task 7: Run the full bake-off + write up the verdict

**Files:**
- Create: `logos-experiments/ner-extraction-bakeoff/README.md`
- Create: `logos-experiments/ner-extraction-bakeoff/results.json` (generated)

- [ ] **Step 1: Run all 4 arms** (needs the OpenAI key configured, as the live Hermes already has):
```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run python ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py
```
Expected: a 4-row table + `results.json`. Re-run once to note any nondeterminism in `distinct_predicates`.

- [ ] **Step 2: Write `README.md`** — capture: how to run, the gold-set provenance (40 curated sentences — 16 `corpus.jsonl` + 24 authored — hand-labeled, compact 20-relation canonical vocabulary), the results table (paste from the run), and the **verdict** per the DESIGN decision criterion: does any arm cut `distinct_predicates` vs baseline while keeping `link_recall` within a small margin and `entity_f1` not materially worse? Name the recommended arm (or "none — baseline still best") and what it implies for the source-side fix.

- [ ] **Step 3: Commit**
```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments
git add ner-extraction-bakeoff/README.md ner-extraction-bakeoff/results.json
git commit -m "feat(lx38): run the bake-off + verdict (README + results)"
```

---

## Notes for the implementer

- **Run from the hermes venv** — the arms import `hermes.*`. The metrics tests also import `hermes.canonical` (needs `inflect`, no network).
- **Determinism:** LLM arms use temperature 0; still re-run to note compactness nondeterminism.
- **The spaCy arm is expected to score low on entity recall** (its model knows PERSON/ORG/GPE, not "narwhal"/"tusk"). That is the free option's ceiling — a finding, report it, don't "fix" it.
- **`HERMES_LLM_MODEL` / `BAKEOFF_CHEAP_MODEL`:** the cheap arm overrides the model via env; confirm the configured provider honors `HERMES_LLM_MODEL` (grep `hermes/llm.py` — if the provider takes `model` differently, pass it through `generate_completion(model=...)` instead, and route `_combined` through `generate_completion` rather than the extractor for the cheap arm).
- **n = 40:** enough to rank moderate effects; read a near-tie between two arms as a near-tie (extend `gold.jsonl`), not a verdict.
