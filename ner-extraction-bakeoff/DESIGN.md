# NER/RE extraction bake-off — design

**Ticket:** c-daly/logos-experiments#38 · **Program:** c-daly/logos#557 (W0.3 source-quality lever) · **Date:** 2026-06-10

## Question

Can we tone down or replace the OpenAI extractor — with spaCy (free/local), a cheaper model, or closed-vocabulary prompt pressure — **without losing entity + relation extraction quality**, and how much does each approach cut relation-vocabulary over-generation (the df=1 source problem)?

Established empirically (2026-06-10): vocabulary *cleanup* (canonicalize + match-before-mint + the full synonym rollup) folds ~13% but moves df=1 only 0.611 → 0.586 — the long tail is genuinely-distinct one-off predicates the LLM RE over-generates. So the binding constraint is the **source**: generate fewer distinct predicates at extraction. This experiment evaluates the source-side levers.

## Method: offline 4-arm bake-off on a hand-labeled gold set

Each arm runs the same extraction on the same 16 gold sentences; scored against hand labels for entities and relations, plus relation-vocabulary compactness. **Offline** — the harness calls the Hermes extractors directly on the text, no graph ingest, no mutation. Deterministic per arm (temperature 0 for the LLM arms).

### Arms

| arm | NER | RE | rationale |
|-----|-----|-----|-----------|
| `baseline` | OpenAI combined (NER+RE, one call), current model | — | the production default; the thing to beat |
| `spacy` | `SpacyNERProvider` | `SpacyRelationExtractor` | free/local; RE maps verbs→~12 fixed relations (compact by construction) |
| `closed_vocab` | OpenAI combined | + known relation vocabulary injected into the prompt with reuse-don't-mint pressure | keeps OpenAI recall, constrains relation sprawl (H2 at the prompt) |
| `cheap_model` | OpenAI combined on a cheaper model tier | — | tests the "tone down the model" cost lever |

The OpenAI arms reuse `hermes.combined_extractor.OpenAICombinedExtractor`; `closed_vocab` is built **in the harness** (a vocab-injected system prompt via `hermes.llm.generate_completion`) so production code is untouched. `spacy` uses `hermes.ner_provider.SpacyNERProvider` + `hermes.relation_extractor.SpacyRelationExtractor`.

## Gold set

Source: `logos-experiments/naming-driven-typing/corpus/corpus.jsonl` (16 curated sentences — animals/vehicles/instruments/plants/weather, simple and unambiguous, e.g. `"A narwhal is a marine mammal whose tusk spirals out from its jaw."`).

Hand-labeled into `gold.jsonl`, one record per source sentence:
```json
{
  "text": "A narwhal is a marine mammal whose tusk spirals out from its jaw.",
  "entities": [
    {"name": "narwhal", "type": "animal"},
    {"name": "marine mammal", "type": "animal"},
    {"name": "tusk", "type": "body_part"},
    {"name": "jaw", "type": "body_part"}
  ],
  "relations": [
    {"source": "narwhal", "relation": "IS_A", "target": "marine mammal"},
    {"source": "tusk", "relation": "PART_OF", "target": "narwhal"}
  ]
}
```
Labels are written by the experimenter (this session), reviewed before scoring. The relation labels are the *canonical, minimal* set a good extractor should find — the recall target that an over-generating arm must still hit.

## Metrics (per arm)

All matching is on **canonicalized** names/relations (`hermes.canonical.canonicalize` for entity names, `canonicalize_predicate` for relations) so surface variation isn't penalized.

1. **Entity F1** — precision/recall/F1 over entity *names* (canonical-name match). Secondary: **type-accuracy** on matched entities.
2. **Relation-link F1** — over **directional** (source → target) pairs, **ignoring the relation label**. "Did the arm find the real relationship?" This is the recall we must not lose. (Symmetric gold relations, if any, are matched in either direction.)
3. **Relation-label F1** — over (source, relation-canonical, target) triples. "Did it label the relation correctly, without over-inventing?"
4. **Compactness** — distinct relation predicates emitted across the 16 sentences; distinct-predicates-per-relation ratio; df=1 fraction over the arm's output.
5. **Cost** — LLM calls + total tokens per arm (spaCy = 0); wall-clock.

**Why link-F1 and label-F1 are separate (the crux):** an arm could "win" compactness by emitting fewer relations — but if it also misses real links, link-recall drops and exposes it. The decision looks for the arm that keeps **link-recall high** while cutting **distinct-predicate count** and keeping **label-F1** reasonable.

## Output

`results.json` + a printed table: one row per arm with entity-F1, type-acc, link-F1, label-F1, distinct-predicates, df=1, calls, tokens. Plus a short verdict: which arm best trades compactness for quality, and whether any free/cheap arm is viable.

## Harness structure

`ner-extraction-bakeoff/`
- `gold.jsonl` — the hand-labeled gold set (16 records).
- `arms.py` — one async `run(text) -> (entities, relations)` per arm; `baseline`/`cheap_model`/`closed_vocab` via the combined extractor + `generate_completion`, `spacy` via the spaCy providers.
- `metrics.py` — pure scoring functions (entity/link/label P-R-F1, compactness) over (predicted, gold); unit-tested on synthetic pairs, no network.
- `run_bakeoff.py` — loads gold, runs each arm over the 16 texts, scores, writes `results.json` + table.
- `README.md` — how to run, the gold-set provenance, the verdict.

Runs in the **hermes venv** (to import the extractors). `metrics.py` is pure and unit-tested; `arms.py`/`run_bakeoff.py` are integration (need the API key / spaCy model).

## Implementation notes & risks

- **spaCy not installed** in the hermes venv (`spaCy not available` at import). The `spacy` arm needs `pip install spacy` + `python -m spacy download en_core_web_sm` — part of evaluating the free option; if it can't load, the arm is reported as "unavailable," not a silent skip.
- **spaCy NER ceiling:** spaCy's default model recognizes PERSON/ORG/GPE/DATE etc., not domain entities like "narwhal"/"tusk" — expect low entity recall on this corpus. That is itself a finding (the free option's ceiling), not a bug.
- **closed_vocab prompt:** the injected vocabulary is the live `logos:ontology:relations` snapshot (or, if absent, the distinct relations from the current graph) — capped to a prompt-sized list of the most frequent; the prompt instructs "reuse an existing relation when one fits; only coin a new one if none does."
- **Determinism:** LLM arms at temperature 0; still re-run twice to note any nondeterminism in the compactness numbers.
- **Small n:** 16 sentences is a first read, not a statistical verdict — enough to see a large compactness effect and a recall cliff. Extend the gold set if the result is borderline.

## Non-goals

- Not wiring any winning arm into production (that's a follow-up if an arm wins).
- Not changing the rollup/cleanup path (orthogonal; this is the source half).
- Not a full reingest — the offline harness isolates the extractor from graph/ingest noise.

## Decision criterion

Recommend an arm if it **reduces distinct-predicate count materially vs baseline while keeping relation-link recall within a small margin of baseline** (and entity-F1 not materially worse). Free/cheap arms are preferred when they clear that bar. If only `closed_vocab` clears it, that's the source fix to pursue; if `spacy`/`cheap_model` clear it, the cost win is real.
