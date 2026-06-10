# NER/RE extraction bake-off (logos-experiments#38)

Offline comparison of 4 extraction arms against a hand-labeled gold set, to
find the source-side lever that cuts relation over-generation (the df=1
problem) without losing recall. Spec: `DESIGN.md`. Program: c-daly/logos#557.

## Run

```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run pip install --quiet spacy && poetry run python -m spacy download en_core_web_sm  # for the spacy arm
poetry run python ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py   # all arms -> results.json
```

Metrics are pure + unit-tested (`metrics.py`, 11 tests). Entity/relation
matching is on canonical forms (`hermes.canonical`). **link F1** scores the
directional (source→target) pair *ignoring the label* ("did it find the
relationship?"); **label F1** requires the canonical relation to match too
("did it label it right, not over-invent?"). **df=1 / distinct_predicates**
measure over-generation.

## Gold set

40 curated sentences (16 from `naming-driven-typing/corpus/corpus.jsonl` + 24
authored in the same clean style) — 112 entities, 64 relations, **20 distinct
gold relations** (the compact reuse target). `gold.jsonl`.

## Results (2026-06-10, n=40)

| arm | entity F1 | type acc | link recall | link F1 | label F1 | distinct preds | df=1 |
|-----|-----------|----------|-------------|---------|----------|----------------|------|
| baseline (gpt-4o-mini, production combined) | 0.731 | 0.11 | 0.388 | 0.336 | 0.18 | 44 | 0.886 |
| **closed_vocab** (gpt-4o-mini + injected vocab) | **0.811** | 0.287 | **0.500** | **0.493** | 0.171 | **32** | **0.594** |
| big_model (gpt-4o, open prompt) | 0.865 | 0.406 | 0.575 | 0.514 | 0.152 | 61 | 0.869 |
| spacy (local en_core_web_sm) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — |

*(baseline's 232 s vs the others' ~50 s is fetch-timeout noise — its production
prompt tries to fetch the Sophia type list, which was unreachable and retried;
not a model-speed signal.)*

## Verdict

**closed_vocab — injecting the known relation vocabulary into the prompt with
reuse-don't-mint pressure — is the source-side lever.** At the *same cheap
model and cost* it cut distinct predicates 44 → 32 and df=1 0.886 → 0.594,
**and** improved entity F1 (0.731 → 0.811) and relation link-recall
(0.388 → 0.500). It reduces over-generation while extracting *better*.

Two clean negatives that matter just as much:

- **A bigger model is counterproductive for sprawl.** `big_model` (gpt-4o)
  had the best recall but the *worst* over-generation (61 distinct
  predicates vs baseline's 44) — a smarter model invents *more* distinct
  labels, not fewer. **Spend is not the lever.** Staying on the cheap model
  is correct; the fix is the prompt, not the tier.
- **spaCy-local is a non-starter for this domain.** `en_core_web_sm` NER
  recognizes proper nouns (people/orgs/places/dates), not common-noun domain
  concepts — it found ~zero entities on this corpus (e.g. "narwhal", "tusk",
  "baker" → nothing), so RE had nothing to relate. Free, but useless here
  without a custom-trained model.

**Caveat / strongest follow-up:** closed_vocab was *handicapped* — the vocab
it injected was the LIVE `logos:ontology:relations` snapshot, i.e. the
already-sprawled ~2,300-relation vocabulary (120 alphabetical entries:
`ABBREVIATED_AS, ABOUT, ABSENT_FROM, …`). It still helped. Injecting a CLEAN,
compact target vocabulary (like the 20 gold relations) should cut df=1
further and lift label F1 (low across all arms because the extractors' free
labels don't match the gold's specific canonical choices). df=1 0.594 is
still above the program's 0.25 gate at this scale — but the direction is now
empirical: **closed-vocabulary prompt pressure with a clean target vocabulary,
on the current cheap model, is the source fix** (W0.3 / lx#38). The rollup
(sophia#192) cleans the existing tail; this stops new sprawl at the source.
