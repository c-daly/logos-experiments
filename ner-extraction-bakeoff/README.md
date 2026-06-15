# NER/RE extraction bake-off (logos-experiments#38)

Offline comparison of 10 extraction arms against a hand-labeled gold set, to
find the source-side lever that cuts relation over-generation (the df=1
problem) without losing recall. Spec: `DESIGN.md`. Program: c-daly/logos#557.

## Run

```bash
cd /home/fearsidhe/projects/logos-workspace/hermes
poetry run pip install --quiet spacy && poetry run python -m spacy download en_core_web_sm  # for the spacy arms
poetry run python ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py            # all arms -> results.json
poetry run python ../logos-experiments/ner-extraction-bakeoff/run_bakeoff.py closed_vocab_clean  # one arm
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

## Arms

| arm | nodes | relations |
|-----|-------|-----------|
| `baseline` | gpt-4o-mini production combined extractor | open prompt |
| `closed_vocab` | gpt-4o-mini, open NER + RE | RE constrained to **live** (sprawled ~2.3k) vocab snapshot |
| `closed_vocab_clean` | gpt-4o-mini, open NER + RE | RE constrained to **clean** ~32-relation vocab |
| `big_model` | gpt-4o, open NER + RE | open prompt |
| `big_model_clean` | gpt-4o, open NER + RE | RE constrained to clean vocab |
| `spacy` | `en_core_web_sm` `doc.ents` | dependency-parse heuristic |
| `spacy_pos` | spaCy NOUN/PROPN tokens | dependency-parse heuristic |
| `spacy_chunks` | spaCy `noun_chunks` (det-stripped) | dependency-parse heuristic |
| `spacy_head` | spaCy chunk **root head noun** ("just the noun") | dependency-parse heuristic |
| `hybrid_clean` | FREE spaCy noun-chunk nodes | cheap gpt-4o-mini relations-only pass, clean vocab |

## Results (2026-06-10, n=40)

| arm | entity F1 | type acc | link recall | link F1 | label F1 | distinct preds | df=1 | s |
|-----|-----------|----------|-------------|---------|----------|----------------|------|---|
| baseline | 0.745 | 0.131 | 0.375 | 0.326 | 0.173 | 43 | 0.884 | 180 |
| closed_vocab | 0.842 | 0.296 | 0.550 | 0.522 | 0.171 | 34 | 0.588 | 51 |
| **closed_vocab_clean** | **0.854** | 0.390 | **0.550** | 0.514 | **0.439** | **26** | 0.538 | 47 |
| big_model | 0.886 | 0.358 | 0.613 | 0.556 | 0.146 | 59 | 0.915 | 40 |
| big_model_clean | 0.843 | **0.442** | **0.588** | **0.545** | **0.460** | 29 | **0.483** | 45 |
| spacy | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — | 1 |
| spacy_pos | 0.662 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — | 0.3 |
| spacy_chunks | 0.596 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — | 0.3 |
| **spacy_head** | **0.713** | 0.0 | 0.0 | 0.0 | 0.0 | 0 | — | 0.3 |
| hybrid_clean | 0.596 | 0.0 | 0.275 | 0.220 | 0.174 | 26 | 0.538 | 59 |

*(baseline's 180 s vs the others' ~45 s is fetch-timeout noise — its production
prompt tries to fetch the Sophia type list, which was unreachable and retried;
not a model-speed signal.)*

## Verdict

**The source-side lever for relation sprawl is a clean closed vocabulary on
the current cheap model — not a bigger model, and not local NER.** Four
findings, each with a clean control:

### 1. Clean closed-vocabulary prompting is the relation lever (and it's free of model spend)

Holding the model fixed at gpt-4o-mini, swapping the injected vocab from the
**live sprawled snapshot** (`closed_vocab`) to a **clean ~32-relation target**
(`closed_vocab_clean`) was the single biggest relation gain in the sweep:

- label F1 **0.171 → 0.439** (≈2.6×) — it labels relations correctly instead of inventing
- distinct predicates **34 → 26**, df=1 0.588 → 0.538
- entity F1 also best-in-cheap-tier at 0.854

The *vocabulary content* moved the needle far more than the *model*. This is
the production change to ship for the NDT source fix: constrain the combined
extractor's RE step to a clean, curated target vocabulary.

### 2. A bigger model is counterproductive for sprawl (open prompt), and only ties on clean vocab

- `big_model` (gpt-4o, **open** prompt) had the best link recall (0.613) but
  the **worst** over-generation in the whole sweep — 59 distinct predicates,
  df=1 0.915. A smarter model with a free hand invents *more* distinct labels.
- `big_model_clean` (gpt-4o + clean vocab) is the nominal best on relations
  (label F1 0.460, df=1 0.483) — but it only edges the **cheap**
  `closed_vocab_clean` (0.439 / 0.538) by a hair, at gpt-4o cost.

**Spend is not the lever — the vocabulary is.** The cheap model with a clean
vocab captures ~95% of the achievable relation quality. Staying on gpt-4o-mini
is correct.

### 3. spaCy can find common-noun NODES for free, but not relations

- Off-the-shelf `en_core_web_sm` `doc.ents` is a **non-starter** (0.0) — it
  recognizes proper nouns (people/orgs/places/dates), not common-noun domain
  concepts ("narwhal", "tusk", "baker" → nothing).
- But spaCy's *syntactic* primitives do surface domain concepts for free:
  **head-noun** (`spacy_head`, the chunk root noun — "just the noun") is the
  best free node primitive at **entity F1 0.713**, within ~0.14 of the best
  LLM (0.854) at **zero cost and ~0.3 s vs ~45 s**. NOUN/PROPN tokens (0.662)
  and noun-chunks (0.596) trail it.
- **No spaCy arm extracts relations** — every dependency-parse heuristic scored
  0 on relation linking. Relations need the LLM.

### 4. You can't cheaply decouple free nodes from LLM relations

`hybrid_clean` (free spaCy chunk nodes + cheap LLM relations-only pass)
**underperforms** the all-LLM `closed_vocab_clean` on relations — link recall
**0.275 vs 0.550**, label F1 0.174 vs 0.439. The spaCy node set (entity F1
0.596) is the bottleneck: the relations-only pass can only link entities spaCy
already surfaced, so every node spaCy misses is a relation it can't recover.
Garbage-in. The LLM extracting entities and relations *together* lets it
co-select the endpoints it intends to link — that coupling is load-bearing.

## What to ship & what's still open

**Ship (source fix, W0.3 / lx#38):** constrain the combined extractor's RE
step to a **clean, curated target relation vocabulary** on the current cheap
model (gpt-4o-mini). Keep the LLM for entity extraction (it co-selects the
endpoints relations link). spaCy head-noun is a viable **free candidate
augmenter / fallback** for nodes (0.713), but cannot stand alone without
killing relation recall.

**Still open:** df=1 0.538 (cheap) / 0.483 (gpt-4o) is still above the
program's **0.25 gate** at n=40. Closed-vocab + clean vocab is the biggest
single source lever but is *necessary, not sufficient*: the gate also needs
(a) the downstream rollup (sophia#192) to collapse the existing sprawled tail,
and (b) a tighter, curated production vocabulary than the ad-hoc 32-entry clean
set used here. The direction is now empirical — **stop new sprawl at the source
with clean closed-vocab prompting; clean the legacy tail with the rollup.**
