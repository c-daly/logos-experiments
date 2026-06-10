# Relation-vocabulary consolidation (W0.3b — logos-experiments#34, epic logos#557)

The one-time cleanup half of the relation-vocabulary fix. The W0.1 probe
(`structural-health/`) found 2,244 distinct predicates on 13,115 edges,
**62.6% occurring exactly once**; the W1 ledger (`mdl-ledger/`) priced the
relation vocabulary at **35,856 bits — 62% of the model cost**. This
instrument proposes a consolidation target for every df=1 predicate so the
existing graph can be cleaned; the extraction-time fix that stops new
one-offs being minted is `hermes#130`.

Run: `NEO4J_PASSWORD=... uv run python propose_mapping.py` → `mapping.csv`.

## What it is — and is NOT

It writes a **reviewable proposal table**, one row per df=1 predicate:
`predicate, df, proposed_target, tier, evidence, review`. **Nothing is
applied to the graph.** The `review` column is filled by hand; an approved
table is applied by a separate, reviewed maintenance step. The matching
machinery (a crude stemmer + token/signature heuristics) only *proposes* —
language labels, it never decides.

Three evidence tiers, strongest first:
- **high** (93) — canonical-form collision: the stemmer folds the one-off
  onto an existing predicate (`ACQUIRE→ACQUIRES`, `BINDS_TO→BIND_TO`) or
  onto a group of fellow one-offs that share a stem (`ACTS_IN→ACT_IN`).
  These are near-mechanical and mostly safe to accept on a glance.
- **medium** (615) — content-token match against a head predicate. Exact
  (`ILLUMINATED_BY→ILLUMINATES`, j=1.0) is trustworthy; **lossy** matches
  drop extra tokens (`STUDIES_RELATIONSHIPS_AMONG→STUDIES`) and need a
  real look — some are right, some collapse a specific relation into a
  vague one.
- **low** (148) — signature only: a head predicate has an edge with the
  same (source-type, target-type) pair. Weak by construction
  (`EQUALS→EXCEEDS`, `CHALLENGES→PRODUCES` are coincidences); treat as
  "here's a candidate, probably wrong."
- **keep** (528) — no evidence, OR a **polarity marker** (`NOT`/`NO`/
  `NEVER`/`CANNOT`/`WITHOUT`/`NON`). Negated predicates are never
  auto-mapped: `DOES_NOT_REFER_TO→REFERS_TO` would flip meaning. Many
  keeps are genuinely distinct relations.
- **embed** (20) — *complementary name-embedding pass* (`embed_evidence.py`):
  for a row the cascade above left as a bare `keep`, the nearest surviving
  predicate by embedding cosine (OpenAI `text-embedding-3`). At **≥0.85** it
  is promoted to a proposal — these are synonyms with **no shared tokens**
  the token pass cannot see (`AFFILIATED_WITH→ASSOCIATED_WITH` 0.87,
  `ANALYSED→ANALYZED` 0.96, `ADOPTED_BY→ADOPTED_IN` 0.91). Below 0.85 the row
  stays `keep` but records its nearest neighbour (`ACCOMPANIED_BY` → nearest
  `ASSOCIATED_WITH` 0.59), so every one-off now carries evidence. The
  machinery only *proposes*; the review column still decides.

## First run (2026-06-10)

| tier | count | trust |
|------|-------|-------|
| high | 93 | accept on glance |
| medium | 615 | exact yes; lossy needs review |
| low | 148 | candidate, usually wrong |
| embed | 20 | synonym proposal (no shared tokens) |
| keep | 528 | distinct, or polarity-guarded |

**Proposal coverage (rows with a fold target): 876/1404 = 62.4%** — up from
61.0%. The embedding pass adds **20** fold proposals the mechanical cascade
structurally cannot find (synonyms with *no shared tokens*). **The ticket's
≥80% gate is still NOT met by the proposer**, and that is reported, not
inflated: ~38% of one-offs have no defensible fold target because the
vocabulary is genuinely open-ended — which is exactly why `hermes#130`/`#140`
(closing the vocabulary at extraction time) is the load-bearing half.

What the embedding pass adds *beyond* those 20 folds is a **review aid, not a
gate number**: every remaining `keep` now carries its nearest survivor and
cosine as evidence (`ACCOMPANIED_BY` → nearest `ASSOCIATED_WITH` 0.59), so all
1404 rows are annotated. That makes the genuinely-distinct keeps cheap to
*confirm* — a 0.59-cosine keep is still a keep, never a proposal, and is **not
counted toward the ≥80% gate**.

### The ≥80% *table* gate is met; the *graph* gate still needs the source fix

An embedding-only sensitivity sweep (nearest survivor per one-off, varying
the map-cutoff) shows what consolidation can buy on df=1 itself:

| embed map-cutoff | projected df=1 |
|------------------|----------------|
| 0.85 | 0.574 |
| 0.72 | 0.484 |
| 0.65 | 0.386 |
| **0.60 (aggressive)** | **0.276** |

Even folding **aggressively at 0.60 cosine**, the tail floors at **df=1 ≈
0.276 — still above the program's 0.25 gate**: ~340 one-offs have no survivor
within 0.60 cosine and are genuinely distinct. **Cleaning the legacy tail is
necessary but not sufficient.** The 0.25 gate is unreachable by consolidation
alone; it needs the extraction-time source fix (`hermes#140` closed-vocab
prompting, the companion to `hermes#130`) to stop new one-offs being minted,
after which df=1 falls as new edges reuse existing predicates. This
quantifies the bake-off's "necessary, not sufficient" verdict
(logos-experiments#38). This instrument's job remains to make the *existing*
tail cheap to review.

### What approving the strong tiers would buy (estimate)

High + exact-medium consolidation folds on the order of ~500–700 distinct
predicates away. At 16 bits per relation in the model term that is roughly
**8k–11k bits off `L_model`** directly, plus cheaper per-edge relation
codes in `L_data_edges` (the dominant term) as the surviving predicates
get higher document frequency. The W1 ledger is the meter: re-run
`mdl-ledger/live.py` after an approved mapping is applied to read the
actual ΔL. Program gate (epic #557): re-run the W0.1 probe after both this
and hermes#130 land — full pass = EV@128 ≥ 0.171 AND df=1 < 0.25.

## Files

- `propose.py` — pure matching/folding functions + `apply_embed_fallback`
  (unit-tested, no Neo4j, no network).
- `embed_evidence.py` — name-embedding nearest-survivor evidence (OpenAI HTTP,
  caches vectors under `.cache/`, git-ignored).
- `propose_mapping.py` — live read-only Neo4j runner → `mapping.csv`; applies
  the embedding pass inline (fail-soft without `OPENAI_API_KEY`).
- `enrich_mapping.py` — post-hoc: add embedding evidence to an existing
  `mapping.csv` off a frozen `snapshot.json`, when Neo4j isn't available.
- `mapping.csv` — the review artifact. `snapshot.json` — frozen relation
  snapshot, so the embedding enrichment is reproducible.
