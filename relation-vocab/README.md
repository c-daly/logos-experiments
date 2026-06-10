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
- **keep** (548) — no evidence, OR a **polarity marker** (`NOT`/`NO`/
  `NEVER`/`CANNOT`/`WITHOUT`/`NON`). Negated predicates are never
  auto-mapped: `DOES_NOT_REFER_TO→REFERS_TO` would flip meaning. Many
  keeps are genuinely distinct relations.

## First run (2026-06-10)

| tier | count | trust |
|------|-------|-------|
| high | 93 | accept on glance |
| medium | 615 | exact yes; lossy needs review |
| low | 148 | candidate, usually wrong |
| keep | 548 | distinct, or polarity-guarded |

**With-evidence coverage: 856/1404 = 61.0% — the ticket's ≥80% gate is NOT
met by the automatic proposer alone**, and that is reported, not papered
over. The shortfall is honest: ~39% of one-offs have no mechanical bridge
to an existing predicate, because the vocabulary is genuinely
open-ended — which is exactly why `hermes#130` (closing the vocabulary at
extraction time) is the load-bearing half. This instrument's job is to
make the *existing* tail cheap to review, not to reach 80% by inflating
weak matches.

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

- `propose.py` — pure matching/folding functions (unit-tested, no Neo4j).
- `propose_mapping.py` — live read-only runner → `mapping.csv`.
- `mapping.csv` — the review artifact (regenerated per run).
