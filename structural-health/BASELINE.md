# Structural-health baseline (W0.1 — logos-experiments#25, epic logos#557)

Is there low-rank relational structure in the HCG? Probe: node ×
(relation, direction, neighbor-type) matrix over semantic edges, TF-IDF,
truncated SVD explained variance, plus source-noise indicators.
Run with: `NEO4J_PASSWORD=... uv run python probe.py`

## Runs

| date | data nodes | sem. edges | edges/node | distinct rels | df=1 frac | EV all k=16/128 | EV non-typing k=16/128 |
|------|-----------|------------|------------|---------------|-----------|------------------|------------------------|
| 2026-06-04 (pre-reset)¹ | 11992 | ~13.1k² | 2.6 | (one-off junk dominant) | — | 0.012 / 0.057 | — |
| 2026-06-10 | 5469 | 13115 | 2.40 | 2244 | 0.626 | 0.066 / 0.148 | 0.025 / 0.110 |

¹ Pre-reset row from the 2026-06-04 probe (vault memo
`all-correction-signals-fail-on-current-graph`): matrix 11992 × 24695,
nnz=31352, top-16 = 1.2%, top-128 = 5.7%. Its matrix construction
predates edge reification and B2's removal of `type_uuid`; the probe was
reimplemented for the current model (see probe.py docstring), so the
comparison is like-for-like in *construction*, not in code lineage.
² The old graph's Neo4j relationship count divided by 2 directions is
approximate; the memo recorded ~2.6 edges/node.

## Gate (thresholds provisional until Chris freezes them — plan §W0.1)

- Explained variance @k=128 ≥ 3× baseline (≥ 0.171): **0.148 — NEAR MISS
  (2.6×)**. Non-typing variant (honest signature number, IS_A excluded):
  0.110.
- df=1 predicate fraction < 0.25: **0.626 — CLEAR FAIL.** 2,244 distinct
  relations on 13,115 edges; ~1,400 appear exactly once (`CONCEIVED_OF`,
  `SENT_BY`, `READ_TO`, `ANNOUNCED`, `OCCURRED_ON` …). The open relation
  vocabulary is unchanged from the 2026-06-04 diagnosis — relations never
  got the canonicalization/NDT treatment that entity names and types got.

## Verdict (2026-06-10 run)

**Mixed — latent structure has improved materially (2.6×–5.5× by k), but
the relation vocabulary is still source-noise.** Per the plan's fail
branch, structural engines (Phase 2) should not build on a 62.6% one-off
predicate vocabulary; the routed work is relation-vocabulary quality at
the source (Hermes extraction + canonical-at-the-boundary for predicates
— the "relations need NDT treatment" thread, vault memo 2026-06-08).

Where the typing layer stands (context, healthy-ish): 841 populated
types, 0 duplicate names, 7 singleton types (0.8%), 36.5% with ≤3 members;
1,215 nodes still placed flat under realm roots (entity/concept/process)
— that is the drainage population, exactly what NDT exists to work on.

Recommendation carried to the epic: recalibrate the gate to split its two
axes — let the judge (W1 MDL ledger, W2 growth prediction) proceed (they
are measurement, not extraction, and they make the source-quality work
scoreable), while Phase 2 engines wait on a relation-vocabulary fix +
re-probe.
