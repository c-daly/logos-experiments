# Structural-health baseline (W0.1 — logos-experiments#25, epic logos#557)

Is there low-rank relational structure in the HCG? Probe: node ×
(relation, direction, neighbor-type) matrix over semantic edges, TF-IDF,
truncated SVD explained variance, plus source-noise indicators.
Run with: `NEO4J_PASSWORD=... uv run python probe.py`

## Runs

| date | data nodes | sem. edges | edges/node | distinct rels | df=1 frac | EV all k=16/128 | EV non-typing k=16/128 | edges/pred | top-10/50 share |
|------|-----------|------------|------------|---------------|-----------|------------------|------------------------|-----------|-----------------|
| 2026-06-04 (pre-reset)¹ | 11992 | ~13.1k² | 2.6 | (one-off junk dominant) | — | 0.012 / 0.057 | — | — | — |
| 2026-06-10 | 5469 | 13115 | 2.40 | 2244 | 0.626 | 0.065 / 0.145 | 0.024 / 0.108 | 5.84 | — |
| 2026-06-11³ | 4499 | 8873 | 1.97 | 951 | 0.591 | 0.202 / 0.497 | 0.173 / 0.478 | 9.33 | — |
| 2026-06-11 (bakeoff arm)⁴ | 1425 | 2775 | 1.95 | 344 | 0.558 | 0.269 / 0.634 | 0.260 / 0.627 | 8.07 | 0.702 / 0.835 |
| ~~2026-06-11 (reference)⁵~~ INVALID | 1942 | 3750 | 1.93 | 516 | 0.583 | 0.214 / 0.529 | 0.188 / 0.514 | 7.27 | 0.659 / 0.791 |
| 2026-06-11 (reference v2)⁶ | 5517 | 12368 | 2.24 | 1952 | 0.606 | 0.218 / 0.407 | 0.187 / 0.382 | 6.34 | 0.584 / 0.672 |

¹ Pre-reset row from the 2026-06-04 probe (vault memo
`all-correction-signals-fail-on-current-graph`): matrix 11992 × 24695,
nnz=31352, top-16 = 1.2%, top-128 = 5.7%. Its matrix construction
predates edge reification and B2's removal of `type_uuid`; the probe was
reimplemented for the current model (see probe.py docstring), so the
comparison is like-for-like in *construction*, not in code lineage.
² The old graph's Neo4j relationship count divided by 2 directions is
approximate; the memo recorded ~2.6 edges/node.
³ Fresh corpus_batch3 (350-block) ingest through hermes with the
usage-ranked H5 window (hermes 297391a). Same-corpus comparison against
the alphabetical-window ingest earlier that day (not in this table; df=1
measured only): distinct non-typing predicates 1640 -> 950 (-42%), df=1
singletons 1002 -> 562 (-44%), df=1 fraction 0.611 -> 0.592. The EV jump
vs the 06-10 row is graph-to-graph (old accumulated graph vs fresh
corpus ingest), not a controlled window comparison.
⁴ Instrument-validation run for the new gate columns: at run time the
live DB held a small NER-bakeoff arm ingest (1,425 data nodes; every
IS_A edge flat to the seeded realm skeleton, 0 specifically-typed
nodes), so the row is not comparable to prior rows and is not a gate
readout. First population of edges/pred and top-10/50 concentration;
prior rows backfilled where derivable (sem. edges ÷ distinct rels).
⁶ THE reference readout (bar-freeze row, replacing the voided ⁵):
corpus_batch3 through MERGED main (hermes b75c3cd: ranked window + name
cleaning + registry types + free typing; sophia 6016f03: snapshot at
boot), single clean feed, settle verified (queue drained, maintenance
converged, 0 errors both service logs in window). Seeded-state
disclosure: the ranked window was COLD at ingest (snapshot held the
~30-predicate smoke-era vocabulary), so minting ran near-open — 1,952
distinct relations; future readouts will carry warm snapshots and the
recipe must record window size per run. Matched-scale comparison: the
06-10 row measured EV@128 = 0.145 at 5,469 data nodes; this row reads
0.407 at 5,517 — 2.8× at equal scale, the cleanest gate evidence yet.
⁵ THE reference-corpus readout (bar-freeze row): clean wipe + skeleton
seed (HCGSeeder), single pass of corpus_batch3 (350 blocks) through the
deployed hermes /llm gateway (4 workers, temperature 0.0; hermes
proposal → sophia background ingestion), settled to queue drain before
probing. Note the size delta vs the ³ row (1,942 vs 4,499 data nodes):
³'s graph plausibly accumulated both same-day feeds (alpha then ranked)
without a wipe between; this row is a single clean feed and is the
recipe future readouts must reproduce.

## Gate (recalibrated 2026-06-11 — readout below is the 2026-06-10 run)

Recalibrated 2026-06-11 (logos#557 comment; plan §W0.1 amended): df=1
fraction no longer gates — it failed to respond to the H5 ranked-window
fix (singleton mints −44%, fraction 0.626 → 0.591) and penalizes
legitimate domain-specific facts. The gate reads on EV@128 ≥ 3× baseline
(≥ 0.171) with edges-per-predicate and top-10/50 concentration judged
alongside. There is no persistent graph at this stage — the DB is wiped
and reseeded with test corpora — so readouts are corpus-controlled: bars
freeze from the first reference-corpus readout, and future readouts
hold the corpus fixed. The ⁴ row (a 50-block bakeoff arm) is not that
readout; the ⁵ row IS.

**Bars frozen 2026-06-11 from the ⁶ reference readout** (the ⁵-derived
floors are void with their row; corpus_batch3 + merged-main recipe per
footnote ⁶; floors for future pipeline changes, Chris
confirms/recalibrates per plan §W0.1):
- EV@128 ≥ 0.171 (3× the 06-04 baseline) — readout **0.407, PASS 2.4×**
  (non-typing 0.382; matched-scale vs the 06-10 row: 0.145 → 0.407 at
  ~5.5k data nodes, 2.8×)
- edges-per-predicate ≥ 6.0 — readout 6.34 (floor ~5% below readout;
  tolerance provisional until a repeat run quantifies variance)
- top-10 concentration ≥ 0.55, top-50 ≥ 0.64 — readout 0.584 / 0.672
  (same provisional tolerance; concentration is one-sided — a CEILING
  may eventually be needed against pathological collapse into few
  predicates)
- Seeded-state rule: every future readout records window size + snapshot
  provenance (the ⁶ row ran cold at ~30 advertised predicates)

**Engines-gate verdict (recalibrated metrics, valid readout): PASS.**

**Engines-gate verdict on the recalibrated metrics: PASS.** Phase 2
engines (W3 rule mining, W4 structural typing, W5 FCA, W6 motifs)
unblock per the plan's gate logic.
The hapax tail is handled by periodic consolidation (sophia#194 rollup,
lx#34 mapping) as a light groomer, not a gate-crosser.

- Explained variance @k=128 ≥ 3× baseline (≥ 0.171): **0.145 — NEAR MISS
  (2.5×)**. Non-typing variant (honest signature number, IS_A excluded):
  0.108. (Numbers are from the deterministic multi-IS_A tie-break added in
  review — 104 data nodes carry multiple IS_A edges; the pre-review run's
  0.148 was order-dependent.)
- df=1 predicate fraction < 0.25: **0.626 — CLEAR FAIL.** 2,244 distinct
  relations on 13,115 edges; ~1,400 appear exactly once (`CONCEIVED_OF`,
  `SENT_BY`, `READ_TO`, `ANNOUNCED`, `OCCURRED_ON` …). The open relation
  vocabulary is unchanged from the 2026-06-04 diagnosis — relations never
  got the canonicalization/NDT treatment that entity names and types got.

## Verdict (2026-06-10 run)

**Mixed — latent structure has improved materially (2.5×–5.4× by k), but
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
