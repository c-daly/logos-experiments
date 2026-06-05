# naming-driven-typing

**Question:** Does ONE catalog-aware LLM naming pass per candidate cluster beat the
embedding wall (flat 3072-dim distance band → over-fragmented, entity-flat typing) —
and which mechanism (reuse, graft, chain) carries the lift?

**Thesis:** embeddings POINT, the graph ASSERTS. Coarse embedding clustering proposes
candidates (recall); the naming pass does the semantic typing — reuse/mint decision,
covering hypernym, IS_A chain, graft target.

**Status:** planned — workspace not yet built.

- **Spec + plan (read first):** vault `10-projects/LOGOS/sophia/plans/naming-driven-typing/`
  (SPEC.md §0 decisions, PLAN.md T1–T7).
- **Epic:** c-daly/logos#553. Hermes contract work: c-daly/hermes#120 (canonicalize),
  c-daly/hermes#121 (/type-cluster v2). Experiment tasks T2/T3/T5/T6/T7 are tracked
  as tickets in THIS repo.
- **Eval:** label-free — structural metrics + eyeball + ablations A0–A6, K-repeat
  CI-lower-bound gating (`goal.yaml` when T7 lands).
- **Gates:** experiment verdict, then the R1 production-integration test. The offline
  run is necessary but not sufficient; no production wiring until both pass.
