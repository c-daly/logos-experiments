# naming-driven-typing

**Question:** Does a catalog-aware naming pass type a coarse cluster better than
the current tier-2/rollup pipeline? Embeddings POINT (cheap coarse clustering);
the graph ASSERTS (one LLM call per cluster returns a covering hypernym, an IS_A
chain to a root, a reuse-or-mint decision, and a graft target). We validate this
OFFLINE on frozen clusters with a SIMULATED placement cascade — no production
mutation.

**Status:** T2-T7 landed and tested — catalog, fixtures layer, cascade
simulator, K-repeat harness with ablation arms A0-A6, and the eval layer.
Issue #13 added the live wiring: the real non-mutation probe
(`harness/probe.py`), the gated K-sample freeze (`harness/freeze.py`), the
reseed driver CLI (`harness/reseed.py`) and the blessed batch3 corpus. The
paid freeze and the graded reseed are operator-driven (see the graded
pipeline below). Verdict narratives are vault-side, post-run.

- **Spec + plan (read first):** vault `10-projects/LOGOS/sophia/plans/naming-driven-typing/`
  (SPEC.md §0 decisions, PLAN.md T1–T7).
- **Epic:** c-daly/logos#553. Hermes contract work: c-daly/hermes#120 (canonicalize),
  c-daly/hermes#121 (/type-cluster v2). Experiment tasks T2/T3/T5/T6/T7 are tracked
  as tickets in THIS repo.
- **Gates:** experiment verdict, then the R1 production-integration test. The offline
  run is necessary but not sufficient; no production wiring until both pass.

## Eval is label-free (Chris 2026-06-05)

No per-cluster coherence labels, no root ground-truth, no external hypernymy
oracle, no label-derived precision/recall/accuracy. Eval is:

1. **Intrinsic STRUCTURAL metrics** read off the cascade output
   (`eval/metrics.py`): `graft_depth_fraction`, semantic `reuse_collapses`
   (string `canonical_merge_collapses` reported separately), `residual_fraction`,
   `raw_partition_violation_rate`, `hallucinated_target_rate`,
   `placement_conflict_rate`, descriptive `root_distribution`, `mean_graft_depth`,
   `new_floated_at_root`.
2. **Eyeballing** the per-cluster decision dump (`eval/metrics.py --eyeball`).
3. **Ablations A0–A6** — full-v2 (A6) must beat naive-LLM (A1) by more than the
   K-repeat noise band (A0 is the MEASURED tier-2/rollup baseline, not asserted).

All metrics are reported `mean ± stdev` over K≥5 repeats; `stability_cv` is
emitted; a criterion passes only at the CI LOWER bound; a comparison "passes"
only if its delta clears the noise band.

## Layout

```
naming-driven-typing/
  goal.yaml                  label-free success criteria + objective
  pyproject.toml             per-experiment env (uv); pytest-only for the eval layer
  harness/run_experiment.py  K-repeat runner: --replay (default) / --live / --freeze — T6
  harness/catalog.py         uuid-keyed enriched catalog + by_norm (read-only) — T2
  harness/cascade.py         simulated placement cascade (no production mutation) — T5
  harness/ablations.py       arm wiring A0-A6 (registry views, prompt transforms)
  harness/fixtures_io.py     canonical freeze/load for clusters, catalog, llm responses
  harness/probe.py           real non-mutation probe (Neo4j count, Redis key hash)
  harness/freeze.py          gated K-sample freeze through the deployed hermes gateway
  harness/reseed.py          RESEED_LIVE-gated clean reseed driver (--graded = batch3)
  corpus/corpus_batch3.jsonl blessed graded corpus (350 blocks / 8 domains)
  eval/metrics.py            structural metrics -> [METRIC] lines + ablation deltas + eyeball dump
  eval/fixtures/             committed snapshot fixtures (label-free; schema contract for T5/T6)
  tests/                     unit tests for the eval layer
  workspace/run_<ts>.json    per-run snapshots (all K repeats; written by harness)
```

Run journals and verdict narratives live in the vault
(`10-projects/LOGOS/`), not in this repo.

## Run

The eval layer is self-contained (uv, no service deps):

```bash
cd naming-driven-typing
uv sync

# Tests:
uv run pytest tests/ -v

# Score a snapshot (default = newest workspace/run_*.json); --eyeball appends
# the human-readable per-cluster decision dump:
uv run python eval/metrics.py eval/fixtures/run_synthetic.json --eyeball
```

The live paths (--live / --freeze / reseed) run in an env where `logos_hcg`, `logos_config`,
`pymilvus`, `neo4j` AND `hermes` import (path-A drives the hermes handler
in-process; the hermes poetry env carries the rest via its foundry pin —
re-verify at execution). Default is `--replay` from frozen fixtures
($0, deterministic).

### Stack (smoke / `--live` only — never the graded run)

| service | default |
|---------|---------|
| Neo4j   | `bolt://localhost:7687` (neo4j / logosdev) |
| Milvus  | `localhost:19530` |
| Hermes  | `http://localhost:17000` (throwaway only; prod URL is asserted untouched) |

### Graded pipeline (issue #13)

Three steps, in order; only step 2 spends money:

```bash
# 1. Clean reseed of the DISPOSABLE stack (graded corpus = the blessed
#    corpus/corpus_batch3.jsonl, 350 blocks / 8 domains, approved 2026-06-05;
#    omit --graded for the 16-block smoke corpus). Freezes clusters+catalog
#    through the canonical fixture writers.
RESEED_LIVE=1 uv run --no-sync python harness/reseed.py --graded

# 2. K-sample freeze (the ONE paid step): K repeats per cluster per
#    prompt-distinct arm (full, naive_llm, no_graft, no_chain; no_reuse and
#    no_gate share the full fixture), pinned temperature=0.0, raw completions
#    frozen in the replayer shape + model snapshot id in fixtures/freeze_meta.json.
#    Prints a cost estimate and refuses without the explicit acknowledgement.
LIVE_RUN=1 HERMES_URL=http://localhost:17000 \
  uv run --no-sync python harness/run_experiment.py --freeze --repeats 5 --yes-i-will-pay

# 3. Graded runs: $0, deterministic, per arm.
uv run --no-sync python harness/run_experiment.py --replay --repeats 5 --ablation full
```

`--live` is a PAID smoke/illustration run only (never the graded one, SPEC
7.6): frozen fixture inputs, the live LLM through the deployed hermes
gateway, and the REAL non-mutation probe (Neo4j type-def count, prod Redis
key `logos:ontology:types` content hash, hermes TypeRegistry count) asserted
before the snapshot persists. Same gates as the freeze: `LIVE_RUN=1`,
`HERMES_URL`, `--yes-i-will-pay`.

## Metrics

`eval/metrics.py` emits `[METRIC] key=value` lines (aggregate metrics emit
`.mean` / `.stdev` / `.cv`):

| metric | meaning | reading |
|--------|---------|---------|
| `reuse_collapses` | semantic reuse merges into a published type | `> 0` (primary) |
| `graft_depth_fraction` | NEW types grafting under a non-root parent | `>= 0.5` (primary) |
| `raw_partition_violation_rate` | RAW LLM id loss/dup before reconciliation | `<= 0.1` (primary) |
| `residual_fraction` | members parked to residual / total | `<= 0.3` (`residual_bloat` flags > 0.4) |
| `stability_cv_max` | worst CV across aggregate metrics | `<= 0.1` (primary) |
| `placement_conflict_rate` | v2 vs rollup parent disagreement | reported go/no-go |

## Results land in

`workspace/run_<ts>.json` — one snapshot per run, holding all K repeats for
every cluster (groups with branch/parent/depth/reuse/graft flags, residual ids,
evicted ids, `raw_partition_ok`), plus the enriched-catalog assertions
(`roots_present_in_live_catalog`, `live_redis_catalog_staleness`). The offline
run is **necessary-but-not-sufficient**: cycle / idempotency / dedup /
residual-durability live in unrun write paths, gated by the blocking R1
production integration test.
