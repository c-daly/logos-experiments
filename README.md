# logos-experiments

Tracked, falsifiable experiments for the LOGOS cognitive architecture.

**Code lives here; knowledge lives in the vault.** Specs, plans, decision records,
and verdict narratives stay in the vault (docs only — no code in the vault, ever).
This repo holds everything executable: harnesses, eval gates, frozen fixtures, and
run outputs worth keeping.

## Structure — one directory per experiment

```
logos-experiments/
├── <experiment-name>/
│   ├── README.md          # question, verdict status, pointer to the vault spec
│   ├── goal.yaml          # falsifiable success criteria (eval-gate schema)
│   ├── pyproject.toml     # per-experiment env (uv) — experiments do not share deps
│   ├── harness/           # runners, builders, simulators
│   ├── eval/              # metrics.py :: compute_metrics(snapshot) -> [METRIC] k=v
│   ├── fixtures/          # frozen inputs (the graded run never reads live state)
│   └── workspace/         # run outputs (run_<ts>.json; large artifacts gitignored)
└── README.md              # this file
```

## Conventions

- **Self-contained experiments.** Each experiment directory carries its own
  environment (`uv` — per-experiment `pyproject.toml`/lockfile), fixtures, and
  docs pointer. Experiments may pin any service repo (hermes, logos-foundry) as
  a dependency; service repos never depend on this repo.
- **Frozen fixtures, replayable runs.** The graded run reads checked-in fixtures
  (`--replay` default, $0, deterministic); live pulls are smoke-only. Databases
  are wiped constantly during dev — reproducibility comes from reseed-from-corpus,
  never from preserved state.
- **Falsifiable gates.** Every experiment has a `goal.yaml` with success criteria
  the eval emits as `[METRIC] key=value`. A verdict is recorded in the vault
  journal — including "necessary but not sufficient" caveats where production
  integration tests gate promotion.
- **Non-mutation.** Harnesses are read-only against live systems; a post-run
  assert verifies nothing changed.
- **Tracking.** Work in this repo gets a ticket in this repo (one PR per ticket,
  branch `<type>/<repo><ticket>-<desc>`). Cross-repo efforts hang off an epic in
  `logos`.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| `naming-driven-typing/` | Does a catalog-aware LLM naming pass beat the embedding wall for emergent typing? | Planned — spec at vault `10-projects/LOGOS/sophia/plans/naming-driven-typing/`, epic c-daly/logos#553 |
