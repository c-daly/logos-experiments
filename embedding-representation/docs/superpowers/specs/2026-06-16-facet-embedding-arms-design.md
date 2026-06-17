# Facet embedding arms — sense vs structure

**Date:** 2026-06-16
**Status:** approved (design)
**Experiment:** `logos-experiments/embedding-representation/`
**Builds on:** `RECORD.md` (the lab notebook) and the vault `2026-06-15-hcg-*` facet thread.

## Motivation

`RECORD.md` closed on an open problem: every representation tried so far is
either chunk-independent but coarse (`name` — best-cut silhouette 0.062) or
more separable but **chunk-coupled** (`sentence` 0.181 at ~180× the random
chunk-neighbour rate — false coherence: the vectors cluster by source passage,
not by type). The target — *fine-grained AND chunk-independent* — was reached
by no arm.

The session-close conclusion (vault `hcg-proper-text-representation-is-facets-not-one-vector`)
reframes the entity not as one vector but as **facets**. Two of those facets
are chunk-independent by construction and untested:

- **sense** — what the entity *means*: a short, chunk-independent gloss.
- **relations** — what the entity is *connected to, and how*: its structural
  signature over the graph.

This experiment adds both as parallel arms and scores them against the existing
bar on the one metric the positive control proved diagnostic (best-cut
silhouette), jointly with the chunk-collapse diagnostic.

## Goal & hypothesis

**Hypothesis.** A chunk-independent facet lifts best-cut silhouette toward
control territory (≥ ~0.18, ideally toward the 0.213 ceiling) **while keeping
`chunk_ratio` at its floor** — the joint outcome (high silhouette *and* low
chunk coupling) that no prior arm achieved.

- `gloss` target: silhouette ↑, `chunk_ratio` ≈ name floor (~37×).
- `relations` target: silhouette ↑, `chunk_ratio` ≈ chance (structure carries
  no chunk signal by construction — a sanity check, not a risk).

## Arms

| arm | what is embedded / compared | expected chunk coupling | prior result |
|---|---|---|---|
| `name` (baseline) | live bare-name vectors, pulled free from `hcg_entity_embeddings` | floor (~37×) | silhouette 0.062 |
| `sentence`, `name_sentence`, `marked` (existing) | kept for continuity | ~180× (false) | 0.18 / 0.13 / 0.15 |
| **`gloss`** *(new)* | LLM definition string → embed (3-large/3072) | should stay ~37× | — |
| **`name_gloss`** *(new)* | `"{name} — {gloss}"` → embed | should stay ~37× | — |
| **`relations`** *(new)* | `build_signature` → weighted-Jaccard distance | chunk-blind by construction | — |
| *control ceiling* | 80 known-category bare names | — | 0.213 (reference only) |

## Scoring — two paths, one metric

Both new arms land on the **same** best-cut silhouette already in `rescore.py`
(`_distance_matrix` → `_agglomerative_partitions` → `_silhouette`), so results
are directly comparable.

- **Text arms** (`name`, `gloss`, `name_gloss`): cosine → best-cut silhouette +
  `chunk_ratio` (`nn_chunk_rate`) + the existing battery, exactly as today.
- **`relations`**: each entity → `build_signature` (Counter of
  `(relation_type, neighbor_type)`) → pairwise **weighted-Jaccard distance
  matrix** (`1 - signature_similarity`) → fed straight into `_silhouette`
  (it consumes a distance matrix, so the metric switch is transparent).
  `chunk_ratio` still computed as a sanity check (expect ≈ chance).
- **Coverage:** report the fraction of the 5,330 entities with a non-empty
  signature; score `relations` on the covered subset and state n explicitly.

## Data plumbing

- **`gloss.py`** (new) — generate one definition per *unique name* (~5,131),
  from `name + its sentence`. Prompt asks for a single self-contained sentence
  defining the concept and **explicitly forbids quoting or paraphrasing the
  passage** (the sentence disambiguates the sense; the output must not carry
  chunk vocabulary). Temperature 0. Cache to `.cache/glosses.json`
  (`{name: gloss}`) — auditable and free on re-run. Generator: OpenAI
  `gpt-4o-mini` via the existing httpx/`OPENAI_API_KEY` path.
- **`signatures.py`** (new, or extend `sample.py`) — for each entity uuid pull
  incident edges via the production HCG adapter (`hcg.query_edges_from` +
  `hcg.get_nodes_batch`), assemble `neighbors=[{relation, neighbor_type}]`
  exactly as `emergence_handler` does, and persist `signatures.json` =
  `{uuid: [[relation, neighbor_type], …]}`. Reuse sophia's `build_signature`
  / `signature_similarity` directly (the experiment already runs in the sophia
  venv — `rescore.py` imports from sophia).
- **`represent.py`** — register `gloss` / `name_gloss` (read pre-generated
  glosses off a name→gloss map).
- **`run.py` / `rescore.py`** — include the new text arms in the embed/score
  loop; add the relations scoring path.
- **`RECORD.md`** — a new dated run entry with the results table + verdict.

## Decisions (approved)

- **A. Include `name_gloss`.** `name_sentence` beat bare `sentence`, so identity
  anchoring helps; the extra arm is near-free once glosses exist.
- **B. Generator = `gpt-4o-mini` direct.** Existing key/httpx path, ~$0.50 for
  ~5k glosses; the generator is not the variable under test. (Embedding model
  stays fixed at `text-embedding-3-large`/3072 — that invariant is the point.)
- **C. Defer the combined `gloss+relations` correspondence arm to v2.** Averaging
  the two distance matrices is the vault's "concept = cross-view correspondence"
  thesis and is the natural payoff — but we look at each pure facet alone first.
- **D. Relations = from-edges only** (mirror production `query_edges_from`).
  Add bidirectional edges only if from-edge coverage turns out poor.

## Scope / non-goals

- Offline, **no graph writes**, same `sample.json`, embedding model fixed at
  3-large/3072.
- **No wiring back into `proposal_builder.py`** — promoting a winning facet into
  the live pipeline is a separate follow-up, exactly as the prior runs were
  scoped.

## Success criteria

The experiment succeeds (as an experiment) if it produces, for `gloss`,
`name_gloss`, and `relations`: a best-cut silhouette and a `chunk_ratio`,
plus relations coverage, in a `RECORD.md` table comparable to the existing
arms — and a clear verdict on whether either facet reaches **high silhouette at
low chunk coupling**. A negative result that is correctly measured is a valid
outcome.

## Risks

- **Gloss leakage.** "Context-disambiguated" must not degrade into "paraphrase
  the sentence", which would silently re-import chunk text. Mitigation: prompt
  constraint + the `chunk_ratio` diagnostic catches it if it happens.
- **Sparse signatures.** Entities with one or zero from-edges yield degenerate
  signatures; surfaced via the coverage number, with bidirectional as fallback.
- **Single-mention corpus** (1.04 mentions/name) still applies — noted in RECORD;
  it bounds what any text facet can denoise.

## Follow-ups (out of scope here)

- v2: `gloss+relations` correspondence arm (combined distance).
- If a facet wins: re-embed across the lx39 model arms, then feed the verdict
  back into `proposal_builder.py` (what to embed) and the extractor.
