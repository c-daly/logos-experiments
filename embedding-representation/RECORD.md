# embedding-representation — lab record

**Question.** *Why does corpus after corpus fail to surface the information it contains?*
The HCG ingests a corpus, extracts entities, embeds them, and types them by
embedding similarity. Retrieval and type formation are only as good as the
geometry of those entity vectors. This experiment isolates **what we embed**
(the input representation) from **which model** (lx39 / `embedding-eval` owns
that axis) and from **how we cluster** — and asks which representation makes
entities sit in an information-bearing geometry.

The metric is intrinsic and **label-free** by deliberate choice (see Decision
log): we do not score against a single "correct" type, because most live
entities have several legitimate types ("pneumococci" is a bacterium *and* an
organism *and* a pathogen *and* a cell). Instead we ask whether the vectors
have the internal and relative structure that information-bearing embeddings
should have.

---

## Root-cause finding (the headline)

`hermes/src/hermes/proposal_builder.py` embeds **bare entity names**:

```python
entity_names = [e["name"] for e in entities]
embed_texts = entity_names + [text]      # the document gets the passage; entities get only their name
```

The document/chunk is embedded with its full text. Each **entity** is embedded
with nothing but its surface string — `"cell wall"`, `"0.1 to 5.0 μm"`,
`"thiomargarita magnifica"`. No context, no disambiguation, no role in the
passage. This is **corpus-invariant**: it does not matter what you ingest, the
entity vectors are a function of the name alone. That is the mechanism behind
"corpus after corpus fails" — every corpus is reduced, at the entity layer, to
a bag of short out-of-context strings, and short-string embeddings of a large
general model live in a narrow, weakly-separated cone.

A second, compounding defect (Step 0): the extractor emits **non-entities** as
first-class nodes — measurements and fragments are embedded and typed
alongside real entities. Visible in the very first sample row: `0.1 to 5.0 μm`,
`1 cm` are live `entity` nodes. Noise in the input caps the coherence of any
representation.

---

## The pipeline is a chain of lossy stages

```
chunk → extract → select → represent → encode → cluster
```

Coherence is capped by the weakest stage. lx39 probes `encode` (model). This
experiment probes `represent` (and audits `extract`/`select` in Step 0). A
better model cannot recover information the representation never carried.

---

## Coherence battery (label-free)

Scored on the (N, D) matrix of unit-normalized entity vectors.

**Internal (is the space using its capacity?)**
- `anisotropy_centroid` — mean cosine of each vector to the global (normalized)
  mean. High ⇒ narrow cone ⇒ poor discrimination. **Lower is better.**
- `anisotropy_pairs` — mean cosine of random i≠j pairs (lx39 definition).
- `effective_rank` — `exp(spectral entropy)` of the singular spectrum; how many
  dimensions actually carry variance. **Higher is better.**
- `participation_ratio` — `(Σλ)² / Σλ²`; second estimate of dims-in-use.
- `intrinsic_dim_twonn` — TwoNN estimator `N / Σ log(r₂/r₁)`; the manifold's
  intrinsic dimensionality. Pathologically low ⇒ collapsed.

**Relative (do neighbourhoods mean anything?)**
- `pairwise_cos_mean` / `pairwise_cos_spread` (std) — how spread out the cloud
  is. **More spread is better** (within reason).
- `nn_margin` — median over points of (cos-dist to 2nd NN)/(cos-dist to 1st
  NN). >1; crisp neighbourhoods score higher. **Higher is better.**
- `hopkins` — clustering tendency; 0.5 = uniform/random, →1 = clusterable.

A **whitening control** (PCA-whiten, then re-score the relative metrics) tells
us whether weak relative structure is genuinely absent or merely masked by the
anisotropic cone.

---

## Baseline — name embeddings (live HCG, openai-3-large / 3072)

Measured on the live `hcg_entity_embeddings` collection (entity name vectors as
they are stored today). This is the bar every richer representation must beat.

| metric                | name baseline | direction |
|-----------------------|---------------|-----------|
| anisotropy_centroid   | **0.405**     | lower better |
| nn_margin             | **1.088**     | higher better |
| pairwise_cos_spread   | **0.067**     | higher better |
| effective_rank        | ~452 / 3072   | higher better |
| intrinsic_dim_twonn   | ~2.1          | higher better |

Reading: vectors sit in a tight cone (cos 0.405 to the mean), neighbourhoods
are nearly flat (2nd neighbour only ~9% farther than 1st), and the cloud is
narrow. An intrinsic dim ~2 on a 3072-dim model means the name vectors live on
an almost trivial manifold — consistent with "short strings of a big model".

### Whitening control — ruled out

PCA-whitening collapses `anisotropy_centroid` 0.405 → ~0.003 (the cone is real
and removable) but `nn_margin` stays flat 1.088 → ~1.091. **Removing the cone
does not create neighbourhood structure that was not there.** The weak geometry
is not a cosmetic cone hiding good clusters — the information is not in the
representation to begin with. → The input representation is the real lever, not
a post-hoc transform.

---

## Experiment design

**Step 0 — extraction / selection audit (parallel).** Quantify what fraction of
live `entity` nodes are non-entities (measurements like `0.1 to 5.0 μm`,
fragments, stopword spans). Caps achievable coherence regardless of
representation. Output: a noise rate + a cleaned-vs-raw battery delta.

**The chunk-collapse confound (why a blind window is wrong).** Measured on the
live sample: **15.5 entities are extracted per `raw_text` chunk** (median 15,
max 37), and **100% of entities share their chunk** with others; `raw_text` is
~1084 chars. A blind ±char window around a mention is therefore ~1/3 shared
passage text, so co-occurring entities collapse onto near-identical vectors —
you measure *chunk* geometry, not *entity* geometry. So context is scoped to the
**sentence** containing the mention, and we add a label-free **chunk-collapse
diagnostic** (`nn_chunk_rate`: fraction of an entity's k-NN that are its own
chunk-mates, vs the random-neighbour rate). High ratio = false coherence.

**Step 1 — input representation.** Same model (openai-3-large, `name` pulled
free from the live `hcg_entity_embeddings` so it *is* the production geometry),
same 5330-entity sample, four input strings per entity:
- `name` — bare surface string (= the live baseline).
- `sentence` — the single sentence containing the mention (name not privileged).
- `name_sentence` — `"{name} — {sentence}"` (identity anchored).
- `marked` — the sentence with the mention wrapped: `… «name» …`.

Score the full battery + whitening control + chunk-collapse on each arm.

**Step 2 — model axis (later, composes with lx39).** Re-embed the winning
representation with local/HF models via harness/ollama to test whether a smaller
or domain model on a *good* representation beats the big model on a *bad* one.

---

## Run 1 — name / sentence / name_sentence / marked (2026-06-15)

`results.json`. Battery (raw), all openai-3-large / 3072, N=5330:

| arm           | anisotropy↓ | nn_margin↑ | spread↑ | eff_rank↑ | idim↑ | hopkins | **chunk_ratio** |
|---------------|-------------|------------|---------|-----------|-------|---------|-----------------|
| name          | 0.406       | 1.107      | 0.067   | 1867      | 1.38  | 0.619   | **37×**         |
| sentence      | 0.331       | 1.000      | 0.083   | 1214      | 0.11  | 0.898   | **177×**        |
| name_sentence | 0.369       | 1.253      | 0.079   | 1617      | 5.08  | 0.713   | **181×**        |
| marked        | 0.354       | 1.276      | 0.083   | 1518      | 3.23  | 0.765   | **187×**        |

`name` reproduces the recorded live baseline (anisotropy 0.406≈0.405, spread
0.067, nn_margin 1.107≈1.088) — the harness is validated.

**The result is negative, and the chunk-collapse diagnostic is why.**
- `name` neighbourhoods are 12% chunk-mates (37× random) — the legitimate
  *topical* floor (entities from one passage are genuinely related).
- Every **context** arm jumps to **~58% chunk-mates (~180×)**: more than half of
  each entity's nearest neighbours are just other entities extracted from the
  same passage. The passage text dominates the vector.
- So the nn_margin / intrinsic-dim *gains* of `name_sentence`/`marked` over
  `name` are **false coherence** — the vectors cluster more tightly, but they
  cluster **by source chunk, not by type**.
- `sentence`-only is degenerate: 2063 unique vectors for 5330 entities
  (co-sentence entities collapse to identical points), nn_margin pinned at
  1.000, intrinsic-dim 0.11.
- Whitening: context arms retain marginally more post-whiten margin than `name`
  (1.08–1.10 vs 1.03), so there *is* a little real relative structure — but it
  is swamped by the chunk artefact.

**Verdict:** naive context injection swaps a name-cone for chunk-blobs. The
target — crisp entity neighbourhoods with *low* chunk dependence (high nn_margin
**and** low chunk_ratio) — is reached by no arm. Context-richness and
chunk-independence are in tension; the next step must **decouple entity signal
from passage signal**.

---

## Run 2 — decoupling entity signal from chunk signal (2026-06-15)

Cross-mention averaging is a **no-op on this corpus**: 5131 distinct names for
5330 entities (1.04 mentions/name, median 1); only 4% of names span ≥2 chunks.
Each entity is mentioned once — there is no cross-context evidence to average the
chunk noise away. So we removed the chunk component *directly* from the
`name_sentence` vectors. `results_decouple.json`:

| arm            | anisotropy↓ | nn_margin↑ | spread↑ | eff_rank↑ | idim↑ | chunk_ratio↓ |
|----------------|-------------|------------|---------|-----------|-------|--------------|
| name_sentence  | 0.369       | 1.253      | 0.079   | 1617      | 5.08  | 181×         |
| chunk_centered | 0.008       | 1.227      | 0.053   | 1780      | 5.26  | **104×**     |
| chunk_residual | 0.299       | 1.236      | 0.055   | 1813      | 5.18  | **141×**     |

**There is real entity signal under the chunk** — removing the chunk mean keeps
nn_margin (1.25→1.23) and intrinsic-dim (~5.1) intact, even raises eff_rank. But
it only **partially** decouples: chunk_ratio falls 181× → 104×, still ~3× the
name floor (37×). We can subtract the chunk *centroid*, but each chunk's entities
share vocabulary/topic *beyond* the mean, and that residual does not subtract
geometrically.

## Step 0 — extraction audit (2026-06-15)

Tested two distinct claims (genericity is NOT noise — `gene`/`energy`/`cell` are
legitimate concepts; only measurements/fragments are errors). `audit.json`:

- **Genuine error rate is small and irrelevant to the geometry.** Errors
  (measurement 3.7%, year/number 2.7%, has_digit 5.3%, too_short 0.9%, too_long
  0.5%) total **9.9%**. Removing them moves the battery by *nothing*: anisotropy
  0.406→0.410, nn_margin 1.110→1.111. Extraction quality is **not** the
  bottleneck. `confidence` is 0.70 everywhere — does not discriminate.
- **The "generic concepts need context most" hypothesis is false.** Common-noun
  concepts (4013) vs proper-named entities (802) have ~identical bare-name
  geometry: anisotropy 0.417 vs 0.424, nn_margin 1.114 vs 1.094. The weak cone /
  flat margins are **uniform across every entity kind** — an intrinsic property
  of bare-name embedding, not of which entities were selected.

| group        | n    | anisotropy | nn_margin | eff_rank | idim |
|--------------|------|------------|-----------|----------|------|
| raw          | 5344 | 0.406      | 1.110     | 1868     | 1.47 |
| minus_errors | 4815 | 0.410      | 1.111     | 1826     | 1.37 |
| common_noun  | 4013 | 0.417      | 1.114     | 1747     | 1.29 |
| proper       | 802  | 0.424      | 1.094     | 609      | 1.52 |

**What the audit rules out:** the corpus, extraction noise, and genericity are
*not* the cause. The weakness is uniform and structural → it is the
representation, confirmed from a third independent angle.

## Positive control — do bare-name embeddings work at all? (2026-06-15)

Reductio (Chris): we ruled out representation and chunk size; either embeddings
are a scam or something else is wrong. Test: push 80 known-category bare names
(animals/countries/math/fruits/emotions/metals/instruments/weather) through the
SAME path (text-embedding-3-large/3072). `control.json`:

| signal                       | control | reading |
|------------------------------|---------|---------|
| within-cat cosine            | 0.423   |         |
| between-cat cosine           | 0.221   |         |
| cosine silhouette            | 0.213   | real clusters (>0.1) |
| **5-NN category purity**     | **0.973** | bare names cluster at 97% |

**Embeddings work.** Bare-name embedding separates known concepts at 97% kNN
purity. The input representation was never the bottleneck.

**The intrinsic battery is non-diagnostic** — the load-bearing negative:

| metric              | control (clusters!) | our entities | 
|---------------------|---------------------|--------------|
| anisotropy_centroid | 0.504               | 0.406        |
| nn_margin           | 1.062               | 1.110        |
| hopkins             | 0.56                | 0.619        |
| intrinsic_dim_twonn | 26.68               | 1.47         |

On anisotropy / nn_margin / hopkins the **known-clustered control scores WORSE
than our entities.** These label-free metrics do **not** measure clusterability
— clustering is relative to a grouping and needs labels (silhouette / kNN
purity). nn_margin is actively backwards: dense clusters *lower* it (1st & 2nd
neighbours are both in-cluster). **Every "weak geometry → representation" claim
in Runs 1–2 and the audit was measuring diversity, not dysfunction.** The only
metric that tracked anything (intrinsic_dim 1.47 vs 26.68) measures spread, not
coherence.

→ Embeddings are fine; the measurement was broken; **the problem is the process
that consumes the embeddings (typing / retrieval), not the vectors.**

## THE ANSWER — retrieval is RAG with the passages removed (2026-06-15)

Chris: the answer is "somewhere else, something not obvious." It is. After ruling
out the entire index/storage side (embeddings work, relations dense + 49%
cross-chunk + one 3596-node component, type hints 100% coverage), the unexamined
subsystem was RETRIEVAL — literally "find information." Traced it:

- There is NO retrieval/QA endpoint. The only "find information" path is
  `hermes._get_sophia_context` → a per-CONVERSATION Redis cache
  (`ContextCache.get_context(conversation_id)`), filled as a side-effect of
  INGESTING turns. *"We never call Sophia synchronously to obtain context."*
- The actual graph search is in `proposal_processor.process` (#1): it runs
  `milvus.search_similar(query_embedding=doc_emb["embedding"])` over
  `SEARCHABLE_COLLECTIONS = (Entity, Concept, State, Process)` — the **bare
  entity-NAME** vector collections — and returns nodes as **name+type+props**,
  with `raw_text` explicitly stripped in `_build_context_message`.
- It does NOT search `Edge` (relations) and NOT `hermes_embeddings` (the 352k
  **passage** vectors that hold the information).

**The flaw:** standard RAG embeds passages and matches query→passage. This
pipeline embeds the query as a **document** vector and matches it against **bare
entity-name** vectors (incompatible regions of the space — a question does not
land near a 2-token name), then returns bare labels. The passages and relations
— the only information-bearing layers — are never queried. So a perfect,
connected, correctly-typed graph still "fails to find information," on every
corpus, because the query path never touches where the information lives.

**This is the real face of the initial bare-name embedding problem:** name
vectors were never a representation you could query with a document embedding.
The damage isn't mainly in typing — it's that retrieval is structurally
incapable regardless of graph quality. Fix direction: query the passage store
(hermes_embeddings) query→passage, search relations, and return raw_text /
propositions — not bare node labels.

## Rescore — representations on the RIGHT metric (2026-06-15)

Runs 1–2 scored representations with the intrinsic battery, which the control
proved non-diagnostic. Re-scored every cached arm with best-cut silhouette (the
metric that separated control 0.213 from junk 0.06) + chunk_ratio. `rescore.json`:

| arm            | best_k | silhouette | chunk_ratio |
|----------------|--------|------------|-------------|
| (control)      |        | 0.213      |             |
| (junk name)    |        | 0.060      |             |
| name           | 266    | 0.062      | 37×         |
| sentence       | 265    | **0.181**  | 177×        |
| name_sentence  | 264    | 0.132      | 181×        |
| marked         | 265    | 0.152      | 187×        |
| chunk_centered | 2      | 0.224      | 104×        |
| chunk_residual | 2      | 0.119      | 141×        |

**Overturns the "representation is a modest lever" finding** (that was the broken
battery): on the typing-relevant metric, context **triples** cluster
separability (name 0.062 → sentence 0.181, control territory). Representation is
THE lever — we just couldn't see it.

**But the lift is entirely chunk-coupled** (chunk_ratio 177–187×), and decoupling
destroys the resolution: `chunk_centered` hits the highest silhouette (0.224) but
at **best_k=2** — one coarse split, useless for fine typing.

**The non-redundant gap:** every arm tried EMBEDS THE PASSAGE (sentence/
name_sentence/marked all contain chunk text → chunk-coupled). Untried category:
semantic resolution WITHOUT passage text — a short generated gloss, the stored
`hermes_type_hint`, or the entity's relations (structural signature). That is the
one axis that could add resolution while staying chunk-independent.

## Typing / emergence — where the embeddings problem actually bites (2026-06-15)

Chris: "this goes back to the initial embeddings problem." Traced the typing
process (`emergence_handler.py` + `emergence_clustering.py`): load members
(embeddings) → `find_emergent_clusters` (PROPOSAL only, samples to 800/pass) →
an LLM (`hermes /type-cluster`) NAMES each cluster + parent → mint/attach.
*"Embeddings only propose; centroids never decide; the LLM-named parent asserts
placement."* Config: variance_threshold 0.6, min_cluster_size 3, max_cluster_size
50, hermes_confidence_floor 0.5.

**Live placement state:** of ~5400 entity→type IS_A edges, **4053 (75%) are
still parked unplaced under the `entity` junk-drawer root.** Only ~1340 drained
into 45 small types (structure 91, chemical compound 55, protein 14, …).

**Why (measured on the 4053 unplaced):**
- variance 0.835 (> 0.6) → pre-filter passes; not a gating bug.
- `find_emergent_clusters` returns 97 clusters → not "zero clusters."
- **best-cut silhouette 0.060 vs the clean control's 0.213** — bare-name vectors
  barely separate the fine-grained within-domain entity mix. Diffuse clusters →
  the LLM namer can't confidently label them (sub-0.5 confidence discarded) →
  members stay in the pool → 75% never drains.

**Connection confirmed:** typing is starved by the bare-name representation's
lack of *fine-grained* resolution. Embeddings separate DISTINCT things (control
0.97 purity / 0.213 silhouette) but not a within-domain CONTINUUM
(protein/enzyme/biopolymer), which is what emergence needs. Caveat: the
junk-drawer is survivorship-biased (easy clusters already drained), so 0.06 is
partly "the hard residue" — but that residue IS the fine continuum.

**The crux everything converges on:** fine-grained AND chunk-independent entity
embeddings. Context adds resolution but couples to chunk (Run 1); cross-mention
averaging would decouple but the corpus is single-mention (Run 2). That tension,
on a repetitive corpus, is the open problem.

## Synthesis (superseded — see control above)

- **Representation matters, but at the margin and with diminishing returns.**
  Context lifts neighbourhood quality over bare names (nn_margin 1.11→1.25,
  intrinsic-dim 1.4→5.1) and the lift survives chunk-mean removal — so it is not
  *all* artefact. But context is heavily chunk-entangled and only partly
  decouplable. Best achievable so far: margin ~1.25 at chunk_ratio ~104×, vs
  name's margin 1.11 at chunk_ratio 37×. A real but modest improvement, traded
  against residual chunk dependence.
- **The data now points at extraction/selection (Step 0) as the dominant lever.**
  The recurring "entities" are generic common nouns (`sun`, `gene`, `distance`,
  `energy`, `cell`, `earth`, `country`, `metal`) and measurements (`0.1 to 5.0
  μm`, `1 cm`). Generic abstractions and measurements **cannot** cluster into
  meaningful types no matter how they are embedded. No representation fixes a bad
  entity set.
- **The single-mention corpus** (1.04 mentions/name) removes the most powerful
  denoiser (cross-context aggregation). Coherence work on richer/repetitive
  corpora may behave very differently.

---

## Decision log

- **Label-free over single-label eval.** Rejected scoring entities against one
  "correct" type — too many legitimate types per entity for a single label to
  be meaningful. Chose the intrinsic battery: information-bearing embeddings
  have characteristic internal + relative structure, measurable without labels.
- **Whitening ruled out as the fix.** Cone is removable but masks no structure
  (nn_margin flat). Representation, not transform, is the lever.
- **Reuse what's already embedded.** `name` is the live production geometry,
  pulled free from `hcg_entity_embeddings` by uuid (no re-embed). The
  `hermes_embeddings` cache (352k strings, all `text-embedding-3-large`) holds
  every name + chunk text ever embedded, queryable by `text` — but its expr
  parser chokes on long/quoted passages, so reverse-lookup is unreliable for
  novel strings; only the new arms hit the API (cached to `.cache/`).
- **Chunk-collapse is the load-bearing diagnostic.** Run 1 shows context arms
  cluster by source chunk (~58% chunk-mate neighbours), not by type. Any future
  representation must be judged on **nn_margin AND chunk_ratio jointly** — a
  high margin with a high chunk_ratio is false coherence.
- **SVD perf.** One thin SVD (with U) per arm, shared across the three spectral
  metrics + whitening (`battery.score`); cap BLAS threads (`OMP_NUM_THREADS=8`)
  to avoid WSL thread-thrash on the 5330×3072 SVD.
- **Offline, no graph writes.** `raw_text`+`start`+`end` on every entity node;
  representations are reconstructable without re-ingest.

## Next steps

The tension to break: **decouple entity signal from passage signal** — high
nn_margin with *low* chunk_ratio.

- [ ] **cross-mention arm.** Many entity names recur across chunks. Represent an
      entity as the **mean of its contextualised mentions across all its
      chunks**; chunk-specific text should average out, leaving the
      type-consistent signal. Direct attack on the confound — expect chunk_ratio
      to fall while margin holds. (Needs name→mentions grouping.)
- [ ] **chunk-residual arm.** Subtract the passage vector: `name_sentence_vec −
      raw_text_vec` (chunk vectors are available / embeddable). Explicitly
      projects out the shared-chunk component.
- [ ] **Step 0 audit.** Noise rate of live entities (measurements/fragments like
      `0.1 to 5.0 μm`, `1 cm`); cleaned-vs-raw battery delta.
- [ ] If a decoupled arm wins (margin↑, chunk_ratio→name's floor): re-embed it
      across the lx39 model arms (Step 2), then feed the verdict back into
      `proposal_builder.py` (what to embed) and the extractor (what to select).

---

## Session close (2026-06-15) — the real conclusion + reframe

The "Next steps" above are SUPERSEDED. The investigation continued (audit,
positive control, retrieval trace) and the conclusion moved. In order:

1. **Embeddings work; the battery was the broken instrument.** Positive control:
   bare names cluster known categories at 97% kNN purity, and the known-clustered
   control scores *worse* than our entities on the intrinsic battery. The
   label-free battery does not measure clusterability — that needs labels. Most
   of the "weak geometry" framing in Runs 1–2 was measuring diversity, not
   dysfunction.
2. **The real bottleneck is RETRIEVAL, not the index.** Index side is sound
   (relations dense + 49% cross-chunk + one 3596-node component; type hints 100%).
   "Find information" matches a *document*-embedding query against *bare entity-name*
   vectors and returns bare labels — RAG with the passages amputated (see "THE
   ANSWER" section). Corpus is never consulted where its information lives.
3. **But that's the text-only special case of a larger design** (vision thread,
   captured in the vault `2026-06-15-hcg-*` memories): the graph is a
   non-linguistic concept substrate; a node is the convergence of many views
   (modal × temporal × experiential); the name was forced to *be* the concept,
   which is the original sin. With ≥2 channels, correspondence grounds identity
   and the lone weak embedding becomes one channel among many.

**Actionable conclusion for text representation** (what Chris asked help with):
stop hunting for THE entity vector. Represent text by its FACETS, each built to
slot into correspondence later:
- **handle** — the name; a lexical handle, *not* the identity (demote it).
- **sense** — a short, *chunk-independent* gloss/definition (the one arm never
  tried; adds resolution without dragging the chunk in).
- **relations** — what the text asserts; the lingua franca + structural identity
  (revive the shelved `structural_signature` — right, just early). Highest leverage.
- **mentions** — the episodic where/when log.

Two concrete first moves when we resume: (1) chunk-independent **sense** embedding
(small experiment, same harness); (2) revive **structural_signature** as a
first-class facet (highest leverage). Full reasoning in the vault memories.

## Run 3 — sense (gloss) facet; structure deferred (2026-06-16)

`rescore.json`. One chunk-independent LLM definition per unique entity name
(gpt-4o-mini, temp 0, 100% coverage of the 5330-entity sample), embedded with
text-embedding-3-large/3072. Scored on best-cut silhouette + chunk_ratio.

| arm | best_k | silhouette | chunk_ratio | |
|-----|--------|------------|-------------|--|
| name (baseline) | 266 | 0.062 | 37× | |
| sentence | 265 | 0.181 | 177× | chunk-coupled (false) |
| name_sentence | 264 | 0.132 | 181× | chunk-coupled (false) |
| marked | 265 | 0.152 | 187× | chunk-coupled (false) |
| chunk_centered | 2 | 0.224 | 104× | one coarse 2-way split |
| **gloss** | 265 | **0.070** | **47×** | |
| **name_gloss** | 265 | **0.069** | **46×** | |
| (control ceiling) | | 0.213 | | clean external categories |

**Verdict — the hypothesis split: chunk-independence held, separability did not.**

1. **Chunk-clean (the good half).** gloss / name_gloss chunk_ratio ~46–47× ≈ the
   name floor (37×), NOT the 177–187× of the passage arms. A generated definition
   encodes entity-level *sense* without importing chunk geometry — the property no
   prior context arm achieved.
2. **No lift (the disappointing half).** silhouette 0.070 ≈ name 0.062, far below
   the 0.213 control. The disambiguated sense added essentially nothing to cluster
   separability.

**Metric caveat.** On this within-domain continuum the only high silhouettes are
chunk artefacts (sentence) or a coarse k=2 split (chunk_centered); the 0.213
control was on artificially distinct categories. So best-cut silhouette may be near
its floor for any fine-grained representation here — we cannot fully separate
"gloss doesn't help" from "the metric can't see it." No positive signal either way.

**Structure (relations) arm — built but DEFERRED.** `signatures.py` + the
weighted-Jaccard scoring helpers are implemented and unit-tested, and the signature
pull is at 100% coverage (mean 2.2 pairs/entity, IS_A-dominated). But scoring
relations on the *current* graph is circular: neighbour-types come from emergence
typing and node identity from entity-resolution — both downstream of the very
bare-name representation under test. The un-confounded test requires a graph
re-ingested with a better representation, then relations measured on that. Deferred
to that spike.

**Bottom line.** A richer single text channel does not manufacture crisp type
structure from a within-domain continuum. Consistent with the 2026-06-15
conclusion that the bottleneck is retrieval (query→passage) and the path forward is
multi-view correspondence — not more single-channel representation arms.
