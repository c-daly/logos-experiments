# structural-edge-prediction

**Question.** On a knowledge graph, does a type-conditioned edge *signature*
predict held-out edges better than (A0) a type-blind marginal baseline and (A2)
embedding-cosine kNN \u2014 and does per-type prediction quality track signature
sharpness ("a kind is whatever predicts")?

**Thesis context.** *Embeddings POINT, the graph ASSERTS.* A type is the
reified IS_A edge; its *signature* is the distribution of relational edges its
members tend to have. If that signature ranks true missing edges above
corrupted fakes \u2014 with no LLM, no training \u2014 structure carries predictive
meaning. Held-out-edge recovery is simultaneously the GENERATION test (the
partial graph proposing the missing edges) and the ERROR-CATCHING test
(ranking fakes low). Same gear, one AUC.

---

## Snapshot schema

The toy fixture and the live exporter both emit this exact JSON:

```json
{
  "nodes": [{"id": "narwhal#1", "type": "whale", "label": "narwhal"}],
  "type_parents": {"whale": "animal", "fish": "animal", "animal": null},
  "edges": [{"src": "narwhal#1", "rel": "has", "dst": "blowhole#1"}],
  "embeddings": {"narwhal#1": [0.1, 0.2, "..."]}
}
```

- `nodes[].type` \u2014 the node\u0027s immediate type id (its IS_A parent type). May be `null`.
- `type_parents` \u2014 the type hierarchy (type id -> parent type id or `null`).
  Used to POOL members up the IS_A chain when building a type\u0027s signature.
- `edges` \u2014 the RELATIONAL edges among entity nodes (`has`, `lives_in`, ...).
  These are what we hold out and predict. The IS_A backbone (`nodes[].type` +
  `type_parents`) is NEVER held out \u2014 a node\u0027s type is always known.
- `embeddings` \u2014 optional; per-node vectors for the A2 arm. If absent, A2 is
  skipped (logged).

---

## Arms

Each scorer takes a candidate edge `(src, rel, dst)` + the training graph and
returns a real-valued plausibility score (higher = more plausible).

| Arm | Scorer | What it knows |
|-----|--------|---------------|
| **A0** marginal | `score_marginal` | global frequency of `(rel, type(dst))` over all train edges, ignoring `type(src)`. The null. |
| **A1** signature | `score_signature` | `P(rel, type(dst) | type(src))` from the most-specific supported type signature; backs off up the IS_A chain. The hypothesis. |
| **A2** embedding kNN | `score_embedding_knn` | fraction of `src`\u0027s cosine-nearest embedded neighbours carrying a matching edge. "Embeddings point." Skipped if no embeddings. |
| **A3** structural kNN (optional) | `score_structural_knn` | fraction of `src`\u0027s Jaccard-nearest (by edge-set) neighbours carrying the pattern. The analogy form. |

The **signature** of a type T is `P(rel, dst_type | src_type == T)` over train
edges whose source is a member of T (members pooled up the IS_A chain). Backoff:
a node carries a signature at every level of its chain \u2014 the most-specific type
with support wins, and the scorer climbs to a supertype when a pattern is unseen
(discounted by 0.5 per level, so a specific hit always outranks a backed-off one).
Sharpness = `-entropy` of the signature distribution (lower entropy = sharper).

---

## Metrics

`eval/metrics.py` prints one `[METRIC] key=value` line per metric (filtered
protocol \u2014 corrupted negatives that already exist in the graph are excluded):

- per arm: `AUC` (P a true edge outranks a random negative), `hits@1`,
  `hits@10`, `MRR`.
- `signature_auc_minus_marginal` (A1_AUC - A0_AUC).
- `signature_minus_embedding_auc` (A1_AUC - A2_AUC), if A2 ran.
- per-type `hits@10` and `sharpness`, then `sharpness_accuracy_corr` =
  Spearman correlation across types between signature sharpness and per-type
  hits@10.

AUC, MRR, hits@k, Shannon entropy, and Spearman are all plain-stdlib helpers
(no numpy/scipy).

---

## Decision rules

Every outcome is informative \u2014 that is the whole point.

| Outcome | Reading | Action |
|---------|---------|--------|
| **A1 \u226b A0** | structure predicts beyond base rates; the mechanism has signal | build the predict-surprise loop |
| **A1 \u2248 A0** | at this density types don\u0027t predict | re-run on a clean public KG (WN18RR) to disambiguate method-failure from density-failure; if A1 wins there but not here, the blocker is graph density/noise (a numbered finding) |
| **A1 > A2** | structure beats embedding proximity | quantified evidence for "embeddings point, graph asserts"; node2vec can wait |
| **A2 > A1** | embeddings already carry the structure | graph not adding predictive value yet (likely density) |
| **sharpness\u2194accuracy correlation present** | "a kind is whatever predicts" is supported | sharpness \u00d7 predictive-lift becomes a label-free type-quality gate for mint/evict |

---

## How to run

Offline toy positive control (zero external services):

```bash
uv sync --group dev
uv run python -m harness.run_experiment --snapshot fixtures/toy_graph.json
uv run python -m eval.metrics workspace/run.json
uv run pytest
```

Real 350-block graph \u2014 export the live HCG first (the only step that touches
Neo4j; gated behind `FREEZE_LIVE=1`), then feed the frozen snapshot to the same
offline runner:

```bash
FREEZE_LIVE=1 NEO4J_PASSWORD=... \\
    uv run python -m harness.freeze_snapshot --n 350
uv run python -m harness.run_experiment --snapshot fixtures/graph_350.json
uv run python -m eval.metrics workspace/run.json
```

Everything except `harness/freeze_snapshot.py` runs with no live stack. The run
is deterministic given `--seed`.

---

## The toy fixture is a positive control

`fixtures/toy_graph.json` has deliberately sharp types: whales
(`has\u2192blowhole`, `lives_in\u2192ocean`, `attr\u2192warm_blooded`) vs fish
(`has\u2192gills`, `lives_in\u2192ocean`, `attr\u2192cold_blooded`). The control
property: the whale signature ranks `narwhal has\u2192blowhole` ABOVE the
corruption `narwhal has\u2192gills`, while the marginal \u2014 blowhole and gills are
equally frequent globally \u2014 is near chance. The tests assert exactly this, and
the end-to-end test asserts `A1_AUC > A0_AUC` on the control.
