# Journal 001 — real-data clustering grid search

**Date:** 2026-06-03
**Run:** `run_real_grid_search.sh 3` (clean reset → 3-round ingest → capture → sweep)

## Hypothesis
The node-type emergence failure (production clusterer ~random ARI, ~10x
over-fragmentation) is a *clustering-config* problem, not an embedding/edge
problem. A grid search over algorithms/preproc/k-selection on REAL captured
embeddings should find a config that materially beats the baseline.

## Setup
- Clean slate: dropped all Milvus collections, FLUSHALL Redis, DETACH DELETE
  Neo4j; restarted Hermes + Sophia with `LOGOS_EMBEDDING_DIM=1536`.
- Fresh cold-start ingest (seed_n=1, 3 rounds) of the 6-domain corpus via the
  live Hermes→Redis→Sophia→HCG loop.
- Captured a static fixture from the live graph: **215 entities, 131 semantic
  edges, 6 ground-truth domains** (`sweep/fixture.json`).
- Full ML menu: scikit-learn + hdbscan + umap-learn installed in the sophia venv.

## Results
**Baseline (production silhouette auto-k):** node_type_ari = **0.039**,
203 members → **59 hierarchy roots** (~10x over-fragmentation), coverage 0.82.

**Best swept node configs (ARI vs domain):**
| algo | preproc | k_mode | n_cl | cover | ARI | purity |
|---|---|---|---|---|---|---|
| kmeans | raw | n_domains | 6 | 1.00 | **0.198** | 0.460 |
| agglomerative_avg | pca50 | n_domains | 6 | 1.00 | 0.197 | 0.423 |
| kmeans | pca50 | n_domains | 6 | 1.00 | 0.190 | 0.451 |
| hdbscan | pca50 | n_domains | 8 | 0.27 | 0.188 | **0.542** |
| agglomerative_ward | raw | n_domains | 6 | 1.00 | 0.179 | 0.437 |

- **Every** top config uses `k_mode=n_domains`. Algorithm barely matters.
- hdbscan: highest purity (0.54–0.62) but low coverage (0.27–0.31) — clean cores,
  ~70% left as noise.
- umap preproc did not reach the top.

**Edges:** agglomerative/kmeans reproduce relation labels (90 clusters,
merge_ratio 1.0 — tautological). Only **hdbscan** merges labels into edge-types
(merge_ratio 7.5, 12 clusters) at 0.726 purity.

## Diagnosis
Bottleneck confirmed = **cluster-count selection, not the algorithm**. Replacing
silhouette auto-k with controlled/target k lifts ARI **0.039 → 0.198 (~5x)**.
But the ceiling is only ~0.2: real NER-noisy linguistic embeddings do not cleanly
separate into domains under any config — supports the dual-embedding (JEPA +
relational) thesis.

## Next directions
1. Replace silhouette auto-k in `emergence_clustering` with a target-k / capped
   cluster-count strategy (or a density floor). Cheapest high-impact fix.
2. For emergent minting, prefer the **hdbscan precision route**: mint types from
   dense cores, leave residue as base `entity` — matches LOGOS bootstrap philosophy.
3. Test whether adding relational/edge structure (not just node embeddings) to
   the node-clustering features raises the ~0.2 ceiling.
