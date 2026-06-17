# edge-embeddings-worth-it

**Question:** Is embedding reified edges worth it? Does clustering edge-nodes
produce edge-types that carve the relation space better than leaving everything
in the `related` junk-drawer?

This experiment grows a multi-domain knowledge graph across several ingestion
rounds against the **live LOGOS stack**, then runs emergence on BOTH:

- the **entity residue** -> node-types (does the graph rediscover the corpus
  domains?), and
- the **reified edge nodes** -> edge-types (do edges cluster into coherent
  relation families, or just noise?).

The edge-clustering coherence is the "worth it" signal.

## Layout

```
edge-embeddings-worth-it/
  goal.yaml                 success criteria + objective
  corpus/corpus.jsonl       ~6 domains x ~17 lines, staged into 4 rounds
  harness/run_experiment.py the runner (file-based; no inline shell ingest)
  eval/metrics.py           ARI / coverage / edge-types / coherence -> [METRIC] lines
  workspace/round_<r>.json  per-round snapshots (written by the harness)
```

## Corpus

Six single-domain domains so entities are unambiguous:
`animals`, `lab_instruments`, `math_physics` (real symbols inline: \u2207 \u222b \u2211 \u2202 \u03c0 \u03b1 \u2264 m/z H\u2082O \u03c8 ...),
`vehicles`, `plants`, `celestial_bodies`.

Rounds grow the graph:
- **round 1** = animals + lab_instruments
- **round 2** += math_physics + vehicles
- **round 3** += plants + celestial_bodies
- **round 4** = relational-density (links already-introduced entities, sometimes
  cross-domain, to fatten the edge population)

## Bootstrap is a parameter (`--seed-n`)

- **N=1** (cold start): seed only `{entity}` (node junk-drawer) and a
  `related` / `RELATED_TO` edge-type-definition under `edge_type`. No centroids.
- **N>1**: additionally seed the first `N-1` foundational types from a fixed
  ladder with **name-derived centroids** (embed the type name via
  Hermes `/embed_text`, then `update_centroid`).

## Run

Must run inside the **sophia poetry env** (so `sophia.maintenance.emergence_*`,
`logos_hcg`, `pymilvus`, `neo4j` import). The harness **resets the live DB** on
every run (intended).

```bash
cd /home/fearsidhe/projects/logos-workspace/sophia

# Full experiment (4 growing rounds, cold-start bootstrap):
poetry run python \
  /home/fearsidhe/projects/vault/10-projects/LOGOS/experiments/edge-embeddings-worth-it/harness/run_experiment.py \
  --seed-n 1 --rounds 4

# Smoke test (~6 round-1 lines, single clustering pass):
poetry run python \
  /home/fearsidhe/projects/vault/10-projects/LOGOS/experiments/edge-embeddings-worth-it/harness/run_experiment.py \
  --seed-n 1 --smoke

# Larger bootstrap:
poetry run python .../harness/run_experiment.py --seed-n 4 --rounds 4
```

### Stack (env-overridable defaults)

| service | default |
|---------|---------|
| Neo4j   | `bolt://localhost:7687` (neo4j / logosdev) |
| Milvus  | `localhost:19530` |
| Hermes  | `http://localhost:17000` |

## Metrics

The harness calls `eval/metrics.py` on the final snapshot and prints
`[METRIC] key=value` lines:

| metric | meaning | threshold |
|--------|---------|-----------|
| `node_type_ari` | ARI of emergent node clusters vs. provenance domains | >= 0.8 (primary) |
| `classification_coverage` | fraction of entity-kind nodes that joined a cluster | >= 0.7 |
| `edge_types_formed` | # edge clusters of size >= min_cluster_size | >= 1 |
| `edge_cluster_coherence` | mean within-cluster relation purity | >= 0.7 |

You can also score a snapshot directly:

```bash
python eval/metrics.py workspace/round_4.json
```

## Results land in

`workspace/round_<r>.json` -- one snapshot per round, each containing node
clusters, the node hierarchy, edge clusters (rendered `src --REL--> tgt`), the
`(source-type -> target-type)` endpoint view, counts, and the
`entity_uuid -> domain` provenance map used as ARI ground truth.
