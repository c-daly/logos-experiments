# Embedding bake-off (lx39 — logos-experiments#39)

Does the HCG actually need `text-embedding-3-large` / 3072, or would a smaller
(or local, free) model do the job just as well? The job is **type
classification** — nodes are placed by embedding similarity — so that's the
metric, alongside the **vector geometry** that explains it.

This is sample-based and offline: it embeds a labeled node sample with each
candidate model and scores it. **No migration, no graph writes** — it informs
whether the all-or-nothing re-embed (every collection must share one dim) is
worth committing to. Switching models does *not* relax the single-dim
constraint; the only wins are storage (½ at 1536, less for ST), cost, and
speed.

## Run

```bash
NEO4J_PASSWORD=... uv run python sample.py    # build sample.json (once)
uv run python run.py                          # embed (cached) + score -> results.json
uv run pytest                                 # metric unit tests
```

OpenAI arms need `OPENAI_API_KEY` (exported); sentence-transformers arms run
locally on CPU (weights download on first run). Vectors cache to `.cache/`.

## Sample

3,879 nodes that are `IS_A` a `type_definition`, restricted to the 351 types
with ≥5 members (one type per node, lexicographic-first). Coarse kinds
(entity/concept/process) are excluded as too easy. `sample.json`.

## Models

| arm | dim | cost |
|-----|-----|------|
| openai-3-large | 3072 | API $0.13/Mtok (current) |
| openai-3-small | 1536 | API $0.02/Mtok |
| st-minilm-l6 (all-MiniLM-L6-v2) | 384 | local / free |
| st-mpnet-base (all-mpnet-base-v2) | 768 | local / free |
| st-bge-base (bge-base-en-v1.5) | 768 | local / free |

## Metrics

**Geometry** (describes the space): `eff_rank` (spectral-entropy effective
rank), `dims_95pct_var` (PCA dims for 95% variance — the decisive number for
"is 1536 enough?"), `anisotropy` (mean random-pair cosine; high = narrow cone).

**Downstream** (does it do the job): `knn_acc_raw` / `knn_acc_whitened`
(leave-one-out cosine kNN type accuracy, before/after PCA-whitening — whitening
often closes model gaps), `silhouette` (type-cluster separation). Metrics are
pure and unit-tested (`metrics.py`, `tests/`).

**Decision rule:** if a cheaper model's `knn_acc` is within ~1–2 points of
`3-large` (especially after whitening), the storage/cost win justifies the
migration — and `dims_95pct_var ≪ 3072` is the geometric reason it can.

## Results (2026-06-10, n=3879, 351 types)

| model | dim | eff_rank | dims@95%var | anisotropy | silhouette | kNN raw | kNN whitened | cost |
|-------|-----|----------|-------------|------------|------------|---------|--------------|------|
| **openai-3-large** | 3072 | 1772 | 1020 | 0.164 | −0.009 | **0.502** | 0.313 | API $0.13/Mtok |
| openai-3-small | 1536 | 1011 | 644 | 0.176 | −0.025 | 0.439 | 0.372 | API $0.02/Mtok |
| st-bge-base | 768 | 534 | 356 | 0.476 | −0.05 | 0.400 | 0.362 | local/free |
| st-mpnet-base | 768 | 519 | 322 | 0.105 | −0.05 | 0.397 | 0.388 | local/free |
| st-minilm-l6 | 384 | 311 | 239 | 0.111 | −0.055 | 0.380 | 0.361 | local/free |

(kNN type-accuracy on 351 fine types; weighted chance ≈ 0.05, so every model
clearly encodes type. Comparison is relative.)

## Verdict — `3-large` earns its size, but the reason is **model quality, not dimensionality**

1. **`3-large` is best by a clear margin**: kNN type-accuracy 0.502, vs 0.439
   for `3-small` (−6 pts) and ~0.40 for the best free model (−10 pts). The
   "mini ≈ large" hypothesis does **not** hold on this task.

2. **The decisive geometry finding flips the intuition about dim.** `3-large`'s
   variance fits in **1020 dims (95%) — below 1536**. So `3-small` has *more
   than enough* dimensional room, yet scores lower. **The 3072→1536 gap is not
   a capacity problem; `3-small` is simply a weaker model.** You can't recover
   it by handing a small model more dimensions — which also means the
   single-dim constraint isn't the lever here; a *better same-dim* model would
   be. (And it means 3-large could likely be Matryoshka-truncated to ~1024
   with little loss — a real storage win *without* changing model.)

3. **Free/local costs ~10 pts** (mpnet/bge @768 ≈ 0.40 vs 0.502). Real quality
   drop, but $0 + private + no rate limits + no single-vendor dependency.

4. **Naive whitening hurts the high-dim models** (`3-large` 0.502 → 0.313): full
   PCA-whitening over-amplifies the many low-variance noise directions. The raw
   numbers are the honest ones; a gentler ABTT pass is the follow-up.

**So:** the mini/local migration is **not** a free storage win — it trades
6–10 pts of type-classification accuracy for the saving. Worth it only if that
tolerance is acceptable or cost/privacy dominate. The cheapest *lossless*
storage win on the table is **Matryoshka-truncating `3-large` to ~1024**, which
the 95%-variance number says is nearly free — and keeps the strongest model.

### Caveats / follow-ups
- One downstream proxy (type-classification via kNN) on **node names** (short
  text). Retrieval and emergence-clustering uses may rank models differently;
  edge *phrases* (longer) are worth a second pass.
- `bge` wants a query-instruction prefix for retrieval; used plain symmetric
  encoding here — mild possible handicap.
- Whitening was naive full-PCA; ABTT (remove top-k only) may change the
  whitened column.
