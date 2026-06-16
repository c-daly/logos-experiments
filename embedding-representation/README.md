# embedding-representation

**Which input representation makes HCG entities cluster coherently?** The HCG
embeds entities by **bare name** (`hermes/proposal_builder.py`:
`embed_texts = entity_names + [text]`), which is corpus-invariant and is the
suspected reason "corpus after corpus fails to surface information". This
experiment holds the model fixed (`text-embedding-3-large` / 3072) and varies
*what we embed*, scoring each arm with a **label-free** coherence battery.

See **RECORD.md** for the full lab notebook (root cause, baseline, decisions).

## Arms

| arm          | embedded string |
|--------------|-----------------|
| name         | bare surface string (reproduces the live baseline) |
| context      | ±160-char window of `raw_text` around the mention |
| name_context | `"{name} — {context}"` |

## Battery (label-free)

Internal: `anisotropy_centroid` (↓), `effective_rank` (↑), `participation_ratio`
(↑), `intrinsic_dim_twonn` (↑). Relative: `pairwise_cos_spread` (↑), `nn_margin`
(↑), `hopkins` (→1). Plus a **whitened** control (cone removal) per arm.

## Run

```bash
NEO4J_PASSWORD=logosdev python sample.py   # build sample.json (once)
OPENAI_API_KEY=... python run.py           # embed (cached) + score -> results.json
```

Offline and **no graph writes** — `raw_text`/`start`/`end` are stored on every
entity node, so every representation is reconstructable without re-ingest.
Vectors cache to `.cache/`.
