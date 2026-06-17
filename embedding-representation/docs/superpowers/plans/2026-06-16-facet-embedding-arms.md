# Facet Embedding Arms (sense + relations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two chunk-independent representation arms — `gloss`/`name_gloss` (the SENSE facet, an LLM definition) and `relations` (the STRUCTURE facet, the entity's `(relation_type, neighbor_type)` signature) — to the `embedding-representation` harness and score them on best-cut silhouette + chunk_ratio against the existing arms.

**Architecture:** Two new modules (`gloss.py`, `signatures.py`) produce the two facets offline; `represent.py`/`run.py`/`rescore.py` gain the new arms. Text arms reuse the existing cosine path (`embed_cached` + `battery` + `best_cut_silhouette`); the relations arm builds a weighted-Jaccard distance matrix from `sophia`'s `build_signature`/`signature_similarity` and feeds it into the *same* agglomerative best-cut silhouette. No graph writes; same `sample.json`; embedding model fixed at `text-embedding-3-large`/3072.

**Tech Stack:** Python 3.13, numpy, httpx (OpenAI embeddings + chat), neo4j driver, pymilvus, and `sophia` (editable-installed) for `structural_signature` + `emergence_clustering`. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-06-16-facet-embedding-arms-design.md`

**Run environment convention (important):** each experiment has its OWN `.venv`; this one doesn't exist yet (earlier runs used the sophia poetry env ad hoc — do not do that). Task 1 creates it. After Task 1, all commands below assume:
- working dir: `/home/fearsidhe/projects/logos-workspace/logos-experiments/embedding-representation`
- `PY=.venv/bin/python`
- secrets exported when noted: `OPENAI_API_KEY`, `NEO4J_PASSWORD` (live Neo4j on `bolt://localhost:7687`, Milvus on `localhost:19530`).

---

## File Structure

| File | Responsibility |
|---|---|
| `gloss.py` *(new)* | Generate one chunk-independent definition per unique entity name (parallel OpenAI chat, cached to `.cache/glosses.json`); expose `load_glosses`/`attach_glosses`. |
| `signatures.py` *(new)* | Pull each entity's `(relation, neighbor_type)` pairs from the reified-edge graph → `signatures.json`; build the weighted-Jaccard distance matrix and the relations scoring helpers. |
| `represent.py` *(modify)* | Register `gloss` / `name_gloss` representation functions in `REPS`. |
| `run.py` *(modify)* | Attach glosses; add `gloss`/`name_gloss` to the embed+battery loop. |
| `rescore.py` *(modify)* | Attach glosses; add `gloss`/`name_gloss` cosine arms and the `relations` distance-matrix arm to the silhouette/chunk_ratio table. |
| `tests/test_gloss.py` *(new)* | Unit-test prompt building + the two representation functions. |
| `tests/test_signatures.py` *(new)* | Unit-test `to_counter`, `signature_distance_matrix`, `nn_chunk_rate_dm`, `best_cut_silhouette_dm`. |
| `RECORD.md` *(modify)* | New dated run entry with the results table + verdict. |

---

## Task 1: Create the experiment's own venv

**Files:** none (environment only)

- [ ] **Step 1: Create the venv and install the experiment's own deps**

Run (from the experiment dir):

```bash
cd /home/fearsidhe/projects/logos-workspace/logos-experiments/embedding-representation
~/.local/bin/uv venv
~/.local/bin/uv sync            # neo4j, httpx, numpy + dev pytest from this dir's pyproject.toml
~/.local/bin/uv pip install -e ../../sophia   # sophia (editable) -> also brings pymilvus/neo4j/numpy
```

- [ ] **Step 2: Verify every import the experiment needs resolves in THIS venv**

Run:

```bash
.venv/bin/python -c "import numpy, httpx, neo4j, pymilvus; \
from sophia.maintenance.structural_signature import build_signature, signature_similarity; \
from sophia.maintenance.emergence_clustering import _agglomerative_partitions, _distance_matrix, _silhouette; \
import pytest; print('ENV OK')"
```

Expected: `ENV OK` (no ModuleNotFoundError).

- [ ] **Step 3: Commit the lock file**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(embedding-representation): own venv (uv) + editable sophia"
```

---

## Task 2: `gloss` / `name_gloss` representation functions

**Files:**
- Modify: `represent.py` (append after `marked_text`, before `REPS`)
- Test: `tests/test_gloss.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gloss.py`:

```python
from represent import REPS, gloss_text, name_gloss_text


def test_gloss_text_uses_gloss_when_present():
    row = {"name": "pneumococcus", "gloss": "a bacterium that causes pneumonia"}
    assert gloss_text(row) == "a bacterium that causes pneumonia"


def test_gloss_text_falls_back_to_name_when_missing():
    assert gloss_text({"name": "pneumococcus"}) == "pneumococcus"
    assert gloss_text({"name": "pneumococcus", "gloss": ""}) == "pneumococcus"


def test_name_gloss_concatenates_with_em_dash():
    row = {"name": "pneumococcus", "gloss": "a bacterium that causes pneumonia"}
    assert name_gloss_text(row) == "pneumococcus — a bacterium that causes pneumonia"


def test_name_gloss_falls_back_to_bare_name():
    assert name_gloss_text({"name": "pneumococcus"}) == "pneumococcus"


def test_arms_registered():
    assert "gloss" in REPS and "name_gloss" in REPS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gloss.py -v`
Expected: FAIL — `ImportError: cannot import name 'gloss_text'`.

- [ ] **Step 3: Add the functions and register the arms**

In `represent.py`, add after `marked_text` and extend `REPS`:

```python
def gloss_text(row: dict) -> str:
    """SENSE facet: the generated chunk-independent definition (attached by
    gloss.attach_glosses). Falls back to the bare name if no gloss is present."""
    return (row.get("gloss") or "").strip() or row["name"]


def name_gloss_text(row: dict) -> str:
    """Identity-anchored sense: '{name} — {gloss}'."""
    g = (row.get("gloss") or "").strip()
    return f"{row['name']} — {g}" if g else row["name"]


REPS = {
    "name": name_text,
    "sentence": sentence_text,
    "name_sentence": name_sentence_text,
    "marked": marked_text,
    "gloss": gloss_text,
    "name_gloss": name_gloss_text,
}
```

(Replace the existing `REPS = {...}` block; do not leave two definitions.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gloss.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add represent.py tests/test_gloss.py
git commit -m "feat(embedding-representation): gloss/name_gloss representation arms"
```

---

## Task 3: Gloss prompt builder

**Files:**
- Create: `gloss.py`
- Test: `tests/test_gloss.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gloss.py`:

```python
from gloss import build_prompt


def test_build_prompt_includes_name_and_sentence():
    p = build_prompt("plasmid", "Bacteria exchange plasmids during conjugation.")
    assert "plasmid" in p
    assert "Bacteria exchange plasmids during conjugation." in p
    # the instruction must forbid leaning on the passage (keeps the gloss
    # chunk-independent, the whole point of this arm)
    assert "not" in p.lower() and "sentence" in p.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gloss.py::test_build_prompt_includes_name_and_sentence -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gloss'`.

- [ ] **Step 3: Create `gloss.py` with the prompt builder**

```python
"""SENSE facet: a chunk-independent gloss per entity, embedded instead of the
bare name.

One short definition per UNIQUE name, generated from the name + the sentence it
was mentioned in, but explicitly FORBIDDEN to quote or paraphrase that sentence.
The sentence only disambiguates the sense; the embedded output is a clean
definition carrying no passage vocabulary -- so unlike the Run-1 context arms it
should NOT cluster by source chunk.

    OPENAI_API_KEY=... .venv/bin/python gloss.py   # generate -> .cache/glosses.json
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
from pathlib import Path

import httpx

from represent import sentence_text

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)
GLOSS_PATH = CACHE / "glosses.json"
GEN_MODEL = "gpt-4o-mini"

_PROMPT = (
    "Give a one-sentence, dictionary-style definition of the term below, as it "
    "is used in the example sentence. Write a self-contained definition of the "
    "concept ONLY. Do NOT quote or paraphrase the example sentence, do NOT "
    "mention the example, and do NOT name other entities from it.\n\n"
    "Term: {name}\n"
    "Example sentence (for disambiguation only): {sentence}\n\n"
    "Definition:"
)


def build_prompt(name: str, sentence: str) -> str:
    return _PROMPT.format(name=name, sentence=sentence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gloss.py::test_build_prompt_includes_name_and_sentence -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gloss.py tests/test_gloss.py
git commit -m "feat(embedding-representation): gloss prompt builder"
```

---

## Task 4: Parallel gloss generation + cache

**Files:**
- Modify: `gloss.py` (add `_chat`, `generate`, `load_glosses`, `attach_glosses`, `main`)

No unit test (network I/O); verified by a smoke run on 3 names, then the full run.

- [ ] **Step 1: Add the generation + accessor functions to `gloss.py`**

Append to `gloss.py`:

```python
def _chat(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (needed to generate glosses)")
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": GEN_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _save(glosses: dict) -> None:
    GLOSS_PATH.write_text(json.dumps(glosses, indent=2, ensure_ascii=False))


def load_glosses() -> dict[str, str]:
    return json.loads(GLOSS_PATH.read_text()) if GLOSS_PATH.exists() else {}


def attach_glosses(sample: list[dict], glosses: dict[str, str]) -> None:
    """Set row['gloss'] in place from the name->gloss map."""
    for row in sample:
        row["gloss"] = glosses.get(row["name"], "")


def generate(sample: list[dict], workers: int = 16, save_every: int = 200) -> dict[str, str]:
    """One gloss per UNIQUE name (names are ~unique on this corpus: 1.04
    mentions/name). Parallel across names -- ~5k serial calls would take ~40 min;
    a thread pool makes it minutes. Cached + checkpointed so re-runs are free and
    a crash only loses the last <=save_every calls."""
    by_name: dict[str, str] = {}
    for row in sample:
        by_name.setdefault(row["name"], sentence_text(row))

    glosses = load_glosses()
    todo = [n for n in by_name if n not in glosses]
    if not todo:
        print(f"all {len(by_name)} glosses cached")
        return glosses
    print(f"generating {len(todo)} glosses ({len(by_name) - len(todo)} cached) ...")

    lock = threading.Lock()
    done = 0

    def work(name: str):
        try:
            return name, _chat(build_prompt(name, by_name[name]))
        except Exception as exc:  # leave it uncached -> retried on re-run
            print(f"  ! {name!r}: {exc}")
            return name, None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, gloss in ex.map(work, todo):
            if gloss is None:
                continue
            with lock:
                glosses[name] = gloss
                done += 1
                if done % save_every == 0:
                    _save(glosses)
                    print(f"  {done}/{len(todo)} glossed")
    _save(glosses)
    print(f"wrote {GLOSS_PATH} ({len(glosses)} glosses)")
    return glosses


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    generate(sample)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 3 names (sanity-check the prompt output)**

Run:

```bash
OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python -c "
import json, gloss
sample = json.loads(open('sample.json').read())[:3]
g = gloss.generate(sample, workers=3)
for r in sample: print(r['name'], '->', g[r['name']])
"
```

Expected: 3 lines, each a clean one-sentence definition that does NOT echo the source passage. If a gloss parrots the sentence, tighten `_PROMPT` and delete `.cache/glosses.json` before re-running. (This 3-name cache will be reused by the full run.)

- [ ] **Step 3: Full generation run**

Run: `OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python gloss.py`
Expected: progress lines, ending `wrote .../.cache/glosses.json (~5131 glosses)`.

- [ ] **Step 4: Verify coverage of the sample**

Run:

```bash
.venv/bin/python -c "
import json, gloss
sample = json.loads(open('sample.json').read())
g = gloss.load_glosses()
have = sum(1 for r in sample if g.get(r['name']))
print(f'gloss coverage: {have}/{len(sample)} = {have/len(sample):.3f}')
"
```

Expected: coverage ≥ 0.99 (only failed API calls missing).

- [ ] **Step 5: Commit the code (NOT the cache — caches are gitignored)**

```bash
git add gloss.py
git commit -m "feat(embedding-representation): parallel cached gloss generation"
```

---

## Task 5: Pull entity relation signatures

**Files:**
- Create: `signatures.py` (fetch + persist; scoring helpers added in Task 6)

- [ ] **Step 1: Create `signatures.py` with the fetch path**

```python
"""RELATIONS facet: each entity's (relation_type, neighbor_type) signature,
pulled from the live reified-edge graph and scored by weighted Jaccard.

Mirrors production sophia.hcg_client.HCGClient.query_edges_from -- FROM-edges
only (the node as edge source). Reified edges are :Node carrying a `relation`
property, linked (edge)-[:FROM]->(source) and (edge)-[:TO]->(target). The
signature is chunk-blind by construction: no passage text touches it.

    NEO4J_PASSWORD=... .venv/bin/python signatures.py   # pull -> signatures.json
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from sophia.maintenance.structural_signature import build_signature, signature_similarity

HERE = Path(__file__).resolve().parent
SIG_PATH = HERE / "signatures.json"

# FROM-edges of each sampled node + the neighbour's type, in one batched read.
_CYPHER = """
MATCH (edge:Node)-[:FROM]->(src:Node)
WHERE src.uuid IN $uuids AND edge.relation IS NOT NULL
OPTIONAL MATCH (edge)-[:TO]->(tgt:Node)
RETURN src.uuid AS uuid, edge.relation AS relation, tgt.type AS neighbor_type
"""


def fetch_signatures(uuids: list[str]) -> dict[str, list[list[str]]]:
    """Return {uuid: [[relation, neighbor_type], ...]} for every uuid (empty
    list if the node has no typed FROM-edges)."""
    from neo4j import GraphDatabase

    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise SystemExit("NEO4J_PASSWORD must be set")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    out: dict[str, list[list[str]]] = {u: [] for u in uuids}

    def work(tx):
        rows: list[dict] = []
        for i in range(0, len(uuids), 1000):
            rows += tx.run(_CYPHER, uuids=uuids[i : i + 1000]).data()
        return rows

    try:
        with driver.session() as s:
            for rec in s.execute_read(work):
                if rec["relation"] and rec["neighbor_type"]:
                    out[rec["uuid"]].append([rec["relation"], rec["neighbor_type"]])
    finally:
        driver.close()
    return out


def load_signatures() -> dict[str, list[list[str]]]:
    return json.loads(SIG_PATH.read_text()) if SIG_PATH.exists() else {}


def to_counter(pairs: list[list[str]]) -> Counter:
    """Reuse production build_signature on stored [relation, neighbor_type] pairs."""
    return build_signature(
        [{"relation": r, "neighbor_type": t} for r, t in pairs]
    )


def main() -> None:
    sample = json.loads((HERE / "sample.json").read_text())
    uuids = [r["uuid"] for r in sample]
    sigs = fetch_signatures(uuids)
    SIG_PATH.write_text(json.dumps(sigs, ensure_ascii=False))
    nonempty = sum(1 for u in uuids if sigs.get(u))
    print(f"wrote {SIG_PATH}")
    print(f"signature coverage: {nonempty}/{len(uuids)} = {nonempty / len(uuids):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Pull the signatures**

Run: `NEO4J_PASSWORD=$NEO4J_PASSWORD .venv/bin/python signatures.py`
Expected: `wrote .../signatures.json` and a `signature coverage: N/5330 = ...` line. **Record the coverage number** — it qualifies the relations result. If coverage is poor (< ~0.5), note it; bidirectional edges (decision D fallback) would be the follow-up.

- [ ] **Step 3: Commit the code (signatures.json is regenerable; gitignore it)**

```bash
grep -qxF "signatures.json" .gitignore || echo "signatures.json" >> .gitignore
git add signatures.py .gitignore
git commit -m "feat(embedding-representation): pull entity relation signatures"
```

---

## Task 6: Relations scoring helpers

**Files:**
- Modify: `signatures.py` (add `signature_distance_matrix`, `nn_chunk_rate_dm`, `best_cut_silhouette_dm`)
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signatures.py`:

```python
from collections import Counter

import numpy as np

from signatures import (
    best_cut_silhouette_dm,
    nn_chunk_rate_dm,
    signature_distance_matrix,
    to_counter,
)


def test_to_counter_tallies_pairs():
    c = to_counter([["IS_A", "concept"], ["IS_A", "concept"], ["CAUSES", "process"]])
    assert c == Counter({("IS_A", "concept"): 2, ("CAUSES", "process"): 1})


def test_distance_matrix_is_zero_on_diagonal_and_symmetric():
    a = Counter({("IS_A", "concept"): 1})
    b = Counter({("IS_A", "concept"): 1})            # identical -> distance 0
    c = Counter({("CAUSES", "process"): 1})          # disjoint  -> distance 1
    dm = signature_distance_matrix([a, b, c])
    assert dm.shape == (3, 3)
    assert np.allclose(np.diag(dm), 0.0)
    assert np.allclose(dm, dm.T)
    assert dm[0, 1] == 0.0
    assert dm[0, 2] == 1.0


def test_nn_chunk_rate_dm_detects_chunk_clustering():
    # 4 points; 0&1 share chunk "A" and are each other's nearest neighbour.
    dm = np.array(
        [[0.0, 0.1, 0.9, 0.9],
         [0.1, 0.0, 0.9, 0.9],
         [0.9, 0.9, 0.0, 0.1],
         [0.9, 0.9, 0.1, 0.0]],
        dtype="float32",
    )
    out = nn_chunk_rate_dm(dm, ["A", "A", "B", "B"], k=1)
    assert out["nn_same_chunk"] == 1.0     # every NN is a chunk-mate
    assert out["ratio"] is not None and out["ratio"] > 1.0


def test_best_cut_silhouette_dm_finds_two_clusters():
    # two tight clusters, far apart -> best cut at k=2, positive silhouette
    dm = np.array(
        [[0.0, 0.05, 0.9, 0.9],
         [0.05, 0.0, 0.9, 0.9],
         [0.9, 0.9, 0.0, 0.05],
         [0.9, 0.9, 0.05, 0.0]],
        dtype="float32",
    )
    k, sil = best_cut_silhouette_dm(dm)
    assert k == 2
    assert sil > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_signatures.py -v`
Expected: FAIL — `ImportError: cannot import name 'signature_distance_matrix'`.

- [ ] **Step 3: Add the helpers to `signatures.py`**

Append to `signatures.py` (the `from sophia... emergence_clustering` import goes at the top with the other imports):

```python
# add near the top imports:
# from sophia.maintenance.emergence_clustering import _agglomerative_partitions, _silhouette


def signature_distance_matrix(sigs: list[Counter]) -> np.ndarray:
    """Symmetric weighted-Jaccard DISTANCE matrix (1 - signature_similarity)."""
    n = len(sigs)
    dm = np.zeros((n, n), dtype="float32")
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - signature_similarity(sigs[i], sigs[j])
            dm[i, j] = dm[j, i] = d
    return dm


def nn_chunk_rate_dm(dm: np.ndarray, chunk_ids: list, k: int = 10) -> dict:
    """Sanity check for the relations arm: fraction of each point's k nearest
    neighbours (by signature distance) that share its chunk, vs the random
    expectation. For a chunk-BLIND representation this should be ~1x."""
    n = len(dm)
    cids = np.asarray(chunk_ids)
    order = np.argsort(dm, axis=1)
    obs = []
    for i in range(n):
        neigh = [j for j in order[i] if j != i][:k]
        if neigh:
            obs.append(float(np.mean(cids[neigh] == cids[i])))
    observed = float(np.mean(obs)) if obs else 0.0
    cnt = Counter(chunk_ids)
    expected = sum(c * (c - 1) for c in cnt.values()) / (n * (n - 1)) if n > 1 else 0.0
    return {
        "nn_same_chunk": round(observed, 3),
        "expected_random": round(expected, 3),
        "ratio": round(observed / expected, 1) if expected else None,
    }


def best_cut_silhouette_dm(dm: np.ndarray) -> tuple[int | None, float | None]:
    """Best-cut silhouette over a precomputed distance matrix, using the SAME
    agglomerative cut that rescore.py uses for the vector arms."""
    n = len(dm)
    parts = _agglomerative_partitions(dm.tolist(), 2, max(2, n // 3))
    if not parts:
        return None, None
    k, lab = max(parts.items(), key=lambda kv: _silhouette(dm.tolist(), kv[1]))
    return int(k), round(_silhouette(dm.tolist(), lab), 4)
```

Add the import at the top of `signatures.py`:

```python
from sophia.maintenance.emergence_clustering import _agglomerative_partitions, _silhouette
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_signatures.py -v`
Expected: PASS (4 passed).

> If `_agglomerative_partitions` / `_silhouette` expect a `list[list[float]]` rather than an ndarray, the `.tolist()` calls above already handle it; if they accept an ndarray directly, the `.tolist()` is harmless. Confirm against `rescore.py::best_cut_silhouette`, which calls them the same way.

- [ ] **Step 5: Commit**

```bash
git add signatures.py tests/test_signatures.py
git commit -m "feat(embedding-representation): relations distance matrix + scoring helpers"
```

---

## Task 7: Wire `gloss` / `name_gloss` into the score runners

**Files:**
- Modify: `run.py` (battery path)
- Modify: `rescore.py` (silhouette path)

- [ ] **Step 1: Add gloss attach + arms to `run.py`**

In `run.py`, add the import near the other imports:

```python
from gloss import attach_glosses, load_glosses
```

In `run.py::main`, immediately after `print(f"sample: ...")` (before `name_vecs = load_name_vectors(uuids)`), attach glosses:

```python
    attach_glosses(sample, load_glosses())
```

Then extend the embed loop tuple:

```python
    for arm in ("sentence", "name_sentence", "marked", "gloss", "name_gloss"):
```

- [ ] **Step 2: Add gloss attach + arms to `rescore.py`**

In `rescore.py`, add the import:

```python
from gloss import attach_glosses, load_glosses
```

In `rescore.py::main`, after `chunk_ids = [r["raw_text"] for r in sample]`, attach glosses:

```python
    attach_glosses(sample, load_glosses())
```

Add the two cosine arms to the `arms` dict (alongside the existing entries):

```python
        "gloss": embed_cached([REPS["gloss"](r) for r in sample]),
        "name_gloss": embed_cached([REPS["name_gloss"](r) for r in sample]),
```

- [ ] **Step 3: Run the silhouette table (the decisive metric)**

Run: `OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python rescore.py`
Expected: the existing table plus `gloss` and `name_gloss` rows, each with `best_k`, `silhouette`, `chunk_ratio`; `rescore.json` updated. **Record the gloss silhouette + chunk_ratio.** Hypothesis: silhouette up vs `name` (0.062), `chunk_ratio` near the name floor (~37×, NOT ~180×).

- [ ] **Step 4: Run the battery (continuity)**

Run: `OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python run.py`
Expected: `results.json` + table now include `gloss`/`name_gloss` battery rows. (Informational; the silhouette in Step 3 is the decision metric.)

- [ ] **Step 5: Commit**

```bash
git add run.py rescore.py
git commit -m "feat(embedding-representation): score gloss/name_gloss arms"
```

---

## Task 8: Wire the `relations` arm into `rescore.py`

**Files:**
- Modify: `rescore.py`

- [ ] **Step 1: Add the relations imports + the scoring block**

In `rescore.py`, add imports (near the top):

```python
import random
from signatures import (
    best_cut_silhouette_dm,
    load_signatures,
    nn_chunk_rate_dm,
    signature_distance_matrix,
    to_counter,
)
```

In `rescore.py::main`, after the vector-arm loop writes `out` (after the `for arm, X in arms.items():` loop, before `(HERE / "rescore.json").write_text(...)`), add:

```python
    # relations arm: distance-matrix path (no embedding vectors)
    sigs_by_uuid = load_signatures()
    covered = [r for r in sample if sigs_by_uuid.get(r["uuid"])]
    cov = len(covered) / len(sample) if sample else 0.0
    if covered:
        sub = (
            covered
            if len(covered) <= SAMPLE
            else [covered[i] for i in random.Random(0).sample(range(len(covered)), SAMPLE)]
        )
        sigs = [to_counter(sigs_by_uuid[r["uuid"]]) for r in sub]
        dm = signature_distance_matrix(sigs)
        rk, rsil = best_cut_silhouette_dm(dm)
        rcr = nn_chunk_rate_dm(dm, [r["raw_text"] for r in sub])["ratio"]
        out["relations"] = {
            "best_k": rk,
            "silhouette": rsil,
            "chunk_ratio": rcr,
            "coverage": round(cov, 3),
            "n": len(sub),
        }
        print(f"{'relations':<16}{rk!s:>8}{rsil!s:>12}{rcr!s:>13}"
              f"  (coverage={cov:.2f}, n={len(sub)})")
    else:
        print("relations: no signatures found (run signatures.py first)")
```

- [ ] **Step 2: Run and capture the relations result**

Run: `OPENAI_API_KEY=$OPENAI_API_KEY .venv/bin/python rescore.py`
Expected: the table now ends with a `relations` row showing `best_k`, `silhouette`, `chunk_ratio`, coverage, n; `rescore.json` has a `relations` entry. **Record silhouette + coverage.** Hypothesis: silhouette up, `chunk_ratio` ≈ 1× (chunk-blind by construction).

- [ ] **Step 3: Commit**

```bash
git add rescore.py
git commit -m "feat(embedding-representation): score relations (structural-signature) arm"
```

---

## Task 9: Record the run + verdict

**Files:**
- Modify: `RECORD.md`

- [ ] **Step 1: Append a dated run section to `RECORD.md`**

Add a new section at the end of `RECORD.md` titled `## Run 3 — facet arms: sense (gloss) + structure (relations) (2026-06-16)`. Fill the table from `rescore.json` with the real numbers captured in Tasks 7–8:

```markdown
## Run 3 — facet arms: sense (gloss) + structure (relations) (2026-06-16)

`rescore.json`. Best-cut silhouette + chunk_ratio (control 0.213 / junk 0.060 reference):

| arm        | best_k | silhouette | chunk_ratio | note |
|------------|--------|------------|-------------|------|
| name       | <fill> | 0.062      | 37×         | baseline |
| sentence   | <fill> | 0.181      | 177×        | chunk-coupled (false) |
| gloss      | <fill> | <fill>     | <fill>      | sense facet |
| name_gloss | <fill> | <fill>     | <fill>      | anchored sense |
| relations  | <fill> | <fill>     | <fill>      | structure facet; coverage <fill>, n <fill> |

**Verdict:** <one paragraph — did either facet reach high silhouette at LOW
chunk_ratio (the fine-grained AND chunk-independent target no prior arm hit)?
State the gloss chunk_ratio vs the name floor, the relations coverage, and
whether the v2 gloss+relations correspondence arm is now worth running.>
```

- [ ] **Step 2: Run the full test suite once more (guard against regressions)**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests in `tests/test_gloss.py` + `tests/test_signatures.py` PASS.

- [ ] **Step 3: Commit**

```bash
git add RECORD.md
git commit -m "docs(embedding-representation): Run 3 — sense + structure facet arms"
```

---

## Self-Review

**Spec coverage:**
- `gloss` arm → Tasks 2–4, 7. ✓
- `name_gloss` arm → Tasks 2, 7. ✓
- `relations` arm (build_signature → weighted-Jaccard → same silhouette) → Tasks 5, 6, 8. ✓
- Coverage reporting for relations → Tasks 5 (pull), 8 (scored subset). ✓
- Two scoring paths, one metric → Tasks 7 (cosine) + 8 (distance matrix), both via best-cut silhouette. ✓
- Battery continuity for text arms → Task 7 Step 4. ✓
- Decision A (`name_gloss`) ✓; B (`gpt-4o-mini`, Task 3/4) ✓; C (defer correspondence arm — noted as Run-3 verdict follow-up, not implemented) ✓; D (from-edges only, Task 5 Cypher; bidirectional as documented fallback) ✓.
- Offline / no graph writes / fixed model / no `proposal_builder.py` change → nothing in any task writes to the graph or touches hermes/sophia source. ✓
- RECORD.md run entry → Task 9. ✓

**Placeholder scan:** the only `<fill>` markers are in the Task 9 RECORD table, which is intentionally filled from the actual run output (Tasks 7–8) — not a code placeholder.

**Type consistency:** `attach_glosses(sample, glosses)` / `load_glosses()` used identically in `gloss.py`, `run.py`, `rescore.py`. `to_counter` → `Counter`; `signature_distance_matrix(list[Counter]) -> ndarray`; `best_cut_silhouette_dm(ndarray) -> (k, sil)`; `nn_chunk_rate_dm(ndarray, list) -> dict` — consistent across `signatures.py` and `tests/test_signatures.py`. `REPS["gloss"]`/`REPS["name_gloss"]` defined in Task 2, consumed in Task 7. ✓
