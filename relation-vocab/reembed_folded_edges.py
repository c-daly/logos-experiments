"""Re-embed the edges renamed by a consolidation apply (lx34b follow-up).

An edge's vector in ``hcg_edge_embeddings`` is the embedding of the phrase
``"{source_name} {relation} {target_name}"`` (hermes ``proposal_builder._embed_edges``).
After ``apply_mapping.py`` renames ``relation`` on a folded edge, that vector
still encodes the OLD surface (e.g. an edge folded ANALYSED->ANALYZED is still
embedded as "... analysed ..."). This recomputes the phrase with the NEW
relation and upserts the fresh vector by uuid.

Runs in the **hermes poetry venv** (not uv): it uses the pipeline's own
``get_embedding_provider`` so the vectors are identical to what ingestion would
produce, plus pymilvus + the neo4j driver.

    cd hermes
    NEO4J_PASSWORD=... poetry run python \\
        ../logos-experiments/relation-vocab/reembed_folded_edges.py            # dry-run
    NEO4J_PASSWORD=... poetry run python \\
        ../logos-experiments/relation-vocab/reembed_folded_edges.py --apply    # upsert Milvus
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTION = "hcg_edge_embeddings"
MODEL = "text-embedding-3-large"


def build_phrase(source_name: str, relation: str, target_name: str) -> str:
    """The exact phrase hermes embeds for an edge (proposal_builder._embed_edges)."""
    return f"{source_name} {relation.lower().replace('_', ' ')} {target_name}"


def latest_rollback() -> str:
    files = sorted(glob.glob(str(HERE / "rollback_*.json")))
    if not files:
        raise SystemExit("no rollback_*.json found; pass --rollback explicitly")
    return files[-1]


def fetch_edges(uuids: list[str]) -> list[dict]:
    """For each edge uuid: its current relation + endpoint names, from Neo4j."""
    from neo4j import GraphDatabase

    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        raise SystemExit("NEO4J_PASSWORD must be set")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pw))

    def work(tx):
        return tx.run(
            "UNWIND $uuids AS u "
            "MATCH (e:Node {type:'edge', uuid: u}) "
            "MATCH (s:Node {uuid: e.source}) MATCH (t:Node {uuid: e.target}) "
            "RETURN e.uuid AS uuid, e.relation AS rel, "
            "s.name AS source_name, t.name AS target_name",
            uuids=uuids,
        ).data()

    try:
        with driver.session() as s:
            return s.execute_read(work)
    finally:
        driver.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", default=None, help="rollback_*.json (default: latest)")
    ap.add_argument("--apply", action="store_true", help="upsert into Milvus")
    args = ap.parse_args()

    rollback_path = args.rollback or latest_rollback()
    rb = json.loads(Path(rollback_path).read_text(encoding="utf-8"))
    changed = [r for r in rb if r["old"] != r["new"]]
    uuids = [r["uuid"] for r in changed]
    print(f"rollback: {rollback_path}  ({len(changed)} relation-changed edges)")

    recs = fetch_edges(uuids)
    items = [
        (r["uuid"], build_phrase(r["source_name"], r["rel"], r["target_name"]))
        for r in recs
        if r["source_name"] and r["target_name"] and r["rel"]
    ]
    print(f"resolved {len(items)}/{len(uuids)} edges (present + named) for re-embed")
    if not items:
        print("nothing to re-embed (rollback empty, or all uuids absent/unnamed)")
        return
    for u, ph in items[:5]:
        print(f"  {u[:8]}  \"{ph}\"")

    if not args.apply:
        print("\nDRY RUN -- no embeddings computed, no Milvus writes. "
              "Re-run with --apply.")
        return

    from logos_config import MilvusConfig
    from pymilvus import Collection, connections

    cfg = MilvusConfig()
    connections.connect(host=cfg.host, port=str(cfg.port))
    col = Collection(COLLECTION)

    # Pin model + dim to the COLLECTION so the re-embed can't drift to a
    # different model than the stored vectors. `poetry run` does not load
    # hermes/.env, so get_embedding_provider() would otherwise fall back to its
    # text-embedding-3-small/1536 default and Milvus would reject the upsert.
    dim = next((f.params["dim"] for f in col.schema.fields if f.params.get("dim")), None)
    if dim is None:
        raise SystemExit(f"no vector field with 'dim' found in {COLLECTION} schema")
    probe = col.query(
        expr=f'uuid == "{items[0][0]}"', output_fields=["embedding_model"], limit=1
    )
    model = probe[0]["embedding_model"] if probe else MODEL
    os.environ["EMBEDDING_MODEL"] = model
    os.environ["LOGOS_EMBEDDING_DIM"] = str(dim)
    print(f"target collection: model={model} dim={dim}")

    from hermes.embedding_provider import get_embedding_provider

    provider = get_embedding_provider()
    if provider._model_name != model or provider._dimension != dim:
        raise SystemExit(
            f"provider {provider._model_name}/{provider._dimension} != "
            f"collection {model}/{dim}; refusing to write mismatched vectors"
        )

    phrases = [ph for _, ph in items]
    vecs: list[list[float]] = []
    for i in range(0, len(phrases), 256):  # chunk under the OpenAI input cap
        vecs.extend(await provider.embed_batch(phrases[i : i + 256]))
        print(f"  embedded {len(vecs)}/{len(phrases)}")
    bad = [len(v) for v in vecs if len(v) != dim]
    if bad:
        raise SystemExit(f"{len(bad)} vectors are not dim {dim}; aborting upsert")

    now = int(time.time())
    col.upsert(
        [
            [u for u, _ in items],          # uuid (primary)
            vecs,                           # embedding
            [model] * len(items),           # embedding_model
            [now] * len(items),             # last_sync
        ]
    )
    col.flush()
    print(f"\nUPSERTED {len(items)} edge embeddings into {COLLECTION}.")


if __name__ == "__main__":
    asyncio.run(main())
