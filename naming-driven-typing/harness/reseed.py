"""Clean reseed + build for the naming-driven-typing experiment.

The live `reseed_and_build` path (added below) clears the graph, seeds the
realm roots, cold-start ingests the curated corpus, runs emergence clustering,
and freezes {clusters, catalog} to fixtures/. It is gated behind RESEED_LIVE=1
and is a smoke/illustration path only — graded runs always --replay the frozen
fixtures (SPEC §7.6).

`clusters_from_node_members` is a pure mapper (no graph) so it is unit-testable
offline: it maps emergence `node_clusters` to the frozen cluster-record schema
{cluster_id, current_name, members:[{id,name}], sample_coverage}.
"""
from __future__ import annotations

from typing import Any


def clusters_from_node_members(
    node_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map emergence node-clusters to frozen cluster records (label-free).

    Each input cluster is `{label, current_name?, members:[{uuid,name}]}`
    (see edge-embeddings workspace/round_*.json node_clusters). Output records
    are `{cluster_id, current_name, members:[{id,name}], sample_coverage}`.

    We send the WHOLE cluster (no down-sampling, SPEC §6.1) so sample_coverage
    is always 1.0 here; the live harness overrides it only if it must truncate.
    No `label`/`labels` field is ever emitted — eval is label-free.
    """
    records: list[dict[str, Any]] = []
    for cluster in node_clusters:
        members = [
            {"id": m["uuid"], "name": m["name"]}
            for m in cluster.get("members", [])
        ]
        records.append(
            {
                "cluster_id": str(cluster["label"]),
                "current_name": cluster.get("current_name", "entity"),
                "members": members,
                "sample_coverage": 1.0,
            }
        )
    return records


import json
import os
import time
from pathlib import Path

from harness.fixtures_io import freeze_catalog, freeze_clusters


def _ingest(text: str, domain: str, hermes_url: str) -> None:
    """Cold-start ingest one block via Hermes (retry on transient errors)."""
    # Lazy: httpx is needed only on the live path; the offline test env
    # (pyproject dependencies = []) must import this module without it.
    import httpx

    last_err: Exception | None = None
    for _ in range(5):
        try:
            resp = httpx.post(
                f"{hermes_url}/ingest",
                json={"text": text, "metadata": {"domain": domain}},
                timeout=60.0,
            )
            resp.raise_for_status()
            return
        except Exception as err:  # noqa: BLE001 — transient ingest retry
            last_err = err
            time.sleep(2.0)
    raise RuntimeError(f"ingest failed after 5 retries: {last_err}")


def reseed_and_build(
    client: Any,
    sync: Any,
    *,
    corpus_path: Path,
    hermes_url: str,
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    """Clear the graph, seed roots, cold-start ingest the corpus, cluster, build.

    Returns {clusters, catalog, meta}. Gated behind RESEED_LIVE=1 by the caller;
    this is a smoke/illustration path — graded runs --replay frozen fixtures
    (SPEC §7.6). Read-write against a DISPOSABLE stack only.
    """
    # Lazy: only the live path needs these heavy deps.
    from logos_hcg.seeder import HCGSeeder

    # edge-embeddings harness lives in the sibling experiment; import its
    # population builder + clustering by adding it to sys.path at call time.
    import sys

    edge_harness = (
        Path(corpus_path).resolve().parents[2]
        / "edge-embeddings-worth-it"
        / "harness"
    )
    if str(edge_harness) not in sys.path:
        sys.path.insert(0, str(edge_harness))
    from run_experiment import build_node_members  # type: ignore[import-not-found]
    from sophia.maintenance.emergence_clustering import find_emergent_clusters

    from harness.catalog import build_enriched_catalog  # T2

    seeder = HCGSeeder(client)
    seeder.clear()
    seeder.seed_type_definitions()

    corpus = [
        json.loads(ln)
        for ln in Path(corpus_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for item in corpus:
        _ingest(item["text"], item["domain"], hermes_url)

    driver = client.driver
    node_members, _, _ = build_node_members(driver, sync, entity_filter=True, dedup=True)
    raw_clusters = find_emergent_clusters(node_members, min_cluster_size=min_cluster_size)

    # find_emergent_clusters returns objects with .label/.members[{uuid,name}];
    # normalize to dicts the pure mapper expects.
    node_cluster_dicts = [
        {
            "label": c.label,
            "current_name": "entity",
            "members": [{"uuid": m.uuid, "name": m.name} for m in c.members],
        }
        for c in raw_clusters
    ]
    clusters = clusters_from_node_members(node_cluster_dicts)
    catalog = build_enriched_catalog(client)

    fixtures_dir = Path(corpus_path).resolve().parents[1] / "fixtures"
    freeze_clusters(clusters, fixtures_dir / "clusters.json")
    freeze_catalog(catalog, fixtures_dir / "catalog.json")

    return {
        "clusters": clusters,
        "catalog": catalog,
        "meta": {
            "n_clusters": len(clusters),
            "n_corpus_blocks": len(corpus),
            "reseeded_at": int(time.time()),
        },
    }
