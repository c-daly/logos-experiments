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

import json
import time
from pathlib import Path
from typing import Any

from harness.fixtures_io import freeze_catalog, freeze_clusters


class ReseedInputError(ValueError):
    """Raised when live input (emergence clusters, corpus items) is malformed."""


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

    Raises ReseedInputError when a cluster lacks ``label`` or a member lacks
    ``uuid``/``name`` — live emergence output is an external input boundary
    and is never coerced.
    """
    records: list[dict[str, Any]] = []
    for i, cluster in enumerate(node_clusters):
        if "label" not in cluster:
            raise ReseedInputError(f"node_clusters[{i}] is missing required key: label")
        raw_members = cluster.get("members")
        if not isinstance(raw_members, list) or not raw_members:
            raise ReseedInputError(
                f"node_clusters[{i}] is missing required non-empty list: members"
            )
        members: list[dict[str, Any]] = []
        for j, member in enumerate(raw_members):
            for key in ("uuid", "name"):
                if key not in member:
                    raise ReseedInputError(
                        f"node_clusters[{i}].members[{j}] is missing "
                        f"required key: {key}"
                    )
            members.append({"id": member["uuid"], "name": member["name"]})
        records.append(
            {
                "cluster_id": str(cluster["label"]),
                "current_name": cluster.get("current_name", "entity"),
                "members": members,
                "sample_coverage": 1.0,
            }
        )
    return records


def validate_corpus_items(corpus: list[Any]) -> None:
    """Check every corpus item is an object carrying text and domain.

    The corpus file is an external input boundary: a malformed line must
    raise a precise error BEFORE any ingest call, not surface as a KeyError
    mid-ingest against a half-seeded graph.
    """
    for i, item in enumerate(corpus):
        if not isinstance(item, dict):
            raise ReseedInputError(
                f"corpus[{i}] is not an object (got {type(item).__name__})"
            )
        for key in ("text", "domain"):
            if key not in item:
                raise ReseedInputError(f"corpus[{i}] is missing required key: {key}")


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

    from harness.catalog import build_catalog_from_client  # T2

    seeder = HCGSeeder(client)
    seeder.clear()
    seeder.seed_type_definitions()

    corpus = [
        json.loads(ln)
        for ln in Path(corpus_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    validate_corpus_items(corpus)
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
    # build_catalog_from_client returns a CatalogResult dataclass; the freeze
    # writers and the replay loader consume the plain-dict envelope.
    catalog_result = build_catalog_from_client(client)
    catalog = {
        "catalog_by_uuid": catalog_result.catalog_by_uuid,
        "by_norm": catalog_result.by_norm,
        "roots_present_in_live_catalog": (
            catalog_result.roots_present_in_live_catalog
        ),
    }

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
