"""Clean reseed + build for the naming-driven-typing experiment.

The live `reseed_and_build` path (added below) clears the graph, seeds the
realm roots, cold-start ingests the curated corpus, runs emergence clustering,
and freezes {clusters, catalog} to fixtures/. It is gated behind RESEED_LIVE=1
and is a smoke/illustration path only — graded runs always --replay the frozen
fixtures (SPEC §7.6).

`clusters_from_node_members` is a pure mapper (no graph) so it is unit-testable
offline: it maps emergence `node_clusters` to the frozen cluster-record schema
{cluster_id, current_name, members:[{id,name}], sample_coverage}.

Driver CLI (issue #13): ``python harness/reseed.py [--graded] [--corpus NAME]``,
gated behind RESEED_LIVE=1. The smoke default corpus is ``corpus/corpus.jsonl``
(16 blocks); the GRADED default is the blessed ``corpus/corpus_batch3.jsonl``
(350 blocks / 8 domains, approved 2026-06-05): run ``--graded`` for the graded
reseed. Cold-start ingest goes through the deployed hermes (``--hermes-url`` /
HERMES_URL) and the frozen {clusters, catalog} land via the canonical fixture
writers (`freeze_clusters` / `freeze_catalog`). Requires the
edge-embeddings-worth-it sibling experiment checked out next to
naming-driven-typing (its harness supplies build_node_members).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Direct-script execution (python harness/reseed.py) puts harness/ -- not the
# experiment root -- on sys.path; shim the root in so the ``harness.*``
# imports resolve (same role as the shim in run_experiment.py).
_EXP_DIR = Path(__file__).resolve().parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from harness.fixtures_io import freeze_catalog, freeze_clusters  # noqa: E402

# Corpus defaults (issue #13): the smoke path keeps the 16-block curated set;
# the GRADED path uses the blessed batch3 corpus checked in byte-identical at
# corpus/corpus_batch3.jsonl (350 blocks / 8 domains, approved 2026-06-05).
SMOKE_CORPUS = "corpus.jsonl"
GRADED_CORPUS = "corpus_batch3.jsonl"


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


def resolve_corpus_path(
    corpus: Optional[str],
    *,
    graded: bool,
    corpus_dir: Path,
) -> Path:
    """Pick the corpus file: explicit ``--corpus`` wins, else the mode default.

    The smoke default is ``corpus.jsonl`` (16 curated blocks); the GRADED
    default is the blessed ``corpus_batch3.jsonl`` (350 blocks / 8 domains,
    approved 2026-06-05). A relative ``--corpus`` resolves under the corpus
    dir first, then as given. A missing file raises ReseedInputError: corpus
    selection must never fall back silently.
    """
    if corpus:
        candidate = Path(corpus)
        if not candidate.is_absolute():
            in_dir = corpus_dir / corpus
            if in_dir.exists():
                return in_dir
        if candidate.exists():
            return candidate
        raise ReseedInputError(f"corpus file not found: {corpus}")
    default_name = GRADED_CORPUS if graded else SMOKE_CORPUS
    path = corpus_dir / default_name
    if not path.exists():
        raise ReseedInputError(f"default corpus missing: {path}")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    """Reseed driver (gated behind RESEED_LIVE=1; see the module docstring).

    The graded entry point is ``--graded``: it defaults the corpus to the
    blessed batch3 set and otherwise runs the same clear -> seed roots ->
    cold-start ingest -> emergence -> freeze pipeline through the canonical
    fixture writers.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            "corpus file name (under corpus/) or path; overrides the "
            "per-mode default"
        ),
    )
    parser.add_argument(
        "--graded",
        action="store_true",
        help=(
            "graded reseed: default corpus is the blessed corpus_batch3.jsonl "
            "(350 blocks / 8 domains)"
        ),
    )
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument(
        "--hermes-url",
        default=os.environ.get("HERMES_URL", "http://localhost:17000"),
        help="deployed hermes used for the cold-start ingest",
    )
    args = parser.parse_args(argv)

    if os.environ.get("RESEED_LIVE") != "1":
        print(
            "[reseed] live reseed is gated: set RESEED_LIVE=1 to clear and "
            "reseed the DISPOSABLE stack (this is a destructive write path)",
            file=sys.stderr,
        )
        return 2

    corpus_path = resolve_corpus_path(
        args.corpus, graded=args.graded, corpus_dir=_EXP_DIR / "corpus"
    )
    fixtures_dir = corpus_path.resolve().parents[1] / "fixtures"
    print(f"[reseed] corpus -> {corpus_path}", flush=True)
    print(f"[reseed] fixtures -> {fixtures_dir}", flush=True)

    # Lazy: only the live path needs the stack clients.
    from logos_hcg.client import HCGClient
    from logos_hcg.sync import HCGMilvusSync

    client = HCGClient(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "logosdev"),
    )
    sync = HCGMilvusSync(
        milvus_host=os.environ.get("MILVUS_HOST", "localhost"),
        milvus_port=os.environ.get("MILVUS_PORT", "19530"),
    )
    result = reseed_and_build(
        client,
        sync,
        corpus_path=corpus_path,
        hermes_url=args.hermes_url,
        min_cluster_size=args.min_cluster_size,
    )
    meta = result["meta"]
    print(
        "[reseed] frozen {n} clusters from {b} corpus blocks".format(
            n=meta["n_clusters"], b=meta["n_corpus_blocks"]
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
