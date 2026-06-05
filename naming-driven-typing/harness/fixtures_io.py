"""Frozen-fixture I/O for the naming-driven-typing experiment (label-free).

Two fixtures, both written deterministically (sorted keys, sorted records,
canonically sorted list content, 2-space indent, trailing newline) so a reseed
produces byte-identical files and git diffs stay reviewable:

  fixtures/clusters.json  -> {version, clusters:[{cluster_id, current_name,
                              members:[{id,name}], sample_coverage}]}
  fixtures/catalog.json   -> {version, catalog_by_uuid, by_norm,
                              roots_present_in_live_catalog}

There is NO labels.json (2026-06-05 override). The cluster validator actively
REJECTS any `label`/`labels` key so a coherence label can never re-enter via a
hand-edited fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURE_VERSION = "1"

# The three realm roots are unconditionally present by name-membership (SPEC §4.3).
_REALM_ROOTS = ("entity", "concept", "process")

# A coherence/root label must never re-enter a label-free fixture.
_FORBIDDEN_CLUSTER_KEYS = ("label", "labels", "coherence", "root_gt", "gold")


class ClusterFixtureError(ValueError):
    """Raised when clusters.json violates the label-free cluster schema."""


class CatalogFixtureError(ValueError):
    """Raised when catalog.json violates the enriched-catalog schema."""


def _write_deterministic(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


# --- clusters -------------------------------------------------------------

def validate_clusters(clusters: list[dict[str, Any]]) -> None:
    """Validate the label-free cluster schema; raise ClusterFixtureError."""
    if not isinstance(clusters, list) or not clusters:
        raise ClusterFixtureError("clusters must be a non-empty list")
    seen_ids: set[str] = set()
    seen_cluster_ids: set[str] = set()
    for i, c in enumerate(clusters):
        if not isinstance(c, dict):
            raise ClusterFixtureError(
                f"clusters[{i}] is not an object (got {type(c).__name__})"
            )
        for forbidden in _FORBIDDEN_CLUSTER_KEYS:
            if forbidden in c:
                raise ClusterFixtureError(
                    f"label-free violation: cluster has forbidden key {forbidden!r}"
                )
        cid = c.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            raise ClusterFixtureError("each cluster needs a non-empty str cluster_id")
        if cid in seen_cluster_ids:
            raise ClusterFixtureError(f"duplicate cluster_id {cid!r}")
        seen_cluster_ids.add(cid)
        cov = c.get("sample_coverage")
        if not isinstance(cov, (int, float)) or not (0.0 < cov <= 1.0):
            raise ClusterFixtureError(
                f"cluster {cid}: sample_coverage must be in (0, 1], got {cov!r}"
            )
        members = c.get("members")
        if not isinstance(members, list) or not members:
            raise ClusterFixtureError(f"cluster {cid}: members must be non-empty list")
        for j, m in enumerate(members):
            if not isinstance(m, dict):
                raise ClusterFixtureError(
                    f"cluster {cid}: members[{j}] is not an object "
                    f"(got {type(m).__name__})"
                )
            mid, name = m.get("id"), m.get("name")
            if not isinstance(mid, str) or not mid:
                raise ClusterFixtureError(f"cluster {cid}: member needs non-empty id")
            if not isinstance(name, str) or not name:
                raise ClusterFixtureError(f"cluster {cid}: member needs non-empty name")
            if mid in seen_ids:
                raise ClusterFixtureError(f"member id {mid!r} not unique across clusters")
            seen_ids.add(mid)


def freeze_clusters(clusters: list[dict[str, Any]], path: Path) -> None:
    """Validate then write clusters deterministically (sorted by cluster_id).

    List content is canonicalized too: `sort_keys=True` only sorts dict KEYS,
    so each set-like `members` list is sorted by member `id` here. Otherwise
    live-return order would leak into the frozen bytes and a future reseed
    could silently break byte-determinism.
    """
    validate_clusters(clusters)
    ordered = [
        {**c, "members": sorted(c["members"], key=lambda m: m["id"])}
        for c in sorted(clusters, key=lambda c: c["cluster_id"])
    ]
    _write_deterministic({"version": FIXTURE_VERSION, "clusters": ordered}, path)


def load_clusters(path: Path) -> list[dict[str, Any]]:
    """Load + validate clusters.json, returning the clusters list."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != FIXTURE_VERSION:
        raise ClusterFixtureError(
            f"clusters fixture version mismatch: {raw.get('version')!r}"
        )
    clusters = raw.get("clusters")
    validate_clusters(clusters)
    return clusters


# --- catalog --------------------------------------------------------------

def validate_catalog(catalog: dict[str, Any]) -> None:
    """Validate the enriched catalog schema; raise CatalogFixtureError."""
    if not isinstance(catalog, dict):
        raise CatalogFixtureError(
            f"catalog must be an object (got {type(catalog).__name__})"
        )
    by_uuid = catalog.get("catalog_by_uuid")
    by_norm = catalog.get("by_norm")
    if not isinstance(by_uuid, dict) or not by_uuid:
        raise CatalogFixtureError("catalog_by_uuid must be a non-empty dict")
    for uuid, rec in by_uuid.items():
        if not isinstance(rec, dict):
            raise CatalogFixtureError(
                f"catalog_by_uuid[{uuid!r}] must be an object "
                f"(got {type(rec).__name__})"
            )
    if not isinstance(by_norm, dict) or not by_norm:
        raise CatalogFixtureError("by_norm must be a non-empty dict")
    if catalog.get("roots_present_in_live_catalog") is not True:
        raise CatalogFixtureError("roots_present_in_live_catalog must be True")
    # by_norm values are LISTS of uuids (SPEC §4.2).
    for norm, uuids in by_norm.items():
        if not isinstance(uuids, list):
            raise CatalogFixtureError(f"by_norm[{norm!r}] must be a list, got {type(uuids)}")
        for k, u in enumerate(uuids):
            if not isinstance(u, str):
                raise CatalogFixtureError(
                    f"by_norm[{norm!r}][{k}] must be a str uuid (got {type(u).__name__})"
                )
            if u not in by_uuid:
                raise CatalogFixtureError(
                    f"by_norm[{norm!r}] references uuid {u!r} "
                    f"absent from catalog_by_uuid"
                )
    # The three realm roots present by is_root name-membership (SPEC §4.3).
    root_names = {
        rec.get("name")
        for rec in by_uuid.values()
        if rec.get("is_root") is True
    }
    for root in _REALM_ROOTS:
        if root not in root_names:
            raise CatalogFixtureError(f"realm root {root!r} absent from catalog (R6)")
        if root not in by_norm or not by_norm[root]:
            raise CatalogFixtureError(f"realm root {root!r} missing from by_norm")


def freeze_catalog(catalog: dict[str, Any], path: Path) -> None:
    """Validate then write catalog deterministically (sorted keys).

    List content is canonicalized too: `by_norm` uuid lists are set-like, so
    they are sorted before write to keep the frozen bytes independent of
    live-return order. Record-level `chain`/`ancestors` lists are root->node
    PATHS: their order is semantic and already deterministic given the
    hierarchy, so they are intentionally NOT sorted.
    """
    validate_catalog(catalog)
    envelope = {
        "version": FIXTURE_VERSION,
        "catalog_by_uuid": catalog["catalog_by_uuid"],
        "by_norm": {
            norm: sorted(uuids) for norm, uuids in catalog["by_norm"].items()
        },
        "roots_present_in_live_catalog": catalog["roots_present_in_live_catalog"],
    }
    _write_deterministic(envelope, path)


def load_catalog(path: Path) -> dict[str, Any]:
    """Load + validate catalog.json; enforce the vehicle-fragments invariant."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != FIXTURE_VERSION:
        raise CatalogFixtureError(
            f"catalog fixture version mismatch: {raw.get('version')!r}"
        )
    validate_catalog(raw)
    # SPEC §4.2: the known fragments case MUST be exercised, else eval
    # criterion (c) is silently un-exercised.
    veh = raw["by_norm"].get("vehicle", [])
    if len(veh) <= 1:
        raise CatalogFixtureError(
            f"by_norm['vehicle'] must have >1 uuid (fragments case), got {veh!r}"
        )
    return raw
