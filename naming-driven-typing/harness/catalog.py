"""Enriched, read-only catalog builder for the naming-driven typing harness.

Builds the harness\u0027s OWN closed-world catalog from the live graph \u2014 it does
NOT read the flat production Redis snapshot for structure and NEVER writes
production (SPEC \u00a74.2). Fixes two production-snapshot blockers:

  * uuid-keyed (not name-keyed) so same-name twins both survive (\u00a74.1).
  * member_count via the type_uuid pull, not the uniformly-0 node prop.

Roots (entity/concept/process) are present by NAME-membership regardless of
is_type_definition flags (\u00a74.3); a missing root fails loudly.

Two 2026-06-05 corrections govern this module:

  * ``parent_uuid`` is derived from the reified IS_A edge walk
    ``(t)<-[:FROM]-(e {relation:\u0027IS_A\u0027})-[:TO]->(parent)`` \u2014 the
    ``ancestors``/``chain``/``depth`` fields are NOT stored; any chain is
    reconstructed by following ``parent_uuid`` upward (\u00a74.4 corrected).
  * Stored names are ASSERTED canonical (``norm_name == name``); drift fails
    loudly with ``non_canonical_names_in_catalog`` \u2014 never a silent repair
    (canonical-at-the-boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# The three realm roots, present unconditionally by name-membership (\u00a74.3).
REALM_ROOTS: frozenset[str] = frozenset({"entity", "concept", "process"})


class RootMissingError(Exception):
    """A realm root is absent from the live catalog (\u00a74.3 \u2014 fail loudly)."""


class NonCanonicalNameError(Exception):
    """A stored type name is not canonical (boundary drift \u2014 fail loudly)."""


@dataclass
class CatalogResult:
    """The enriched, read-only catalog and its derived indices."""

    catalog_by_uuid: dict[str, dict[str, Any]]
    by_norm: dict[str, list[str]]
    roots_present_in_live_catalog: bool
    root_uuids: dict[str, str]


def build_catalog(
    type_defs: list[dict[str, Any]],
    member_counter: Callable[[str], int],
    is_a_edges: list[dict[str, Any]],
    *,
    canonicalize: Callable[[str], str],
) -> CatalogResult:
    """Build the enriched catalog from raw type-def rows + IS_A edge rows.

    Args:
        type_defs: rows as returned by ``HCGClient.get_all_type_definitions``
            \u2014 each ``{"uuid": str, "name": str, "properties": dict}``.
        member_counter: ``uuid -> member_count``; the caller pulls live
            membership via ``get_nodes_by_type_uuid`` (the props value is 0).
        is_a_edges: reified IS_A edge rows (``{"source", "target",
            "relation", ...}``) \u2014 the read-only edge walk flattened to
            ``source -> target`` pairs. Edges whose endpoints are not both
            catalog uuids (instance membership edges) and non-IS_A relations
            are ignored.
        canonicalize: the SHARED ``hermes.canonical.canonicalize`` (T1).

    Returns:
        A :class:`CatalogResult`.

    Raises:
        NonCanonicalNameError: if any stored name is not canonical
            (``norm_name != name``) \u2014 never silently repaired.
        RootMissingError: if any of entity/concept/process is absent.
    """
    catalog_uuids = {row["uuid"] for row in type_defs}

    # parent_uuid from the reified IS_A edge walk (\u00a74.4 corrected): only
    # IS_A edges BETWEEN catalog uuids count (instance->type membership
    # edges are not type placement). Deterministic: smallest target uuid
    # wins hypothetical duplicate-edge ties.
    parent_by_uuid: dict[str, str] = {}
    for edge in is_a_edges:
        if edge.get("relation") != "IS_A":
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source in catalog_uuids and target in catalog_uuids:
            if source not in parent_by_uuid or target < parent_by_uuid[source]:
                parent_by_uuid[source] = target

    catalog_by_uuid: dict[str, dict[str, Any]] = {}
    non_canonical: list[tuple[str, str, str]] = []
    for row in type_defs:
        uuid = row["uuid"]
        name = row["name"]
        norm_name = canonicalize(name)
        if norm_name != name:
            non_canonical.append((uuid, name, norm_name))
        catalog_by_uuid[uuid] = {
            "uuid": uuid,
            "name": name,
            "norm_name": norm_name,
            "member_count": int(member_counter(uuid)),
            "parent_uuid": parent_by_uuid.get(uuid),
            "is_root": name in REALM_ROOTS,
        }

    # Canonical-at-the-boundary: stored names must ALREADY be canonical.
    # Drift is a boundary violation \u2014 fail loudly, never silently repair.
    if non_canonical:
        offenders = ", ".join(
            f"{uuid}: {name!r} -> {norm!r}"
            for uuid, name, norm in sorted(non_canonical)
        )
        raise NonCanonicalNameError(
            f"non_canonical_names_in_catalog={len(non_canonical)} \u2014 stored "
            f"type names must already be canonical; offenders: {offenders}"
        )

    # by_norm: multi-valued, built from EVERY uuid (\u00a74.2). Deterministic
    # ordering (sorted uuids) for idempotent disambiguation downstream.
    by_norm: dict[str, list[str]] = {}
    for uuid, entry in catalog_by_uuid.items():
        by_norm.setdefault(entry["norm_name"], []).append(uuid)
    for norm in by_norm:
        by_norm[norm] = sorted(by_norm[norm])

    # Roots present by name-membership, regardless of flags (\u00a74.3).
    root_uuids: dict[str, str] = {}
    for entry in catalog_by_uuid.values():
        if entry["is_root"] and entry["name"] not in root_uuids:
            root_uuids[entry["name"]] = entry["uuid"]
    missing = sorted(REALM_ROOTS - set(root_uuids))
    if missing:
        raise RootMissingError(
            "realm root(s) absent from live catalog: "
            + ", ".join(missing)
            + " (a root-less catalog measures graft-to-nothing, SPEC \u00a74.3)"
        )

    return CatalogResult(
        catalog_by_uuid=catalog_by_uuid,
        by_norm=by_norm,
        roots_present_in_live_catalog=True,
        root_uuids={k: root_uuids[k] for k in sorted(root_uuids)},
    )


def build_catalog_from_client(
    client: Any,
    *,
    canonicalize: Callable[[str], str] | None = None,
) -> CatalogResult:
    """Build the catalog from a live, READ-ONLY HCGClient.

    Pulls all type-def rows and the reified IS_A edges (\u00a74.4 corrected \u2014
    one read-only edge pull, no stored ancestors), then counts members per
    type via the post-#505 ``type_uuid`` property (``get_nodes_by_type_uuid``);
    the base ``entity`` junk-drawer is counted by node-type scan
    (``list_all_nodes``), mirroring ``emergence_handler._member_rows``.
    Performs no writes.

    Args:
        client: an HCGClient exposing ``get_all_type_definitions``,
            ``list_all_edges``, ``get_nodes_by_type_uuid``, and
            ``list_all_nodes`` (reads only).
        canonicalize: override for the shared normalizer; defaults to
            ``hermes.canonical.canonicalize`` (T1), imported LAZILY so the
            offline tests never need hermes installed.
    """
    if canonicalize is None:
        from hermes.canonical import canonicalize as _canonicalize

        canonicalize = _canonicalize

    type_defs = client.get_all_type_definitions()
    name_by_uuid = {row["uuid"]: row["name"] for row in type_defs}
    is_a_edges = client.list_all_edges(relation_type="IS_A") or []

    def member_counter(uuid: str) -> int:
        # The base junk-drawer is resolved by node-type scan; any minted
        # type by its authoritative type_uuid property (\u00a74.2, mirrors
        # emergence_handler._member_rows).
        if name_by_uuid.get(uuid) == "entity":
            return len(client.list_all_nodes(node_type="entity") or [])
        rows = client.get_nodes_by_type_uuid(uuid) or []
        return len([n for n in rows if n and "uuid" in n])

    return build_catalog(
        type_defs, member_counter, is_a_edges, canonicalize=canonicalize
    )
