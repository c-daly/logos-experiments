"""Simulate-only placement cascade for the naming-driven-typing experiment.

Takes the validated /type-cluster groups (T4) + the enriched catalog (T2) and
produces one PlacementRecord per group: branch (G1 reuse / G2 graft / G3 root /
residual), resolved graft parent, floor/ceiling gate outcomes, eviction proxy,
and an event log. Implements SPEC §5.1-§5.7, §5.12, §5.14.

*** NO graph writes, no LLM calls, no network, no Neo4j/Milvus/Redis. ***
Every [WRITE-PATH] action (retype / mint / edge / centroid) is RECORDED in the
PlacementRecord, never executed. The resolver is deterministic so the harness
can assert a twice-run identical proxy (SPEC §5.11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Shared canonicalize() from hermes.canonical (T1). Fall back to a minimal
# local shim if hermes is not installed in this env -- keeps the
# simulator standalone-testable; the experiment run env has hermes (path A).
try:  # pragma: no cover - import wiring
    from hermes.canonical import canonicalize  # type: ignore
except Exception:  # pragma: no cover - shim path

    def canonicalize(name: str) -> str:
        return " ".join(name.strip().lower().split())


# ---- tunable params (surfaced as [METRIC] params by the harness) --------

MIN_DEPTH: int = 2          # FLOOR: covering_depth = len(chain)-1 must be >=
MAX_WORDS: int = 3          # CEILING word-count trigger
EVICT_LEVEL_DROP: int = 2   # G-EVICT-1 default N

CONJUNCTION_TOKENS: frozenset[str] = frozenset(
    {"and", "or", "&", "/", "related", "feature-of", "part-of", "with"}
)

# assign_to resolving to any of these => NEVER reuse; coerce NEW + graft-under.
PROTECTED_ROOT_NAMES: frozenset[str] = frozenset(
    {"entity", "concept", "process", "root", "node", "cognition"}
)

# chain[-1] whitelist (G-CW-7). G3 emergence terminal is 'entity' only, but the
# whitelist also admits concept/process for catalog-published roots.
VALID_TERMINAL_ROOTS: frozenset[str] = frozenset({"entity", "concept", "process"})


@dataclass(frozen=True)
class PlacementRecord:
    """One group's simulated placement. Frozen, with tuple
    collection fields, so records are deeply immutable (determinism)."""

    cluster_id: str
    branch: str  # G1_REUSE | G2_GRAFT | G3_ROOT | RESIDUAL
    assign_to: str
    name: str
    chain: tuple[str, ...]
    member_ids: tuple[str, ...]
    resolved_parent_uuid: Optional[str]
    resolved_parent_name: Optional[str]
    covering_depth: int
    floor_ok: bool
    ceiling_ok: bool
    over_specified: bool
    evicted_ids: tuple[str, ...] = ()
    residual_ids: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    self_reported: bool = True
    # Re-parent bookkeeping (2026-06-06 contract): placement is a re-parent --
    # the kept subgraph drops its existing IS_A edge and gains one to the chosen
    # parent. `minted` is True when a NEW type node was created (mint), False
    # when the subgraph attaches under an existing type (reuse). `chain` is
    # always () now -- the IS_A structure lives in the graph, never rebuilt.
    minted: bool = False
    removed_parent_uuid: Optional[str] = None


# ---- gates --------------------------------------------------------------

def floor_ok(chain: list[str], min_depth: int = MIN_DEPTH) -> bool:
    """covering_depth = len(chain) - 1; pass iff >= min_depth (SPEC §5.5)."""
    covering_depth = max(len(chain) - 1, 0)
    return covering_depth >= min_depth


def ceiling_violation(name: str) -> bool:
    """word_count(name) > MAX_WORDS OR a whole-word conjunction (SPEC §5.5)."""
    words = name.split()
    if len(words) > MAX_WORDS:
        return True
    tokens = set(re.split(r"[\s]+", name.lower().strip()))
    # treat '/' and '&' as their own tokens even without surrounding spaces
    if "/" in name or "&" in name:
        return True
    return bool(tokens & CONJUNCTION_TOKENS)


# ---- resolver -----------------------------------------------------------

def _disambiguate(uuids: list[str], catalog_by_uuid: dict[str, dict]) -> str:
    """Deterministic multi-match pick: most member_count -> shallowest
    ancestors -> smallest uuid (SPEC §5.3)."""

    def sort_key(u: str) -> tuple[int, int, str]:
        rec = catalog_by_uuid[u]
        # negate member_count so 'most' sorts first under ascending sort
        return (-int(rec.get("member_count", 0)), len(rec.get("ancestors", [])), u)

    return min(uuids, key=sort_key)


def _is_protected_root(uuid: str, catalog_by_uuid: dict[str, dict]) -> bool:
    rec = catalog_by_uuid.get(uuid)
    if rec is None:
        return False
    return bool(rec.get("is_root")) or rec.get("name") in PROTECTED_ROOT_NAMES


def resolve_deepest_ancestor(
    chain: list[str],
    catalog_by_uuid: dict[str, dict],
    by_norm: dict[str, list[str]],
    minted_names_this_pass: Optional[set[str]] = None,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Walk chain[1:] specific-first; return the first element that matches a
    published catalog node by normalized name (SPEC §5.2). Multi-match resolved
    deterministically (§5.3). Skips names minted THIS pass (freeze-snapshot,
    §5.12). Root-only => the root uuid. No root at all => default entity +
    CHAIN_NO_ROOT.
    """
    events: list[str] = []
    minted = minted_names_this_pass or set()
    # chain[1:] = candidate ancestors (skip chain[0], the type being minted).
    for element in chain[1:]:
        norm = canonicalize(element)
        if not norm:
            continue
        if norm in minted:
            # minted this pass -> not yet a published graft target (§5.12)
            events.append(f"FREEZE_SKIP:{norm}")
            continue
        candidates = by_norm.get(norm)
        if candidates:
            chosen = _disambiguate(list(candidates), catalog_by_uuid)
            return chosen, catalog_by_uuid[chosen]["name"], events
    # nothing matched (or only minted-this-pass). Find a terminal root.
    terminal = canonicalize(chain[-1]) if chain else ""
    if terminal not in VALID_TERMINAL_ROOTS:
        events.append("CHAIN_NO_ROOT")
        terminal = "entity"
    root_uuids = by_norm.get(terminal, [])
    if root_uuids:
        chosen = _disambiguate(list(root_uuids), catalog_by_uuid)
        return chosen, catalog_by_uuid[chosen]["name"], events
    # catalog has no such root published (should not happen: T2 asserts roots)
    events.append("ROOT_MISSING")
    return None, terminal, events


# ---- eviction proxy -----------------------------------------------------

def simulate_eviction(
    member_ids: list[str],
    chain: list[str],
    catalog_by_uuid: dict[str, dict],
    by_norm: dict[str, list[str]],
    level_drop: int = EVICT_LEVEL_DROP,
) -> tuple[list[str], list[str]]:
    """Offline eviction proxy (SPEC §5.14, G-EVICT-1).

    The offline harness has no per-member covering-depth signal (that is the
    namer's judgment on the live members), so it is CONSERVATIVE: evict
    nothing, return (all_kept, []). The write-path enforces real eviction.
    """
    return list(member_ids), []


# ---- per-group simulation ----------------------------------------------

def simulate_group(
    group: dict,
    input_ids: set[str],
    catalog_by_uuid: dict[str, dict],
    by_norm: dict[str, list[str]],
    min_depth: int = MIN_DEPTH,
    minted_names_this_pass: Optional[set[str]] = None,
    enforce_ceiling: bool = True,
) -> PlacementRecord:
    """Place one group; record-only, no graph writes (SPEC §5.1)."""
    events: list[str] = []
    assign_to = str(group.get("assign_to", "NEW"))
    name = str(group.get("name", ""))
    cluster_id = str(group.get("cluster_id", name))
    chain = list(group.get("chain", []))
    # member_ids restricted to this request's input (closed-world).
    member_ids = [m for m in group.get("member_ids", []) if m in input_ids]
    over_specified = bool(group.get("over_specified", False))
    covering_depth = max(len(chain) - 1, 0)

    f_ok = floor_ok(chain, min_depth=min_depth)
    # A5/no-gate seam: CEILING off => c_ok is vacuously True (FLOOR is
    # disabled by min_depth=0, since covering_depth >= 0 always holds).
    c_ok = not ceiling_violation(name) if enforce_ceiling else True

    def residual(extra_events: list[str]) -> PlacementRecord:
        return PlacementRecord(
            cluster_id=cluster_id,
            branch="RESIDUAL",
            assign_to=assign_to,
            name=name,
            chain=tuple(chain),
            member_ids=tuple(member_ids),
            resolved_parent_uuid=None,
            resolved_parent_name=None,
            covering_depth=covering_depth,
            floor_ok=f_ok,
            ceiling_ok=c_ok,
            over_specified=over_specified,
            evicted_ids=(),
            residual_ids=tuple(member_ids),
            events=tuple(events + extra_events),
        )

    # FLOOR / CEILING gates first: a gated group never mints (§5.5).
    if not f_ok:
        return residual(["FLOOR_VIOLATION"])
    if not c_ok:
        # offline: split is recorded as residual (re-segment is a [WRITE-PATH]).
        return residual(["CEILING_VIOLATION_SPLIT_DEFERRED"])

    # G1 -- reuse-node.
    if assign_to != "NEW":
        if assign_to in catalog_by_uuid:
            if _is_protected_root(assign_to, catalog_by_uuid):
                # G-CW-PROTECTED: never reuse a root; coerce NEW + graft-under.
                events.append(f"PROTECTED_ROOT_COERCE_NEW:{assign_to}")
                # fall through to G2/G3 below
            else:
                return PlacementRecord(
                    cluster_id=cluster_id,
                    branch="G1_REUSE",
                    assign_to=assign_to,
                    name=name,
                    chain=tuple(chain),
                    member_ids=tuple(member_ids),
                    resolved_parent_uuid=assign_to,
                    resolved_parent_name=catalog_by_uuid[assign_to]["name"],
                    covering_depth=covering_depth,
                    floor_ok=f_ok,
                    ceiling_ok=c_ok,
                    over_specified=over_specified,
                    evicted_ids=(),
                    residual_ids=(),
                    events=tuple(events),
                )
        else:
            events.append(f"UNRESOLVED_ASSIGN_TO:{assign_to}")
            # fall through to NEW handling

    # G2 / G3 -- mint. Resolve graft parent (deepest published ancestor).
    parent_uuid, parent_name, resolve_events = resolve_deepest_ancestor(
        chain, catalog_by_uuid, by_norm, minted_names_this_pass
    )
    events += resolve_events

    # Was the resolved parent a non-root published node (G2) or a root (G3)?
    is_root_parent = parent_uuid is not None and _is_protected_root(
        parent_uuid, catalog_by_uuid
    )
    if parent_uuid is None or is_root_parent:
        branch = "G3_ROOT"
        # G-CW-7: terminal must be a whitelist root; otherwise BAD_ROOT->entity.
        terminal = canonicalize(chain[-1]) if chain else ""
        if terminal not in VALID_TERMINAL_ROOTS:
            events.append(f"BAD_ROOT:{terminal}->entity")
    else:
        branch = "G2_GRAFT"

    kept, evicted = simulate_eviction(
        member_ids, chain, catalog_by_uuid, by_norm
    )

    return PlacementRecord(
        cluster_id=cluster_id,
        branch=branch,
        assign_to=assign_to,
        name=name,
        chain=tuple(chain),
        member_ids=tuple(kept),
        resolved_parent_uuid=parent_uuid,
        resolved_parent_name=parent_name,
        covering_depth=covering_depth,
        floor_ok=f_ok,
        ceiling_ok=c_ok,
        over_specified=over_specified,
        evicted_ids=tuple(evicted),
        residual_ids=tuple(evicted),
        events=tuple(events),
    )


# ---- full pass ----------------------------------------------------------

def _depth_via_catalog(uuid: Optional[str], catalog_by_uuid: dict[str, dict]) -> int:
    """Number of IS_A hops from `uuid` to the top of the existing structure.

    A READ of structure that already exists in the catalog (mirrors the graph's
    parent_uuid spine) -- never a reconstructed/stored chain (2026-06-06).
    """
    depth = 0
    seen: set[str] = set()
    cur = uuid
    while cur and cur in catalog_by_uuid and cur not in seen:
        seen.add(cur)
        cur = catalog_by_uuid[cur].get("parent_uuid")
        depth += 1
    return depth


def _root_target(
    by_norm: dict[str, list[str]], catalog_by_uuid: dict[str, dict], realm: str = "entity"
) -> tuple[Optional[str], str]:
    uuids = by_norm.get(realm)
    if uuids:
        chosen = _disambiguate(list(uuids), catalog_by_uuid)
        return chosen, catalog_by_uuid[chosen]["name"]
    return None, realm


def simulate_cluster_placement(
    *,
    cluster_id: str,
    member_ids: list[str],
    name: str,
    parent: Optional[str],
    residual_ids: list[str],
    catalog_by_uuid: dict[str, dict],
    by_norm: dict[str, list[str]],
    current_parent_uuid: Optional[str] = None,
    enforce_ceiling: bool = True,
) -> PlacementRecord:
    """Place ONE named cluster (2026-06-06 contract). Record-only, no writes.

    The namer returns {name, parent, outliers}; placement is a RE-PARENT of the
    kept subgraph (members - outliers): drop its existing IS_A edge
    (`removed_parent_uuid`), add one to the chosen parent (`resolved_parent_uuid`).
    parent=None => REUSE `name` as an existing type (attach under it, no new
    node). parent set => MINT `name` under that existing parent (new node).
    Attaching/minting under a domain root is always legal. No chain is built --
    ancestry already exists in the graph; depth is a catalog read.
    """
    events: list[str] = []
    canon = canonicalize(name)
    over_specified = ceiling_violation(name)
    residual_set = {m for m in residual_ids if m in set(member_ids)}
    kept = [m for m in member_ids if m not in residual_set]

    if not kept:
        return PlacementRecord(
            cluster_id=cluster_id, branch="RESIDUAL", assign_to="NEW", name=canon,
            chain=(), member_ids=(), resolved_parent_uuid=None,
            resolved_parent_name=None, covering_depth=0, floor_ok=True,
            ceiling_ok=not over_specified, over_specified=over_specified,
            residual_ids=tuple(sorted(residual_set)),
            events=tuple(events + ["ALL_MEMBERS_RESIDUAL"]),
            minted=False, removed_parent_uuid=current_parent_uuid,
        )

    if parent is None:
        # REUSE: `name` should match an existing type; attach the subgraph there.
        uuids = by_norm.get(canon)
        if uuids:
            target = _disambiguate(list(uuids), catalog_by_uuid)
            if _is_protected_root(target, catalog_by_uuid):
                events.append("REUSE_ROOT_COERCE_MINT")
                parent_uuid, parent_name = target, catalog_by_uuid[target]["name"]
                branch, minted, assign_to, placed = "G3_ROOT", True, "NEW", canon
            else:
                parent_uuid = target
                parent_name = catalog_by_uuid[target]["name"]
                branch, minted, assign_to, placed = "G1_REUSE", False, target, parent_name
        else:
            events.append("REUSE_UNRESOLVED_MINT_ROOT")
            parent_uuid, parent_name = _root_target(by_norm, catalog_by_uuid)
            branch, minted, assign_to, placed = "G3_ROOT", True, "NEW", canon
    else:
        # MINT `name` under an existing parent.
        canon_parent = canonicalize(parent)
        puuids = by_norm.get(canon_parent)
        if puuids:
            parent_uuid = _disambiguate(list(puuids), catalog_by_uuid)
            parent_name = catalog_by_uuid[parent_uuid]["name"]
        else:
            events.append("PARENT_UNRESOLVED:" + str(parent))
            parent_uuid, parent_name = _root_target(by_norm, catalog_by_uuid)
        is_root = bool(catalog_by_uuid.get(parent_uuid or "", {}).get("is_root")) or (
            parent_uuid is not None and _is_protected_root(parent_uuid, catalog_by_uuid)
        )
        branch = "G3_ROOT" if is_root else "G2_GRAFT"
        minted, assign_to, placed = True, "NEW", canon

    covering_depth = _depth_via_catalog(parent_uuid, catalog_by_uuid) + 1
    return PlacementRecord(
        cluster_id=cluster_id, branch=branch, assign_to=assign_to, name=placed,
        chain=(), member_ids=tuple(kept), resolved_parent_uuid=parent_uuid,
        resolved_parent_name=parent_name, covering_depth=covering_depth,
        floor_ok=True,  # re-parent under an existing type/root is always legal
        ceiling_ok=(not over_specified) if enforce_ceiling else True,
        over_specified=over_specified, residual_ids=tuple(sorted(residual_set)),
        events=tuple(events), minted=minted, removed_parent_uuid=current_parent_uuid,
    )


def simulate_cascade(
    groups: list[dict],
    catalog_by_uuid: dict[str, dict],
    by_norm: dict[str, list[str]],
    minted_names_this_pass: Optional[set[str]] = None,
    min_depth: int = MIN_DEPTH,
    enforce_ceiling: bool = True,
) -> list[PlacementRecord]:
    """Run the cascade over a full /type-cluster response (SPEC §5.1, §5.12).

    Freeze-snapshot guard: a type minted during this pass is NOT a graft target
    in the same pass. Two-pass derivation: FLOOR/CEILING verdicts depend only
    on (chain, name), never on the minted set, so gates are evaluated first and
    minted is built ONLY from groups that pass them AND mint (assign_to == NEW
    or unresolvable). A gated-out group never mints (SPEC 5.5, 5.12), so its
    name cannot block sibling graft targets. Deterministic: pure function
    of (groups, catalog), so a twice-run proxy is identical (§5.11).
    """
    input_ids = {
        m
        for g in groups
        for m in g.get("member_ids", [])
    }
    if minted_names_this_pass is None:
        minted: set[str] = set()
        for g in groups:
            # Gates first (SPEC 5.5): a FLOOR/CEILING-gated group routes to
            # RESIDUAL and never mints, so its name must not block sibling
            # graft targets (SPEC 5.12).
            if not floor_ok(list(g.get("chain", [])), min_depth=min_depth):
                continue
            if enforce_ceiling and ceiling_violation(str(g.get("name", ""))):
                continue
            at = str(g.get("assign_to", "NEW"))
            mints = at == "NEW" or at not in catalog_by_uuid
            if mints:
                minted.add(canonicalize(str(g.get("name", ""))))
    else:
        minted = set(minted_names_this_pass)

    records: list[PlacementRecord] = []
    for group in groups:
        records.append(
            simulate_group(
                group,
                input_ids,
                catalog_by_uuid,
                by_norm,
                min_depth=min_depth,
                minted_names_this_pass=minted,
                enforce_ceiling=enforce_ceiling,
            )
        )
    return records
