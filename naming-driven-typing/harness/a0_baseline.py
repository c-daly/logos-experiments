"""A0 measured rollup-baseline driver (SPEC 7.4 A0; logos-experiments#12).

Runs the REAL sophia production typing pipeline -- emergence minting
(``EmergenceHandler`` + ``mint_type``) followed by the type rollup
(``TypeRollupHandler``) -- over the SAME frozen fixtures the v2 arms use
(``fixtures/clusters.json`` + ``fixtures/catalog.json``), against a
DISPOSABLE namespaced subgraph of a live Neo4j, then READS the resulting
graph back and emits a snapshot in the exact T6 schema
(``run_experiment.build_snapshot``), so ``eval/metrics.py``
``ablation_deltas`` compares the v2 arms against what production actually
does today -- measured, not asserted. No placement logic is
reimplemented: every mint, reconcile, re-parent and graft below is
unmodified production code; this module only seeds inputs, runs the
pipeline, and reads the graph back.

Graph outcome -> cascade branch mapping
=======================================
Per-cluster ``cascade.branches`` records reuse the T5 vocabulary and the
exact ``PlacementRecord`` field names (key parity is asserted):

``G1_REUSE``
    The cluster members were retyped onto a type-def that existed BEFORE
    the pass (a seeded published catalog type): emergence
    match-before-mint reconciled the cluster into published structure
    instead of minting. ``assign_to`` / ``resolved_parent_uuid`` carry
    the FROZEN catalog uuid of the reused type.
``G2_GRAFT``
    A NEW type-def was minted for the cluster and its final IS_A chain
    crosses a published (pre-seeded, non-root) catalog type: the
    pipeline grafted the mint under published structure.
    ``resolved_parent_*`` is the nearest published ancestor (frozen
    catalog uuid); ``covering_depth`` = len(chain) - 1.
``G3_ROOT``
    A NEW type-def was minted but its final chain reaches a realm root
    (``entity`` / ``concept`` / ``process``) without crossing any
    published type: the mint floated at the root -- the flat-shelf
    outcome. Same-pass super-types minted by the rollup add chain depth
    but are NOT published structure, so they never turn a G3 into a G2.
residual (no branch record)
    Members whose authoritative ``type_uuid`` still points at the
    disposable junk-drawer seed after the pass were never typed by
    production (below min cluster size, low namer confidence, or namer
    outliers). They are recorded in ``cascade.residual_ids`` only,
    mirroring the v2 endpoint-residual convention -- the fragmentation
    signal.

Vacuous fields. ``raw_partition_ok`` is always True and
``sample_coverage`` always 1.0: the production pipeline consumes whole
clusters and has no LLM partition contract, so neither quantity is
measured here -- do not read them as a baseline result. Likewise
``floor_ok`` / ``ceiling_ok`` are recorded True on every branch because
production applies no FLOOR/CEILING naming gates.

K = 1 repeats. The production pipeline is deterministic given a fixed
graph and the deterministic namer/embedding seams below (the only
randomness, the uuid4 suffix in minted uuids, does not affect
structure), so the snapshot carries a single repeat; metrics aggregation
handles n = 1 and the K-repeat noise band comes from the v2 arms.

Seams (mirroring sophia tests/integration/test_r1_graft_invariants.py):
Neo4j is LIVE -- every node write, IS_A swing and member retype goes
through the production HCGClient write path. Milvus and the Hermes namer
are in-process fakes injected through the same constructor seams the
production builders use: ``FakeMilvus`` serves member vectors and minted
centroids; the namer labels a cluster by its dominant centroid axis and
embeds the namespace token in every label so minted uuids inherit it
(teardown safety). The namer never proposes a covering ``parent`` (the
production namer may), so the rollup path that roots a super-type under
a named closest cover is unreachable in this baseline -- a documented
namer-seam limitation, not a measured production verdict.

Embedding geometry. The frozen fixtures carry member NAMES only, so the
driver synthesizes fixture-grounded vectors: each frozen cluster gets
its own axis (members jittered apart), published catalog types sharing a
normalized name share an axis (offset apart), and every main axis
carries a distinct tie-break epsilon (a Sidon sequence, so all pairwise
inter-group distances are distinct and agglomeration never tie-breaks on
graph read order). Under this geometry semantic-affinity reuse of
published types is unreachable (no fixture signal links cluster members
to catalog names); the baseline measures the STRUCTURAL placement
behavior of the pipeline, which is what SPEC 7.4 A0 gates on (root
distribution, graft depth, fragmentation).

Config. Mirrors the R1 suite (variance_threshold=0.05,
type_match_threshold=0.9, hermes_confidence_floor=0.5,
rollup_min_cluster_size=2, rollup_min_supercluster_size=2) except
``min_cluster_size=2``: production caps the partition search at
k_max = n_members // min_cluster_size, and with the 8-member fixture
drawer a floor of 3 caps k at 2, forcing frozen clusters to merge as a
fixture-size artifact. The floor of 2 keeps every frozen cluster
expressible; all placement logic is untouched.

Non-mutation discipline. Every seeded uuid embeds a per-run namespace
token; the rollup is read-side filtered to the token (write paths remain
production code); teardown DETACH-DELETEs every node carrying the token
and then VERIFIES zero residue; the (uuid -> ancestors) map of type-defs
outside the namespace must be unchanged by the pass; the shared
``type_entity`` realm root is created only when absent and removed again
only if this run created it. A breach raises BEFORE any snapshot is
persisted.

Run (env-gated; refuses to run without the explicit opt-in):
    A0_LIVE=1 uv run --no-sync python harness/a0_baseline.py --out workspace/

Stack env (URI/user default; the credential is mandatory):
    NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j
    NEO4J_PASSWORD=<required -- no default credential>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FIXTURES = EXP / "fixtures"
WORKSPACE = EXP / "workspace"

# Direct-script execution shim (same role as in harness/run_experiment.py);
# harness.* imports below stay deferred so this module never trips E402 and
# stays importable both as a script and as a package module.
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

ABLATION = "rollup_baseline"
_MODEL_LABEL = "sophia-production-pipeline"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_ROOT_MARKER = "a0-baseline-fixture"

# Canonical realm-root uuids (the seeded top of the ontology); the chain walk
# in map_outcomes terminates on these.
_REALM_ROOT_UUIDS = {
    "type_entity": "entity",
    "type_concept": "concept",
    "type_process": "process",
}

# Mian-Chowla (Sidon) sequence: pairwise differences are all distinct, so the
# per-axis tie-break epsilons never produce equal inter-group distances
# (equal distances would let agglomeration tie-break on graph read order,
# the one ordering seam a live graph does not pin down).
# Greedy Mian-Chowla (B2/Sidon) sequence, extended on demand: all pairwise
# differences distinct, which keeps the epsilon tie-breaks collision-free.
# The first 11 terms are the original hand-rolled table; the graded corpus
# (142 clusters + ~220 published norms) blew past it (#18).
_SIDON = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96]
_EPS_CEILING = 0.96  # max epsilon: keep tie-breaks well under the unit axis


def _extend_sidon(upto: int) -> None:
    """Grow _SIDON greedily until it has at least ``upto + 1`` terms."""
    s = _SIDON
    sums = {a + b for i, a in enumerate(s) for b in s[i:]}
    while len(s) <= upto:
        cand = s[-1] + 1
        while True:
            new_sums = {cand + a for a in s}
            new_sums.add(2 * cand)
            if len(new_sums) == len(s) + 1 and sums.isdisjoint(new_sums):
                break
            cand += 1
        s.append(cand)
        sums |= new_sums
_MEMBER_JITTER = 0.01
_PUBLISHED_OFFSET = 0.05


class A0LiveGateError(RuntimeError):
    """The live driver was invoked without the explicit A0_LIVE=1 opt-in."""


class ZeroResidueViolation(RuntimeError):
    """Teardown left namespaced nodes behind (disposable-graph breach)."""


def _branch_keys() -> frozenset[str]:
    """The T6 branch schema IS the T5 PlacementRecord schema.

    Key parity is derived from the dataclass at call time so the branch
    records this driver emits can never drift from what the arms emit.
    """
    from harness.cascade import PlacementRecord  # deferred: path shim above

    return frozenset(f.name for f in fields(PlacementRecord))


# ------------------------------------------------------------------ geometry


def sidon_epsilon(index: int, *, scale: float) -> float:
    """Tie-break epsilon for main axis ``index`` (distinct pairwise gaps).

    ``scale`` comes from the plan: raw B2 terms grow ~quadratically (term
    365 is ~6.6e4), so a fixed multiplier would swamp the unit main axis at
    graded scale; the plan normalizes its LARGEST term to _EPS_CEILING and
    common scaling preserves gap distinctness.
    """
    _extend_sidon(index)
    return _SIDON[index] * scale


def _norm_key(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def geometry_plan(clusters: list[dict], catalog: dict) -> dict[str, Any]:
    """Pure axis plan for the synthetic vectors (see module docstring).

    Frozen clusters get one main axis each (fixture order); published
    non-root catalog types grouped by normalized name share one main axis
    per group (sorted for determinism). Three service axes follow: the
    published in-group offset axis, the member jitter axis, and the
    tie-break epsilon axis.
    """
    cluster_axis = {str(c["cluster_id"]): i for i, c in enumerate(clusters)}
    norms = sorted(
        {
            _norm_key(rec.get("norm_name") or rec.get("name", ""))
            for rec in catalog.get("catalog_by_uuid", {}).values()
            if not rec.get("is_root")
        }
    )
    publish_axis = {n: len(cluster_axis) + i for i, n in enumerate(norms)}
    n_main = len(cluster_axis) + len(publish_axis)
    _extend_sidon(max(n_main - 1, 0))
    eps_scale = _EPS_CEILING / max(_SIDON[max(n_main - 1, 0)], 1)
    return {
        "cluster_axis": cluster_axis,
        "publish_axis": publish_axis,
        "off_axis": n_main,
        "jitter_axis": n_main + 1,
        "eps_axis": n_main + 2,
        "dim": n_main + 3,
        "eps_scale": eps_scale,
    }


def axis_vector(
    plan: dict[str, Any],
    axis: int,
    *,
    jitter: float = 0.0,
    off: float = 0.0,
) -> list[float]:
    """Unit vector on a main axis + its epsilon/jitter/offset components."""
    vec = [0.0] * plan["dim"]
    vec[axis] = 1.0
    vec[plan["eps_axis"]] = sidon_epsilon(axis, scale=plan["eps_scale"])
    vec[plan["jitter_axis"]] = jitter
    vec[plan["off_axis"]] = off
    return vec


def member_vector(
    plan: dict[str, Any], cluster_id: str, position: int, count: int
) -> list[float]:
    """Vector for member ``position`` of ``count`` in a frozen cluster."""
    jitter = _MEMBER_JITTER * (position - (count - 1) / 2.0)
    return axis_vector(plan, plan["cluster_axis"][str(cluster_id)], jitter=jitter)


def published_vector(
    plan: dict[str, Any], norm: str, position: int, count: int
) -> list[float]:
    """Centroid vector for a published catalog type within its norm group."""
    off = _PUBLISHED_OFFSET * (position - (count - 1) / 2.0)
    return axis_vector(plan, plan["publish_axis"][norm], off=off)


# ------------------------------------------------------- published seed plan


def published_rows(catalog: dict, ns: str) -> list[dict[str, Any]]:
    """Pure seed plan for the published (non-root) catalog types.

    Catalog chains are root-FIRST with the type itself last. Each row maps
    a frozen catalog uuid to a namespaced live uuid, derives the seeded
    ``ancestors`` ([root, node] + chain-without-self) and resolves the
    live IS_A parent: the chain element just above the type when that
    element is itself a published catalog name, else the realm root the
    chain is rooted in. Rows are ordered parents-first (shallow chains
    first) so seeding can create each IS_A edge as it goes.
    """
    recs = [
        (cat_uuid, rec)
        for cat_uuid, rec in sorted(catalog.get("catalog_by_uuid", {}).items())
        if not rec.get("is_root")
    ]
    recs.sort(key=lambda kv: (len(kv[1].get("chain") or []), kv[0]))
    group_size: dict[str, int] = {}
    for _, rec in recs:
        norm = _norm_key(rec.get("norm_name") or rec.get("name", ""))
        group_size[norm] = group_size.get(norm, 0) + 1
    live_by_norm: dict[str, str] = {}
    group_seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, (cat_uuid, rec) in enumerate(recs):
        chain = [str(c) for c in (rec.get("chain") or [])]
        name = rec.get("name", "")
        norm = _norm_key(rec.get("norm_name") or name)
        live_uuid = f"type_{ns}pub{index}"
        parent_name = _norm_key(chain[-2]) if len(chain) >= 2 else "entity"
        if parent_name in live_by_norm:
            parent_uuid = live_by_norm[parent_name]
        elif f"type_{parent_name}" in _REALM_ROOT_UUIDS:
            parent_uuid = f"type_{parent_name}"
        else:
            # Fail closed to the chain realm root (chain is root-first).
            realm = _norm_key(chain[0]) if chain else "entity"
            parent_uuid = (
                f"type_{realm}"
                if f"type_{realm}" in _REALM_ROOT_UUIDS
                else "type_entity"
            )
        position = group_seen.get(norm, 0)
        group_seen[norm] = position + 1
        rows.append(
            {
                "live_uuid": live_uuid,
                "catalog_uuid": rec.get("uuid", cat_uuid),
                "name": name,
                "norm": norm,
                "ancestors": ["root", "node"] + chain[:-1],
                "parent_uuid": parent_uuid,
                "group_position": position,
                "group_size": group_size[norm],
            }
        )
        # First seeded uuid wins the norm key (deterministic; only used to
        # chain deeper published types under their published parent).
        live_by_norm.setdefault(norm, live_uuid)
    return rows


def realm_roots_from_catalog(catalog: dict) -> dict[str, dict]:
    """name -> frozen catalog record for the realm roots (``is_root``)."""
    return {
        _norm_key(rec.get("name", "")): {
            "uuid": rec.get("uuid", cat_uuid),
            "name": rec.get("name"),
        }
        for cat_uuid, rec in catalog.get("catalog_by_uuid", {}).items()
        if rec.get("is_root")
    }


# --------------------------------------------- graph outcome -> T6 cascade


def _walk_chain(
    leaf_uuid: str,
    *,
    seed_uuid: str,
    type_name_of: dict[str, str],
    parent_of: dict[str, Optional[str]],
    published: dict[str, dict],
) -> tuple[list[str], list[str], Optional[str]]:
    """Root-LAST name chain from a live leaf type-def, walking IS_A upward.

    The disposable junk-drawer seed is rewritten to ``entity`` (it stands
    in for the entity junk drawer; the raw uuid path is preserved for the
    events log). Returns (chain_names, raw_uuid_path, first_published_hit).
    Cycle-guarded so a corrupt adjacency cannot hang the read-back.
    """
    chain = [type_name_of.get(leaf_uuid, leaf_uuid)]
    raw = [leaf_uuid]
    published_hit: Optional[str] = None
    seen = {leaf_uuid}
    cur = leaf_uuid
    while True:
        parent = parent_of.get(cur)
        if parent is None or parent in seen:
            break
        seen.add(parent)
        raw.append(parent)
        if parent == seed_uuid:
            chain.append("entity")
            break
        if parent in _REALM_ROOT_UUIDS:
            chain.append(_REALM_ROOT_UUIDS[parent])
            break
        chain.append(type_name_of.get(parent, parent))
        if published_hit is None and parent in published:
            published_hit = parent
        cur = parent
    return chain, raw, published_hit


def _branch_record(**values: Any) -> dict[str, Any]:
    """Build one branch dict with EXACT PlacementRecord key parity.

    ``floor_ok`` / ``ceiling_ok`` are vacuous for the baseline (module
    docstring); ``self_reported`` is False because every value here is
    measured from the graph, not self-reported by a namer.
    """
    record: dict[str, Any] = {
        "floor_ok": True,
        "ceiling_ok": True,
        "over_specified": False,
        "evicted_ids": [],
        "residual_ids": [],
        "events": [],
        "self_reported": False,
        # A0 measures the production rollup, which mints types; it never
        # records the dropped old edge (re-parent bookkeeping is the v2
        # write-path's, #18).
        "minted": True,
        "removed_parent_uuid": None,
    }
    record.update(values)
    expected = _branch_keys()
    missing = expected - set(record)
    extra = set(record) - expected
    if missing or extra:
        raise ValueError(
            f"branch schema drift: missing={sorted(missing)} extra={sorted(extra)}"
        )
    return record


def map_outcomes(
    clusters: list[dict],
    *,
    ns: str,
    seed_uuid: str,
    member_type: dict[str, Optional[str]],
    type_name_of: dict[str, str],
    parent_of: dict[str, Optional[str]],
    published: dict[str, dict],
    realm_roots: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    """Pure mapping: graph read-back -> per-cluster T6 cascade dicts.

    ``member_type`` maps live member uuids to their final authoritative
    ``type_uuid``; ``published`` maps seeded live uuids to their frozen
    catalog records; ``realm_roots`` maps realm names to frozen catalog
    root records. Branch vocabulary: module docstring. A type serving
    members of more than one frozen cluster yields one branch per cluster
    (each carrying only that cluster member ids) plus an
    ``A0_TYPE_SPANS_CLUSTERS`` event.
    """
    type_clusters: dict[str, set[str]] = {}
    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        for member in cluster.get("members", []):
            member_id = str(member["id"])
            target = member_type.get(f"{ns}-{member_id}")
            if target and target != seed_uuid and target in type_name_of:
                type_clusters.setdefault(target, set()).add(cid)

    out: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        by_type: dict[str, list[str]] = {}
        residual: list[str] = []
        for member in cluster.get("members", []):
            member_id = str(member["id"])
            target = member_type.get(f"{ns}-{member_id}")
            if not target or target == seed_uuid or target not in type_name_of:
                residual.append(member_id)
                continue
            by_type.setdefault(target, []).append(member_id)
        branches: list[dict[str, Any]] = []
        for target, member_ids in by_type.items():
            chain, raw, published_hit = _walk_chain(
                target,
                seed_uuid=seed_uuid,
                type_name_of=type_name_of,
                parent_of=parent_of,
                published=published,
            )
            events = ["A0_RAW_IS_A:" + " -> ".join(raw)]
            if len(type_clusters.get(target, set())) > 1:
                events.append("A0_TYPE_SPANS_CLUSTERS")
            if target in published:
                cat = published[target]
                branches.append(
                    _branch_record(
                        cluster_id=cid,
                        branch="G1_REUSE",
                        assign_to=cat["uuid"],
                        name=type_name_of[target],
                        chain=chain,
                        member_ids=member_ids,
                        resolved_parent_uuid=cat["uuid"],
                        resolved_parent_name=cat.get("name"),
                        covering_depth=max(len(chain) - 1, 0),
                        events=events + [f"A0_REUSE_LIVE:{target}"],
                    )
                )
                continue
            if published_hit is not None:
                cat = published[published_hit]
                branch = "G2_GRAFT"
                parent_uuid: Optional[str] = cat["uuid"]
                parent_name = cat.get("name")
                events.append(f"A0_GRAFT_VIA_LIVE:{published_hit}")
            else:
                branch = "G3_ROOT"
                root_rec = realm_roots.get(chain[-1]) if chain else None
                parent_uuid = (root_rec or {}).get("uuid")
                parent_name = (root_rec or {}).get("name") or (
                    chain[-1] if chain else None
                )
            branches.append(
                _branch_record(
                    cluster_id=cid,
                    branch=branch,
                    assign_to="NEW",
                    name=type_name_of[target],
                    chain=chain,
                    member_ids=member_ids,
                    resolved_parent_uuid=parent_uuid,
                    resolved_parent_name=parent_name,
                    covering_depth=max(len(chain) - 1, 0),
                    events=events,
                )
            )
        out[cid] = {"branches": branches, "residual_ids": residual}
    return out


# ------------------------------------------------------------ T6 snapshot


def build_a0_snapshot(
    clusters: list[dict],
    cascades: dict[str, dict[str, Any]],
    catalog: dict,
    *,
    run_ts: Optional[str] = None,
    model: str = _MODEL_LABEL,
    catalog_mode: str = "seeded_disposable_graph",
) -> dict[str, Any]:
    """Assemble the A0 snapshot through the SAME T6 builder the arms use.

    K = 1 (deterministic pipeline; module docstring) and
    ``raw_partition_ok`` / ``sample_coverage`` are vacuous for the
    baseline (module docstring).
    """
    from harness.run_experiment import build_snapshot  # deferred (shim)

    run_ts = run_ts or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cluster_results = []
    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        cluster_results.append(
            {
                "cluster_id": cid,
                "current_name": cluster.get("current_name", ""),
                "member_count": len(cluster.get("members", [])),
                "repeats": [
                    {
                        "repeat": 0,
                        "request_id": f"{cid}::0",
                        "raw_partition_ok": True,
                        "sample_coverage": 1.0,
                        "cascade": cascades[cid],
                    }
                ],
            }
        )
    return build_snapshot(
        cluster_results=cluster_results,
        catalog=catalog,
        ablation=ABLATION,
        model=model,
        repeats=1,
        catalog_mode=catalog_mode,
        roots_present=bool(catalog.get("roots_present_in_live_catalog", False)),
        run_ts=run_ts,
    )


# ------------------------------------------------- metrics dialect bridge


def _branch_to_group(
    branch: dict[str, Any], catalog_by_uuid: Optional[dict[str, dict]] = None
) -> dict[str, Any]:
    from harness.cascade import _realm_root_name  # READ helper (no stored root)

    kind = branch.get("branch")
    # root is a READ off the one parent edge -- walk resolved_parent_uuid to
    # its realm root in the existing structure (2026-06-06: never stored on
    # the placement). Falls back to the parent name / entity without a catalog.
    parent_uuid_for_root = branch.get("resolved_parent_uuid")
    if catalog_by_uuid:
        root = _realm_root_name(parent_uuid_for_root, catalog_by_uuid)
    else:
        root = branch.get("resolved_parent_name") or "entity"
    is_reuse = kind == "G1_REUSE"
    is_grafted = kind == "G2_GRAFT"
    parent_uuid = branch.get("resolved_parent_uuid")
    parent_name = branch.get("resolved_parent_name")
    return {
        "assign_to": branch.get("assign_to"),
        "name": branch.get("name"),
        "chain": list(branch.get("chain") or []),
        "member_ids": list(branch.get("member_ids") or []),
        "over_specified": bool(branch.get("over_specified", False)),
        "branch": kind,
        "covering_depth": int(branch.get("covering_depth", 0)),
        "is_grafted": is_grafted,
        "is_reuse": is_reuse,
        "reuse_target_uuid": parent_uuid if is_reuse else None,
        "graft_parent_uuid": parent_uuid if is_grafted else None,
        "graft_parent_name": parent_name if is_grafted else None,
        "canonical_merged_into": None,
        "placement_conflict": False,
        "root": root,
    }


def as_metrics_snapshot(
    snapshot: dict[str, Any], catalog: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Bridge a T6 harness snapshot to the eval/metrics consumption shape.

    ``compute_metrics`` reads per-repeat ``groups`` / ``residual_ids`` and
    per-cluster ``total_members``; the T6 harness writes per-repeat
    ``cascade{branches, residual_ids}`` and ``member_count``. This bridge
    derives the former from the latter for ANY T6 snapshot (the A0
    baseline or a v2 arm), so the deltas path consumes both identically.
    RESIDUAL branches contribute their members to ``residual_ids`` and
    emit no group (matching the metrics residual semantics).
    """
    catalog_by_uuid = (catalog or {}).get("catalog_by_uuid", {})
    out = {k: v for k, v in snapshot.items() if k != "clusters"}
    out.setdefault(
        "roots_present_in_live_catalog", bool(snapshot.get("roots_present", False))
    )
    clusters_out = []
    for cluster in snapshot.get("clusters", []):
        repeats_out = []
        coverages = [1.0]
        for rep in cluster.get("repeats", []):
            coverages.append(float(rep.get("sample_coverage", 1.0)))
            cascade = rep.get("cascade") or {}
            branches = list(cascade.get("branches") or [])
            groups = [
                _branch_to_group(b, catalog_by_uuid)
                for b in branches
                if b.get("branch") != "RESIDUAL"
            ]
            residual: set[str] = set(cascade.get("residual_ids") or [])
            evicted: set[str] = set()
            for b in branches:
                if b.get("branch") == "RESIDUAL":
                    residual.update(b.get("member_ids") or [])
                residual.update(b.get("residual_ids") or [])
                evicted.update(b.get("evicted_ids") or [])
            repeats_out.append(
                {
                    "raw_partition_ok": bool(rep.get("raw_partition_ok", True)),
                    "residual_ids": sorted(residual),
                    "evicted_ids": sorted(evicted),
                    "groups": groups,
                }
            )
        clusters_out.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "current_name": cluster.get("current_name", ""),
                "total_members": int(
                    cluster.get("member_count", cluster.get("total_members", 0))
                ),
                "sample_coverage": min(coverages),
                "repeats": repeats_out,
            }
        )
    out["clusters"] = clusters_out
    return out


def deltas_arm_entry(
    snapshot: dict[str, Any],
    catalog: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    """(arm_name, {metric -> aggregate}) entry for the ``ablation_deltas`` input.

    Bridges the snapshot to the metrics dialect, computes the label-free
    structural metrics, and keeps only the aggregate-shaped entries (dicts
    carrying a ``mean``) -- the shape ``ablation_deltas`` consumes per arm.
    Scalar passthroughs and the descriptive ``root_distribution`` are not
    delta-comparable and are dropped. Works identically for the A0 baseline
    snapshot and any v2 arm snapshot, so the by-arm input assembles as
    ``dict(deltas_arm_entry(s) for s in snapshots)``.
    """
    from eval.metrics import compute_metrics  # deferred: path shim above

    metrics = compute_metrics(as_metrics_snapshot(snapshot, catalog))
    aggregates = {
        name: agg
        for name, agg in metrics.items()
        if isinstance(agg, dict) and "mean" in agg
    }
    return str(snapshot.get("ablation", "")), aggregates


# ----------------------------------------------------------------- live half
# Everything below talks to the live stack; sophia imports stay deferred so
# the pure half above is importable without the live environment.


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class FakeMilvus:
    """In-memory vector store mirroring the sophia R1 integration fake.

    Serves the entity collection (member vectors) and TypeCentroid
    (seeded or minted centroids); ``find_nearest_types`` ranks the
    centroid store by cosine -- exactly the contract the production
    handlers rely on.
    """

    def __init__(self) -> None:
        self.members: dict[str, dict] = {}
        self.centroids: dict[str, dict] = {}

    def get_embedding(self, node_type: str, uuid: str) -> Optional[dict]:
        store = self.centroids if node_type == "TypeCentroid" else self.members
        row = store.get(uuid)
        return dict(row) if row else None

    def update_centroid(
        self, type_uuid: str, centroid: list[float], model: str
    ) -> None:
        self.centroids[type_uuid] = {
            "embedding": list(centroid),
            "model": model,
            "embedding_model": model,
        }

    def find_nearest_types(
        self, query_embedding: list[float], top_k: int = 1
    ) -> list[dict]:
        ranked = sorted(
            self.centroids.items(),
            key=lambda kv: -_cosine(query_embedding, kv[1]["embedding"]),
        )
        return [{"uuid": u} for u, _ in ranked[:top_k]]


def _dominant_axis(vectors: list[list[float]]) -> int:
    centroid = [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]
    return max(range(len(centroid)), key=lambda i: abs(centroid[i]))


def axis_labels(plan: dict[str, Any], ns: str) -> dict[int, str]:
    """Deterministic ns-embedding labels for every main axis.

    The token rides into ``mint_type`` slugs (type_<slug>_<hex8>), so every
    minted uuid carries the namespace and teardown can find it.
    """
    labels = {axis: f"{ns} c{cid}" for cid, axis in plan["cluster_axis"].items()}
    for norm, axis in plan["publish_axis"].items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in norm)
        labels[axis] = f"{ns} p{slug}"
    return labels


def make_axis_namer(labels_by_axis: dict[int, str]) -> Callable[..., Any]:
    """Deterministic namer seam: label = dominant centroid axis.

    Mirrors the R1 suite namer: the same cluster always gets the same
    label (re-runs hit dedup / no-op anchors, not namer noise); an
    unlabeled axis is recorded and surfaced AFTER the run (production
    swallows per-node exceptions), never silently defaulted. The namer
    never proposes a graft ``parent`` (module docstring, Seams).
    """
    from sophia.maintenance.emergence_types import NameResult

    unexpected_axes: list[int] = []

    def name_fn(cluster: Any, candidates: Any, hermes_url: Any, token: Any) -> Any:
        axis = _dominant_axis(cluster.embeddings)
        label = labels_by_axis.get(axis)
        if label is None:
            unexpected_axes.append(axis)
            return NameResult(label="", description="", confidence=0.0, removed=[])
        return NameResult(label=label, description="", confidence=0.9, removed=[])

    name_fn.unexpected_axes = unexpected_axes  # type: ignore[attr-defined]
    return name_fn


def _maintenance_config() -> Any:
    from sophia.maintenance.config import MaintenanceConfig

    # Mirrors the R1 integration suite except min_cluster_size (module
    # docstring, Config).
    return MaintenanceConfig(
        variance_threshold=0.05,
        min_cluster_size=2,
        hermes_confidence_floor=0.5,
        type_match_threshold=0.9,
        rollup_min_cluster_size=2,
        rollup_min_supercluster_size=2,
    )


def _connect() -> Any:
    # Fail loudly on a missing credential (same rule as the live probe and
    # reseed entry points on c-daly/logos-experiments#13): never fall back
    # to a default password on a live-graph writer.
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise A0LiveGateError(
            "NEO4J_PASSWORD must be set explicitly for the A0 live driver "
            "(refusing a default credential)"
        )
    from sophia.hcg_client import HCGClient

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    client = HCGClient(neo4j_uri=uri, neo4j_username=user, neo4j_password=password)
    client._execute_query("RETURN 1 AS ok", {})  # connectivity probe
    return client


def _ensure_root(hcg: Any) -> bool:
    """Create the shared realm root if absent; True when this run created it.

    Hoisted OUT of _seed_graph so run_live learns the flag before any
    seeding write can fail: a mid-seed exception must still tear the root
    down in the finally (PR #17 review, P1).
    """
    if hcg.get_node("type_entity") is not None:
        return False
    hcg.add_node(
        name="entity",
        node_type="type_definition",
        uuid="type_entity",
        properties={"ancestors": ["root", "node"]},
        source=_ROOT_MARKER,
    )
    return True


def _close_quietly(hcg: Any) -> None:
    """Close the client without masking an in-flight teardown signal.

    A close() failure must never replace ZeroResidueViolation (or any other
    teardown diagnostic) as the propagating exception (PR #17 review).
    """
    try:
        hcg.close()
    except Exception as err:
        print(
            f"[a0] hcg.close() failed during cleanup: {err}",
            file=sys.stderr,
            flush=True,
        )


def _outside_type_defs(hcg: Any, ns: str) -> dict[str, list[str]]:
    """uuid -> stored ancestors for every type-def OUTSIDE the namespace.

    Captured after seeding and compared after the pass: the namespaced
    rollup loader must keep production writes inside the namespace, and
    ancestor equality (not just count) catches re-parent churn.
    """
    return {
        td["uuid"]: list((td.get("properties") or {}).get("ancestors") or [])
        for td in (hcg.get_all_type_definitions() or [])
        if td.get("uuid") and ns not in td["uuid"]
    }


def _seed_graph(
    hcg: Any,
    milvus: FakeMilvus,
    ns: str,
    clusters: list[dict],
    catalog: dict,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Seed the disposable namespaced inputs (module docstring, Seams)."""
    source = f"a0-{ns}"

    published: dict[str, dict] = {}
    for row in published_rows(catalog, ns):
        hcg.add_node(
            name=row["name"],
            node_type="type_definition",
            uuid=row["live_uuid"],
            properties={"ancestors": list(row["ancestors"])},
            source=source,
        )
        hcg.add_edge(row["live_uuid"], row["parent_uuid"], "IS_A")
        milvus.update_centroid(
            type_uuid=row["live_uuid"],
            centroid=published_vector(
                plan, row["norm"], row["group_position"], row["group_size"]
            ),
            model=_EMBED_MODEL,
        )
        published[row["live_uuid"]] = {
            "uuid": row["catalog_uuid"],
            "name": row["name"],
        }

    # One namespaced junk-drawer type holds ALL frozen-cluster members (the
    # stand-in for the production entity drawer; its name must differ from
    # the literal "entity" so membership resolves via the type_uuid property
    # scan, not the global node-type scan).
    seed_uuid = f"type_{ns}drawer"
    hcg.add_node(
        name=f"{ns} drawer",
        node_type="type_definition",
        uuid=seed_uuid,
        properties={"ancestors": ["root", "node"]},
        source=source,
    )

    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        members = cluster.get("members", [])
        for position, member in enumerate(members):
            member_id = str(member["id"])
            live_uuid = f"{ns}-{member_id}"
            # type_uuid is the authoritative pointer (the production retype
            # write target); set at creation -- add_node merges properties
            # over the standard props, so the end state is identical to the
            # add-then-update two-step at half the roundtrips (PR #17 review).
            hcg.add_node(
                name=str(member.get("name", member_id)),
                node_type="entity",
                uuid=live_uuid,
                properties={"type": "entity", "type_uuid": seed_uuid},
                source=source,
            )
            milvus.members[live_uuid] = {
                "embedding": member_vector(plan, cid, position, len(members)),
                "model": _EMBED_MODEL,
            }

    return {
        "seed_uuid": seed_uuid,
        "published": published,
    }


def _run_pipeline(
    hcg: Any, milvus: FakeMilvus, ns: str, seed_uuid: str, namer: Any
) -> None:
    """Drive the unmodified production handlers over the seeded inputs."""
    from sophia.maintenance.emergence_handler import (
        EmergenceHandler,
        current_categories,
        load_type_members,
    )
    from sophia.maintenance.type_minting import mint_type
    from sophia.maintenance.type_rollup_handler import TypeRollupHandler

    config = _maintenance_config()
    emergence = EmergenceHandler(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://stub.invalid",
        token="stub",
        load_members=lambda type_uuid: load_type_members(hcg, milvus, type_uuid),
        name_fn=namer,
        mint_fn=mint_type,
        candidates_fn=lambda: current_categories(hcg),
    )

    class NamespacedRollup(TypeRollupHandler):
        """Production rollup with a namespace-filtered LOADER.

        Read-side isolation only (mirrors the sophia R1 suite): every
        write below it is unmodified production code on the live graph.
        """

        def _load_type_layer(self) -> list[dict]:
            return [
                r
                for r in super()._load_type_layer()
                if ns in (r.get("uuid") or "")
            ]

    rollup = NamespacedRollup(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://stub.invalid",
        token="stub",
        name_fn=namer,
        mint_fn=mint_type,
    )

    emergence.run(seed_uuid)
    rollup.run()
    if namer.unexpected_axes:
        # Production swallows per-node failures; surface the namer surprise
        # OUTSIDE the swallower (sophia PR #174 discipline).
        raise RuntimeError(
            f"A0 namer hit unlabeled axes: {sorted(set(namer.unexpected_axes))}"
        )


def _read_graph(
    hcg: Any, ns: str, clusters: list[dict]
) -> tuple[dict[str, Optional[str]], dict[str, str], dict[str, Optional[str]]]:
    """Read back member pointers and the namespaced type layer."""
    member_uuids: list[str] = []
    for cluster in clusters:
        for member in cluster.get("members", []):
            member_id = str(member["id"])
            member_uuids.append(f"{ns}-{member_id}")
    rows = {
        r["uuid"]: r
        for r in (hcg.get_nodes_batch(member_uuids) or [])
        if r and "uuid" in r
    }
    member_type: dict[str, Optional[str]] = {}
    for live_uuid in member_uuids:
        row = rows.get(live_uuid) or {}
        member_type[live_uuid] = row.get("type_uuid") or (
            row.get("properties") or {}
        ).get("type_uuid")

    type_name_of: dict[str, str] = {}
    parent_of: dict[str, Optional[str]] = {}
    for td in hcg.get_all_type_definitions() or []:
        uuid = td.get("uuid") or ""
        if ns not in uuid:
            continue
        type_name_of[uuid] = td.get("name") or uuid
        parent_of[uuid] = next(
            (
                e.get("target")
                for e in (hcg.query_edges_from(uuid) or [])
                if e.get("relation") == "IS_A"
            ),
            None,
        )
    return member_type, type_name_of, parent_of


def _teardown(hcg: Any, ns: str, created_root: bool) -> None:
    """DETACH-DELETE everything carrying the token; verify zero residue."""
    try:
        hcg._execute_query(
            "MATCH (n:Node) WHERE n.uuid CONTAINS $t OR n.source CONTAINS $t "
            "OR n.target CONTAINS $t DETACH DELETE n",
            {"t": ns},
        )
        rows = hcg._execute_query(
            "MATCH (n:Node) WHERE n.uuid CONTAINS $t OR n.source CONTAINS $t "
            "OR n.target CONTAINS $t RETURN count(n) AS c",
            {"t": ns},
        )
        residue = rows[0]["c"] if rows else None
        if residue != 0:
            raise ZeroResidueViolation(
                f"teardown left {residue!r} namespaced node(s) for token {ns!r}"
            )
    finally:
        # The shared root does not carry the namespace token, so the token
        # sweep above never removes it; guarantee its cleanup even when the
        # residue check raises (PR #17 review).
        if created_root:
            hcg._execute_query(
                "MATCH (n:Node {uuid: $u, source: $s}) DETACH DELETE n",
                {"u": "type_entity", "s": _ROOT_MARKER},
            )


def run_live(out_dir: Path = WORKSPACE, fixtures_dir: Path = FIXTURES) -> Path:
    """Seed, run the production pipeline, read back, snapshot, tear down.

    Fail-closed ordering mirrors the T6 harness: the teardown, the
    zero-residue check and the outside-namespace non-mutation check all
    complete BEFORE the snapshot is persisted, so a violation never
    leaves a poisoned run_<ts>.json behind for eval consumers.
    """
    if os.environ.get("A0_LIVE") != "1":
        raise A0LiveGateError(
            "A0_LIVE=1 is required: the A0 baseline seeds and tears down a "
            "disposable namespaced subgraph on the live Neo4j "
            "(NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)."
        )
    from harness.fixtures_io import load_catalog, load_clusters  # path shim
    from harness.run_experiment import NonMutationViolation

    clusters = load_clusters(Path(fixtures_dir) / "clusters.json")
    catalog = load_catalog(Path(fixtures_dir) / "catalog.json")
    plan = geometry_plan(clusters, catalog)
    ns = f"a0b{uuid4().hex[:8]}"
    milvus = FakeMilvus()
    hcg = _connect()
    snapshot: Optional[dict[str, Any]] = None
    # Learn the flag before any seeding write can fail: _seed_graph used to
    # create the root internally, which orphaned type_entity when a later
    # seeding write raised (PR #17 review, P1).
    created_root = _ensure_root(hcg)
    try:
        seeded = _seed_graph(hcg, milvus, ns, clusters, catalog, plan)
        outside_before = _outside_type_defs(hcg, ns)
        namer = make_axis_namer(axis_labels(plan, ns))
        _run_pipeline(hcg, milvus, ns, seeded["seed_uuid"], namer)
        outside_after = _outside_type_defs(hcg, ns)
        if outside_after != outside_before:
            raise NonMutationViolation(
                "type-defs outside the namespace changed during the pass "
                f"({len(outside_before)} -> {len(outside_after)} tracked rows)"
            )
        member_type, type_name_of, parent_of = _read_graph(hcg, ns, clusters)
        cascades = map_outcomes(
            clusters,
            ns=ns,
            seed_uuid=seeded["seed_uuid"],
            member_type=member_type,
            type_name_of=type_name_of,
            parent_of=parent_of,
            published=seeded["published"],
            realm_roots=realm_roots_from_catalog(catalog),
        )
        snapshot = build_a0_snapshot(clusters, cascades, catalog)
    finally:
        try:
            _teardown(hcg, ns, created_root)
        finally:
            _close_quietly(hcg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts_value = str(snapshot["run_ts"])
    out_path = out_dir / f"run_{run_ts_value}.json"
    out_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A0 measured rollup baseline (env-gated: A0_LIVE=1)"
    )
    parser.add_argument(
        "--out", default=str(WORKSPACE), help="snapshot output directory"
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES), help="frozen fixtures directory"
    )
    args = parser.parse_args(argv)
    if os.environ.get("A0_LIVE") != "1":
        print(
            "[a0] refusing to run: set A0_LIVE=1 explicitly. The A0 baseline "
            "seeds and tears down a DISPOSABLE namespaced subgraph on the "
            "live Neo4j (NEO4J_URI / NEO4J_USER default bolt://localhost:7687 "
            "/ neo4j; NEO4J_PASSWORD must be set explicitly).",
            file=sys.stderr,
        )
        return 2
    out_path = run_live(Path(args.out), Path(args.fixtures_dir))
    print(f"[a0] snapshot -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
