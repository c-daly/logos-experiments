"""Harness: is embedding reified edges worth it?

Grows a knowledge graph across N ingestion rounds against the LIVE LOGOS stack,
then runs emergence on BOTH the entity residue (node-types) and the reified edge
nodes (edge-types), recording per-round snapshots that ``eval/metrics.py`` scores.

Run (full):   poetry run python harness/run_experiment.py --seed-n 1 --rounds 4
Run (smoke):  poetry run python harness/run_experiment.py --seed-n 1 --smoke

MUST be run inside the sophia poetry env so ``sophia.maintenance.emergence_*``,
``logos_hcg`` and ``pymilvus`` import. Resets the live DB on every run (fine).

Stack defaults (overridable via env):
  NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=logosdev
  MILVUS_HOST=localhost            MILVUS_PORT=19530
  HERMES_URL=http://localhost:17000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests
from neo4j import GraphDatabase

from logos_hcg.client import HCGClient
from logos_hcg.seeder import HCGSeeder
from logos_hcg.sync import HCGMilvusSync
from sophia.maintenance.emergence_clustering import (
    find_emergent_clusters,
    find_emergent_hierarchy,
)
from sophia.maintenance.emergence_types import Member

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CORPUS = EXP / "corpus" / "corpus.jsonl"
WORKSPACE = EXP / "workspace"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "logosdev")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")

# Node types that are domain ENTITIES (not ontology scaffolding / cognition /
# reserved system nodes). The NER pipeline assigns "entity" plus emergent
# sub-types like "animal", "biological_entity", "scientific_instrument"; we treat
# anything that is not type_definition/edge/edge_type/reserved_*/cognition as an
# entity-kind candidate for node clustering.
_NON_ENTITY_TYPES = {
    "type_definition",
    "edge",
    "edge_type",
    "cognition",
    "node",
}

# Purely-structural / ontological relations. These are NOT semantic edges --
# IS_A / COMPONENT_OF are scaffolding the seeder + type hierarchy mint, and
# every IS_A edge embeds to the identical "RELATIONSHIP: IS_A" vector, so they
# would dominate (and tautologically dominate the coherence of) the edge-type
# clustering pool. We keep them in the GRAPH but EXCLUDE them from the
# edge-discovery clustering population. (see build_edge_members)
#
# NOTE (post #505 membership-change): instance->type IS_A edges no longer exist
# -- membership is now carried by the entity ``type_uuid`` property, not a
# reified IS_A edge. So this partition is largely a no-op now: the only IS_A
# edges remaining are the few TAXONOMY IS_A among type-definitions (type_def ->
# type_def). Harmless to keep; it just excludes those taxonomy IS_A.
STRUCTURAL_EDGE_RELATIONS = {"IS_A", "COMPONENT_OF"}

# Light junk filter for the node-clustering population ONLY (never touches the
# graph). The live NER pipeline emits generic single-token fragments ("speed",
# "mass", "light", ...) that are not real entities; we drop them so the
# observational node clustering measures real entity structure rather than NER
# debris. Preference order: if a node carries a numeric confidence/score
# property we threshold on it; otherwise we fall back to this small stoplist.
# Heuristic is intentionally minimal + documented (toggle: --no-entity-filter).
_ENTITY_STOPLIST = {
    "speed",
    "tremendous speed",
    "mass",
    "prey",
    "structure",
    "light",
    "wavelength",
    "size",
    "shape",
    "color",
    "colour",
    "surface",
    "region",
    "area",
    "thing",
    "object",
    "matter",
    "energy",
    "force",
}
_ENTITY_CONFIDENCE_THRESHOLD = 0.5

# A larger bootstrap (N>1) seeds these foundational types with name-derived
# centroids in addition to the always-present {entity, related}.
_BOOTSTRAP_LADDER = [
    "object",
    "location",
    "concept",
    "animal",
    "instrument",
    "vehicle",
    "plant",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Hermes
# ---------------------------------------------------------------------------


def hermes_embed(text: str) -> tuple[list[float], str]:
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(
                f"{HERMES_URL}/embed_text", json={"text": text}, timeout=120
            )
            r.raise_for_status()
            d = r.json()
            return d["embedding"], d.get("model", "unknown")
        except Exception as e:  # transient upstream (e.g. OpenAI 5xx)
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"hermes_embed failed after 5 retries for {text!r}: {last_err}")


def hermes_ingest(text: str, domain: str, rnd: int) -> None:
    """Ingest one sentence via /llm with the cheap deterministic echo provider.

    The echo completion is trivial, but Hermes still runs NER + embeddings and
    Sophia writes the resulting entities/edges -- which is all we need.
    """
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(
                f"{HERMES_URL}/llm",
                json={
                    "prompt": text,
                    "provider": "echo",
                    "metadata": {
                        "experiment": "edge-emb",
                        "domain": domain,
                        "round": rnd,
                    },
                },
                timeout=180,
            )
            r.raise_for_status()
            return
        except Exception as e:  # transient upstream
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"hermes_ingest failed after 5 retries: {last_err}")


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------


def entity_uuids(driver: Any) -> dict[str, dict[str, Any]]:
    """Return {uuid: {name, type, confidence}} for every entity-kind node.

    ``confidence`` is whatever scoring property the node carries (coalesced
    across the property names the pipeline has used); ``None`` if absent.
    Used by the light junk filter on the node-clustering population.
    """
    cypher = (
        "MATCH (n:Node) "
        "WHERE NOT n.type IN $excluded "
        "AND NOT n.type STARTS WITH 'reserved_' "
        "AND NOT n.type STARTS WITH 'type_' "
        "RETURN n.uuid AS uuid, n.name AS name, n.type AS type, "
        "coalesce(n.confidence, n.score, n.ner_confidence) AS confidence"
    )
    out: dict[str, dict[str, Any]] = {}
    with driver.session() as s:
        for rec in s.run(cypher, excluded=list(_NON_ENTITY_TYPES)):
            out[rec["uuid"]] = {
                "name": rec["name"],
                "type": rec["type"],
                "confidence": rec["confidence"],
            }
    return out


def edge_nodes(driver: Any) -> list[dict[str, Any]]:
    """Return reified edge nodes with their relation + endpoint uuids."""
    cypher = (
        "MATCH (e:Node {type: 'edge'}) "
        "RETURN e.uuid AS uuid, e.relation AS relation, "
        "e.source AS source, e.target AS target, e.name AS name"
    )
    with driver.session() as s:
        return [dict(rec) for rec in s.run(cypher)]


def node_type_of(driver: Any, uuid: str) -> str | None:
    with driver.session() as s:
        rec = s.run(
            "MATCH (n:Node {uuid: $u}) RETURN n.type AS t", u=uuid
        ).single()
        return rec["t"] if rec else None


# ---------------------------------------------------------------------------
# Bootstrap / seeding
# ---------------------------------------------------------------------------


def add_edge_type_definition(client: HCGClient, name: str, relation: str) -> None:
    """Create an edge-type-definition node under the edge_type root.

    Mirrors seeder.seed_type_definitions's EDGE_TYPES loop: a node of
    node_type=='edge_type' with an IS_A edge to type_edge_type. This is the
    `related` junk-drawer that edge-type emergence tries to split.
    """
    node_uuid = client.add_node(
        uuid=f"type_edge_{name.lower()}",
        name=relation,
        node_type="edge_type",
    )
    client.add_edge(
        source_uuid=node_uuid,
        target_uuid="type_edge_type",
        relation="IS_A",
    )


def bootstrap(
    client: HCGClient, sync: HCGMilvusSync, seed_n: int
) -> dict[str, Any]:
    """Reset the graph and seed the bootstrap ontology for the given N.

    N=1  -> cold start: ensure {entity} node-type-def + {related/RELATED_TO}
            edge-type-def, NO centroids (true cold-start).
    N>1  -> additionally seed the first (N-1) foundational types from the ladder
            with NAME-DERIVED centroids (embed the type name, update_centroid).
    """
    seeder = HCGSeeder(client)
    log("[bootstrap] clearing graph ...")
    seeder.clear()
    log("[bootstrap] seeding core type hierarchy ...")
    seeder.seed_type_definitions()  # entity, object, location, edge_type, IS_A ...

    # The `related` junk-drawer edge-type (NOT in EDGE_TYPES) -- the thing edge
    # emergence is meant to carve up.
    add_edge_type_definition(client, name="related", relation="RELATED_TO")

    seeded_centroids: list[str] = []
    if seed_n > 1:
        n_extra = seed_n - 1
        for type_name in _BOOTSTRAP_LADDER[:n_extra]:
            embedding, model = hermes_embed(
                f"A {type_name}: a kind of entity in the world."
            )
            sync.update_centroid(
                type_uuid=f"type_{type_name}",
                centroid=embedding,
                model=model,
            )
            seeded_centroids.append(type_name)
        log(f"[bootstrap] N={seed_n}: name-derived centroids -> {seeded_centroids}")
    else:
        log("[bootstrap] N=1 cold start: no centroids seeded")

    return {"seed_n": seed_n, "seeded_centroids": seeded_centroids}


# ---------------------------------------------------------------------------
# Edge embeddings
# ---------------------------------------------------------------------------


def embed_edges(
    driver: Any,
    sync: HCGMilvusSync,
    relation_cache: dict[str, tuple[list[float], str]],
) -> int:
    """(Re)embed EVERY reified edge with a uniform RELATIONSHIP: <relation> vector.

    Text is \"RELATIONSHIP: <relation>\", cached per relation label. We OVERRIDE
    any pre-existing Edge-collection embedding (e.g. a Hermes triple embedding
    minted at ingest) rather than fill-only-missing: the edge space must be
    homogeneous -- every edge embedded by the SAME function -- for the edge-type
    clustering to be valid. (IS_A/COMPONENT_OF are embedded here too but get
    partitioned out of the clustering pool downstream; that is harmless.)
    Returns the number of edge embeddings upserted this call.
    """
    edges = edge_nodes(driver)
    pending: list[dict[str, Any]] = []
    for e in edges:
        uuid = e["uuid"]
        relation = e.get("relation") or "RELATED_TO"
        if relation not in relation_cache:
            relation_cache[relation] = hermes_embed(f"RELATIONSHIP: {relation}")
        emb, model = relation_cache[relation]
        pending.append({"uuid": uuid, "embedding": emb, "model": model})

    if pending:
        sync.batch_upsert_embeddings("Edge", pending)
    return len(pending)


# ---------------------------------------------------------------------------
# Member construction
# ---------------------------------------------------------------------------


def _is_junk_entity(info: dict[str, Any]) -> bool:
    """Light, documented heuristic: is this node an obvious NER fragment?

    Two INDEPENDENT signals -- drop the node if EITHER fires:
      1. Numeric confidence/score below ``_ENTITY_CONFIDENCE_THRESHOLD`` (0.5),
         when the node carries a *discriminative* score. NOTE: the current live
         NER pipeline stamps a constant ``confidence=0.7`` on every entity
         (junk included), so this signal is presently a no-op -- which is why
         the stoplist below is applied unconditionally rather than only as a
         fallback. If the pipeline later emits real per-entity confidences,
         this branch starts doing useful work automatically.
      2. Name in a tiny stoplist of generic single-token fragments
         ("speed", "mass", "light", ...) the pipeline emits as spurious
         entities.
    Applied ONLY to the harness node-clustering population -- never to graph.
    """
    conf = info.get("confidence")
    if isinstance(conf, (int, float)) and conf < _ENTITY_CONFIDENCE_THRESHOLD:
        return True
    name = (info.get("name") or "").strip().lower()
    return name in _ENTITY_STOPLIST


_DEDUP_COSINE_THRESHOLD = 0.98


def _dedup_node_members(
    members: list[Member],
    type_by_uuid: dict[str, str],
    entity_domains: dict[str, str] | None,
) -> tuple[list[Member], dict[str, str], int]:
    """BAND-AID dedup of the node-clustering population (NOT the graph).

    The live ingestion pipeline currently mints a fresh entity node per mention
    (e.g. "peregrine falcon" appears once per line that names it), so the same
    real-world entity shows up as several near-identical Members and inflates /
    smears the node clusters. Here we collapse those duplicates into a single
    representative *for clustering only* -- the graph is untouched.

    Grouping (union of two signals):
      1. exact lowercased ``name`` match;
      2. cosine >= ``_DEDUP_COSINE_THRESHOLD`` (0.98) on the raw entity
         embedding, even when names differ (alias / surface-form variants).
    Each group collapses to one rep: the first member, carrying the group MEAN
    embedding and the MAJORITY provenance domain (from ``entity_domains`` when
    available, else the majority ``current_type``) as its ``current_type``.
    Emits ``[METRIC] node_dedup_merged=<count>`` (members removed).

    NOTE: this is a measurement-side band-aid. The real fix lives in the
    pipeline: dedup-by-identity on ingest + accumulate-by-flavor, so one node
    per entity carries the union of its observed surface forms / domains.
    """
    if not members:
        print("[METRIC] node_dedup_merged=0", flush=True)
        return members, type_by_uuid, 0

    # Pre-normalise embeddings for cheap cosine via dot product.
    vecs: list[Any] = []
    for m in members:
        v = np.asarray(m.embedding, dtype=float)
        n = float(np.linalg.norm(v))
        vecs.append(v / n if n > 0 else v)

    group_of: list[int] = [-1] * len(members)
    reps: list[int] = []  # indices of representative members, one per group
    name_to_group: dict[str, int] = {}
    for i, m in enumerate(members):
        key = (m.name or "").strip().lower()
        # 1. exact lowercased-name match -> existing group.
        g = name_to_group.get(key) if key else None
        # 2. else near-identical embedding to an existing rep.
        if g is None:
            for gi, rep_idx in enumerate(reps):
                if float(np.dot(vecs[i], vecs[rep_idx])) >= _DEDUP_COSINE_THRESHOLD:
                    g = gi
                    break
        if g is None:
            g = len(reps)
            reps.append(i)
        group_of[i] = g
        if key:
            name_to_group.setdefault(key, g)

    n_groups = len(reps)
    if n_groups == len(members):
        print("[METRIC] node_dedup_merged=0", flush=True)
        return members, type_by_uuid, 0

    # Collect group membership in order.
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, g in enumerate(group_of):
        groups[g].append(i)

    deduped: list[Member] = []
    new_type_by_uuid: dict[str, str] = {}
    for member_idxs in groups:
        mean_emb = np.mean(
            [np.asarray(members[i].embedding, dtype=float) for i in member_idxs],
            axis=0,
        )
        # Preserve the MAJORITY provenance domain: vote on the ingest-tracked
        # domain (entity_domains, uuid -> domain) across the group, then pick
        # the representative member from that winning domain so the surviving
        # rep carries the majority provenance. Falls back to the first member
        # when no domain provenance is available for the group.
        domain_votes: Counter[str] = Counter()
        for i in member_idxs:
            dom = (entity_domains or {}).get(members[i].uuid)
            if dom:
                domain_votes[dom] += 1
        rep_idx = member_idxs[0]
        if domain_votes:
            majority_domain = domain_votes.most_common(1)[0][0]
            for i in member_idxs:
                if (entity_domains or {}).get(members[i].uuid) == majority_domain:
                    rep_idx = i
                    break
        rep = members[rep_idx]
        # current_type carries the live-minted type-flavor (e.g. animal_xxx);
        # keep the group majority so clustering metadata stays representative.
        type_votes: Counter[str] = Counter(
            members[i].current_type for i in member_idxs if members[i].current_type
        )
        rep_type = (
            type_votes.most_common(1)[0][0] if type_votes else rep.current_type
        )
        merged = Member(
            uuid=rep.uuid,
            name=rep.name,
            embedding=list(mean_emb),
            signature=Counter(),
            current_type=rep_type,
            hermes_type_hint=rep.hermes_type_hint,
            neighbors=[],
            model=rep.model,
        )
        deduped.append(merged)
        new_type_by_uuid[rep.uuid] = rep_type

    merged_count = len(members) - len(deduped)
    print(f"[METRIC] node_dedup_merged={merged_count}", flush=True)
    return deduped, new_type_by_uuid, merged_count


def build_node_members(
    driver: Any,
    sync: HCGMilvusSync,
    entity_filter: bool = True,
    dedup: bool = True,
    entity_domains: dict[str, str] | None = None,
) -> tuple[list[Member], dict[str, str], int]:
    """Build Members for entity-kind nodes that have an entity embedding.

    Observational by construction: every Member is built from the node's RAW
    Entity embedding (``sync.get_embedding("Entity", uuid)``) and clustered on
    that vector. Sophia's live-minted ``type`` is carried only as ``current_type``
    metadata (and in ``type_by_uuid``) -- it does NOT drive the clustering, so
    the measurement is independent of the live scheduler's mints.

    When ``entity_filter`` is on (default), obvious NER-fragment nodes are
    dropped from this population (see ``_is_junk_entity``). Returns
    ``(members, type_by_uuid, n_dropped)``.
    """
    nodes = entity_uuids(driver)
    members: list[Member] = []
    type_by_uuid: dict[str, str] = {}
    n_dropped = 0
    for uuid, info in nodes.items():
        if entity_filter and _is_junk_entity(info):
            n_dropped += 1
            continue
        try:
            rec = sync.get_embedding("Entity", uuid)
        except Exception:
            rec = None
        if not rec or not rec.get("embedding"):
            continue
        emb = list(rec["embedding"])
        members.append(
            Member(
                uuid=uuid,
                name=info["name"],
                embedding=emb,
                signature=Counter(),
                current_type=info["type"],
                hermes_type_hint=None,
                neighbors=[],
                model=rec.get("embedding_model"),
            )
        )
        type_by_uuid[uuid] = info["type"]
    if dedup:
        # Band-aid: collapse duplicate/near-identical entities BEFORE clustering
        # (clustering population only -- never the graph). Emits
        # [METRIC] node_dedup_merged=<count>.
        members, type_by_uuid, _merged = _dedup_node_members(
            members, type_by_uuid, entity_domains
        )
    else:
        print("[METRIC] node_dedup_merged=0", flush=True)
    return members, type_by_uuid, n_dropped


def build_edge_members(
    driver: Any, sync: HCGMilvusSync
) -> tuple[list[Member], dict[str, dict[str, Any]], int]:
    """Build Members for reified edges that have a RELATIONSHIP embedding.

    Purely-structural relations (``STRUCTURAL_EDGE_RELATIONS`` = IS_A,
    COMPONENT_OF) are PARTITIONED OUT of the edge-discovery clustering pool:
    they stay in the graph but are not clustered into edge-types (they are
    ontology scaffolding and would otherwise swamp the semantic-edge signal).
    Returns ``(members, meta, n_structural_excluded)``; ``len(members)`` is the
    semantic-edge count remaining in the pool.
    """
    edges = edge_nodes(driver)
    members: list[Member] = []
    meta: dict[str, dict[str, Any]] = {}
    n_structural = 0
    for e in edges:
        uuid = e["uuid"]
        relation = e.get("relation") or "RELATED_TO"
        if relation in STRUCTURAL_EDGE_RELATIONS:
            n_structural += 1
            continue
        try:
            rec = sync.get_embedding("Edge", uuid)
        except Exception:
            rec = None
        if not rec or not rec.get("embedding"):
            continue
        members.append(
            Member(
                uuid=uuid,
                name=relation,
                embedding=list(rec["embedding"]),
                signature=Counter(),
                current_type="related",
                hermes_type_hint=None,
                neighbors=[],
                model=rec.get("embedding_model"),
            )
        )
        meta[uuid] = e
    return members, meta, n_structural


# ---------------------------------------------------------------------------
# Clustering wrappers
# ---------------------------------------------------------------------------


def cluster_nodes(
    members: list[Member], min_cluster_size: int
) -> list[dict[str, Any]]:
    if len(members) < 2 * min_cluster_size:
        return []
    clusters = find_emergent_clusters(
        members, min_cluster_size=min_cluster_size, variance_threshold=0.0
    )
    out = []
    for i, cl in enumerate(clusters):
        out.append(
            {
                "label": i,
                "members": [{"uuid": m.uuid, "name": m.name} for m in cl.members],
            }
        )
    return out


def cluster_node_hierarchy(
    members: list[Member], min_cluster_size: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discover the node-type hierarchy and the top-level-root assignment.

    Returns ``(hierarchy_describe, assignment)`` where ``assignment`` maps each
    entity uuid to the index of its TOP-LEVEL hierarchy root (a HierarchyNode
    root carries ALL leaf members beneath it). This coarse, hierarchy-granularity
    assignment -- not the over-fragmented flat clusters -- is what the
    node_type_ari metric scores against the provenance domain labels. Entities
    not under any root are left unassigned here and treated as their own
    singleton by the metric.
    """
    if len(members) < 2 * min_cluster_size:
        return [], {}
    roots = find_emergent_hierarchy(
        members, min_cluster_size=min_cluster_size, variance_threshold=0.0
    )

    def describe(node: Any) -> dict[str, Any]:
        return {
            "size": len(node.members),
            "sample": [m.name for m in node.members[:6]],
            "children": [describe(ch) for ch in node.children],
        }

    assignment: dict[str, int] = {}
    for root_idx, root in enumerate(roots):
        for m in root.members:
            assignment.setdefault(m.uuid, root_idx)

    return [describe(r) for r in roots], assignment


def cluster_edges(
    members: list[Member],
    meta: dict[str, dict[str, Any]],
    min_cluster_size: int,
) -> list[dict[str, Any]]:
    if len(members) < 2 * min_cluster_size:
        return []
    clusters = find_emergent_clusters(
        members, min_cluster_size=min_cluster_size, variance_threshold=0.0
    )
    out = []
    for i, cl in enumerate(clusters):
        edges = []
        for m in cl.members:
            e = meta.get(m.uuid, {})
            edges.append(
                {
                    "uuid": m.uuid,
                    "relation": e.get("relation", m.name),
                    "source": e.get("source"),
                    "target": e.get("target"),
                }
            )
        out.append({"label": i, "edges": edges})
    return out


def endpoint_type_view(
    driver: Any, meta: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Group edges by (source-type, target-type) as a structural baseline view."""
    type_cache: dict[str, str | None] = {}

    def t(u: str | None) -> str:
        if not u:
            return "?"
        if u not in type_cache:
            type_cache[u] = node_type_of(driver, u)
        return type_cache[u] or "?"

    counts: Counter = Counter()
    for e in meta.values():
        counts[f"{t(e.get('source'))} -> {t(e.get('target'))}"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus() -> list[dict[str, Any]]:
    rows = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    seed_n: int,
    rounds: int,
    smoke: bool,
    min_cluster_size: int,
    entity_filter: bool = True,
    dedup: bool = True,
    max_per_domain: int = 0,
) -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()

    if smoke:
        # ~6 lines from round 1, single clustering pass.
        r1 = [c for c in corpus if c["round"] == 1]
        animals = [c for c in r1 if c["domain"] == "animals"][:3]
        instruments = [c for c in r1 if c["domain"] == "lab_instruments"][:3]
        corpus = animals + instruments
        rounds = 1
        log(f"[smoke] using {len(corpus)} corpus lines, single pass")

    client = HCGClient(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    sync = HCGMilvusSync(milvus_host=MILVUS_HOST, milvus_port=str(MILVUS_PORT))
    sync.connect()
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    relation_cache: dict[str, tuple[list[float], str]] = {}
    entity_domains: dict[str, str] = {}
    # uuid -> source sentence the entity / edge was first extracted from. Used
    # by the capture+sweep to embed a CONTEXT vector (and fuse it with the
    # name / RELATIONSHIP-label vector). The graph itself does not keep this
    # join, so we record it here at ingest.
    entity_sentences: dict[str, str] = {}
    edge_sentences: dict[str, str] = {}
    ingested = 0

    try:
        boot = bootstrap(client, sync, seed_n)
        log(f"[bootstrap] done: {boot}")

        for r in range(1, rounds + 1):
            log(f"\n===== ROUND {r} =====")
            this_round = [c for c in corpus if c["round"] <= r]
            if max_per_domain > 0:
                # Trim each domain to at most N CUMULATIVE lines (in corpus
                # order). Deterministic + a stable growing prefix, so the
                # idx<ingested skip below stays correct across rounds.
                seen_per_domain: Counter[str] = Counter()
                capped: list[dict[str, Any]] = []
                for c in this_round:
                    dom = c["domain"]
                    if seen_per_domain[dom] >= max_per_domain:
                        continue
                    seen_per_domain[dom] += 1
                    capped.append(c)
                this_round = capped
            for idx, item in enumerate(this_round):
                # Skip lines already ingested in prior rounds.
                if idx < ingested:
                    continue
                before = set(entity_uuids(driver).keys())
                before_edges = {e["uuid"] for e in edge_nodes(driver)}
                hermes_ingest(item["text"], item["domain"], r)
                # Poll until new entities appear (ingestion is ~synchronous).
                new: set[str] = set()
                for _ in range(20):
                    after = entity_uuids(driver)
                    new = set(after.keys()) - before
                    if new:
                        break
                    time.sleep(0.5)
                for u in new:
                    # Record provenance ONLY for entities first seen for this
                    # line's domain (later cross-domain lines don't overwrite).
                    entity_domains.setdefault(u, item["domain"])
                    entity_sentences.setdefault(u, item["text"])
                # Edges first seen this line came from this sentence.
                for e in edge_nodes(driver):
                    if e["uuid"] not in before_edges:
                        edge_sentences.setdefault(e["uuid"], item["text"])
            ingested = len(this_round)
            ents = entity_uuids(driver)
            log(
                f"[round {r}] ingested {len(this_round)} lines -> "
                f"{len(ents)} entity-kind nodes, "
                f"{len(edge_nodes(driver))} edges"
            )

            # (a) edge embeddings
            n_new_edges = embed_edges(driver, sync, relation_cache)
            log(f"[round {r}] edge embeddings created this round: {n_new_edges}")

            # (b) node clustering (observational: clusters RAW entity
            # embeddings, independent of Sophia's live-minted types). The
            # light junk filter drops obvious NER fragments from this pool.
            node_members, _, n_node_dropped = build_node_members(
                driver,
                sync,
                entity_filter=entity_filter,
                dedup=dedup,
                entity_domains=entity_domains,
            )
            node_clusters = cluster_nodes(node_members, min_cluster_size)
            node_hierarchy, node_hierarchy_assignment = cluster_node_hierarchy(
                node_members, min_cluster_size
            )
            log(
                f"[round {r}] node members={len(node_members)} "
                f"(entity_filter={'on' if entity_filter else 'off'}, "
                f"dropped={n_node_dropped}) "
                f"flat_clusters={len(node_clusters)} "
                f"hierarchy_roots={len(node_hierarchy)}"
            )

            # (c) edge clustering (IS_A/COMPONENT_OF partitioned out of pool)
            edge_members, edge_meta, n_structural = build_edge_members(driver, sync)
            edge_clusters = cluster_edges(edge_members, edge_meta, min_cluster_size)
            endpoint_view = endpoint_type_view(driver, edge_meta)
            n_total_edges = len(edge_nodes(driver))
            log(
                f"[round {r}] edges total={n_total_edges} "
                f"structural_excluded={n_structural} "
                f"semantic_pool={len(edge_members)} "
                f"edge_clusters={len(edge_clusters)}"
            )

            # Resolve entity NAMES for readable edge cluster output.
            name_by_uuid = {u: i["name"] for u, i in ents.items()}
            for cl in edge_clusters:
                for e in cl["edges"]:
                    e["label"] = (
                        f"{name_by_uuid.get(e.get('source'), '?')} "
                        f"--{e.get('relation')}--> "
                        f"{name_by_uuid.get(e.get('target'), '?')}"
                    )

            # (d) snapshot
            snapshot = {
                "round": r,
                "seed_n": seed_n,
                "smoke": smoke,
                "min_cluster_size": min_cluster_size,
                "counts": {
                    "entity_nodes": len(ents),
                    "edge_nodes": n_total_edges,
                    "node_members": len(node_members),
                    "node_dropped_junk": n_node_dropped,
                    "edge_members": len(edge_members),
                    "edge_structural_excluded": n_structural,
                    "edge_embeddings_new": n_new_edges,
                },
                "entity_filter": entity_filter,
                "entity_domains": entity_domains,
                "entity_sentences": entity_sentences,
                "edge_sentences": edge_sentences,
                "node_clusters": node_clusters,
                "node_hierarchy": node_hierarchy,
                # uuid -> top-level hierarchy-root index. node_type_ari is scored
                # against THIS (hierarchy granularity), not the flat clusters.
                "node_hierarchy_assignment": node_hierarchy_assignment,
                "edge_clusters": edge_clusters,
                "endpoint_type_view": endpoint_view,
            }
            out_path = WORKSPACE / f"round_{r}.json"
            out_path.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log(f"[round {r}] snapshot -> {out_path}")

            _print_round_summary(snapshot, name_by_uuid)

        # Final metrics on the last snapshot.
        from importlib import util as _util

        metrics_path = EXP / "eval" / "metrics.py"
        spec = _util.spec_from_file_location("edge_emb_metrics", metrics_path)
        mod = _util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        last = json.loads(
            (WORKSPACE / f"round_{rounds}.json").read_text(encoding="utf-8")
        )
        log("\n===== METRICS (final round) =====")
        mod.emit(mod.compute_metrics(last))
    finally:
        driver.close()
        sync.disconnect()
        client.close()


def _print_round_summary(
    snapshot: dict[str, Any], name_by_uuid: dict[str, str]
) -> None:
    log("  -- node clusters --")
    if not snapshot["node_clusters"]:
        log("     (none)")
    for cl in snapshot["node_clusters"]:
        names = [m["name"] for m in cl["members"]]
        log(f"     cluster {cl['label']}: {names}")
    log("  -- edge clusters --")
    if not snapshot["edge_clusters"]:
        log("     (none)")
    for cl in snapshot["edge_clusters"]:
        rels = Counter(e["relation"] for e in cl["edges"])
        log(f"     cluster {cl['label']}: {dict(rels)}")
        for e in cl["edges"][:5]:
            log(f"        {e.get('label', e['relation'])}")
    view = snapshot['endpoint_type_view']
    log(f"  -- endpoint (src-type -> tgt-type) view -- {view}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-n", type=int, default=1)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--min-cluster-size", type=int, default=2)
    # Light NER-fragment filter on the node-clustering population (default ON).
    p.add_argument(
        "--entity-filter",
        dest="entity_filter",
        action="store_true",
        default=True,
        help="drop obvious NER-fragment entities from node clustering (default)",
    )
    p.add_argument(
        "--no-entity-filter",
        dest="entity_filter",
        action="store_false",
        help="keep all entities in the node-clustering population",
    )
    # Light entity dedup on the node-clustering population (default ON). Band-aid
    # for the pipeline minting one node per mention; never touches the graph.
    p.add_argument(
        "--dedup",
        dest="dedup",
        action="store_true",
        default=True,
        help="collapse duplicate/near-identical entities before clustering (default)",
    )
    p.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        help="keep every entity node in the clustering population",
    )
    p.add_argument(
        "--max-per-domain",
        type=int,
        default=0,
        help="cap each domain to at most N cumulative ingested lines (0 = no cap)",
    )
    args = p.parse_args()
    run(
        seed_n=args.seed_n,
        rounds=args.rounds,
        smoke=args.smoke,
        min_cluster_size=args.min_cluster_size,
        entity_filter=args.entity_filter,
        dedup=args.dedup,
        max_per_domain=args.max_per_domain,
    )


if __name__ == "__main__":
    sys.exit(main())
