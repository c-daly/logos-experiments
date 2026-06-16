"""edge-emergence P1 pilot, RUN-2 design (lx#46) — read-only induction.

Run-1 (run_20260611_192846.json) failed its gate and earned three changes:
  (a) the ledger adjudicates on INDUCED endpoint roles, not realm-flat
      membership (its only semantic brake, -log2 P(tgt_type|rel), is
      uninformative on a flat graph — every merge won its 16-bit saving);
  (b) merge proposals come from LABEL-TEXT embedding clusters and must be
      CONFIRMED by endpoint-pair distribution similarity before the ledger
      votes (run 1's edge-phrase instance embeddings were endpoint-dominated
      — stratum 2 restated stratum 1 instead of corroborating it);
  (c) per-class ARI was degenerate (constant partition) — replaced by the
      pair-histogram cosine as the agreement measure.

Funnel: label clusters propose -> pair-hist similarity confirms -> role-rich
dL adjudicates. Senses per surface from pair-clusters; splits adjudicated
the same way. Read-only; the only write is workspace/run_<ts>.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mdl-ledger"))

from ledger import Snapshot, delta  # noqa: E402

from stages import (  # noqa: E402
    cluster_embeddings,
    directional_veto,
    instance_reversal_veto,
    is_candidate_relation,
    mapping_eval,
    merge_relations,
    pair_hist_similarity,
    pair_vectors,
    role_membership,
    role_pair_hist,
    split_relation,
)

MIN_SENSE_MASS = 5
NODE_COVERAGE_HARD_STOP = 0.50
SIM_CONFIRM_BAR = 0.5  # provisional — freeze on readout (spec convention)
MIN_SURFACE_INSTANCES = 2  # surfaces with 1 instance carry no distribution

MAPPING_CSV = Path(__file__).resolve().parent.parent / "relation-vocab" / "mapping.csv"
HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")


def _driver():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit("NEO4J_PASSWORD must be set explicitly (no default credential)")
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
    )


def fetch_graph(driver):
    with driver.session() as s:
        edges = [
            (r["uuid"], r["rel"], r["src"], r["tgt"])
            for r in s.run(
                "MATCH (e:Node {type:'edge'}) "
                "RETURN e.uuid AS uuid, e.relation AS rel, e.source AS src, e.target AS tgt"
            )
            if r["uuid"] and r["rel"] and r["src"] and r["tgt"]
        ]
        kinds = {
            r["uuid"]: r["kind"]
            for r in s.run(
                "MATCH (n:Node) WHERE NOT n.type IN ['edge'] "
                "RETURN n.uuid AS uuid, n.type AS kind"
            )
            if r["uuid"]
        }
    candidates = [e for e in edges if is_candidate_relation(e[1])]
    return edges, candidates, kinds


def fetch_node_embeddings(uuids: list[str]) -> dict[str, np.ndarray]:
    from pymilvus import Collection, connections

    connections.connect(
        host=os.environ.get("MILVUS_HOST", "localhost"),
        port=os.environ.get("MILVUS_PORT", "19530"),
        timeout=20,
    )
    want = set(uuids)
    out: dict[str, np.ndarray] = {}
    for name in (
        "hcg_entity_embeddings",
        "hcg_concept_embeddings",
        "hcg_process_embeddings",
        "hcg_state_embeddings",
    ):
        col = Collection(name)
        col.load(timeout=120)
        ids = sorted(want - set(out))
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            expr = "uuid in [" + ",".join(f'"{u}"' for u in chunk) + "]"
            for row in col.query(expr=expr, output_fields=["uuid", "embedding"]):
                out[row["uuid"]] = np.asarray(row["embedding"], dtype=np.float32)
    return out


def label_text(rel: str) -> str:
    return rel.lower().replace("_", " ")


def fetch_label_embeddings(surfaces: list[str]) -> dict[str, np.ndarray]:
    """Label-text vectors: hermes_embeddings cache by `text`, then /embed_text
    for misses (which re-persists them — the cache warms itself)."""
    from pymilvus import Collection, connections

    connections.connect(
        host=os.environ.get("MILVUS_HOST", "localhost"),
        port=os.environ.get("MILVUS_PORT", "19530"),
        timeout=20,
    )
    texts = {label_text(s): s for s in surfaces}
    out: dict[str, np.ndarray] = {}
    col = Collection("hermes_embeddings")
    col.load(timeout=120)
    keys = sorted(texts)
    for i in range(0, len(keys), 300):
        chunk = keys[i : i + 300]
        safe = [t.replace("\\", "").replace('"', "") for t in chunk]
        expr = "text in [" + ",".join(f'"{t}"' for t in safe) + "]"
        for row in col.query(expr=expr, output_fields=["text", "embedding"]):
            s = texts.get(row["text"])
            if s is not None and s not in out:
                out[s] = np.asarray(row["embedding"], dtype=np.float32)

    missing = [s for s in surfaces if s not in out]
    print(
        f"[pilot2] label cache: {len(out)}/{len(surfaces)} hit, embedding {len(missing)}",
        file=sys.stderr,
    )
    if missing:
        import httpx

        async def embed_all() -> None:
            sem = asyncio.Semaphore(4)
            async with httpx.AsyncClient(timeout=60.0) as client:

                async def one(s: str) -> None:
                    async with sem:
                        try:
                            r = await client.post(
                                f"{HERMES_URL}/embed_text",
                                json={"text": label_text(s)},
                            )
                            r.raise_for_status()
                            emb = r.json().get("embedding")
                            if emb:
                                out[s] = np.asarray(emb, dtype=np.float32)
                        except Exception as exc:  # fail-soft: surface skipped
                            print(f"  [embed] {s}: {str(exc)[:60]}", file=sys.stderr)

                await asyncio.gather(*(one(s) for s in missing))

        asyncio.run(embed_all())
    return out


def main() -> None:
    t0 = time.time()
    driver = _driver()
    try:
        all_edges, candidates, kinds = fetch_graph(driver)
    finally:
        driver.close()

    data_kinds = {
        u: k for u, k in kinds.items() if k not in ("type_definition", "edge_type")
    }
    provenance = {
        "design": "run5 (run4 + magnitude hygiene + instance-level reversal veto)",
        "graph": "corpus_batch3 reference v2 (BASELINE.md row 6)",
        "pipeline": "merged main: hermes b75c3cd, sophia 6016f03",
        "consolidation": "pre (mapping.csv is gold)",
        "n_edges_total": len(all_edges),
        "n_edges_candidate": len(candidates),
        "n_nodes": len(data_kinds),
    }
    print(f"[pilot2] edges={len(all_edges)} candidates={len(candidates)}", file=sys.stderr)

    # Node embeddings -> induced roles -> role-rich snapshot
    node_uuids = sorted({e[2] for e in candidates} | {e[3] for e in candidates})
    node_emb = fetch_node_embeddings(node_uuids)
    node_cov = len(node_emb) / len(node_uuids) if node_uuids else 0.0
    out: dict = {"provenance": provenance, "coverage": {"node": round(node_cov, 4)}}
    if node_cov < NODE_COVERAGE_HARD_STOP:
        out["verdict"] = f"HARD STOP: node coverage {node_cov:.1%}"
        _write(out)
        return
    membership, parents = role_membership(
        node_emb, data_kinds, whiten_top=5, normalize_first=True
    )
    n_roles = sum(1 for p in parents.values() if p is not None)
    print(f"[pilot2] roles induced: {n_roles}", file=sys.stderr)

    triples = [(rel, src, tgt) for _, rel, src, tgt in candidates]
    snapshot = Snapshot(membership=membership, type_parents=parents, edges=tuple(triples))

    # Endpoint-pair clusters over ROLE-tagged... (vectors stay embedding-level)
    pair_X, kept_idx = pair_vectors(triples, node_emb)
    pair_labels_kept = cluster_embeddings(pair_X) if len(kept_idx) else np.array([])
    pair_labels = np.full(len(triples), -2)
    for j, i in enumerate(kept_idx):
        pair_labels[i] = pair_labels_kept[j]

    # Per-surface signatures: directional ROLE-pair histograms (run 3 — the
    # confirmation substrate; reversible, enabling the converse veto) and
    # pair-cluster histograms (kept for senses).
    inst_by_surface: dict[str, list[int]] = defaultdict(list)
    sense_hist: dict[str, Counter] = defaultdict(Counter)
    for i, (rel, _, _) in enumerate(triples):
        inst_by_surface[rel].append(i)
        if pair_labels[i] >= 0:
            sense_hist[rel][int(pair_labels[i])] += 1
    hist: dict[str, Counter] = {
        rel: role_pair_hist(triples, idxs, membership)
        for rel, idxs in inst_by_surface.items()
    }
    node_pairs: dict[str, set] = {
        rel: {(triples[i][1], triples[i][2]) for i in idxs}
        for rel, idxs in inst_by_surface.items()
    }

    # Label-embedding proposals
    surfaces = sorted(
        s for s, idxs in inst_by_surface.items() if len(idxs) >= MIN_SURFACE_INSTANCES
    )
    label_emb = fetch_label_embeddings(surfaces)
    label_cov = len(label_emb) / len(surfaces) if surfaces else 0.0
    out["coverage"]["label"] = round(label_cov, 4)
    emb_surfaces = [s for s in surfaces if s in label_emb]
    L = np.vstack([label_emb[s] for s in emb_surfaces]) if emb_surfaces else np.empty((0, 0))
    lab = (
        cluster_embeddings(L, min_cluster_size=2, min_samples=1)
        if len(emb_surfaces)
        else np.array([])
    )
    clusters: dict[int, list[str]] = defaultdict(list)
    for s, c in zip(emb_surfaces, lab):
        if c >= 0:
            clusters[int(c)].append(s)
    print(
        f"[pilot2] label clusters: {len(clusters)} over {len(emb_surfaces)} surfaces",
        file=sys.stderr,
    )

    # Funnel: propose -> confirm (pair-hist cosine) -> adjudicate (dL)
    funnel = {"proposed": 0, "confirmed": 0, "accepted": 0}
    merge_records = []
    accepted_pairs: list[tuple[str, str]] = []
    accepted_dl = 0.0
    for c, members in sorted(clusters.items()):
        counts = {s: len(inst_by_surface[s]) for s in members}
        winner = max(members, key=lambda s: counts[s])
        for loser in members:
            if loser == winner:
                continue
            funnel["proposed"] += 1
            # Belt-and-braces (run 4): the GRANULAR brake (run-2's
            # pair-cluster histograms) must agree AND the role-pair converse
            # veto must not fire. Run 3 proved a brake is only as good as its
            # substrate — so the working one stays connected while roles
            # mature.
            sim = pair_hist_similarity(sense_hist[loser], sense_hist[winner])
            vetoed = directional_veto(
                hist[loser], hist[winner]
            ) or instance_reversal_veto(node_pairs[loser], node_pairs[winner])
            confirmed = sim >= SIM_CONFIRM_BAR and not vetoed
            d = None
            accepted = False
            if confirmed:
                funnel["confirmed"] += 1
                d = delta(snapshot, merge_relations(loser, winner))
                accepted = d < 0
                if accepted:
                    funnel["accepted"] += 1
                    accepted_dl += d
                    accepted_pairs.append((loser, winner))
            merge_records.append(
                {
                    "loser": loser,
                    "winner": winner,
                    "pair_sim": round(sim, 3),
                    "converse_veto": vetoed,
                    "confirmed": confirmed,
                    "dL": None if d is None else round(d, 2),
                    "accepted": accepted,
                }
            )

    # Senses per surface: >=2 pair-clusters with mass — split the largest
    # minority sense, adjudicated on the role-rich snapshot.
    split_records = []
    multi_sense_split_found = False
    for s, h in sense_hist.items():
        senses = [(k, m) for k, m in h.most_common() if m >= MIN_SENSE_MASS]
        if len(senses) < 2:
            continue
        minority_sense = senses[1][0]
        idxs = [
            i
            for i in inst_by_surface[s]
            if pair_labels[i] == minority_sense
        ]
        if len(idxs) < MIN_SENSE_MASS:
            continue
        op = split_relation(s, idxs, f"{s}~SENSE{minority_sense}")
        d = delta(snapshot, op)
        accepted = d < 0
        if accepted:
            accepted_dl += d
            multi_sense_split_found = True
        split_records.append(
            {"surface": s, "sense_mass": len(idxs), "dL": round(d, 2), "accepted": accepted}
        )

    # Induced classes from accepted merges (union-find) -> gold eval
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in accepted_pairs:
        parent[find(a)] = find(b)
    classes: dict[str, set[str]] = defaultdict(set)
    for s in parent:
        classes[find(s)].add(s)
    induced = [c for c in classes.values() if len(c) > 1]

    import csv

    gold_pairs = []
    with open(MAPPING_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("predicate") and row.get("proposed_target"):
                gold_pairs.append((row["predicate"], row["proposed_target"]))
    m_eval = mapping_eval(induced, gold_pairs)
    # Gold-overlap guard (run 3): the mapping speaks the OLD vocabulary; raw
    # P/R is structurally zero when surfaces are disjoint. Restrict to pairs
    # whose surfaces BOTH exist in the gold vocabulary.
    gold_vocab = {a.lower() for a, _ in gold_pairs} | {b.lower() for _, b in gold_pairs}
    graph_vocab = {s.lower() for s in surfaces}
    overlap = gold_vocab & graph_vocab
    induced_in_gold = [
        c for c in induced if sum(1 for s in c if s.lower() in gold_vocab) >= 2
    ]
    m_eval_restricted = mapping_eval(induced_in_gold, gold_pairs)

    out.update(
        {
            "n_roles": n_roles,
            "n_surfaces_considered": len(surfaces),
            "n_label_clusters": len(clusters),
            "funnel": funnel,
            "merges": sorted(
                merge_records, key=lambda r: (not r["accepted"], r.get("dL") or 0)
            )[:60],
            "splits": split_records[:20],
            "mapping_eval": {
                k: (v if not isinstance(v, list) else v[:25]) for k, v in m_eval.items()
            },
            "gold_overlap": {
                "gold_vocab": len(gold_vocab),
                "graph_vocab": len(graph_vocab),
                "shared_surfaces": len(overlap),
                "restricted_eval": {
                    k: (v if not isinstance(v, list) else v[:10])
                    for k, v in m_eval_restricted.items()
                },
            },
            "hand_label_request": [
                f"{a} -> {b}" for a, b in accepted_pairs
            ],
            "gate": {
                "multi_sense_split_found": multi_sense_split_found,
                "aggregate_dL_accepted": round(accepted_dl, 1),
                "pass": multi_sense_split_found and accepted_dl < 0,
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    _write(out)
    print(
        f"[pilot2] funnel={funnel} gate_pass={out['gate']['pass']} "
        f"P={m_eval['precision']:.3f} R={m_eval['recall']:.3f}",
        file=sys.stderr,
    )


def _write(out: dict) -> None:
    ws = Path(__file__).resolve().parent / "workspace"
    ws.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = ws / f"run_{ts}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[pilot2] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
