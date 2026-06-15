"""Offline harness: structural edge prediction.

Holds out a fraction of the relational edges, builds type signatures + the
marginal baseline from the SURVIVING train edges only (the IS_A backbone is
never held out), then ranks each held-out positive against ``neg_per_pos``
corrupted negatives under every available arm. Writes a deterministic snapshot
that ``eval/metrics.py::compute_metrics`` scores into ``[METRIC]`` lines.

Negative sampling is the standard FILTERED protocol (a corrupted edge that
already exists anywhere in the graph is rejected), with a tunable fraction of
HARD negatives: corruptions whose dst type is a SIBLING of the true dst type
under the same parent (e.g. blowhole -> gills, both anatomy). Hard negatives
are the type-confusable fakes the marginal cannot tell apart but the signature
can \u2014 they are what makes the test discriminating rather than trivial.

Runs offline on the toy fixture \u2014 no LLM, no Neo4j, no Milvus, no training.

Run (toy):
    uv run python -m harness.run_experiment --snapshot fixtures/toy_graph.json

The real 350-block graph is exported by ``harness/freeze_snapshot.py`` (the
only file that touches the live stack), then fed to the same command.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

# Direct/module execution puts harness/ on sys.path; shim the experiment root
# in so the absolute ``harness.*`` imports resolve (same role as conftest.py).
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from harness.scorers import (  # noqa: E402
    TrainGraph,
    build_train_graph,
    score_embedding_knn,
    score_marginal,
    score_signature,
    score_structural_knn,
)
from harness.signature import as_dict, build_signatures  # noqa: E402
from harness.snapshot_io import (  # noqa: E402
    Snapshot,
    load_snapshot,
    node_type,
)


def split_edges(
    edges: list[dict[str, str]], test_frac: float, rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Uniformly hold out ``test_frac`` of the edges (seeded, deterministic).

    Returns ``(train_edges, test_edges)``. The split sorts a copy first so the
    shuffle order depends only on the seed, never on input ordering.
    """
    ordered = sorted(edges, key=lambda e: (e["src"], e["rel"], e["dst"]))
    rng.shuffle(ordered)
    n_test = max(1, int(round(len(ordered) * test_frac))) if ordered else 0
    test = ordered[:n_test]
    train = ordered[n_test:]
    return train, test


def _edge_key(e: dict[str, str]) -> tuple[str, str, str]:
    return (e["src"], e["rel"], e["dst"])


def _sibling_type_nodes(
    snapshot: Snapshot, true_dst: str
) -> list[str]:
    """Nodes whose type is a SIBLING of ``true_dst``\u0027s type (same parent).

    These are the type-confusable hard-negative targets: a different subtype
    under the same parent (e.g. fish_anatomy vs whale_anatomy under anatomy).
    Excludes ``true_dst`` and nodes of its own type. Empty if the type has no
    parent or no siblings.
    """
    dst_type = node_type(snapshot, true_dst)
    if dst_type is None:
        return []
    parent = snapshot.type_parents.get(dst_type)
    if parent is None:
        return []
    siblings = {
        t
        for t, p in snapshot.type_parents.items()
        if p == parent and t != dst_type
    }
    if not siblings:
        return []
    return sorted(
        n["id"]
        for n in snapshot.nodes
        if n.get("type") in siblings and n["id"] != true_dst
    )


def corrupt_dst(
    pos: dict[str, str],
    snapshot: Snapshot,
    all_edge_keys: set[tuple[str, str, str]],
    node_ids: list[str],
    neg_per_pos: int,
    rng: random.Random,
    *,
    hard_neg_frac: float = 0.6,
) -> list[dict[str, str]]:
    """Generate ``neg_per_pos`` dst-corrupted negatives (filtered protocol).

    First fills up to ``hard_neg_frac`` of the slots with HARD negatives \u2014 a
    sibling-type dst (type-confusable). Remaining slots are random filtered
    corruptions. Every corrupted edge must NOT exist anywhere in the full edge
    set; self-corruption back to the true dst is skipped. Deterministic.
    """
    src, rel, true_dst = pos["src"], pos["rel"], pos["dst"]
    negs: list[dict[str, str]] = []
    used: set[str] = set()

    def _try_add(cand: str) -> bool:
        if cand in used or cand == true_dst or cand == src:
            return False
        if (src, rel, cand) in all_edge_keys:
            return False
        used.add(cand)
        negs.append({"src": src, "rel": rel, "dst": cand})
        return True

    n_hard = int(round(neg_per_pos * hard_neg_frac))
    hard_pool = _sibling_type_nodes(snapshot, true_dst)
    rng.shuffle(hard_pool)
    for cand in hard_pool:
        if len(negs) >= n_hard:
            break
        _try_add(cand)

    # Fill the remaining slots with random filtered corruptions.
    rand_pool = [n for n in node_ids if n not in used]
    rng.shuffle(rand_pool)
    for cand in rand_pool:
        if len(negs) >= neg_per_pos:
            break
        _try_add(cand)

    return negs


def _score_arm(
    arm: str, graph: TrainGraph, e: dict[str, str], k: int
) -> float | None:
    """Dispatch a single candidate edge to one arm\u0027s scorer."""
    src, rel, dst = e["src"], e["rel"], e["dst"]
    if arm == "A0_marginal":
        return score_marginal(graph, src, rel, dst)
    if arm == "A1_signature":
        return score_signature(graph, src, rel, dst)
    if arm == "A2_embedding_knn":
        return score_embedding_knn(graph, src, rel, dst, k=k)
    if arm == "A3_structural_knn":
        return score_structural_knn(graph, src, rel, dst, k=k)
    raise ValueError(f"unknown arm {arm!r}")


def run(
    snapshot: Snapshot,
    *,
    test_frac: float = 0.2,
    neg_per_pos: int = 5,
    seed: int = 0,
    k: int = 10,
    hard_neg_frac: float = 0.6,
) -> dict[str, Any]:
    """Run the held-out-edge recovery experiment; return a snapshot dict.

    Deterministic given ``seed``. Builds signatures/marginals from train edges
    only; the IS_A backbone (node types + hierarchy) is fully retained.
    """
    rng = random.Random(seed)
    train_edges, test_edges = split_edges(snapshot.edges, test_frac, rng)

    all_edge_keys = {_edge_key(e) for e in snapshot.edges}
    node_ids = sorted(n["id"] for n in snapshot.nodes)

    signatures = build_signatures(train_edges, snapshot)
    graph = build_train_graph(snapshot, train_edges, signatures)

    arms = ["A0_marginal", "A1_signature"]
    if snapshot.has_embeddings():
        arms.append("A2_embedding_knn")
    else:
        print("[run] no embeddings in snapshot \u2014 skipping A2_embedding_knn")
    arms.append("A3_structural_knn")

    # One ranking group per held-out positive: [positive, neg, neg, ...].
    groups: list[dict[str, Any]] = []
    for pos in sorted(test_edges, key=_edge_key):
        negs = corrupt_dst(
            pos,
            snapshot,
            all_edge_keys,
            node_ids,
            neg_per_pos,
            rng,
            hard_neg_frac=hard_neg_frac,
        )
        src_type = node_type(snapshot, pos["src"])
        group: dict[str, Any] = {
            "positive": pos,
            "src_type": src_type,
            "n_neg": len(negs),
            "scores": {},
        }
        for arm in arms:
            pos_score = _score_arm(arm, graph, pos, k)
            if pos_score is None:
                continue
            neg_scores = [_score_arm(arm, graph, ng, k) for ng in negs]
            neg_scores = [s for s in neg_scores if s is not None]
            group["scores"][arm] = {
                "positive": pos_score,
                "negatives": neg_scores,
            }
        groups.append(group)

    sig_export = {t: as_dict(sig) for t, sig in sorted(signatures.items())}

    return {
        "params": {
            "test_frac": test_frac,
            "neg_per_pos": neg_per_pos,
            "seed": seed,
            "k": k,
            "hard_neg_frac": hard_neg_frac,
        },
        "arms": arms,
        "n_train_edges": len(train_edges),
        "n_test_edges": len(test_edges),
        "groups": groups,
        "signatures": sig_export,
    }


def _summarize(run_snapshot: dict[str, Any]) -> None:
    """Print a short human summary (full metrics live in eval/metrics.py)."""
    print(
        "[run] arms={arms} train_edges={tr} test_edges={te} groups={g}".format(
            arms=",".join(run_snapshot["arms"]),
            tr=run_snapshot["n_train_edges"],
            te=run_snapshot["n_test_edges"],
            g=len(run_snapshot["groups"]),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="path to snapshot JSON")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10, help="kNN neighbourhood size")
    parser.add_argument(
        "--hard-neg-frac",
        type=float,
        default=0.6,
        help="fraction of negatives drawn as sibling-type hard negatives",
    )
    parser.add_argument(
        "--out",
        default=str(EXP / "workspace" / "run.json"),
        help="output snapshot path",
    )
    args = parser.parse_args(argv)

    snapshot = load_snapshot(args.snapshot)
    run_snapshot = run(
        snapshot,
        test_frac=args.test_frac,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
        k=args.k,
        hard_neg_frac=args.hard_neg_frac,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(run_snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    _summarize(run_snapshot)
    print(f"[run] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
