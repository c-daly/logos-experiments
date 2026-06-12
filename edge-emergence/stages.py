"""Pure pilot stages for edge-emergence P1 (lx#46).

Everything here is graph-free and unit-testable: candidate filtering
(constitutive-tier exclusion), clustering wrappers, endpoint-pair vectors,
cross-strata agreement, gold-mapping evaluation, and the two ledger Ops the
pilot adjudicates with (relation merge / sense split), composed against the
mdl-ledger Snapshot contract (lx#26).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

# Constitutive tier (design §2a): never candidates. Typing relations carry
# the spine; the reserved namespace is the grammar's own vocabulary.
TYPING_RELATIONS = {"IS_A", "INSTANCE_OF", "SUBTYPE_OF"}


def is_candidate_relation(rel: str) -> bool:
    """Empirical-tier relations only — the tier rule, stated once."""
    if rel in TYPING_RELATIONS:
        return False
    if rel.startswith("_"):
        return False
    return True


def cluster_embeddings(
    vectors: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """HDBSCAN over L2-normalized vectors; -1 = unclassed (the allowed tail)."""
    if len(vectors) < min_cluster_size:
        return np.full(len(vectors), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(
        normalized
    ).labels_


def pair_vectors(
    edges: Sequence[tuple[str, str, str]],
    node_embeddings: dict[str, np.ndarray],
    *,
    normalize_endpoints: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Joint (from ⊕ to) endpoint vectors per edge — a sense is defined by
    the PAIRING, so endpoints are clustered jointly, never independently.

    Returns (vectors, kept_indices); edges with missing endpoint embeddings
    are skipped and reported by the caller as coverage loss.
    """
    vecs: list[np.ndarray] = []
    kept: list[int] = []
    for i, (_, src, tgt) in enumerate(edges):
        se, te = node_embeddings.get(src), node_embeddings.get(tgt)
        if se is None or te is None:
            continue
        if normalize_endpoints:
            # Each endpoint contributes equally to the joint direction —
            # otherwise one high-magnitude endpoint owns the pair vector
            # before the post-concat L2 ever sees it (run 5).
            se = se / max(float(np.linalg.norm(se)), 1e-12)
            te = te / max(float(np.linalg.norm(te)), 1e-12)
        vecs.append(np.concatenate([se, te]))
        kept.append(i)
    if not vecs:
        return np.empty((0, 0)), []
    return np.vstack(vecs), kept


def agreement_ari(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Cross-strata agreement: ARI between two partitions of the same items."""
    return float(adjusted_rand_score(labels_a, labels_b))


def mapping_eval(
    induced_classes: Iterable[set[str]],
    gold_pairs: Iterable[tuple[str, str]],
    *,
    normalize: Callable[[str], str] = lambda s: s.lower(),
) -> dict:
    """Precision/recall of induced equivalences vs the approved mapping.

    Match unit = unordered normalized surface pair. FN/FP lists are returned
    verbatim — they are the interesting reading (spec, lx#46).
    """

    def pairs_of(cls: set[str]) -> set[frozenset[str]]:
        names = sorted(normalize(x) for x in cls)
        return {
            frozenset((a, b))
            for i, a in enumerate(names)
            for b in names[i + 1 :]
            if a != b
        }

    induced: set[frozenset[str]] = set()
    for cls in induced_classes:
        induced |= pairs_of(set(cls))
    gold = {frozenset((normalize(a), normalize(b))) for a, b in gold_pairs}
    gold = {p for p in gold if len(p) == 2}

    tp_pairs = induced & gold
    tp = len(tp_pairs)
    precision = tp / len(induced) if induced else 0.0
    recall = tp / len(gold) if gold else 0.0
    return {
        "tp": tp,
        "n_induced_pairs": len(induced),
        "n_gold_pairs": len(gold),
        "precision": precision,
        "recall": recall,
        "fn": sorted(tuple(sorted(p)) for p in gold - induced),
        "fp": sorted(tuple(sorted(p)) for p in induced - gold),
    }


# -- ledger Ops (compose with mdl-ledger's Snapshot/delta contract) ----------


@dataclass(frozen=True)
class _Op:
    """Matches mdl-ledger's Op duck type: a pure callable Snapshot -> Snapshot
    (delta() invokes the op directly); .apply kept as the readable alias."""

    name: str
    apply: Callable[[Any], Any]

    def __call__(self, s: Any) -> Any:
        return self.apply(s)


def merge_relations(loser: str, winner: str) -> _Op:
    """Label-equivalence rule: every loser-surface edge re-labels to winner.

    Degree-zero logical induction — a merge is a compression move (one fewer
    16-bit relation surface) accepted only if delta(s, op) < 0.
    """

    def apply(s: Any) -> Any:
        edges = tuple(
            (winner if rel == loser else rel, src, tgt) for rel, src, tgt in s.edges
        )
        return type(s)(
            membership=dict(s.membership),
            type_parents=dict(s.type_parents),
            edges=edges,
        )

    return _Op(name=f"merge_relations({loser}->{winner})", apply=apply)


def split_relation(rel: str, edge_indices: Sequence[int], new_name: str) -> _Op:
    """Sense split: re-label the selected instances of *rel* as *new_name*.

    Pays +16 bits of model for the new surface; accepted only when the
    sense's tighter target-type distribution earns it back in edge terms —
    the marginal-information story, adjudicated by the ledger.
    """
    chosen = set(edge_indices)

    def apply(s: Any) -> Any:
        edges = tuple(
            (new_name if (i in chosen and r == rel) else r, src, tgt)
            for i, (r, src, tgt) in enumerate(s.edges)
        )
        return type(s)(
            membership=dict(s.membership),
            type_parents=dict(s.type_parents),
            edges=edges,
        )

    return _Op(name=f"split_relation({rel}->{new_name}, n={len(chosen)})", apply=apply)


def role_membership(
    node_embeddings: dict[str, "np.ndarray"],
    kinds: dict[str, str],
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    pca_dims: int | None = 50,
    whiten_top: int | None = 5,
    normalize_first: bool = False,
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Induced endpoint roles for the ledger's membership (run-2 fix).

    Run 1 proved the ledger blind on realm-flat membership: its only
    semantic brake is -log2 P(tgt_type | rel), which is uninformative when
    every node sits at a realm root. Cluster node embeddings into roles;
    unclassed/unembedded nodes keep their realm kind — the antecedent
    fallback (a dissolved or unearned role drops to the antecedent, never
    to nowhere). Roles are parented under their dominant realm, realms are
    roots: marginal bits per hop.
    """
    ids = sorted(node_embeddings)
    membership: dict[str, str] = dict(kinds)
    parents: dict[str, str | None] = {k: None for k in set(kinds.values())}
    if not ids:
        return membership, parents
    X = np.vstack([node_embeddings[u] for u in ids])
    if normalize_first:
        # Magnitude hygiene (run 5): norms correlate with text length and
        # frequency, not meaning — without this, centering and the ABTT SVD
        # are dominated by high-norm vectors. Standard ABTT order is
        # normalize -> center -> drop.
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    # Run 3: plain PCA kept the embedding space's dominant COMMON directions
    # (anisotropy) — exactly the wrong ones — and roles still collapsed.
    # ABTT whitening (W9, lx#31) first: center, REMOVE the top components,
    # then reduce what remains.
    Xc = X - X.mean(axis=0, keepdims=True)
    if (whiten_top or pca_dims) and Xc.shape[0] > (whiten_top or 0) + 2:
        vt = np.linalg.svd(Xc, full_matrices=False)[2]
        if whiten_top:
            top = vt[:whiten_top]
            Xc = Xc - (Xc @ top.T) @ top
            vt = vt[whiten_top:]
        if pca_dims is not None and vt.shape[0] > pca_dims:
            Xc = Xc @ vt[:pca_dims].T
    X = Xc
    labels = cluster_embeddings(
        X, min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    by_role: dict[int, list[str]] = {}
    for u, lab in zip(ids, labels):
        if lab >= 0:
            by_role.setdefault(int(lab), []).append(u)
    for lab, members in by_role.items():
        role = f"role_{lab}"
        realm_counts = Counter(kinds.get(u, "entity") for u in members)
        parents[role] = realm_counts.most_common(1)[0][0]
        for u in members:
            membership[u] = role
    return membership, parents


def pair_hist_similarity(h1: "Counter", h2: "Counter") -> float:
    """Cosine similarity of two endpoint-pair-cluster histograms.

    The run-2 confirmation step: two surfaces are merge candidates only if
    their instances live in similar (from, to) role-pair distributions —
    stratum-1 evidence that the surfaces mean the same relation.
    """
    keys = set(h1) | set(h2)
    if not keys or not h1 or not h2:
        return 0.0
    v1 = np.array([h1.get(k, 0) for k in sorted(keys)], dtype=float)
    v2 = np.array([h2.get(k, 0) for k in sorted(keys)], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / denom) if denom else 0.0


def role_pair_hist(
    triples: Sequence[tuple[str, str, str]],
    idxs: Sequence[int],
    membership: dict[str, str],
) -> "Counter":
    """Directional (from_role, to_role) histogram for a set of edge instances.

    Run-3 confirmation substrate: role-pair tuples are computable in both
    directions for free, which the pair-CLUSTER ids of run 2 were not —
    enabling the converse veto.
    """
    h: Counter = Counter()
    for i in idxs:
        _, src, tgt = triples[i]
        rs, rt = membership.get(src), membership.get(tgt)
        if rs is not None and rt is not None:
            h[(rs, rt)] += 1
    return h


def directional_veto(h_a: "Counter", h_b: "Counter") -> bool:
    """True when B reads as A's CONVERSE: A matches reversed-B better than B.

    Catches PHASE_TRANSITION_TO vs _FROM, WITHIN vs BETWEEN (run-2's error
    signature). Same-polarity antonyms (INCREASES/DECREASES) remain a known
    residual — direction statistics cannot see negation.
    """
    rev_b: Counter = Counter({(t, s): n for (s, t), n in h_b.items()})
    return pair_hist_similarity(h_a, rev_b) > pair_hist_similarity(h_a, h_b)


def instance_reversal_veto(
    pairs_a: set[tuple[str, str]], pairs_b: set[tuple[str, str]]
) -> bool:
    """True when B's instances are A's REVERSED node pairs.

    Run 4: the role-pair converse veto is blind to converses whose domain
    and range are the SAME role (PHASE_TRANSITION_TO/FROM is substance ->
    substance both ways). At instance level the signature is unmistakable:
    the same (src, tgt) uuids appear flipped.
    """
    rev_b = {(t, s) for s, t in pairs_b}
    reversed_overlap = len(pairs_a & rev_b)
    forward_overlap = len(pairs_a & pairs_b)
    return reversed_overlap > forward_overlap and reversed_overlap > 0
