"""W0.3b mapping proposer (logos-experiments#34, epic logos#557).

Proposes a consolidation target for every df=1 predicate, with evidence
and a confidence tier. Output feeds mapping.csv for human review --
NOTHING here is applied to the graph; the review column decides.

Three evidence passes, strongest first; a predicate takes the first that
fires:

  high   -- canonical-form collision: the crude stemmer folds the one-off
            onto an existing predicate (or onto a group of fellow
            one-offs, which consolidate to the group's lexicographically
            first member).
  medium -- content-token match against a head predicate (df >= HEAD_DF):
            the head's content tokens are a subset of the one-off's
            (lossy compound: AFFECTS_PURCHASING_POWER_OF -> AFFECTS), or
            Jaccard >= TOKEN_JACCARD.
  low    -- signature match: a head predicate has edges with the same
            (source-type, target-type) pair as the one-off's single edge.
  keep   -- no evidence; the predicate may be genuinely distinct.

The stemmer is deliberately crude (recall over precision -- a human
reviews every row); it is deterministic and guards short tokens and -SS.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

STOPWORDS = {
    "OF", "TO", "BY", "IN", "ON", "AT", "FOR", "WITH", "THE", "A", "AN",
    "AS", "UP", "FROM", "DURING", "INTO", "OVER", "UNDER",
}
# negation/modality markers: folding these onto a positive head flips
# meaning (DOES_NOT_REFER_TO -> REFERS_TO). Never auto-map; force review.
POLARITY_MARKERS = {"NOT", "NO", "NEVER", "CANNOT", "CANT", "WITHOUT", "NON"}
HEAD_DF = 3
TOKEN_JACCARD = 0.34
# tokens ending in these are not plurals -- never S-strip (ALIAS, BASIS,
# CHAOS, VIRUS, ...); same family as the hermes canonicalize() guards
NO_S_STRIP = ("SS", "US", "IS", "AS", "OS")
# realm-fallback kinds: a signature pair made only of these is
# non-discriminative (it matches half the graph) and is not evidence
REALM_KINDS = {"entity", "concept", "process", "action", "state", "plan", "goal", "?"}


@dataclass(frozen=True)
class Edge:
    relation: str
    src_type: str
    tgt_type: str


@dataclass
class Row:
    predicate: str
    df: int
    target: str
    tier: str  # high | medium | low | keep
    evidence: str
    review: str = field(default="")


def fold_token(t: str) -> str:
    """Crude deterministic stem: -ING/-IES/-ED/-S suffixes, then stem-E."""
    if len(t) >= 6 and t.endswith("ING"):
        t = t[:-3]
    elif len(t) >= 5 and t.endswith("IES"):
        t = t[:-3] + "Y"
    elif len(t) >= 5 and t.endswith("ED"):
        t = t[:-2]
    elif len(t) >= 4 and t.endswith("S") and not t.endswith(NO_S_STRIP):
        t = t[:-1]
    if len(t) >= 5 and t.endswith("E"):
        t = t[:-1]
    return t


def canon(predicate: str) -> str:
    return "_".join(fold_token(t) for t in predicate.split("_"))


def content_tokens(predicate: str) -> set[str]:
    return {
        fold_token(t) for t in predicate.split("_") if t not in STOPWORDS
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def propose_mappings(edges: list[Edge]) -> list[Row]:
    df = Counter(e.relation for e in edges)
    one_offs = sorted(r for r, c in df.items() if c == 1)
    heads = sorted(r for r, c in df.items() if c >= HEAD_DF)

    canon_of = {r: canon(r) for r in df}
    by_canon: dict[str, list[str]] = defaultdict(list)
    for r in sorted(df):
        by_canon[canon_of[r]].append(r)

    head_tokens = {h: content_tokens(h) for h in heads}
    head_pairs: dict[str, Counter] = defaultdict(Counter)
    for e in edges:
        if e.relation in head_tokens:
            head_pairs[e.relation][(e.src_type, e.tgt_type)] += 1
    pairs_of: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for e in edges:
        pairs_of[e.relation].add((e.src_type, e.tgt_type))

    rows: list[Row] = []
    for p in one_offs:
        if set(p.split("_")) & POLARITY_MARKERS:
            # a negated/modal predicate must not be folded onto its
            # positive form; surface it for review rather than mapping it.
            rows.append(Row(p, 1, "", "keep", "polarity marker -- review (do not auto-map)"))
            continue
        group = by_canon[canon_of[p]]
        established = [q for q in group if q != p and df[q] > 1]
        fellows = [q for q in group if q != p and df[q] == 1]

        if established:
            target = max(established, key=lambda q: (df[q], q))
            rows.append(Row(p, 1, target, "high", f"canon collision: {canon_of[p]}"))
            continue
        if fellows:
            # the group's representative is its shortest member (the base
            # form beats inflections; lexicographic tie-break)
            representative = min(group, key=lambda r: (len(r), r))
            if representative != p:
                rows.append(
                    Row(
                        p, 1, representative, "high",
                        f"canon group of one-offs: {canon_of[p]}",
                    )
                )
                continue

        ptoks = content_tokens(p)
        best, best_j, lossy = "", 0.0, False
        for h in heads:
            htoks = head_tokens[h]
            j = jaccard(ptoks, htoks)
            subset = bool(htoks) and htoks <= ptoks
            if subset or j >= TOKEN_JACCARD:
                score = max(j, 0.999 if subset and j < TOKEN_JACCARD else j)
                if score > best_j or (score == best_j and h < best):
                    best, best_j, lossy = h, score, bool(ptoks - htoks)
        if best:
            note = " (lossy: extra tokens dropped)" if lossy else ""
            rows.append(
                Row(p, 1, best, "medium", f"token match vs {best}, j={best_j:.2f}{note}")
            )
            continue

        sig_hits: list[tuple[int, str, tuple[str, str]]] = []
        for h in heads:
            for pair in pairs_of[p]:
                if set(pair) <= REALM_KINDS:
                    continue  # realm-only pairs are not evidence
                if head_pairs[h][pair]:
                    sig_hits.append((head_pairs[h][pair], h, pair))
        if sig_hits:
            count, h, pair = max(sig_hits, key=lambda x: (x[0], x[1]))
            rows.append(
                Row(
                    p, 1, h, "low",
                    f"signature: ({pair[0]}->{pair[1]}) seen in {h} x{count}",
                )
            )
            continue

        rows.append(Row(p, 1, "", "keep", ""))

    tier_order = {"high": 0, "medium": 1, "low": 2, "keep": 3}
    rows.sort(key=lambda r: (tier_order[r.tier], r.predicate))
    return rows
