"""W2 scoring + null rankings (logos-experiments#27).

Two nulls, per the program plan:
- random ranking over the same candidate set (chance under the candidate
  restriction);
- recency: rank pairs by how recently their endpoints were active -- the
  bursty-ingest baseline topology has to beat to claim the *shape* knows
  anything beyond "recent things stay busy".
Preferential attachment (in predict.py) serves as the degree-only
configuration-model null.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence

Pair = frozenset


def precision_at_k(
    ranked: Sequence[tuple[Pair, float]], test_pairs: set[Pair], ks: Iterable[int]
) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in ks:
        prefix = ranked[:k]
        if not prefix:
            out[k] = 0.0
            continue
        hits = sum(1 for p, _ in prefix if p in test_pairs)
        out[k] = hits / len(prefix)
    return out


def auc_sampled(
    score: Callable[[Pair], float],
    test_pairs: set[Pair],
    negatives: Sequence[Pair],
    seed: int = 0,
    n_samples: int = 10_000,
) -> float:
    """P(score(positive) > score(negative)) over sampled pos/neg pairs,
    ties counted half (standard rank-AUC)."""
    rng = random.Random(seed)
    pos = sorted(test_pairs, key=lambda p: tuple(sorted(p)))
    if not pos or not negatives:
        return 0.5
    wins = ties = 0
    for _ in range(n_samples):
        sp = score(rng.choice(pos))
        sn = score(rng.choice(negatives))
        if sp > sn:
            wins += 1
        elif sp == sn:
            ties += 1
    return (wins + 0.5 * ties) / n_samples


def random_ranking(candidates: Iterable[Pair], seed: int) -> list[tuple[Pair, float]]:
    cands = sorted(candidates, key=lambda p: tuple(sorted(p)))
    rng = random.Random(seed)
    rng.shuffle(cands)
    return [(p, 0.0) for p in cands]


def recency_ranking(
    candidates: Iterable[Pair], last_seen: dict[str, int]
) -> list[tuple[Pair, float]]:
    """Rank by the most recent activity among the pair's endpoints
    (last_seen: node -> last training-edge index)."""
    scored = [
        (p, float(max(last_seen.get(n, -1) for n in p))) for p in candidates
    ]
    return sorted(scored, key=lambda ps: (-ps[1], tuple(sorted(ps[0]))))
