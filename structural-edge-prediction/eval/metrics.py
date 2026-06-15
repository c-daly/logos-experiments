"""Score a run snapshot into ``[METRIC] key=value`` lines.

Reads ``workspace/run.json`` (the output of ``harness/run_experiment.py``) and
computes, per arm, the filtered ranking metrics \u2014 AUC, hits@1, hits@10, MRR
\u2014 plus the cross-arm deltas the goal.yaml cares about and the
sharpness/accuracy correlation across types.

AUC, MRR, hits@k, Shannon entropy, and Spearman correlation are all
implemented in PLAIN stdlib (no numpy/scipy) \u2014 small helpers below. Mirrors
the ``[METRIC] key=value`` precedent of the sibling experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

# Per-type hits@10 requires at least this many test positives for a type to
# enter the sharpness/accuracy correlation (a 1-positive type is noise).
MIN_TYPE_POSITIVES = 2


# --------------------------------------------------------------------------- #
# stdlib statistical helpers
# --------------------------------------------------------------------------- #
def auc_pos_vs_neg(positive: float, negatives: list[float]) -> float | None:
    """P(positive outranks a random negative) within one ranking group.

    Ties count as 0.5 (the standard AUC tie convention). ``None`` if the group
    has no negatives. Averaged across groups, this is the rank-AUC.
    """
    if not negatives:
        return None
    wins = sum(1.0 for n in negatives if positive > n)
    ties = sum(1.0 for n in negatives if positive == n)
    return (wins + 0.5 * ties) / len(negatives)


def rank_of_positive(positive: float, negatives: list[float]) -> int:
    """1-based rank of the positive among [positive + negatives], desc.

    Ties are pessimistic: a negative equal to the positive is counted as
    outranking it (rank pushed down). Lowest rank is best (=1).
    """
    return 1 + sum(1 for n in negatives if n >= positive)


def reciprocal_rank(positive: float, negatives: list[float]) -> float:
    """Reciprocal of the positive\u0027s pessimistic rank."""
    return 1.0 / rank_of_positive(positive, negatives)


def hits_at_k(positive: float, negatives: list[float], k: int) -> float:
    """1.0 if the positive\u0027s pessimistic rank is <= k, else 0.0."""
    return 1.0 if rank_of_positive(positive, negatives) <= k else 0.0


def mean(xs: list[float]) -> float:
    """Arithmetic mean (0.0 for an empty list)."""
    return sum(xs) / len(xs) if xs else 0.0


def _rankdata(values: list[float]) -> list[float]:
    """Fractional (average) ranks of ``values`` \u2014 1-based, ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average of the tie block
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (Pearson on fractional ranks).

    ``None`` if fewer than two pairs or if either ranked series is constant
    (zero variance \u2014 correlation undefined). Clamped to [-1, 1] to absorb
    float rounding. Plain stdlib.
    """
    if len(xs) != len(ys):
        raise ValueError("spearman_corr: length mismatch")
    n = len(xs)
    if n < 2:
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return None
    corr = cov / (vx ** 0.5 * vy ** 0.5)
    # Clamp away float drift (a perfect monotone pair can read as 0.99999...).
    return max(-1.0, min(1.0, corr))


# --------------------------------------------------------------------------- #
# metric assembly
# --------------------------------------------------------------------------- #
def _arm_metrics(groups: list[dict[str, Any]], arm: str) -> dict[str, float]:
    """Filtered AUC / hits@1 / hits@10 / MRR for one arm across all groups."""
    aucs: list[float] = []
    h1: list[float] = []
    h10: list[float] = []
    rr: list[float] = []
    for g in groups:
        sc = g["scores"].get(arm)
        if sc is None:
            continue
        pos = sc["positive"]
        negs = sc["negatives"]
        auc = auc_pos_vs_neg(pos, negs)
        if auc is None:
            continue
        aucs.append(auc)
        h1.append(hits_at_k(pos, negs, 1))
        h10.append(hits_at_k(pos, negs, 10))
        rr.append(reciprocal_rank(pos, negs))
    return {
        "AUC": mean(aucs),
        "hits@1": mean(h1),
        "hits@10": mean(h10),
        "MRR": mean(rr),
        "n": float(len(aucs)),
    }


def _per_type_hits10(
    groups: list[dict[str, Any]], arm: str
) -> dict[str, float]:
    """Mean hits@10 of ``arm`` per source type, for types with enough positives."""
    by_type: dict[str, list[float]] = {}
    for g in groups:
        sc = g["scores"].get(arm)
        if sc is None or g.get("src_type") is None:
            continue
        by_type.setdefault(g["src_type"], []).append(
            hits_at_k(sc["positive"], sc["negatives"], 10)
        )
    return {
        t: mean(v)
        for t, v in by_type.items()
        if len(v) >= MIN_TYPE_POSITIVES
    }


def compute_metrics(run: dict[str, Any]) -> None:
    """Print every ``[METRIC] key=value`` line for a run snapshot."""
    groups = run["groups"]
    arms = run["arms"]

    arm_stats: dict[str, dict[str, float]] = {}
    for arm in arms:
        stats = _arm_metrics(groups, arm)
        arm_stats[arm] = stats
        for key in ("AUC", "hits@1", "hits@10", "MRR"):
            print(f"[METRIC] {arm}.{key}={round(stats[key], 4)}")

    a0 = arm_stats.get("A0_marginal", {}).get("AUC")
    a1 = arm_stats.get("A1_signature", {}).get("AUC")
    a2 = arm_stats.get("A2_embedding_knn", {}).get("AUC")

    if a1 is not None:
        print(f"[METRIC] signature_auc={round(a1, 4)}")
    if a0 is not None and a1 is not None:
        print(f"[METRIC] signature_auc_minus_marginal={round(a1 - a0, 4)}")
    if a1 is not None and a2 is not None:
        print(f"[METRIC] signature_minus_embedding_auc={round(a1 - a2, 4)}")

    # Sharpness <-> accuracy: per-type signature sharpness vs per-type hits@10.
    per_type_h10 = _per_type_hits10(groups, "A1_signature")
    sig_export = run.get("signatures", {})
    sharp_xs: list[float] = []
    acc_ys: list[float] = []
    for t in sorted(per_type_h10):
        sig = sig_export.get(t)
        if sig is None:
            continue
        sharpness = float(sig["sharpness"])
        sharp_xs.append(sharpness)
        acc_ys.append(per_type_h10[t])
        print(f"[METRIC] type.{t}.hits@10={round(per_type_h10[t], 4)}")
        print(f"[METRIC] type.{t}.sharpness={round(sharpness, 4)}")

    corr = spearman_corr(sharp_xs, acc_ys) if len(sharp_xs) >= 2 else None
    if corr is not None:
        print(f"[METRIC] sharpness_accuracy_corr={round(corr, 4)}")
    else:
        print("[METRIC] sharpness_accuracy_corr=nan")


def main() -> None:
    path = EXP / "workspace" / "run.json"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    if not path.exists():
        print("[METRIC] error=no_run_snapshot_found")
        return
    run = json.loads(path.read_text(encoding="utf-8"))
    compute_metrics(run)


if __name__ == "__main__":
    main()
