"""Pure scoring for the NER/RE bake-off (logos-experiments#38).

Entity and relation matching is on CANONICAL forms (hermes.canonical) so
surface variation (case/plural/morphology) is not penalized. No network.
"""

from __future__ import annotations

from hermes.canonical import (
    canonicalize,
    canonicalize_predicate,  # noqa: F401 — staged for relation scoring (next task)
)


def prf(*, pred: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1 over two sets. Empty/empty == perfect."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def _cname(name: str) -> str:
    return canonicalize(name or "")


def score_entities(pred: list[dict], gold: list[dict]) -> dict:
    """Entity precision/recall/F1 over canonical names + type-accuracy on
    the names that matched."""
    pred_names = {_cname(e["name"]) for e in pred if e.get("name")}
    gold_names = {_cname(e["name"]) for e in gold if e.get("name")}
    p, r, f = prf(pred=pred_names, gold=gold_names)

    gold_type = {_cname(e["name"]): e.get("type", "") for e in gold if e.get("name")}
    pred_type = {_cname(e["name"]): e.get("type", "") for e in pred if e.get("name")}
    matched = pred_names & gold_names
    type_ok = sum(1 for n in matched if pred_type.get(n) == gold_type.get(n))
    type_accuracy = type_ok / len(matched) if matched else (1.0 if not gold else 0.0)

    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "type_accuracy": type_accuracy,
        "pred_count": len(pred_names),
        "gold_count": len(gold_names),
    }
