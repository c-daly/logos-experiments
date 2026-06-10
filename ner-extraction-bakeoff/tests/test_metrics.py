import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import prf, score_entities


def test_prf_basic():
    p, r, f = prf(pred={"a", "b", "c"}, gold={"b", "c", "d"})
    assert p == 2 / 3
    assert r == 2 / 3
    assert round(f, 3) == 0.667


def test_prf_empty_pred_is_zero():
    assert prf(pred=set(), gold={"a"}) == (0.0, 0.0, 0.0)


def test_prf_empty_gold_pred_empty_is_perfect():
    assert prf(pred=set(), gold=set()) == (1.0, 1.0, 1.0)


def test_score_entities_matches_on_canonical_name():
    pred = [{"name": "Cheetahs", "type": "animal"}]
    gold = [{"name": "cheetah", "type": "animal"}]
    out = score_entities(pred, gold)
    assert out["recall"] == 1.0
    assert out["type_accuracy"] == 1.0


def test_score_entities_type_accuracy_independent_of_name_f1():
    pred = [{"name": "cheetah", "type": "vehicle"}]
    gold = [{"name": "cheetah", "type": "animal"}]
    out = score_entities(pred, gold)
    assert out["f1"] == 1.0
    assert out["type_accuracy"] == 0.0
