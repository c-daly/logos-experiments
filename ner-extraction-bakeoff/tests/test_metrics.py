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


from metrics import score_relation_labels, score_relation_links


def test_relation_links_ignore_label():
    pred = [{"source": "tusk", "relation": "GROWS_FROM", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_links(pred, gold)
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0


def test_relation_links_directional():
    pred = [{"source": "narwhal", "relation": "PART_OF", "target": "tusk"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_links(pred, gold)
    assert out["recall"] == 0.0


def test_relation_labels_require_canonical_relation_match():
    pred = [{"source": "tusk", "relation": "GROWS_FROM", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_labels(pred, gold)
    assert out["f1"] == 0.0


def test_relation_labels_match_on_canonical_predicate():
    pred = [{"source": "tusk", "relation": "parts of", "target": "narwhal"}]
    gold = [{"source": "tusk", "relation": "PART_OF", "target": "narwhal"}]
    out = score_relation_labels(pred, gold)
    assert out["recall"] == 1.0


from metrics import compactness


def test_compactness_counts_distinct_canonical_predicates():
    rels = [
        {"source": "a", "relation": "CARRIES", "target": "b"},
        {"source": "c", "relation": "carried", "target": "d"},
        {"source": "e", "relation": "PART_OF", "target": "f"},
    ]
    out = compactness(rels)
    assert out["distinct_predicates"] == 2
    assert out["total_relations"] == 3
    assert out["df1_fraction"] == 0.5


def test_compactness_empty():
    out = compactness([])
    assert out["distinct_predicates"] == 0 and out["df1_fraction"] == 0.0
