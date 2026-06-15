"""run_experiment: positive control (A1_AUC > A0_AUC) + determinism."""

from __future__ import annotations

import json

from eval.metrics import _arm_metrics
from harness.run_experiment import run


def _arm_auc(run_snapshot, arm):
    return _arm_metrics(run_snapshot["groups"], arm)["AUC"]


def test_positive_control_signature_beats_marginal(toy_snapshot):
    # On the deliberately-sharp toy graph, the type signature must recover
    # held-out edges better than the type-blind marginal.
    rs = run(toy_snapshot, test_frac=0.3, neg_per_pos=5, seed=0, k=5)
    a1 = _arm_auc(rs, "A1_signature")
    a0 = _arm_auc(rs, "A0_marginal")
    assert a1 > a0
    # Sanity floor: the signature clears well-above chance.
    assert a1 >= 0.6


def test_run_is_deterministic_across_two_runs(toy_snapshot):
    a = run(toy_snapshot, test_frac=0.3, neg_per_pos=5, seed=7, k=5)
    b = run(toy_snapshot, test_frac=0.3, neg_per_pos=5, seed=7, k=5)
    # Byte-identical serialisation given the same seed.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_a2_runs_when_embeddings_present(toy_snapshot):
    rs = run(toy_snapshot, test_frac=0.3, neg_per_pos=5, seed=0, k=5)
    assert "A2_embedding_knn" in rs["arms"]


def test_a2_skipped_when_embeddings_absent(toy_snapshot):
    # Strip embeddings -> A2 must drop out of the arm list.
    toy_snapshot.embeddings = {}
    rs = run(toy_snapshot, test_frac=0.3, neg_per_pos=5, seed=0, k=5)
    assert "A2_embedding_knn" not in rs["arms"]
