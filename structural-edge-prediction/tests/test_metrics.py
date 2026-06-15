"""metrics: AUC / MRR / hits@k / Spearman on tiny hand-computed inputs."""

from __future__ import annotations

import math

from eval.metrics import (
    auc_pos_vs_neg,
    compute_metrics,
    hits_at_k,
    rank_of_positive,
    reciprocal_rank,
    spearman_corr,
)


def test_auc_clean_win():
    # positive beats both negatives outright.
    assert auc_pos_vs_neg(1.0, [0.0, 0.5]) == 1.0


def test_auc_tie_counts_as_half():
    # one strict win (0.5>0.0) + one tie (0.5==0.5) over 2 negatives.
    assert auc_pos_vs_neg(0.5, [0.5, 0.0]) == 0.75


def test_auc_none_without_negatives():
    assert auc_pos_vs_neg(1.0, []) is None


def test_rank_and_reciprocal_rank():
    assert rank_of_positive(1.0, [0.0, 0.5]) == 1
    # pessimistic ties: both negatives >= 0.5 push the positive to rank 3.
    assert rank_of_positive(0.5, [0.5, 0.9]) == 3
    assert reciprocal_rank(1.0, [0.0, 0.5]) == 1.0
    assert reciprocal_rank(0.5, [0.5, 0.9]) == 1.0 / 3.0


def test_hits_at_k():
    assert hits_at_k(1.0, [0.0, 0.5], 1) == 1.0
    assert hits_at_k(0.5, [0.5, 0.9], 1) == 0.0
    assert hits_at_k(0.5, [0.5, 0.9], 10) == 1.0


def test_spearman_perfect_monotone():
    # Float math leaves a perfect monotone pair at 1.0 within tolerance.
    assert math.isclose(
        spearman_corr([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0, rel_tol=1e-9
    )
    assert math.isclose(
        spearman_corr([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0, rel_tol=1e-9
    )


def test_spearman_with_a_tie():
    # xs has a tie block -> fractional ranks [1, 2.5, 2.5, 4]; ys monotone.
    corr = spearman_corr([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert corr is not None
    assert math.isclose(corr, 0.9486832980505138, rel_tol=1e-9)


def test_spearman_constant_series_is_none():
    # zero variance in ys -> correlation undefined.
    assert spearman_corr([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None
    assert spearman_corr([1.0], [2.0]) is None


def test_compute_metrics_runs_on_a_minimal_run(capsys):
    # A hand-built two-group run: A1 perfectly ranks, A0 is at chance.
    run = {
        "arms": ["A0_marginal", "A1_signature"],
        "groups": [
            {
                "src_type": "whale",
                "scores": {
                    "A0_marginal": {"positive": 0.5, "negatives": [0.5]},
                    "A1_signature": {"positive": 1.0, "negatives": [0.0]},
                },
            },
            {
                "src_type": "whale",
                "scores": {
                    "A0_marginal": {"positive": 0.5, "negatives": [0.5]},
                    "A1_signature": {"positive": 1.0, "negatives": [0.0]},
                },
            },
        ],
        "signatures": {
            "whale": {"type_id": "whale", "sharpness": -1.0, "support": 4},
        },
    }
    compute_metrics(run)
    out = capsys.readouterr().out
    assert "[METRIC] A1_signature.AUC=1.0" in out
    assert "[METRIC] A0_marginal.AUC=0.5" in out
    assert "[METRIC] signature_auc_minus_marginal=0.5" in out
