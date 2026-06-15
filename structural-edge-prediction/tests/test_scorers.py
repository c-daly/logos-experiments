"""scorers: the signature separates blowhole from gills; the marginal does not.

The toy positive control. blowhole (whale_anatomy) and gills (fish_anatomy) are
equally frequent GLOBALLY (6 has-edges each), so the type-blind marginal cannot
tell them apart for narwhal. The whale signature can: whales source
has->whale_anatomy with high probability; has->fish_anatomy is unseen at the
whale level and only reachable by a discounted backoff up to the animal level.
"""

from __future__ import annotations

from harness.scorers import build_train_graph, score_marginal, score_signature
from harness.signature import build_signatures


def _graph(toy_snapshot):
    sigs = build_signatures(toy_snapshot.edges, toy_snapshot)
    return build_train_graph(toy_snapshot, toy_snapshot.edges, sigs)


def test_signature_ranks_blowhole_above_gills(toy_snapshot):
    g = _graph(toy_snapshot)
    s_blowhole = score_signature(g, "narwhal#1", "has", "blowhole#1")
    s_gills = score_signature(g, "narwhal#1", "has", "gills#1")
    # The true edge strongly dominates the type-confusable corruption.
    assert s_blowhole > s_gills
    assert s_blowhole > 0.0
    # gills is only reachable by a discounted animal-level backoff, so it
    # scores far below the direct whale-level blowhole hit.
    assert s_gills < 0.5 * s_blowhole


def test_marginal_does_not_separate_blowhole_from_gills(toy_snapshot):
    g = _graph(toy_snapshot)
    m_blowhole = score_marginal(g, "narwhal#1", "has", "blowhole#1")
    m_gills = score_marginal(g, "narwhal#1", "has", "gills#1")
    # whale_anatomy and fish_anatomy each appear in exactly 6 has-edges
    # globally, so the type-blind marginal scores them identically -- near
    # chance, no source-type signal.
    assert m_blowhole == m_gills


def test_marginal_and_signature_disagree_on_the_control(toy_snapshot):
    g = _graph(toy_snapshot)
    # The marginal cannot rank the true edge above the fake; the signature can.
    sig_gap = score_signature(g, "narwhal#1", "has", "blowhole#1") - score_signature(
        g, "narwhal#1", "has", "gills#1"
    )
    marg_gap = score_marginal(g, "narwhal#1", "has", "blowhole#1") - score_marginal(
        g, "narwhal#1", "has", "gills#1"
    )
    assert sig_gap > marg_gap
    assert marg_gap == 0.0
