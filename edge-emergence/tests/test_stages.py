"""Unit tests for the pure pilot stages (lx#46) — synthetic fixtures only."""

import numpy as np
import pytest

from stages import (
    agreement_ari,
    cluster_embeddings,
    is_candidate_relation,
    mapping_eval,
    merge_relations,
    pair_vectors,
    split_relation,
)


class TestCandidateFilter:
    def test_typing_relations_excluded(self):
        for rel in ("IS_A", "INSTANCE_OF", "SUBTYPE_OF"):
            assert not is_candidate_relation(rel)

    def test_reserved_namespace_excluded(self):
        assert not is_candidate_relation("_RESERVED_THING")

    def test_descriptive_relation_included(self):
        assert is_candidate_relation("CHASES")


class TestClustering:
    def test_two_blobs_two_clusters(self):
        # Centers away from the origin: embeddings are never near-zero, and
        # L2 normalization of an origin-centered blob is pathological.
        rng = np.random.default_rng(0)
        ca, cb = np.zeros(8), np.zeros(8)
        ca[0] = 1.0
        cb[1] = 1.0
        a = ca + rng.normal(0.0, 0.02, size=(20, 8))
        b = cb + rng.normal(0.0, 0.02, size=(20, 8))
        labels = cluster_embeddings(np.vstack([a, b]))
        la, lb = set(labels[:20]), set(labels[20:])
        la.discard(-1)
        lb.discard(-1)
        assert len(la) == 1 and len(lb) == 1 and la != lb

    def test_uniform_noise_mostly_unclassed(self):
        rng = np.random.default_rng(1)
        labels = cluster_embeddings(rng.uniform(size=(40, 16)))
        assert (labels == -1).sum() >= 20


class TestPairVectors:
    def test_concatenates_endpoint_embeddings(self):
        node_emb = {"a": np.ones(4), "b": np.zeros(4)}
        edges = [("REL", "a", "b")]
        v, kept = pair_vectors(edges, node_emb)
        assert kept == [0]
        assert v.shape == (1, 8)
        # endpoints are unit-normalized before concat (run 5): ones(4) -> 0.5s,
        # the zero vector stays zero via the norm clip
        assert v[0][:4] == pytest.approx([0.5] * 4)
        assert (v[0][4:] == 0).all()

    def test_skips_edges_with_missing_embeddings(self):
        node_emb = {"a": np.ones(4)}
        v, kept = pair_vectors([("REL", "a", "missing")], node_emb)
        assert kept == [] and v.shape[0] == 0


class TestAgreement:
    def test_identical_partitions_ari_1(self):
        assert agreement_ari([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)

    def test_independent_partitions_ari_low(self):
        assert agreement_ari([0, 0, 1, 1], [0, 1, 0, 1]) < 0.5


class TestMappingEval:
    def test_precision_recall_against_gold_pairs(self):
        induced = [{"CHASES", "PURSUES"}, {"EATS", "DEVOURS", "CONSUMES"}]
        gold = [("chases", "pursues"), ("eats", "devours"), ("sips", "drinks")]
        out = mapping_eval(induced, gold)
        # induced pairs: chases-pursues, eats-devours, eats-consumes,
        # devours-consumes -> 4; gold hits: chases-pursues, eats-devours -> 2
        assert out["tp"] == 2
        assert out["precision"] == pytest.approx(2 / 4)
        assert out["recall"] == pytest.approx(2 / 3)
        assert ("drinks", "sips") in out["fn"]  # canonical sorted-pair form


class TestLedgerOps:
    def _snapshot(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mdl-ledger"))
        from ledger import Snapshot

        return Snapshot(
            membership={"n1": "animal", "n2": "animal", "n3": "place"},
            type_parents={"animal": None, "place": None},
            edges=(
                ("CHASES", "n1", "n2"),
                ("PURSUES", "n1", "n2"),
                ("LIVES_IN", "n1", "n3"),
            ),
        )

    def test_merge_relations_reduces_model_cost(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mdl-ledger"))
        from ledger import delta

        s = self._snapshot()
        d = delta(s, merge_relations("PURSUES", "CHASES"))
        # one fewer 16-bit relation surface; data terms shift less than that
        assert d < 0

    def test_split_relation_relabels_selected_edges(self):
        s = self._snapshot()
        op = split_relation("CHASES", [0], "CHASES_PREY")
        s2 = op.apply(s)
        rels = [r for r, _, _ in s2.edges]
        assert "CHASES_PREY" in rels and rels.count("CHASES") == 0


class TestRoleMembership:
    def test_clustered_nodes_get_roles_others_keep_realm(self):
        from stages import role_membership

        rng = np.random.default_rng(2)
        ca, cb = np.zeros(8), np.zeros(8)
        ca[0], cb[1] = 1.0, 1.0
        emb = {}
        for i in range(6):
            emb[f"a{i}"] = ca + rng.normal(0, 0.02, 8)
            emb[f"b{i}"] = cb + rng.normal(0, 0.02, 8)
        kinds = {**{f"a{i}": "entity" for i in range(6)},
                 **{f"b{i}": "concept" for i in range(6)},
                 "lone": "process"}  # no embedding -> realm fallback
        membership, parents = role_membership(emb, kinds, min_cluster_size=5, whiten_top=None)
        roles = {v for k, v in membership.items() if k != "lone"}
        assert len(roles) == 2 and all(r.startswith("role_") for r in roles)
        assert membership["lone"] == "process"  # antecedent fallback
        # every role parented under its dominant realm; realms are roots
        for r in roles:
            assert parents[r] in ("entity", "concept")
        assert parents["process"] is None


class TestPairHistSimilarity:
    def test_identical_distributions_similarity_1(self):
        from collections import Counter

        from stages import pair_hist_similarity

        h = Counter({1: 10, 2: 5})
        assert pair_hist_similarity(h, Counter(h)) == pytest.approx(1.0)

    def test_disjoint_distributions_similarity_0(self):
        from collections import Counter

        from stages import pair_hist_similarity

        assert pair_hist_similarity(Counter({1: 10}), Counter({2: 10})) == pytest.approx(0.0)

    def test_empty_histogram_is_zero(self):
        from collections import Counter

        from stages import pair_hist_similarity

        assert pair_hist_similarity(Counter(), Counter({1: 3})) == 0.0


class TestRolePairHist:
    def test_builds_directional_histograms(self):
        from collections import Counter

        from stages import role_pair_hist

        membership = {"a": "role_0", "b": "role_1", "c": "role_0"}
        triples = [("R", "a", "b"), ("R", "c", "b"), ("R", "b", "a")]
        fwd = role_pair_hist(triples, [0, 1, 2], membership)
        assert fwd == Counter(
            {("role_0", "role_1"): 2, ("role_1", "role_0"): 1}
        )


class TestDirectionalVeto:
    def test_converse_pair_vetoed(self):
        from collections import Counter

        from stages import directional_veto

        # A's pairs run role_0->role_1; B's run role_1->role_0 (converse)
        a = Counter({("r0", "r1"): 10})
        b = Counter({("r1", "r0"): 10})
        assert directional_veto(a, b) is True

    def test_same_direction_not_vetoed(self):
        from collections import Counter

        from stages import directional_veto

        a = Counter({("r0", "r1"): 10, ("r0", "r2"): 2})
        b = Counter({("r0", "r1"): 8, ("r0", "r2"): 3})
        assert directional_veto(a, b) is False


class TestRoleMembershipPCA:
    def test_high_dim_blobs_recover_roles_with_pca(self):
        from stages import role_membership

        rng = np.random.default_rng(3)
        dim = 200
        ca, cb = np.zeros(dim), np.zeros(dim)
        ca[0], cb[1] = 1.0, 1.0
        emb = {}
        for i in range(8):
            emb[f"a{i}"] = ca + rng.normal(0, 0.02, dim)
            emb[f"b{i}"] = cb + rng.normal(0, 0.02, dim)
        kinds = {**{f"a{i}": "entity" for i in range(8)},
                 **{f"b{i}": "concept" for i in range(8)}}
        membership, parents = role_membership(
            emb, kinds, min_cluster_size=5, pca_dims=10, whiten_top=None
        )
        roles = set(membership.values())
        assert len([r for r in roles if r.startswith("role_")]) == 2


class TestWhitenedRoles:
    def test_anisotropic_blobs_need_whitening(self):
        """Blob structure buried under dominant common directions (the 3-large
        anisotropy that collapsed roles in runs 2-3) is recovered when the
        top components are dropped (ABTT, W9/lx#31)."""
        from stages import role_membership

        rng = np.random.default_rng(4)
        dim = 60
        emb = {}
        kinds = {}
        for i in range(20):
            for blob, realm in ((0, "entity"), (1, "concept")):
                v = np.zeros(dim)
                v[10 + blob] = 1.0                      # the real role signal
                v += rng.normal(0, 0.02, dim)            # small noise
                v[:3] += rng.normal(0, 12.0, 3)           # dominant shared dirs
                emb[f"{realm}{i}"] = v
                kinds[f"{realm}{i}"] = realm
        membership, _ = role_membership(
            emb, kinds, min_cluster_size=8, pca_dims=10, whiten_top=3
        )
        roles = {v for v in membership.values() if v.startswith("role_")}
        assert len(roles) == 2
        # blobs must not be mixed: every entity-blob member shares one role
        a_roles = {membership[f"entity{i}"] for i in range(20)}
        b_roles = {membership[f"concept{i}"] for i in range(20)}
        assert len(a_roles) == 1 and len(b_roles) == 1 and a_roles != b_roles


class TestMagnitudeHygiene:
    def test_pair_vector_halves_contribute_equally(self):
        from stages import pair_vectors

        node_emb = {"big": np.array([100.0, 0.0]), "small": np.array([0.0, 0.01])}
        v, kept = pair_vectors([("R", "big", "small")], node_emb)
        assert kept == [0]
        half = v.shape[1] // 2
        assert np.linalg.norm(v[0][:half]) == pytest.approx(
            np.linalg.norm(v[0][half:])
        )

    def test_roles_recovered_despite_wild_magnitudes(self):
        from stages import role_membership

        rng = np.random.default_rng(5)
        dim = 60
        emb, kinds = {}, {}
        for i in range(20):
            for blob, realm in ((0, "entity"), (1, "concept")):
                v = np.zeros(dim)
                v[10 + blob] = 1.0
                v += rng.normal(0, 0.02, dim)
                v[:3] += rng.normal(0, 12.0, 3)      # anisotropy
                v *= rng.uniform(0.5, 80.0)           # wild magnitude spread
                emb[f"{realm}{i}"] = v
                kinds[f"{realm}{i}"] = realm
        membership, _ = role_membership(
            emb, kinds, min_cluster_size=8, pca_dims=10, whiten_top=3,
            normalize_first=True,
        )
        a = {membership[f"entity{i}"] for i in range(20)}
        b = {membership[f"concept{i}"] for i in range(20)}
        assert len(a) == 1 and len(b) == 1 and a != b
        assert all(r.startswith("role_") for r in a | b)


class TestInstanceReversalVeto:
    def test_reversed_instances_vetoed(self):
        from stages import instance_reversal_veto

        a = {("ice", "water"), ("graphite", "diamond")}
        b = {("water", "ice"), ("diamond", "graphite")}
        assert instance_reversal_veto(a, b) is True

    def test_shared_forward_instances_not_vetoed(self):
        from stages import instance_reversal_veto

        a = {("x", "y"), ("p", "q")}
        b = {("x", "y"), ("r", "s")}
        assert instance_reversal_veto(a, b) is False

    def test_disjoint_instances_not_vetoed(self):
        from stages import instance_reversal_veto

        assert instance_reversal_veto({("a", "b")}, {("c", "d")}) is False
