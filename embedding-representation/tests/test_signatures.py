from collections import Counter

import numpy as np

from signatures import (
    best_cut_silhouette_dm,
    nn_chunk_rate_dm,
    signature_distance_matrix,
    to_counter,
)


def test_to_counter_tallies_pairs():
    c = to_counter([["IS_A", "concept"], ["IS_A", "concept"], ["CAUSES", "process"]])
    assert c == Counter({("IS_A", "concept"): 2, ("CAUSES", "process"): 1})


def test_distance_matrix_is_zero_on_diagonal_and_symmetric():
    a = Counter({("IS_A", "concept"): 1})
    b = Counter({("IS_A", "concept"): 1})            # identical -> distance 0
    c = Counter({("CAUSES", "process"): 1})          # disjoint  -> distance 1
    dm = signature_distance_matrix([a, b, c])
    assert dm.shape == (3, 3)
    assert np.allclose(np.diag(dm), 0.0)
    assert np.allclose(dm, dm.T)
    assert dm[0, 1] == 0.0
    assert dm[0, 2] == 1.0


def test_nn_chunk_rate_dm_detects_chunk_clustering():
    # 4 points; 0&1 share chunk "A" and are each other's nearest neighbour.
    dm = np.array(
        [[0.0, 0.1, 0.9, 0.9],
         [0.1, 0.0, 0.9, 0.9],
         [0.9, 0.9, 0.0, 0.1],
         [0.9, 0.9, 0.1, 0.0]],
        dtype="float32",
    )
    out = nn_chunk_rate_dm(dm, ["A", "A", "B", "B"], k=1)
    assert out["nn_same_chunk"] == 1.0     # every NN is a chunk-mate
    assert out["ratio"] is not None and out["ratio"] > 1.0


def test_best_cut_silhouette_dm_finds_two_clusters():
    # two tight clusters, far apart -> best cut at k=2, positive silhouette
    dm = np.array(
        [[0.0, 0.05, 0.9, 0.9],
         [0.05, 0.0, 0.9, 0.9],
         [0.9, 0.9, 0.0, 0.05],
         [0.9, 0.9, 0.05, 0.0]],
        dtype="float32",
    )
    k, sil = best_cut_silhouette_dm(dm)
    assert k == 2
    assert sil > 0.5
