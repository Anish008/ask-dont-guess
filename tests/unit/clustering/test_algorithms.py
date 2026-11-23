"""Test clustering algorithms."""

from adg.clustering.algorithms import cluster_samples
from adg.core.types import PairwiseRelations, Sample


def test_cluster_samples_greedy_bidirectional():
    """Test greedy bidirectional clustering."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="London", sample_id=2),
    ]

    # Create relations: 0-1 equivalent, 2 separate
    scores = [
        [1.0, 1.0, 0.1],
        [1.0, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ]
    equivalence = [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=3,
        n_pairs_total=3,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="greedy_bidirectional")

    # Should produce 2 clusters
    assert result.n_clusters == 2
    assert len(result.clusters) == 2
    assert result.algorithm == "greedy_bidirectional"

    # Clusters should be sorted by size descending
    assert result.clusters[0].size >= result.clusters[1].size

    # Largest cluster should have samples 0 and 1
    assert result.clusters[0].size == 2
    assert set(result.clusters[0].sample_ids) == {0, 1}

    # Smallest cluster should have sample 2
    assert result.clusters[1].size == 1
    assert result.clusters[1].sample_ids == [2]

    # Singletons count
    assert result.singletons == 1


def test_cluster_samples_agglomerative():
    """Test agglomerative hierarchical clustering."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="London", sample_id=2),
        Sample(interpretation="I4", answer="Berlin", sample_id=3),
    ]

    # Create relations: 0-1 equivalent, 2 and 3 separate
    scores = [
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.2],
        [0.1, 0.1, 0.2, 1.0],
    ]
    equivalence = [
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=6,
        n_pairs_total=6,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="agglomerative")

    # Should produce 3 clusters (0-1 merged, 2 alone, 3 alone)
    assert result.n_clusters == 3
    assert result.algorithm == "agglomerative"

    # Find the largest cluster (should contain 0 and 1)
    largest_cluster = result.clusters[0]
    assert largest_cluster.size == 2
    assert set(largest_cluster.sample_ids) == {0, 1}


def test_cluster_samples_connected_components():
    """Test connected components clustering."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="Paris", sample_id=2),
        Sample(interpretation="I4", answer="London", sample_id=3),
    ]

    # Create relations: 0-1-2 form a connected component (transitive)
    # 0-1 equivalent, 1-2 equivalent (so 0-2 should be in same cluster via transitivity)
    scores = [
        [1.0, 1.0, 0.4, 0.1],
        [1.0, 1.0, 1.0, 0.1],
        [0.4, 1.0, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ]
    equivalence = [
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, False],
        [False, False, False, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=6,
        n_pairs_total=6,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="connected_components")

    # Should produce 2 clusters: {0, 1, 2} and {3}
    assert result.n_clusters == 2
    assert result.algorithm == "connected_components"

    # Largest cluster should contain 0, 1, 2 (via transitivity)
    largest_cluster = result.clusters[0]
    assert largest_cluster.size == 3
    assert set(largest_cluster.sample_ids) == {0, 1, 2}

    # Smallest cluster should contain 3
    smallest_cluster = result.clusters[1]
    assert smallest_cluster.size == 1
    assert smallest_cluster.sample_ids == [3]


def test_cluster_samples_all_separate():
    """Test when all samples are in separate clusters."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="London", sample_id=1),
        Sample(interpretation="I3", answer="Berlin", sample_id=2),
    ]

    # Create relations: no equivalences
    scores = [
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ]
    equivalence = [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=3,
        n_pairs_total=3,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="greedy_bidirectional")

    # Should produce 3 clusters, all singletons
    assert result.n_clusters == 3
    assert result.singletons == 3

    for cluster in result.clusters:
        assert cluster.size == 1


def test_cluster_samples_all_together():
    """Test when all samples are in one cluster."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="Paris", sample_id=2),
    ]

    # Create relations: all equivalent
    scores = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    equivalence = [
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=3,
        n_pairs_total=3,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="connected_components")

    # Should produce 1 cluster
    assert result.n_clusters == 1
    assert result.singletons == 0
    assert result.clusters[0].size == 3
    assert set(result.clusters[0].sample_ids) == {0, 1, 2}


def test_cluster_samples_empty():
    """Test with empty sample list."""
    samples: list[Sample] = []

    relations = PairwiseRelations(
        scores=[],
        equivalence=[],
        n_pairs_computed=0,
        n_pairs_total=0,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="greedy_bidirectional")

    assert result.n_clusters == 0
    assert result.singletons == 0
    assert len(result.clusters) == 0


def test_cluster_samples_single():
    """Test with single sample."""
    samples = [Sample(interpretation="I1", answer="Paris", sample_id=0)]

    relations = PairwiseRelations(
        scores=[[1.0]],
        equivalence=[[True]],
        n_pairs_computed=0,
        n_pairs_total=0,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="agglomerative")

    assert result.n_clusters == 1
    assert result.singletons == 1
    assert result.clusters[0].size == 1
    assert result.clusters[0].sample_ids == [0]


def test_cluster_samples_representative():
    """Test that representative answer and interpretation are set."""
    samples = [
        Sample(interpretation="Interp1", answer="Paris", sample_id=0),
        Sample(interpretation="Interp2", answer="Paris", sample_id=1),
    ]

    scores = [[1.0, 1.0], [1.0, 1.0]]
    equivalence = [[True, True], [True, True]]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=1,
        n_pairs_total=1,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="greedy_bidirectional")

    # Should have 1 cluster with 2 samples
    assert result.n_clusters == 1
    cluster = result.clusters[0]

    # Representative should be from first sample in cluster
    assert cluster.representative_answer == "Paris"
    assert cluster.representative_interpretation in ["Interp1", "Interp2"]


def test_cluster_samples_cluster_id_renumbering():
    """Test that cluster IDs are renumbered after sorting."""
    samples = [
        Sample(interpretation="I1", answer="A", sample_id=0),
        Sample(interpretation="I2", answer="B", sample_id=1),
        Sample(interpretation="I3", answer="B", sample_id=2),
        Sample(interpretation="I4", answer="B", sample_id=3),
    ]

    # B cluster (1,2,3) is larger than A cluster (0)
    scores = [
        [1.0, 0.1, 0.1, 0.1],
        [0.1, 1.0, 1.0, 1.0],
        [0.1, 1.0, 1.0, 1.0],
        [0.1, 1.0, 1.0, 1.0],
    ]
    equivalence = [
        [True, False, False, False],
        [False, True, True, True],
        [False, True, True, True],
        [False, True, True, True],
    ]

    relations = PairwiseRelations(
        scores=scores,
        equivalence=equivalence,
        n_pairs_computed=6,
        n_pairs_total=6,
        n_guard_applied=0,
        mode="soft",
    )

    result = cluster_samples(samples, relations, algorithm="greedy_bidirectional")

    # Clusters should be sorted by size descending
    assert result.clusters[0].cluster_id == 0
    assert result.clusters[1].cluster_id == 1

    # Largest cluster (B) should be first
    assert result.clusters[0].size == 3
    assert result.clusters[1].size == 1
