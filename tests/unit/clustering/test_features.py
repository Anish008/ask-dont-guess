"""Test cluster features computation."""

from adg.clustering.features import compute_aggregate_features
from adg.core.types import Cluster, ClusteringResult, PairwiseRelations, Sample, SamplingResult


def test_compute_aggregate_features_basic():
    """Test basic aggregate features computation."""
    # Create sampling result
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0, is_hedged=False),
        Sample(interpretation="I2", answer="Paris", sample_id=1, is_hedged=False),
        Sample(interpretation="I3", answer="London", sample_id=2, is_hedged=True),
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=3,
        n_valid=3,
        n_refusals=0,
        seed=42,
    )

    # Create clustering result: 2 clusters
    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="I1",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2],
            size=1,
            representative_answer="London",
            representative_interpretation="I3",
        ),
    ]
    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=2,
        algorithm="greedy_bidirectional",
        singletons=1,
    )

    # Create relations
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

    # Compute features
    features = compute_aggregate_features(
        sampling_result,
        clustering_result,
        relations,
        significance_rho=0.25,
        gamma=0.5,
        interpretation_divergence=0.0,
    )

    # Check basic counts
    assert features.n_clusters == 2
    assert features.singletons == 1

    # Check masses
    assert features.p1 == 2 / 3  # Largest cluster: 2/3
    assert features.p2 == 1 / 3  # Second cluster: 1/3

    # Check coverage (both clusters significant at rho=0.25)
    assert features.coverage == 1.0  # 2/3 + 1/3 = 1.0
    assert features.n_significant == 2

    # Check rates
    assert features.refusal_rate == 0.0
    assert features.invalid_rate == 0.0
    assert features.hedge_rate == 1 / 3  # 1 hedged out of 3

    # Check separation (should be low score between clusters)
    assert features.separation == 0.1

    # Check derived scores
    assert features.ask_score == 0.0  # (1 - 2/3)^0.5 * 0.0 = 0.0
    assert features.abstain_score == 0.0  # max(0, 0) * (1 - 2/3) = 0.0
    assert features.answer_score == 2 / 3  # 2/3 * (1 - 0.0) = 2/3


def test_compute_aggregate_features_single_cluster():
    """Test features with single cluster (consensus)."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0, is_hedged=False),
        Sample(interpretation="I2", answer="Paris", sample_id=1, is_hedged=False),
        Sample(interpretation="I3", answer="Paris", sample_id=2, is_hedged=False),
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=3,
        n_valid=3,
        n_refusals=0,
        seed=42,
    )

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1, 2],
            size=3,
            representative_answer="Paris",
            representative_interpretation="I1",
        ),
    ]
    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=1,
        algorithm="greedy_bidirectional",
        singletons=0,
    )

    scores = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    relations = PairwiseRelations(
        scores=scores,
        equivalence=[[True] * 3] * 3,
        n_pairs_computed=3,
        n_pairs_total=3,
        n_guard_applied=0,
        mode="soft",
    )

    features = compute_aggregate_features(
        sampling_result, clustering_result, relations, significance_rho=0.25
    )

    # Perfect consensus
    assert features.n_clusters == 1
    assert features.p1 == 1.0
    assert features.p2 == 0.0  # No second cluster
    assert features.coverage == 1.0
    assert features.n_significant == 1
    assert features.separation == 1.0  # No separation needed (< 2 significant clusters)


def test_compute_aggregate_features_many_singletons():
    """Test features with many singleton clusters (high disagreement)."""
    samples = [
        Sample(interpretation="I1", answer="A", sample_id=0, is_hedged=False),
        Sample(interpretation="I2", answer="B", sample_id=1, is_hedged=False),
        Sample(interpretation="I3", answer="C", sample_id=2, is_hedged=False),
        Sample(interpretation="I4", answer="D", sample_id=3, is_hedged=False),
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=4,
        n_valid=4,
        n_refusals=0,
        seed=42,
    )

    clusters = [
        Cluster(
            cluster_id=i,
            sample_ids=[i],
            size=1,
            representative_answer=f"{chr(65+i)}",
            representative_interpretation=f"I{i+1}",
        )
        for i in range(4)
    ]
    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=4,
        algorithm="greedy_bidirectional",
        singletons=4,
    )

    # All different
    scores = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    relations = PairwiseRelations(
        scores=scores,
        equivalence=[[i == j for j in range(4)] for i in range(4)],
        n_pairs_computed=6,
        n_pairs_total=6,
        n_guard_applied=0,
        mode="soft",
    )

    features = compute_aggregate_features(
        sampling_result, clustering_result, relations, significance_rho=0.25
    )

    # High disagreement
    assert features.n_clusters == 4
    assert features.singletons == 4
    assert features.p1 == 0.25  # Largest cluster: 1/4
    assert features.p2 == 0.25  # Second cluster: 1/4
    assert features.coverage == 1.0  # All clusters significant
    assert features.n_significant == 4


def test_compute_aggregate_features_with_refusals():
    """Test features with refusals."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0, is_hedged=False),
        Sample(interpretation="I2", answer="Paris", sample_id=1, is_hedged=False),
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=5,
        n_valid=2,
        n_refusals=3,  # 3 refusals
        seed=42,
    )

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="I1",
        ),
    ]
    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=1,
        algorithm="greedy_bidirectional",
        singletons=0,
    )

    scores = [[1.0, 1.0], [1.0, 1.0]]
    relations = PairwiseRelations(
        scores=scores,
        equivalence=[[True, True], [True, True]],
        n_pairs_computed=1,
        n_pairs_total=1,
        n_guard_applied=0,
        mode="soft",
    )

    features = compute_aggregate_features(
        sampling_result, clustering_result, relations, interpretation_divergence=0.0
    )

    # Check refusal rate
    assert features.refusal_rate == 3 / 5  # 3 refusals out of 5 samples
    assert features.invalid_rate == 0.0

    # Check abstain score (affected by refusal rate)
    # abstain_score = max(refusal_rate, invalid_rate) * (1 - p1)
    # = 0.6 * (1 - 1.0) = 0.0
    assert features.abstain_score == 0.0


def test_compute_aggregate_features_with_interpretation_divergence():
    """Test features with interpretation divergence."""
    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0, is_hedged=False),
        Sample(interpretation="I2", answer="Paris", sample_id=1, is_hedged=False),
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=2,
        n_valid=2,
        n_refusals=0,
        seed=42,
    )

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="I1",
        ),
    ]
    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=1,
        algorithm="greedy_bidirectional",
        singletons=0,
    )

    scores = [[1.0, 1.0], [1.0, 1.0]]
    relations = PairwiseRelations(
        scores=scores,
        equivalence=[[True, True], [True, True]],
        n_pairs_computed=1,
        n_pairs_total=1,
        n_guard_applied=0,
        mode="soft",
    )

    features = compute_aggregate_features(
        sampling_result,
        clustering_result,
        relations,
        gamma=0.5,
        interpretation_divergence=0.8,  # High divergence
    )

    # Check derived scores with divergence
    # ask_score = (1 - p1)^gamma * interpretation_divergence = (1 - 1.0)^0.5 * 0.8 = 0.0
    assert features.ask_score == 0.0

    # answer_score = p1 * (1 - interpretation_divergence) = 1.0 * (1 - 0.8) = 0.2
    assert abs(features.answer_score - 0.2) < 1e-10


def test_compute_aggregate_features_significance_threshold():
    """Test significance threshold (rho) filtering."""
    samples = [
        Sample(interpretation=f"I{i}", answer=f"A{i}", sample_id=i, is_hedged=False)
        for i in range(10)
    ]
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=10,
        n_valid=10,
        n_refusals=0,
        seed=42,
    )

    # 1 large cluster (5 samples), 5 small clusters (1 sample each)
    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1, 2, 3, 4],
            size=5,
            representative_answer="A0",
            representative_interpretation="I0",
        ),
    ] + [
        Cluster(
            cluster_id=i,
            sample_ids=[i + 4],
            size=1,
            representative_answer=f"A{i+4}",
            representative_interpretation=f"I{i+4}",
        )
        for i in range(1, 6)
    ]

    clustering_result = ClusteringResult(
        clusters=clusters,
        n_clusters=6,
        algorithm="greedy_bidirectional",
        singletons=5,
    )

    scores = [[1.0 if i == j else 0.0 for j in range(10)] for i in range(10)]
    relations = PairwiseRelations(
        scores=scores,
        equivalence=[[i == j for j in range(10)] for i in range(10)],
        n_pairs_computed=45,
        n_pairs_total=45,
        n_guard_applied=0,
        mode="soft",
    )

    # With rho=0.25, only cluster with size >= 2.5 is significant
    features = compute_aggregate_features(
        sampling_result, clustering_result, relations, significance_rho=0.25
    )

    # Only the large cluster (mass=0.5) is significant
    assert features.n_significant == 1
    assert features.p1 == 0.5
    assert features.coverage == 0.5  # Only significant cluster


def test_compute_aggregate_features_empty():
    """Test features with no samples."""
    samples: list[Sample] = []
    sampling_result = SamplingResult(
        samples=samples,
        n_samples=0,
        n_valid=0,
        n_refusals=0,
        seed=42,
    )

    clustering_result = ClusteringResult(
        clusters=[],
        n_clusters=0,
        algorithm="greedy_bidirectional",
        singletons=0,
    )

    relations = PairwiseRelations(
        scores=[],
        equivalence=[],
        n_pairs_computed=0,
        n_pairs_total=0,
        n_guard_applied=0,
        mode="soft",
    )

    features = compute_aggregate_features(sampling_result, clustering_result, relations)

    # All zeros
    assert features.n_clusters == 0
    assert features.p1 == 0.0
    assert features.p2 == 0.0
    assert features.coverage == 0.0
    assert features.separation == 1.0  # Default when < 2 significant clusters
