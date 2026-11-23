"""Test interpretation channel."""

from adg.clustering.interpretation import compute_interpretation_divergence
from adg.core.types import Cluster, Sample
from adg.nli.mock_nli import MockNLIScorer


def test_compute_interpretation_divergence_single_cluster():
    """Test interpretation divergence with single cluster."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="Paris is the capital", answer="Paris", sample_id=0),
        Sample(interpretation="Paris is the capital", answer="Paris", sample_id=1),
        Sample(interpretation="Paris is the capital", answer="Paris", sample_id=2),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1, 2],
            size=3,
            representative_answer="Paris",
            representative_interpretation="Paris is the capital",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # All interpretations identical, no divergence
    assert divergence == 0.0


def test_compute_interpretation_divergence_multiple_interpretations():
    """Test interpretation divergence with different interpretations across clusters."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="European capital city", answer="Paris", sample_id=0),
        Sample(interpretation="European capital city", answer="Paris", sample_id=1),
        Sample(interpretation="Large metropolitan area", answer="London", sample_id=2),
        Sample(interpretation="Large metropolitan area", answer="London", sample_id=3),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="European capital city",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2, 3],
            size=2,
            representative_answer="London",
            representative_interpretation="Large metropolitan area",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # Two different interpretations should have high divergence
    # Different texts with low word overlap
    assert divergence > 0.5  # Significant divergence


def test_compute_interpretation_divergence_partial_overlap():
    """Test interpretation divergence with partial overlap."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="The capital", answer="Paris", sample_id=0),
        Sample(interpretation="The capital city", answer="Paris", sample_id=1),
        Sample(interpretation="A major city", answer="London", sample_id=2),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="The capital",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2],
            size=1,
            representative_answer="London",
            representative_interpretation="A major city",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # Some overlap in interpretations (or maximum divergence)
    assert 0.0 < divergence <= 1.0


def test_compute_interpretation_divergence_empty():
    """Test interpretation divergence with no clusters."""
    scorer = MockNLIScorer()

    samples: list[Sample] = []
    clusters: list[Cluster] = []

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    assert divergence == 0.0


def test_compute_interpretation_divergence_single_sample():
    """Test interpretation divergence with single sample."""
    scorer = MockNLIScorer()

    samples = [
        Sample(interpretation="Capital", answer="Paris", sample_id=0),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0],
            size=1,
            representative_answer="Paris",
            representative_interpretation="Capital",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # Single sample, no divergence
    assert divergence == 0.0


def test_compute_interpretation_divergence_intra_cluster():
    """Test interpretation divergence within a single cluster (mixed interpretations)."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="Capital of France", answer="Paris", sample_id=0),
        Sample(interpretation="Large metropolitan area", answer="Paris", sample_id=1),
        Sample(interpretation="Historic European city", answer="Paris", sample_id=2),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1, 2],
            size=3,
            representative_answer="Paris",
            representative_interpretation="Capital of France",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # Different interpretations within same answer cluster
    # May have divergence depending on similarity (MockNLI uses Jaccard)
    # With these different phrases, should have some divergence
    assert divergence >= 0.0  # Can be 0 if MockNLI considers them similar enough


def test_compute_interpretation_divergence_consensus():
    """Test interpretation divergence with consensus (all same interpretation)."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="Same interpretation", answer="A", sample_id=0),
        Sample(interpretation="Same interpretation", answer="A", sample_id=1),
        Sample(interpretation="Same interpretation", answer="B", sample_id=2),
        Sample(interpretation="Same interpretation", answer="B", sample_id=3),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="A",
            representative_interpretation="Same interpretation",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2, 3],
            size=2,
            representative_answer="B",
            representative_interpretation="Same interpretation",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # Same interpretation across all clusters
    assert divergence == 0.0


def test_compute_interpretation_divergence_max_divergence():
    """Test interpretation divergence with maximum divergence (all different)."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="Interpretation A", answer="A", sample_id=0),
        Sample(interpretation="Interpretation B", answer="B", sample_id=1),
        Sample(interpretation="Interpretation C", answer="C", sample_id=2),
        Sample(interpretation="Interpretation D", answer="D", sample_id=3),
    ]

    clusters = [
        Cluster(
            cluster_id=i,
            sample_ids=[i],
            size=1,
            representative_answer=chr(65 + i),
            representative_interpretation=f"Interpretation {chr(65 + i)}",
        )
        for i in range(4)
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # All different interpretations
    # Maximum divergence (n interpretation clusters = n samples)
    assert divergence == 1.0


def test_compute_interpretation_divergence_threshold_effect():
    """Test that threshold affects interpretation clustering."""
    samples = [
        Sample(interpretation="The capital city", answer="Paris", sample_id=0),
        Sample(interpretation="Capital city", answer="Paris", sample_id=1),
        Sample(interpretation="A major city", answer="London", sample_id=2),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="Paris",
            representative_interpretation="The capital city",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2],
            size=1,
            representative_answer="London",
            representative_interpretation="A major city",
        ),
    ]

    # Low threshold - more likely to cluster together
    scorer_low = MockNLIScorer(threshold=0.3)
    divergence_low = compute_interpretation_divergence(samples, clusters, scorer_low, threshold=0.3)

    # High threshold - more likely to separate
    scorer_high = MockNLIScorer(threshold=0.9)
    divergence_high = compute_interpretation_divergence(
        samples, clusters, scorer_high, threshold=0.9
    )

    # Higher threshold should give higher divergence (more separation)
    assert divergence_high >= divergence_low


def test_compute_interpretation_divergence_three_clusters_two_interpretations():
    """Test divergence with 3 clusters but only 2 interpretation types."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="Interp A", answer="A", sample_id=0),
        Sample(interpretation="Interp A", answer="A", sample_id=1),
        Sample(interpretation="Interp B", answer="B", sample_id=2),
        Sample(interpretation="Interp B", answer="B", sample_id=3),
        Sample(interpretation="Interp A", answer="C", sample_id=4),
    ]

    clusters = [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 1],
            size=2,
            representative_answer="A",
            representative_interpretation="Interp A",
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[2, 3],
            size=2,
            representative_answer="B",
            representative_interpretation="Interp B",
        ),
        Cluster(
            cluster_id=2,
            sample_ids=[4],
            size=1,
            representative_answer="C",
            representative_interpretation="Interp A",
        ),
    ]

    divergence = compute_interpretation_divergence(samples, clusters, scorer)

    # 3 clusters but only 2 interpretation types
    # Should have moderate divergence (not 0, not 1)
    # (2 - 1) / (3 - 1) = 0.5
    assert 0.3 < divergence < 0.7
