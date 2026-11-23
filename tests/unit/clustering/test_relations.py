"""Test pairwise relations computation."""

from adg.clustering.relations import compute_pairwise_relations
from adg.core.types import Sample
from adg.nli.mock_nli import MockNLIScorer


def test_compute_pairwise_relations_basic():
    """Test basic pairwise relations computation."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="London", sample_id=1),
        Sample(interpretation="I3", answer="Paris", sample_id=2),
    ]

    result = compute_pairwise_relations(samples, scorer, query_text="What is the capital?")

    # Check structure
    assert len(result.scores) == 3
    assert len(result.equivalence) == 3
    assert result.n_pairs_total == 3  # 3 * 2 / 2

    # Check diagonal
    for i in range(3):
        assert result.scores[i][i] == 1.0
        assert result.equivalence[i][i] is True

    # Check symmetry
    for i in range(3):
        for j in range(3):
            assert result.scores[i][j] == result.scores[j][i]
            assert result.equivalence[i][j] == result.equivalence[j][i]


def test_compute_pairwise_relations_deduplication():
    """Test deduplication for identical answers."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="paris", sample_id=1),  # Same (case-insensitive)
        Sample(interpretation="I3", answer="  Paris  ", sample_id=2),  # Same (whitespace)
        Sample(interpretation="I4", answer="London", sample_id=3),
    ]

    result = compute_pairwise_relations(samples, scorer, query_text="What is the capital?")

    # Samples 0, 1, 2 should be deduplicated (identical normalized answers)
    assert result.scores[0][1] == 1.0
    assert result.equivalence[0][1] is True

    assert result.scores[0][2] == 1.0
    assert result.equivalence[0][2] is True

    assert result.scores[1][2] == 1.0
    assert result.equivalence[1][2] is True

    # Sample 3 is different
    assert result.scores[0][3] < 1.0
    assert result.equivalence[0][3] is False


def test_compute_pairwise_relations_empty():
    """Test with empty sample list."""
    scorer = MockNLIScorer()

    samples: list[Sample] = []

    result = compute_pairwise_relations(samples, scorer, query_text="Test")

    assert len(result.scores) == 0
    assert len(result.equivalence) == 0
    assert result.n_pairs_total == 0


def test_compute_pairwise_relations_single():
    """Test with single sample."""
    scorer = MockNLIScorer()

    samples = [Sample(interpretation="I1", answer="Paris", sample_id=0)]

    result = compute_pairwise_relations(samples, scorer, query_text="Test")

    assert len(result.scores) == 1
    assert result.scores[0][0] == 1.0
    assert result.equivalence[0][0] is True
    assert result.n_pairs_total == 0


def test_compute_pairwise_relations_input_formats():
    """Test different input formats."""
    scorer = MockNLIScorer()

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
    ]

    # Test qa format
    result_qa = compute_pairwise_relations(samples, scorer, query_text="What?", input_format="qa")
    assert result_qa.n_pairs_computed == 1

    # Test answer_only format
    result_answer = compute_pairwise_relations(
        samples, scorer, query_text="What?", input_format="answer_only"
    )
    assert result_answer.n_pairs_computed == 1

    # Test interpretation_answer format
    result_interp = compute_pairwise_relations(
        samples, scorer, query_text="What?", input_format="interpretation_answer"
    )
    assert result_interp.n_pairs_computed == 1


def test_compute_pairwise_relations_all_unique():
    """Test when all answers are unique (no deduplication)."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="London", sample_id=1),
        Sample(interpretation="I3", answer="Berlin", sample_id=2),
    ]

    result = compute_pairwise_relations(samples, scorer, query_text="What is the capital?")

    # All pairs should be computed
    assert result.n_pairs_computed == 3

    # Different answers should have lower scores
    assert result.scores[0][1] < 1.0
    assert result.scores[0][2] < 1.0
    assert result.scores[1][2] < 1.0


def test_compute_pairwise_relations_all_same():
    """Test when all answers are identical."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="Paris", sample_id=2),
        Sample(interpretation="I4", answer="Paris", sample_id=3),
    ]

    result = compute_pairwise_relations(samples, scorer, query_text="What is the capital?")

    # All pairs should be equivalent (deduplicated)
    for i in range(4):
        for j in range(4):
            assert result.scores[i][j] == 1.0
            assert result.equivalence[i][j] is True
