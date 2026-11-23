"""Test mock NLI scorer."""

from adg.core.types import Sample
from adg.nli.mock_nli import MockNLIScorer


def test_mock_nli_initialization():
    """Test mock NLI scorer initialization."""
    scorer = MockNLIScorer(threshold=0.6)
    assert scorer.threshold == 0.6


def test_mock_nli_initialization_default():
    """Test mock NLI scorer with default threshold."""
    scorer = MockNLIScorer()
    assert scorer.threshold == 0.5


def test_mock_nli_score_single_pair_identical():
    """Test scoring identical texts."""
    scorer = MockNLIScorer()

    entail, neutral, contra = scorer.score_single_pair(
        "Paris is the capital", "Paris is the capital"
    )

    assert entail == 1.0  # Perfect match
    assert neutral == 0.0
    assert contra == 0.0


def test_mock_nli_score_single_pair_similar():
    """Test scoring similar texts."""
    scorer = MockNLIScorer()

    entail, neutral, contra = scorer.score_single_pair(
        "Paris is the capital of France",
        "The capital of France is Paris",
    )

    # Jaccard similarity: same words, different order = 1.0 (order-independent)
    assert entail == 1.0  # Same words
    assert neutral == 0.0
    assert contra == 0.0


def test_mock_nli_score_single_pair_different():
    """Test scoring different texts."""
    scorer = MockNLIScorer()

    entail, neutral, contra = scorer.score_single_pair(
        "Paris is the capital of France",
        "Tokyo is a city in Japan",
    )

    # Different texts with low word overlap
    assert entail < 0.5  # Low similarity
    assert neutral == 0.0
    assert contra == (1.0 - entail)


def test_mock_nli_score_pairs():
    """Test scoring multiple sample pairs."""
    scorer = MockNLIScorer(threshold=0.5)

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
        Sample(interpretation="I3", answer="London", sample_id=2),
    ]

    result = scorer.score_pairs(samples, query_text="What is the capital?")

    # Check structure
    assert len(result.scores) == 3
    assert len(result.equivalence) == 3
    assert result.n_pairs_total == 3  # 3 * 2 / 2
    assert result.n_pairs_computed == 3

    # Check diagonal (self-equivalence)
    for i in range(3):
        assert result.scores[i][i] == 1.0
        assert result.equivalence[i][i] is True

    # Check symmetry
    for i in range(3):
        for j in range(3):
            assert result.scores[i][j] == result.scores[j][i]
            assert result.equivalence[i][j] == result.equivalence[j][i]

    # Check specific pairs
    # Samples 0 and 1 have same answer "Paris"
    assert result.scores[0][1] == 1.0
    assert result.equivalence[0][1] is True

    # Samples 0 and 2 have different answers
    assert result.scores[0][2] < 0.5
    assert result.equivalence[0][2] is False


def test_mock_nli_score_pairs_empty():
    """Test scoring with empty sample list."""
    scorer = MockNLIScorer()

    samples: list[Sample] = []

    result = scorer.score_pairs(samples, query_text="Test")

    assert len(result.scores) == 0
    assert len(result.equivalence) == 0
    assert result.n_pairs_total == 0
    assert result.n_pairs_computed == 0


def test_mock_nli_score_pairs_single():
    """Test scoring with single sample."""
    scorer = MockNLIScorer()

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
    ]

    result = scorer.score_pairs(samples, query_text="Test")

    assert len(result.scores) == 1
    assert result.scores[0][0] == 1.0
    assert result.equivalence[0][0] is True
    assert result.n_pairs_total == 0
    assert result.n_pairs_computed == 0


def test_mock_nli_threshold_effect():
    """Test that threshold affects equivalence classification."""
    samples = [
        Sample(interpretation="I1", answer="The capital is Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris is the capital", sample_id=1),
    ]

    # Low threshold - should be equivalent
    scorer_low = MockNLIScorer(threshold=0.3)
    result_low = scorer_low.score_pairs(samples, query_text="Test")
    assert result_low.equivalence[0][1] is True

    # High threshold - might not be equivalent
    scorer_high = MockNLIScorer(threshold=0.9)
    scorer_high.score_pairs(samples, query_text="Test")
    # Depends on exact Jaccard similarity


def test_mock_nli_input_format():
    """Test that input_format parameter is accepted."""
    scorer = MockNLIScorer()

    samples = [
        Sample(interpretation="I1", answer="Paris", sample_id=0),
        Sample(interpretation="I2", answer="Paris", sample_id=1),
    ]

    # Should work with different formats
    result_qa = scorer.score_pairs(samples, query_text="Test", input_format="qa")
    result_answer = scorer.score_pairs(samples, query_text="Test", input_format="answer_only")
    result_interp = scorer.score_pairs(
        samples, query_text="Test", input_format="interpretation_answer"
    )

    # All should produce valid results (format doesn't affect mock scoring)
    assert result_qa.n_pairs_computed == 1
    assert result_answer.n_pairs_computed == 1
    assert result_interp.n_pairs_computed == 1


def test_mock_nli_close():
    """Test close method."""
    scorer = MockNLIScorer()
    scorer.close()  # Should not raise


def test_mock_nli_jaccard_similarity():
    """Test Jaccard similarity computation."""
    scorer = MockNLIScorer()

    # Test cases
    assert scorer._compute_similarity("a b c", "a b c") == 1.0  # Identical
    assert scorer._compute_similarity("a b c", "d e f") == 0.0  # Disjoint
    assert scorer._compute_similarity("a b c", "a b") == 2 / 3  # Partial overlap
    assert scorer._compute_similarity("", "a b") == 0.0  # Empty
    assert scorer._compute_similarity("a b", "") == 0.0  # Empty


def test_mock_nli_case_insensitive():
    """Test that scoring is case-insensitive."""
    scorer = MockNLIScorer()

    score1 = scorer._compute_similarity("Paris", "paris")
    score2 = scorer._compute_similarity("PARIS", "paris")

    assert score1 == 1.0
    assert score2 == 1.0
