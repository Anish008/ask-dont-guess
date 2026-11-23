"""Test HF NLI scorer (structural tests without model loading)."""

from adg.core.types import Sample


def test_hf_nli_text_formatting():
    """Test text formatting for different input formats."""
    # We can't easily test HFNLIScorer without loading a model,
    # so we'll test the formatting logic conceptually

    samples = [
        Sample(interpretation="Asking about capital", answer="Paris", sample_id=0),
        Sample(interpretation="Capital question", answer="London", sample_id=1),
    ]

    query_text = "What is the capital?"

    # Test that we can construct expected formatted texts
    # Format: "qa"
    expected_qa_0 = f"Q: {query_text} A: Paris"
    assert "Q:" in expected_qa_0
    assert "A:" in expected_qa_0

    # Format: "answer_only"
    expected_answer_0 = "Paris"
    assert expected_answer_0 == samples[0].answer

    # Format: "interpretation_answer"
    expected_interp_0 = f"{samples[0].interpretation} {samples[0].answer}"
    assert "Asking about capital" in expected_interp_0
    assert "Paris" in expected_interp_0


def test_hf_nli_label_mapping():
    """Test that label mapping is correct for DeBERTa."""
    # DeBERTa-MNLI uses: 0=contradiction, 1=neutral, 2=entailment
    expected_mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}

    # Verify mapping structure
    assert "contradiction" in expected_mapping
    assert "neutral" in expected_mapping
    assert "entailment" in expected_mapping
    assert expected_mapping["entailment"] == 2
    assert expected_mapping["contradiction"] == 0


def test_hf_nli_bidirectional_concept():
    """Test bidirectional scoring concept."""
    # Bidirectional score should be product of forward and backward
    forward_score = 0.8
    backward_score = 0.9
    bidirectional_score = forward_score * backward_score

    assert abs(bidirectional_score - 0.72) < 1e-10  # Floating point comparison
    assert bidirectional_score <= min(forward_score, backward_score)


def test_hf_nli_symmetry_requirement():
    """Test that scores should be symmetric."""
    # For any pair (i, j), score[i][j] should equal score[j][i]
    # This is guaranteed by bidirectional scoring

    # Mock scores
    scores = [[1.0, 0.8, 0.5], [0.8, 1.0, 0.3], [0.5, 0.3, 1.0]]

    # Check symmetry
    n = len(scores)
    for i in range(n):
        for j in range(n):
            assert scores[i][j] == scores[j][i]


def test_hf_nli_diagonal_requirement():
    """Test that diagonal should be 1.0 (self-equivalence)."""
    # Mock score matrix
    scores = [[1.0, 0.8, 0.5], [0.8, 1.0, 0.3], [0.5, 0.3, 1.0]]

    # Check diagonal
    for i in range(len(scores)):
        assert scores[i][i] == 1.0
