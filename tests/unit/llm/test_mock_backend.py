"""Test mock LLM backend."""

from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend


def test_mock_backend_initialization():
    """Test mock backend initialization."""
    backend = MockLLMBackend()
    assert backend is not None


def test_mock_backend_sample():
    """Test mock backend sampling."""
    backend = MockLLMBackend()
    query = Query(text="What is the capital of France?", query_id="test_q1")

    result = backend.sample(
        query=query,
        n_samples=5,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=42,
    )

    assert result.n_samples == 5
    assert len(result.samples) == 5
    assert result.n_valid == 5
    assert result.n_refusals == 0
    assert result.seed == 42

    # Check all samples are valid
    for sample in result.samples:
        assert sample.is_valid
        assert sample.interpretation != ""
        assert sample.answer != ""
        assert sample.confidence in ["high", "medium", "low"]


def test_mock_backend_determinism():
    """Test that mock backend is deterministic with same seed."""
    backend = MockLLMBackend()
    query = Query(text="What is 2 + 2?", query_id="test_q2")

    result1 = backend.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=123,
    )

    result2 = backend.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=123,
    )

    # Same seed should produce identical results
    for s1, s2 in zip(result1.samples, result2.samples, strict=False):
        assert s1.interpretation == s2.interpretation
        assert s1.answer == s2.answer
        assert s1.confidence == s2.confidence


def test_mock_backend_different_seeds():
    """Test that different seeds produce different results."""
    backend = MockLLMBackend()
    query = Query(text="Test question", query_id="test_q3")

    result1 = backend.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=1,
    )

    result2 = backend.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=2,
    )

    # Different seeds should produce different results
    different = False
    for s1, s2 in zip(result1.samples, result2.samples, strict=False):
        if s1.answer != s2.answer:
            different = True
            break

    assert different, "Different seeds should produce different outputs"


def test_mock_backend_generate():
    """Test mock backend generate method."""
    backend = MockLLMBackend()
    prompt = "Test prompt"

    response = backend.generate(prompt, temperature=0.0, max_new_tokens=50, seed=42)

    assert isinstance(response, str)
    assert len(response) > 0


def test_mock_backend_generate_determinism():
    """Test generate method determinism."""
    backend = MockLLMBackend()
    prompt = "Generate this"

    response1 = backend.generate(prompt, seed=100)
    response2 = backend.generate(prompt, seed=100)

    assert response1 == response2


def test_mock_backend_close():
    """Test mock backend close method."""
    backend = MockLLMBackend()
    backend.close()  # Should not raise


def test_mock_backend_sample_ids():
    """Test that sample IDs are correctly assigned."""
    backend = MockLLMBackend()
    query = Query(text="Test", query_id="q1")

    result = backend.sample(
        query=query,
        n_samples=5,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=1,
    )

    for i, sample in enumerate(result.samples):
        assert sample.sample_id == i


def test_mock_backend_raw_text():
    """Test that raw_text is populated."""
    backend = MockLLMBackend()
    query = Query(text="Test", query_id="q1")

    result = backend.sample(
        query=query,
        n_samples=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=1,
    )

    sample = result.samples[0]
    assert sample.raw_text is not None
    assert "Interpretation:" in sample.raw_text
    assert "Answer:" in sample.raw_text
    assert "Confidence:" in sample.raw_text


def test_mock_backend_different_queries():
    """Test that different queries produce different outputs."""
    backend = MockLLMBackend()

    query1 = Query(text="Question 1", query_id="q1")
    query2 = Query(text="Question 2", query_id="q2")

    result1 = backend.sample(
        query1, n_samples=1, temperature=0.8, top_p=0.95, max_new_tokens=100, seed=1
    )
    result2 = backend.sample(
        query2, n_samples=1, temperature=0.8, top_p=0.95, max_new_tokens=100, seed=1
    )

    # Different questions should produce different answers
    assert result1.samples[0].answer != result2.samples[0].answer
