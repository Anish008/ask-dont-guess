"""Test sampling orchestration."""

from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.sampling.sampler import Sampler


def test_sampler_initialization():
    """Test sampler initialization."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    assert sampler.backend is backend


def test_sampler_sample_with_explicit_seed():
    """Test sampling with explicit seed."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query = Query(text="Test question", query_id="q1")
    result = sampler.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=42,
    )

    assert result.n_samples == 3
    assert len(result.samples) == 3
    assert result.seed == 42


def test_sampler_sample_with_derived_seed():
    """Test sampling with seed derived from query_id."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query = Query(text="Test question", query_id="deterministic_id")
    result = sampler.sample(
        query=query,
        n_samples=2,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=None,  # Should derive from query_id
    )

    assert result.n_samples == 2
    assert len(result.samples) == 2
    # Seed should be derived deterministically
    assert result.seed is not None


def test_sampler_determinism_with_derived_seed():
    """Test that sampling is deterministic when seed is derived."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query = Query(text="Test", query_id="same_id")

    result1 = sampler.sample(
        query=query,
        n_samples=2,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=None,
    )

    result2 = sampler.sample(
        query=query,
        n_samples=2,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=None,
    )

    # Same query_id should produce same seed and same results
    assert result1.seed == result2.seed
    for s1, s2 in zip(result1.samples, result2.samples, strict=False):
        assert s1.answer == s2.answer


def test_sampler_different_queries_different_seeds():
    """Test that different query_ids produce different seeds."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query1 = Query(text="Test", query_id="id1")
    query2 = Query(text="Test", query_id="id2")

    result1 = sampler.sample(
        query=query1,
        n_samples=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=None,
    )

    result2 = sampler.sample(
        query=query2,
        n_samples=1,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=None,
    )

    # Different query_ids should produce different seeds
    assert result1.seed != result2.seed


def test_sampler_close():
    """Test sampler close method."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    sampler.close()  # Should call backend.close()


def test_sampler_respects_parameters():
    """Test that sampler passes parameters to backend correctly."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query = Query(text="Test", query_id="q1")

    # Test with specific parameters
    result = sampler.sample(
        query=query,
        n_samples=5,
        temperature=0.9,
        top_p=0.99,
        max_new_tokens=200,
        seed=123,
    )

    assert result.n_samples == 5
    assert result.seed == 123
    assert len(result.samples) == 5


def test_sampler_handles_valid_samples():
    """Test that sampler correctly reports valid samples."""
    backend = MockLLMBackend()
    sampler = Sampler(backend)

    query = Query(text="Test", query_id="q1")
    result = sampler.sample(
        query=query,
        n_samples=3,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
        seed=1,
    )

    # Mock backend should produce all valid samples
    assert result.n_valid == 3
    assert result.n_refusals == 0
