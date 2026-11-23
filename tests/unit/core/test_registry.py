"""Tests for core/registry.py - backend and method registration."""

import pytest
from adg.core.errors import BackendNotFoundError, MethodNotFoundError
from adg.core.registry import (
    BackendRegistry,
    MethodRegistry,
    NLIScorerRegistry,
    SimUserRegistry,
    backend_registry,
    method_registry,
    nli_registry,
    register_backend,
    register_method,
    register_nli,
    register_simuser,
    simuser_registry,
)
from adg.core.types import (
    ADGResult,
    DecisionTrace,
    PairwiseRelations,
    Query,
    Sample,
    SamplingResult,
    SimUserReply,
)

# ==============================================================================
# Test Backend Classes
# ==============================================================================


class TestLLMBackend:
    """Test LLM backend."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sample(self, query, n_samples, temperature, top_p, max_new_tokens, seed):
        samples = [Sample(interpretation="I", answer="A", sample_id=i) for i in range(n_samples)]
        return SamplingResult(
            samples=samples, n_samples=n_samples, n_valid=n_samples, n_refusals=0, seed=seed
        )

    def generate(self, prompt, temperature=0.0, max_new_tokens=96, seed=None):
        return "Generated"

    def close(self):
        pass


class TestNLIScorer:
    """Test NLI scorer."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score_pairs(self, samples, query_text, input_format="qa"):
        n = len(samples)
        scores = [[1.0 if i == j else 0.5 for j in range(n)] for i in range(n)]
        equivalence = [[i == j for j in range(n)] for i in range(n)]
        return PairwiseRelations(
            scores=scores, equivalence=equivalence, n_pairs_computed=0, n_pairs_total=0
        )

    def score_single_pair(self, premise, hypothesis):
        return (0.7, 0.2, 0.1)

    def close(self):
        pass


class TestMethod:
    """Test method."""

    name: str = "test_method"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, query, llm_backend, nli_scorer, config):
        from adg.core.types import CostRecord

        sampling_result = llm_backend.sample(query, 1, 0.7, 0.9, 96, 0)
        trace = DecisionTrace(action="ANSWER", rule_fired="TEST", reasons=["Test"])
        return ADGResult(
            query=query,
            action="ANSWER",
            decision_trace=trace,
            final_answer="Test",
            sampling_result=sampling_result,
            cost=CostRecord(),
        )


class TestSimUser:
    """Test simulated user."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reply(self, input_data):
        return SimUserReply(text="Test reply", chosen_option_index=0, behavior="choose_option")


# ==============================================================================
# Backend Registry Tests
# ==============================================================================


def test_backend_registry_creation():
    """Test creating a new backend registry."""
    registry = BackendRegistry()
    assert len(registry.list_backends()) == 0


def test_backend_registry_register():
    """Test registering a backend."""
    registry = BackendRegistry()
    registry.register("test_backend", TestLLMBackend)

    assert "test_backend" in registry.list_backends()


def test_backend_registry_get():
    """Test getting a registered backend."""
    registry = BackendRegistry()
    registry.register("test_backend", TestLLMBackend)

    backend = registry.get("test_backend", test_param="value")

    assert isinstance(backend, TestLLMBackend)
    assert backend.kwargs["test_param"] == "value"


def test_backend_registry_get_not_found():
    """Test getting an unregistered backend raises error."""
    registry = BackendRegistry()
    registry.register("backend1", TestLLMBackend)

    with pytest.raises(BackendNotFoundError) as exc_info:
        registry.get("nonexistent")

    assert exc_info.value.backend_name == "nonexistent"
    assert "backend1" in str(exc_info.value)


# ==============================================================================
# NLI Scorer Registry Tests
# ==============================================================================


def test_nli_registry_register_and_get():
    """Test registering and getting an NLI scorer."""
    registry = NLIScorerRegistry()
    registry.register("test_nli", TestNLIScorer)

    scorer = registry.get("test_nli", test_param="value")

    assert isinstance(scorer, TestNLIScorer)
    assert scorer.kwargs["test_param"] == "value"
    assert "test_nli" in registry.list_scorers()


# ==============================================================================
# Method Registry Tests
# ==============================================================================


def test_method_registry_register_and_get():
    """Test registering and getting a method."""
    registry = MethodRegistry()
    registry.register("test_method", TestMethod)

    method = registry.get("test_method", test_param="value")

    assert isinstance(method, TestMethod)
    assert method.kwargs["test_param"] == "value"
    assert "test_method" in registry.list_methods()


def test_method_registry_get_not_found():
    """Test getting an unregistered method raises error."""
    registry = MethodRegistry()
    registry.register("method1", TestMethod)

    with pytest.raises(MethodNotFoundError) as exc_info:
        registry.get("nonexistent")

    assert exc_info.value.method_name == "nonexistent"
    assert "method1" in str(exc_info.value)


# ==============================================================================
# Simulated User Registry Tests
# ==============================================================================


def test_simuser_registry_register_and_get():
    """Test registering and getting a simulated user."""
    registry = SimUserRegistry()
    registry.register("test_user", TestSimUser)

    user = registry.get("test_user", test_param="value")

    assert isinstance(user, TestSimUser)
    assert user.kwargs["test_param"] == "value"
    assert "test_user" in registry.list_users()


# ==============================================================================
# Global Registry Tests
# ==============================================================================


def test_global_backend_registry():
    """Test the global backend_registry instance."""
    # Clean up first (in case other tests polluted it)
    for name in backend_registry.list_backends():
        if name.startswith("test_"):
            backend_registry._backends.pop(name, None)

    backend_registry.register("test_global_backend", TestLLMBackend)

    assert "test_global_backend" in backend_registry.list_backends()
    backend = backend_registry.get("test_global_backend")
    assert isinstance(backend, TestLLMBackend)


# ==============================================================================
# Decorator Tests
# ==============================================================================


def test_register_backend_decorator():
    """Test @register_backend decorator."""

    @register_backend("decorated_backend")
    class DecoratedBackend(TestLLMBackend):
        pass

    assert "decorated_backend" in backend_registry.list_backends()
    backend = backend_registry.get("decorated_backend")
    assert isinstance(backend, DecoratedBackend)


def test_register_nli_decorator():
    """Test @register_nli decorator."""

    @register_nli("decorated_nli")
    class DecoratedNLI(TestNLIScorer):
        pass

    assert "decorated_nli" in nli_registry.list_scorers()
    scorer = nli_registry.get("decorated_nli")
    assert isinstance(scorer, DecoratedNLI)


def test_register_method_decorator():
    """Test @register_method decorator."""

    @register_method("decorated_method")
    class DecoratedMethod(TestMethod):
        pass

    assert "decorated_method" in method_registry.list_methods()
    method = method_registry.get("decorated_method")
    assert isinstance(method, DecoratedMethod)


def test_register_simuser_decorator():
    """Test @register_simuser decorator."""

    @register_simuser("decorated_user")
    class DecoratedUser(TestSimUser):
        pass

    assert "decorated_user" in simuser_registry.list_users()
    user = simuser_registry.get("decorated_user")
    assert isinstance(user, DecoratedUser)


# ==============================================================================
# Integration Tests
# ==============================================================================


def test_registry_integration():
    """Test using registered backends in a pipeline."""
    # Register components
    backend_registry.register("int_backend", TestLLMBackend)
    nli_registry.register("int_nli", TestNLIScorer)
    method_registry.register("int_method", TestMethod)

    # Get components
    backend = backend_registry.get("int_backend")
    scorer = nli_registry.get("int_nli")
    method = method_registry.get("int_method")

    # Use them
    query = Query(text="Test", query_id="q1")
    from adg.config.schema import AppConfig

    config = AppConfig()

    result = method.run(query, backend, scorer, config)

    assert isinstance(result, ADGResult)
    assert result.action == "ANSWER"


def test_registry_multiple_backends():
    """Test registering multiple backends and listing them."""
    registry = BackendRegistry()

    registry.register("backend_a", TestLLMBackend)
    registry.register("backend_b", TestLLMBackend)
    registry.register("backend_c", TestLLMBackend)

    backends = registry.list_backends()

    assert len(backends) >= 3
    assert "backend_a" in backends
    assert "backend_b" in backends
    assert "backend_c" in backends
