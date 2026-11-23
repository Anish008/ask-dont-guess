"""Tests for core/protocols.py - protocol conformance."""

from adg.config.schema import AppConfig
from adg.core.protocols import LLMBackend, NLIScorer, SimulatedUser
from adg.core.types import (
    ADGResult,
    DecisionTrace,
    GoldInterpretation,
    PairwiseRelations,
    Query,
    Sample,
    SamplingResult,
    SimUserInput,
    SimUserReply,
)

# ==============================================================================
# Mock Implementations for Testing
# ==============================================================================


class MockLLMBackend:
    """Mock LLM backend for testing protocol conformance."""

    def sample(
        self,
        query: Query,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> SamplingResult:
        """Mock sampling that returns deterministic samples."""
        samples = [
            Sample(
                interpretation=f"Interpretation {i}",
                answer=f"Answer {i}",
                sample_id=i,
            )
            for i in range(n_samples)
        ]
        return SamplingResult(
            samples=samples,
            n_samples=n_samples,
            n_valid=n_samples,
            n_refusals=0,
            seed=seed,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 96,
        seed: int | None = None,
    ) -> str:
        """Mock generation."""
        return f"Generated response to: {prompt[:20]}..."

    def close(self) -> None:
        """Mock close."""


class MockNLIScorer:
    """Mock NLI scorer for testing protocol conformance."""

    def score_pairs(
        self,
        samples: list[Sample],
        query_text: str,
        input_format: str = "qa",
    ) -> PairwiseRelations:
        """Mock NLI scoring that returns identity matrix."""
        n = len(samples)
        scores = [[1.0 if i == j else 0.5 for j in range(n)] for i in range(n)]
        equivalence = [[i == j for j in range(n)] for i in range(n)]

        return PairwiseRelations(
            scores=scores,
            equivalence=equivalence,
            n_pairs_computed=n * (n - 1) // 2,
            n_pairs_total=n * (n - 1) // 2,
        )

    def score_single_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, float, float]:
        """Mock single pair scoring."""
        # Return (entailment, neutral, contradiction)
        return (0.7, 0.2, 0.1)

    def close(self) -> None:
        """Mock close."""


class MockMethod:
    """Mock method for testing protocol conformance."""

    name: str = "mock_method"

    def run(
        self,
        query: Query,
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        config: AppConfig,
    ) -> ADGResult:
        """Mock run that always answers."""
        from adg.core.types import CostRecord

        sampling_result = llm_backend.sample(query, 2, 0.7, 0.9, 96, 0)
        trace = DecisionTrace(action="ANSWER", rule_fired="MOCK", reasons=["Testing"])

        return ADGResult(
            query=query,
            action="ANSWER",
            decision_trace=trace,
            final_answer="Mock answer",
            sampling_result=sampling_result,
            cost=CostRecord(),
        )


class MockSimulatedUser:
    """Mock simulated user for testing protocol conformance."""

    def reply(self, input_data: SimUserInput) -> SimUserReply:
        """Mock reply that always chooses first option."""
        return SimUserReply(
            text="The first option",
            chosen_option_index=0,
            behavior="choose_option",
            leakage_flag=False,
        )


# ==============================================================================
# Protocol Conformance Tests
# ==============================================================================


def test_llm_backend_protocol_conformance():
    """Test that MockLLMBackend conforms to LLMBackend protocol."""
    backend = MockLLMBackend()

    # Check protocol conformance
    assert isinstance(backend, LLMBackend)

    # Test sample method
    query = Query(text="Test query", query_id="q1")
    result = backend.sample(
        query, n_samples=3, temperature=0.7, top_p=0.9, max_new_tokens=96, seed=42
    )

    assert isinstance(result, SamplingResult)
    assert len(result.samples) == 3
    assert result.n_samples == 3

    # Test generate method
    response = backend.generate("Test prompt", temperature=0.0)
    assert isinstance(response, str)
    assert len(response) > 0

    # Test close method
    backend.close()  # Should not raise


def test_nli_scorer_protocol_conformance():
    """Test that MockNLIScorer conforms to NLIScorer protocol."""
    scorer = MockNLIScorer()

    # Check protocol conformance
    assert isinstance(scorer, NLIScorer)

    # Test score_pairs method
    samples = [
        Sample(interpretation="I1", answer="A1", sample_id=0),
        Sample(interpretation="I2", answer="A2", sample_id=1),
    ]
    relations = scorer.score_pairs(samples, query_text="Test query")

    assert isinstance(relations, PairwiseRelations)
    assert len(relations.scores) == 2
    assert len(relations.equivalence) == 2

    # Test score_single_pair method
    entail, neutral, contra = scorer.score_single_pair("Premise", "Hypothesis")
    assert isinstance(entail, float)
    assert isinstance(neutral, float)
    assert isinstance(contra, float)
    assert abs(entail + neutral + contra - 1.0) < 0.01  # Should sum to ~1

    # Test close method
    scorer.close()


def test_method_protocol_conformance():
    """Test that MockMethod conforms to Method protocol."""
    method = MockMethod()

    # Check protocol conformance (methods and attributes)
    assert hasattr(method, "name")
    assert hasattr(method, "run")
    assert callable(method.run)
    assert method.name == "mock_method"

    # Test run method
    query = Query(text="Test", query_id="q1")
    backend = MockLLMBackend()
    scorer = MockNLIScorer()
    config = AppConfig()

    result = method.run(query, backend, scorer, config)

    assert isinstance(result, ADGResult)
    assert result.action == "ANSWER"
    assert result.final_answer is not None


def test_simulated_user_protocol_conformance():
    """Test that MockSimulatedUser conforms to SimulatedUser protocol."""
    user = MockSimulatedUser()

    # Check protocol conformance
    assert isinstance(user, SimulatedUser)

    # Test reply method
    intended = GoldInterpretation(interpretation="The country", answers=["Tbilisi"])
    input_data = SimUserInput(
        original_question="What is the capital of Georgia?",
        intended=intended,
        other_interpretations=[],
        clarifying_question="Do you mean the state or country?",
        options=["The state", "The country"],
    )

    reply = user.reply(input_data)

    assert isinstance(reply, SimUserReply)
    assert isinstance(reply.text, str)
    assert reply.behavior in ["choose_option", "describe_intent", "none_of_these"]


def test_protocol_method_signatures():
    """Test that protocol methods have correct signatures."""
    # LLMBackend
    backend = MockLLMBackend()
    assert callable(backend.sample)
    assert callable(backend.generate)
    assert callable(backend.close)

    # NLIScorer
    scorer = MockNLIScorer()
    assert callable(scorer.score_pairs)
    assert callable(scorer.score_single_pair)
    assert callable(scorer.close)

    # Method
    method = MockMethod()
    assert callable(method.run)
    assert hasattr(method, "name")

    # SimulatedUser
    user = MockSimulatedUser()
    assert callable(user.reply)


def test_protocol_integration():
    """Test that protocols work together in an integration scenario."""
    query = Query(text="Test integration", query_id="int_001")
    backend = MockLLMBackend()
    scorer = MockNLIScorer()
    method = MockMethod()
    config = AppConfig()

    # Run full pipeline with mock implementations
    result = method.run(query, backend, scorer, config)

    # Verify result structure
    assert result.query == query
    assert result.action in ["ANSWER", "ASK", "ABSTAIN"]
    assert result.sampling_result is not None
    assert result.decision_trace is not None

    # Clean up
    backend.close()
    scorer.close()
