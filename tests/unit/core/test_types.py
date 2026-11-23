"""Tests for core/types.py - pydantic data models."""


import pytest
from adg.core.types import (
    ADGResult,
    AggregateFeatures,
    ClarifyingQuestion,
    Cluster,
    ClusteringResult,
    CostRecord,
    DecisionTrace,
    Example,
    GoldInterpretation,
    InterpretationEvidence,
    PairwiseRelations,
    Query,
    ResolvedInterpretation,
    Sample,
    SamplingResult,
    SessionState,
    SimUserInput,
    SimUserReply,
    Turn,
)
from pydantic import ValidationError


def test_query_creation():
    """Test Query model creation."""
    query = Query(text="What is the capital of Georgia?", query_id="test_001")

    assert query.text == "What is the capital of Georgia?"
    assert query.query_id == "test_001"
    assert query.metadata == {}


def test_query_with_metadata():
    """Test Query with metadata."""
    query = Query(
        text="Test question",
        query_id="q1",
        metadata={"source": "test", "timestamp": 123456},
    )

    assert query.metadata["source"] == "test"
    assert query.metadata["timestamp"] == 123456


def test_sample_creation():
    """Test Sample model creation."""
    sample = Sample(
        interpretation="The US state of Georgia",
        answer="Atlanta",
        confidence="high",
        sample_id=0,
    )

    assert sample.interpretation == "The US state of Georgia"
    assert sample.answer == "Atlanta"
    assert sample.confidence == "high"
    assert sample.sample_id == 0
    assert sample.is_valid is True
    assert sample.is_refusal is False
    assert sample.is_hedged is False


def test_sample_with_flags():
    """Test Sample with refusal and hedge flags."""
    sample = Sample(
        interpretation="Unknown",
        answer="I don't know",
        sample_id=1,
        is_refusal=True,
    )

    assert sample.is_refusal is True
    assert sample.is_valid is True

    hedged_sample = Sample(
        interpretation="Multiple meanings",
        answer="Depends on...",
        sample_id=2,
        is_hedged=True,
        hedged_interpretations=["Meaning A", "Meaning B"],
    )

    assert hedged_sample.is_hedged is True
    assert len(hedged_sample.hedged_interpretations) == 2


def test_sampling_result():
    """Test SamplingResult model."""
    samples = [Sample(interpretation="A", answer="Ans A", sample_id=i) for i in range(8)]

    result = SamplingResult(samples=samples, n_samples=8, n_valid=7, n_refusals=1, seed=42)

    assert len(result.samples) == 8
    assert result.n_samples == 8
    assert result.n_valid == 7
    assert result.n_refusals == 1
    assert result.seed == 42


def test_pairwise_relations():
    """Test PairwiseRelations model."""
    relations = PairwiseRelations(
        scores=[[1.0, 0.8], [0.8, 1.0]],
        equivalence=[[True, True], [True, True]],
        n_pairs_computed=1,
        n_pairs_total=1,
        mode="soft",
    )

    assert relations.scores[0][1] == 0.8
    assert relations.equivalence[0][1] is True
    assert relations.mode == "soft"


def test_cluster():
    """Test Cluster model."""
    cluster = Cluster(
        cluster_id=0,
        sample_ids=[0, 1, 2],
        size=3,
        representative_answer="Atlanta",
        representative_interpretation="US state",
    )

    assert cluster.cluster_id == 0
    assert len(cluster.sample_ids) == 3
    assert cluster.size == 3
    assert cluster.representative_answer == "Atlanta"


def test_clustering_result():
    """Test ClusteringResult model."""
    clusters = [
        Cluster(cluster_id=0, sample_ids=[0, 1], size=2, representative_answer="Ans 1"),
        Cluster(cluster_id=1, sample_ids=[2], size=1, representative_answer="Ans 2"),
    ]

    result = ClusteringResult(
        clusters=clusters,
        n_clusters=2,
        algorithm="agglomerative",
        singletons=1,
    )

    assert len(result.clusters) == 2
    assert result.n_clusters == 2
    assert result.algorithm == "agglomerative"
    assert result.singletons == 1


def test_aggregate_features():
    """Test AggregateFeatures model."""
    features = AggregateFeatures(
        n_clusters=2,
        n_significant=1,
        singletons=0,
        p1=0.6,
        p2=0.3,
        coverage=0.9,
        separation=0.2,
        refusal_rate=0.1,
        hedge_rate=0.05,
        invalid_rate=0.0,
        ask_score=0.4,
        abstain_score=0.1,
        answer_score=0.6,
    )

    assert features.p1 == 0.6
    assert features.n_significant == 1
    assert features.ask_score == 0.4


def test_interpretation_evidence():
    """Test InterpretationEvidence model."""
    evidence = InterpretationEvidence(
        enabled=True,
        n_interpretation_clusters=2,
        interpretation_divergence=0.4,
        hedge_evidence=0.1,
    )

    assert evidence.enabled is True
    assert evidence.n_interpretation_clusters == 2
    assert evidence.interpretation_divergence == 0.4


def test_decision_trace():
    """Test DecisionTrace model."""
    trace = DecisionTrace(
        action="ASK",
        rule_fired="R2a",
        reasons=["Multiple significant clusters detected"],
        thresholds_used={"tau_answer": 0.7},
        scores_computed={"p1": 0.5, "ask_score": 0.6},
    )

    assert trace.action == "ASK"
    assert trace.rule_fired == "R2a"
    assert len(trace.reasons) == 1
    assert trace.thresholds_used["tau_answer"] == 0.7


def test_clarifying_question():
    """Test ClarifyingQuestion model."""
    question = ClarifyingQuestion(
        question_text="Do you mean Georgia the US state or Georgia the country?",
        options=["The US state", "The country"],
        mode="template",
        cluster_ids=[0, 1],
    )

    assert "The US state" in question.options
    assert len(question.options) == 2
    assert question.mode == "template"


def test_resolved_interpretation():
    """Test ResolvedInterpretation model."""
    resolved = ResolvedInterpretation(option_index=0, confidence=0.95, method="exact_match")

    assert resolved.option_index == 0
    assert resolved.confidence == 0.95
    assert resolved.method == "exact_match"


def test_cost_record():
    """Test CostRecord model."""
    cost = CostRecord(
        n_llm_calls=2,
        n_samples=8,
        prompt_tokens=150,
        completion_tokens=200,
        n_nli_pairs=28,
        n_nli_forward_passes=4,
        wall_time_ms=1500.0,
        turns=1,
    )

    assert cost.n_samples == 8
    assert cost.wall_time_ms == 1500.0
    assert cost.turns == 1


def test_adg_result():
    """Test ADGResult model (main pipeline output)."""
    query = Query(text="Test query", query_id="q1")
    samples = [Sample(interpretation="I", answer="A", sample_id=0)]
    sampling_result = SamplingResult(samples=samples, n_samples=1, n_valid=1, n_refusals=0, seed=0)
    trace = DecisionTrace(action="ANSWER", rule_fired="R1", reasons=["High confidence"])
    cost = CostRecord()

    result = ADGResult(
        query=query,
        action="ANSWER",
        decision_trace=trace,
        final_answer="Test answer",
        sampling_result=sampling_result,
        cost=cost,
    )

    assert result.action == "ANSWER"
    assert result.final_answer == "Test answer"
    assert result.query.query_id == "q1"


def test_turn():
    """Test Turn model."""
    turn = Turn(
        turn_id=0,
        user_message="What is 2+2?",
        system_response="4",
        action="ANSWER",
    )

    assert turn.turn_id == 0
    assert turn.user_message == "What is 2+2?"
    assert turn.action == "ANSWER"


def test_session_state():
    """Test SessionState model."""
    session = SessionState(session_id="sess_001", state="IDLE")

    assert session.session_id == "sess_001"
    assert session.state == "IDLE"
    assert len(session.turns) == 0
    assert session.ttl_seconds == 3600


def test_gold_interpretation():
    """Test GoldInterpretation model."""
    interp = GoldInterpretation(
        interpretation="The US state",
        answers=["Atlanta", "GA capital"],
    )

    assert interp.interpretation == "The US state"
    assert len(interp.answers) == 2


def test_example():
    """Test Example model (dataset example)."""
    example = Example(
        example_id="ambigqa_001",
        query="What is the capital of Georgia?",
        is_ambiguous=True,
        interpretations=[
            GoldInterpretation(interpretation="US state", answers=["Atlanta"]),
            GoldInterpretation(interpretation="Country", answers=["Tbilisi"]),
        ],
        dataset="ambigqa",
        split="dev",
    )

    assert example.is_ambiguous is True
    assert len(example.interpretations) == 2
    assert example.dataset == "ambigqa"


def test_simuser_input():
    """Test SimUserInput model."""
    intended = GoldInterpretation(interpretation="The country", answers=["Tbilisi"])
    sim_input = SimUserInput(
        original_question="What is the capital of Georgia?",
        intended=intended,
        other_interpretations=[],
        clarifying_question="Do you mean the US state or the country?",
        options=["The US state", "The country"],
    )

    assert sim_input.intended.interpretation == "The country"
    assert len(sim_input.options) == 2


def test_simuser_reply():
    """Test SimUserReply model."""
    reply = SimUserReply(
        text="The country",
        chosen_option_index=1,
        behavior="choose_option",
        leakage_flag=False,
    )

    assert reply.chosen_option_index == 1
    assert reply.behavior == "choose_option"
    assert reply.leakage_flag is False


def test_model_serialization():
    """Test pydantic model serialization."""
    query = Query(text="Test", query_id="q1")

    # Test JSON serialization
    json_dict = query.model_dump()
    assert json_dict["text"] == "Test"
    assert json_dict["query_id"] == "q1"

    # Test round-trip
    query2 = Query(**json_dict)
    assert query2.text == query.text


def test_validation_errors():
    """Test pydantic validation errors."""
    # Missing required field
    with pytest.raises(ValidationError):
        Query(text="Test")  # Missing query_id

    # Invalid literal type
    with pytest.raises(ValidationError):
        Sample(
            interpretation="I",
            answer="A",
            confidence="invalid",  # Must be "high", "medium", or "low"
            sample_id=0,
        )
