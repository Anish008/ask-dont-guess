"""Test decision engine."""

from adg.core.types import Action, AggregateFeatures
from adg.decision.engine import DecisionEngine


def test_decision_engine_initialization():
    """Test decision engine initialization."""
    engine = DecisionEngine()
    assert engine.thresholds is not None
    assert "refusal_rate" in engine.thresholds


def test_decision_engine_initialization_custom_thresholds():
    """Test decision engine with custom thresholds."""
    custom_thresholds = {
        "refusal_rate": 0.5,
        "invalid_rate": 0.5,
        "ask_score": 0.6,
        "answer_score": 0.7,
    }
    engine = DecisionEngine(thresholds=custom_thresholds)
    assert engine.thresholds == custom_thresholds


def test_decision_engine_decide_answer():
    """Test decision engine makes answer decision."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=1,
        n_significant=1,
        singletons=0,
        p1=0.9,
        p2=0.1,
        coverage=1.0,
        separation=1.0,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.1,
        abstain_score=0.0,
        answer_score=0.9,
    )

    decision = engine.decide(features)

    assert decision.action == Action.ANSWER
    assert decision.rule_name == "answer"
    assert decision.confidence > 0.0
    assert len(decision.confidence_scores) == 3


def test_decision_engine_decide_ask():
    """Test decision engine makes ask decision."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=2,
        n_significant=2,
        singletons=0,
        p1=0.5,
        p2=0.5,
        coverage=1.0,
        separation=0.3,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.6,
        abstain_score=0.0,
        answer_score=0.4,
    )

    decision = engine.decide(features)

    assert decision.action == Action.ASK
    assert decision.rule_name == "ask"
    assert decision.confidence > 0.0


def test_decision_engine_decide_abstain():
    """Test decision engine makes abstain decision."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=1,
        n_significant=1,
        singletons=0,
        p1=0.8,
        p2=0.0,
        coverage=0.8,
        separation=1.0,
        refusal_rate=0.5,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.0,
        abstain_score=0.1,
        answer_score=0.8,
    )

    decision = engine.decide(features)

    assert decision.action == Action.ABSTAIN
    assert decision.rule_name == "abstain_refusal"


def test_decision_engine_confidence_scores_normalized():
    """Test that confidence scores are normalized to sum to 1."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=2,
        n_significant=2,
        singletons=0,
        p1=0.6,
        p2=0.4,
        coverage=1.0,
        separation=0.5,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.3,
        abstain_score=0.1,
        answer_score=0.6,
    )

    decision = engine.decide(features)

    # Confidence scores should sum to 1
    total = sum(decision.confidence_scores.values())
    assert abs(total - 1.0) < 1e-6


def test_decision_engine_confidence_scores_all_actions():
    """Test that confidence scores include all actions."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=1,
        n_significant=1,
        singletons=0,
        p1=0.8,
        p2=0.0,
        coverage=0.8,
        separation=1.0,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.1,
        abstain_score=0.1,
        answer_score=0.8,
    )

    decision = engine.decide(features)

    assert Action.ANSWER in decision.confidence_scores
    assert Action.ASK in decision.confidence_scores
    assert Action.ABSTAIN in decision.confidence_scores


def test_decision_engine_confidence_scores_zero_sum():
    """Test confidence scores when all features are zero."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=0,
        n_significant=0,
        singletons=0,
        p1=0.0,
        p2=0.0,
        coverage=0.0,
        separation=1.0,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.0,
        abstain_score=0.0,
        answer_score=0.0,
    )

    decision = engine.decide(features)

    # Should have uniform distribution when all zeros
    assert decision.confidence_scores[Action.ANSWER] == 1.0 / 3.0
    assert decision.confidence_scores[Action.ASK] == 1.0 / 3.0
    assert decision.confidence_scores[Action.ABSTAIN] == 1.0 / 3.0


def test_decision_engine_update_thresholds():
    """Test updating thresholds."""
    engine = DecisionEngine()

    original_threshold = engine.thresholds["answer_score"]

    engine.update_thresholds({"answer_score": 0.8})

    assert engine.thresholds["answer_score"] == 0.8
    assert engine.thresholds["answer_score"] != original_threshold


def test_decision_engine_update_thresholds_partial():
    """Test partial threshold updates."""
    engine = DecisionEngine()

    original_refusal = engine.thresholds["refusal_rate"]

    engine.update_thresholds({"answer_score": 0.8})

    # answer_score should be updated
    assert engine.thresholds["answer_score"] == 0.8

    # refusal_rate should remain unchanged
    assert engine.thresholds["refusal_rate"] == original_refusal


def test_decision_engine_confidence_highest_for_chosen_action():
    """Test that chosen action has highest confidence (usually)."""
    engine = DecisionEngine()

    features = AggregateFeatures(
        n_clusters=1,
        n_significant=1,
        singletons=0,
        p1=0.9,
        p2=0.1,
        coverage=1.0,
        separation=1.0,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.05,
        abstain_score=0.05,
        answer_score=0.9,
    )

    decision = engine.decide(features)

    # Chosen action should have highest confidence
    chosen_confidence = decision.confidence_scores[decision.action]
    for action, confidence in decision.confidence_scores.items():
        if action != decision.action:
            assert chosen_confidence >= confidence
