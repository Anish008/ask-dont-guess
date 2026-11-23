"""Test decision rules."""

from adg.core.types import Action, AggregateFeatures
from adg.decision.rules import DEFAULT_THRESHOLDS, ORDERED_RULES, apply_rules


def test_ordered_rules_structure():
    """Test that ordered rules are properly structured."""
    assert len(ORDERED_RULES) == 4

    # Check rule order
    assert ORDERED_RULES[0].name == "abstain_refusal"
    assert ORDERED_RULES[0].action == Action.ABSTAIN

    assert ORDERED_RULES[1].name == "abstain_invalid"
    assert ORDERED_RULES[1].action == Action.ABSTAIN

    assert ORDERED_RULES[2].name == "ask"
    assert ORDERED_RULES[2].action == Action.ASK

    assert ORDERED_RULES[3].name == "answer"
    assert ORDERED_RULES[3].action == Action.ANSWER


def test_default_thresholds():
    """Test default thresholds are defined."""
    assert "refusal_rate" in DEFAULT_THRESHOLDS
    assert "invalid_rate" in DEFAULT_THRESHOLDS
    assert "ask_score" in DEFAULT_THRESHOLDS
    assert "answer_score" in DEFAULT_THRESHOLDS

    # All should be between 0 and 1
    for threshold in DEFAULT_THRESHOLDS.values():
        assert 0.0 <= threshold <= 1.0


def test_apply_rules_abstain_refusal():
    """Test abstain decision due to high refusal rate."""
    features = AggregateFeatures(
        n_clusters=1,
        n_significant=1,
        singletons=0,
        p1=0.8,
        p2=0.0,
        coverage=0.8,
        separation=1.0,
        refusal_rate=0.5,  # High refusal rate
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.0,
        abstain_score=0.1,
        answer_score=0.8,
    )

    action, rule_name = apply_rules(features)

    assert action == Action.ABSTAIN
    assert rule_name == "abstain_refusal"


def test_apply_rules_abstain_invalid():
    """Test abstain decision due to high invalid rate."""
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
        invalid_rate=0.5,  # High invalid rate
        ask_score=0.0,
        abstain_score=0.4,
        answer_score=0.6,
    )

    action, rule_name = apply_rules(features)

    assert action == Action.ABSTAIN
    assert rule_name == "abstain_invalid"


def test_apply_rules_ask():
    """Test ask decision due to high ask score."""
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
        ask_score=0.6,  # High ask score
        abstain_score=0.0,
        answer_score=0.4,
    )

    action, rule_name = apply_rules(features)

    assert action == Action.ASK
    assert rule_name == "ask"


def test_apply_rules_answer():
    """Test answer decision due to high answer score."""
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
        answer_score=0.9,  # High answer score
    )

    action, rule_name = apply_rules(features)

    assert action == Action.ANSWER
    assert rule_name == "answer"


def test_apply_rules_default_abstain():
    """Test default abstain when no rule matches."""
    features = AggregateFeatures(
        n_clusters=3,
        n_significant=3,
        singletons=3,
        p1=0.33,
        p2=0.33,
        coverage=1.0,
        separation=0.5,
        refusal_rate=0.0,
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.2,  # Below threshold
        abstain_score=0.0,
        answer_score=0.3,  # Below threshold
    )

    action, rule_name = apply_rules(features)

    assert action == Action.ABSTAIN
    assert rule_name == "default_abstain"


def test_apply_rules_custom_thresholds():
    """Test applying rules with custom thresholds."""
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
        ask_score=0.0,
        abstain_score=0.0,
        answer_score=0.4,  # Below default threshold (0.5) but above custom (0.3)
    )

    # With default thresholds, should not match answer rule
    action, rule_name = apply_rules(features)
    assert action == Action.ABSTAIN
    assert rule_name == "default_abstain"

    # With custom threshold, should match answer rule
    custom_thresholds = {"answer_score": 0.3}
    action, rule_name = apply_rules(features, custom_thresholds)
    assert action == Action.ANSWER
    assert rule_name == "answer"


def test_apply_rules_priority():
    """Test that rules are checked in priority order."""
    # Both refusal_rate and ask_score are high
    # refusal rule should match first (higher priority)
    features = AggregateFeatures(
        n_clusters=2,
        n_significant=2,
        singletons=0,
        p1=0.5,
        p2=0.5,
        coverage=1.0,
        separation=0.3,
        refusal_rate=0.5,  # High (matches first rule)
        hedge_rate=0.0,
        invalid_rate=0.0,
        ask_score=0.6,  # Also high (matches third rule)
        abstain_score=0.4,
        answer_score=0.2,
    )

    action, rule_name = apply_rules(features)

    # Should match first rule (abstain_refusal) due to priority
    assert action == Action.ABSTAIN
    assert rule_name == "abstain_refusal"


def test_apply_rules_all_zeros():
    """Test with all feature scores at zero."""
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

    action, rule_name = apply_rules(features)

    # Should default to abstain
    assert action == Action.ABSTAIN
    assert rule_name == "default_abstain"
