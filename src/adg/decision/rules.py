"""Decision rules for ANSWER/ASK/ABSTAIN.

Per spec §4.8: Ordered rule list with thresholds.
"""

from collections.abc import Callable
from dataclasses import dataclass

from adg.core.logging import get_logger
from adg.core.types import Action, AggregateFeatures

log = get_logger(__name__)


@dataclass
class DecisionRule:
    """A single decision rule.

    Args:
        name: Rule identifier
        condition: Function that takes AggregateFeatures and threshold, returns bool
        action: Action to take if condition is True
        feature_key: Key name of the feature used (for calibration)
    """

    name: str
    condition: Callable[[AggregateFeatures, float], bool]
    action: Action
    feature_key: str


# Default thresholds from spec §4.8
DEFAULT_THRESHOLDS = {
    "refusal_rate": 0.3,
    "invalid_rate": 0.3,
    "ask_score": 0.4,
    "answer_score": 0.5,
}


def _check_abstain_refusal(features: AggregateFeatures, threshold: float) -> bool:
    """Check if refusal rate exceeds threshold."""
    return features.refusal_rate > threshold


def _check_abstain_invalid(features: AggregateFeatures, threshold: float) -> bool:
    """Check if invalid rate exceeds threshold."""
    return features.invalid_rate > threshold


def _check_ask(features: AggregateFeatures, threshold: float) -> bool:
    """Check if ask score exceeds threshold."""
    return features.ask_score > threshold


def _check_answer(features: AggregateFeatures, threshold: float) -> bool:
    """Check if answer score exceeds threshold."""
    return features.answer_score > threshold


# Ordered rule list per spec §4.8
# Rules are checked in order; first match wins
ORDERED_RULES = [
    DecisionRule(
        name="abstain_refusal",
        condition=_check_abstain_refusal,
        action=Action.ABSTAIN,
        feature_key="refusal_rate",
    ),
    DecisionRule(
        name="abstain_invalid",
        condition=_check_abstain_invalid,
        action=Action.ABSTAIN,
        feature_key="invalid_rate",
    ),
    DecisionRule(
        name="ask",
        condition=_check_ask,
        action=Action.ASK,
        feature_key="ask_score",
    ),
    DecisionRule(
        name="answer",
        condition=_check_answer,
        action=Action.ANSWER,
        feature_key="answer_score",
    ),
]


def apply_rules(
    features: AggregateFeatures,
    thresholds: dict[str, float] | None = None,
) -> tuple[Action, str]:
    """Apply ordered rules to determine action.

    Per spec §4.8: Rules checked in order, first match wins.
    If no rule matches, default to ABSTAIN.

    Args:
        features: Aggregate features from clustering
        thresholds: Optional threshold overrides (uses DEFAULT_THRESHOLDS if None)

    Returns:
        (action, rule_name) tuple
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    for rule in ORDERED_RULES:
        threshold = thresholds.get(rule.feature_key, DEFAULT_THRESHOLDS[rule.feature_key])
        if rule.condition(features, threshold):
            log.debug(
                "rule_matched",
                rule=rule.name,
                action=rule.action,
                threshold=threshold,
            )
            return rule.action, rule.name

    # Default: ABSTAIN if no rule matches
    log.debug("no_rule_matched", default_action="abstain")
    return Action.ABSTAIN, "default_abstain"


__all__ = ["DecisionRule", "ORDERED_RULES", "DEFAULT_THRESHOLDS", "apply_rules"]
