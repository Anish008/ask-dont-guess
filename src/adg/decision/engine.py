"""Decision engine for ADG system.

Per spec §4.8: Rule-based decision with confidence scores.
"""

from dataclasses import dataclass

from adg.core.logging import get_logger
from adg.core.types import Action, AggregateFeatures, Decision
from adg.decision.rules import DEFAULT_THRESHOLDS, apply_rules

log = get_logger(__name__)


@dataclass
class DecisionContext:
    """Context for decision-making.

    Args:
        features: Aggregate features from clustering
        thresholds: Threshold values for rules
        rule_name: Name of matched rule
    """

    features: AggregateFeatures
    thresholds: dict[str, float]
    rule_name: str


class DecisionEngine:
    """Decision engine for ANSWER/ASK/ABSTAIN.

    Per spec §4.8: Applies ordered rules with optional threshold overrides.
    """

    def __init__(self, thresholds: dict[str, float] | None = None):
        """Initialize decision engine.

        Args:
            thresholds: Optional threshold overrides (uses defaults if None)
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        log.info("decision_engine_initialized", thresholds=self.thresholds)

    def decide(self, features: AggregateFeatures) -> Decision:
        """Make decision based on aggregate features.

        Args:
            features: Aggregate features from clustering

        Returns:
            Decision with action and confidence scores
        """
        log.debug(
            "decision_input",
            p1=round(features.p1, 3),
            p2=round(features.p2, 3),
            ask_score=round(features.ask_score, 3),
            answer_score=round(features.answer_score, 3),
            abstain_score=round(features.abstain_score, 3),
        )

        # Apply rules
        action, rule_name = apply_rules(features, self.thresholds)

        # Compute confidence scores (normalized)
        # Use the three derived scores from features
        confidence_scores = self._compute_confidence_scores(features)

        log.info(
            "decision_made",
            action=action,
            rule=rule_name,
            confidence=round(confidence_scores[action], 3),
        )

        return Decision(
            action=action,
            confidence=confidence_scores[action],
            confidence_scores=confidence_scores,
            rule_name=rule_name,
        )

    def _compute_confidence_scores(self, features: AggregateFeatures) -> dict[Action, float]:
        """Compute confidence scores for each action.

        Uses the derived scores from features (§4.6):
        - answer_score: p1 * (1 - interpretation_divergence)
        - ask_score: (1 - p1)^gamma * interpretation_divergence
        - abstain_score: max(refusal_rate, invalid_rate) * (1 - p1)

        Normalizes to sum to 1.0.

        Args:
            features: Aggregate features

        Returns:
            Dict mapping Action to confidence score
        """
        raw_scores = {
            Action.ANSWER: features.answer_score,
            Action.ASK: features.ask_score,
            Action.ABSTAIN: features.abstain_score,
        }

        # Normalize to sum to 1.0
        total = sum(raw_scores.values())
        if total > 0:
            normalized = {action: score / total for action, score in raw_scores.items()}
        else:
            # All zeros: uniform distribution
            normalized = {
                Action.ANSWER: 1.0 / 3.0,
                Action.ASK: 1.0 / 3.0,
                Action.ABSTAIN: 1.0 / 3.0,
            }

        return normalized

    def update_thresholds(self, thresholds: dict[str, float]) -> None:
        """Update decision thresholds.

        Args:
            thresholds: New threshold values (partial updates allowed)
        """
        self.thresholds.update(thresholds)
        log.info("thresholds_updated", thresholds=self.thresholds)


__all__ = ["DecisionContext", "DecisionEngine"]
