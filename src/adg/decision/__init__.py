"""Decision engine for ADG system.

Per spec §4.8: Rule-based decision with threshold calibration.
"""

from adg.decision.calibration import (
    ThresholdCalibrator,
    accuracy_metric,
    ask_f1_metric,
    ask_precision_metric,
)
from adg.decision.engine import DecisionContext, DecisionEngine
from adg.decision.rules import (
    DEFAULT_THRESHOLDS,
    ORDERED_RULES,
    DecisionRule,
    apply_rules,
)

__all__ = [
    "DecisionEngine",
    "DecisionContext",
    "DecisionRule",
    "ORDERED_RULES",
    "DEFAULT_THRESHOLDS",
    "apply_rules",
    "ThresholdCalibrator",
    "accuracy_metric",
    "ask_precision_metric",
    "ask_f1_metric",
]
