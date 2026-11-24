"""Threshold calibration for decision engine.

Per spec §4.8: Grid search to find optimal thresholds on dev split.
"""

from collections.abc import Callable

import numpy as np

from adg.core.logging import get_logger
from adg.core.types import Action, AggregateFeatures, Decision
from adg.decision.engine import DecisionEngine
from adg.decision.rules import DEFAULT_THRESHOLDS

log = get_logger(__name__)


class ThresholdCalibrator:
    """Calibrates decision thresholds on dev data.

    Uses grid search to find thresholds that maximize a given metric.
    """

    def __init__(
        self,
        grid_resolution: int = 10,
        threshold_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        """Initialize calibrator.

        Args:
            grid_resolution: Number of points to test per threshold (default 10)
            threshold_ranges: Optional ranges for each threshold (min, max)
                Default: (0.0, 1.0) for all thresholds
        """
        self.grid_resolution = grid_resolution
        self.threshold_ranges = threshold_ranges or {
            "refusal_rate": (0.0, 1.0),
            "invalid_rate": (0.0, 1.0),
            "ask_score": (0.0, 1.0),
            "answer_score": (0.0, 1.0),
        }
        log.info(
            "calibrator_initialized",
            grid_resolution=grid_resolution,
            threshold_ranges=self.threshold_ranges,
        )

    def calibrate(
        self,
        dev_features: list[AggregateFeatures],
        dev_labels: list[Action],
        metric_fn: Callable[[list[Decision], list[Action]], float],
        optimize_thresholds: list[str] | None = None,
    ) -> dict[str, float]:
        """Calibrate thresholds to maximize metric on dev data.

        Uses grid search over specified thresholds.

        Args:
            dev_features: List of AggregateFeatures for dev examples
            dev_labels: Ground truth actions for dev examples
            metric_fn: Function that takes (predictions, labels) and returns score
            optimize_thresholds: List of threshold names to optimize
                If None, optimizes all thresholds

        Returns:
            Best threshold configuration
        """
        if optimize_thresholds is None:
            optimize_thresholds = list(DEFAULT_THRESHOLDS.keys())

        log.info(
            "calibration_start",
            n_dev=len(dev_features),
            optimize_thresholds=optimize_thresholds,
        )

        # Start with default thresholds
        best_thresholds = DEFAULT_THRESHOLDS.copy()
        best_score = self._evaluate(dev_features, dev_labels, best_thresholds, metric_fn)

        # Grid search for each threshold independently (greedy)
        for threshold_name in optimize_thresholds:
            min_val, max_val = self.threshold_ranges[threshold_name]
            grid = np.linspace(min_val, max_val, self.grid_resolution)

            best_val = best_thresholds[threshold_name]
            for val in grid:
                # Test this threshold value
                test_thresholds = best_thresholds.copy()
                test_thresholds[threshold_name] = float(val)

                score = self._evaluate(dev_features, dev_labels, test_thresholds, metric_fn)

                if score > best_score:
                    best_score = score
                    best_val = float(val)

            # Update best threshold
            best_thresholds[threshold_name] = best_val

            log.debug(
                "threshold_optimized",
                name=threshold_name,
                value=round(best_val, 3),
                score=round(best_score, 3),
            )

        log.info(
            "calibration_complete",
            best_score=round(best_score, 3),
            best_thresholds={k: round(v, 3) for k, v in best_thresholds.items()},
        )

        return best_thresholds

    def _evaluate(
        self,
        dev_features: list[AggregateFeatures],
        dev_labels: list[Action],
        thresholds: dict[str, float],
        metric_fn: Callable[[list[Decision], list[Action]], float],
    ) -> float:
        """Evaluate a threshold configuration.

        Args:
            dev_features: Dev features
            dev_labels: Dev labels
            thresholds: Threshold configuration to test
            metric_fn: Metric function

        Returns:
            Metric score
        """
        engine = DecisionEngine(thresholds=thresholds)
        predictions = [engine.decide(features) for features in dev_features]
        return metric_fn(predictions, dev_labels)


def accuracy_metric(predictions: list[Decision], labels: list[Action]) -> float:
    """Compute accuracy of predictions.

    Args:
        predictions: List of Decision objects
        labels: Ground truth actions

    Returns:
        Accuracy (fraction correct)
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(
        1 for pred, label in zip(predictions, labels, strict=False) if pred.action == label
    )
    return correct / len(predictions)


def ask_precision_metric(predictions: list[Decision], labels: list[Action]) -> float:
    """Compute precision of ASK predictions.

    Args:
        predictions: List of Decision objects
        labels: Ground truth actions

    Returns:
        Precision (TP / (TP + FP))
    """
    ask_preds = [pred.action == Action.ASK for pred in predictions]
    ask_labels = [label == Action.ASK for label in labels]

    true_positives = sum(
        1 for pred, label in zip(ask_preds, ask_labels, strict=False) if pred and label
    )
    predicted_positives = sum(ask_preds)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def ask_f1_metric(predictions: list[Decision], labels: list[Action]) -> float:
    """Compute F1 score for ASK predictions.

    Args:
        predictions: List of Decision objects
        labels: Ground truth actions

    Returns:
        F1 score
    """
    ask_preds = [pred.action == Action.ASK for pred in predictions]
    ask_labels = [label == Action.ASK for label in labels]

    true_positives = sum(
        1 for pred, label in zip(ask_preds, ask_labels, strict=False) if pred and label
    )
    predicted_positives = sum(ask_preds)
    actual_positives = sum(ask_labels)

    if predicted_positives == 0 or actual_positives == 0:
        return 0.0

    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


__all__ = [
    "ThresholdCalibrator",
    "accuracy_metric",
    "ask_f1_metric",
    "ask_precision_metric",
]
