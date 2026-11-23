"""Test threshold calibration."""

from adg.core.types import Action, AggregateFeatures, Decision
from adg.decision.calibration import (
    ThresholdCalibrator,
    accuracy_metric,
    ask_f1_metric,
    ask_precision_metric,
)


def test_threshold_calibrator_initialization():
    """Test calibrator initialization."""
    calibrator = ThresholdCalibrator(grid_resolution=5)
    assert calibrator.grid_resolution == 5
    assert "refusal_rate" in calibrator.threshold_ranges


def test_threshold_calibrator_initialization_custom_ranges():
    """Test calibrator with custom threshold ranges."""
    custom_ranges = {
        "refusal_rate": (0.1, 0.9),
        "invalid_rate": (0.1, 0.9),
        "ask_score": (0.2, 0.8),
        "answer_score": (0.3, 0.7),
    }
    calibrator = ThresholdCalibrator(threshold_ranges=custom_ranges)
    assert calibrator.threshold_ranges == custom_ranges


def test_threshold_calibrator_calibrate():
    """Test threshold calibration."""
    # Create simple dev set
    dev_features = [
        # High answer score -> should be ANSWER
        AggregateFeatures(
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
        ),
        # High ask score -> should be ASK
        AggregateFeatures(
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
        ),
    ]

    dev_labels = [Action.ANSWER, Action.ASK]

    calibrator = ThresholdCalibrator(grid_resolution=5)
    best_thresholds = calibrator.calibrate(
        dev_features, dev_labels, accuracy_metric, optimize_thresholds=["answer_score"]
    )

    # Should return threshold dict
    assert isinstance(best_thresholds, dict)
    assert "answer_score" in best_thresholds


def test_accuracy_metric():
    """Test accuracy metric."""
    predictions = [
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={Action.ANSWER: 0.9, Action.ASK: 0.05, Action.ABSTAIN: 0.05},
            rule_name="answer",
        ),
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={Action.ANSWER: 0.2, Action.ASK: 0.6, Action.ABSTAIN: 0.2},
            rule_name="ask",
        ),
        Decision(
            action=Action.ABSTAIN,
            confidence=0.5,
            confidence_scores={
                Action.ANSWER: 0.25,
                Action.ASK: 0.25,
                Action.ABSTAIN: 0.5,
            },
            rule_name="abstain_refusal",
        ),
    ]

    labels = [Action.ANSWER, Action.ASK, Action.ANSWER]  # Last one is wrong

    accuracy = accuracy_metric(predictions, labels)

    # 2 out of 3 correct
    assert accuracy == 2 / 3


def test_accuracy_metric_all_correct():
    """Test accuracy metric with all correct predictions."""
    predictions = [
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={Action.ANSWER: 0.9, Action.ASK: 0.05, Action.ABSTAIN: 0.05},
            rule_name="answer",
        ),
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={Action.ANSWER: 0.2, Action.ASK: 0.6, Action.ABSTAIN: 0.2},
            rule_name="ask",
        ),
    ]

    labels = [Action.ANSWER, Action.ASK]

    accuracy = accuracy_metric(predictions, labels)

    assert accuracy == 1.0


def test_accuracy_metric_empty():
    """Test accuracy metric with empty lists."""
    predictions = []
    labels = []

    accuracy = accuracy_metric(predictions, labels)

    assert accuracy == 0.0


def test_ask_precision_metric():
    """Test ASK precision metric."""
    predictions = [
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={},
            rule_name="answer",
        ),
    ]

    labels = [Action.ASK, Action.ANSWER, Action.ANSWER]  # First ASK correct, second ASK wrong

    precision = ask_precision_metric(predictions, labels)

    # 1 TP, 1 FP -> precision = 1/2
    assert precision == 0.5


def test_ask_precision_metric_no_predictions():
    """Test ASK precision when no ASK predictions."""
    predictions = [
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={},
            rule_name="answer",
        ),
    ]

    labels = [Action.ASK]

    precision = ask_precision_metric(predictions, labels)

    assert precision == 0.0


def test_ask_f1_metric():
    """Test ASK F1 score."""
    predictions = [
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={},
            rule_name="answer",
        ),
    ]

    labels = [Action.ASK, Action.ANSWER, Action.ASK]  # 1 TP, 1 FP, 1 FN

    f1 = ask_f1_metric(predictions, labels)

    # Precision = 1/2, Recall = 1/2, F1 = 0.5
    assert abs(f1 - 0.5) < 1e-6


def test_ask_f1_metric_perfect():
    """Test ASK F1 score with perfect predictions."""
    predictions = [
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={},
            rule_name="answer",
        ),
    ]

    labels = [Action.ASK, Action.ANSWER]

    f1 = ask_f1_metric(predictions, labels)

    assert f1 == 1.0


def test_ask_f1_metric_no_predictions():
    """Test ASK F1 when no ASK predictions."""
    predictions = [
        Decision(
            action=Action.ANSWER,
            confidence=0.9,
            confidence_scores={},
            rule_name="answer",
        ),
    ]

    labels = [Action.ASK]

    f1 = ask_f1_metric(predictions, labels)

    assert f1 == 0.0


def test_ask_f1_metric_no_labels():
    """Test ASK F1 when no ASK labels."""
    predictions = [
        Decision(
            action=Action.ASK,
            confidence=0.6,
            confidence_scores={},
            rule_name="ask",
        ),
    ]

    labels = [Action.ANSWER]

    f1 = ask_f1_metric(predictions, labels)

    assert f1 == 0.0
