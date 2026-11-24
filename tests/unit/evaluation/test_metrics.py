"""Tests for evaluation metrics."""

import pytest
from adg.evaluation.metrics import (
    compute_asking_metrics,
    compute_three_way_metrics,
    normalize_answer,
    score_answer,
)


def test_normalize_answer() -> None:
    """Test answer normalization."""
    answer = "The capital is Paris, France!"
    normalized = normalize_answer(answer)

    assert normalized == "capital is paris france"


def test_score_answer_exact_match() -> None:
    """Test exact match scoring."""
    predicted = "Paris"
    gold = ["Paris", "paris"]

    em, f1 = score_answer(predicted, gold)

    assert em == 1.0
    assert f1 == 1.0


def test_score_answer_partial_match() -> None:
    """Test partial token match."""
    predicted = "Paris, France"
    gold = ["Paris"]

    em, f1 = score_answer(predicted, gold)

    assert em == 0.0  # Not exact
    assert f1 > 0.0  # Has token overlap
    assert f1 < 1.0


def test_score_answer_no_match() -> None:
    """Test no match."""
    predicted = "London"
    gold = ["Paris"]

    em, f1 = score_answer(predicted, gold)

    assert em == 0.0
    assert f1 == 0.0


def test_compute_asking_metrics_perfect() -> None:
    """Test asking metrics with perfect predictions."""
    predictions = ["ASK", "ASK", "ANSWER", "ANSWER"]
    gold_ambiguous = [True, True, False, False]

    metrics = compute_asking_metrics(predictions, gold_ambiguous)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.support_ask == 2
    assert metrics.support_total == 4


def test_compute_asking_metrics_with_errors() -> None:
    """Test asking metrics with prediction errors."""
    predictions = ["ASK", "ANSWER", "ASK", "ANSWER"]
    gold_ambiguous = [True, True, False, False]

    metrics = compute_asking_metrics(predictions, gold_ambiguous)

    # Precision: 1 correct ASK out of 2 ASK predictions
    assert metrics.precision == 0.5
    # Recall: 1 correct ASK out of 2 gold ambiguous
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(0.5)


def test_compute_asking_metrics_with_scores() -> None:
    """Test AUROC computation with ask scores."""
    predictions = ["ASK", "ANSWER", "ASK", "ANSWER"]
    gold_ambiguous = [True, True, False, False]
    scores = [0.9, 0.3, 0.8, 0.2]

    metrics = compute_asking_metrics(predictions, gold_ambiguous, scores)

    # AUROC should be computed
    assert metrics.auroc is not None
    assert 0.0 <= metrics.auroc <= 1.0


def test_compute_three_way_metrics() -> None:
    """Test 3-way classification metrics."""
    predictions = ["ANSWER", "ASK", "ABSTAIN", "ANSWER"]
    gold_actions = ["ANSWER", "ASK", "ABSTAIN", "ASK"]
    gold_ambiguous = [False, True, False, True]

    metrics = compute_three_way_metrics(
        predictions,
        gold_actions,
        gold_ambiguous,
    )

    assert metrics.accuracy == 0.75  # 3 out of 4 correct
    assert metrics.macro_f1 > 0.0
    assert len(metrics.confusion) == 3
    assert len(metrics.confusion[0]) == 3


def test_compute_three_way_unnecessary_ask_rate() -> None:
    """Test unnecessary ask rate on clear questions."""
    predictions = ["ASK", "ANSWER", "ASK", "ANSWER"]
    gold_actions = ["ANSWER", "ANSWER", "ABSTAIN", "ABSTAIN"]
    gold_ambiguous = [False, False, False, False]

    metrics = compute_three_way_metrics(
        predictions,
        gold_actions,
        gold_ambiguous,
    )

    # 2 ASK on 4 clear questions
    assert metrics.unnecessary_ask_rate == 0.5
