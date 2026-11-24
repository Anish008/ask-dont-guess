"""Tests for statistical tests."""

import pytest
from adg.evaluation.stats import (
    bootstrap_confidence_interval,
    mcnemar_test,
    paired_bootstrap_test,
)


def test_bootstrap_confidence_interval() -> None:
    """Test bootstrap CI computation."""
    data = [0.8, 0.85, 0.9, 0.82, 0.88]

    mean, lower, upper = bootstrap_confidence_interval(
        data,
        confidence=0.95,
        n_bootstrap=1000,
        seed=42,
    )

    assert 0.0 <= mean <= 1.0
    assert lower < mean < upper
    assert lower >= 0.0
    assert upper <= 1.0


def test_bootstrap_ci_empty_data() -> None:
    """Test bootstrap CI with empty data."""
    mean, lower, upper = bootstrap_confidence_interval(
        [],
        n_bootstrap=100,
    )

    assert mean == 0.0
    assert lower == 0.0
    assert upper == 0.0


def test_paired_bootstrap_test_significant() -> None:
    """Test paired bootstrap with significant difference."""
    # Method A consistently better than B
    data_a = [0.9, 0.85, 0.88, 0.92, 0.87]
    data_b = [0.7, 0.65, 0.68, 0.72, 0.67]

    p_value, effect_size = paired_bootstrap_test(
        data_a,
        data_b,
        n_bootstrap=1000,
        seed=42,
    )

    # Should be significant (low p-value)
    assert p_value < 0.1
    # Should have positive effect size (A > B)
    assert effect_size > 0


def test_paired_bootstrap_test_not_significant() -> None:
    """Test paired bootstrap with no difference."""
    # Same data
    data_a = [0.8, 0.85, 0.9, 0.82, 0.88]
    data_b = [0.8, 0.85, 0.9, 0.82, 0.88]

    p_value, effect_size = paired_bootstrap_test(
        data_a,
        data_b,
        n_bootstrap=1000,
        seed=42,
    )

    # Should not be significant (high p-value)
    assert p_value > 0.5
    # Should have near-zero effect size
    assert abs(effect_size) < 0.1


def test_paired_bootstrap_length_mismatch() -> None:
    """Test paired bootstrap with length mismatch."""
    data_a = [0.8, 0.85, 0.9]
    data_b = [0.7, 0.65]

    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_test(data_a, data_b)


def test_mcnemar_test_significant() -> None:
    """Test McNemar test with significant difference."""
    # Method A is better
    pred_a = [True, True, True, False]
    pred_b = [True, False, False, False]
    gold = [True, True, True, False]

    p_value, statistic = mcnemar_test(pred_a, pred_b, gold)

    # Should be significant (Method A gets 2 that B misses)
    assert p_value < 0.5
    assert statistic > 0


def test_mcnemar_test_no_difference() -> None:
    """Test McNemar test with no difference."""
    # Both methods same
    pred_a = [True, True, False, False]
    pred_b = [True, True, False, False]
    gold = [True, True, False, False]

    p_value, statistic = mcnemar_test(pred_a, pred_b, gold)

    # No discordant pairs
    assert p_value == 1.0
    assert statistic == 0.0


def test_mcnemar_test_length_mismatch() -> None:
    """Test McNemar test with length mismatch."""
    pred_a = [True, True]
    pred_b = [True]
    gold = [True, True]

    with pytest.raises(ValueError, match="same length"):
        mcnemar_test(pred_a, pred_b, gold)
