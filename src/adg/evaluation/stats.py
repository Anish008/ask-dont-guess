"""Statistical tests and confidence intervals.

Per spec §12.2: Bootstrap CIs, paired tests, McNemar.
"""

import numpy as np
from scipy import stats

from adg.core.logging import get_logger

log = get_logger(__name__)


def bootstrap_confidence_interval(
    data: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    data_array = np.array(data)

    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)
    mean = np.mean(data_array)

    return float(mean), float(lower), float(upper)


def paired_bootstrap_test(
    data_a: list[float],
    data_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap test for comparing two methods.

    Args:
        data_a: Metric values for method A
        data_b: Metric values for method B (paired)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (p_value, effect_size)
    """
    if len(data_a) != len(data_b):
        raise ValueError("Data must be paired (same length)")

    rng = np.random.default_rng(seed)
    data_a_array = np.array(data_a)
    data_b_array = np.array(data_b)

    # Observed difference
    observed_diff = np.mean(data_a_array) - np.mean(data_b_array)

    # Bootstrap null distribution
    # (sample with replacement from paired differences)
    diffs = data_a_array - data_b_array

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_diffs = rng.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs.append(np.mean(sample_diffs))

    # Two-tailed p-value
    bootstrap_diffs_array = np.array(bootstrap_diffs)
    p_value = np.mean(np.abs(bootstrap_diffs_array) >= np.abs(observed_diff))

    # Effect size (Cohen's d for paired data)
    effect_size = observed_diff / np.std(diffs) if np.std(diffs) > 0 else 0.0

    return float(p_value), float(effect_size)


def mcnemar_test(
    pred_a: list[bool],
    pred_b: list[bool],
    gold: list[bool],
) -> tuple[float, float]:
    """McNemar test for paired binary predictions.

    Args:
        pred_a: Predictions from method A
        pred_b: Predictions from method B
        gold: Gold labels

    Returns:
        Tuple of (p_value, test_statistic)
    """
    if len(pred_a) != len(pred_b) or len(pred_a) != len(gold):
        raise ValueError("All inputs must have same length")

    # Contingency table
    # Both correct, A correct & B wrong, A wrong & B correct, both wrong
    correct_a = [pa == g for pa, g in zip(pred_a, gold, strict=False)]
    correct_b = [pb == g for pb, g in zip(pred_b, gold, strict=False)]

    a_right_b_wrong = sum(1 for ca, cb in zip(correct_a, correct_b, strict=False) if ca and not cb)
    a_wrong_b_right = sum(1 for ca, cb in zip(correct_a, correct_b, strict=False) if not ca and cb)

    # McNemar's test on discordant pairs
    # Only a_right_b_wrong and a_wrong_b_right matter
    if a_right_b_wrong + a_wrong_b_right == 0:
        # No discordant pairs
        return 1.0, 0.0

    # Continuity correction for small counts
    statistic = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / (
        a_right_b_wrong + a_wrong_b_right
    )

    # Chi-square distribution with 1 df
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return float(p_value), float(statistic)


__all__ = [
    "bootstrap_confidence_interval",
    "mcnemar_test",
    "paired_bootstrap_test",
]
