"""Evaluation framework for ADG.

Per spec §12: Metrics, E2E evaluation, experiment runner, statistics.
"""

from adg.evaluation.metrics import (
    compute_asking_metrics,
    compute_three_way_metrics,
    normalize_answer,
    score_answer,
)

__all__ = [
    "compute_asking_metrics",
    "compute_three_way_metrics",
    "normalize_answer",
    "score_answer",
]
