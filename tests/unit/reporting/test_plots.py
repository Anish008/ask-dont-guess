"""Tests for plot generation."""

from pathlib import Path

import matplotlib.pyplot as plt
from adg.reporting.plots import (
    plot_budget_sweep,
    plot_cluster_histogram,
    plot_confusion_matrix,
    plot_feature_scatter,
    plot_risk_coverage_curve,
)


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Test confusion matrix plotting."""
    cm = [[45, 3, 2], [5, 38, 7], [2, 4, 44]]
    labels = ["ANSWER", "ASK", "ABSTAIN"]

    output_path = tmp_path / "confusion"

    fig = plot_confusion_matrix(cm, labels, output_path=output_path)

    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "confusion.pdf").exists()
    assert (tmp_path / "confusion.png").exists()

    plt.close(fig)


def test_plot_risk_coverage_curve(tmp_path: Path) -> None:
    """Test risk-coverage curve plotting."""
    coverages = [[1.0, 0.8, 0.6], [1.0, 0.9, 0.7]]
    accuracies = [[0.65, 0.72, 0.85], [0.56, 0.58, 0.62]]
    methods = ["ADG-Core", "AlwaysAnswer"]

    output_path = tmp_path / "risk_coverage"

    fig = plot_risk_coverage_curve(
        coverages,
        accuracies,
        methods,
        output_path=output_path,
    )

    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "risk_coverage.pdf").exists()

    plt.close(fig)


def test_plot_budget_sweep(tmp_path: Path) -> None:
    """Test budget sweep plotting."""
    budgets = [1, 4, 8, 16]
    metrics = {
        "ADG-Core": [0.65, 0.78, 0.84, 0.86],
        "SemanticEntropy": [0.60, 0.72, 0.80, 0.82],
    }

    output_path = tmp_path / "budget_sweep"

    fig = plot_budget_sweep(
        budgets,
        metrics,
        metric_name="Ask F1",
        output_path=output_path,
    )

    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "budget_sweep.pdf").exists()

    plt.close(fig)


def test_plot_feature_scatter(tmp_path: Path) -> None:
    """Test feature scatter plotting."""
    p1_values = [0.9, 0.8, 0.5, 0.6, 0.4]
    delta_values = [0.3, 0.25, 0.15, 0.1, 0.05]
    gold_actions = ["ANSWER", "ANSWER", "ASK", "ABSTAIN", "ABSTAIN"]

    output_path = tmp_path / "feature_scatter"

    fig = plot_feature_scatter(
        p1_values,
        delta_values,
        gold_actions,
        output_path=output_path,
    )

    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "feature_scatter.pdf").exists()

    plt.close(fig)


def test_plot_cluster_histogram(tmp_path: Path) -> None:
    """Test cluster histogram plotting."""
    cluster_counts = [1, 1, 2, 3, 2, 1, 4, 3]
    gold_actions = [
        "ANSWER",
        "ANSWER",
        "ASK",
        "ASK",
        "ASK",
        "ABSTAIN",
        "ASK",
        "ABSTAIN",
    ]

    output_path = tmp_path / "cluster_histogram"

    fig = plot_cluster_histogram(
        cluster_counts,
        gold_actions,
        output_path=output_path,
    )

    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "cluster_histogram.pdf").exists()

    plt.close(fig)


def test_plot_without_save() -> None:
    """Test plotting without saving to file."""
    cm = [[45, 3, 2], [5, 38, 7], [2, 4, 44]]
    labels = ["ANSWER", "ASK", "ABSTAIN"]

    fig = plot_confusion_matrix(cm, labels)

    assert isinstance(fig, plt.Figure)

    plt.close(fig)
