"""Tests for table generation."""

from pathlib import Path

import pytest
from adg.reporting.tables import (
    format_ci,
    generate_ablation_table,
    generate_cost_table,
    generate_main_results_table,
)


@pytest.fixture
def sample_results() -> list[dict]:
    """Sample results data."""
    return [
        {
            "method": "ADG-Core",
            "ask_precision": 0.85,
            "ask_recall": 0.82,
            "ask_f1": 0.835,
            "auroc": 0.91,
            "macro_f1": 0.78,
            "e2e_accuracy": 0.74,
        },
        {
            "method": "AlwaysAnswer",
            "ask_precision": 0.0,
            "ask_recall": 0.0,
            "ask_f1": 0.0,
            "auroc": None,
            "macro_f1": 0.42,
            "e2e_accuracy": 0.56,
        },
    ]


@pytest.fixture
def sample_costs() -> list[dict]:
    """Sample cost data."""
    return [
        {
            "method": "ADG-Core",
            "mean_samples": 8.0,
            "mean_llm_calls": 1.0,
            "mean_tokens": 756.3,
            "mean_nli_pairs": 28.0,
            "mean_latency_ms": 1250.5,
        },
        {
            "method": "AlwaysAnswer",
            "mean_samples": 1.0,
            "mean_llm_calls": 1.0,
            "mean_tokens": 94.5,
            "mean_nli_pairs": 0.0,
            "mean_latency_ms": 85.2,
        },
    ]


def test_generate_main_results_markdown(
    sample_results: list[dict],
    tmp_path: Path,
) -> None:
    """Test main results table in markdown format."""
    output_path = tmp_path / "results.md"

    table_str = generate_main_results_table(
        sample_results,
        output_format="markdown",
        output_path=output_path,
    )

    assert "ADG-Core" in table_str
    assert "AlwaysAnswer" in table_str
    assert "ask_precision" in table_str
    assert output_path.exists()


def test_generate_main_results_latex(sample_results: list[dict]) -> None:
    """Test main results table in LaTeX format."""
    table_str = generate_main_results_table(
        sample_results,
        output_format="latex",
    )

    assert "\\begin{tabular}" in table_str
    assert "ADG-Core" in table_str
    assert "\\end{tabular}" in table_str


def test_generate_main_results_csv(sample_results: list[dict]) -> None:
    """Test main results table in CSV format."""
    table_str = generate_main_results_table(
        sample_results,
        output_format="csv",
    )

    assert "method,ask_precision" in table_str
    assert "ADG-Core," in table_str


def test_generate_cost_table(sample_costs: list[dict], tmp_path: Path) -> None:
    """Test cost table generation."""
    output_path = tmp_path / "costs.md"

    table_str = generate_cost_table(
        sample_costs,
        output_format="markdown",
        output_path=output_path,
    )

    assert "ADG-Core" in table_str
    assert "mean_samples" in table_str
    assert output_path.exists()


def test_generate_ablation_table(tmp_path: Path) -> None:
    """Test ablation table generation."""
    ablations = [
        {
            "variant": "Greedy clustering",
            "ask_f1": 0.82,
            "macro_f1": 0.75,
        },
        {
            "variant": "Agglomerative clustering",
            "ask_f1": 0.84,
            "macro_f1": 0.77,
        },
    ]

    output_path = tmp_path / "ablations.md"

    table_str = generate_ablation_table(
        ablations,
        output_format="markdown",
        output_path=output_path,
    )

    assert "Greedy clustering" in table_str
    assert "Agglomerative clustering" in table_str
    assert output_path.exists()


def test_generate_empty_results() -> None:
    """Test table generation with empty results."""
    table_str = generate_main_results_table([])

    assert "No results" in table_str


def test_format_ci() -> None:
    """Test confidence interval formatting."""
    formatted = format_ci(0.85, 0.82, 0.88)

    assert "0.850" in formatted
    assert "0.030" in formatted
    assert "±" in formatted


def test_invalid_format(sample_results: list[dict]) -> None:
    """Test error handling for invalid format."""
    with pytest.raises(ValueError, match="Unknown format"):
        generate_main_results_table(sample_results, output_format="invalid")
