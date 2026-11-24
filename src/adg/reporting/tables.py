"""Table generation for reports.

Per spec §12.5: Markdown, LaTeX (booktabs), CSV/JSON output.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from adg.core.logging import get_logger

log = get_logger(__name__)


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Convert DataFrame to markdown table.

    ponytail: Simple markdown generation without tabulate dependency.

    Args:
        df: DataFrame to convert
        floatfmt: Float format string

    Returns:
        Markdown table string
    """
    # Get column names
    cols = df.columns.tolist()

    # Build header
    lines = []
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    lines.append(header)
    lines.append(separator)

    # Build rows
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                # Format floats
                precision = int(floatfmt.split(".")[1].rstrip("f"))
                cells.append(f"{val:.{precision}f}")
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append("-")
            else:
                cells.append(str(val))

        row_str = "| " + " | ".join(cells) + " |"
        lines.append(row_str)

    return "\n".join(lines)


def generate_main_results_table(
    results: list[dict[str, Any]],
    output_format: Literal["markdown", "latex", "csv"] = "markdown",
    output_path: Path | None = None,
) -> str:
    """Generate main results table.

    Per spec §12.5: Table 1 with Ask P/R/F1, AUROC, 3-way macro-F1,
    E2E accuracy, mean samples, mean latency, with ± CI.

    Args:
        results: List of result dicts with method, metrics, CIs
        output_format: Output format (markdown, latex, csv)
        output_path: Optional path to save table

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Format table based on output format
    if output_format == "markdown":
        table_str = _df_to_markdown(df, floatfmt=".3f")
    elif output_format == "latex":
        # booktabs style
        table_str = df.to_latex(
            index=False,
            float_format="%.3f",
            caption="Main Results",
            label="tab:main_results",
        )
    elif output_format == "csv":
        table_str = df.to_csv(index=False)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table_str)
        log.info("table_saved", path=str(output_path), format=output_format)

    return table_str


def generate_cost_table(
    results: list[dict[str, Any]],
    output_format: Literal["markdown", "latex", "csv"] = "markdown",
    output_path: Path | None = None,
) -> str:
    """Generate cost breakdown table.

    Per spec §12.5: Table 3 with samples, calls, tokens,
    NLI pairs, latency by method.

    Args:
        results: List of cost dicts with method, cost metrics
        output_format: Output format (markdown, latex, csv)
        output_path: Optional path to save table

    Returns:
        Formatted table string
    """
    if not results:
        return "No cost data to display."

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Format table based on output format
    if output_format == "markdown":
        table_str = _df_to_markdown(df, floatfmt=".2f")
    elif output_format == "latex":
        table_str = df.to_latex(
            index=False,
            float_format="%.2f",
            caption="Cost Breakdown by Method",
            label="tab:costs",
        )
    elif output_format == "csv":
        table_str = df.to_csv(index=False)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table_str)
        log.info("cost_table_saved", path=str(output_path), format=output_format)

    return table_str


def generate_ablation_table(
    results: list[dict[str, Any]],
    output_format: Literal["markdown", "latex", "csv"] = "markdown",
    output_path: Path | None = None,
) -> str:
    """Generate ablation study table.

    Per spec §12.5: Table 4 with ablations (clustering algorithm,
    threshold choice, etc.).

    Args:
        results: List of ablation result dicts
        output_format: Output format (markdown, latex, csv)
        output_path: Optional path to save table

    Returns:
        Formatted table string
    """
    if not results:
        return "No ablation data to display."

    df = pd.DataFrame(results)

    if output_format == "markdown":
        table_str = _df_to_markdown(df, floatfmt=".3f")
    elif output_format == "latex":
        table_str = df.to_latex(
            index=False,
            float_format="%.3f",
            caption="Ablation Study Results",
            label="tab:ablations",
        )
    elif output_format == "csv":
        table_str = df.to_csv(index=False)
    else:
        raise ValueError(f"Unknown format: {output_format}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table_str)
        log.info("ablation_table_saved", path=str(output_path), format=output_format)

    return table_str


def format_ci(mean: float, lower: float, upper: float) -> str:
    """Format mean with confidence interval.

    Args:
        mean: Mean value
        lower: Lower bound
        upper: Upper bound

    Returns:
        Formatted string like "0.85 ± 0.03"
    """
    margin = (upper - lower) / 2
    return f"{mean:.3f} ± {margin:.3f}"


__all__ = [
    "generate_main_results_table",
    "generate_cost_table",
    "generate_ablation_table",
    "format_ci",
]
