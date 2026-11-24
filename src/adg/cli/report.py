"""Report generation CLI commands.

Per spec §12.5: Generate tables and figures from run artifacts.
"""

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from adg.core.logging import get_logger
from adg.reporting.plots import (
    plot_budget_sweep,
    plot_cluster_histogram,
    plot_confusion_matrix,
    plot_feature_scatter,
    plot_risk_coverage_curve,
)
from adg.reporting.tables import (
    generate_ablation_table,
    generate_cost_table,
    generate_main_results_table,
)

app = typer.Typer(help="Generate reports from experiment results")
console = Console()
log = get_logger(__name__)


@app.command()
def generate(  # noqa: PLR0912, PLR0915
    run_id: str = typer.Argument(..., help="Run ID to generate report for"),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("results/reports"),
        "--output",
        "-o",
        help="Output directory for reports",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Table format (markdown, latex, csv)",
    ),
) -> None:
    """Generate tables and figures from a run.

    Per spec §12.5: Creates main results table, cost table, ablation table,
    and all figures (confusion matrix, risk-coverage, budget sweep, etc.).
    """
    rprint(f"[bold]Generating report for run:[/bold] {run_id}")

    # Check for run directory
    run_dir = Path("results/runs") / run_id
    if not run_dir.exists():
        rprint(f"[red]Error:[/red] Run directory not found: {run_dir}")
        raise typer.Exit(1)

    # Load manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        rprint(f"[red]Error:[/red] Manifest not found: {manifest_path}")
        raise typer.Exit(1)

    with manifest_path.open() as f:
        manifest = json.load(f)

    # Check for mock runs
    if manifest.get("mock", False):
        rprint("[yellow]Warning:[/yellow] This is a mock run.")
        rprint("Per spec: Mock runs should not be reported as real experiments.")
        if not typer.confirm("Continue anyway?"):
            raise typer.Exit(0)

    # Load metrics
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        rprint(f"[red]Error:[/red] Metrics not found: {metrics_path}")
        raise typer.Exit(1)

    with metrics_path.open() as f:
        metrics = json.load(f)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    rprint("[bold]Generating tables...[/bold]")

    # Main results table (ponytail: placeholder structure)
    if "results" in metrics:
        generate_main_results_table(
            metrics["results"],
            output_format=format,
            output_path=output_dir / f"main_results.{format}",
        )
        rprint(f"  ✓ Main results table: {output_dir / f'main_results.{format}'}")

    # Cost table
    if "costs" in metrics:
        generate_cost_table(
            metrics["costs"],
            output_format=format,
            output_path=output_dir / f"costs.{format}",
        )
        rprint(f"  ✓ Cost table: {output_dir / f'costs.{format}'}")

    # Ablations table
    if "ablations" in metrics:
        generate_ablation_table(
            metrics["ablations"],
            output_format=format,
            output_path=output_dir / f"ablations.{format}",
        )
        rprint(f"  ✓ Ablation table: {output_dir / f'ablations.{format}'}")

    # Generate figures
    rprint("[bold]Generating figures...[/bold]")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix (ponytail: only if data available)
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        plot_confusion_matrix(
            cm["matrix"],
            cm.get("labels", ["ANSWER", "ASK", "ABSTAIN"]),
            output_path=figures_dir / "confusion_matrix",
        )
        rprint(f"  ✓ Confusion matrix: {figures_dir / 'confusion_matrix.pdf'}")

    # Risk-coverage curves
    if "risk_coverage" in metrics:
        rc = metrics["risk_coverage"]
        plot_risk_coverage_curve(
            rc["coverages"],
            rc["accuracies"],
            rc["methods"],
            output_path=figures_dir / "risk_coverage",
        )
        rprint(f"  ✓ Risk-coverage curves: {figures_dir / 'risk_coverage.pdf'}")

    # Budget sweep
    if "budget_sweep" in metrics:
        bs = metrics["budget_sweep"]
        plot_budget_sweep(
            bs["budgets"],
            bs["metrics"],
            metric_name=bs.get("metric_name", "F1"),
            output_path=figures_dir / "budget_sweep",
        )
        rprint(f"  ✓ Budget sweep: {figures_dir / 'budget_sweep.pdf'}")

    # Feature scatter
    if "feature_scatter" in metrics:
        fs = metrics["feature_scatter"]
        plot_feature_scatter(
            fs["p1_values"],
            fs["delta_values"],
            fs["gold_actions"],
            output_path=figures_dir / "feature_scatter",
        )
        rprint(f"  ✓ Feature scatter: {figures_dir / 'feature_scatter.pdf'}")

    # Cluster histogram
    if "cluster_histogram" in metrics:
        ch = metrics["cluster_histogram"]
        plot_cluster_histogram(
            ch["cluster_counts"],
            ch["gold_actions"],
            output_path=figures_dir / "cluster_histogram",
        )
        rprint(f"  ✓ Cluster histogram: {figures_dir / 'cluster_histogram.pdf'}")

    rprint("\n[bold green]✓ Report generated successfully![/bold green]")
    rprint(f"Output directory: {output_dir}")


@app.command()
def list() -> None:
    """List available runs."""
    runs_dir = Path("results/runs")

    if not runs_dir.exists():
        rprint("[yellow]No runs directory found.[/yellow]")
        return

    runs = [d for d in runs_dir.iterdir() if d.is_dir()]

    if not runs:
        rprint("[yellow]No runs found.[/yellow]")
        return

    rprint(f"[bold]Available runs ({len(runs)}):[/bold]\n")

    for run in sorted(runs):
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)

            mock_badge = "[yellow][MOCK][/yellow]" if manifest.get("mock") else ""
            rprint(f"  • {run.name} {mock_badge}")

            if "config" in manifest:
                config = manifest["config"]
                rprint(f"    Methods: {', '.join(config.get('methods', []))}")
                rprint(f"    Datasets: {', '.join(config.get('datasets', []))}")
        else:
            rprint(f"  • {run.name} [dim](no manifest)[/dim]")


if __name__ == "__main__":
    app()
