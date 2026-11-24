"""Experiment commands for running evaluations.

Per spec §12: Grid search over methods, models, datasets.
"""

from pathlib import Path

import typer
import yaml
from rich import print as rprint

from adg.evaluation.runner import ExperimentConfig

app = typer.Typer(help="Experiment management")


@app.command()
def run(
    config_file: Path = typer.Argument(..., help="Experiment config YAML"),  # noqa: B008
    mock: bool = typer.Option(True, help="Use mock backends"),
) -> None:
    """Run experiment from config file."""
    if not config_file.exists():
        rprint(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    # Load config
    with config_file.open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = ExperimentConfig(**config_dict)

    rprint(f"[bold]Experiment:[/bold] {config.name}")
    rprint(f"Methods: {', '.join(config.methods)}")
    rprint(f"Datasets: {', '.join(config.datasets)}")
    rprint(f"Seeds: {config.seeds}")

    if not mock:
        rprint("[yellow]Real models require full baseline implementations[/yellow]")
        rprint("TODO: Load methods and backends from config")
        raise typer.Exit(1)

    # TODO: Create methods dict from config
    # TODO: Load examples from datasets
    # simuser = RuleBasedSimulatedUser(similarity_threshold=0.7)
    # runner = ExperimentRunner(config=config, simulated_user=simuser)
    # result = runner.run(methods=methods, examples=examples)
    # output_path = runner.save_results(result)
    # rprint(f"\n[green]Results saved:[/green] {output_path}")

    rprint("\n[yellow]Full experiment runner pending:[/yellow]")
    rprint("1. Method loading from config")
    rprint("2. Dataset loading")
    rprint("3. Real model support")
    raise typer.Exit(1)


@app.command()
def list_runs(
    runs_dir: Path = typer.Option(  # noqa: B008
        Path("results/runs"),
        "--dir",
        "-d",
        help="Runs directory",
    ),
) -> None:
    """List completed experiment runs."""
    if not runs_dir.exists():
        rprint(f"[yellow]No runs directory: {runs_dir}[/yellow]")
        return

    results = list(runs_dir.glob("*_results.json"))

    if not results:
        rprint("[yellow]No experiment results found[/yellow]")
        return

    rprint(f"[bold]Found {len(results)} experiment runs:[/bold]\n")
    for result_file in sorted(results):
        rprint(f"  • {result_file.stem.replace('_results', '')}")


if __name__ == "__main__":
    app()
