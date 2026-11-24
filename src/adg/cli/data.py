"""Data commands for dataset building.

Per spec §9: Load and process datasets.
"""

from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(help="Dataset management")


@app.command()
def build(
    datasets: list[str] = typer.Option(  # noqa: B008
        ["ambigqa", "nq_open", "clamber"],
        "--dataset",
        "-d",
        help="Datasets to build",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("data/processed"),
        "--output",
        "-o",
        help="Output directory",
    ),
) -> None:
    """Build processed datasets.

    TODO(verify): Actual dataset loading requires verification per spec §9.1.
    Loaders exist but need testing with real data.
    """
    rprint("[yellow]Dataset building requires verification (Phase 11 TODO)[/yellow]")
    rprint(f"Datasets: {', '.join(datasets)}")
    rprint(f"Output: {output_dir}")
    rprint("\nSteps needed:")
    rprint("1. Verify dataset URLs and schemas")
    rprint("2. Test loaders with real data")
    rprint("3. Run deduplication")
    rprint("4. Create splits")
    raise typer.Exit(1)


@app.command()
def verify() -> None:
    """Verify dataset schemas and access.

    TODO(verify): Test each loader's verify_schema() method.
    """
    rprint("[yellow]Dataset verification not yet implemented[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
