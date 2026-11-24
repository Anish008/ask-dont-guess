"""Main CLI entrypoint.

Per spec: Typer CLI with rich progress bars and help text.
"""

import typer
from rich import print as rprint

from adg.cli.report import app as report_app

app = typer.Typer(
    name="adg",
    help="Ask, Don't Guess - Ambiguity detection and clarification system",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(report_app, name="report")


@app.command()
def version() -> None:
    """Show version information."""
    rprint("[bold]Ask, Don't Guess[/bold] v0.1.0")
    rprint("Research implementation of ambiguity-aware QA")


@app.callback()
def main() -> None:
    """
    Ask, Don't Guess (ADG) CLI.

    Training-free system for detecting query ambiguity and generating
    clarifying questions using semantic clustering.
    """


if __name__ == "__main__":
    app()
