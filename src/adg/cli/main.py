"""Main CLI entrypoint.

Per spec: Typer CLI with rich progress bars and help text.
"""

import typer
from rich import print as rprint

from adg.cli.analyze import analyze_query
from adg.cli.chat import chat_interactive
from adg.cli.data import app as data_app
from adg.cli.experiment import app as experiment_app
from adg.cli.report import app as report_app

app = typer.Typer(
    name="adg",
    help="Ask, Don't Guess - Ambiguity detection and clarification system",
    no_args_is_help=True,
)

# Add commands and subcommands
app.command(name="analyze")(analyze_query)
app.command(name="chat")(chat_interactive)
app.add_typer(data_app, name="data")
app.add_typer(experiment_app, name="experiment")
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
