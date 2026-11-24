"""Analyze command for single-query analysis.

Per spec: CLI for analyzing a single query.
"""

import typer
from rich import print as rprint
from rich.panel import Panel

from adg.conversation.manager import ConversationManager
from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer


def analyze_query(
    query: str = typer.Argument(..., help="Query text to analyze"),
    mock: bool = typer.Option(True, help="Use mock backends (no model loading)"),
) -> None:
    """Analyze a single query and show decision."""
    # ponytail: mock only for now, real models need config loading
    if not mock:
        rprint("[red]Real models not yet implemented. Use --mock.[/red]")
        raise typer.Exit(1)

    llm = MockLLMBackend()
    nli = MockNLIScorer()
    manager = ConversationManager(llm_backend=llm, nli_scorer=nli)

    # Create query and start session
    session = manager.start_session("cli")
    q = Query(query_id="cli_0", text=query)

    # Process
    rprint(f"\n[bold]Query:[/bold] {query}")
    session, response, decision = manager.process_query(session, q)

    # Show result
    action_colors = {"ANSWER": "green", "ASK": "yellow", "ABSTAIN": "red"}
    action_str = decision.action if isinstance(decision.action, str) else decision.action.value
    color = action_colors.get(action_str, "white")

    rprint(Panel(response, title=f"[{color}]{action_str}[/{color}]"))

    # Show decision details
    rprint("\n[bold]Decision Details:[/bold]")
    rprint(f"  Rule: {decision.rule_name}")
    rprint(f"  Confidence: {decision.confidence:.3f}")
    rprint("  Scores:")
    for action, score in decision.confidence_scores.items():
        rprint(f"    {action}: {score:.3f}")
