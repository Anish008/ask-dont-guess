"""Chat command for interactive conversation.

Per spec: CLI for multi-turn chat.
"""

import typer
from rich import print as rprint
from rich.prompt import Prompt

from adg.conversation.manager import ConversationManager
from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer


def chat_interactive(
    mock: bool = typer.Option(True, help="Use mock backends (no model loading)"),
) -> None:
    """Start interactive chat session."""
    if not mock:
        rprint("[red]Real models not yet implemented. Use --mock.[/red]")
        raise typer.Exit(1)

    llm = MockLLMBackend()
    nli = MockNLIScorer()
    manager = ConversationManager(llm_backend=llm, nli_scorer=nli)

    session = manager.start_session("cli_chat")

    rprint("[bold]Ask, Don't Guess - Interactive Chat[/bold]")
    rprint("Type 'quit' or 'exit' to end.\n")

    while True:
        user_input = Prompt.ask("[cyan]You[/cyan]")

        if user_input.lower() in {"quit", "exit", "q"}:
            rprint("[dim]Goodbye![/dim]")
            break

        q = Query(query_id=f"cli_{len(session.turns)}", text=user_input)
        session, response, decision = manager.process_query(session, q)

        action_emoji = {"ANSWER": "✅", "ASK": "❓", "ABSTAIN": "⚠️"}
        emoji = action_emoji.get(decision.action, "")

        rprint(f"\n[bold green]ADG[/bold green] {emoji}: {response}\n")
