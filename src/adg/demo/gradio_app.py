"""Gradio demo application for ADG.

Per spec §10.2: Quick demo with mock backends showing ANSWER/ASK/ABSTAIN flow.
"""

from pathlib import Path

import gradio as gr
import yaml

from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer

log = get_logger(__name__)

# Load curated examples
EXAMPLES_PATH = Path(__file__).parent / "examples.yaml"


def load_examples() -> dict:
    """Load curated demo examples.

    Returns:
        Dict of example categories
    """
    if not EXAMPLES_PATH.exists():
        return {"clear": [], "ambiguous": [], "unknown": []}

    with EXAMPLES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Initialize manager with mock backends (ponytail: global for demo simplicity)
llm_backend = MockLLMBackend()
nli_scorer = MockNLIScorer()
manager = ConversationManager(
    llm_backend=llm_backend,
    nli_scorer=nli_scorer,
)


def process_query(query_text: str, session_id: str = "demo") -> tuple[str, str, str]:
    """Process a query through ADG.

    Args:
        query_text: User's question
        session_id: Session identifier

    Returns:
        Tuple of (response, action_badge, decision_info)
    """
    if not query_text.strip():
        return "Please enter a question.", "", ""

    try:
        # Create query
        query = Query(query_id=f"{session_id}_query", text=query_text)

        # Get or create session
        session = manager.get_session(session_id)
        if session is None:
            session = manager.start_session(session_id)

        # Process query
        session, response, decision = manager.process_query(session, query)

        # Format response based on action
        action = decision.action

        if action == "ANSWER":
            response_text = f"**Answer:** {response.message}"
            action_badge = "✅ ANSWER"
        elif action == "ASK":
            response_text = f"**Question:** {response.message}\n\n"
            if response.options:
                response_text += "**Options:**\n"
                for i, opt in enumerate(response.options, 1):
                    response_text += f"{i}. {opt}\n"
            action_badge = "❓ ASK"
        else:  # ABSTAIN
            response_text = f"**Not confident:** {response.message}"
            action_badge = "⚠️ ABSTAIN"

        # Format decision info
        decision_info = f"""**Decision Rule:** {decision.rule_name}
**Confidence:** {decision.confidence:.2f}

**Confidence Scores:**
"""
        for key, value in decision.confidence_scores.items():
            decision_info += f"- {key}: {value:.3f}\n"

        # Add trace info if available
        if decision.trace:
            decision_info += "\n**Trace:**\n"
            for key, value in decision.trace.items():
                decision_info += f"- {key}: {value}\n"

        return response_text, action_badge, decision_info

    except Exception as e:
        log.error("demo_error", error=str(e))
        return f"Error: {e}", "❌ ERROR", ""


def reset_session(session_id: str = "demo") -> str:
    """Reset the demo session.

    Args:
        session_id: Session identifier

    Returns:
        Status message
    """
    try:
        manager.reset_session(session_id)
        return "Session reset successfully."
    except Exception as e:
        return f"Error resetting session: {e}"


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface.

    Returns:
        Gradio Blocks interface
    """
    examples = load_examples()

    # Flatten examples for Gradio
    example_questions = []
    for category in ["clear", "ambiguous", "unknown"]:
        for ex in examples.get(category, []):
            example_questions.append([ex["question"]])

    with gr.Blocks(title="Ask, Don't Guess Demo") as demo:
        gr.Markdown(
            """
            # Ask, Don't Guess (ADG)

            A system that **detects ambiguity** in questions and asks for clarification
            instead of guessing.

            Try asking a question below, or select from the examples!
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is the capital of Georgia?",
                    lines=2,
                )

                submit_btn = gr.Button("Ask", variant="primary")
                reset_btn = gr.Button("Reset Session", variant="secondary")

                gr.Markdown("### Examples")
                gr.Examples(
                    examples=example_questions[:8],  # Show 8 examples
                    inputs=[query_input],
                    label="Try these:",
                )

            with gr.Column(scale=3):
                action_badge = gr.Textbox(
                    label="Action",
                    interactive=False,
                )

                response_output = gr.Markdown(label="Response")

                with gr.Accordion("Decision Details", open=False):
                    decision_info = gr.Markdown()

        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[response_output, action_badge, decision_info],
        )

        reset_btn.click(
            fn=reset_session,
            outputs=[response_output],
        )

        gr.Markdown(
            """
            ---
            **Note:** This demo uses mock backends for demonstration.
            In production, ADG uses real language models (Qwen2.5) and NLI scoring (DeBERTa).

            **Actions:**
            - ✅ **ANSWER**: Clear question, model is confident
            - ❓ **ASK**: Ambiguous question, needs clarification
            - ⚠️ **ABSTAIN**: Model doesn't know the answer
            """
        )

    return demo


def main() -> None:
    """Launch the Gradio demo."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
