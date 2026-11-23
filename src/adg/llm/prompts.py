"""Prompt templates for LLM backends.

Per spec §4.3: structured vs plain prompts.
"""

# ==============================================================================
# Structured Prompt (with interpretation, answer, confidence)
# ==============================================================================

STRUCTURED_PROMPT_TEMPLATE = """Question: {question}

Answer the question above. First, write your interpretation of what is being asked. Then provide your answer. Finally, rate your confidence (high/medium/low).

Interpretation:
Answer:
Confidence:"""


# ==============================================================================
# Plain Prompt (question only)
# ==============================================================================

PLAIN_PROMPT_TEMPLATE = """Question: {question}

Answer:"""


# ==============================================================================
# Clarification Generation Prompt
# ==============================================================================

CLARIFICATION_PROMPT_TEMPLATE = """You are helping to disambiguate an ambiguous question by generating a clarifying question.

Original question: {question}

The question has multiple possible interpretations:
{interpretations}

Generate a natural, clear question that asks the user which interpretation they meant. Present the interpretations as options.

Clarifying question:"""


# ==============================================================================
# Helper Functions
# ==============================================================================


def format_structured_prompt(question: str) -> str:
    """Format a structured prompt with interpretation/answer/confidence fields.

    Args:
        question: User's question

    Returns:
        Formatted prompt string
    """
    return STRUCTURED_PROMPT_TEMPLATE.format(question=question)


def format_plain_prompt(question: str) -> str:
    """Format a plain prompt with question only.

    Args:
        question: User's question

    Returns:
        Formatted prompt string
    """
    return PLAIN_PROMPT_TEMPLATE.format(question=question)


def format_clarification_prompt(question: str, interpretations: list[str]) -> str:
    """Format a clarification generation prompt.

    Args:
        question: Original ambiguous question
        interpretations: List of possible interpretations

    Returns:
        Formatted prompt string
    """
    interp_text = "\n".join(f"{i+1}. {interp}" for i, interp in enumerate(interpretations))
    return CLARIFICATION_PROMPT_TEMPLATE.format(
        question=question,
        interpretations=interp_text,
    )


__all__ = [
    "CLARIFICATION_PROMPT_TEMPLATE",
    "PLAIN_PROMPT_TEMPLATE",
    "STRUCTURED_PROMPT_TEMPLATE",
    "format_clarification_prompt",
    "format_plain_prompt",
    "format_structured_prompt",
]
