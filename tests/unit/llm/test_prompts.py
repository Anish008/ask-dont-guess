"""Test prompt formatting."""

from adg.llm.prompts import (
    format_clarification_prompt,
    format_plain_prompt,
    format_structured_prompt,
)


def test_format_structured_prompt():
    """Test structured prompt formatting."""
    question = "What is the capital of France?"
    prompt = format_structured_prompt(question)

    assert "Question: What is the capital of France?" in prompt
    assert "Interpretation:" in prompt
    assert "Answer:" in prompt
    assert "Confidence:" in prompt


def test_format_plain_prompt():
    """Test plain prompt formatting."""
    question = "What is 2 + 2?"
    prompt = format_plain_prompt(question)

    assert "Question: What is 2 + 2?" in prompt
    assert "Answer:" in prompt
    assert "Interpretation:" not in prompt


def test_format_clarification_prompt():
    """Test clarification generation prompt formatting."""
    question = "Who is the president?"
    interpretations = [
        "President of the United States",
        "President of France",
        "President of the company",
    ]

    prompt = format_clarification_prompt(question, interpretations)

    assert "Who is the president?" in prompt
    assert "President of the United States" in prompt
    assert "President of France" in prompt
    assert "President of the company" in prompt
    assert "1." in prompt
    assert "2." in prompt
    assert "3." in prompt


def test_structured_prompt_multiple_questions():
    """Test structured prompt with multiple questions."""
    question1 = "Question 1?"
    question2 = "Question 2?"

    prompt1 = format_structured_prompt(question1)
    prompt2 = format_structured_prompt(question2)

    assert question1 in prompt1
    assert question2 in prompt2
    assert prompt1 != prompt2
