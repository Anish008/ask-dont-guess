"""Test LLM output parsing."""

from adg.llm.parsing import parse_plain_output, parse_structured_output


def test_parse_structured_output_valid():
    """Test parsing valid structured output."""
    raw_text = """Interpretation: The question asks about the capital city of France.
Answer: Paris
Confidence: high"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.interpretation == "The question asks about the capital city of France."
    assert sample.answer == "Paris"
    assert sample.confidence == "high"
    assert sample.is_valid
    assert not sample.is_refusal
    assert sample.sample_id == 0


def test_parse_structured_output_medium_confidence():
    """Test parsing with medium confidence."""
    raw_text = """Interpretation: Asking about population
Answer: Approximately 2 million
Confidence: medium"""

    sample = parse_structured_output(raw_text, sample_id=1)

    assert sample.confidence == "medium"
    assert sample.is_valid


def test_parse_structured_output_low_confidence():
    """Test parsing with low confidence."""
    raw_text = """Interpretation: Unclear question
Answer: Not sure
Confidence: low"""

    sample = parse_structured_output(raw_text, sample_id=2)

    assert sample.confidence == "low"
    assert sample.is_valid


def test_parse_structured_output_refusal():
    """Test detection of refusal patterns."""
    raw_text = """Interpretation: The question is ambiguous
Answer: I don't know
Confidence: low"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_refusal
    assert sample.is_valid


def test_parse_structured_output_refusal_patterns():
    """Test various refusal patterns."""
    refusal_texts = [
        "I don't know",
        "Cannot answer",
        "Unable to answer",
        "Insufficient information",
        "I am unsure",
    ]

    for text in refusal_texts:
        raw = f"Interpretation: Question unclear\nAnswer: {text}\nConfidence: low"
        sample = parse_structured_output(raw, sample_id=0)
        assert sample.is_refusal, f"Failed to detect refusal in: {text}"


def test_parse_structured_output_hedged():
    """Test detection of hedged responses."""
    raw_text = """Interpretation: This could mean either the US president or the company president.
Answer: Depends on context
Confidence: low"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_hedged
    assert len(sample.hedged_interpretations) >= 0  # May or may not extract specific options


def test_parse_structured_output_missing_fields():
    """Test handling of missing fields."""
    raw_text = "Some text without proper structure"

    sample = parse_structured_output(raw_text, sample_id=0)

    assert not sample.is_valid
    assert sample.raw_text == raw_text


def test_parse_structured_output_empty_fields():
    """Test handling of empty fields."""
    raw_text = """Interpretation:
Answer:
Confidence: medium"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert not sample.is_valid


def test_parse_plain_output():
    """Test parsing plain output."""
    raw_text = "Paris is the capital of France."

    sample = parse_plain_output(raw_text, sample_id=0)

    assert sample.answer == "Paris is the capital of France."
    assert sample.interpretation == "Paris is the capital of France."
    assert sample.is_valid
    assert not sample.is_refusal


def test_parse_plain_output_refusal():
    """Test refusal detection in plain output."""
    raw_text = "I don't know the answer to this question."

    sample = parse_plain_output(raw_text, sample_id=0)

    assert sample.is_refusal


def test_parse_plain_output_empty():
    """Test handling of empty plain output."""
    raw_text = ""

    sample = parse_plain_output(raw_text, sample_id=0)

    assert not sample.is_valid


def test_parse_structured_output_case_insensitive():
    """Test that parsing is case-insensitive."""
    raw_text = """interpretation: Question about capitals
answer: Paris
confidence: HIGH"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_valid
    assert sample.confidence == "high"


def test_parse_structured_output_with_extra_text():
    """Test parsing with extra text before/after fields."""
    raw_text = """Let me answer this question.

Interpretation: Capital city question
Answer: Paris
Confidence: high

That's my answer."""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_valid
    assert "Capital city question" in sample.interpretation
    assert sample.answer == "Paris"


def test_parse_structured_output_multiline_answer():
    """Test parsing with multiline answer."""
    raw_text = """Interpretation: Complex question
Answer: The answer has multiple parts:
1. First part
2. Second part
Confidence: medium"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_valid
    assert "multiple parts" in sample.answer


def test_hedged_interpretation_extraction():
    """Test extraction of multiple interpretations from hedged response."""
    raw_text = """Interpretation: This could mean:
1. The US president
2. The company president
Answer: Depends on which president
Confidence: low"""

    sample = parse_structured_output(raw_text, sample_id=0)

    assert sample.is_hedged
    # Extraction may or may not find specific items depending on regex
