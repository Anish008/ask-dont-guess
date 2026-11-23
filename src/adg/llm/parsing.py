"""Parse structured LLM outputs into Sample objects.

Per spec §4.3: extract interpretation/answer/confidence from structured prompt responses.
"""

import re

from adg.core.logging import get_logger
from adg.core.types import Sample

log = get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

# Refusal patterns (case-insensitive)
REFUSAL_PATTERNS = [
    r"\bi\s+don'?t\s+know\b",
    r"\bcannot\s+answer\b",
    r"\bunable\s+to\s+answer\b",
    r"\bno\s+information\b",
    r"\binsufficient\s+information\b",
    r"\bcannot\s+determine\b",
    r"\bunsure\b",
    r"\buncertain\b",
]

# Hedge patterns indicating multiple interpretations
HEDGE_PATTERNS = [
    r"\bdepends\s+on\b",
    r"\bcould\s+mean\b",
    r"\bmight\s+refer\s+to\b",
    r"\beither.*or\b",
    r"\bmultiple\s+interpretations\b",
]


# ==============================================================================
# Parsing Functions
# ==============================================================================


def parse_structured_output(raw_text: str, sample_id: int) -> Sample:
    """Parse structured LLM output into a Sample object.

    Args:
        raw_text: Raw text from LLM generation
        sample_id: Index in the sampling batch

    Returns:
        Sample object with parsed fields

    Raises:
        ParsingError: If output format is invalid
    """
    # Extract fields using regex
    interpretation_match = re.search(
        r"Interpretation:\s*(.+?)(?=\nAnswer:|\nConfidence:|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    answer_match = re.search(
        r"Answer:\s*(.+?)(?=\nConfidence:|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    confidence_match = re.search(
        r"Confidence:\s*(high|medium|low)",
        raw_text,
        re.IGNORECASE,
    )

    # Validate required fields
    if not interpretation_match or not answer_match:
        # Try to extract any text as answer if parsing fails
        log.warning(
            "structured_parse_failed",
            raw_text=raw_text[:200],
            sample_id=sample_id,
        )
        # Mark as invalid
        return Sample(
            interpretation="",
            answer="",
            sample_id=sample_id,
            is_valid=False,
            raw_text=raw_text,
        )

    # Extract text
    interpretation = interpretation_match.group(1).strip()
    answer = answer_match.group(1).strip()
    confidence = confidence_match.group(1).lower() if confidence_match else "medium"

    # Empty check and cross-field contamination check
    if not interpretation or not answer:
        log.warning(
            "empty_fields",
            interpretation_empty=not interpretation,
            answer_empty=not answer,
            sample_id=sample_id,
        )
        return Sample(
            interpretation=interpretation or "",
            answer=answer or "",
            sample_id=sample_id,
            is_valid=False,
            raw_text=raw_text,
        )

    # Check for field contamination (one field containing another field's label)
    if any(keyword in interpretation.lower() for keyword in ["answer:", "confidence:"]):
        log.warning("field_contamination", field="interpretation", sample_id=sample_id)
        return Sample(
            interpretation=interpretation,
            answer=answer,
            sample_id=sample_id,
            is_valid=False,
            raw_text=raw_text,
        )

    if "interpretation:" in answer.lower() or "confidence:" in answer.lower():
        log.warning("field_contamination", field="answer", sample_id=sample_id)
        return Sample(
            interpretation=interpretation,
            answer=answer,
            sample_id=sample_id,
            is_valid=False,
            raw_text=raw_text,
        )

    # Detect refusals
    is_refusal = _is_refusal(answer) or _is_refusal(interpretation)

    # Detect hedged responses
    is_hedged = _is_hedged(interpretation)
    hedged_interpretations = []
    if is_hedged:
        hedged_interpretations = _extract_hedged_interpretations(interpretation)

    return Sample(
        interpretation=interpretation,
        answer=answer,
        confidence=confidence,  # type: ignore[arg-type]
        sample_id=sample_id,
        is_valid=True,
        is_refusal=is_refusal,
        is_hedged=is_hedged,
        hedged_interpretations=hedged_interpretations,
        raw_text=raw_text,
    )


def parse_plain_output(raw_text: str, sample_id: int) -> Sample:
    """Parse plain LLM output (no structured fields).

    Args:
        raw_text: Raw text from LLM generation
        sample_id: Index in the sampling batch

    Returns:
        Sample object with answer set to raw text
    """
    answer = raw_text.strip()

    if not answer:
        return Sample(
            interpretation="",
            answer="",
            sample_id=sample_id,
            is_valid=False,
            raw_text=raw_text,
        )

    is_refusal = _is_refusal(answer)

    return Sample(
        interpretation=answer,  # Use answer as interpretation for plain mode
        answer=answer,
        sample_id=sample_id,
        is_valid=True,
        is_refusal=is_refusal,
        raw_text=raw_text,
    )


# ==============================================================================
# Helper Functions
# ==============================================================================


def _is_refusal(text: str) -> bool:
    """Check if text contains refusal patterns.

    Args:
        text: Text to check

    Returns:
        True if refusal detected
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in REFUSAL_PATTERNS)


def _is_hedged(text: str) -> bool:
    """Check if text contains hedge patterns indicating multiple interpretations.

    Args:
        text: Text to check

    Returns:
        True if hedging detected
    """
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in HEDGE_PATTERNS)


def _extract_hedged_interpretations(text: str) -> list[str]:
    """Extract individual interpretations from a hedged response.

    Args:
        text: Hedged interpretation text

    Returns:
        List of individual interpretations (may be empty)
    """
    # Look for numbered or bulleted lists
    # Match patterns like "1. ...", "- ...", "* ..."
    list_items = re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*])\s*(.+?)(?=\n|$)", text)

    if list_items:
        return [item.strip() for item in list_items if item.strip()]

    # Look for "either X or Y" patterns
    either_or = re.search(r"either\s+(.+?)\s+or\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if either_or:
        return [either_or.group(1).strip(), either_or.group(2).strip()]

    # If no clear structure, return empty list
    return []


__all__ = [
    "HEDGE_PATTERNS",
    "REFUSAL_PATTERNS",
    "parse_plain_output",
    "parse_structured_output",
]
