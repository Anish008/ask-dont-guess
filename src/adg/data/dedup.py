"""Cross-dataset deduplication.

Per spec §9.3: AmbigQA questions are drawn from NQ-open.
NQ-open control set must exclude any question appearing in AmbigQA.
"""

import re

from adg.core.logging import get_logger
from adg.data.schema import Example

log = get_logger(__name__)


def normalize_question(question: str) -> str:
    """Normalize question for deduplication matching.

    Per spec §9.3: Lowercase, strip punctuation and whitespace.

    Args:
        question: Raw question text

    Returns:
        Normalized question
    """
    # Lowercase
    normalized = question.lower()

    # Strip punctuation and extra whitespace
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def deduplicate_nq_open_against_ambigqa(
    nq_open_examples: list[Example],
    ambigqa_examples: list[Example],
) -> tuple[list[Example], int]:
    """Remove NQ-open examples that appear in AmbigQA.

    Per spec §9.3: Leakage prevention - AmbigQA ⊂ NQ-open.

    Args:
        nq_open_examples: NQ-open examples
        ambigqa_examples: AmbigQA examples

    Returns:
        Tuple of (deduplicated NQ-open examples, number removed)
    """
    # Build set of normalized AmbigQA questions
    ambigqa_questions = {normalize_question(ex.question) for ex in ambigqa_examples}

    log.info(
        "dedup_starting",
        nq_open_count=len(nq_open_examples),
        ambigqa_question_count=len(ambigqa_questions),
    )

    # Filter NQ-open
    deduplicated = []
    removed_count = 0

    for ex in nq_open_examples:
        normalized = normalize_question(ex.question)
        if normalized in ambigqa_questions:
            removed_count += 1
            log.debug(
                "dedup_removed",
                example_id=ex.example_id,
                question=ex.question[:50],
            )
        else:
            deduplicated.append(ex)

    log.info(
        "dedup_complete",
        original_count=len(nq_open_examples),
        removed_count=removed_count,
        final_count=len(deduplicated),
    )

    return deduplicated, removed_count


__all__ = [
    "deduplicate_nq_open_against_ambigqa",
    "normalize_question",
]
