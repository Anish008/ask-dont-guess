"""Common data schema for evaluation datasets.

Per spec §9.2: Unified schema for AmbigQA, CLAMBER, NQ-open.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class GoldInterpretation(BaseModel):
    """Single interpretation of an ambiguous question.

    Per spec §9.2: Represents one meaning with its disambiguated form.

    Args:
        interp_id: Stable ID like "{example_id}::i{k}"
        disambiguated_question: Clarified question for this interpretation
        answers: Gold answer aliases for this interpretation
        description: Short label for the interpretation (optional)
    """

    interp_id: str
    disambiguated_question: str
    answers: list[str]
    description: str | None = None


class Example(BaseModel):
    """Unified example schema across datasets.

    Per spec §9.2: Common format for AmbigQA, CLAMBER, NQ-open.

    Args:
        example_id: Stable ID like "{dataset}:{split}:{original_id}"
        dataset: Source dataset name
        split: Dataset split (train/dev/test/all)
        original_id: Original ID from source dataset
        question: The user's original question
        is_ambiguous: Whether question is ambiguous (dataset-annotated)
        ambiguity_category: Ambiguity category (e.g., CLAMBER taxonomy)
        interpretations: Gold interpretations (≥2 if ambiguous, 1 if clear)
        gold_answers: Union of all answers (convenience for clear questions)
        gold_clarifying_question: Gold clarifying question if available
        source_metadata: Untouched original fields
        annotation_notes: Optional notes about annotations
    """

    example_id: str
    dataset: Literal["ambigqa", "nq_open", "clamber", "clariq"]
    split: Literal["train", "dev", "test", "all"]
    original_id: str

    # Question and ambiguity
    question: str
    is_ambiguous: bool | None = None  # Dataset-annotated
    ambiguity_category: str | None = None

    # Gold data
    interpretations: list[GoldInterpretation] = Field(default_factory=list)
    gold_answers: list[str] = Field(default_factory=list)
    gold_clarifying_question: str | None = None

    # Metadata
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    annotation_notes: str | None = None


__all__ = [
    "Example",
    "GoldInterpretation",
]
