"""AmbigQA dataset loader.

Per spec §9.1: Load AmbigQA from Hugging Face Hub.
TODO(verify): Dataset ID, fields, license before real implementation.
"""

from collections.abc import Iterator
from pathlib import Path

from adg.core.errors import DatasetError
from adg.core.logging import get_logger
from adg.data.schema import Example, GoldInterpretation

log = get_logger(__name__)


class AmbigQALoader:
    """Loader for AmbigQA dataset.

    Per spec §9.1: Ambiguous open-domain questions with disambiguated QA pairs.
    Source: Hugging Face Hub (TODO(verify): exact dataset_id)
    """

    def __init__(self, dataset_id: str = "sewon/ambig_qa", config: str = "light"):
        """Initialize AmbigQA loader.

        Args:
            dataset_id: Hugging Face dataset ID (TODO(verify) before use)
            config: Dataset config (light/full)
        """
        self.dataset_id = dataset_id
        self.config = config

        log.info("ambigqa_loader_init", dataset_id=dataset_id, config=config)

    def load(
        self,
        split: str = "validation",
        cache_dir: Path | None = None,
    ) -> Iterator[Example]:
        """Load AmbigQA examples.

        Args:
            split: Dataset split (train/validation)
            cache_dir: Cache directory for HF datasets

        Yields:
            Example objects in common schema

        Raises:
            DatasetError: If loading fails
        """
        # TODO(verify): Actual implementation requires:
        # 1. Install datasets library
        # 2. Verify dataset_id exists on HF Hub
        # 3. Verify field names match spec
        # 4. Verify license
        log.warning(
            "ambigqa_load_not_implemented",
            message="AmbigQA loader is a placeholder. Real implementation requires dataset verification.",
        )
        raise DatasetError(
            "AmbigQA loader not yet implemented. "
            "Must verify dataset schema per spec §9.1 before loading."
        )

    def verify_schema(self) -> dict[str, str]:
        """Verify AmbigQA schema.

        Returns:
            Metadata dict with dataset_id, fields, license

        Raises:
            DatasetError: If verification fails
        """
        # TODO(verify): Load dataset and check fields:
        # Expected fields: id, question, annotations (with type, answer, qaPairs)
        # Record findings in docs/VERIFIED_FACTS.md
        return {
            "dataset_id": self.dataset_id,
            "config": self.config,
            "status": "unverified",
            "expected_fields": "id, question, annotations",
            "license": "TODO(verify)",
        }

    def _map_to_common_schema(self, raw_example: dict) -> Example:
        """Map raw AmbigQA example to common schema.

        Per spec §9.2:
        - multipleQAs → is_ambiguous=True, each qaPair becomes GoldInterpretation
        - singleAnswer → is_ambiguous=False, one interpretation

        Args:
            raw_example: Raw example from HF datasets

        Returns:
            Example in common schema
        """
        example_id = f"ambigqa:validation:{raw_example['id']}"

        # Extract annotations (first annotation as primary)
        annotations = raw_example.get("annotations", [{}])[0]
        annotation_type = annotations.get("type")

        is_ambiguous = annotation_type == "multipleQAs"

        interpretations: list[GoldInterpretation] = []
        gold_answers: list[str] = []

        if is_ambiguous:
            # Multiple interpretations
            qa_pairs = annotations.get("qaPairs", [])
            for i, qa_pair in enumerate(qa_pairs):
                interp = GoldInterpretation(
                    interp_id=f"{example_id}::i{i}",
                    disambiguated_question=qa_pair.get("question", ""),
                    answers=qa_pair.get("answer", []),
                )
                interpretations.append(interp)
                gold_answers.extend(qa_pair.get("answer", []))
        else:
            # Single answer
            interp = GoldInterpretation(
                interp_id=f"{example_id}::i0",
                disambiguated_question=raw_example["question"],
                answers=annotations.get("answer", []),
            )
            interpretations.append(interp)
            gold_answers = annotations.get("answer", [])

        return Example(
            example_id=example_id,
            dataset="ambigqa",
            split="dev",
            original_id=raw_example["id"],
            question=raw_example["question"],
            is_ambiguous=is_ambiguous,
            interpretations=interpretations,
            gold_answers=gold_answers,
            source_metadata={"annotations": annotations},
        )


__all__ = ["AmbigQALoader"]
