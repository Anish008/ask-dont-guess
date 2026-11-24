"""NQ-open dataset loader.

Per spec §9.1: Load NQ-open from Hugging Face Hub.
TODO(verify): Dataset ID, fields, license before real implementation.
"""

from collections.abc import Iterator
from pathlib import Path

from adg.core.errors import DatasetError
from adg.core.logging import get_logger
from adg.data.schema import Example, GoldInterpretation

log = get_logger(__name__)


class NQOpenLoader:
    """Loader for NQ-open dataset.

    Per spec §9.1: Open-domain QA from Natural Questions.
    Source: Hugging Face Hub (TODO(verify): exact dataset_id)
    """

    def __init__(self, dataset_id: str = "nq_open"):
        """Initialize NQ-open loader.

        Args:
            dataset_id: Hugging Face dataset ID (TODO(verify) before use)
        """
        self.dataset_id = dataset_id

        log.info("nq_open_loader_init", dataset_id=dataset_id)

    def load(
        self,
        split: str = "validation",
        cache_dir: Path | None = None,
    ) -> Iterator[Example]:
        """Load NQ-open examples.

        Args:
            split: Dataset split (train/validation/test)
            cache_dir: Cache directory for HF datasets

        Yields:
            Example objects in common schema

        Raises:
            DatasetError: If loading fails
        """
        # TODO(verify): Actual implementation requires:
        # 1. Install datasets library
        # 2. Verify dataset_id exists on HF Hub
        # 3. Verify field names (expected: question, answer)
        # 4. Verify license
        log.warning(
            "nq_open_load_not_implemented",
            message="NQ-open loader is a placeholder. Real implementation requires dataset verification.",
        )
        raise DatasetError(
            "NQ-open loader not yet implemented. "
            "Must verify dataset schema per spec §9.1 before loading."
        )

    def verify_schema(self) -> dict[str, str]:
        """Verify NQ-open schema.

        Returns:
            Metadata dict with dataset_id, fields, license

        Raises:
            DatasetError: If verification fails
        """
        # TODO(verify): Load dataset and check fields:
        # Expected fields: question, answer (list[str])
        # Record findings in docs/VERIFIED_FACTS.md
        return {
            "dataset_id": self.dataset_id,
            "status": "unverified",
            "expected_fields": "question, answer",
            "license": "TODO(verify)",
        }

    def _map_to_common_schema(self, raw_example: dict, example_idx: int) -> Example:
        """Map raw NQ-open example to common schema.

        Per spec §9.2: NQ-open → is_ambiguous=False, one interpretation.

        Args:
            raw_example: Raw example from HF datasets
            example_idx: Index for generating example_id

        Returns:
            Example in common schema
        """
        example_id = f"nq_open:validation:{example_idx}"

        # NQ-open has single answer (not ambiguous)
        answers = raw_example.get("answer", [])

        interp = GoldInterpretation(
            interp_id=f"{example_id}::i0",
            disambiguated_question=raw_example["question"],
            answers=answers,
        )

        return Example(
            example_id=example_id,
            dataset="nq_open",
            split="dev",
            original_id=str(example_idx),
            question=raw_example["question"],
            is_ambiguous=False,
            interpretations=[interp],
            gold_answers=answers,
        )


__all__ = ["NQOpenLoader"]
