"""CLAMBER dataset loader.

Per spec §9.1: Load CLAMBER from official GitHub repository.
TODO(verify): Repository URL, file format, fields, license before real implementation.
"""

from collections.abc import Iterator
from pathlib import Path

from adg.core.errors import DatasetError
from adg.core.logging import get_logger
from adg.data.schema import Example, GoldInterpretation

log = get_logger(__name__)


class CLAMBERLoader:
    """Loader for CLAMBER dataset.

    Per spec §9.1: Taxonomy and benchmark of ambiguous information needs.
    Source: Official GitHub repository (TODO(verify): exact URL)
    """

    def __init__(self, repo_url: str | None = None):
        """Initialize CLAMBER loader.

        Args:
            repo_url: GitHub repository URL (TODO(verify) before use)
        """
        self.repo_url = repo_url or "TODO(verify): official CLAMBER repo"

        log.info("clamber_loader_init", repo_url=self.repo_url)

    def load(
        self,
        split: str = "all",
        cache_dir: Path | None = None,
    ) -> Iterator[Example]:
        """Load CLAMBER examples.

        Args:
            split: Dataset split (may need to create splits if none exist)
            cache_dir: Cache directory for downloaded data

        Yields:
            Example objects in common schema

        Raises:
            DatasetError: If loading fails
        """
        # TODO(verify): Actual implementation requires:
        # 1. Clone/download from GitHub
        # 2. Verify file format (JSON/CSV/JSONL?)
        # 3. Verify fields (which field marks "needs clarification", ambiguity category)
        # 4. Verify license
        # 5. Create deterministic splits if none exist
        log.warning(
            "clamber_load_not_implemented",
            message="CLAMBER loader is a placeholder. Real implementation requires dataset verification.",
        )
        raise DatasetError(
            "CLAMBER loader not yet implemented. "
            "Must verify dataset schema per spec §9.1 before loading."
        )

    def verify_schema(self) -> dict[str, str]:
        """Verify CLAMBER schema.

        Returns:
            Metadata dict with repo_url, fields, license

        Raises:
            DatasetError: If verification fails
        """
        # TODO(verify): Download and check:
        # - File format
        # - Fields: clarification_needed flag, ambiguity category, gold clarifying question
        # - License
        # Record findings in docs/VERIFIED_FACTS.md
        return {
            "repo_url": self.repo_url,
            "status": "unverified",
            "expected_fields": "needs_clarification, ambiguity_category, gold_clarifying_question",
            "license": "TODO(verify)",
        }

    def _map_to_common_schema(self, raw_example: dict, example_idx: int) -> Example:
        """Map raw CLAMBER example to common schema.

        Per spec §9.2: CLAMBER → map clarification_needed to is_ambiguous.

        Args:
            raw_example: Raw example from CLAMBER
            example_idx: Index for generating example_id

        Returns:
            Example in common schema
        """
        example_id = f"clamber:all:{example_idx}"

        # Map clarification_needed to is_ambiguous
        is_ambiguous = raw_example.get("needs_clarification", False)
        ambiguity_category = raw_example.get("ambiguity_category")
        gold_clarifying_q = raw_example.get("gold_clarifying_question")

        # CLAMBER may not provide interpretations, used for ASK metrics only
        interpretations: list[GoldInterpretation] = []
        if "interpretations" in raw_example:
            # TODO: Map if provided
            pass

        return Example(
            example_id=example_id,
            dataset="clamber",
            split="all",
            original_id=str(example_idx),
            question=raw_example.get("question", ""),
            is_ambiguous=is_ambiguous,
            ambiguity_category=ambiguity_category,
            interpretations=interpretations,
            gold_clarifying_question=gold_clarifying_q,
            source_metadata=raw_example,
        )


__all__ = ["CLAMBERLoader"]
