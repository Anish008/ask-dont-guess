"""Base dataset loader protocol.

Per spec §9: Common interface for all dataset loaders.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from adg.data.schema import Example


class DatasetLoader(Protocol):
    """Protocol for dataset loaders.

    All loaders (AmbigQA, CLAMBER, NQ-open) implement this interface.
    """

    def load(
        self,
        split: str = "dev",
        cache_dir: Path | None = None,
    ) -> Iterator[Example]:
        """Load dataset examples.

        Args:
            split: Dataset split to load (train/dev/test)
            cache_dir: Cache directory for downloaded data

        Yields:
            Example objects in common schema

        Raises:
            DatasetError: If loading fails
        """
        ...

    def verify_schema(self) -> dict[str, str]:
        """Verify dataset schema and return metadata.

        Returns:
            Dict with dataset_id, fields, license, etc.
        """
        ...


__all__ = ["DatasetLoader"]
