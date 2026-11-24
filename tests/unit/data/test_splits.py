"""Tests for dataset splitting."""

import pytest
from adg.data.schema import Example
from adg.data.splits import create_calibration_split, create_dev_subset


class TestCalibrationSplit:
    """Tests for calibration/test splitting."""

    def test_split_proportions(self) -> None:
        """Test that split creates correct proportions."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
                is_ambiguous=i % 2 == 0,
            )
            for i in range(100)
        ]

        calibration, test = create_calibration_split(examples, calibration_fraction=0.3, seed=42)

        assert len(calibration) + len(test) == 100
        # Allow some tolerance for stratification
        assert 25 <= len(calibration) <= 35  # ~30%
        assert 65 <= len(test) <= 75  # ~70%

    def test_stratification(self) -> None:
        """Test that stratification maintains ambiguous/clear ratios."""
        # Create 50 ambiguous, 50 clear
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
                is_ambiguous=i < 50,
            )
            for i in range(100)
        ]

        calibration, test = create_calibration_split(examples, calibration_fraction=0.3, seed=42)

        # Count ambiguous in each split
        cal_ambiguous = sum(1 for ex in calibration if ex.is_ambiguous)
        test_ambiguous = sum(1 for ex in test if ex.is_ambiguous)

        # Stratification should preserve ~50% ambiguous in both
        cal_ratio = cal_ambiguous / len(calibration) if calibration else 0
        test_ratio = test_ambiguous / len(test) if test else 0

        assert 0.4 <= cal_ratio <= 0.6
        assert 0.4 <= test_ratio <= 0.6

    def test_deterministic_split(self) -> None:
        """Test that same seed produces same split."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
                is_ambiguous=i % 2 == 0,
            )
            for i in range(50)
        ]

        cal1, test1 = create_calibration_split(examples, seed=42)
        cal2, test2 = create_calibration_split(examples, seed=42)

        # Same seed should produce same split
        cal1_ids = {ex.example_id for ex in cal1}
        cal2_ids = {ex.example_id for ex in cal2}
        assert cal1_ids == cal2_ids

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different splits."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
                is_ambiguous=i % 2 == 0,
            )
            for i in range(50)
        ]

        cal1, _ = create_calibration_split(examples, seed=42)
        cal2, _ = create_calibration_split(examples, seed=123)

        # Different seeds should produce different splits
        cal1_ids = {ex.example_id for ex in cal1}
        cal2_ids = {ex.example_id for ex in cal2}
        assert cal1_ids != cal2_ids

    def test_invalid_fraction(self) -> None:
        """Test that invalid fraction raises error."""
        examples = [
            Example(
                example_id="ex:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="Q",
            )
        ]

        with pytest.raises(ValueError, match="calibration_fraction"):
            create_calibration_split(examples, calibration_fraction=0.0)

        with pytest.raises(ValueError, match="calibration_fraction"):
            create_calibration_split(examples, calibration_fraction=1.0)


class TestDevSubset:
    """Tests for dev subset creation."""

    def test_subset_size(self) -> None:
        """Test that subset has correct size."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
            )
            for i in range(200)
        ]

        subset = create_dev_subset(examples, limit=50, stratify=False, seed=42)

        assert len(subset) == 50

    def test_full_dataset_when_limit_exceeds(self) -> None:
        """Test that returns full dataset when limit exceeds size."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
            )
            for i in range(50)
        ]

        subset = create_dev_subset(examples, limit=100, seed=42)

        assert len(subset) == 50
        subset_ids = {ex.example_id for ex in subset}
        example_ids = {ex.example_id for ex in examples}
        assert subset_ids == example_ids

    def test_stratified_subset(self) -> None:
        """Test stratified sampling maintains ratio."""
        # 70 ambiguous, 30 clear
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
                is_ambiguous=i < 70,
            )
            for i in range(100)
        ]

        subset = create_dev_subset(examples, limit=50, stratify=True, seed=42)

        ambiguous_count = sum(1 for ex in subset if ex.is_ambiguous)
        ratio = ambiguous_count / len(subset)

        # Should maintain ~70% ambiguous
        assert 0.6 <= ratio <= 0.8

    def test_deterministic_subset(self) -> None:
        """Test that same seed produces same subset."""
        examples = [
            Example(
                example_id=f"ex:{i}",
                dataset="ambigqa",
                split="dev",
                original_id=str(i),
                question=f"Q{i}",
            )
            for i in range(100)
        ]

        subset1 = create_dev_subset(examples, limit=30, seed=42)
        subset2 = create_dev_subset(examples, limit=30, seed=42)

        ids1 = {ex.example_id for ex in subset1}
        ids2 = {ex.example_id for ex in subset2}
        assert ids1 == ids2
