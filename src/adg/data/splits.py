"""Deterministic dataset splitting.

Per spec §9.3: Create calibration/test splits with seeded stratification.
"""

import random
from collections import defaultdict

from adg.core.logging import get_logger
from adg.data.schema import Example

log = get_logger(__name__)


def create_calibration_split(
    examples: list[Example],
    calibration_fraction: float = 0.3,
    seed: int = 42,
    stratify_by: str = "is_ambiguous",
) -> tuple[list[Example], list[Example]]:
    """Create calibration/test split with stratification.

    Per spec §9.3: Hold out 30% for calibration, stratified by ambiguity.
    Uses seeded random split for determinism.

    Args:
        examples: Examples to split
        calibration_fraction: Fraction for calibration (default 0.3)
        seed: Random seed for reproducibility
        stratify_by: Field to stratify by (default: is_ambiguous)

    Returns:
        Tuple of (calibration_examples, test_examples)
    """
    MIN_FRACTION = 0.0
    MAX_FRACTION = 1.0
    if not MIN_FRACTION < calibration_fraction < MAX_FRACTION:
        msg = f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        raise ValueError(msg)

    log.info(
        "creating_calibration_split",
        total=len(examples),
        calibration_fraction=calibration_fraction,
        seed=seed,
        stratify_by=stratify_by,
    )

    # Group by stratification field
    strata: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        key = str(getattr(ex, stratify_by, "unknown"))
        strata[key].append(ex)

    # Split each stratum
    calibration_examples = []
    test_examples = []

    rng = random.Random(seed)

    for stratum_key, stratum_examples in strata.items():
        # Shuffle within stratum
        shuffled = list(stratum_examples)
        rng.shuffle(shuffled)

        # Split
        n_calibration = int(len(shuffled) * calibration_fraction)
        calibration_examples.extend(shuffled[:n_calibration])
        test_examples.extend(shuffled[n_calibration:])

        log.debug(
            "stratum_split",
            stratum=stratum_key,
            total=len(shuffled),
            calibration=n_calibration,
            test=len(shuffled) - n_calibration,
        )

    log.info(
        "split_complete",
        calibration_count=len(calibration_examples),
        test_count=len(test_examples),
    )

    return calibration_examples, test_examples


def create_dev_subset(
    examples: list[Example],
    limit: int = 100,
    stratify: bool = True,
    seed: int = 42,
) -> list[Example]:
    """Create small dev subset for fast iteration.

    Per spec §9.3: Stratified sampling for balanced small sets.

    Args:
        examples: Examples to sample from
        limit: Maximum number of examples
        stratify: Whether to stratify by is_ambiguous
        seed: Random seed for reproducibility

    Returns:
        Subset of examples
    """
    if limit >= len(examples):
        return examples

    log.info(
        "creating_dev_subset",
        total=len(examples),
        limit=limit,
        stratify=stratify,
        seed=seed,
    )

    rng = random.Random(seed)

    if not stratify:
        # Simple random sample
        return rng.sample(examples, limit)

    # Stratified sampling
    strata: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        key = str(ex.is_ambiguous)
        strata[key].append(ex)

    # Sample proportionally from each stratum
    subset = []
    for stratum_examples in strata.values():
        n_from_stratum = int(limit * len(stratum_examples) / len(examples))
        n_from_stratum = min(n_from_stratum, len(stratum_examples))
        subset.extend(rng.sample(stratum_examples, n_from_stratum))

    # Fill remaining to reach limit
    if len(subset) < limit:
        remaining = [ex for ex in examples if ex not in subset]
        n_remaining = limit - len(subset)
        subset.extend(rng.sample(remaining, min(n_remaining, len(remaining))))

    log.info("dev_subset_created", final_count=len(subset))

    return subset


__all__ = [
    "create_calibration_split",
    "create_dev_subset",
]
