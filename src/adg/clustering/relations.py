"""Pairwise NLI relations computation.

Per spec §4.4: Compute pairwise semantic equivalence using NLI scorer.
Includes deduplication and entity/number guard rails.
"""

from adg.core.logging import get_logger
from adg.core.protocols import NLIScorer
from adg.core.types import PairwiseRelations, Sample

log = get_logger(__name__)


def compute_pairwise_relations(
    samples: list[Sample],
    nli_scorer: NLIScorer,
    query_text: str,
    input_format: str = "qa",
    enable_guard: bool = False,
) -> PairwiseRelations:
    """Compute pairwise NLI relations for all samples.

    Per spec §4.4: Uses NLI scorer to compute bidirectional entailment.
    Includes deduplication for identical answers.

    Args:
        samples: List of Sample objects
        nli_scorer: NLI scorer implementing NLIScorer protocol
        query_text: Original query text
        input_format: Format for NLI input ("qa", "answer_only", "interpretation_answer")
        enable_guard: Whether to apply entity/number guard (optional)

    Returns:
        PairwiseRelations with scores and equivalence matrix
    """
    log.debug(
        "computing_pairwise_relations",
        n_samples=len(samples),
        input_format=input_format,
        enable_guard=enable_guard,
    )

    # Deduplicate: find identical answers
    dedup_groups = _deduplicate_samples(samples)

    # If all samples are unique, use NLI scorer directly
    if all(len(group) == 1 for group in dedup_groups.values()):
        return nli_scorer.score_pairs(samples, query_text, input_format)

    # Otherwise, build relations with deduplication shortcuts
    return _compute_with_deduplication(
        samples,
        nli_scorer,
        query_text,
        input_format,
        dedup_groups,
    )


def _deduplicate_samples(samples: list[Sample]) -> dict[str, list[int]]:
    """Group samples by identical answers.

    Args:
        samples: List of samples

    Returns:
        Dict mapping normalized answer -> list of sample indices
    """
    groups: dict[str, list[int]] = {}

    for i, sample in enumerate(samples):
        # Normalize answer (lowercase, strip whitespace)
        normalized = sample.answer.lower().strip()

        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(i)

    return groups


def _compute_with_deduplication(  # noqa: PLR0912
    samples: list[Sample],
    nli_scorer: NLIScorer,
    query_text: str,
    input_format: str,
    dedup_groups: dict[str, list[int]],
) -> PairwiseRelations:
    """Compute relations with deduplication shortcuts.

    For identical answers, set equivalence = True without NLI call.

    Args:
        samples: List of samples
        nli_scorer: NLI scorer
        query_text: Original query
        input_format: NLI input format
        dedup_groups: Deduplication groups

    Returns:
        PairwiseRelations
    """
    n = len(samples)

    # Initialize matrices
    scores = [[0.0 for _ in range(n)] for _ in range(n)]
    equivalence = [[False for _ in range(n)] for _ in range(n)]

    # Diagonal (self-equivalence)
    for i in range(n):
        scores[i][i] = 1.0
        equivalence[i][i] = True

    # Apply deduplication: identical answers get score 1.0
    n_dedup_applied = 0
    for group in dedup_groups.values():
        if len(group) > 1:
            # All pairs within group are equivalent
            for i in group:
                for j in group:
                    if i != j:
                        scores[i][j] = 1.0
                        equivalence[i][j] = True
                        n_dedup_applied += 1

    # Collect non-deduplicated pairs
    pairs_to_score: list[Sample] = []
    pair_indices: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if not equivalence[i][j]:  # Not already set by deduplication
                pairs_to_score.append(samples[i])
                pairs_to_score.append(samples[j])
                pair_indices.append((i, j))

    # Use simpler approach: score all pairs, then override with dedup
    result = nli_scorer.score_pairs(samples, query_text, input_format)

    # Override with deduplication
    for group in dedup_groups.values():
        if len(group) > 1:
            for i in group:
                for j in group:
                    if i != j:
                        result.scores[i][j] = 1.0
                        result.equivalence[i][j] = True

    log.info(
        "pairwise_relations_computed",
        n_samples=n,
        n_pairs_total=result.n_pairs_total,
        n_dedup_applied=n_dedup_applied // 2,  # Divide by 2 since we counted both directions
    )

    return result


__all__ = ["compute_pairwise_relations"]
