"""Mock NLI scorer for testing.

Per spec §4.4: rule-based test scorer with no model loading.
Uses simple heuristics for semantic equivalence.
"""

from adg.core.logging import get_logger
from adg.core.types import PairwiseRelations, Sample

log = get_logger(__name__)


class MockNLIScorer:
    """Rule-based mock NLI scorer for testing.

    Uses simple string similarity heuristics.
    No model loading required.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize mock NLI scorer.

        Args:
            threshold: Similarity threshold for equivalence (default 0.5)
        """
        self.threshold = threshold
        log.info("mock_nli_initialized", threshold=threshold)

    def score_pairs(
        self,
        samples: list[Sample],
        query_text: str,
        input_format: str = "qa",
    ) -> PairwiseRelations:
        """Compute NLI scores for all pairs of samples.

        Args:
            samples: List of Sample objects to compare
            query_text: Original query (unused in mock)
            input_format: "qa", "answer_only", or "interpretation_answer" (unused in mock)

        Returns:
            PairwiseRelations with symmetric scores matrix
        """
        n = len(samples)
        log.debug("mock_nli_scoring", n_samples=n, input_format=input_format)

        # Initialize score matrix (symmetric)
        scores = [[0.0 for _ in range(n)] for _ in range(n)]
        equivalence = [[False for _ in range(n)] for _ in range(n)]

        # Diagonal is always 1.0 (self-equivalence)
        for i in range(n):
            scores[i][i] = 1.0
            equivalence[i][i] = True

        # Compute pairwise scores
        n_pairs_computed = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Use answer similarity for mock scoring
                score = self._compute_similarity(samples[i].answer, samples[j].answer)
                scores[i][j] = score
                scores[j][i] = score  # Symmetric

                # Equivalence based on threshold
                is_equiv = score >= self.threshold
                equivalence[i][j] = is_equiv
                equivalence[j][i] = is_equiv

                n_pairs_computed += 1

        n_pairs_total = n * (n - 1) // 2

        log.info(
            "mock_nli_scoring_complete",
            n_samples=n,
            n_pairs_computed=n_pairs_computed,
            n_pairs_total=n_pairs_total,
        )

        return PairwiseRelations(
            scores=scores,
            equivalence=equivalence,
            n_pairs_computed=n_pairs_computed,
            n_pairs_total=n_pairs_total,
            n_guard_applied=0,
            mode="soft",
        )

    def score_single_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, float, float]:
        """Score a single premise-hypothesis pair.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            (entailment_prob, neutral_prob, contradiction_prob)
        """
        # Compute similarity
        similarity = self._compute_similarity(premise, hypothesis)

        # Map similarity to NLI probabilities
        # High similarity → high entailment
        # Low similarity → high contradiction
        entailment_prob = similarity
        contradiction_prob = 1.0 - similarity
        neutral_prob = 0.0  # Mock doesn't use neutral

        return (entailment_prob, neutral_prob, contradiction_prob)

    def close(self) -> None:
        """Clean up resources (no-op for mock)."""
        log.debug("mock_nli_closed")

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score in [0, 1]
        """
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Exact match
        if t1 == t2:
            return 1.0

        # Empty check
        if not t1 or not t2:
            return 0.0

        # Jaccard similarity on words
        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


__all__ = ["MockNLIScorer"]
