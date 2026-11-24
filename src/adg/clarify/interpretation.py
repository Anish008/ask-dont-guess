"""Reply resolution for clarifying questions.

Per spec §4.10: Map user replies back to selected interpretations.
"""

from adg.core.logging import get_logger
from adg.core.types import ClarifyingQuestion, ResolvedInterpretation

log = get_logger(__name__)


class ReplyResolver:
    """Resolves user replies to clarifying questions."""

    def __init__(self, fuzzy_match_threshold: float = 0.6):
        """Initialize reply resolver.

        Args:
            fuzzy_match_threshold: Threshold for fuzzy text matching (default 0.6)
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold
        log.info("reply_resolver_initialized", threshold=fuzzy_match_threshold)

    def resolve(
        self,
        reply: str,
        question: ClarifyingQuestion,
    ) -> ResolvedInterpretation:
        """Resolve user reply to selected interpretation.

        Args:
            reply: User's reply text
            question: The clarifying question that was asked

        Returns:
            ResolvedInterpretation with selected interpretation and confidence
        """
        log.debug("resolving_reply", reply=reply[:50], n_options=len(question.options))

        # Try numeric resolution first
        selected_idx, confidence = self._try_numeric_resolution(reply, question.options)

        if selected_idx is not None:
            log.info(
                "reply_resolved",
                method="exact_match",
                selected_idx=selected_idx,
                confidence=confidence,
            )
            return ResolvedInterpretation(
                option_index=selected_idx,
                confidence=confidence,
                method="exact_match",
            )

        # Try text matching
        selected_idx, confidence, is_exact = self._try_text_matching(reply, question.options)

        if selected_idx is not None:
            method = "exact_match" if is_exact else "fuzzy"
            log.info(
                "reply_resolved",
                method=method,
                selected_idx=selected_idx,
                confidence=confidence,
            )
            return ResolvedInterpretation(
                option_index=selected_idx,
                confidence=confidence,
                method=method,
            )

        # Fallback: unclear resolution
        log.warning("reply_resolution_unclear", reply=reply[:50])
        return ResolvedInterpretation(
            option_index=None,  # No match
            confidence=0.0,
            method="none",
        )

    def _try_numeric_resolution(
        self,
        reply: str,
        options: list[str],
    ) -> tuple[int | None, float]:
        """Try to resolve reply as numeric choice.

        Args:
            reply: User reply
            options: Available options

        Returns:
            (selected_index, confidence) or (None, 0.0) if not numeric
        """
        reply_stripped = reply.strip()

        # Try parsing as integer
        try:
            choice = int(reply_stripped)
            # 1-indexed -> 0-indexed
            idx = choice - 1

            if 0 <= idx < len(options):
                return idx, 1.0  # High confidence for valid numeric choice
            else:
                log.debug("numeric_choice_out_of_range", choice=choice, n_options=len(options))
                return None, 0.0

        except ValueError:
            # Not a numeric choice
            return None, 0.0

    def _try_text_matching(
        self,
        reply: str,
        options: list[str],
    ) -> tuple[int | None, float, bool]:
        """Try to match reply text to option.

        Uses fuzzy string matching to find best match.

        Args:
            reply: User reply
            options: Available options

        Returns:
            (selected_index, confidence, is_exact) or (None, 0.0, False) if no match
        """
        reply_normalized = reply.lower().strip()

        # First try exact match
        for i, option in enumerate(options):
            if reply_normalized == option.lower().strip():
                return i, 1.0, True

        # Try substring match
        for i, option in enumerate(options):
            option_normalized = option.lower().strip()

            # Check if reply is substring of option or vice versa
            if reply_normalized in option_normalized:
                confidence = len(reply_normalized) / len(option_normalized)
                if confidence >= self.fuzzy_match_threshold:
                    return i, confidence, False

            if option_normalized in reply_normalized:
                confidence = len(option_normalized) / len(reply_normalized)
                if confidence >= self.fuzzy_match_threshold:
                    return i, confidence, False

        # Try word overlap matching
        reply_words = set(reply_normalized.split())
        best_idx = None
        best_score = 0.0

        for i, option in enumerate(options):
            option_words = set(option.lower().strip().split())

            if not reply_words or not option_words:
                continue

            # Jaccard similarity
            intersection = reply_words & option_words
            union = reply_words | option_words
            score = len(intersection) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.fuzzy_match_threshold:
            return best_idx, best_score, False

        # No match found
        return None, 0.0, False


__all__ = ["ReplyResolver"]
