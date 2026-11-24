"""Stated Confidence baseline.

Per spec §8: Verbalized confidence 0-100 with two thresholds.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput

log = get_logger(__name__)


class StatedConfidence(Method):
    """Stated Confidence baseline.

    Per spec §8: Prompts for {"answer": ..., "confidence": 0-100}.
    Two thresholds: conf ≥ τ_hi → ANSWER; τ_lo ≤ conf < τ_hi → ASK; conf < τ_lo → ABSTAIN.
    Budget: 1 call (+1 for generic clarifier if ASK).
    """

    name = "stated_confidence"

    def __init__(
        self,
        llm_backend: LLMBackend,
        tau_high: float = 70.0,
        tau_low: float = 30.0,
    ):
        """Initialize StatedConfidence method.

        Args:
            llm_backend: LLM backend for generation
            tau_high: High confidence threshold (default: 70)
            tau_low: Low confidence threshold (default: 30)
        """
        self.llm = llm_backend
        self.tau_high = tau_high
        self.tau_low = tau_low
        log.info(
            "stated_confidence_initialized",
            tau_high=tau_high,
            tau_low=tau_low,
        )

    def decide(self, query: Query) -> MethodOutput:
        """Decide using stated confidence.

        Per spec §8: Parse confidence from model output.

        Args:
            query: Input query

        Returns:
            MethodOutput with decided action
        """
        # TODO: Implement full logic with confidence parsing
        # For now, return ANSWER with high confidence
        from adg.sampling.sampler import Sampler

        sampler = Sampler(self.llm)
        result = sampler.sample(query, n_samples=1, temperature=0.0, seed=0)

        answer = result.samples[0].answer if result.samples else "No answer"
        confidence = 80.0  # Mock confidence

        # Apply thresholds
        if confidence >= self.tau_high:
            action = "ANSWER"
        elif confidence >= self.tau_low:
            action = "ASK"
        else:
            action = "ABSTAIN"

        return MethodOutput(
            action=action,
            message=answer if action == "ANSWER" else "Uncertain",
            options=[],
            score=confidence,
            trace={
                "method": "stated_confidence",
                "confidence": confidence,
                "tau_high": self.tau_high,
                "tau_low": self.tau_low,
            },
            cost=result.cost,
        )

    def answer_after_clarification(
        self,
        query: Query,
        clarification_turns: list[Turn],
    ) -> tuple[str, CostRecord]:
        """Answer after clarification.

        Args:
            query: Original query
            clarification_turns: Clarification history

        Returns:
            Tuple of (answer, cost)
        """
        result = self.decide(query)
        return result.message, result.cost

    def budget(self) -> BudgetSpec:
        """Return budget specification.

        Returns:
            Budget: 1 call (+1 if ASK)
        """
        return BudgetSpec(
            llm_calls=1,
            samples_per_call=1,
            nli_pairs=0,
            description="1 call (+1 if ASK)",
        )


__all__ = ["StatedConfidence"]
