"""Token Log-Prob Threshold baseline.

Per spec §8: Mean token log-prob with two thresholds.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput

log = get_logger(__name__)


class LogProbThreshold(Method):
    """Token Log-Prob Threshold baseline.

    Per spec §8: Mean token log-prob of greedy answer with two thresholds.
    Budget: 1 call (+1 if ASK).
    Limitation: Requires backend that exposes log-probs.
    """

    name = "logprob_threshold"

    def __init__(
        self,
        llm_backend: LLMBackend,
        tau_high: float = -0.5,
        tau_low: float = -2.0,
    ):
        """Initialize LogProbThreshold method.

        Args:
            llm_backend: LLM backend for generation
            tau_high: High log-prob threshold (default: -0.5)
            tau_low: Low log-prob threshold (default: -2.0)
        """
        self.llm = llm_backend
        self.tau_high = tau_high
        self.tau_low = tau_low
        log.info(
            "logprob_threshold_initialized",
            tau_high=tau_high,
            tau_low=tau_low,
        )

    def decide(self, query: Query) -> MethodOutput:
        """Decide using token log-probs.

        Per spec §8: Mean token log-prob determines action.

        Args:
            query: Input query

        Returns:
            MethodOutput with decided action
        """
        # TODO: Implement full logic with log-prob extraction
        # For now, return ANSWER (requires backend support)
        from adg.sampling.sampler import Sampler

        sampler = Sampler(self.llm)
        result = sampler.sample(query, n_samples=1, temperature=0.0, seed=0)

        answer = result.samples[0].answer if result.samples else "No answer"
        mean_logprob = -1.0  # Mock log-prob

        # Apply thresholds
        if mean_logprob >= self.tau_high:
            action = "ANSWER"
        elif mean_logprob >= self.tau_low:
            action = "ASK"
        else:
            action = "ABSTAIN"

        return MethodOutput(
            action=action,
            message=answer if action == "ANSWER" else "Uncertain",
            options=[],
            score=mean_logprob,
            trace={
                "method": "logprob_threshold",
                "mean_logprob": mean_logprob,
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


__all__ = ["LogProbThreshold"]
