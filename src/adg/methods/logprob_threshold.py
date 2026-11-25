"""Token Log-Prob Threshold baseline.

Per spec §8: Mean token log-prob with two thresholds.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.llm.hf_backend import HFLLMBackend
from adg.llm.prompts import format_structured_prompt
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
        # Check if backend supports log-probs
        if not isinstance(self.llm, HFLLMBackend):
            # Mark as unsupported
            log.warning("logprob_unsupported", backend=type(self.llm).__name__)
            return MethodOutput(
                action="ABSTAIN",
                message="Log-prob threshold unsupported for this backend",
                options=[],
                score=None,
                trace={"method": "logprob_threshold", "supported": False},
                cost=CostRecord(),
            )

        # Generate with log-probs (greedy)
        prompt = format_structured_prompt(query.text)
        answer, mean_logprob = self.llm.generate_with_logprobs(
            prompt, temperature=0.0, max_new_tokens=96, seed=0
        )

        # Apply two-threshold logic
        if mean_logprob >= self.tau_high:
            action = "ANSWER"
            message = answer
        elif mean_logprob >= self.tau_low:
            action = "ASK"
            message = "Could you clarify?"
        else:
            action = "ABSTAIN"
            message = "I'm not confident in my answer."

        return MethodOutput(
            action=action,
            message=message,
            options=[],
            score=mean_logprob,
            trace={
                "method": "logprob_threshold",
                "mean_logprob": mean_logprob,
                "tau_high": self.tau_high,
                "tau_low": self.tau_low,
            },
            cost=CostRecord(input_tokens=0, output_tokens=0),  # TODO: proper cost tracking
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
