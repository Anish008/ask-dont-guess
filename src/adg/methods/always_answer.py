"""Always Answer baseline.

Per spec §8: Always ANSWER with the greedy answer.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput

log = get_logger(__name__)


class AlwaysAnswer(Method):
    """Always Answer baseline method.

    Per spec §8: Always returns ANSWER with greedy generation.
    Budget: 1 greedy call per query.
    """

    name = "always_answer"

    def __init__(self, llm_backend: LLMBackend):
        """Initialize AlwaysAnswer method.

        Args:
            llm_backend: LLM backend for generation
        """
        self.llm = llm_backend
        log.info("always_answer_initialized")

    def decide(self, query: Query) -> MethodOutput:
        """Always ANSWER with greedy generation.

        Per spec §8: No decision logic, just generate.

        Args:
            query: Input query

        Returns:
            MethodOutput with ANSWER action
        """
        # Generate greedy answer (temperature=0)
        from adg.sampling.sampler import Sampler

        sampler = Sampler(self.llm)
        result = sampler.sample(
            query,
            n_samples=1,
            temperature=0.0,
            top_p=0.95,
            max_new_tokens=96,
            seed=0,
        )

        answer = result.samples[0].answer if result.samples else "No answer generated"

        return MethodOutput(
            action="ANSWER",
            message=answer,
            options=[],
            score=None,  # No uncertainty score
            trace={
                "method": "always_answer",
                "greedy": True,
            },
            cost=result.cost,
        )

    def answer_after_clarification(
        self,
        query: Query,
        clarification_turns: list[Turn],
    ) -> tuple[str, CostRecord]:
        """Answer after clarification (not applicable for this method).

        Args:
            query: Original query
            clarification_turns: Clarification history

        Returns:
            Tuple of (answer, cost)
        """
        # This method never asks, so this shouldn't be called
        # But if it is, just answer again
        result = self.decide(query)
        return result.message, result.cost

    def budget(self) -> BudgetSpec:
        """Return budget specification.

        Returns:
            Budget: 1 greedy call
        """
        return BudgetSpec(
            llm_calls=1,
            samples_per_call=1,
            nli_pairs=0,
            description="1 greedy call",
        )


__all__ = ["AlwaysAnswer"]
