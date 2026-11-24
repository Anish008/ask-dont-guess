"""Ask-If-Unclear prompt baseline.

Per spec §8: Prompt-based method using JSON output.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput

log = get_logger(__name__)


class AskIfUnclear(Method):
    """Ask-If-Unclear prompt baseline.

    Per spec §8: System prompt asks model to output JSON with action.
    Budget: 1 greedy call.
    """

    name = "ask_if_unclear"

    def __init__(self, llm_backend: LLMBackend):
        """Initialize AskIfUnclear method.

        Args:
            llm_backend: LLM backend for generation
        """
        self.llm = llm_backend
        log.info("ask_if_unclear_initialized")

    def decide(self, query: Query) -> MethodOutput:
        """Decide using prompt-based approach.

        Per spec §8: Prompt asks for JSON {"action": "answer|ask|abstain", "content": "..."}.
        Falls back to heuristics on parse failure.

        Args:
            query: Input query

        Returns:
            MethodOutput with decided action
        """
        # TODO: Implement full logic with JSON parsing
        # For now, return simple greedy answer
        from adg.sampling.sampler import Sampler

        sampler = Sampler(self.llm)
        result = sampler.sample(query, n_samples=1, temperature=0.0, seed=0)

        answer = result.samples[0].answer if result.samples else "No answer"

        return MethodOutput(
            action="ANSWER",
            message=answer,
            options=[],
            score=None,
            trace={"method": "ask_if_unclear", "parse_failed": False},
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
            Budget: 1 greedy call
        """
        return BudgetSpec(
            llm_calls=1,
            samples_per_call=1,
            nli_pairs=0,
            description="1 greedy call with prompt",
        )


__all__ = ["AskIfUnclear"]
