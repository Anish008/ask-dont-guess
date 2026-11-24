"""Base protocol and types for methods.

Per spec §6.11 and §8: Common interface for fair comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from adg.core.types import CostRecord, Query, Turn


class BudgetSpec(BaseModel):
    """Budget declaration for a method.

    Per spec §8: Declared samples/calls per query.

    Args:
        llm_calls: Number of LLM calls per query
        samples_per_call: Samples per LLM call (for sampling methods)
        nli_pairs: Approximate NLI forward passes
        description: Human-readable budget description
    """

    llm_calls: int
    samples_per_call: int = 1
    nli_pairs: int = 0
    description: str


class MethodOutput(BaseModel):
    """Output from a method's decide() call.

    Per spec §6.11: Common output format for all methods.

    Args:
        action: ANSWER/ASK/ABSTAIN decision
        message: Response text (answer or clarifying question)
        options: Options for ASK action (empty for ANSWER/ABSTAIN)
        score: Continuous ask/uncertainty score for AUROC (optional)
        trace: Method-specific decision trace
        cost: Actual cost incurred
    """

    action: Literal["ANSWER", "ASK", "ABSTAIN"]
    message: str
    options: list[str] = Field(default_factory=list)
    score: float | None = None  # For AUROC
    trace: dict[str, Any] = Field(default_factory=dict)
    cost: CostRecord


class Method(ABC):
    """Base class for all methods (baselines + ADG).

    Per spec §6.11: Common interface for fair comparison.
    All methods use the same LLM instance and evaluation code.
    """

    name: str

    @abstractmethod
    def decide(self, query: Query) -> MethodOutput:
        """Make ANSWER/ASK/ABSTAIN decision for a query.

        Args:
            query: Input query

        Returns:
            MethodOutput with action, message, and trace
        """
        ...

    @abstractmethod
    def answer_after_clarification(
        self,
        query: Query,
        clarification_turns: list[Turn],
    ) -> tuple[str, CostRecord]:
        """Generate final answer after clarification turn(s).

        Args:
            query: Original query
            clarification_turns: History of clarification exchanges

        Returns:
            Tuple of (answer text, cost record)
        """
        ...

    @abstractmethod
    def budget(self) -> BudgetSpec:
        """Declare expected budget per query.

        Returns:
            Budget specification
        """
        ...


__all__ = [
    "BudgetSpec",
    "Method",
    "MethodOutput",
]
