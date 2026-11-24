"""ADG+IC method (with interpretation channel).

Per spec §8: Full feature set + interpretation channel enabled.
"""

from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend, NLIScorer
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput

log = get_logger(__name__)


class ADGIC(Method):
    """ADG+IC method (with interpretation channel).

    Per spec §8: Full ADG pipeline with interpretation channel enabled.
    Budget: N structured samples + NLI pairs (answers + interpretations) (+1 if LLM clarifier).
    """

    name = "adg_ic"

    def __init__(
        self,
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        n_samples: int = 8,
    ):
        """Initialize ADG+IC method.

        Args:
            llm_backend: LLM backend for sampling
            nli_scorer: NLI scorer for clustering
            n_samples: Number of samples (default: 8)
        """
        self.manager = ConversationManager(
            llm_backend=llm_backend,
            nli_scorer=nli_scorer,
            n_samples=n_samples,
            interpretation_channel_enabled=True,  # IC variant
        )
        self.n_samples = n_samples
        log.info("adg_ic_initialized", n_samples=n_samples)

    def decide(self, query: Query) -> MethodOutput:
        """Decide using ADG+IC pipeline.

        Per spec §8: Full ADG pipeline with interpretation channel.

        Args:
            query: Input query

        Returns:
            MethodOutput with decided action
        """
        # Create session and process query
        session = self.manager.start_session(f"adg_ic_{query.query_id}")
        session, response, decision = self.manager.process_query(session, query)

        # Extract options if ASK
        options = []
        if decision.action == "ASK" and session.pending_clarification:
            options = session.pending_clarification.options

        # TODO: Extract actual cost from pipeline
        from adg.core.types import CostRecord

        cost = CostRecord(
            llm_calls=1,
            samples=self.n_samples,
            prompt_tokens=0,
            completion_tokens=0,
            nli_forward_passes=0,
        )

        return MethodOutput(
            action=decision.action,
            message=response,
            options=options,
            score=decision.confidence_scores.get("ask_score", decision.confidence),
            trace={
                "method": "adg_ic",
                "rule_name": decision.rule_name,
                "confidence": decision.confidence,
                "confidence_scores": decision.confidence_scores,
                "interpretation_channel": True,
            },
            cost=cost,
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
        # TODO: Implement resolution using manager
        from adg.core.types import CostRecord

        cost = CostRecord(
            llm_calls=0,
            samples=0,
            prompt_tokens=0,
            completion_tokens=0,
            nli_forward_passes=0,
        )
        return "Resolved answer", cost

    def budget(self) -> BudgetSpec:
        """Return budget specification.

        Returns:
            Budget: N samples + NLI pairs (answers + interpretations)
        """
        # With interpretation channel, NLI on both answers and interpretations
        nli_pairs = self.n_samples * (self.n_samples - 1)  # 2x for answers + interpretations

        return BudgetSpec(
            llm_calls=1,
            samples_per_call=self.n_samples,
            nli_pairs=nli_pairs,
            description=f"{self.n_samples} structured samples + {nli_pairs} NLI pairs",
        )


__all__ = ["ADGIC"]
