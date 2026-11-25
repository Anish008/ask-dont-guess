"""Ask-If-Unclear prompt baseline.

Per spec §8: Prompt-based method using JSON output.
"""

import json
import re

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.llm.parsing import REFUSAL_PATTERNS
from adg.methods.base import BudgetSpec, Method, MethodOutput
from adg.sampling.sampler import Sampler

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
        # Augment query with instruction
        augmented_query = Query(
            query_id=query.query_id,
            text=f"""If this question is ambiguous, ask ONE clarifying question. If you don't know, say "I don't know". Otherwise, answer directly.

Question: {query.text}

Respond ONLY with valid JSON:
{{"action": "answer", "content": "..."}} OR {{"action": "ask", "content": "..."}} OR {{"action": "abstain", "content": "..."}}""",
        )

        # Sample once, greedy
        sampler = Sampler(self.llm)
        result = sampler.sample(
            augmented_query, n_samples=1, temperature=0.0, top_p=1.0, max_new_tokens=200, seed=0
        )

        response = result.samples[0].raw_text if result.samples else ""

        # JSON parse with fallback
        parse_failed = False
        try:
            # Strip markdown if present
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            action = data["action"].lower()
            content = data["content"]

            if action not in ["answer", "ask", "abstain"]:
                raise ValueError(f"bad action: {action}")

            action = action.upper()

        except (json.JSONDecodeError, ValueError, KeyError):
            parse_failed = True
            log.warning("json_parse_failed", response=response[:100])

            # Fallback: ? → ASK, refusal → ABSTAIN, else ANSWER
            if response.strip().endswith("?"):
                action, content = "ASK", response.strip()
            elif any(re.search(p, response.lower()) for p in REFUSAL_PATTERNS):
                action, content = "ABSTAIN", "I don't know"
            else:
                action, content = "ANSWER", response.strip()

        return MethodOutput(
            action=action,
            message=content,
            options=[],
            score=None,
            trace={"method": "ask_if_unclear", "parse_failed": parse_failed},
            cost=CostRecord(),  # TODO: track actual cost
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
