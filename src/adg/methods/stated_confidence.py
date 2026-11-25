"""Stated Confidence baseline.

Per spec §8: Verbalized confidence 0-100 with two thresholds.
"""

import json
import re

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput
from adg.sampling.sampler import Sampler

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
        # Augment query with confidence instruction
        augmented_query = Query(
            query_id=query.query_id,
            text=f"""Answer this question and rate your confidence from 0 (no confidence) to 100 (completely certain).

Question: {query.text}

Respond ONLY with valid JSON:
{{"answer": "your answer", "confidence": 75}}""",
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
            # Strip markdown
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            answer = data["answer"]
            confidence = float(data["confidence"])

            # Clamp confidence to [0, 100]
            confidence = max(0.0, min(100.0, confidence))

        except (json.JSONDecodeError, ValueError, KeyError):
            parse_failed = True
            log.warning("confidence_parse_failed", response=response[:100])

            # Fallback: extract answer, default confidence=50
            answer = response.strip()
            # Try regex for confidence number
            conf_match = re.search(r"\b(\d+)(?:\s*%|\s+out of 100)\b", response)
            confidence = float(conf_match.group(1)) if conf_match else 50.0

        # Apply two-threshold logic
        if confidence >= self.tau_high:
            action = "ANSWER"
            message = answer
        elif confidence >= self.tau_low:
            action = "ASK"
            message = "Could you clarify?"
        else:
            action = "ABSTAIN"
            message = "I'm not confident I can answer this."

        return MethodOutput(
            action=action,
            message=message,
            options=[],
            score=confidence,
            trace={
                "method": "stated_confidence",
                "confidence": confidence,
                "parse_failed": parse_failed,
                "tau_high": self.tau_high,
                "tau_low": self.tau_low,
            },
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
            Budget: 1 call (+1 if ASK)
        """
        return BudgetSpec(
            llm_calls=1,
            samples_per_call=1,
            nli_pairs=0,
            description="1 call (+1 if ASK)",
        )


__all__ = ["StatedConfidence"]
