"""Protocol definitions for extensible backends and methods.

Per spec: LLMBackend, NLIScorer, Method, SimulatedUser protocols for pluggable implementations.
"""

from typing import Protocol, runtime_checkable

from adg.config.schema import AppConfig
from adg.core.types import (
    ADGResult,
    Example,
    PairwiseRelations,
    Query,
    Sample,
    SamplingResult,
    SimUserInput,
    SimUserReply,
)

# ==============================================================================
# LLM Backend Protocol
# ==============================================================================


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends (HF, vLLM, OpenAI, Mock).

    Per spec §4.3: backends must implement sampling and generation.
    """

    def sample(
        self,
        query: Query,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> SamplingResult:
        """Sample N answers from the LLM.

        Args:
            query: Input query
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility

        Returns:
            SamplingResult with N samples (some may be invalid/refusals)

        Raises:
            GenerationError: If generation fails after retries
        """
        ...

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 96,
        seed: int | None = None,
    ) -> str:
        """Generate a single response (for clarification, disambiguation, etc.).

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 for greedy)
            max_new_tokens: Maximum tokens to generate
            seed: Optional random seed

        Returns:
            Generated text

        Raises:
            GenerationError: If generation fails
        """
        ...

    def close(self) -> None:
        """Clean up resources (models, connections, etc.)."""
        ...


# ==============================================================================
# NLI Scorer Protocol
# ==============================================================================


@runtime_checkable
class NLIScorer(Protocol):
    """Protocol for NLI scoring backends (HF, Mock).

    Per spec §4.4: score bidirectional entailment for semantic equivalence.
    """

    def score_pairs(
        self,
        samples: list[Sample],
        query_text: str,
        input_format: str = "qa",
    ) -> PairwiseRelations:
        """Compute NLI scores for all pairs of samples.

        Args:
            samples: List of Sample objects to compare
            query_text: Original query (for contextualization)
            input_format: "qa" (default), "answer_only", or "interpretation_answer"

        Returns:
            PairwiseRelations with symmetric scores matrix S[i][j]

        Raises:
            NLIError: If scoring fails
        """
        ...

    def score_single_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, float, float]:
        """Score a single premise-hypothesis pair.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            (entailment_prob, neutral_prob, contradiction_prob)

        Raises:
            NLIError: If scoring fails
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...


# ==============================================================================
# Method Protocol
# ==============================================================================


@runtime_checkable
class Method(Protocol):
    """Protocol for decision methods (ADG, baselines).

    Per spec §12: methods implement run() and optional calibrate().
    """

    name: str  # Method identifier (e.g., "adg_ic", "semantic_entropy")

    def run(
        self,
        query: Query,
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        config: AppConfig,
    ) -> ADGResult:
        """Run the method on a single query.

        Args:
            query: Input query
            llm_backend: LLM backend for sampling
            nli_scorer: NLI scorer for semantic equivalence
            config: Application configuration

        Returns:
            ADGResult with action, decision trace, and intermediate outputs

        Raises:
            ADGError: On method-specific failures
        """
        ...

    def calibrate(
        self,
        examples: list[Example],
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        config: AppConfig,
        objective: str = "macro_f1",
    ) -> dict[str, float]:
        """Calibrate thresholds on a development set.

        Args:
            examples: Calibration examples
            llm_backend: LLM backend
            nli_scorer: NLI scorer
            config: Configuration
            objective: Metric to optimize ("macro_f1", "ask_f1", etc.)

        Returns:
            Dict of calibrated thresholds

        Note:
            Not all methods require calibration. Default: return empty dict.
        """
        return {}


# ==============================================================================
# Simulated User Protocol
# ==============================================================================


@runtime_checkable
class SimulatedUser(Protocol):
    """Protocol for simulated users (rule-based, LLM-based).

    Per spec §11: simulates user response to clarifying questions.
    """

    def reply(self, input_data: SimUserInput) -> SimUserReply:
        """Generate user reply to a clarifying question.

        Args:
            input_data: Context including intent, options, clarifying question

        Returns:
            SimUserReply with text, chosen option, and behavior

        Raises:
            ADGError: If reply generation fails
        """
        ...


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "LLMBackend",
    "Method",
    "NLIScorer",
    "SimulatedUser",
]
