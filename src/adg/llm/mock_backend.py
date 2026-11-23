"""Mock LLM backend for testing.

Per spec §4.3: deterministic test backend with no model loading.
"""

import hashlib

from adg.core.logging import get_logger
from adg.core.types import Query, Sample, SamplingResult

log = get_logger(__name__)


class MockLLMBackend:
    """Deterministic mock LLM backend for testing.

    Generates predictable outputs based on query and seed.
    No model loading required.
    """

    def __init__(self) -> None:
        """Initialize mock backend."""
        log.info("mock_llm_initialized")

    def sample(
        self,
        query: Query,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> SamplingResult:
        """Generate mock samples deterministically.

        Args:
            query: Input query
            n_samples: Number of samples to generate
            temperature: Ignored (mock is deterministic)
            top_p: Ignored (mock is deterministic)
            max_new_tokens: Ignored (mock generates fixed-length text)
            seed: Random seed for determinism

        Returns:
            SamplingResult with n_samples mock samples
        """
        log.debug(
            "mock_sampling",
            query_id=query.query_id,
            n_samples=n_samples,
            seed=seed,
        )

        samples = []
        for i in range(n_samples):
            # Deterministic hash-based generation
            sample_seed = seed + i
            interpretation, answer, confidence = self._generate_sample(query.text, sample_seed)

            # Create sample
            sample = Sample(
                interpretation=interpretation,
                answer=answer,
                confidence=confidence,  # type: ignore[arg-type]
                sample_id=i,
                is_valid=True,
                is_refusal=False,
                raw_text=f"Interpretation: {interpretation}\nAnswer: {answer}\nConfidence: {confidence}",
            )
            samples.append(sample)

        # Count valid and refusal samples
        n_valid = sum(1 for s in samples if s.is_valid and not s.is_refusal)
        n_refusals = sum(1 for s in samples if s.is_refusal)

        return SamplingResult(
            samples=samples,
            n_samples=n_samples,
            n_valid=n_valid,
            n_refusals=n_refusals,
            seed=seed,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 96,
        seed: int | None = None,
    ) -> str:
        """Generate a single mock response.

        Args:
            prompt: Input prompt
            temperature: Ignored (mock is deterministic)
            max_new_tokens: Ignored (mock generates fixed-length text)
            seed: Random seed for determinism

        Returns:
            Generated text
        """
        # Use seed or hash of prompt for determinism
        effective_seed = seed if seed is not None else self._hash_text(prompt)

        # Simple deterministic generation
        hash_val = self._hash_text(prompt + str(effective_seed))
        response_id = int(hash_val[:8], 16) % 3

        responses = [
            "This is a mock response A.",
            "This is a mock response B.",
            "This is a mock response C.",
        ]

        return responses[response_id]

    def close(self) -> None:
        """Clean up resources (no-op for mock)."""
        log.debug("mock_llm_closed")

    def _generate_sample(self, query_text: str, sample_seed: int) -> tuple[str, str, str]:
        """Generate a single mock sample deterministically.

        Args:
            query_text: Query text
            sample_seed: Seed for this sample

        Returns:
            (interpretation, answer, confidence) tuple
        """
        # Hash query + seed for determinism
        hash_val = self._hash_text(query_text + str(sample_seed))
        variant = int(hash_val[:8], 16) % 5

        # Generate diverse mock responses
        if variant == 0:
            interpretation = f"Asking about {query_text[:20]}..."
            answer = f"The answer is {hash_val[:8]}"
            confidence = "high"
        elif variant == 1:
            interpretation = f"The question refers to {query_text[:20]}..."
            answer = f"Response variant {hash_val[:8]}"
            confidence = "medium"
        elif variant == 2:  # noqa: PLR2004
            interpretation = f"Interpreting as {query_text[:20]}..."
            answer = f"Answer option {hash_val[:8]}"
            confidence = "low"
        elif variant == 3:  # noqa: PLR2004
            interpretation = f"Understanding the query as {query_text[:20]}..."
            answer = f"Result {hash_val[:8]}"
            confidence = "high"
        else:
            interpretation = f"Query meaning: {query_text[:20]}..."
            answer = f"Output {hash_val[:8]}"
            confidence = "medium"

        return interpretation, answer, confidence

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text to hex string.

        Args:
            text: Input text

        Returns:
            16-character hex hash
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]


__all__ = ["MockLLMBackend"]
