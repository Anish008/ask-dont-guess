"""Seeded sampling orchestration.

Per spec §4.3: manages batched LLM sampling with retry logic and seed derivation.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.seeding import derive_seed
from adg.core.types import Query, SamplingResult

log = get_logger(__name__)


class Sampler:
    """Orchestrates seeded sampling from LLM backends.

    Handles:
    - Seed derivation per query
    - Batch size management
    - Retry logic (delegated to backend)
    """

    def __init__(self, backend: LLMBackend) -> None:
        """Initialize sampler.

        Args:
            backend: LLM backend implementing LLMBackend protocol
        """
        self.backend = backend
        log.debug("sampler_initialized", backend=type(backend).__name__)

    def sample(
        self,
        query: Query,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None = None,
        global_seed: int = 0,
    ) -> SamplingResult:
        """Sample N answers for a query.

        Args:
            query: Input query
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate per sample
            seed: Base seed (if None, derive from query_id)
            global_seed: Global seed for derivation (default 0)

        Returns:
            SamplingResult with N samples
        """
        # Derive seed from query if not provided
        if seed is None:
            seed = derive_seed(global_seed, query.query_id)

        log.debug(
            "sampling_start",
            query_id=query.query_id,
            n_samples=n_samples,
            seed=seed,
        )

        # Delegate to backend
        result = self.backend.sample(
            query=query,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        log.info(
            "sampling_complete",
            query_id=query.query_id,
            n_samples=result.n_samples,
            n_valid=result.n_valid,
            n_refusals=result.n_refusals,
        )

        return result

    def close(self) -> None:
        """Clean up backend resources."""
        self.backend.close()


__all__ = ["Sampler"]
