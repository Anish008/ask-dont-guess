"""Semantic Entropy baseline.

Per spec §8: Discrete SE over meaning clusters (Kuhn et al.).
Faithful replication using strict bidirectional entailment.
"""

import math

from adg.clustering.algorithms import cluster_samples
from adg.clustering.relations import compute_pairwise_relations
from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend, NLIScorer
from adg.core.types import CostRecord, Query, Turn
from adg.methods.base import BudgetSpec, Method, MethodOutput
from adg.sampling.sampler import Sampler

log = get_logger(__name__)


class SemanticEntropy(Method):
    """Semantic Entropy baseline (Kuhn et al.).

    Per spec §8: Plain sampling, N samples, strict bidirectional entailment,
    greedy clustering, entropy H with two thresholds.
    Budget: N samples + NLI pairs (+1 if ASK).
    """

    name = "semantic_entropy"

    def __init__(
        self,
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        n_samples: int = 8,
        tau1: float = 0.5,
        tau2: float = 1.5,
    ):
        """Initialize SemanticEntropy method.

        Args:
            llm_backend: LLM backend for sampling
            nli_scorer: NLI scorer for clustering
            n_samples: Number of samples (default: 8)
            tau1: Low entropy threshold (default: 0.5)
            tau2: High entropy threshold (default: 1.5)
        """
        self.llm = llm_backend
        self.nli = nli_scorer
        self.n_samples = n_samples
        self.tau1 = tau1
        self.tau2 = tau2
        log.info(
            "semantic_entropy_initialized",
            n_samples=n_samples,
            tau1=tau1,
            tau2=tau2,
        )

    def decide(self, query: Query) -> MethodOutput:
        """Decide using semantic entropy.

        Per spec §8: H ≤ τ1 → ANSWER; τ1 < H ≤ τ2 → ASK; H > τ2 → ABSTAIN.

        Args:
            query: Input query

        Returns:
            MethodOutput with decided action
        """
        # Plain sampling (no structured format)
        plain_query = Query(query_id=query.query_id, text=f"Question: {query.text}\n\nAnswer:")

        sampler = Sampler(self.llm)
        result = sampler.sample(
            plain_query,
            n_samples=self.n_samples,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=96,
            seed=0,
        )

        # Filter to valid non-refusal samples
        valid_samples = [s for s in result.samples if s.is_valid and not s.is_refusal]

        MIN_SAMPLES_FOR_CLUSTERING = 2
        if len(valid_samples) < MIN_SAMPLES_FOR_CLUSTERING:
            # Not enough samples → ABSTAIN
            return MethodOutput(
                action="ABSTAIN",
                message="I'm not confident I can answer this.",
                options=[],
                score=None,
                trace={
                    "method": "semantic_entropy",
                    "n_samples": len(result.samples),
                    "n_valid": len(valid_samples),
                },
                cost=CostRecord(),  # TODO: track actual cost
            )

        # Compute NLI pairwise relations (strict bidirectional entailment)
        relations = compute_pairwise_relations(
            valid_samples, self.nli, query.text, input_format="answer_only"
        )

        # Cluster with greedy bidirectional (faithful to Kuhn et al.)
        clustering_result = cluster_samples(
            valid_samples, relations, algorithm="greedy_bidirectional"
        )

        # Compute discrete entropy: H = -sum(p_i * log(p_i))
        clusters = clustering_result.clusters
        total = sum(c.size for c in clusters)
        probs = [c.size / total for c in clusters]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Apply two-threshold decision
        if entropy <= self.tau1:
            # Low entropy → ANSWER with majority cluster's answer
            action = "ANSWER"
            majority_cluster = max(clusters, key=lambda c: c.size)
            majority_idx = majority_cluster.members[0]
            message = valid_samples[majority_idx].answer
        elif entropy <= self.tau2:
            # Medium entropy → ASK
            action = "ASK"
            message = "Could you clarify what you mean?"
        else:
            # High entropy → ABSTAIN
            action = "ABSTAIN"
            message = "I'm not confident I can answer this accurately."

        return MethodOutput(
            action=action,
            message=message,
            options=[],
            score=entropy,
            trace={
                "method": "semantic_entropy",
                "entropy": entropy,
                "n_samples": self.n_samples,
                "n_valid": len(valid_samples),
                "n_clusters": len(clusters),
                "tau1": self.tau1,
                "tau2": self.tau2,
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
            Budget: N samples + NLI pairs
        """
        # Approximate NLI pairs: N*(N-1)/2 for pairwise comparisons
        nli_pairs = self.n_samples * (self.n_samples - 1) // 2

        return BudgetSpec(
            llm_calls=1,
            samples_per_call=self.n_samples,
            nli_pairs=nli_pairs,
            description=f"{self.n_samples} samples + {nli_pairs} NLI pairs (+1 if ASK)",
        )


__all__ = ["SemanticEntropy"]
