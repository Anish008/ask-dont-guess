"""Semantic clustering for grouping equivalent LLM outputs.

Per spec §4.5-4.7: Pairwise NLI scoring, clustering algorithms, and features.
"""

from adg.clustering.algorithms import cluster_samples
from adg.clustering.features import compute_aggregate_features
from adg.clustering.interpretation import compute_interpretation_divergence
from adg.clustering.relations import compute_pairwise_relations

__all__ = [
    "cluster_samples",
    "compute_aggregate_features",
    "compute_interpretation_divergence",
    "compute_pairwise_relations",
]
