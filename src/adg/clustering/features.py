"""Cluster features computation.

Per spec §4.6: Compute features from cluster structure for decision engine.
"""

from adg.core.logging import get_logger
from adg.core.types import AggregateFeatures, ClusteringResult, PairwiseRelations, SamplingResult

log = get_logger(__name__)

# Minimum number of significant clusters needed for separation computation
_MIN_CLUSTERS_FOR_SEPARATION = 2


def compute_aggregate_features(
    sampling_result: SamplingResult,
    clustering_result: ClusteringResult,
    relations: PairwiseRelations,
    significance_rho: float = 0.25,
    gamma: float = 0.5,
    interpretation_divergence: float = 0.0,
) -> AggregateFeatures:
    """Compute aggregate features for decision engine.

    Per spec §4.6: Derives features from clusters, masses, and relations.

    Args:
        sampling_result: Sampling results with n_valid, n_refusals
        clustering_result: Clustering results with clusters
        relations: Pairwise relations for intra-cluster agreement
        significance_rho: Significance threshold (default 0.25)
        gamma: Exponent for ask_score (default 0.5)
        interpretation_divergence: Divergence from interpretation channel (default 0.0)

    Returns:
        AggregateFeatures with all decision-relevant features
    """
    n_samples = sampling_result.n_samples
    n_valid = sampling_result.n_valid
    n_refusals = sampling_result.n_refusals
    clusters = clustering_result.clusters

    # Count invalid samples
    invalid_samples = n_samples - n_valid - n_refusals

    # Compute masses (discrete: cluster size / n_valid)
    masses = [cluster.size / n_valid if n_valid > 0 else 0.0 for cluster in clusters]

    # Identify significant clusters
    significant_clusters = [
        cluster for cluster, mass in zip(clusters, masses, strict=False) if mass >= significance_rho
    ]
    n_significant = len(significant_clusters)

    # Compute p1 (largest cluster share)
    p1 = masses[0] if masses else 0.0

    # Compute p2 (second-largest cluster share)
    p2 = masses[1] if len(masses) > 1 else 0.0

    # Compute coverage (sum of significant cluster masses)
    coverage = sum(mass for mass in masses if mass >= significance_rho)

    # Compute separation (min pairwise score between significant clusters)
    separation = _compute_separation(significant_clusters, relations)

    # Compute hedge rate
    hedge_rate = (
        sum(1 for s in sampling_result.samples if s.is_hedged) / n_samples if n_samples > 0 else 0.0
    )

    # Compute rates
    refusal_rate = n_refusals / n_samples if n_samples > 0 else 0.0
    invalid_rate = invalid_samples / n_samples if n_samples > 0 else 0.0

    # Compute derived scores (§4.6)
    # ask_score = (1 - p1)^gamma * interpretation_divergence
    ask_score = ((1 - p1) ** gamma) * interpretation_divergence

    # abstain_score = max(refusal_rate, invalid_rate) * (1 - p1)
    abstain_score = max(refusal_rate, invalid_rate) * (1 - p1)

    # answer_score = p1 * (1 - interpretation_divergence)
    answer_score = p1 * (1 - interpretation_divergence)

    log.debug(
        "aggregate_features_computed",
        n_clusters=len(clusters),
        n_significant=n_significant,
        p1=round(p1, 3),
        p2=round(p2, 3),
        coverage=round(coverage, 3),
    )

    return AggregateFeatures(
        n_clusters=len(clusters),
        n_significant=n_significant,
        singletons=clustering_result.singletons,
        p1=p1,
        p2=p2,
        coverage=coverage,
        separation=separation,
        refusal_rate=refusal_rate,
        hedge_rate=hedge_rate,
        invalid_rate=invalid_rate,
        ask_score=ask_score,
        abstain_score=abstain_score,
        answer_score=answer_score,
    )


def _compute_separation(
    significant_clusters: list,
    relations: PairwiseRelations,
) -> float:
    """Compute minimum pairwise score between significant clusters.

    Args:
        significant_clusters: List of significant clusters
        relations: Pairwise relations

    Returns:
        Minimum score, or 1.0 if < 2 significant clusters
    """
    if len(significant_clusters) < _MIN_CLUSTERS_FOR_SEPARATION:
        return 1.0  # No separation needed

    scores = relations.scores
    min_score = 1.0

    for i, cluster_i in enumerate(significant_clusters):
        for cluster_j in significant_clusters[i + 1 :]:
            # Compute average score between clusters
            total = 0.0
            count = 0
            for si in cluster_i.sample_ids:
                for sj in cluster_j.sample_ids:
                    total += scores[si][sj]
                    count += 1

            avg_score = total / count if count > 0 else 0.0
            min_score = min(min_score, avg_score)

    return min_score


__all__ = ["compute_aggregate_features"]
