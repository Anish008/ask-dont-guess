"""Interpretation channel for detecting multiple interpretations.

Per spec §4.7: Cluster interpretations to detect interpretation divergence.
"""

from adg.core.logging import get_logger
from adg.core.protocols import NLIScorer
from adg.core.types import Cluster, PairwiseRelations, Sample

log = get_logger(__name__)


def compute_interpretation_divergence(
    samples: list[Sample],
    clusters: list[Cluster],
    nli_scorer: NLIScorer,
    threshold: float = 0.5,
) -> float:
    """Compute interpretation divergence across clusters.

    Per spec §4.7: Use NLI to cluster interpretations and detect divergence.

    Args:
        samples: List of samples
        clusters: Clusters from answer clustering
        nli_scorer: NLI scorer for interpretation comparison
        threshold: Threshold for interpretation equivalence

    Returns:
        Interpretation divergence score (0.0 = consensus, 1.0 = max divergence)
    """
    if len(clusters) == 0:
        return 0.0

    log.debug(
        "computing_interpretation_divergence",
        n_samples=len(samples),
        n_clusters=len(clusters),
    )

    # Extract representative interpretations from each cluster
    representative_interpretations = []
    for cluster in clusters:
        # Use first sample's interpretation as representative
        rep_sample_id = cluster.sample_ids[0]
        representative_interpretations.append(samples[rep_sample_id].interpretation)

    # If only one cluster, check interpretation variation within it
    if len(clusters) == 1:
        cluster = clusters[0]
        if cluster.size == 1:
            return 0.0  # Single sample, no divergence

        # Compare interpretations within the cluster
        interpretations = [samples[sid].interpretation for sid in cluster.sample_ids]
        intra_divergence = _compute_intra_cluster_interpretation_divergence(
            interpretations, nli_scorer, threshold
        )
        return intra_divergence

    # Multiple clusters: compute inter-cluster interpretation divergence
    inter_divergence = _compute_inter_cluster_interpretation_divergence(
        representative_interpretations, nli_scorer, threshold
    )

    log.info(
        "interpretation_divergence_computed",
        n_clusters=len(clusters),
        divergence=round(inter_divergence, 3),
    )

    return inter_divergence


def _compute_intra_cluster_interpretation_divergence(
    interpretations: list[str],
    nli_scorer: NLIScorer,
    threshold: float,
) -> float:
    """Compute interpretation divergence within a single cluster.

    Args:
        interpretations: List of interpretation texts
        nli_scorer: NLI scorer
        threshold: Equivalence threshold

    Returns:
        Divergence score (0.0 = all equivalent, 1.0 = all different)
    """
    n = len(interpretations)
    if n <= 1:
        return 0.0

    # Create pseudo-samples for NLI scoring
    pseudo_samples = [
        Sample(interpretation=interp, answer="", sample_id=i)
        for i, interp in enumerate(interpretations)
    ]

    # Score interpretation pairs
    result = nli_scorer.score_pairs(pseudo_samples, query_text="", input_format="answer_only")

    # Count non-equivalent pairs
    non_equiv_count = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if result.scores[i][j] < threshold:
                non_equiv_count += 1

    # Divergence = fraction of non-equivalent pairs
    divergence = non_equiv_count / total_pairs if total_pairs > 0 else 0.0
    return divergence


def _compute_inter_cluster_interpretation_divergence(
    interpretations: list[str],
    nli_scorer: NLIScorer,
    threshold: float,
) -> float:
    """Compute interpretation divergence across multiple clusters.

    Args:
        interpretations: List of representative interpretation texts
        nli_scorer: NLI scorer
        threshold: Equivalence threshold

    Returns:
        Divergence score (0.0 = consensus, 1.0 = max divergence)
    """
    n = len(interpretations)
    if n <= 1:
        return 0.0

    # Create pseudo-samples for NLI scoring
    pseudo_samples = [
        Sample(interpretation=interp, answer=interp, sample_id=i)
        for i, interp in enumerate(interpretations)
    ]

    # Score interpretation pairs (use answer_only since we put interpretation in answer)
    result = nli_scorer.score_pairs(pseudo_samples, query_text="", input_format="answer_only")

    # Cluster interpretations
    interpretation_clusters = _cluster_interpretations(result, threshold)

    # Divergence based on number of interpretation clusters
    # 1 cluster = 0.0, n clusters = 1.0
    n_interp_clusters = len(interpretation_clusters)
    divergence = (n_interp_clusters - 1) / (n - 1) if n > 1 else 0.0

    log.debug(
        "inter_cluster_divergence",
        n_interpretations=n,
        n_interp_clusters=n_interp_clusters,
        divergence=round(divergence, 3),
    )

    return divergence


def _cluster_interpretations(
    relations: PairwiseRelations,
    threshold: float,
) -> list[list[int]]:
    """Cluster interpretations using connected components.

    Args:
        relations: Pairwise relations for interpretations
        threshold: Equivalence threshold

    Returns:
        List of clusters (each cluster is a list of sample indices)
    """
    n = len(relations.scores)
    if n == 0:
        return []

    # Build adjacency list based on threshold
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if relations.scores[i][j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components via DFS
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue

        # DFS to find component
        component = []
        stack = [i]

        while stack:
            node = stack.pop()
            if visited[node]:
                continue

            visited[node] = True
            component.append(node)

            for neighbor in adj[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)

        clusters.append(sorted(component))

    return clusters


__all__ = ["compute_interpretation_divergence"]
