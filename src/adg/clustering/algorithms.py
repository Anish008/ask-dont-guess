"""Clustering algorithms for semantic equivalence.

Per spec §4.5: Implements 3 clustering algorithms for grouping semantically equivalent answers.
"""

from typing import Literal

from adg.core.logging import get_logger
from adg.core.types import Cluster, ClusteringResult, PairwiseRelations, Sample

log = get_logger(__name__)

# Default equivalence threshold for agglomerative clustering
_AGGLOMERATIVE_THRESHOLD = 0.5


def cluster_samples(
    samples: list[Sample],
    relations: PairwiseRelations,
    algorithm: Literal[
        "greedy_bidirectional", "agglomerative", "connected_components"
    ] = "agglomerative",
) -> ClusteringResult:
    """Cluster samples into semantically equivalent groups.

    Args:
        samples: List of samples to cluster
        relations: Pairwise relations (equivalence matrix)
        algorithm: Clustering algorithm to use

    Returns:
        ClusteringResult with clusters ordered by size descending
    """
    log.debug("clustering_samples", n_samples=len(samples), algorithm=algorithm)

    if algorithm == "greedy_bidirectional":
        clusters = _greedy_bidirectional(samples, relations)
    elif algorithm == "agglomerative":
        clusters = _agglomerative(samples, relations)
    elif algorithm == "connected_components":
        clusters = _connected_components(samples, relations)
    else:
        msg = f"Unknown clustering algorithm: {algorithm}"
        raise ValueError(msg)

    # Sort clusters by size descending
    clusters.sort(key=lambda c: c.size, reverse=True)

    # Renumber cluster IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    # Count singletons
    singletons = sum(1 for c in clusters if c.size == 1)

    log.info(
        "clustering_complete",
        algorithm=algorithm,
        n_clusters=len(clusters),
        singletons=singletons,
    )

    return ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        algorithm=algorithm,
        singletons=singletons,
    )


def _greedy_bidirectional(
    samples: list[Sample],
    relations: PairwiseRelations,
) -> list[Cluster]:
    """Greedy clustering based on equivalence matrix.

    Per spec §4.5: Greedily merge samples that are equivalent.

    Args:
        samples: List of samples
        relations: Pairwise relations

    Returns:
        List of clusters
    """
    n = len(samples)
    equivalence = relations.equivalence

    # Track which samples are assigned
    assigned = [False] * n
    clusters: list[Cluster] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new cluster with sample i
        cluster_members = [i]
        assigned[i] = True

        # Add all equivalent samples
        for j in range(i + 1, n):
            if not assigned[j] and equivalence[i][j]:
                cluster_members.append(j)
                assigned[j] = True

        # Create cluster
        cluster = Cluster(
            cluster_id=len(clusters),
            sample_ids=cluster_members,
            size=len(cluster_members),
            representative_answer=samples[cluster_members[0]].answer,
            representative_interpretation=samples[cluster_members[0]].interpretation,
        )
        clusters.append(cluster)

    return clusters


def _agglomerative(
    samples: list[Sample],
    relations: PairwiseRelations,
) -> list[Cluster]:
    """Agglomerative hierarchical clustering.

    Per spec §4.5: Hierarchical clustering with average linkage.

    Args:
        samples: List of samples
        relations: Pairwise relations

    Returns:
        List of clusters
    """
    n = len(samples)
    scores = relations.scores

    # Initialize: each sample in its own cluster
    clusters_dict: dict[int, list[int]] = {i: [i] for i in range(n)}

    # Merge until no more equivalent pairs
    while True:
        # Find best merge candidate
        best_score = -1.0
        best_pair = None

        cluster_ids = list(clusters_dict.keys())
        for i, ci in enumerate(cluster_ids):
            for cj in cluster_ids[i + 1 :]:
                # Compute average linkage score
                avg_score = _average_linkage_score(clusters_dict[ci], clusters_dict[cj], scores)

                # Check if above threshold (use equivalence as proxy)
                # Average score >= 0.5 (or use first pair's equivalence)
                if avg_score > best_score:
                    best_score = avg_score
                    best_pair = (ci, cj)

        # Stop if no good merges (threshold for equivalence)
        if best_pair is None or best_score < _AGGLOMERATIVE_THRESHOLD:
            break

        # Merge best pair
        ci, cj = best_pair
        clusters_dict[ci].extend(clusters_dict[cj])
        del clusters_dict[cj]

    # Convert to Cluster objects
    clusters = []
    for cluster_id, member_ids in clusters_dict.items():
        cluster = Cluster(
            cluster_id=cluster_id,
            sample_ids=member_ids,
            size=len(member_ids),
            representative_answer=samples[member_ids[0]].answer,
            representative_interpretation=samples[member_ids[0]].interpretation,
        )
        clusters.append(cluster)

    return clusters


def _connected_components(
    samples: list[Sample],
    relations: PairwiseRelations,
) -> list[Cluster]:
    """Connected components clustering.

    Per spec §4.5: Graph-based clustering using equivalence edges.

    Args:
        samples: List of samples
        relations: Pairwise relations

    Returns:
        List of clusters
    """
    n = len(samples)
    equivalence = relations.equivalence

    # Build adjacency list
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if equivalence[i][j]:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components via DFS
    visited = [False] * n
    clusters: list[Cluster] = []

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

        # Create cluster
        cluster = Cluster(
            cluster_id=len(clusters),
            sample_ids=sorted(component),
            size=len(component),
            representative_answer=samples[component[0]].answer,
            representative_interpretation=samples[component[0]].interpretation,
        )
        clusters.append(cluster)

    return clusters


def _average_linkage_score(
    cluster1: list[int],
    cluster2: list[int],
    scores: list[list[float]],
) -> float:
    """Compute average linkage score between two clusters.

    Args:
        cluster1: Sample indices in cluster 1
        cluster2: Sample indices in cluster 2
        scores: Pairwise score matrix

    Returns:
        Average score across all pairs
    """
    if not cluster1 or not cluster2:
        return 0.0

    total = 0.0
    count = 0

    for i in cluster1:
        for j in cluster2:
            total += scores[i][j]
            count += 1

    return total / count if count > 0 else 0.0


__all__ = ["cluster_samples"]
