"""Core type definitions for ADG (§4.1 of spec).

All pydantic models for the ADG system, shared across modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ==============================================================================
# Input and Query
# ==============================================================================


class Query(BaseModel):
    """User query with metadata.

    Args:
        text: The user's question (normalized, NFKC)
        query_id: Unique identifier for caching and tracking
        metadata: Optional metadata (source, timestamp, etc.)
    """

    text: str
    query_id: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


# ==============================================================================
# Sampling Outputs
# ==============================================================================


class Sample(BaseModel):
    """Single sampled answer from the LLM.

    Per spec §4.3: structured prompt outputs interpretation + answer + confidence.
    """

    # Required fields (from structured prompt)
    interpretation: str  # LLM's understanding of the question
    answer: str  # The generated answer
    confidence: Literal["high", "medium", "low"] = "medium"

    # Metadata
    sample_id: int  # Index in the sampling batch (0 to n_samples-1)
    is_valid: bool = True  # False if empty/unparseable
    is_refusal: bool = False  # "I don't know" / "Cannot answer"
    is_hedged: bool = False  # Enumerates multiple interpretations
    hedged_interpretations: list[str] = Field(default_factory=list)

    # Raw outputs (optional)
    raw_text: str | None = None
    logprobs: list[float] | None = None
    tokens: list[str] | None = None


class SamplingResult(BaseModel):
    """Results from sampling N answers.

    Args:
        samples: List of Sample objects
        n_samples: Total requested (including invalid)
        n_valid: Number of valid, non-refusal samples
        n_refusals: Number of refusal samples
        seed: Seed used for sampling
    """

    samples: list[Sample]
    n_samples: int
    n_valid: int
    n_refusals: int
    seed: int


# ==============================================================================
# NLI and Relations
# ==============================================================================


class PairwiseRelations(BaseModel):
    """NLI-based pairwise relations matrix.

    Per spec §4.4: S[i][j] = bidirectional entailment score (0-1).
    """

    # Scores: S[i][j] in [0, 1] where 1 = strong bidirectional entailment
    scores: list[list[float]]  # N x N symmetric matrix

    # Equivalence binary matrix: i ≡ j (soft or strict mode)
    equivalence: list[list[bool]]  # N x N symmetric

    # Metadata
    n_pairs_computed: int  # Actual NLI calls (after deduplication)
    n_pairs_total: int  # N*(N-1)/2
    n_guard_applied: int = 0  # Times entity/number guard fired
    mode: Literal["soft", "strict"] = "soft"


# ==============================================================================
# Clustering
# ==============================================================================


class Cluster(BaseModel):
    """Cluster of semantically equivalent answers.

    Per spec §4.5: clusters are ordered by size (n1 ≥ n2 ≥ ... ≥ nK).
    """

    cluster_id: int  # 0-indexed, ordered by size
    sample_ids: list[int]  # Indices into SamplingResult.samples
    size: int  # len(sample_ids)

    # Representative elements
    representative_answer: str  # From the cluster (e.g., most common)
    representative_interpretation: str | None = None  # From interpretation channel


class ClusteringResult(BaseModel):
    """Results from semantic clustering.

    Args:
        clusters: List of Cluster objects, ordered by size descending
        algorithm: Which clustering algorithm was used
    """

    clusters: list[Cluster]
    n_clusters: int
    algorithm: Literal["greedy_bidirectional", "agglomerative", "connected_components"]
    singletons: int = 0  # Number of singleton clusters (noise)


# ==============================================================================
# Cluster Features (§4.6)
# ==============================================================================


class ClusterFeatures(BaseModel):
    """Features computed from cluster structure.

    Per spec §4.6: significance, mass, intra-cluster agreement.
    """

    # Cluster-level
    cluster_id: int
    is_significant: bool  # size / n_valid >= rho (significance threshold)
    mass: float  # Discrete: size/n_valid or Likelihood-weighted
    intra_agreement: float  # Average S[i][j] within cluster

    # Global features
    n_significant: int  # Number of significant clusters
    p1: float  # Share of largest cluster
    coverage: float  # Sum of masses for significant clusters
    separation: float  # Min pairwise S between significant clusters


class AggregateFeatures(BaseModel):
    """Aggregate features for decision engine.

    Per spec §4.6: derived from all clusters.
    """

    # Cluster structure
    n_clusters: int
    n_significant: int
    singletons: int

    # Masses and shares
    p1: float  # Largest cluster share
    p2: float  # Second-largest cluster share (0 if K < 2)
    coverage: float  # Covered by significant clusters
    separation: float  # Min distance between significant clusters

    # Confidence and noise
    refusal_rate: float  # n_refusals / n_samples
    hedge_rate: float  # Fraction of samples marked hedged
    invalid_rate: float  # (n_samples - n_valid) / n_samples

    # Derived scores (§4.6)
    ask_score: float  # (1 - p1)^gamma * I where I = interpretation_divergence
    abstain_score: float  # max(refusal_rate, invalid_rate) * (1 - p1)
    answer_score: float  # p1 * I_flipped


# ==============================================================================
# Interpretation Channel (§4.7)
# ==============================================================================


class InterpretationEvidence(BaseModel):
    """Evidence from the interpretation channel [ER].

    Per spec §4.7: NLI clustering on self-reported interpretations.
    """

    enabled: bool
    n_interpretation_clusters: int = 0
    interpretation_divergence: float = 0.0  # 1 - p1_interpretations
    hedge_evidence: float = 0.0  # Fraction of hedged samples with ≥2 alternatives

    # Cluster mappings
    interpretation_to_answer_cluster: dict[int, list[int]] = Field(default_factory=dict)


# ==============================================================================
# Decision Outputs (§4.9)
# ==============================================================================


class Action(str):
    """Decision action.

    Per spec §4.9: ANSWER (confident), ASK (ambiguous), ABSTAIN (don't know).
    """

    ANSWER: str = "ANSWER"
    ASK: str = "ASK"
    ABSTAIN: str = "ABSTAIN"


class Decision(BaseModel):
    """Decision output from decision engine.

    Args:
        action: The chosen action (ANSWER, ASK, or ABSTAIN)
        confidence: Confidence score for the chosen action
        confidence_scores: Confidence scores for all actions
        rule_name: Name of the rule that was matched
    """

    action: Literal["ANSWER", "ASK", "ABSTAIN"]
    confidence: float
    confidence_scores: dict[str, float]
    rule_name: str


class DecisionTrace(BaseModel):
    """Decision trace for explainability.

    Per spec §4.9: which rules fired, intermediate values, reasons.
    """

    action: Literal["ANSWER", "ASK", "ABSTAIN"]
    rule_fired: str  # e.g., "R1", "R2a", "FALLBACK"
    reasons: list[str]  # Plain-English explanations

    # Intermediate values used in decision
    thresholds_used: dict[str, float] = Field(default_factory=dict)
    scores_computed: dict[str, float] = Field(default_factory=dict)

    # Feature values at decision time
    features: AggregateFeatures | None = None
    interpretation_evidence: InterpretationEvidence | None = None


# ==============================================================================
# Clarification (§4.10)
# ==============================================================================


class ClarifyingQuestion(BaseModel):
    """Generated clarifying question.

    Per spec §4.10: asks user to disambiguate between interpretations.
    """

    question_text: str  # Natural language question
    options: list[str]  # Interpretations (never raw answers)
    mode: Literal["template", "llm"] = "llm"

    # Cluster context
    cluster_ids: list[int]  # Which clusters this question distinguishes


class ResolvedInterpretation(BaseModel):
    """Resolved user reply to clarifying question.

    Per spec §4.11: maps reply to an option or none.
    """

    option_index: int | None  # Index into ClarifyingQuestion.options (None if unresolved)
    confidence: float  # Match confidence [0, 1]
    method: Literal["exact_match", "fuzzy", "nli", "llm", "none"]


# ==============================================================================
# Main Pipeline Output (§4.1)
# ==============================================================================


class ADGResult(BaseModel):
    """Complete output from the ADG pipeline.

    Per spec §4.1: combines all intermediate outputs and final decision.
    """

    # Input
    query: Query

    # Outputs
    action: Literal["ANSWER", "ASK", "ABSTAIN"]
    decision_trace: DecisionTrace

    # If action == ANSWER
    final_answer: str | None = None
    answer_interpretation: str | None = None  # Which interpretation was answered

    # If action == ASK
    clarifying_question: ClarifyingQuestion | None = None

    # Intermediate results (for analysis and debugging)
    sampling_result: SamplingResult
    relations: PairwiseRelations | None = None
    clustering_result: ClusteringResult | None = None
    aggregate_features: AggregateFeatures | None = None
    interpretation_evidence: InterpretationEvidence | None = None

    # Metadata
    cost: CostRecord
    timestamp: datetime = Field(default_factory=datetime.now)
    config_hash: str | None = None  # For reproducibility


# ==============================================================================
# Conversation State (§4.11)
# ==============================================================================


class Turn(BaseModel):
    """Single conversation turn.

    Per spec §4.11: user message and system response.
    """

    turn_id: int  # 0-indexed
    user_message: str
    system_response: str
    action: Literal["ANSWER", "ASK", "ABSTAIN"]

    # Full ADG result for this turn
    adg_result: ADGResult | None = None

    timestamp: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    """Multi-turn conversation session.

    Per spec §4.11: persistent state across turns.
    """

    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    state: Literal["IDLE", "AWAITING_CLARIFICATION", "ANSWERED", "ABSTAINED"] = "IDLE"

    # Context for clarification resolution
    pending_clarification: ClarifyingQuestion | None = None
    pending_clusters: list[Cluster] | None = None

    # Session metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 3600


# ==============================================================================
# Cost Tracking (§4.1)
# ==============================================================================


class CostRecord(BaseModel):
    """Resource usage tracking.

    Per spec §4.1: tracks samples, tokens, NLI calls, latency, cost.
    """

    # LLM usage
    n_llm_calls: int = 0
    n_samples: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # NLI usage
    n_nli_pairs: int = 0  # Actual pairs computed (after dedup)
    n_nli_forward_passes: int = 0  # Batched forward passes

    # Performance
    wall_time_ms: float = 0.0
    gpu_peak_mem_mb: float | None = None
    api_cost_usd: float | None = None

    # Multi-turn
    turns: int = 1


# ==============================================================================
# Evaluation Data Structures
# ==============================================================================


class GoldInterpretation(BaseModel):
    """Gold-standard interpretation with answers.

    Used for evaluation (datasets like AmbigQA).
    """

    interpretation: str  # Disambiguated question
    answers: list[str]  # Gold answer aliases
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class Example(BaseModel):
    """Dataset example for evaluation.

    Per spec §9: combines query, gold interpretations, annotations.
    """

    example_id: str
    query: str
    is_ambiguous: bool  # Dataset annotation

    # Gold interpretations (empty list for clear questions)
    interpretations: list[GoldInterpretation] = Field(default_factory=list)

    # For clear questions: single set of answer aliases
    answers: list[str] = Field(default_factory=list)

    # Metadata
    dataset: str  # "ambigqa", "nq_open", "clamber"
    split: str  # "dev", "test", "calibration"
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


# ==============================================================================
# Simulated User (§11)
# ==============================================================================


class SimUserInput(BaseModel):
    """Input to simulated user.

    Per spec §11.1: provides context for user to respond to clarification.
    """

    original_question: str
    intended: GoldInterpretation  # User's hidden intent
    other_interpretations: list[GoldInterpretation]
    clarifying_question: str
    options: list[str]  # Options offered by the system


class SimUserReply(BaseModel):
    """Simulated user's reply to clarifying question.

    Per spec §11.1: how the user responds.
    """

    text: str  # Natural language reply
    chosen_option_index: int | None  # If an option was chosen
    behavior: Literal["choose_option", "describe_intent", "none_of_these"]
    leakage_flag: bool = False  # True if reply leaked the answer


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Main output
    "ADGResult",
    # Decision
    "Action",
    "Decision",
    "AggregateFeatures",
    "ClarifyingQuestion",
    # Clustering
    "Cluster",
    "ClusterFeatures",
    "ClusteringResult",
    # Cost
    "CostRecord",
    "DecisionTrace",
    "Example",
    # Evaluation
    "GoldInterpretation",
    # Interpretation
    "InterpretationEvidence",
    # Relations
    "PairwiseRelations",
    # Input/Query
    "Query",
    "ResolvedInterpretation",
    # Sampling
    "Sample",
    "SamplingResult",
    "SessionState",
    # Simulated user
    "SimUserInput",
    "SimUserReply",
    # Conversation
    "Turn",
]
