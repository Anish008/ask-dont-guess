"""Core data models and types for ADG."""

from adg.core import device, errors, hashing, logging, protocols, registry, seeding, types

# Re-export commonly used errors
from adg.core.errors import (
    ADGError,
    GenerationError,
    InvalidQueryError,
    NLIError,
    QueryTooLongError,
)

# Re-export protocols
from adg.core.protocols import LLMBackend, Method, NLIScorer, SimulatedUser

# Re-export registries
from adg.core.registry import (
    backend_registry,
    method_registry,
    nli_registry,
    register_backend,
    register_method,
    register_nli,
    register_simuser,
    simuser_registry,
)

# Re-export commonly used types
from adg.core.types import (
    Action,
    ADGResult,
    AggregateFeatures,
    ClarifyingQuestion,
    Cluster,
    ClusterFeatures,
    ClusteringResult,
    CostRecord,
    DecisionTrace,
    Example,
    GoldInterpretation,
    InterpretationEvidence,
    PairwiseRelations,
    Query,
    ResolvedInterpretation,
    Sample,
    SamplingResult,
    SessionState,
    SimUserInput,
    SimUserReply,
    Turn,
)

__all__ = [
    # Errors
    "ADGError",
    "ADGResult",
    "Action",
    "AggregateFeatures",
    "ClarifyingQuestion",
    "Cluster",
    "ClusterFeatures",
    "ClusteringResult",
    "CostRecord",
    "DecisionTrace",
    "Example",
    "GenerationError",
    "GoldInterpretation",
    "InterpretationEvidence",
    "InvalidQueryError",
    # Protocols
    "LLMBackend",
    "Method",
    "NLIError",
    "NLIScorer",
    "PairwiseRelations",
    # Types
    "Query",
    "QueryTooLongError",
    "ResolvedInterpretation",
    "Sample",
    "SamplingResult",
    "SessionState",
    "SimUserInput",
    "SimUserReply",
    "SimulatedUser",
    "Turn",
    # Registries
    "backend_registry",
    # Modules
    "device",
    "errors",
    "hashing",
    "logging",
    "method_registry",
    "nli_registry",
    "protocols",
    "register_backend",
    "register_method",
    "register_nli",
    "register_simuser",
    "registry",
    "seeding",
    "simuser_registry",
    "types",
]
