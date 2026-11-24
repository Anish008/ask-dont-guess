"""API request/response schemas.

Per spec §10.1: Request/response models for FastAPI endpoints.
"""

from pydantic import BaseModel, Field

from adg.core.types import ADGResult, Decision, SessionState

# Re-export core types
__all__ = [
    "ADGResult",
    "Decision",
    "SessionState",
    "ChatRequest",
    "ChatResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "HealthResponse",
    "StatsResponse",
]


class ChatRequest(BaseModel):
    """Chat request."""

    session_id: str | None = Field(None, description="Session ID (auto-generated if not provided)")
    message: str = Field(..., description="User message", min_length=1)


class ChatResponse(BaseModel):
    """Chat response."""

    session_id: str
    message: str
    action: str  # ANSWER, ASK, ABSTAIN
    options: list[str] | None = None
    decision: Decision
    turn_number: int


class AnalyzeRequest(BaseModel):
    """Analysis request with optional overrides."""

    query: str = Field(..., description="Query text", min_length=1)
    n_samples: int | None = Field(None, ge=1, le=16, description="Number of samples")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    clustering_algorithm: str | None = Field(None, description="Clustering algorithm")


class AnalyzeResponse(BaseModel):
    """Analysis response."""

    query: str
    action: str  # ANSWER, ASK, ABSTAIN
    message: str
    options: list[str] | None = None
    decision: Decision


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
    device: str
    version: str


class StatsResponse(BaseModel):
    """Live statistics response."""

    total_requests: int
    decisions_by_action: dict[str, int]
    mean_latency_ms: float
