"""Statistics endpoint.

Per spec §10.1: Live statistics since server start.
"""

from fastapi import APIRouter

from adg.api.schemas import StatsResponse

router = APIRouter(prefix="/v1")

# ponytail: simple in-memory counters
_stats = {
    "total_requests": 0,
    "decisions": {"ANSWER": 0, "ASK": 0, "ABSTAIN": 0},
    "total_latency_ms": 0.0,
}


def increment_stats(action: str, latency_ms: float) -> None:
    """Increment statistics.

    Args:
        action: Decision action
        latency_ms: Request latency in milliseconds
    """
    _stats["total_requests"] += 1
    _stats["decisions"][action] = _stats["decisions"].get(action, 0) + 1
    _stats["total_latency_ms"] += latency_ms


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get live statistics.

    Returns:
        Statistics response
    """
    total = _stats["total_requests"]
    mean_latency = _stats["total_latency_ms"] / total if total > 0 else 0.0

    return StatsResponse(
        total_requests=total,
        decisions_by_action=_stats["decisions"],
        mean_latency_ms=mean_latency,
    )
