"""Health check endpoint.

Per spec §10.1: Health/readiness check.
"""

from fastapi import APIRouter, Depends

from adg.api.deps import get_conversation_manager, get_version
from adg.api.schemas import HealthResponse
from adg.conversation.manager import ConversationManager
from adg.core.device import detect_hardware

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: ConversationManager = Depends(get_conversation_manager),  # noqa: B008
    version: str = Depends(get_version),
) -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status with model info
    """
    hw_info = detect_hardware()

    return HealthResponse(
        status="ok",
        models_loaded=True,  # Manager exists means models loaded
        device=hw_info.device.value,
        version=version,
    )
