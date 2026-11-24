"""Session management endpoints.

Per spec §10.1: Get and delete sessions.
"""

from fastapi import APIRouter, Depends, HTTPException

from adg.api.deps import get_conversation_manager
from adg.api.schemas import SessionState
from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/v1/sessions")


@router.get("/{session_id}", response_model=SessionState)
async def get_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),  # noqa: B008
) -> SessionState:
    """Get session state.

    Args:
        session_id: Session ID
        manager: Conversation manager

    Returns:
        Session state

    Raises:
        HTTPException: If session not found
    """
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    log.info("session_retrieved", session_id=session_id)
    return session


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    manager: ConversationManager = Depends(get_conversation_manager),  # noqa: B008
) -> None:
    """Delete session.

    Args:
        session_id: Session ID
        manager: Conversation manager
    """
    manager.reset_session(session_id)
    log.info("session_deleted", session_id=session_id)
