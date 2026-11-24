"""Chat endpoint.

Per spec §10.1: Multi-turn chat with session management.
"""

import uuid

from fastapi import APIRouter, Depends

from adg.api.deps import get_conversation_manager
from adg.api.schemas import ChatRequest, ChatResponse
from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.types import Query

log = get_logger(__name__)
router = APIRouter(prefix="/v1")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    manager: ConversationManager = Depends(get_conversation_manager),  # noqa: B008
) -> ChatResponse:
    """Process chat message.

    Args:
        request: Chat request with message and optional session_id
        manager: Conversation manager

    Returns:
        Chat response with action and decision
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Get or create session
    session = manager.get_session(session_id)
    if session is None:
        session = manager.start_session(session_id)
        log.info("session_created", session_id=session_id)

    # Create query
    query = Query(query_id=f"{session_id}_{session.turn_number}", text=request.message)

    # Process query
    session, response_text, decision = manager.process_query(session, query)

    # Get options if this is an ASK action
    options = None
    if session.clarifying_question is not None:
        options = session.clarifying_question.options

    log.info(
        "chat_processed",
        session_id=session_id,
        action=decision.action.value,
        turn=session.turn_number,
    )

    return ChatResponse(
        session_id=session_id,
        message=response_text,
        action=decision.action.value,
        options=options,
        decision=decision,
        turn_number=session.turn_number,
    )
