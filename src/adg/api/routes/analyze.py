"""Analysis endpoint.

Per spec §10.1: One-shot analysis without session.
"""

from fastapi import APIRouter, Depends

from adg.api.deps import get_conversation_manager
from adg.api.schemas import AnalyzeRequest, AnalyzeResponse
from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.core.types import Query

log = get_logger(__name__)
router = APIRouter(prefix="/v1")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    manager: ConversationManager = Depends(get_conversation_manager),  # noqa: B008
) -> AnalyzeResponse:
    """Analyze query without session.

    Args:
        request: Analysis request with query and optional overrides
        manager: Conversation manager

    Returns:
        ADG result with decision and clusters
    """
    # Create query
    query = Query(query_id="analyze", text=request.query)

    # Create temporary session
    session = manager.start_session("temp")

    # Apply overrides if provided
    # ponytail: simple override via manager attributes
    original_n = manager.n_samples
    original_temp = manager.temperature
    original_top_p = manager.top_p
    original_algo = manager.clustering_algorithm

    if request.n_samples is not None:
        manager.n_samples = request.n_samples
    if request.temperature is not None:
        manager.temperature = request.temperature
    if request.top_p is not None:
        manager.top_p = request.top_p
    if request.clustering_algorithm is not None:
        manager.clustering_algorithm = request.clustering_algorithm

    try:
        # Process query
        session, response_text, decision = manager.process_query(session, query)

        # Get options if this is an ASK action
        options = None
        if session.clarifying_question is not None:
            options = session.clarifying_question.options

        log.info(
            "analyze_complete",
            action=decision.action.value,
            n_samples=manager.n_samples,
        )

        return AnalyzeResponse(
            query=request.query,
            action=decision.action.value,
            message=response_text,
            options=options,
            decision=decision,
        )

    finally:
        # Restore original settings
        manager.n_samples = original_n
        manager.temperature = original_temp
        manager.top_p = original_top_p
        manager.clustering_algorithm = original_algo
