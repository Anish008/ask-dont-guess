"""Conversation state management.

Per spec §4.11: Track multi-turn conversation sessions.
"""

from datetime import datetime

from adg.core.logging import get_logger
from adg.core.types import ClarifyingQuestion, Cluster, SessionState, Turn

log = get_logger(__name__)


def create_session(session_id: str, ttl_seconds: int = 3600) -> SessionState:
    """Create a new conversation session.

    Args:
        session_id: Unique session identifier
        ttl_seconds: Session time-to-live in seconds (default 1 hour)

    Returns:
        New SessionState
    """
    log.info("session_created", session_id=session_id, ttl_seconds=ttl_seconds)
    return SessionState(
        session_id=session_id,
        ttl_seconds=ttl_seconds,
    )


def add_turn(
    session: SessionState,
    user_message: str,
    system_response: str,
    action: str,
) -> SessionState:
    """Add a turn to the conversation history.

    Args:
        session: Current session state
        user_message: User's message
        system_response: System's response
        action: Action taken (ANSWER, ASK, ABSTAIN)

    Returns:
        Updated SessionState
    """
    turn_id = len(session.turns)
    turn = Turn(
        turn_id=turn_id,
        user_message=user_message,
        system_response=system_response,
        action=action,
    )

    session.turns.append(turn)
    session.last_updated = datetime.now()

    log.info(
        "turn_added",
        session_id=session.session_id,
        turn_id=turn_id,
        action=action,
    )

    return session


def set_awaiting_clarification(
    session: SessionState,
    clarifying_question: ClarifyingQuestion,
    clusters: list[Cluster],
) -> SessionState:
    """Set session to await user clarification.

    Args:
        session: Current session state
        clarifying_question: Question to ask user
        clusters: Clusters that generated the question

    Returns:
        Updated SessionState
    """
    session.state = "AWAITING_CLARIFICATION"
    session.pending_clarification = clarifying_question
    session.pending_clusters = clusters
    session.last_updated = datetime.now()

    log.info(
        "awaiting_clarification",
        session_id=session.session_id,
        n_options=len(clarifying_question.options),
    )

    return session


def clear_pending_clarification(session: SessionState) -> SessionState:
    """Clear pending clarification after resolution.

    Args:
        session: Current session state

    Returns:
        Updated SessionState
    """
    session.state = "IDLE"
    session.pending_clarification = None
    session.pending_clusters = None
    session.last_updated = datetime.now()

    log.info("clarification_cleared", session_id=session.session_id)

    return session


def set_answered(session: SessionState) -> SessionState:
    """Mark session as answered.

    Args:
        session: Current session state

    Returns:
        Updated SessionState
    """
    session.state = "ANSWERED"
    session.last_updated = datetime.now()

    log.info("session_answered", session_id=session.session_id)

    return session


def set_abstained(session: SessionState) -> SessionState:
    """Mark session as abstained.

    Args:
        session: Current session state

    Returns:
        Updated SessionState
    """
    session.state = "ABSTAINED"
    session.last_updated = datetime.now()

    log.info("session_abstained", session_id=session.session_id)

    return session


def is_expired(session: SessionState) -> bool:
    """Check if session has expired.

    Args:
        session: Session state to check

    Returns:
        True if session is expired
    """
    elapsed = (datetime.now() - session.last_updated).total_seconds()
    expired = elapsed > session.ttl_seconds

    if expired:
        log.info(
            "session_expired",
            session_id=session.session_id,
            elapsed_seconds=elapsed,
        )

    return expired


__all__ = [
    "add_turn",
    "clear_pending_clarification",
    "create_session",
    "is_expired",
    "set_abstained",
    "set_answered",
    "set_awaiting_clarification",
]
