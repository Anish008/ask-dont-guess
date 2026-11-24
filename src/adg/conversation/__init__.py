"""Conversation management for multi-turn interactions.

Per spec §4.11: Track and orchestrate multi-turn conversations.
"""

from adg.conversation.manager import ConversationManager
from adg.conversation.state import (
    add_turn,
    clear_pending_clarification,
    create_session,
    is_expired,
    set_abstained,
    set_answered,
    set_awaiting_clarification,
)

__all__ = [
    "ConversationManager",
    "create_session",
    "add_turn",
    "set_awaiting_clarification",
    "clear_pending_clarification",
    "set_answered",
    "set_abstained",
    "is_expired",
]
