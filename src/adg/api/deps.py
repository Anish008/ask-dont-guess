"""Dependency injection for FastAPI.

Per spec §10.1: Provide conversation manager and settings via DI.
"""

from functools import lru_cache

from adg.conversation.manager import ConversationManager
from adg.core.logging import get_logger
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer

log = get_logger(__name__)

# Global manager instance (ponytail: single instance for simplicity)
_manager: ConversationManager | None = None


def get_conversation_manager() -> ConversationManager:
    """Get conversation manager instance.

    Returns:
        ConversationManager instance
    """
    global _manager  # noqa: PLW0603

    if _manager is None:
        # ponytail: mock backends for now, real models via config later
        llm_backend = MockLLMBackend()
        nli_scorer = MockNLIScorer()
        _manager = ConversationManager(
            llm_backend=llm_backend,
            nli_scorer=nli_scorer,
        )
        log.info("conversation_manager_created")

    return _manager


@lru_cache
def get_version() -> str:
    """Get ADG version.

    Returns:
        Version string
    """
    return "0.1.0"
