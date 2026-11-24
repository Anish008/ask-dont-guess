"""LLM-based simulated user.

Per spec §11.3: GPT-4o-mini or local LLM with strict prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adg.core.errors import NotImplementedError as ADGNotImplementedError
from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.simuser.base import SimUserInput, SimUserReply

if TYPE_CHECKING:
    from adg.simuser.rule_user import RuleBasedUser

log = get_logger(__name__)


class LLMUser:
    """LLM-based simulated user.

    Per spec §11.3: Uses LLM with strict prompt, temperature 0,
    JSON output. Falls back to rule user on leakage.

    TODO: Implement actual LLM prompting and JSON parsing.
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        fallback_user: RuleBasedUser | None = None,
    ):
        """Initialize LLM user.

        Args:
            llm_backend: LLM for generating replies
            fallback_user: Fallback on leakage
        """
        self.llm = llm_backend
        self.fallback = fallback_user
        log.info("llm_user_initialized")

    def reply(self, input_data: SimUserInput) -> SimUserReply:
        """Generate LLM-based reply.

        Per spec §11.3: LLM with strict prompt, check leakage,
        fall back to rule user if needed.

        Args:
            input_data: Input with intent and clarifying question

        Returns:
            Reply from LLM or fallback
        """
        # TODO: Implement LLM prompting
        # 1. Format prompt with simuser.txt template
        # 2. Call LLM with temperature=0
        # 3. Parse JSON output
        # 4. Check leakage
        # 5. Retry once if leakage detected
        # 6. Fall back to rule user if still leaking

        raise ADGNotImplementedError(
            message="LLM user not yet implemented",
            hint="Use RuleBasedUser for now, or implement LLM prompting",
        )


__all__ = ["LLMUser"]
