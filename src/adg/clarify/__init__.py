"""Clarification generation and reply resolution.

Per spec §4.10: Generate clarifying questions and resolve user replies.
"""

from adg.clarify.generator import QuestionGenerator
from adg.clarify.interpretation import ReplyResolver

__all__ = ["QuestionGenerator", "ReplyResolver"]
