"""Simulated user protocol.

Per spec §11.1: Base protocol for simulated users.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from adg.data.schema import GoldInterpretation


class SimUserInput(BaseModel):
    """Input to simulated user.

    Args:
        original_question: Original ambiguous question
        intended: Hidden intent (disambiguated question)
        other_interpretations: Other gold interpretations
        clarifying_question: Question from the system
        options: Options offered (may be empty)
    """

    original_question: str
    intended: GoldInterpretation
    other_interpretations: list[GoldInterpretation]
    clarifying_question: str
    options: list[str] = Field(default_factory=list)


class SimUserReply(BaseModel):
    """Reply from simulated user.

    Args:
        text: Reply text
        chosen_option_index: Index of chosen option (if applicable)
        behavior: How the user replied
        leakage_flag: Whether reply leaked gold answer
    """

    text: str
    chosen_option_index: int | None = None
    behavior: Literal["choose_option", "describe_intent", "none_of_these", "generic"]
    leakage_flag: bool = False


class SimulatedUser(Protocol):
    """Protocol for simulated users.

    Per spec §11: Simulated users reply to clarifying questions
    without leaking gold answers.
    """

    def reply(self, input_data: SimUserInput) -> SimUserReply:
        """Generate a reply to a clarifying question.

        Args:
            input_data: Input with intent and clarifying question

        Returns:
            Reply with text and behavior flag
        """
        ...


__all__ = [
    "SimulatedUser",
    "SimUserInput",
    "SimUserReply",
]
