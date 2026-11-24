"""End-to-end multi-turn evaluation.

Per spec §12.1: Run method with simulated user, measure accuracy and turns.
"""

from typing import Literal

from pydantic import BaseModel, Field

from adg.core.logging import get_logger
from adg.core.types import CostRecord, Query
from adg.data.schema import Example, GoldInterpretation
from adg.evaluation.metrics import score_answer
from adg.methods.base import Method
from adg.simuser.base import SimulatedUser, SimUserInput

log = get_logger(__name__)


class E2EResult(BaseModel):
    """End-to-end evaluation result.

    Args:
        example_id: Example identifier
        intended_interp_id: Intended interpretation ID
        final_action: Final action (ANSWER or ABSTAIN)
        final_answer: Final answer text (if applicable)
        exact_match: EM score
        token_f1: Token F1 score
        num_turns: Number of clarification turns
        total_cost: Aggregated cost
        leakage_events: Number of leakage events
        trace: Full execution trace
    """

    example_id: str
    intended_interp_id: str
    final_action: Literal["ANSWER", "ABSTAIN"]
    final_answer: str | None = None
    exact_match: float = 0.0
    token_f1: float = 0.0
    num_turns: int = 0
    total_cost: CostRecord
    leakage_events: int = 0
    trace: dict = Field(default_factory=dict)


def run_e2e_evaluation(
    method: Method,
    example: Example,
    intended_interpretation: GoldInterpretation,
    simulated_user: SimulatedUser,
    max_turns: int = 3,
) -> E2EResult:
    """Run end-to-end evaluation on one example.

    Per spec §12.1: Method decides → simuser replies → method answers.

    Args:
        method: Method to evaluate
        example: Test example
        intended_interpretation: Hidden user intent
        simulated_user: Simulated user for clarification
        max_turns: Maximum clarification turns

    Returns:
        E2E result with accuracy and cost
    """
    query = Query(
        query_id=example.example_id,
        text=example.question,
    )

    # Initial decision
    output = method.decide(query)

    total_cost = output.cost
    num_turns = 0
    leakage_events = 0

    trace = {
        "initial_decision": output.action,
        "turns": [],
    }

    # Multi-turn loop
    while output.action == "ASK" and num_turns < max_turns:
        num_turns += 1

        # Simulated user replies
        sim_input = SimUserInput(
            original_question=example.question,
            intended=intended_interpretation,
            other_interpretations=[
                interp
                for interp in example.interpretations
                if interp.interp_id != intended_interpretation.interp_id
            ],
            clarifying_question=output.message,
            options=output.options,
        )

        sim_reply = simulated_user.reply(sim_input)

        if sim_reply.leakage_flag:
            leakage_events += 1

        trace["turns"].append(
            {
                "clarifying_question": output.message,
                "options": output.options,
                "user_reply": sim_reply.text,
                "chosen_option": sim_reply.chosen_option_index,
            }
        )

        # Method answers after clarification
        # TODO: This needs conversation state management
        # For now, just call decide again (ponytail: simplest path)
        output = method.decide(query)

        # Accumulate cost
        total_cost = CostRecord(
            llm_calls=total_cost.llm_calls + output.cost.llm_calls,
            samples=total_cost.samples + output.cost.samples,
            prompt_tokens=total_cost.prompt_tokens + output.cost.prompt_tokens,
            completion_tokens=total_cost.completion_tokens + output.cost.completion_tokens,
            nli_forward_passes=total_cost.nli_forward_passes + output.cost.nli_forward_passes,
        )

    # Final action
    if output.action == "ANSWER":
        final_action = "ANSWER"
        final_answer = output.message

        # Score answer
        em, f1 = score_answer(
            final_answer,
            intended_interpretation.answers,
        )
    else:
        # ABSTAIN or max turns reached
        final_action = "ABSTAIN"
        final_answer = None
        em, f1 = 0.0, 0.0

    return E2EResult(
        example_id=example.example_id,
        intended_interp_id=intended_interpretation.interp_id,
        final_action=final_action,
        final_answer=final_answer,
        exact_match=em,
        token_f1=f1,
        num_turns=num_turns,
        total_cost=total_cost,
        leakage_events=leakage_events,
        trace=trace,
    )


__all__ = [
    "E2EResult",
    "run_e2e_evaluation",
]
