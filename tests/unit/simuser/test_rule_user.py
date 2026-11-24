"""Tests for rule-based simulated user."""

import pytest
from adg.data.schema import GoldInterpretation
from adg.simuser.base import SimUserInput
from adg.simuser.rule_user import RuleBasedUser


@pytest.fixture
def rule_user() -> RuleBasedUser:
    """Create rule-based user."""
    # ponytail: Lower threshold for simple string matching
    return RuleBasedUser(threshold=0.3)


def test_rule_user_chooses_matching_option(rule_user: RuleBasedUser) -> None:
    """Test user chooses option that matches intent."""
    intended = GoldInterpretation(
        interp_id="georgia_country",
        disambiguated_question="What is the capital of Georgia the country?",
        answers=["Tbilisi"],
        description="the country in the Caucasus",
    )

    input_data = SimUserInput(
        original_question="What is the capital of Georgia?",
        intended=intended,
        other_interpretations=[],
        clarifying_question="Do you mean Georgia the US state or Georgia the country?",
        options=["The US state", "The country"],
    )

    reply = rule_user.reply(input_data)

    assert reply.behavior == "choose_option"
    assert reply.chosen_option_index is not None
    assert not reply.leakage_flag


def test_rule_user_handles_generic_question(rule_user: RuleBasedUser) -> None:
    """Test user gives minimal reply to generic question."""
    intended = GoldInterpretation(
        interp_id="test_1",
        disambiguated_question="What is the capital of France?",
        answers=["Paris"],
    )

    input_data = SimUserInput(
        original_question="What is the capital of France?",
        intended=intended,
        other_interpretations=[],
        clarifying_question="Could you clarify your question?",
        options=[],
    )

    reply = rule_user.reply(input_data)

    assert reply.behavior == "generic"
    assert reply.text == "I just mean what I asked."


def test_rule_user_restates_intent_when_no_match(
    rule_user: RuleBasedUser,
) -> None:
    """Test user restates intent when options don't match."""
    intended = GoldInterpretation(
        interp_id="georgia_country",
        disambiguated_question="What is the capital of Georgia the country?",
        answers=["Tbilisi"],
        description="the country in the Caucasus",
    )

    input_data = SimUserInput(
        original_question="What is the capital of Georgia?",
        intended=intended,
        other_interpretations=[],
        clarifying_question="Do you mean the city or the state?",
        options=["The city", "The state"],
    )

    reply = rule_user.reply(input_data)

    assert reply.behavior == "none_of_these"
    assert reply.chosen_option_index is None
    # Should not leak answer
    assert "tbilisi" not in reply.text.lower()
