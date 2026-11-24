"""Tests for leakage detection."""

from adg.data.schema import GoldInterpretation
from adg.simuser.leakage import check_leakage, normalize_for_leakage


def test_normalize_for_leakage() -> None:
    """Test text normalization for leakage detection."""
    text = "The capital is Paris, France!"
    normalized = normalize_for_leakage(text)

    assert normalized == "the capital is paris france"


def test_check_leakage_detects_answer() -> None:
    """Test leakage detection with gold answer."""
    interpretation = GoldInterpretation(
        interp_id="test_1",
        disambiguated_question="What is the capital of France?",
        answers=["Paris", "paris"],
    )

    reply = "I mean the country, and the answer is Paris."

    assert check_leakage(reply, interpretation) is True


def test_check_leakage_allows_safe_reply() -> None:
    """Test safe reply without answer leakage."""
    interpretation = GoldInterpretation(
        interp_id="test_1",
        disambiguated_question="What is the capital of France?",
        answers=["Paris"],
    )

    reply = "I mean the country in Europe."

    assert check_leakage(reply, interpretation) is False


def test_check_leakage_ignores_short_answers() -> None:
    """Test that very short answers don't trigger false positives."""
    interpretation = GoldInterpretation(
        interp_id="test_1",
        disambiguated_question="Is it true?",
        answers=["yes", "no"],
    )

    reply = "Yes, I mean the country."

    # "yes" is too short, should not trigger leakage
    assert check_leakage(reply, interpretation) is False
