"""Tests for method base classes."""

from adg.methods.base import BudgetSpec, MethodOutput


def test_budget_spec_creation() -> None:
    """Test BudgetSpec creation."""
    budget = BudgetSpec(
        llm_calls=1,
        samples_per_call=8,
        nli_pairs=28,
        description="Test budget",
    )

    assert budget.llm_calls == 1
    assert budget.samples_per_call == 8
    assert budget.nli_pairs == 28
    assert budget.description == "Test budget"


def test_method_output_creation() -> None:
    """Test MethodOutput creation."""
    from adg.core.types import CostRecord

    cost = CostRecord(
        llm_calls=1,
        samples=8,
        prompt_tokens=100,
        completion_tokens=50,
        nli_forward_passes=28,
    )

    output = MethodOutput(
        action="ANSWER",
        message="Paris",
        options=[],
        score=0.95,
        trace={"method": "test"},
        cost=cost,
    )

    assert output.action == "ANSWER"
    assert output.message == "Paris"
    assert output.score == 0.95
    assert output.trace["method"] == "test"


def test_method_output_defaults() -> None:
    """Test MethodOutput with default values."""
    from adg.core.types import CostRecord

    cost = CostRecord(
        llm_calls=1,
        samples=1,
        prompt_tokens=0,
        completion_tokens=0,
        nli_forward_passes=0,
    )

    output = MethodOutput(
        action="ANSWER",
        message="Test",
        cost=cost,
    )

    assert output.options == []
    assert output.score is None
    assert output.trace == {}
