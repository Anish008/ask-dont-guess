"""Tests for AlwaysAnswer baseline."""

import pytest
from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.methods.always_answer import AlwaysAnswer


@pytest.fixture
def method() -> AlwaysAnswer:
    """Create AlwaysAnswer method with mock backend."""
    llm = MockLLMBackend()
    return AlwaysAnswer(llm)


def test_always_returns_answer(method: AlwaysAnswer) -> None:
    """Test that AlwaysAnswer always returns ANSWER action."""
    query = Query(query_id="test_1", text="What is 2+2?")
    output = method.decide(query)

    assert output.action == "ANSWER"
    assert output.message != ""
    assert output.options == []


def test_budget_spec(method: AlwaysAnswer) -> None:
    """Test budget specification."""
    budget = method.budget()

    assert budget.llm_calls == 1
    assert budget.samples_per_call == 1
    assert budget.nli_pairs == 0


def test_method_name(method: AlwaysAnswer) -> None:
    """Test method name."""
    assert method.name == "always_answer"
