"""Integration tests for full ADG pipeline.

Tests the complete ANSWER/ASK/ABSTAIN flow with mock backends.
"""

import pytest
from adg.conversation.manager import ConversationManager
from adg.core.types import Query
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer


@pytest.fixture
def mock_manager() -> ConversationManager:
    """Create ConversationManager with mock backends."""
    llm = MockLLMBackend()
    nli = MockNLIScorer()
    return ConversationManager(
        llm_backend=llm,
        nli_scorer=nli,
        n_samples=8,
        temperature=0.7,
        top_p=0.95,
    )


def test_clear_question_answers(mock_manager: ConversationManager) -> None:
    """Test clear question produces a valid action."""
    session = mock_manager.start_session("test_clear")

    query = Query(
        query_id="clear_1",
        text="What is the capital of France?",
    )

    session, response, decision = mock_manager.process_query(session, query)

    # Should produce valid action (mock is deterministic, actual action varies)
    assert decision.action in ["ANSWER", "ASK", "ABSTAIN"]
    assert 0.0 <= decision.confidence <= 1.0
    assert response != ""
    assert session.state in ["IDLE", "AWAITING_CLARIFICATION", "ANSWERED", "ABSTAINED"]


def test_ambiguous_question_produces_valid_action(mock_manager: ConversationManager) -> None:
    """Test ambiguous question produces valid action."""
    session = mock_manager.start_session("test_ambiguous")

    query = Query(
        query_id="ambiguous_1",
        text="What is the capital of Georgia?",
    )

    session, response, decision = mock_manager.process_query(session, query)

    # Should produce valid action
    assert decision.action in ["ANSWER", "ASK", "ABSTAIN"]
    assert response != ""

    # If ASK, should have clarification options
    if decision.action == "ASK":
        assert session.state == "AWAITING_CLARIFICATION"
        assert session.pending_clarification is not None
        assert len(session.pending_clarification.options) >= 2


def test_unknown_question_produces_valid_action(mock_manager: ConversationManager) -> None:
    """Test unanswerable question produces valid action."""
    session = mock_manager.start_session("test_unknown")

    query = Query(
        query_id="unknown_1",
        text="What is the most popular flavor of ice cream in Antarctica?",
    )

    session, response, decision = mock_manager.process_query(session, query)

    # Should produce valid action
    assert decision.action in ["ANSWER", "ASK", "ABSTAIN"]
    assert response != ""


def test_clarification_resolution(mock_manager: ConversationManager) -> None:
    """Test multi-turn conversation flow."""
    session = mock_manager.start_session("test_resolution")

    # First turn
    query = Query(
        query_id="ambiguous_2",
        text="What is the capital of Georgia?",
    )
    session, _response1, decision1 = mock_manager.process_query(session, query)

    # If ASK, test clarification reply
    if decision1.action == "ASK":
        reply = "the country"
        session, _response2 = mock_manager.process_clarification_reply(session, reply)
        # Should have resolved
        assert len(session.turns) == 2
    else:
        # For ANSWER/ABSTAIN, just one turn
        assert len(session.turns) == 1


def test_deterministic_outcomes(mock_manager: ConversationManager) -> None:
    """Test that same query produces deterministic outcomes with mocks."""
    query = Query(query_id="deterministic_1", text="Test query")

    # Run twice
    session1 = mock_manager.start_session("det_1")
    _, _response1, decision1 = mock_manager.process_query(session1, query)

    session2 = mock_manager.start_session("det_2")
    _, _response2, decision2 = mock_manager.process_query(session2, query)

    # Same action (mock backend is deterministic by query hash)
    assert decision1.action == decision2.action


def test_session_state_tracking(mock_manager: ConversationManager) -> None:
    """Test that session state is properly tracked through turns."""
    session = mock_manager.start_session("test_state", ttl_seconds=3600)

    assert session.session_id == "test_state"
    assert len(session.turns) == 0
    assert session.state == "IDLE"

    query = Query(query_id="state_1", text="What is 2+2?")
    session, _, _decision = mock_manager.process_query(session, query)

    # Should have one turn now
    assert len(session.turns) == 1
    assert session.state in ["IDLE", "AWAITING_CLARIFICATION", "ANSWERED", "ABSTAINED"]
