"""Tests for conversation manager."""

import pytest
from adg.conversation.manager import ConversationManager
from adg.core.types import Action, Query
from adg.llm.mock_backend import MockLLMBackend
from adg.nli.mock_nli import MockNLIScorer


@pytest.fixture
def mock_llm():
    """Create mock LLM backend."""
    return MockLLMBackend()


@pytest.fixture
def mock_nli():
    """Create mock NLI scorer."""
    return MockNLIScorer()


@pytest.fixture
def conversation_manager(mock_llm, mock_nli):
    """Create conversation manager with mocks."""
    return ConversationManager(
        llm_backend=mock_llm,
        nli_scorer=mock_nli,
        n_samples=5,  # Small for testing
    )


class TestConversationManagerInit:
    """Tests for ConversationManager initialization."""

    def test_init_with_all_dependencies(self, mock_llm, mock_nli):
        """Test initialization with all dependencies."""
        manager = ConversationManager(
            llm_backend=mock_llm,
            nli_scorer=mock_nli,
        )

        assert manager.sampler is not None
        assert manager.sampler.backend is mock_llm
        assert manager.nli_scorer is mock_nli
        assert manager.decision_engine is not None
        assert manager.question_generator is not None
        assert manager.reply_resolver is not None

    def test_init_with_custom_params(self, mock_llm, mock_nli):
        """Test initialization with custom parameters."""
        manager = ConversationManager(
            llm_backend=mock_llm,
            nli_scorer=mock_nli,
            n_samples=20,
            significance_rho=0.3,
            clustering_algorithm="greedy_bidirectional",
        )

        assert manager.n_samples == 20
        assert manager.significance_rho == 0.3
        assert manager.clustering_algorithm == "greedy_bidirectional"


class TestStartSession:
    """Tests for starting sessions."""

    def test_start_session_basic(self, conversation_manager):
        """Test basic session creation."""
        session = conversation_manager.start_session("test_session")

        assert session.session_id == "test_session"
        assert session.state == "IDLE"
        assert len(session.turns) == 0

    def test_start_session_custom_ttl(self, conversation_manager):
        """Test session creation with custom TTL."""
        session = conversation_manager.start_session("test_session", ttl_seconds=7200)

        assert session.ttl_seconds == 7200


class TestProcessQuery:
    """Tests for processing queries."""

    def test_process_query_with_query_object(self, conversation_manager):
        """Test processing query with Query object."""
        session = conversation_manager.start_session("test_session")

        query = Query(query_id="q1", text="What is a python?")
        updated_session, response, decision = conversation_manager.process_query(session, query)

        assert updated_session.session_id == "test_session"
        assert len(updated_session.turns) == 1
        assert isinstance(response, str)
        assert decision.action in [Action.ANSWER, Action.ASK, Action.ABSTAIN]

    def test_process_query_with_string(self, conversation_manager):
        """Test processing query with string."""
        session = conversation_manager.start_session("test_session")

        updated_session, response, _decision = conversation_manager.process_query(
            session, "What is Python?"
        )

        assert len(updated_session.turns) == 1
        assert isinstance(response, str)

    def test_process_query_answer_action(self, conversation_manager):
        """Test processing query that leads to ANSWER."""
        session = conversation_manager.start_session("test_session")

        # Use a query that should produce consensus
        query = Query(query_id="q1", text="What is 2+2?")
        updated_session, response, decision = conversation_manager.process_query(session, query)

        # With mock backend, behavior is deterministic
        assert decision.action in [Action.ANSWER, Action.ASK, Action.ABSTAIN]

        if decision.action == Action.ANSWER:
            assert updated_session.state == "ANSWERED"
            assert len(response) > 0

    def test_process_query_ask_action(self, conversation_manager):
        """Test processing query that leads to ASK."""
        session = conversation_manager.start_session("test_session")

        # Mock backend should sometimes produce divergent answers
        query = Query(query_id="q1", text="Python")
        updated_session, response, decision = conversation_manager.process_query(session, query)

        if decision.action == Action.ASK:
            assert updated_session.state == "AWAITING_CLARIFICATION"
            assert updated_session.pending_clarification is not None
            assert updated_session.pending_clusters is not None
            assert "1." in response or "2." in response  # Numbered options

    def test_process_query_abstain_action(self, conversation_manager):
        """Test processing query that leads to ABSTAIN."""
        session = conversation_manager.start_session("test_session")

        query = Query(query_id="q1", text="Test query")
        updated_session, response, decision = conversation_manager.process_query(session, query)

        if decision.action == Action.ABSTAIN:
            assert updated_session.state == "ABSTAINED"
            assert "not confident" in response.lower() or "cannot" in response.lower()

    def test_process_query_turn_added(self, conversation_manager):
        """Test that query adds turn to session."""
        session = conversation_manager.start_session("test_session")

        query = Query(query_id="q1", text="What is Python?")
        updated_session, response, decision = conversation_manager.process_query(session, query)

        assert len(updated_session.turns) == 1
        turn = updated_session.turns[0]
        assert turn.user_message == "What is Python?"
        assert turn.system_response == response
        assert turn.action == decision.action


class TestProcessClarificationReply:
    """Tests for processing clarification replies."""

    def test_process_clarification_reply_valid_selection(self, conversation_manager):
        """Test processing valid clarification reply."""
        # Create session with pending clarification
        session = conversation_manager.start_session("test_session")

        # First, generate a query that requires clarification
        query = Query(query_id="q1", text="Python")
        session, _, decision = conversation_manager.process_query(session, query)

        # If we got ASK action, test clarification reply
        if decision.action == Action.ASK:
            # Reply with numeric selection
            updated_session, _response = conversation_manager.process_clarification_reply(
                session, "1"
            )

            assert updated_session.state == "ANSWERED"
            assert updated_session.pending_clarification is None
            assert len(updated_session.turns) == 2  # Original + clarification reply

    def test_process_clarification_reply_invalid_state(self, conversation_manager):
        """Test that processing clarification requires correct state."""
        session = conversation_manager.start_session("test_session")

        # Try to process clarification when not awaiting
        with pytest.raises(ValueError, match="not awaiting clarification"):
            conversation_manager.process_clarification_reply(session, "1")

    def test_process_clarification_reply_unresolved(self, conversation_manager):
        """Test processing unresolvable clarification reply."""
        session = conversation_manager.start_session("test_session")

        # Generate query with clarification
        query = Query(query_id="q1", text="Python")
        session, _, decision = conversation_manager.process_query(session, query)

        if decision.action == Action.ASK:
            # Reply with invalid selection
            _updated_session, response = conversation_manager.process_clarification_reply(
                session, "invalid choice xyz"
            )

            # Should handle gracefully
            assert isinstance(response, str)


class TestGetRepresentativeAnswer:
    """Tests for getting representative answer."""

    def test_get_representative_answer_with_clusters(self, conversation_manager):
        """Test getting answer when clusters exist."""
        from adg.core.types import Cluster, Sample, SamplingResult

        clusters = [
            Cluster(
                cluster_id=0,
                sample_ids=[0, 1],
                representative_answer="Answer A",
                size=2,
            ),
            Cluster(
                cluster_id=1,
                sample_ids=[2],
                representative_answer="Answer B",
                size=1,
            ),
        ]

        samples = [
            Sample(
                sample_id=i,
                query_id="q1",
                answer=f"Answer {i}",
                interpretation=f"Interp {i}",
                is_valid=True,
                is_refusal=False,
            )
            for i in range(3)
        ]

        sampling_result = SamplingResult(
            query_id="q1",
            samples=samples,
            n_samples=3,
            n_valid=3,
            n_refusals=0,
            n_invalid=0,
            seed=42,
        )

        answer = conversation_manager._get_representative_answer(clusters, sampling_result)

        assert answer == "Answer A"

    def test_get_representative_answer_empty_clusters(self, conversation_manager):
        """Test getting answer when no clusters."""
        from adg.core.types import Sample, SamplingResult

        samples = [
            Sample(
                sample_id=0,
                query_id="q1",
                answer="Answer",
                interpretation="Interp",
                is_valid=True,
                is_refusal=False,
            )
        ]

        sampling_result = SamplingResult(
            query_id="q1",
            samples=samples,
            n_samples=1,
            n_valid=1,
            n_refusals=0,
            n_invalid=0,
            seed=42,
        )

        answer = conversation_manager._get_representative_answer([], sampling_result)

        assert "couldn't generate" in answer.lower()


class TestMultiTurnWorkflow:
    """Integration tests for multi-turn conversations."""

    def test_full_answer_workflow(self, conversation_manager):
        """Test complete conversation with direct answer."""
        session = conversation_manager.start_session("multi_turn_1")

        query = Query(query_id="q1", text="What is 2+2?")
        session, _response, decision = conversation_manager.process_query(session, query)

        # Should complete in one turn if ANSWER
        if decision.action == Action.ANSWER:
            assert session.state == "ANSWERED"
            assert len(session.turns) == 1

    def test_full_clarification_workflow(self, conversation_manager):
        """Test complete conversation with clarification."""
        session = conversation_manager.start_session("multi_turn_2")

        # Initial query
        query = Query(query_id="q1", text="Python")
        session, _response, decision = conversation_manager.process_query(session, query)

        # If clarification is needed
        if decision.action == Action.ASK:
            # Provide clarification
            session, _final_response = conversation_manager.process_clarification_reply(
                session, "1"
            )

            assert session.state == "ANSWERED"
            assert len(session.turns) == 2

    def test_multiple_queries_same_session(self, conversation_manager):
        """Test multiple queries in same session."""
        session = conversation_manager.start_session("multi_query")

        # First query
        query1 = Query(query_id="q1", text="What is Python?")
        session, _, _ = conversation_manager.process_query(session, query1)
        turns_after_first = len(session.turns)

        # Second query (if state allows)
        if session.state != "AWAITING_CLARIFICATION":
            query2 = Query(query_id="q2", text="What is Java?")
            session, _, _ = conversation_manager.process_query(session, query2)

            assert len(session.turns) > turns_after_first
