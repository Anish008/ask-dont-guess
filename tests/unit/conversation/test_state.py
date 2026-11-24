"""Tests for conversation state management."""

from datetime import datetime, timedelta

import pytest
from adg.conversation import state as conv_state
from adg.core.types import ClarifyingQuestion, Cluster, SessionState


@pytest.fixture
def sample_session() -> SessionState:
    """Create a sample session for testing."""
    return conv_state.create_session("test_session_123", ttl_seconds=3600)


@pytest.fixture
def sample_clarifying_question() -> ClarifyingQuestion:
    """Create a sample clarifying question."""
    return ClarifyingQuestion(
        question_text="Which did you mean?",
        options=["Option A", "Option B"],
        cluster_ids=[0, 1],
    )


@pytest.fixture
def sample_clusters() -> list[Cluster]:
    """Create sample clusters."""
    return [
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


class TestCreateSession:
    """Tests for session creation."""

    def test_create_session_basic(self):
        """Test basic session creation."""
        session = conv_state.create_session("session_1")

        assert session.session_id == "session_1"
        assert session.ttl_seconds == 3600  # Default
        assert session.state == "IDLE"
        assert len(session.turns) == 0
        assert session.pending_clarification is None
        assert session.pending_clusters is None

    def test_create_session_custom_ttl(self):
        """Test session creation with custom TTL."""
        session = conv_state.create_session("session_2", ttl_seconds=7200)

        assert session.session_id == "session_2"
        assert session.ttl_seconds == 7200

    def test_create_session_timestamps(self):
        """Test that timestamps are set on creation."""
        before = datetime.now()
        session = conv_state.create_session("session_3")
        after = datetime.now()

        assert before <= session.created_at <= after
        assert before <= session.last_updated <= after


class TestAddTurn:
    """Tests for adding turns."""

    def test_add_turn_first(self, sample_session: SessionState):
        """Test adding first turn."""
        session = conv_state.add_turn(
            sample_session,
            user_message="What is Python?",
            system_response="Python is a programming language",
            action="ANSWER",
        )

        assert len(session.turns) == 1
        turn = session.turns[0]
        assert turn.turn_id == 0
        assert turn.user_message == "What is Python?"
        assert turn.system_response == "Python is a programming language"
        assert turn.action == "ANSWER"

    def test_add_turn_multiple(self, sample_session: SessionState):
        """Test adding multiple turns."""
        session = sample_session

        session = conv_state.add_turn(session, "First query", "First response", "ANSWER")
        session = conv_state.add_turn(session, "Second query", "Second response", "ASK")

        assert len(session.turns) == 2
        assert session.turns[0].turn_id == 0
        assert session.turns[1].turn_id == 1
        assert session.turns[0].action == "ANSWER"
        assert session.turns[1].action == "ASK"

    def test_add_turn_updates_timestamp(self, sample_session: SessionState):
        """Test that adding turn updates last_updated."""
        original_time = sample_session.last_updated

        session = conv_state.add_turn(sample_session, "Query", "Response", "ANSWER")

        assert session.last_updated >= original_time


class TestSetAwaitingClarification:
    """Tests for setting clarification state."""

    def test_set_awaiting_clarification(
        self,
        sample_session: SessionState,
        sample_clarifying_question: ClarifyingQuestion,
        sample_clusters: list[Cluster],
    ):
        """Test setting session to await clarification."""
        session = conv_state.set_awaiting_clarification(
            sample_session,
            sample_clarifying_question,
            sample_clusters,
        )

        assert session.state == "AWAITING_CLARIFICATION"
        assert session.pending_clarification == sample_clarifying_question
        assert session.pending_clusters == sample_clusters

    def test_set_awaiting_clarification_updates_timestamp(
        self,
        sample_session: SessionState,
        sample_clarifying_question: ClarifyingQuestion,
        sample_clusters: list[Cluster],
    ):
        """Test that setting clarification updates timestamp."""
        original_time = sample_session.last_updated

        session = conv_state.set_awaiting_clarification(
            sample_session,
            sample_clarifying_question,
            sample_clusters,
        )

        assert session.last_updated >= original_time


class TestClearPendingClarification:
    """Tests for clearing clarification."""

    def test_clear_pending_clarification(
        self,
        sample_session: SessionState,
        sample_clarifying_question: ClarifyingQuestion,
        sample_clusters: list[Cluster],
    ):
        """Test clearing pending clarification."""
        # First set clarification
        session = conv_state.set_awaiting_clarification(
            sample_session,
            sample_clarifying_question,
            sample_clusters,
        )

        # Then clear it
        session = conv_state.clear_pending_clarification(session)

        assert session.state == "IDLE"
        assert session.pending_clarification is None
        assert session.pending_clusters is None

    def test_clear_updates_timestamp(
        self,
        sample_session: SessionState,
        sample_clarifying_question: ClarifyingQuestion,
        sample_clusters: list[Cluster],
    ):
        """Test that clearing updates timestamp."""
        session = conv_state.set_awaiting_clarification(
            sample_session,
            sample_clarifying_question,
            sample_clusters,
        )
        time_before_clear = session.last_updated

        session = conv_state.clear_pending_clarification(session)

        assert session.last_updated >= time_before_clear


class TestSetAnswered:
    """Tests for setting answered state."""

    def test_set_answered(self, sample_session: SessionState):
        """Test setting session to answered."""
        session = conv_state.set_answered(sample_session)

        assert session.state == "ANSWERED"

    def test_set_answered_updates_timestamp(self, sample_session: SessionState):
        """Test that setting answered updates timestamp."""
        original_time = sample_session.last_updated

        session = conv_state.set_answered(sample_session)

        assert session.last_updated >= original_time


class TestSetAbstained:
    """Tests for setting abstained state."""

    def test_set_abstained(self, sample_session: SessionState):
        """Test setting session to abstained."""
        session = conv_state.set_abstained(sample_session)

        assert session.state == "ABSTAINED"

    def test_set_abstained_updates_timestamp(self, sample_session: SessionState):
        """Test that setting abstained updates timestamp."""
        original_time = sample_session.last_updated

        session = conv_state.set_abstained(sample_session)

        assert session.last_updated >= original_time


class TestIsExpired:
    """Tests for session expiration."""

    def test_is_expired_fresh_session(self, sample_session: SessionState):
        """Test that fresh session is not expired."""
        assert not conv_state.is_expired(sample_session)

    def test_is_expired_old_session(self, sample_session: SessionState):
        """Test that old session is expired."""
        # Manually set last_updated to past
        sample_session.last_updated = datetime.now() - timedelta(seconds=7200)

        assert conv_state.is_expired(sample_session)

    def test_is_expired_boundary(self, sample_session: SessionState):
        """Test expiration at exact TTL boundary."""
        # Set to exactly TTL seconds ago
        sample_session.last_updated = datetime.now() - timedelta(seconds=sample_session.ttl_seconds)

        # Should be expired (elapsed >= ttl)
        assert conv_state.is_expired(sample_session)

    def test_is_expired_custom_ttl(self):
        """Test expiration with custom TTL."""
        session = conv_state.create_session("session_x", ttl_seconds=10)

        # Fresh
        assert not conv_state.is_expired(session)

        # Set to 15 seconds ago (> 10 second TTL)
        session.last_updated = datetime.now() - timedelta(seconds=15)
        assert conv_state.is_expired(session)


class TestSessionWorkflow:
    """Integration tests for complete workflows."""

    def test_answer_workflow(self, sample_session: SessionState):
        """Test complete answer workflow."""
        session = sample_session

        # Add query turn
        session = conv_state.add_turn(
            session,
            "What is Python?",
            "Python is a programming language",
            "ANSWER",
        )

        # Set answered
        session = conv_state.set_answered(session)

        assert len(session.turns) == 1
        assert session.state == "ANSWERED"
        assert session.turns[0].action == "ANSWER"

    def test_clarification_workflow(
        self,
        sample_session: SessionState,
        sample_clarifying_question: ClarifyingQuestion,
        sample_clusters: list[Cluster],
    ):
        """Test complete clarification workflow."""
        session = sample_session

        # Add initial query turn (ASK)
        session = conv_state.add_turn(
            session,
            "Python?",
            "Which did you mean?",
            "ASK",
        )

        # Set awaiting clarification
        session = conv_state.set_awaiting_clarification(
            session,
            sample_clarifying_question,
            sample_clusters,
        )

        assert session.state == "AWAITING_CLARIFICATION"
        assert session.pending_clarification is not None

        # Add clarification reply turn
        session = conv_state.add_turn(
            session,
            "Option A",
            "Answer A",
            "ANSWER",
        )

        # Clear clarification
        session = conv_state.clear_pending_clarification(session)
        session = conv_state.set_answered(session)

        assert len(session.turns) == 2
        assert session.state == "ANSWERED"
        assert session.pending_clarification is None

    def test_abstain_workflow(self, sample_session: SessionState):
        """Test complete abstain workflow."""
        session = sample_session

        # Add query turn
        session = conv_state.add_turn(
            session,
            "Unclear query",
            "I cannot answer this",
            "ABSTAIN",
        )

        # Set abstained
        session = conv_state.set_abstained(session)

        assert len(session.turns) == 1
        assert session.state == "ABSTAINED"
        assert session.turns[0].action == "ABSTAIN"
