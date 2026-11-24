"""Multi-turn conversation orchestration.

Per spec §4.11: Manage conversation flow with clarifications.
"""

from adg.clarify.generator import QuestionGenerator
from adg.clarify.interpretation import ReplyResolver
from adg.clustering.algorithms import cluster_samples
from adg.clustering.features import compute_aggregate_features
from adg.clustering.interpretation import compute_interpretation_divergence
from adg.clustering.relations import compute_pairwise_relations
from adg.conversation import state as conv_state
from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend, NLIScorer
from adg.core.types import (
    Action,
    Cluster,
    Decision,
    Query,
    SamplingResult,
    SessionState,
)
from adg.decision.engine import DecisionEngine
from adg.sampling.sampler import Sampler

log = get_logger(__name__)


class ConversationManager:
    """Manages multi-turn conversations with clarifications.

    Orchestrates the full ADG pipeline with conversation state management.
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        nli_scorer: NLIScorer,
        decision_engine: DecisionEngine | None = None,
        question_generator: QuestionGenerator | None = None,
        reply_resolver: ReplyResolver | None = None,
        n_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        significance_rho: float = 0.25,
        clustering_algorithm: str = "agglomerative",
    ):
        """Initialize conversation manager.

        Args:
            llm_backend: LLM backend for sampling
            nli_scorer: NLI scorer for clustering
            decision_engine: Decision engine (default: create new)
            question_generator: Question generator (default: create new)
            reply_resolver: Reply resolver (default: create new)
            n_samples: Number of samples to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens per sample
            significance_rho: Significance threshold for clustering
            clustering_algorithm: Clustering algorithm to use
        """
        self.sampler = Sampler(llm_backend)
        self.nli_scorer = nli_scorer
        self.decision_engine = decision_engine or DecisionEngine()
        self.question_generator = question_generator or QuestionGenerator()
        self.reply_resolver = reply_resolver or ReplyResolver()

        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.significance_rho = significance_rho
        self.clustering_algorithm = clustering_algorithm

        log.info(
            "conversation_manager_initialized",
            n_samples=n_samples,
            clustering_algorithm=clustering_algorithm,
        )

    def start_session(self, session_id: str, ttl_seconds: int = 3600) -> SessionState:
        """Start a new conversation session.

        Args:
            session_id: Unique session identifier
            ttl_seconds: Session TTL in seconds

        Returns:
            New SessionState
        """
        return conv_state.create_session(session_id, ttl_seconds)

    def process_query(
        self,
        session: SessionState,
        query: Query | str,
    ) -> tuple[SessionState, str, Decision]:
        """Process a user query and update session.

        Args:
            session: Current session state
            query: User query (Query object or string)

        Returns:
            (updated_session, response_text, decision)
        """
        # Convert string to Query if needed
        if isinstance(query, str):
            query = Query(query_id=f"{session.session_id}_q{len(session.turns)}", text=query)

        log.info(
            "processing_query",
            session_id=session.session_id,
            query_id=query.query_id,
        )

        # Sample answers
        sampling_result = self.sampler.sample(
            query=query,
            n_samples=self.n_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )

        # Compute pairwise relations
        relations = compute_pairwise_relations(
            samples=sampling_result.samples,
            nli_scorer=self.nli_scorer,
            query_text=query.text,
        )

        # Cluster samples
        clustering_result = cluster_samples(
            samples=sampling_result.samples,
            relations=relations,
            algorithm=self.clustering_algorithm,
        )

        # Compute interpretation divergence
        interp_divergence = compute_interpretation_divergence(
            samples=sampling_result.samples,
            clusters=clustering_result.clusters,
            nli_scorer=self.nli_scorer,
        )

        # Compute aggregate features
        features = compute_aggregate_features(
            sampling_result=sampling_result,
            clustering_result=clustering_result,
            relations=relations,
            significance_rho=self.significance_rho,
            interpretation_divergence=interp_divergence,
        )

        # Make decision
        decision = self.decision_engine.decide(features)

        log.info(
            "decision_made",
            session_id=session.session_id,
            action=decision.action,
            confidence=round(decision.confidence, 3),
        )

        # Handle action
        if decision.action == Action.ANSWER:
            # Return most common answer from largest cluster
            response_text = self._get_representative_answer(
                clustering_result.clusters,
                sampling_result,
            )
            session = conv_state.add_turn(session, query.text, response_text, Action.ANSWER)
            session = conv_state.set_answered(session)

        elif decision.action == Action.ASK:
            # Generate clarifying question
            clarifying_question = self.question_generator.generate(
                query_text=query.text,
                clusters=clustering_result.clusters,
                samples=sampling_result.samples,
            )
            response_text = clarifying_question.question_text
            session = conv_state.add_turn(session, query.text, response_text, Action.ASK)
            session = conv_state.set_awaiting_clarification(
                session, clarifying_question, clustering_result.clusters
            )

        else:  # ABSTAIN
            response_text = "I'm not confident enough to answer this question."
            session = conv_state.add_turn(session, query.text, response_text, Action.ABSTAIN)
            session = conv_state.set_abstained(session)

        return session, response_text, decision

    def process_clarification_reply(
        self,
        session: SessionState,
        reply: str,
    ) -> tuple[SessionState, str]:
        """Process user's reply to a clarifying question.

        Args:
            session: Current session state (must be AWAITING_CLARIFICATION)
            reply: User's reply text

        Returns:
            (updated_session, response_text)
        """
        if session.state != "AWAITING_CLARIFICATION":
            msg = f"Session not awaiting clarification (state: {session.state})"
            raise ValueError(msg)

        if session.pending_clarification is None:
            msg = "No pending clarification question"
            raise ValueError(msg)

        log.info(
            "processing_clarification_reply",
            session_id=session.session_id,
            reply=reply[:50],
        )

        # Resolve reply
        resolved = self.reply_resolver.resolve(reply, session.pending_clarification)

        # Get selected interpretation
        if resolved.option_index is not None:
            selected_cluster = session.pending_clusters[resolved.option_index]
            response_text = selected_cluster.representative_answer

            log.info(
                "clarification_resolved",
                session_id=session.session_id,
                selected_option=resolved.option_index,
                confidence=round(resolved.confidence, 3),
            )
        else:
            response_text = "I couldn't understand your selection. Could you try again?"
            log.warning(
                "clarification_unresolved",
                session_id=session.session_id,
                confidence=resolved.confidence,
            )

        # Update session
        session = conv_state.add_turn(session, reply, response_text, Action.ANSWER)
        session = conv_state.clear_pending_clarification(session)
        session = conv_state.set_answered(session)

        return session, response_text

    def _get_representative_answer(
        self,
        clusters: list[Cluster],
        sampling_result: SamplingResult,
    ) -> str:
        """Get representative answer from largest cluster.

        Args:
            clusters: Sorted clusters (largest first)
            sampling_result: Sampling result with samples

        Returns:
            Representative answer text
        """
        if not clusters:
            return "I couldn't generate a confident answer."

        # Return representative answer from largest cluster
        largest_cluster = clusters[0]
        return largest_cluster.representative_answer


__all__ = ["ConversationManager"]
