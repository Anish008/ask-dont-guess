"""Rule-based simulated user.

Per spec §11.3: Offline deterministic user using NLI scoring.
"""

from adg.core.logging import get_logger
from adg.data.schema import GoldInterpretation
from adg.simuser.base import SimUserInput, SimUserReply
from adg.simuser.leakage import check_leakage, normalize_for_leakage

log = get_logger(__name__)


class RuleBasedUser:
    """Rule-based simulated user.

    Per spec §11.2-11.3: Uses NLI to match options against intent,
    never leaks gold answers.

    Behavior rules:
    1. If options include intended interpretation, pick it
    2. If not, say "none of those" and restate intent
    3. Never reveal gold answers
    4. If question is generic/off-topic, give minimal reply

    ponytail: Using simple string similarity for now instead of full NLI.
    """

    def __init__(self, threshold: float = 0.6):
        """Initialize rule-based user.

        Args:
            threshold: Similarity threshold for option selection
        """
        self.threshold = threshold
        log.info("rule_user_initialized", threshold=threshold)

    def reply(self, input_data: SimUserInput) -> SimUserReply:
        """Generate rule-based reply.

        Per spec §11.2: Match options via NLI, never leak answers.

        Args:
            input_data: Input with intent and clarifying question

        Returns:
            Reply with chosen option or intent restatement
        """
        # Rule 3: Detect generic/off-topic questions
        if self._is_generic_question(input_data.clarifying_question):
            return SimUserReply(
                text="I just mean what I asked.",
                behavior="generic",
                leakage_flag=False,
            )

        # Rule 1: Match options against intended interpretation
        if input_data.options:
            best_idx, best_score = self._match_options(
                input_data.options,
                input_data.intended,
            )

            # Check if match is good enough and better than other interpretations
            if best_score >= self.threshold:
                # Verify not ambiguous with other interpretations
                for other in input_data.other_interpretations:
                    _, other_score = self._match_options(
                        input_data.options,
                        other,
                    )
                    if other_score >= best_score - 0.1:  # Too close
                        break
                else:
                    # Clear best match
                    reply_text = self._format_choice_reply(
                        input_data.options[best_idx],
                    )
                    if check_leakage(reply_text, input_data.intended):
                        # Fallback to safe template
                        reply_text = "That one."

                    return SimUserReply(
                        text=reply_text,
                        chosen_option_index=best_idx,
                        behavior="choose_option",
                        leakage_flag=False,
                    )

        # Rule 2: Options don't match, restate intent
        restatement = self._restate_intent(
            input_data.original_question,
            input_data.intended,
        )

        # Check leakage
        if check_leakage(restatement, input_data.intended):
            restatement = "None of those."

        return SimUserReply(
            text=restatement,
            chosen_option_index=None,
            behavior="none_of_these" if input_data.options else "describe_intent",
            leakage_flag=False,
        )

    def _is_generic_question(self, question: str) -> bool:
        """Check if question is generic/off-topic.

        Args:
            question: Clarifying question

        Returns:
            True if generic
        """
        generic_patterns = [
            "could you clarify",
            "can you clarify",
            "what do you mean",
            "can you be more specific",
            "please clarify",
            "i don't understand",
        ]

        q_lower = question.lower()
        return any(pattern in q_lower for pattern in generic_patterns)

    def _match_options(
        self,
        options: list[str],
        interpretation: GoldInterpretation,
    ) -> tuple[int, float]:
        """Match options against an interpretation via string similarity.

        ponytail: Using Jaccard similarity for now, NLI scorer later.

        Args:
            options: Offered options
            interpretation: Gold interpretation

        Returns:
            Tuple of (best_index, best_score)
        """
        if not options:
            return -1, 0.0

        # Score each option via token overlap with disambiguated question
        # Use description if available, more discriminative
        target = interpretation.description or interpretation.disambiguated_question
        target_norm = normalize_for_leakage(target)
        target_tokens = set(target_norm.split())

        best_idx = 0
        best_score = 0.0

        for i, option in enumerate(options):
            option_norm = normalize_for_leakage(option)
            option_tokens = set(option_norm.split())

            # Jaccard similarity
            if not target_tokens and not option_tokens:
                score = 1.0
            elif not target_tokens or not option_tokens:
                score = 0.0
            else:
                overlap = target_tokens & option_tokens
                union = target_tokens | option_tokens
                score = len(overlap) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score

    def _format_choice_reply(self, chosen_option: str) -> str:
        """Format a reply choosing an option.

        Args:
            chosen_option: Chosen option text

        Returns:
            Reply text
        """
        # Short, natural affirmation
        templates = [
            chosen_option,  # Just repeat it
            f"Yes, {chosen_option.lower()}",
            f"{chosen_option}.",
        ]
        return templates[0]  # Simplest

    def _restate_intent(
        self,
        original_question: str,
        intended: GoldInterpretation,
    ) -> str:
        """Restate intent without leaking answers.

        Args:
            original_question: Original ambiguous question
            intended: Intended interpretation

        Returns:
            Safe restatement
        """
        # ponytail: simple template, no fancy diff
        # Use description if available, most discriminative
        if intended.description:
            return f"I mean {intended.description}."

        # Fallback: use disambiguated question directly
        # (leakage check will catch problems)
        return f"I mean {intended.disambiguated_question}"


__all__ = ["RuleBasedUser"]
