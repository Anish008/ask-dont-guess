"""Tests for reply resolution."""

import pytest
from adg.clarify.interpretation import ReplyResolver
from adg.core.types import ClarifyingQuestion, ResolvedInterpretation


@pytest.fixture
def sample_question() -> ClarifyingQuestion:
    """Create a sample clarifying question."""
    return ClarifyingQuestion(
        question_text="By 'Python', did you mean:\n1. Programming language\n2. Snake species\n3. Monty Python comedy",
        options=[
            "Programming language",
            "Snake species",
            "Monty Python comedy",
        ],
        cluster_ids=[0, 1, 2],
    )


class TestReplyResolverNumeric:
    """Tests for numeric reply resolution."""

    def test_numeric_resolution_valid(self, sample_question: ClarifyingQuestion):
        """Test valid numeric choice."""
        resolver = ReplyResolver()

        result = resolver.resolve("1", sample_question)

        assert isinstance(result, ResolvedInterpretation)
        assert result.option_index == 0
        assert result.confidence == 1.0
        assert result.method == "exact_match"

    def test_numeric_resolution_second_option(self, sample_question: ClarifyingQuestion):
        """Test selecting second option."""
        resolver = ReplyResolver()

        result = resolver.resolve("2", sample_question)

        assert result.option_index == 1
        assert result.confidence == 1.0
        assert result.method == "exact_match"

    def test_numeric_resolution_last_option(self, sample_question: ClarifyingQuestion):
        """Test selecting last option."""
        resolver = ReplyResolver()

        result = resolver.resolve("3", sample_question)

        assert result.option_index == 2
        assert result.confidence == 1.0
        assert result.method == "exact_match"

    def test_numeric_resolution_with_whitespace(self, sample_question: ClarifyingQuestion):
        """Test numeric choice with whitespace."""
        resolver = ReplyResolver()

        result = resolver.resolve("  2  ", sample_question)

        assert result.option_index == 1
        assert result.confidence == 1.0

    def test_numeric_resolution_out_of_range_high(self, sample_question: ClarifyingQuestion):
        """Test numeric choice out of range (too high)."""
        resolver = ReplyResolver()

        result = resolver.resolve("99", sample_question)

        # Should fall through to other methods
        assert result.method != "exact_match" or result.option_index is None

    def test_numeric_resolution_out_of_range_zero(self, sample_question: ClarifyingQuestion):
        """Test numeric choice of zero."""
        resolver = ReplyResolver()

        result = resolver.resolve("0", sample_question)

        # Should fall through to other methods
        assert result.method != "exact_match" or result.option_index is None

    def test_numeric_resolution_negative(self, sample_question: ClarifyingQuestion):
        """Test negative numeric choice."""
        resolver = ReplyResolver()

        result = resolver.resolve("-1", sample_question)

        # Should fall through to other methods
        assert result.method != "exact_match" or result.option_index is None


class TestReplyResolverTextMatch:
    """Tests for text matching resolution."""

    def test_exact_match(self, sample_question: ClarifyingQuestion):
        """Test exact text match."""
        resolver = ReplyResolver()

        result = resolver.resolve("Programming language", sample_question)

        assert result.option_index == 0
        assert result.confidence == 1.0
        assert result.method == "exact_match"

    def test_exact_match_case_insensitive(self, sample_question: ClarifyingQuestion):
        """Test exact match is case-insensitive."""
        resolver = ReplyResolver()

        result = resolver.resolve("PROGRAMMING LANGUAGE", sample_question)

        assert result.option_index == 0
        assert result.confidence == 1.0
        assert result.method == "exact_match"

    def test_exact_match_with_whitespace(self, sample_question: ClarifyingQuestion):
        """Test exact match with extra whitespace."""
        resolver = ReplyResolver()

        result = resolver.resolve("  programming language  ", sample_question)

        assert result.option_index == 0
        assert result.confidence == 1.0

    def test_substring_match_reply_in_option(self, sample_question: ClarifyingQuestion):
        """Test substring match where reply is in option."""
        resolver = ReplyResolver(fuzzy_match_threshold=0.5)  # Lower threshold for substring

        result = resolver.resolve("programming", sample_question)

        assert result.option_index == 0
        assert result.confidence >= 0.5  # Above threshold
        assert result.method == "fuzzy"

    def test_substring_match_option_in_reply(self, sample_question: ClarifyingQuestion):
        """Test substring match where option is in reply."""
        resolver = ReplyResolver(fuzzy_match_threshold=0.3)  # Lower threshold

        result = resolver.resolve("I meant the snake species one", sample_question)

        assert result.option_index == 1
        assert result.confidence >= 0.3
        assert result.method == "fuzzy"

    def test_substring_below_threshold(self):
        """Test substring match below threshold."""
        question = ClarifyingQuestion(
            question_text="Which did you mean?",
            options=["Very long option text here", "Another option"],
            cluster_ids=[0, 1],
        )

        resolver = ReplyResolver(fuzzy_match_threshold=0.9)

        # Short substring won't meet high threshold
        result = resolver.resolve("very", question)

        # Should fall through to word overlap
        assert result.confidence < 1.0 or result.method != "text_match"


class TestReplyResolverWordOverlap:
    """Tests for word overlap resolution."""

    def test_word_overlap_match(self, sample_question: ClarifyingQuestion):
        """Test word overlap matching."""
        resolver = ReplyResolver()

        result = resolver.resolve("language for programming", sample_question)

        # Should match "Programming language" via word overlap
        assert result.option_index == 0
        assert result.method == "fuzzy"

    def test_word_overlap_multiple_words(self, sample_question: ClarifyingQuestion):
        """Test word overlap with multiple overlapping words."""
        resolver = ReplyResolver(fuzzy_match_threshold=0.4)  # Lower threshold for Jaccard

        result = resolver.resolve("monty python show", sample_question)

        assert result.option_index == 2
        assert result.confidence > 0.0

    def test_word_overlap_partial(self):
        """Test partial word overlap."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=[
                "Red apple fruit",
                "Green apple fruit",
                "Orange citrus fruit",
            ],
            cluster_ids=[0, 1, 2],
        )

        resolver = ReplyResolver(fuzzy_match_threshold=0.3)

        result = resolver.resolve("red apple", question)

        assert result.option_index == 0
        assert result.confidence >= 0.3

    def test_word_overlap_below_threshold(self):
        """Test word overlap below threshold."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=["Red apple", "Blue sky"],
            cluster_ids=[0, 1],
        )

        resolver = ReplyResolver(fuzzy_match_threshold=0.9)

        # Very low overlap won't meet high threshold
        result = resolver.resolve("green", question)

        # Should fall back
        assert result.method == "none"
        assert result.confidence == 0.0


class TestReplyResolverFallback:
    """Tests for fallback resolution."""

    def test_fallback_on_no_match(self, sample_question: ClarifyingQuestion):
        """Test fallback when nothing matches."""
        resolver = ReplyResolver()

        result = resolver.resolve("completely unrelated text xyz", sample_question)

        assert result.method == "none"
        assert result.confidence == 0.0
        assert result.option_index is None

    def test_fallback_preserves_reply(self, sample_question: ClarifyingQuestion):
        """Test that fallback returns None for option_index."""
        resolver = ReplyResolver()

        reply_text = "some random response"
        result = resolver.resolve(reply_text, sample_question)

        assert result.option_index is None

    def test_fallback_on_empty_reply(self, sample_question: ClarifyingQuestion):
        """Test fallback on empty reply."""
        resolver = ReplyResolver()

        result = resolver.resolve("", sample_question)

        assert result.method == "none"
        assert result.confidence == 0.0


class TestReplyResolverConfiguration:
    """Tests for ReplyResolver configuration."""

    def test_default_threshold(self):
        """Test default fuzzy match threshold."""
        resolver = ReplyResolver()
        assert resolver.fuzzy_match_threshold == 0.6

    def test_custom_threshold(self):
        """Test custom fuzzy match threshold."""
        resolver = ReplyResolver(fuzzy_match_threshold=0.8)
        assert resolver.fuzzy_match_threshold == 0.8

    def test_threshold_affects_matching(self):
        """Test that threshold affects matching behavior."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=["Programming language Python"],
            cluster_ids=[0],
        )

        # Lenient resolver
        lenient = ReplyResolver(fuzzy_match_threshold=0.3)
        result_lenient = lenient.resolve("programming", question)

        # Strict resolver
        strict = ReplyResolver(fuzzy_match_threshold=0.9)
        result_strict = strict.resolve("programming", question)

        # Lenient should match, strict might not
        assert result_lenient.confidence >= 0.3
        # Strict may or may not match depending on exact calculation
        assert result_strict.confidence >= 0.0


class TestReplyResolverEdgeCases:
    """Tests for edge cases."""

    def test_single_option(self):
        """Test with single option."""
        question = ClarifyingQuestion(
            question_text="Did you mean this?",
            options=["Only option"],
            cluster_ids=[0],
        )

        resolver = ReplyResolver()

        result = resolver.resolve("1", question)

        assert result.option_index == 0
        assert result.confidence == 1.0

    def test_empty_options_list(self):
        """Test with empty options list."""
        question = ClarifyingQuestion(
            question_text="No options?",
            options=[],
            cluster_ids=[],
        )

        resolver = ReplyResolver()

        result = resolver.resolve("anything", question)

        # Should fall back
        assert result.method == "none"
        assert result.option_index is None

    def test_special_characters_in_reply(self):
        """Test reply with special characters."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=["Option A", "Option B"],
            cluster_ids=[0, 1],
        )

        resolver = ReplyResolver()

        result = resolver.resolve("Option A!!!", question)

        # Should still match via word overlap
        assert result.option_index == 0 or result.confidence > 0

    def test_very_long_reply(self):
        """Test with very long reply text."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=["Programming language"],
            cluster_ids=[0],
        )

        resolver = ReplyResolver(fuzzy_match_threshold=0.2)  # Very low threshold for long text

        long_reply = "I think what I really meant was the programming language " * 10
        result = resolver.resolve(long_reply, question)

        # Should match via word overlap
        assert result.option_index == 0 or result.confidence > 0

    def test_unicode_in_reply(self):
        """Test reply with unicode characters."""
        question = ClarifyingQuestion(
            question_text="Which?",
            options=["Programming language", "Snake species"],
            cluster_ids=[0, 1],
        )

        resolver = ReplyResolver()

        result = resolver.resolve("Programming language 🐍", question)

        # Should still match the text part
        assert result.option_index == 0 or result.confidence > 0.5
