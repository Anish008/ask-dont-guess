"""Tests for deduplication."""


from adg.data.dedup import deduplicate_nq_open_against_ambigqa, normalize_question
from adg.data.schema import Example


class TestNormalization:
    """Tests for question normalization."""

    def test_normalize_basic(self) -> None:
        """Test basic normalization."""
        result = normalize_question("What is the capital of France?")
        assert result == "what is the capital of france"

    def test_normalize_punctuation(self) -> None:
        """Test punctuation removal."""
        result = normalize_question("Who's the author?!")
        assert result == "whos the author"

    def test_normalize_whitespace(self) -> None:
        """Test whitespace normalization."""
        result = normalize_question("What   is    the    answer?")
        assert result == "what is the answer"

    def test_normalize_mixed(self) -> None:
        """Test mixed normalization."""
        result = normalize_question("  What's  the  capital?!  ")
        assert result == "whats the capital"


class TestDeduplication:
    """Tests for NQ-open deduplication against AmbigQA."""

    def test_no_overlap(self) -> None:
        """Test when no questions overlap."""
        nq_examples = [
            Example(
                example_id="nq:1",
                dataset="nq_open",
                split="dev",
                original_id="1",
                question="Question 1",
            ),
            Example(
                example_id="nq:2",
                dataset="nq_open",
                split="dev",
                original_id="2",
                question="Question 2",
            ),
        ]

        ambigqa_examples = [
            Example(
                example_id="ambig:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="Question 3",
            ),
        ]

        deduplicated, removed_count = deduplicate_nq_open_against_ambigqa(
            nq_examples, ambigqa_examples
        )

        assert len(deduplicated) == 2
        assert removed_count == 0

    def test_full_overlap(self) -> None:
        """Test when all NQ questions appear in AmbigQA."""
        nq_examples = [
            Example(
                example_id="nq:1",
                dataset="nq_open",
                split="dev",
                original_id="1",
                question="What is the capital of Georgia?",
            ),
        ]

        ambigqa_examples = [
            Example(
                example_id="ambig:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="What is the capital of Georgia?",
            ),
        ]

        deduplicated, removed_count = deduplicate_nq_open_against_ambigqa(
            nq_examples, ambigqa_examples
        )

        assert len(deduplicated) == 0
        assert removed_count == 1

    def test_partial_overlap(self) -> None:
        """Test when some questions overlap."""
        nq_examples = [
            Example(
                example_id=f"nq:{i}",
                dataset="nq_open",
                split="dev",
                original_id=str(i),
                question=f"Question {i}",
            )
            for i in range(5)
        ]

        ambigqa_examples = [
            Example(
                example_id="ambig:0",
                dataset="ambigqa",
                split="dev",
                original_id="0",
                question="Question 1",  # Matches nq:1
            ),
            Example(
                example_id="ambig:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="Question 3",  # Matches nq:3
            ),
        ]

        deduplicated, removed_count = deduplicate_nq_open_against_ambigqa(
            nq_examples, ambigqa_examples
        )

        assert len(deduplicated) == 3  # 5 - 2
        assert removed_count == 2

    def test_case_insensitive_matching(self) -> None:
        """Test that matching is case-insensitive."""
        nq_examples = [
            Example(
                example_id="nq:1",
                dataset="nq_open",
                split="dev",
                original_id="1",
                question="What is the Capital of France?",
            ),
        ]

        ambigqa_examples = [
            Example(
                example_id="ambig:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="what is the capital of france?",
            ),
        ]

        deduplicated, removed_count = deduplicate_nq_open_against_ambigqa(
            nq_examples, ambigqa_examples
        )

        assert len(deduplicated) == 0
        assert removed_count == 1

    def test_punctuation_normalized_matching(self) -> None:
        """Test that punctuation differences don't prevent matching."""
        nq_examples = [
            Example(
                example_id="nq:1",
                dataset="nq_open",
                split="dev",
                original_id="1",
                question="Who's the author?",
            ),
        ]

        ambigqa_examples = [
            Example(
                example_id="ambig:1",
                dataset="ambigqa",
                split="dev",
                original_id="1",
                question="Whos the author",
            ),
        ]

        deduplicated, removed_count = deduplicate_nq_open_against_ambigqa(
            nq_examples, ambigqa_examples
        )

        assert len(deduplicated) == 0
        assert removed_count == 1
