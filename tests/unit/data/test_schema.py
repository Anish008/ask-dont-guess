"""Tests for data schema models."""


from adg.data.schema import Example, GoldInterpretation


class TestGoldInterpretation:
    """Tests for GoldInterpretation model."""

    def test_create_interpretation(self) -> None:
        """Test creating a gold interpretation."""
        interp = GoldInterpretation(
            interp_id="ex1::i0",
            disambiguated_question="What is the capital of France?",
            answers=["Paris"],
        )

        assert interp.interp_id == "ex1::i0"
        assert interp.disambiguated_question == "What is the capital of France?"
        assert interp.answers == ["Paris"]
        assert interp.description is None

    def test_interpretation_with_description(self) -> None:
        """Test interpretation with description field."""
        interp = GoldInterpretation(
            interp_id="ex1::i0",
            disambiguated_question="What is the capital of Georgia the country?",
            answers=["Tbilisi"],
            description="the country Georgia",
        )

        assert interp.description == "the country Georgia"


class TestExample:
    """Tests for Example model."""

    def test_create_clear_example(self) -> None:
        """Test creating a clear (non-ambiguous) example."""
        interp = GoldInterpretation(
            interp_id="nq:1::i0",
            disambiguated_question="What is 2+2?",
            answers=["4", "four"],
        )

        example = Example(
            example_id="nq_open:dev:1",
            dataset="nq_open",
            split="dev",
            original_id="1",
            question="What is 2+2?",
            is_ambiguous=False,
            interpretations=[interp],
            gold_answers=["4", "four"],
        )

        assert example.example_id == "nq_open:dev:1"
        assert example.dataset == "nq_open"
        assert example.is_ambiguous is False
        assert len(example.interpretations) == 1
        assert example.gold_answers == ["4", "four"]

    def test_create_ambiguous_example(self) -> None:
        """Test creating an ambiguous example with multiple interpretations."""
        interp1 = GoldInterpretation(
            interp_id="ambigqa:1::i0",
            disambiguated_question="What is the capital of Georgia the US state?",
            answers=["Atlanta"],
            description="the US state",
        )
        interp2 = GoldInterpretation(
            interp_id="ambigqa:1::i1",
            disambiguated_question="What is the capital of Georgia the country?",
            answers=["Tbilisi"],
            description="the country",
        )

        example = Example(
            example_id="ambigqa:dev:1",
            dataset="ambigqa",
            split="dev",
            original_id="1",
            question="What is the capital of Georgia?",
            is_ambiguous=True,
            interpretations=[interp1, interp2],
            gold_answers=["Atlanta", "Tbilisi"],
        )

        assert example.is_ambiguous is True
        assert len(example.interpretations) == 2
        assert example.interpretations[0].description == "the US state"
        assert example.interpretations[1].description == "the country"

    def test_example_with_metadata(self) -> None:
        """Test example with source metadata."""
        example = Example(
            example_id="clamber:all:1",
            dataset="clamber",
            split="all",
            original_id="1",
            question="Book a table",
            is_ambiguous=True,
            ambiguity_category="underspecified",
            source_metadata={"original_category": "entity", "confidence": 0.95},
        )

        assert example.ambiguity_category == "underspecified"
        assert example.source_metadata["original_category"] == "entity"

    def test_example_defaults(self) -> None:
        """Test default field values."""
        example = Example(
            example_id="test:dev:1",
            dataset="ambigqa",
            split="dev",
            original_id="1",
            question="Test?",
        )

        assert example.is_ambiguous is None
        assert example.ambiguity_category is None
        assert example.interpretations == []
        assert example.gold_answers == []
        assert example.gold_clarifying_question is None
        assert example.source_metadata == {}
        assert example.annotation_notes is None
