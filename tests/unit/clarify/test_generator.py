"""Tests for clarifying question generation."""

import pytest
from adg.clarify.generator import QuestionGenerator
from adg.core.types import ClarifyingQuestion, Cluster, Sample


@pytest.fixture
def mock_samples() -> list[Sample]:
    """Create mock samples for testing."""
    return [
        Sample(
            sample_id=0,
            query_id="q1",
            answer="Python is a programming language",
            interpretation="programming language Python",
            is_valid=True,
            is_refusal=False,
        ),
        Sample(
            sample_id=1,
            query_id="q1",
            answer="Python is a snake",
            interpretation="snake species Python",
            is_valid=True,
            is_refusal=False,
        ),
        Sample(
            sample_id=2,
            query_id="q1",
            answer="Python is a programming language used for coding",
            interpretation="programming language Python",
            is_valid=True,
            is_refusal=False,
        ),
    ]


@pytest.fixture
def mock_clusters(mock_samples: list[Sample]) -> list[Cluster]:
    """Create mock clusters for testing."""
    return [
        Cluster(
            cluster_id=0,
            sample_ids=[0, 2],
            representative_answer="Python is a programming language",
            representative_interpretation="Python: programming language Python",
            size=2,
        ),
        Cluster(
            cluster_id=1,
            sample_ids=[1],
            representative_answer="Python is a snake",
            representative_interpretation="Python: snake species Python",
            size=1,
        ),
    ]


class TestQuestionGeneratorTemplate:
    """Tests for template-based question generation."""

    def test_basic_template_generation(
        self,
        mock_samples: list[Sample],
        mock_clusters: list[Cluster],
    ):
        """Test basic template question generation."""
        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="Python",
            clusters=mock_clusters,
            samples=mock_samples,
            max_options=4,
        )

        assert isinstance(question, ClarifyingQuestion)
        assert len(question.options) == 2
        assert "Python" in question.question_text
        assert "1." in question.question_text
        assert "2." in question.question_text

        # Check that interpretations are cleaned
        for option in question.options:
            assert option  # Not empty
            assert option[0].isupper()  # Capitalized

    def test_template_two_options(
        self,
        mock_samples: list[Sample],
        mock_clusters: list[Cluster],
    ):
        """Test template with exactly two options uses 'did you mean'."""
        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="Python",
            clusters=mock_clusters,
            samples=mock_samples,
            max_options=2,
        )

        assert "did you mean:" in question.question_text.lower()
        assert len(question.options) == 2

    def test_template_multiple_options(
        self,
        mock_samples: list[Sample],
    ):
        """Test template with more than two options uses 'which of these'."""
        clusters = [
            Cluster(
                cluster_id=i,
                sample_ids=[i],
                representative_answer=f"Answer {i}",
                representative_interpretation=f"Interpretation {i}",
                size=1,
            )
            for i in range(3)
        ]

        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="Test query",
            clusters=clusters,
            samples=mock_samples,
            max_options=4,
        )

        assert "which of these did you mean:" in question.question_text.lower()
        assert len(question.options) == 3

    def test_template_max_options_limit(
        self,
        mock_samples: list[Sample],
    ):
        """Test that max_options limits number of options."""
        clusters = [
            Cluster(
                cluster_id=i,
                sample_ids=[i % len(mock_samples)],
                representative_answer=f"Answer {i}",
                representative_interpretation=f"Interpretation {i}",
                size=1,
            )
            for i in range(10)
        ]

        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="Test query",
            clusters=clusters,
            samples=mock_samples,
            max_options=4,
        )

        assert len(question.options) == 4

    def test_template_interpretation_cleaning(self):
        """Test that interpretations are cleaned properly."""
        clusters = [
            Cluster(
                cluster_id=0,
                sample_ids=[0],
                representative_answer="Answer 1",
                representative_interpretation="test query: some interpretation",
                size=1,
            ),
            Cluster(
                cluster_id=1,
                sample_ids=[1],
                representative_answer="Answer 2",
                representative_interpretation="Test query - another interpretation",
                size=1,
            ),
        ]

        samples = [
            Sample(
                sample_id=0,
                query_id="q1",
                answer="Answer 1",
                interpretation="test query: some interpretation",
                is_valid=True,
                is_refusal=False,
            ),
            Sample(
                sample_id=1,
                query_id="q1",
                answer="Answer 2",
                interpretation="Test query - another interpretation",
                is_valid=True,
                is_refusal=False,
            ),
        ]

        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="test query",
            clusters=clusters,
            samples=samples,
            max_options=4,
        )

        # Check that query prefix is removed
        for option in question.options:
            assert not option.lower().startswith("test query")
            # Check capitalization
            assert option[0].isupper()

    def test_template_empty_clusters(self):
        """Test handling of empty cluster list."""
        generator = QuestionGenerator(mode="template")

        question = generator.generate(
            query_text="Test query",
            clusters=[],
            samples=[],
            max_options=4,
        )

        assert isinstance(question, ClarifyingQuestion)
        assert len(question.options) == 0


class TestQuestionGeneratorLLM:
    """Tests for LLM-based question generation."""

    @pytest.fixture
    def mock_llm_backend(self):
        """Create a mock LLM backend."""

        class MockLLM:
            def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
                # Return properly formatted response
                return """QUESTION: Which interpretation of your query did you mean?
OPTIONS:
1. Programming language Python
2. Snake species Python"""

        return MockLLM()

    def test_llm_generation_success(
        self,
        mock_llm_backend,
        mock_samples: list[Sample],
        mock_clusters: list[Cluster],
    ):
        """Test successful LLM-based question generation."""
        generator = QuestionGenerator(mode="llm", llm_backend=mock_llm_backend)

        question = generator.generate(
            query_text="Python",
            clusters=mock_clusters,
            samples=mock_samples,
            max_options=4,
        )

        assert isinstance(question, ClarifyingQuestion)
        assert len(question.options) == 2
        assert "Programming language Python" in question.options
        assert "Snake species Python" in question.options

    def test_llm_generation_fallback(
        self,
        mock_samples: list[Sample],
        mock_clusters: list[Cluster],
    ):
        """Test fallback to template on LLM parse failure."""

        class BadMockLLM:
            def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
                # Return malformed response
                return "This is not a properly formatted response"

        generator = QuestionGenerator(mode="llm", llm_backend=BadMockLLM())

        question = generator.generate(
            query_text="Python",
            clusters=mock_clusters,
            samples=mock_samples,
            max_options=4,
        )

        # Should fall back to template format
        assert isinstance(question, ClarifyingQuestion)
        assert len(question.options) == 2

    def test_llm_mode_requires_backend(self):
        """Test that LLM mode requires llm_backend."""
        with pytest.raises(ValueError, match="llm_backend required"):
            QuestionGenerator(mode="llm", llm_backend=None)

    def test_llm_prompt_building(
        self,
        mock_llm_backend,
        mock_samples: list[Sample],
        mock_clusters: list[Cluster],
    ):
        """Test that LLM prompt is built correctly."""
        generator = QuestionGenerator(mode="llm", llm_backend=mock_llm_backend)

        # Build prompt directly
        interpretations = [c.representative_interpretation for c in mock_clusters]
        prompt = generator._build_llm_prompt("Python", interpretations)

        assert "Python" in prompt
        assert "ambiguous" in prompt.lower()
        assert "QUESTION:" in prompt
        assert "OPTIONS:" in prompt


class TestQuestionGeneratorInitialization:
    """Tests for QuestionGenerator initialization."""

    def test_default_mode(self):
        """Test default mode is template."""
        generator = QuestionGenerator()
        assert generator.mode == "template"

    def test_template_mode_no_backend_required(self):
        """Test template mode doesn't require backend."""
        generator = QuestionGenerator(mode="template")
        assert generator.llm_backend is None

    def test_llm_mode_stores_backend(self):
        """Test LLM mode stores backend."""

        class DummyBackend:
            pass

        backend = DummyBackend()
        generator = QuestionGenerator(mode="llm", llm_backend=backend)
        assert generator.llm_backend is backend
