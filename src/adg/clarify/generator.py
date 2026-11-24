"""Clarifying question generation.

Per spec §4.10: Generate questions to disambiguate between interpretations.
"""

from adg.core.logging import get_logger
from adg.core.protocols import LLMBackend
from adg.core.types import ClarifyingQuestion, Cluster, Sample

log = get_logger(__name__)

# Template formatting constants
_BINARY_CHOICE_COUNT = 2


class QuestionGenerator:
    """Generates clarifying questions from clusters.

    Supports template-based and LLM-based generation modes.
    """

    def __init__(
        self,
        mode: str = "template",
        llm_backend: LLMBackend | None = None,
    ):
        """Initialize question generator.

        Args:
            mode: Generation mode ("template" or "llm")
            llm_backend: LLM backend for LLM-based generation (required if mode="llm")
        """
        self.mode = mode
        self.llm_backend = llm_backend

        if mode == "llm" and llm_backend is None:
            msg = "llm_backend required when mode='llm'"
            raise ValueError(msg)

        log.info("question_generator_initialized", mode=mode)

    def generate(
        self,
        query_text: str,
        clusters: list[Cluster],
        samples: list[Sample],
        max_options: int = 4,
    ) -> ClarifyingQuestion:
        """Generate clarifying question from clusters.

        Args:
            query_text: Original user query
            clusters: List of clusters (should be sorted by size descending)
            samples: All samples (for accessing interpretations)
            max_options: Maximum number of options to present (default 4)

        Returns:
            ClarifyingQuestion with question text and options
        """
        log.debug(
            "generating_clarifying_question",
            mode=self.mode,
            n_clusters=len(clusters),
            max_options=max_options,
        )

        if self.mode == "template":
            return self._generate_template(query_text, clusters, samples, max_options)
        else:
            return self._generate_llm(query_text, clusters, samples, max_options)

    def _generate_template(
        self,
        query_text: str,
        clusters: list[Cluster],
        samples: list[Sample],
        max_options: int,
    ) -> ClarifyingQuestion:
        """Generate question using templates.

        Args:
            query_text: Original query
            clusters: Clusters sorted by size
            samples: All samples
            max_options: Max options to show

        Returns:
            ClarifyingQuestion
        """
        # Take top clusters up to max_options
        top_clusters = clusters[:max_options]

        # Extract representative interpretations
        options = []
        for cluster in top_clusters:
            interp = cluster.representative_interpretation
            # Clean up interpretation (remove query prefix if present)
            cleaned_interp = self._clean_interpretation(interp, query_text)
            options.append(cleaned_interp)

        # Generate template question
        question_text = self._format_template_question(query_text, options)

        # Extract cluster IDs
        cluster_ids = [cluster.cluster_id for cluster in top_clusters]

        log.info(
            "clarifying_question_generated",
            mode="template",
            n_options=len(options),
        )

        return ClarifyingQuestion(
            question_text=question_text,
            options=options,
            mode="template",
            cluster_ids=cluster_ids,
        )

    def _generate_llm(
        self,
        query_text: str,
        clusters: list[Cluster],
        samples: list[Sample],
        max_options: int,
    ) -> ClarifyingQuestion:
        """Generate question using LLM.

        Args:
            query_text: Original query
            clusters: Clusters sorted by size
            samples: All samples
            max_options: Max options to show

        Returns:
            ClarifyingQuestion
        """
        # Take top clusters
        top_clusters = clusters[:max_options]

        # Build prompt for LLM
        interpretations = [cluster.representative_interpretation for cluster in top_clusters]

        prompt = self._build_llm_prompt(query_text, interpretations)

        # Generate with LLM
        assert self.llm_backend is not None
        response = self.llm_backend.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200,
        )

        # Parse LLM response (simple parsing for now)
        parsed_question_text, options = self._parse_llm_response(response, interpretations)

        # Extract cluster IDs
        cluster_ids = [cluster.cluster_id for cluster in top_clusters]

        log.info(
            "clarifying_question_generated",
            mode="llm",
            n_options=len(options),
        )

        return ClarifyingQuestion(
            question_text=parsed_question_text,
            options=options,
            mode="llm",
            cluster_ids=cluster_ids,
        )

    def _clean_interpretation(self, interpretation: str, query_text: str) -> str:
        """Clean interpretation text for display.

        Args:
            interpretation: Raw interpretation text
            query_text: Original query

        Returns:
            Cleaned interpretation
        """
        # Remove common query prefixes
        cleaned = interpretation.strip()

        # If interpretation starts with query, remove it
        if cleaned.lower().startswith(query_text.lower()):
            cleaned = cleaned[len(query_text) :].strip()

        # Remove leading punctuation
        while cleaned and cleaned[0] in ":-,.;":
            cleaned = cleaned[1:].strip()

        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _format_template_question(self, query_text: str, options: list[str]) -> str:
        """Format template question text.

        Args:
            query_text: Original query
            options: List of interpretation options

        Returns:
            Question text
        """
        if len(options) == _BINARY_CHOICE_COUNT:
            question = f'By "{query_text}", did you mean:\n'
        else:
            question = f'By "{query_text}", which of these did you mean:\n'

        for i, option in enumerate(options, 1):
            question += f"{i}. {option}\n"

        return question.strip()

    def _build_llm_prompt(self, query_text: str, interpretations: list[str]) -> str:
        """Build prompt for LLM-based generation.

        Args:
            query_text: Original query
            interpretations: List of interpretations

        Returns:
            Prompt string
        """
        prompt = f"""The user asked: "{query_text}"

This question is ambiguous. Here are possible interpretations:
"""
        for i, interp in enumerate(interpretations, 1):
            prompt += f"{i}. {interp}\n"

        prompt += """
Generate a natural clarifying question asking the user which interpretation they meant.
Format your response as:
QUESTION: <question text>
OPTIONS:
1. <option 1>
2. <option 2>
...
"""
        return prompt

    def _parse_llm_response(
        self,
        response: str,
        fallback_interpretations: list[str],
    ) -> tuple[str, list[str]]:
        """Parse LLM response into question and options.

        Args:
            response: LLM response text
            fallback_interpretations: Fallback interpretations if parsing fails

        Returns:
            (question_text, options) tuple
        """
        try:
            lines = response.strip().split("\n")

            # Extract question
            question = ""
            options = []
            in_options = False

            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith("QUESTION:"):
                    question = stripped_line[9:].strip()
                elif stripped_line.startswith("OPTIONS:"):
                    in_options = True
                elif in_options and stripped_line and stripped_line[0].isdigit():
                    # Parse option line (e.g., "1. Option text")
                    # Remove number prefix
                    option = (
                        stripped_line.split(".", 1)[1].strip()
                        if "." in stripped_line
                        else stripped_line
                    )
                    options.append(option)

            if not question or not options:
                raise ValueError("Failed to parse question or options")

            return question, options

        except Exception:
            # Fallback to template if parsing fails
            log.warning("llm_response_parse_failed", falling_back_to_template=True)
            question = self._format_template_question("your query", fallback_interpretations)
            return question, fallback_interpretations


__all__ = ["QuestionGenerator"]
