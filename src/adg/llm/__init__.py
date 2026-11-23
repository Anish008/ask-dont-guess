"""LLM inference module.

Provides backends for local and API-based LLM inference.
"""

from adg.llm.hf_backend import HFLLMBackend
from adg.llm.mock_backend import MockLLMBackend
from adg.llm.parsing import parse_plain_output, parse_structured_output
from adg.llm.prompts import (
    format_clarification_prompt,
    format_plain_prompt,
    format_structured_prompt,
)

__all__ = [
    # Backends
    "HFLLMBackend",
    "MockLLMBackend",
    # Prompts
    "format_clarification_prompt",
    "format_plain_prompt",
    "format_structured_prompt",
    # Parsing
    "parse_plain_output",
    "parse_structured_output",
]
