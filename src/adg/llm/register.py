"""Register LLM backends with the global registry.

This module registers all available LLM backends when imported.
"""

from adg.core.registry import backend_registry
from adg.llm.hf_backend import HFLLMBackend
from adg.llm.mock_backend import MockLLMBackend

# Register backends
backend_registry.register("mock", MockLLMBackend)
backend_registry.register("hf", HFLLMBackend)

__all__ = []  # No exports, just side effects
