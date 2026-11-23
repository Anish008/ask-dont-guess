"""Register NLI scorers with the global registry.

This module registers all available NLI scorers when imported.
"""

from adg.core.registry import nli_registry
from adg.nli.hf_nli import HFNLIScorer
from adg.nli.mock_nli import MockNLIScorer

# Register NLI scorers
nli_registry.register("mock", MockNLIScorer)
nli_registry.register("hf", HFNLIScorer)

__all__ = []  # No exports, just side effects
