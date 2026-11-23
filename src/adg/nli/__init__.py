"""NLI inference module.

Provides scorers for semantic equivalence using Natural Language Inference.
"""

from adg.nli.hf_nli import HFNLIScorer
from adg.nli.mock_nli import MockNLIScorer

__all__ = [
    "HFNLIScorer",
    "MockNLIScorer",
]
