"""Dataset pipeline for ADG evaluation.

Loaders for AmbigQA, CLAMBER, NQ-open with common schema.
"""

from adg.data.schema import Example, GoldInterpretation

__all__ = [
    "Example",
    "GoldInterpretation",
]
