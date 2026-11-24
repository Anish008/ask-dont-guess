"""Methods: baselines and ADG variants.

Per spec §8: Common interface for 5 baselines + ADG-Core + ADG+IC.
"""

from adg.methods.base import BudgetSpec, Method, MethodOutput

__all__ = [
    "BudgetSpec",
    "Method",
    "MethodOutput",
]
