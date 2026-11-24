"""Reporting module for tables and figures.

Per spec §12.5: Generate Markdown/LaTeX tables and matplotlib figures.
"""

from adg.reporting.tables import generate_cost_table, generate_main_results_table

__all__ = [
    "generate_main_results_table",
    "generate_cost_table",
]
