"""FastAPI service for ADG.

Per spec §10.1: REST API for chat, analysis, and experiments.
"""

from adg.api.app import create_app

__all__ = ["create_app"]
