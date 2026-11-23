"""Structured logging setup with structlog.

Per spec §6.12 and §16.1: JSON in production, pretty console in dev.
Never log API keys or full user content in production.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    truncate_queries: bool = True,
    max_query_len: int = 200,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, use JSON output (for production). If False, pretty console.
        truncate_queries: If True, truncate query text in logs (production safety)
        max_query_len: Maximum query length to log before truncation

    Example:
        >>> configure_logging(level="DEBUG", json_format=False)
        >>> log = get_logger(__name__)
        >>> log.info("pipeline_start", query_id="123", model="qwen2.5-1.5b")
    """
    # Set stdlib logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Shared processors (always applied)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add query truncation in production
    if truncate_queries:

        def truncate_query_processor(
            logger: Any, method_name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
            """Truncate query text to avoid logging sensitive/long content."""
            if "query" in event_dict and isinstance(event_dict["query"], str):
                q = event_dict["query"]
                if len(q) > max_query_len:
                    event_dict["query"] = q[:max_query_len] + "..."
                    event_dict["query_truncated"] = True
            return event_dict

        shared_processors.append(truncate_query_processor)

    # Choose renderer based on format
    if json_format:
        # Production: JSON lines
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console with colors
        processors = [
            *shared_processors,
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:  # structlog.stdlib.BoundLogger, but mypy struggles with it
    """Get a structlog logger for the given module.

    Args:
        name: Usually __name__ from the calling module

    Returns:
        A bound logger ready to use

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("event_name", key="value", count=42)
    """
    return structlog.get_logger(name)


# Configure with sensible defaults on import (can be reconfigured)
# Default: INFO level, pretty console (dev mode)
configure_logging(level="INFO", json_format=False)
