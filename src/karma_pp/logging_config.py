"""Configure structlog for the karma-pp application."""

import logging
import sys

import structlog


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
) -> None:
    """Configure structlog and standard library logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_logs: If True, output JSON; otherwise use console-friendly format.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if not json_logs
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
