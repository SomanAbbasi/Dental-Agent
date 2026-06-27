import logging
import structlog
from app.config.settings import get_settings


def setup_logging() -> None:
    """
    Configure structlog for structured JSON-style logging.
    
    """
    settings = get_settings()
    is_dev = settings.app_env == "development"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if is_dev
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    setup_logging()
    return structlog.get_logger(name)