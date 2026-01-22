import structlog
import logging
import inspect
from typing import Any, Optional
from backend.config import get_settings

# Module-level flag to prevent multiple configuration
_logging_configured = False


def _add_module_info(logger: Any, method_name: str, event_dict: structlog.types.EventDict) -> structlog.types.EventDict:
    """
    Custom processor to add module/file information to log records.

    This enhances the default logger name with more explicit file information.
    """
    # Get logger name (module name) if available
    logger_name = event_dict.get('logger', 'unknown')

    # Convert module name to a more readable format
    if logger_name.startswith(('backend.', 'nl2sql.')):
        # Extract just the meaningful part for project modules
        module_parts = logger_name.split('.')
        if len(module_parts) >= 2:
            # Keep last 2 parts (e.g., "services.schema_indexer" from "nl2sql.services.schema_indexer")
            event_dict['module'] = '.'.join(module_parts[-2:])
        else:
            event_dict['module'] = module_parts[-1]
    else:
        event_dict['module'] = logger_name

    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""

    global _logging_configured

    # ---- guard: run only once ----
    if _logging_configured:
        return
    _logging_configured = True

    settings = get_settings()

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.app.log_level),
        handlers=[logging.StreamHandler()]
    )

    # Configure structlog with enhanced file/module tracking
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,  # Adds 'logger' field with module name
            structlog.stdlib.add_log_level,    # Adds 'level' field
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),  # Adds 'timestamp' field (ISO8601)
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _add_module_info,  # Custom processor to add file info
            structlog.processors.JSONRenderer()  # Final JSON output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ to get the module name

    Returns:
        Configured structlog logger with JSON output

    Usage:
        logger = get_logger(__name__)
        logger.info("Schema indexing started", tables_count=14, trace_id="abc-123")

        # Output: {"timestamp": "2024-01-22T10:30:00Z", "level": "info",
        #          "logger": "backend.services.schema_indexer", "module": "services.schema_indexer",
        #          "event": "Schema indexing started", "tables_count": 14, "trace_id": "abc-123"}
    """
    return structlog.get_logger(name)


def get_module_logger() -> structlog.stdlib.BoundLogger:
    """
    Get a logger for the calling module automatically.

    This is a convenience function that automatically determines the module name.

    Returns:
        Configured structlog logger for the calling module

    Note:
        Falls back to 'unknown' module name if frame inspection fails.
    """
    module_name = 'unknown'
    frame = None

    try:
        # Get the calling frame to determine the module name
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_frame = frame.f_back
            module_name = caller_frame.f_globals.get('__name__', 'unknown')
    except (AttributeError, RuntimeError):
        # Frame inspection can fail in some environments (e.g., some REPL implementations)
        pass
    finally:
        # Clean up frame references to avoid potential memory leaks
        if frame is not None:
            del frame

    return get_logger(module_name)
