"""
Middleware and exception handlers for the NL2SQL FastAPI application.

This module contains:
- HTTP middleware for request/response processing
- Centralized exception handlers for all custom exceptions
- Logging, tracing, and security headers

Exception Handling Strategy:
- All NL2SQLException subclasses are caught and converted to JSON responses
- Each exception type maps to a specific HTTP status code
- Responses include trace_id for debugging
- Structured logging captures full error context

Usage in main.py:
    from .api.middleware import register_exception_handlers
    register_exception_handlers(app)
"""

from datetime import datetime, timezone
from typing import Callable, Dict, Type

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..utils.logging import get_module_logger
from ..utils.tracing import generate_trace_id, set_trace_id, current_trace_id
from ..domain.responses import ErrorResponse
from ..domain.errors import (
    NL2SQLException,
    ValidationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ConfigurationError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    SchemaError,
    VectorStoreError,
    LLMError,
    EmbeddingError,
    SQLValidationError,
    SQLGenerationError,
    QueryExecutionError,
    StorageError,
    StorageConnectionError,
    StorageFileError,
    ServiceUnavailableError,
)

# Initialize logger for this module
logger = get_module_logger()


# =============================================================================
# Middleware Functions
# =============================================================================


async def trace_id_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to generate and manage trace IDs for each request.

    - Extracts trace_id from X-Trace-ID header if provided
    - Generates a new UUID trace_id if not provided
    - Sets trace_id in context for the entire request lifecycle
    - Adds trace_id to response headers
    """
    # Generate or extract trace ID
    trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()

    # Set trace ID in context for the entire request lifecycle
    set_trace_id(trace_id)

    # Process request
    response = await call_next(request)

    # Add trace ID to response headers
    response.headers["X-Trace-ID"] = trace_id

    return response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log HTTP requests and responses.

    Logs:
    - Request: method, URL, user-agent, client IP
    - Response: status code, duration in milliseconds
    - Adds X-Process-Time header with duration
    """
    start_time = datetime.now(timezone.utc)
    trace_id = current_trace_id()

    # Log request
    logger.info(
        "HTTP request started",
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
        client_ip=request.client.host if request.client else None,
        trace_id=trace_id
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(round(duration_ms, 2))

    # Log response
    logger.info(
        "HTTP request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
        trace_id=trace_id
    )

    return response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to add security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    """
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Only add HSTS in production/HTTPS environments
    # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


def _create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Dict | None = None
) -> JSONResponse:
    """
    Create a standardized JSON error response.

    All error responses follow this structure:
    {
        "error": "error_code",
        "message": "Human readable message",
        "details": {...},  // Optional additional context
        "trace_id": "uuid",
        "timestamp": "ISO8601"
    }
    """
    trace_id = current_trace_id()

    error_response = ErrorResponse(
        error=error_code.lower(),
        message=message,
        details=details,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc)
    )

    # Use mode="json" to ensure datetime objects are serialized to ISO strings
    # (standard json.dumps doesn't handle datetime)
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json", exclude_none=True)
    )


async def nl2sql_exception_handler(request: Request, exc: NL2SQLException) -> JSONResponse:
    """
    Handler for all NL2SQLException subclasses.

    Maps exception attributes to HTTP response:
    - exc.http_status -> HTTP status code
    - exc.error_code -> error field in response
    - exc.message -> message field in response
    - exc.details -> details field in response
    """
    trace_id = current_trace_id()

    # Log with appropriate level based on status code
    log_level = "warning" if exc.http_status < 500 else "error"
    getattr(logger, log_level)(
        f"{exc.__class__.__name__}: {exc.message}",
        error_code=exc.error_code,
        http_status=exc.http_status,
        details=exc.details,
        url=str(request.url),
        method=request.method,
        trace_id=trace_id
    )

    return _create_error_response(
        status_code=exc.http_status,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details if exc.details else None
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for FastAPI/Pydantic validation errors.

    Converts Pydantic validation errors to standardized format:
    - HTTP 422 Unprocessable Entity
    - Details include field-level error information
    """
    trace_id = current_trace_id()

    # Extract validation error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(
        "Request validation failed",
        error_count=len(errors),
        errors=errors,
        url=str(request.url),
        method=request.method,
        trace_id=trace_id
    )

    return _create_error_response(
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"errors": errors}
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handler for Starlette/FastAPI HTTP exceptions.

    Converts standard HTTP exceptions to standardized format.
    """
    trace_id = current_trace_id()

    # Map status codes to error codes
    error_code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }

    error_code = error_code_map.get(exc.status_code, "HTTP_ERROR")

    log_level = "warning" if exc.status_code < 500 else "error"
    getattr(logger, log_level)(
        f"HTTP {exc.status_code}: {exc.detail}",
        error_code=error_code,
        http_status=exc.status_code,
        url=str(request.url),
        method=request.method,
        trace_id=trace_id
    )

    return _create_error_response(
        status_code=exc.status_code,
        error_code=error_code,
        message=str(exc.detail) if exc.detail else f"HTTP {exc.status_code} error"
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Handler for ValueError exceptions.

    ValueError is commonly raised for invalid input that passes Pydantic
    validation but fails business logic validation.

    Maps to HTTP 400 Bad Request.
    """
    trace_id = current_trace_id()

    logger.warning(
        f"ValueError: {str(exc)}",
        error_type="ValueError",
        url=str(request.url),
        method=request.method,
        trace_id=trace_id
    )

    return _create_error_response(
        status_code=400,
        error_code="BAD_REQUEST",
        message=str(exc)
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global fallback handler for unhandled exceptions.

    - Logs full error details for debugging
    - Returns generic 500 error to client (no internal details exposed)
    - Always includes trace_id for correlation
    """
    trace_id = current_trace_id()

    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method,
        trace_id=trace_id,
        exc_info=True  # Include stack trace in logs
    )

    # Don't expose internal error details to client
    return _create_error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="An internal server error occurred. Please try again later."
    )


# =============================================================================
# Exception Handler Registration
# =============================================================================


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with the FastAPI application.

    This function should be called during app initialization to ensure
    all custom exceptions are properly handled.

    Exception handling priority (first match wins):
    1. Specific NL2SQLException subclasses
    2. RequestValidationError (Pydantic)
    3. StarletteHTTPException (FastAPI/Starlette)
    4. ValueError
    5. General Exception (fallback)

    Usage:
        app = FastAPI()
        register_exception_handlers(app)
    """
    # Register handler for all NL2SQLException subclasses
    # FastAPI will use this for any exception that inherits from NL2SQLException
    # Note: type: ignore needed because FastAPI's add_exception_handler has overly strict typing
    # that doesn't account for specific exception subclass handlers (this is correct usage)
    app.add_exception_handler(NL2SQLException, nl2sql_exception_handler)  # type: ignore[arg-type]

    # Register specific handlers for framework exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]

    # Fallback handler for any unhandled exceptions
    app.add_exception_handler(Exception, general_exception_handler)  # type: ignore[arg-type]

    logger.info(
        "Exception handlers registered",
        handlers=[
            "NL2SQLException",
            "RequestValidationError",
            "StarletteHTTPException",
            "ValueError",
            "Exception (fallback)"
        ]
    )


# =============================================================================
# OpenAPI Error Response Models (for documentation)
# =============================================================================

# These are used in route decorators to document error responses
# Example usage in routes:
#   @app.post("/endpoint", responses=ERROR_RESPONSES)

ERROR_RESPONSES = {
    400: {
        "description": "Bad Request - The request was malformed or invalid",
        "content": {
            "application/json": {
                "example": {
                    "error": "bad_request",
                    "message": "Invalid request format",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    404: {
        "description": "Not Found - The requested resource was not found",
        "content": {
            "application/json": {
                "example": {
                    "error": "not_found",
                    "message": "Resource not found",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    422: {
        "description": "Validation Error - Request validation failed",
        "content": {
            "application/json": {
                "example": {
                    "error": "validation_error",
                    "message": "Request validation failed",
                    "details": {
                        "errors": [
                            {"field": "query", "message": "field required", "type": "missing"}
                        ]
                    },
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    429: {
        "description": "Rate Limit Exceeded - Too many requests",
        "content": {
            "application/json": {
                "example": {
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Please try again later.",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error - An unexpected error occurred",
        "content": {
            "application/json": {
                "example": {
                    "error": "internal_error",
                    "message": "An internal server error occurred. Please try again later.",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    },
    503: {
        "description": "Service Unavailable - A required service is not available",
        "content": {
            "application/json": {
                "example": {
                    "error": "service_unavailable",
                    "message": "Database service is temporarily unavailable",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    }
}