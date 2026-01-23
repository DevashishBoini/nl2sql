"""
Middleware components for the NL2SQL FastAPI application.

This module contains all HTTP middleware for request/response processing,
logging, tracing, error handling, and security headers.
"""

from datetime import datetime, timezone

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ..utils.logging import get_module_logger
from ..utils.tracing import generate_trace_id, set_trace_id, current_trace_id
from ..domain.responses import ErrorResponse

# Initialize logger for this module
logger = get_module_logger()


async def trace_id_middleware(request: Request, call_next) -> Response:
    """Middleware to generate and manage trace IDs for each request."""

    # Generate or extract trace ID
    trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()

    # Set trace ID in context for the entire request lifecycle
    set_trace_id(trace_id)

    # Process request
    response = await call_next(request)

    # Add trace ID to response headers
    response.headers["X-Trace-ID"] = trace_id

    return response


async def logging_middleware(request: Request, call_next) -> Response:
    """Middleware to log HTTP requests and responses."""

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


async def security_headers_middleware(request: Request, call_next) -> Response:
    """Middleware to add security headers to all responses."""

    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Only add HSTS in production/HTTPS environments
    # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""

    trace_id = current_trace_id()

    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method,
        trace_id=trace_id
    )

    error_response = ErrorResponse(
        error="internal_server_error",
        message="An internal server error occurred",
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc)
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )