"""
Main FastAPI application for the NL2SQL system.

This module sets up the FastAPI application with proper logging,
tracing, and error handling middleware.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .utils.logging import configure_logging, get_module_logger
from .utils.tracing import current_trace_id
from .domain.responses import HealthResponse
from .api.middleware import (
    trace_id_middleware,
    logging_middleware,
    security_headers_middleware,
    general_exception_handler
)
from .config import get_settings
from .infrastructure.database_client import DatabaseClient


# Configure logging on module import
configure_logging()
logger = get_module_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting NL2SQL API server", version="0.1.0")

    # Initialize database client
    settings = get_settings()
    db_client = DatabaseClient(settings.database)

    try:
        await db_client.connect()
        logger.info("Database client connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect database client: {e}")
        # Continue without database - health check will report status

    # Store in app state for dependency injection
    app.state.db_client = db_client

    yield

    # Shutdown
    logger.info("Shutting down NL2SQL API server")

    # Close database connections
    if hasattr(app.state, "db_client"):
        await app.state.db_client.close()
        logger.info("Database client closed")


# Create FastAPI application
app = FastAPI(
    title="NL2SQL API",
    description="Natural Language to SQL conversion API with vector-based schema retrieval",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register middleware in correct order (last registered = first executed)
# app.middleware("http")(security_headers_middleware)
app.middleware("http")(logging_middleware)
app.middleware("http")(trace_id_middleware)

# Register exception handlers
app.exception_handler(Exception)(general_exception_handler)


# API Routes
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Union[str, None]]:
    """Root endpoint returning basic API information."""

    trace_id = current_trace_id()
    logger.info("Root endpoint accessed", trace_id=trace_id)

    return {
        "message": "NL2SQL API",
        "version": "0.1.0",
        "trace_id": trace_id
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check endpoint with comprehensive system status."""

    trace_id = current_trace_id()
    logger.info("Health check endpoint accessed", trace_id=trace_id)

    # Check database status
    database_status = "not_configured"
    if hasattr(app.state, "db_client"):
        db_health = await app.state.db_client.health_check()
        database_status = db_health.get("status", "unknown")

    # Determine overall status
    overall_status = "healthy" if database_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        database_status=database_status,
        vector_store_status="not_configured",
        llm_service_status="not_configured",
        embedding_service_status="not_configured"
    )


# FastAPI app is now ready to be imported and run by uvicorn or other ASGI servers
# Use scripts/run_dev.py for development or poetry run uvicorn src.backend.main:app