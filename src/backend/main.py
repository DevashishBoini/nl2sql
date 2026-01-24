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
from .api.dependencies import (
    SettingsDep,
    OptionalDatabaseClientDep,
    OptionalStorageClientDep,
    OptionalLLMClientDep,
    OptionalEmbeddingClientDep
)
from .config import get_settings
from .infrastructure.database_client import DatabaseClient
from .infrastructure.storage_client import StorageClient
from .infrastructure.llm_client import LLMClient
from .infrastructure.embedding_client import EmbeddingClient


# Configure logging on module import
configure_logging()
logger = get_module_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting NL2SQL API server", version="0.1.0")

    # Load settings once at startup
    settings = get_settings()
    app.state.settings = settings
    logger.info("Settings loaded successfully")

    # Initialize database client
    db_client = DatabaseClient(settings.database)
    try:
        await db_client.connect()
        logger.info("Database client connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect database client: {e}")
        # Continue without database - health check will report status

    # Initialize storage client
    storage_client = StorageClient(settings.storage)
    try:
        await storage_client.connect()
        logger.info("Storage client connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect storage client: {e}")
        # Continue without storage - health check will report status

    # Initialize LLM client
    llm_client = LLMClient(settings.llm)
    try:
        await llm_client.connect()
        logger.info("LLM client connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect LLM client: {e}")
        # Continue without LLM - health check will report status

    # Initialize embedding client
    embedding_client = EmbeddingClient(settings.embedding)
    try:
        await embedding_client.connect()
        logger.info("Embedding client connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect embedding client: {e}")
        # Continue without embeddings - health check will report status

    # Store in app state for dependency injection
    app.state.db_client = db_client
    app.state.storage_client = storage_client
    app.state.llm_client = llm_client
    app.state.embedding_client = embedding_client

    yield

    # Shutdown
    logger.info("Shutting down NL2SQL API server")

    # Close all clients
    if hasattr(app.state, "db_client"):
        await app.state.db_client.close()
        logger.info("Database client closed")

    if hasattr(app.state, "storage_client"):
        await app.state.storage_client.close()
        logger.info("Storage client closed")

    if hasattr(app.state, "llm_client"):
        await app.state.llm_client.close()
        logger.info("LLM client closed")

    if hasattr(app.state, "embedding_client"):
        await app.state.embedding_client.close()
        logger.info("Embedding client closed")


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
async def root(settings: SettingsDep) -> Dict[str, Union[str, None]]:
    """Root endpoint returning basic API information."""

    trace_id = current_trace_id()
    logger.info("Root endpoint accessed", trace_id=trace_id)

    return {
        "message": "NL2SQL API",
        "version": "0.1.0",
        "trace_id": trace_id,
        "log_level": settings.app.log_level
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(
    db_client: OptionalDatabaseClientDep,
    storage_client: OptionalStorageClientDep,
    llm_client: OptionalLLMClientDep,
    embedding_client: OptionalEmbeddingClientDep
) -> HealthResponse:
    """Health check endpoint with comprehensive system status."""

    trace_id = current_trace_id()
    logger.info("Health check endpoint accessed", trace_id=trace_id)

    # Check database status
    database_status = "not_configured"
    if db_client:
        db_health = await db_client.health_check()
        database_status = db_health.get("status", "unknown")

    # Check storage status
    storage_status = "not_configured"
    if storage_client:
        storage_health = await storage_client.health_check()
        storage_status = storage_health.get("status", "unknown")

    # Check LLM status
    llm_status = "not_configured"
    if llm_client:
        llm_status = "healthy" if llm_client.is_connected() else "unhealthy"

    # Check embedding status
    embedding_status = "not_configured"
    if embedding_client:
        embedding_status = "healthy" if embedding_client.is_connected() else "unhealthy"

    # Determine overall status
    overall_status = "healthy" if (
        database_status == "healthy" and
        storage_status == "healthy" and
        llm_status == "healthy" and
        embedding_status == "healthy"
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        database_status=database_status,
        vector_store_status=storage_status,
        llm_service_status=llm_status,
        embedding_service_status=embedding_status
    )


# FastAPI app is now ready to be imported and run by uvicorn or other ASGI servers
# Use scripts/run_dev.py for development or poetry run uvicorn src.backend.main:app