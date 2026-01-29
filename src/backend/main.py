"""
Main FastAPI application for the NL2SQL system.

This module sets up the FastAPI application with proper logging,
tracing, and error handling middleware.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Union

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

from .utils.logging import configure_logging, get_module_logger
from .utils.tracing import get_trace_id
from .domain.responses import (
    HealthResponse,
    SchemaSummaryResponse,
    IndexSchemaResponse,
    VectorSchemaSearchResponse,
    VectorStatsResponse,
    SchemaSearchResult,
    DropCollectionResponse,
    NL2SQLQueryResponse,
)
from .domain.requests import (
    IndexSchemaRequest,
    SchemaSearchRequest,
    DropCollectionRequest,
    NL2SQLQueryRequest,
)
from .api.middleware import (
    trace_id_middleware,
    logging_middleware,
    register_exception_handlers,
    ERROR_RESPONSES,
)
from .api.dependencies import (
    SettingsDep,
    SchemaServiceDep,
    VectorServiceDep,
    NL2SQLServiceDep,
    OptionalDatabaseClientDep,
    OptionalStorageClientDep,
    OptionalLLMClientDep,
    OptionalEmbeddingClientDep,
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

    # Store clients in app state for dependency injection
    # Note: VectorRepository is created on-demand in dependencies.py
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

# Register all exception handlers (NL2SQLException, ValidationError, HTTPException, etc.)
register_exception_handlers(app)


# API Routes
@app.get("/", tags=["Root"])
async def root(settings: SettingsDep) -> Dict[str, Union[str, None]]:
    """
    Root endpoint returning basic API information.

    **Response**: Dict with message, version, trace_id, log_level
    """

    trace_id = get_trace_id()
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
    embedding_client: OptionalEmbeddingClientDep,
) -> HealthResponse:
    """
    Health check endpoint with comprehensive system status.

    **Response Model**: `HealthResponse`
    - status: Overall health (healthy/degraded)
    - database_status, vector_store_status, llm_service_status, embedding_service_status
    """

    trace_id = get_trace_id()
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

    # Vector store status depends on DB + embedding clients
    # VectorRepository is created on-demand, so check dependencies
    vector_store_status = "not_configured"
    if db_client and embedding_client:
        if database_status == "healthy" and embedding_status == "healthy":
            vector_store_status = "healthy"
        else:
            vector_store_status = "degraded"

    # Determine overall status
    overall_status = "healthy" if (
        database_status == "healthy" and
        storage_status == "healthy" and
        llm_status == "healthy" and
        embedding_status == "healthy" and
        vector_store_status == "healthy"
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        database_status=database_status,
        vector_store_status=vector_store_status,
        llm_service_status=llm_status,
        embedding_service_status=embedding_status
    )


@app.get("/test-schema", response_model=SchemaSummaryResponse, tags=["Testing"])
async def test_schema(schema_service: SchemaServiceDep) -> SchemaSummaryResponse:
    """
    Test endpoint to fetch schema information using SchemaService.

    Returns complete schema summary including tables, columns, and relationships.

    **Response Model**: `SchemaSummaryResponse`
    - schema_name, table_count, relationship_count
    - tables: List of TableNode
    - relationships: List of RelationshipNode
    """
    trace_id = get_trace_id()
    logger.info("Test schema endpoint accessed", trace_id=trace_id)

    # Get schema summary via Service layer
    summary = await schema_service.get_schema_summary()

    logger.info(
        "Schema summary fetched successfully",
        table_count=summary["table_count"],
        relationship_count=summary["relationship_count"],
        trace_id=trace_id
    )

    return SchemaSummaryResponse(
        trace_id=trace_id,
        schema_name=summary["schema_name"],
        table_count=summary["table_count"],
        relationship_count=summary["relationship_count"],
        tables=summary["tables"],
        relationships=summary["relationships"],
        sample_table=summary.get("sample_table"),
        sample_columns=summary.get("sample_columns", [])
    )


# -------------------------
# Vector Store Endpoints
# -------------------------

@app.post(
    "/api/v1/schema/index",
    response_model=IndexSchemaResponse,
    tags=["Schema Indexing"],
    responses={
        **{k: v for k, v in ERROR_RESPONSES.items() if k in [422, 500, 503]},
    },
)
async def index_schema(
    vector_service: VectorServiceDep,
    settings: SettingsDep,
    request: IndexSchemaRequest = Body(default=IndexSchemaRequest())
) -> IndexSchemaResponse:
    """
    Index schema into vector store for semantic search.

    This endpoint extracts all schema elements (tables, columns, relationships)
    from the specified PostgreSQL schema, transforms them into embeddings,
    and stores them in the vector store.

    **Request Model**: `IndexSchemaRequest`
    - schema_name: PostgreSQL schema to index (default: from config)
    - replace_existing: If True, delete existing embeddings first

    **Response Model**: `IndexSchemaResponse`
    - tables_indexed, columns_indexed, relationships_indexed, total_documents

    **Admin operation**: This endpoint should be called after schema changes
    or when setting up the system for the first time.

    **Possible Errors**:
    - 422: Invalid request parameters
    - 500: Schema extraction or embedding generation failed
    - 503: Database or embedding service unavailable
    """
    trace_id = get_trace_id()

    # Use database default schema if not provided in request
    schema_name = request.schema_name or settings.database.default_schema

    logger.info(
        "Schema indexing requested",
        schema_name=schema_name,
        replace_existing=request.replace_existing,
        trace_id=trace_id
    )

    # Index schema via vector service
    stats = await vector_service.index_schema(
        schema=schema_name,
        replace_existing=request.replace_existing
    )

    logger.info(
        "Schema indexing completed",
        schema_name=stats.schema_name,
        tables_indexed=stats.tables_indexed,
        columns_indexed=stats.columns_indexed,
        relationships_indexed=stats.relationships_indexed,
        total_documents=stats.total_documents,
        trace_id=trace_id
    )

    return IndexSchemaResponse(
        trace_id=trace_id,
        schema_name=stats.schema_name,
        tables_indexed=stats.tables_indexed,
        columns_indexed=stats.columns_indexed,
        relationships_indexed=stats.relationships_indexed,
        total_documents=stats.total_documents,
    )


@app.post(
    "/api/v1/schema/search",
    response_model=VectorSchemaSearchResponse,
    tags=["Schema Retrieval"],
    responses={
        **{k: v for k, v in ERROR_RESPONSES.items() if k in [422, 500, 503]},
    },
)
async def search_schema(
    request: SchemaSearchRequest,
    vector_service: VectorServiceDep,
    settings: SettingsDep
) -> VectorSchemaSearchResponse:
    """
    Search schema using semantic similarity.

    This endpoint performs vector similarity search on indexed schema elements,
    returning the most relevant tables, columns, and relationships for a given query.

    **Request Model**: `SchemaSearchRequest`
    - query: Natural language search query (required)
    - top_k: Number of results to return (default: from config)
    - node_types: Filter by node type ["table", "column", "relationship"]
    - min_similarity: Minimum similarity threshold (0.0-1.0)

    **Response Model**: `VectorSchemaSearchResponse`
    - results: List of SchemaSearchResult with content, metadata, similarity_score
    - result_count: Number of results returned

    **Debugging endpoint**: Useful for testing RAG retrieval quality and understanding
    which schema elements are retrieved for specific queries.

    **Possible Errors**:
    - 422: Invalid query or search parameters
    - 500: Embedding or search operation failed
    - 503: Vector store or embedding service unavailable
    """
    trace_id = get_trace_id()

    # Use config defaults if not provided in request
    top_k = request.top_k if request.top_k is not None else settings.vector_store.default_k
    min_similarity = request.min_similarity if request.min_similarity is not None else settings.vector_store.default_min_similarity

    logger.info(
        "Schema search requested",
        query=request.query,
        k=top_k,
        min_similarity=min_similarity,
        trace_id=trace_id
    )

    # Search schema via vector service
    results = await vector_service.search_schema(
        query=request.query,
        k=top_k,
        schema_name=request.schema_name,
        node_types=request.node_types,
        min_similarity=min_similarity
    )

    # Convert VectorSearchResult to SchemaSearchResult
    search_results = [
        SchemaSearchResult(
            content=result.content,
            metadata=result.metadata,
            similarity_score=result.similarity_score
        )
        for result in results
    ]

    logger.info(
        "Schema search completed",
        result_count=len(search_results),
        trace_id=trace_id
    )

    return VectorSchemaSearchResponse(
        trace_id=trace_id,
        query=request.query,
        results=search_results,
        result_count=len(search_results)
    )


@app.get("/api/v1/vector/stats", response_model=VectorStatsResponse, tags=["Vector Store"])
async def get_vector_stats(
    vector_service: VectorServiceDep
) -> VectorStatsResponse:
    """
    Get vector store statistics.

    Returns counts of indexed documents by type and schema.
    Useful for monitoring and verifying indexing operations.

    **Response Model**: `VectorStatsResponse`
    - total_documents: Total number of indexed vectors
    - node_type_counts: Dict mapping node_type -> count
    - schema_counts: Dict mapping schema_name -> count
    """
    trace_id = get_trace_id()
    logger.info("Vector stats requested", trace_id=trace_id)

    # Get stats via vector service
    stats = await vector_service.get_collection_stats()

    logger.info(
        "Vector stats fetched",
        total_documents=stats.total_documents,
        trace_id=trace_id
    )

    return VectorStatsResponse(
        trace_id=trace_id,
        total_documents=stats.total_documents,
        node_type_counts=stats.node_type_counts,
        schema_counts=stats.schema_counts,
    )


@app.delete(
    "/api/v1/vector/collection",
    response_model=DropCollectionResponse,
    tags=["Vector Store Admin"],
    responses={
        **{k: v for k, v in ERROR_RESPONSES.items() if k in [400, 422, 500, 503]},
    },
)
async def drop_vector_collection(
    request: DropCollectionRequest,
    vector_service: VectorServiceDep
) -> DropCollectionResponse:
    """
    Drop vector collection (table and/or indexes).

    **DESTRUCTIVE ADMIN OPERATION**: This permanently deletes the vector store data.

    **Request Model**: `DropCollectionRequest`
    - confirm: Must be True to proceed (safety check)
    - drop_table: If True, drop entire table; if False, drop only HNSW index

    **Response Model**: `DropCollectionResponse`
    - success: Boolean indicating operation success
    - action: Description of action taken ("dropped_table" or "dropped_index")
    - table_name, index_name, indexes_dropped

    **Options**:
    - `drop_table=True`: Drops the entire `schema_embeddings` table, including all indexes (CASCADE)
    - `drop_table=False`: Drops only the HNSW index, keeping the table and data

    **Safety requirement**: Must set `confirm=true` in request body to proceed.

    **When to use**:
    - Before changing embedding dimensions
    - To reset the vector store
    - To drop and recreate indexes with different parameters
    - During development/testing

    **Warning**: After dropping the table, you must call `POST /api/v1/schema/index` to rebuild.

    **Possible Errors**:
    - 400: Missing confirmation (confirm=true required)
    - 422: Invalid request parameters
    - 500: Database operation failed
    - 503: Database service unavailable
    """
    trace_id = get_trace_id()

    # Safety check - require explicit confirmation
    if not request.confirm:
        from .domain.errors import BadRequestError
        logger.warning(
            "Drop collection attempted without confirmation",
            trace_id=trace_id
        )
        raise BadRequestError(
            message="Must set confirm=true to drop collection",
            details={"field": "confirm", "required_value": True}
        )

    logger.info(
        "Drop collection requested",
        drop_table=request.drop_table,
        trace_id=trace_id
    )

    # Drop collection via vector service
    result = await vector_service.drop_collection(drop_table=request.drop_table)

    logger.info(
        "Vector collection dropped",
        action=result.action,
        trace_id=trace_id
    )

    return DropCollectionResponse(
        trace_id=trace_id,
        success=True,
        action=result.action,
        table_name=result.table_name,
        index_name=result.index_name,
        indexes_dropped=result.indexes_dropped,
    )


# -------------------------
# NL2SQL Query Endpoint
# -------------------------

@app.post(
    "/nl2sql/query",
    response_model=NL2SQLQueryResponse,
    tags=["NL2SQL"],
    responses={
        **{k: v for k, v in ERROR_RESPONSES.items() if k in [400, 422, 429, 500, 503]},
    },
)
async def nl2sql_query(
    request: NL2SQLQueryRequest,
    nl2sql_service: NL2SQLServiceDep,
    settings: SettingsDep,
) -> NL2SQLQueryResponse:
    """
    Convert natural language query to SQL and execute it.

    This endpoint implements a production-safe NL2SQL pipeline:

    1. **Retrieval**: Vector search for relevant columns and relationships
    2. **Filtering**: Deterministic filtering (code-based, not LLM)
    3. **Context**: Attach table descriptions
    4. **Generation**: Single LLM call for SQL generation
    5. **Validation**: Static checks + EXPLAIN validation
    6. **Execution**: Read-only SQL execution

    **Request Model**: `NL2SQLQueryRequest`
    - query: Natural language question (required)
    - schema_name: PostgreSQL schema to query (default: "public")
    - limit: Maximum rows to return (capped by config max_row_limit)
    - timeout_seconds: SQL execution timeout

    **Response Model**: `NL2SQLQueryResponse`
    - status: QueryStatus (completed/failed)
    - sql: Generated SQL query (if successful)
    - results: QueryExecutionResult with rows, column_names, row_count
    - grounding: SchemaGrounding (tables, columns, relationships used)
    - provenance: SchemaProvenance (retrieved_nodes, validation_steps, retries)
    - error_message, error_step: Details if failed

    **Key Safety Features**:
    - SELECT queries only (no INSERT/UPDATE/DELETE/DDL)
    - Read-only database connection
    - Hard caps on tables/columns/relationships
    - SQL validation before execution
    - Retry loop with error feedback

    **Possible Errors**:
    - 400: Invalid query format or empty query
    - 422: SQL validation failed (unsafe SQL, syntax error)
    - 429: LLM rate limit exceeded
    - 500: SQL generation failed after retries
    - 503: Database, LLM, or embedding service unavailable
    """
    trace_id = get_trace_id()

    logger.info(
        "NL2SQL query requested",
        query_length=len(request.query),
        schema_name=request.schema_name,
        limit=request.limit,
        trace_id=trace_id,
    )

    # Validate row limit against config
    row_limit = min(
        request.limit or settings.nl2sql.default_row_limit,
        settings.nl2sql.max_row_limit,
    )

    # Execute the NL2SQL pipeline
    response = await nl2sql_service.execute_query(
        user_query=request.query,
        schema_name=request.schema_name or "public",
        row_limit=row_limit,
        timeout_seconds=request.timeout_seconds,
    )

    logger.info(
        "NL2SQL query completed",
        status=response.status,
        sql_generated=response.sql is not None,
        rows_returned=response.results.row_count if response.results else 0,
        retries=response.provenance.retries,
        trace_id=trace_id,
    )

    return response


# FastAPI app is now ready to be imported and run by uvicorn or other ASGI servers
# Use scripts/run_dev.py for development or poetry run uvicorn src.backend.main:app