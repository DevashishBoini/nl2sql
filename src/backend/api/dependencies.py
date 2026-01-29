"""
FastAPI dependencies for dependency injection.

This module provides reusable dependencies that can be injected into
API route handlers following proper layered architecture:
- Services (SchemaService, VectorService) for business logic
- Settings for configuration
- Optional client dependencies for health checks only

Routes should depend on services, not infrastructure clients directly.
"""

from typing import Annotated

from fastapi import Depends, Request

from ..infrastructure.database_client import DatabaseClient
from ..infrastructure.storage_client import StorageClient
from ..infrastructure.llm_client import LLMClient
from ..infrastructure.embedding_client import EmbeddingClient
from ..repositories.schema_repository import SchemaRepository
from ..repositories.vector_repository import VectorRepository
from ..repositories.schema_filtering import SchemaFilteringRepository
from ..repositories.sql_generation import SQLGenerationRepository
from ..repositories.sql_validation import SQLValidationRepository
from ..repositories.sql_execution import SQLExecutionRepository
from ..services.schema_service import SchemaService
from ..services.vector_service import VectorService
from ..services.nl2sql_service import NL2SQLService
from ..config import Settings, NL2SQLConfig


def get_settings(request: Request) -> Settings:
    """
    Dependency to get the settings from app state.

    Usage in routes:
        @app.get("/config")
        async def get_config(settings: SettingsDep):
            return {"log_level": settings.app.log_level}

    Args:
        request: FastAPI request object

    Returns:
        Settings instance

    Raises:
        RuntimeError: If settings are not initialized
    """
    if not hasattr(request.app.state, "settings"):
        raise RuntimeError("Settings not initialized")

    return request.app.state.settings


# Optional dependency getters for health checks and endpoints that need graceful degradation
def get_db_client_optional(request: Request) -> DatabaseClient | None:
    """Get database client if available, None otherwise."""
    return getattr(request.app.state, "db_client", None)


def get_storage_client_optional(request: Request) -> StorageClient | None:
    """Get storage client if available, None otherwise."""
    return getattr(request.app.state, "storage_client", None)


def get_llm_client_optional(request: Request) -> LLMClient | None:
    """Get LLM client if available, None otherwise."""
    return getattr(request.app.state, "llm_client", None)


def get_embedding_client_optional(request: Request) -> EmbeddingClient | None:
    """Get embedding client if available, None otherwise."""
    return getattr(request.app.state, "embedding_client", None)


def get_schema_service(request: Request) -> SchemaService:
    """
    Dependency to get a SchemaService instance.

    This creates a SchemaService with SchemaRepository, following proper
    layered architecture (API → Service → Repository → Infrastructure).

    Usage in routes:
        @app.get("/schema/tables")
        async def get_tables(schema_service: SchemaServiceDep):
            tables = await schema_service.get_all_tables()
            return {"tables": tables}

    Args:
        request: FastAPI request object

    Returns:
        SchemaService instance

    Raises:
        RuntimeError: If database client is not initialized
    """
    if not hasattr(request.app.state, "db_client"):
        raise RuntimeError("Database client not initialized")
    if not hasattr(request.app.state, "settings"):
        raise RuntimeError("Settings not initialized")

    db_client = request.app.state.db_client
    storage_client = getattr(request.app.state, "storage_client", None)
    settings = request.app.state.settings

    # Get YAML config from settings
    yaml_path = settings.storage.schema_yaml_path
    yaml_bucket = settings.storage.default_bucket

    schema_repo = SchemaRepository(db_client, settings.schema_indexing)
    schema_service = SchemaService(
        schema_repository=schema_repo,
        storage_client=storage_client,
        yaml_path=yaml_path,
        yaml_bucket=yaml_bucket
    )

    return schema_service


def get_vector_service(request: Request) -> VectorService:
    """
    Dependency to get a VectorService instance.

    This creates a VectorService with full dependency tree:
    VectorService → VectorRepository → DatabaseClient (shared)
    VectorService → SchemaService → SchemaRepository → DatabaseClient (shared)

    Uses shared DatabaseClient for both schema and vector operations.

    Usage in routes:
        @app.post("/schema/index")
        async def index_schema(vector_service: VectorServiceDep):
            stats = await vector_service.index_schema(schema="public")
            return stats

    Args:
        request: FastAPI request object

    Returns:
        VectorService instance

    Raises:
        RuntimeError: If required clients are not initialized
    """
    if not hasattr(request.app.state, "db_client"):
        raise RuntimeError("Database client not initialized")
    if not hasattr(request.app.state, "embedding_client"):
        raise RuntimeError("Embedding client not initialized")
    if not hasattr(request.app.state, "settings"):
        raise RuntimeError("Settings not initialized")

    db_client = request.app.state.db_client
    embedding_client = request.app.state.embedding_client
    storage_client = getattr(request.app.state, "storage_client", None)
    settings = request.app.state.settings

    # Build SchemaService (needed for indexing)
    yaml_path = settings.storage.schema_yaml_path
    yaml_bucket = settings.storage.default_bucket

    schema_repo = SchemaRepository(db_client, settings.schema_indexing)
    schema_service = SchemaService(
        schema_repository=schema_repo,
        storage_client=storage_client,
        yaml_path=yaml_path,
        yaml_bucket=yaml_bucket
    )

    # Build VectorRepository with shared DatabaseClient
    vector_repo = VectorRepository(
        db_client=db_client,
        embedding_client=embedding_client,
        config=settings.vector_store
    )

    # Build VectorService
    batch_size = settings.vector_store.batch_size
    vector_service = VectorService(
        vector_repository=vector_repo,
        schema_service=schema_service,
        batch_size=batch_size
    )

    return vector_service


def get_nl2sql_service(request: Request) -> NL2SQLService:
    """
    Dependency to get an NL2SQLService instance.

    This creates an NL2SQLService with full repository tree:
    NL2SQLService (orchestrator)
      ├── VectorRepository (schema retrieval)
      ├── SchemaFilteringRepository (deterministic filtering)
      ├── SQLGenerationRepository (LLM-based generation)
      ├── SQLValidationRepository (SQL validation)
      └── SQLExecutionRepository (SQL execution)

    Usage in routes:
        @app.post("/nl2sql/query")
        async def nl2sql_query(nl2sql_service: NL2SQLServiceDep):
            response = await nl2sql_service.execute_query(query)
            return response

    Args:
        request: FastAPI request object

    Returns:
        NL2SQLService instance

    Raises:
        RuntimeError: If required clients are not initialized
    """
    if not hasattr(request.app.state, "db_client"):
        raise RuntimeError("Database client not initialized")
    if not hasattr(request.app.state, "embedding_client"):
        raise RuntimeError("Embedding client not initialized")
    if not hasattr(request.app.state, "llm_client"):
        raise RuntimeError("LLM client not initialized")
    if not hasattr(request.app.state, "settings"):
        raise RuntimeError("Settings not initialized")

    db_client = request.app.state.db_client
    embedding_client = request.app.state.embedding_client
    llm_client = request.app.state.llm_client
    settings = request.app.state.settings

    # Build NL2SQL config from settings
    nl2sql_config = NL2SQLConfig(
        retrieval_top_k=settings.nl2sql.retrieval_top_k,
        max_tables=settings.nl2sql.max_tables,
        max_columns=settings.nl2sql.max_columns,
        max_relationships=settings.nl2sql.max_relationships,
        max_retries=settings.nl2sql.max_retries,
        default_row_limit=settings.nl2sql.default_row_limit,
        max_row_limit=settings.nl2sql.max_row_limit,
        query_timeout_seconds=settings.nl2sql.query_timeout_seconds,
    )

    # Build repositories
    vector_repo = VectorRepository(
        db_client=db_client,
        embedding_client=embedding_client,
        config=settings.vector_store
    )

    schema_filtering_repo = SchemaFilteringRepository(config=nl2sql_config)

    sql_generation_repo = SQLGenerationRepository(llm_client=llm_client, config=settings.llm)

    sql_validation_repo = SQLValidationRepository(db_client=db_client)

    sql_execution_repo = SQLExecutionRepository(db_client=db_client)

    # Build NL2SQLService (thin orchestrator)
    nl2sql_service = NL2SQLService(
        vector_repository=vector_repo,
        schema_filtering_repository=schema_filtering_repo,
        sql_generation_repository=sql_generation_repo,
        sql_validation_repository=sql_validation_repo,
        sql_execution_repository=sql_execution_repo,
        config=nl2sql_config,
    )

    return nl2sql_service


# Type aliases for cleaner dependency injection
# Service dependencies (used in API routes)
SchemaServiceDep = Annotated[SchemaService, Depends(get_schema_service)]
VectorServiceDep = Annotated[VectorService, Depends(get_vector_service)]
NL2SQLServiceDep = Annotated[NL2SQLService, Depends(get_nl2sql_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Optional client dependencies (used in health checks)
OptionalDatabaseClientDep = Annotated[DatabaseClient | None, Depends(get_db_client_optional)]
OptionalStorageClientDep = Annotated[StorageClient | None, Depends(get_storage_client_optional)]
OptionalLLMClientDep = Annotated[LLMClient | None, Depends(get_llm_client_optional)]
OptionalEmbeddingClientDep = Annotated[EmbeddingClient | None, Depends(get_embedding_client_optional)]
