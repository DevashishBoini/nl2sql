"""
FastAPI dependencies for dependency injection.

This module provides reusable dependencies that can be injected into
API route handlers for accessing shared resources like database clients,
storage clients, configuration, and other services.
"""

from typing import Annotated

from fastapi import Depends, Request

from ..infrastructure.database_client import DatabaseClient
from ..infrastructure.storage_client import StorageClient
from ..infrastructure.llm_client import LLMClient
from ..infrastructure.embedding_client import EmbeddingClient
from ..repositories.schema_repository import SchemaRepository
from ..services.schema_service import SchemaService
from ..config import Settings


def get_db_client(request: Request) -> DatabaseClient:
    """
    Dependency to get the database client from app state.

    Usage in routes:
        @app.get("/users")
        async def get_users(db: DatabaseClientDep):
            results = await db.execute_query_readonly("SELECT * FROM users LIMIT 10")
            return results

    Args:
        request: FastAPI request object

    Returns:
        DatabaseClient instance

    Raises:
        RuntimeError: If database client is not initialized
    """
    if not hasattr(request.app.state, "db_client"):
        raise RuntimeError("Database client not initialized")

    return request.app.state.db_client


def get_storage_client(request: Request) -> StorageClient:
    """
    Dependency to get the storage client from app state.

    Usage in routes:
        @app.get("/files")
        async def list_files(storage: StorageClientDep):
            files = await storage.list_files("documents")
            return {"files": files}

    Args:
        request: FastAPI request object

    Returns:
        StorageClient instance

    Raises:
        RuntimeError: If storage client is not initialized
    """
    if not hasattr(request.app.state, "storage_client"):
        raise RuntimeError("Storage client not initialized")

    return request.app.state.storage_client


def get_llm_client(request: Request) -> LLMClient:
    """
    Dependency to get the LLM client from app state.

    Usage in routes:
        @app.post("/generate")
        async def generate_text(prompt: str, llm: LLMClientDep):
            response = await llm.generate(prompt)
            return {"response": response}

    Args:
        request: FastAPI request object

    Returns:
        LLMClient instance

    Raises:
        RuntimeError: If LLM client is not initialized
    """
    if not hasattr(request.app.state, "llm_client"):
        raise RuntimeError("LLM client not initialized")

    return request.app.state.llm_client


def get_embedding_client(request: Request) -> EmbeddingClient:
    """
    Dependency to get the embedding client from app state.

    Usage in routes:
        @app.post("/embed")
        async def embed_text(text: str, embedding: EmbeddingClientDep):
            vector = await embedding.embed_text(text)
            return {"vector": vector, "dimension": len(vector)}

    Args:
        request: FastAPI request object

    Returns:
        EmbeddingClient instance

    Raises:
        RuntimeError: If embedding client is not initialized
    """
    if not hasattr(request.app.state, "embedding_client"):
        raise RuntimeError("Embedding client not initialized")

    return request.app.state.embedding_client


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

    db_client = request.app.state.db_client
    storage_client = getattr(request.app.state, "storage_client", None)
    settings = getattr(request.app.state, "settings", None)

    # Get YAML config from settings if available
    yaml_path = None
    yaml_bucket = None
    if settings:
        yaml_path = settings.storage.schema_yaml_path
        yaml_bucket = settings.storage.default_bucket

    schema_repo = SchemaRepository(db_client)
    schema_service = SchemaService(
        schema_repository=schema_repo,
        storage_client=storage_client,
        yaml_path=yaml_path,
        yaml_bucket=yaml_bucket
    )

    return schema_service


# Type aliases for cleaner dependency injection
# Required dependencies (raise error if not available)
DatabaseClientDep = Annotated[DatabaseClient, Depends(get_db_client)]
StorageClientDep = Annotated[StorageClient, Depends(get_storage_client)]
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
EmbeddingClientDep = Annotated[EmbeddingClient, Depends(get_embedding_client)]
SchemaServiceDep = Annotated[SchemaService, Depends(get_schema_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Optional dependencies (return None if not available - for health checks)
OptionalDatabaseClientDep = Annotated[DatabaseClient | None, Depends(get_db_client_optional)]
OptionalStorageClientDep = Annotated[StorageClient | None, Depends(get_storage_client_optional)]
OptionalLLMClientDep = Annotated[LLMClient | None, Depends(get_llm_client_optional)]
OptionalEmbeddingClientDep = Annotated[EmbeddingClient | None, Depends(get_embedding_client_optional)]
