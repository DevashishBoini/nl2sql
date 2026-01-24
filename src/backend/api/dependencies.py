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
from ..config import Settings, get_settings


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


# Type aliases for cleaner dependency injection
DatabaseClientDep = Annotated[DatabaseClient, Depends(get_db_client)]
StorageClientDep = Annotated[StorageClient, Depends(get_storage_client)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
