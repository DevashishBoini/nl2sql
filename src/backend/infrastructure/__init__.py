"""
Infrastructure layer for external integrations.

This module contains clients for external services like databases,
LLM providers, vector stores, and other third-party integrations.
"""

from .database_client import DatabaseClient
from .storage_client import StorageClient

__all__ = ["DatabaseClient", "StorageClient"]
