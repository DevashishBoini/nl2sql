"""
API request models for the NL2SQL system.

These models define the structure for all incoming API requests,
ensuring type safety and validation at API boundaries.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for natural language to SQL query conversion."""

    query: str = Field(..., description="Natural language query to convert to SQL", min_length=1)
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of results to return",
        ge=1,
    )
    include_execution_plan: bool = Field(
        default=False,
        description="Whether to include SQL execution plan in response"
    )


class SchemaIndexRequest(BaseModel):
    """Request model for schema indexing operations."""

    schema_name: Optional[str] = Field(
        default=None,
        description="Specific schema to index (if None, indexes all schemas)"
    )
    force_reindex: bool = Field(
        default=False,
        description="Whether to force reindexing even if schema is already indexed"
    )
    include_sample_values: bool = Field(
        default=True,
        description="Whether to extract sample values from columns"
    )


class SchemaSearchRequest(BaseModel):
    """Request model for schema search operations."""

    query: str = Field(..., description="Search query for schema elements", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(
        default=None,
        description="Number of top similar schema elements to return (uses config default if not provided)",
        ge=1,
        le=50
    )
    node_types: Optional[List[str]] = Field(
        default=None,
        description="Filter results by node types (e.g., ['table', 'column', 'relationship'])"
    )
    min_similarity: Optional[float] = Field(
        default=None,
        description="Minimum similarity score threshold (uses config default if not provided)",
        ge=0.0,
        le=1.0
    )
    schema_name: Optional[str] = Field(
        default=None,
        description="Filter results by specific schema name"
    )


class IndexSchemaRequest(BaseModel):
    """Request model for indexing schema into vector store."""

    schema_name: Optional[str] = Field(
        default=None,
        description="PostgreSQL schema name to index (uses database default schema if not provided)"
    )
    replace_existing: bool = Field(
        default=True,
        description="Whether to delete existing vectors before indexing. "
                    "Set to true for clean re-indexing, false to preserve existing data."
    )


class DropCollectionRequest(BaseModel):
    """Request model for dropping vector collection."""

    drop_table: bool = Field(
        default=True,
        description="If True, drop entire table. If False, only drop HNSW index."
    )
    confirm: bool = Field(
        ...,
        description="Must be set to True to confirm deletion (safety check)"
    )


# -------------------------
# NL2SQL Query Models
# -------------------------

class NL2SQLQueryRequest(BaseModel):
    """Request model for NL2SQL query endpoint."""

    query: str = Field(
        ...,
        description="Natural language query to convert to SQL",
        min_length=1,
        max_length=2000
    )
    limit: Optional[int] = Field(
        default=100,
        description="Maximum number of rows to return from SQL execution",
        ge=1
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        description="SQL execution timeout in seconds",
        ge=1,
        le=60
    )
    schema_name: Optional[str] = Field(
        default="public",
        description="PostgreSQL schema to query"
    )