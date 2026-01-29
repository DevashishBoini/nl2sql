"""
API request models for the NL2SQL system.

These models define the structure for all incoming API requests,
ensuring type safety and validation at API boundaries.

All fields include detailed descriptions that appear in Swagger/OpenAPI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for natural language to SQL query conversion."""

    query: str = Field(
        ...,
        description="Natural language query to convert to SQL. "
                    "Example: 'Show me the top 10 customers by total payments'",
        min_length=1
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of result rows to return. "
                    "If not specified, uses the default limit from configuration. "
                    "Value is capped at the maximum allowed limit.",
        ge=1,
    )
    include_execution_plan: bool = Field(
        default=False,
        description="If true, includes the SQL EXPLAIN output in the response. "
                    "Useful for debugging query performance."
    )


class SchemaIndexRequest(BaseModel):
    """Request model for schema indexing operations."""

    schema_name: Optional[str] = Field(
        default=None,
        description="PostgreSQL schema name to index (e.g., 'public', 'myschema'). "
                    "If not provided, indexes all accessible schemas."
    )
    force_reindex: bool = Field(
        default=False,
        description="If true, forces re-indexing even if schema is already indexed. "
                    "Existing vectors for the schema will be replaced."
    )
    include_sample_values: bool = Field(
        default=True,
        description="If true, extracts sample values from columns and stores them in metadata. "
                    "Sample values help the LLM understand data patterns and formats."
    )


class SchemaSearchRequest(BaseModel):
    """
    Request model for semantic schema search.

    Use this endpoint to test RAG retrieval quality and understand
    which schema elements match a given query.
    """

    query: str = Field(
        ...,
        description="Natural language query to search for relevant schema elements. "
                    "Example: 'customer email addresses' or 'payment transactions'. "
                    "The query is embedded and compared against indexed schema elements.",
        min_length=1,
        max_length=1000,
        json_schema_extra={"example": "customer payment amounts"}
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of most similar schema elements to return. "
                    "If not provided, uses the default from configuration (typically 5-15). "
                    "Higher values return more results but may include less relevant matches.",
        ge=1,
        le=50,
        json_schema_extra={"example": 10}
    )
    node_types: Optional[List[str]] = Field(
        default=None,
        description="Filter results by schema element types. "
                    "Valid values: 'table', 'column', 'relationship'. "
                    "If not provided, returns all types. "
                    "Example: ['column', 'relationship'] to exclude table-level results.",
        json_schema_extra={"example": ["column", "relationship"]}
    )
    min_similarity: Optional[float] = Field(
        default=None,
        description="Minimum cosine similarity score (0.0 to 1.0) to include in results. "
                    "Higher values (e.g., 0.7) return only highly relevant matches. "
                    "Lower values (e.g., 0.3) return more results including partial matches. "
                    "If not provided, uses the default from configuration.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.5}
    )
    schema_name: Optional[str] = Field(
        default=None,
        description="Filter results to a specific PostgreSQL schema (e.g., 'public'). "
                    "If not provided, searches across all indexed schemas.",
        json_schema_extra={"example": "public"}
    )


class IndexSchemaRequest(BaseModel):
    """
    Request model for indexing database schema into vector store.

    This is an admin operation that extracts schema metadata (tables, columns,
    relationships) and creates embeddings for semantic search.
    """

    schema_name: Optional[str] = Field(
        default=None,
        description="PostgreSQL schema name to index (e.g., 'public', 'sales'). "
                    "If not provided, uses the default schema from database configuration. "
                    "Only tables in this schema will be indexed.",
        json_schema_extra={"example": "public"}
    )
    replace_existing: bool = Field(
        default=True,
        description="If true, deletes all existing vectors for this schema before indexing. "
                    "Recommended for clean re-indexing after schema changes. "
                    "If false, new vectors are added alongside existing ones (may cause duplicates).",
        json_schema_extra={"example": True}
    )


class DropCollectionRequest(BaseModel):
    """
    Request model for dropping vector collection (DESTRUCTIVE operation).

    Requires explicit confirmation to prevent accidental data loss.
    """

    drop_table: bool = Field(
        default=True,
        description="If true, drops the entire schema_embeddings table (CASCADE). "
                    "All indexes and data are permanently deleted. "
                    "If false, only drops the HNSW index, keeping the table and data intact.",
        json_schema_extra={"example": True}
    )
    confirm: bool = Field(
        ...,
        description="Safety confirmation flag. MUST be set to true to proceed with deletion. "
                    "This prevents accidental data loss from mistyped commands.",
        json_schema_extra={"example": True}
    )


# =============================================================================
# NL2SQL Query Models
# =============================================================================

class NL2SQLQueryRequest(BaseModel):
    """
    Request model for the main NL2SQL query endpoint.

    Converts a natural language question into SQL, validates it,
    and executes it against the database.
    """

    query: str = Field(
        ...,
        description="Natural language question to convert to SQL. "
                    "Be specific about what data you want and any filters/sorting. "
                    "Examples: "
                    "'Show me the top 10 customers by total payment amount', "
                    "'List all films rated PG-13 released after 2005', "
                    "'How many rentals did each store have last month?'",
        min_length=1,
        max_length=2000,
        json_schema_extra={"example": "Show me the top 10 customers by total payment amount"}
    )
    limit: Optional[int] = Field(
        default=100,
        description="Maximum number of rows to return from the SQL query result. "
                    "Default is 100 rows. "
                    "Value is capped at the maximum allowed limit (typically 1000). "
                    "The generated SQL will include a LIMIT clause with this value.",
        ge=1,
        json_schema_extra={"example": 100}
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        description="Maximum time in seconds to wait for SQL query execution. "
                    "Queries exceeding this timeout will be cancelled. "
                    "Default is 30 seconds. Maximum allowed is 60 seconds.",
        ge=1,
        le=60,
        json_schema_extra={"example": 30}
    )
    schema_name: Optional[str] = Field(
        default="public",
        description="PostgreSQL schema to query (e.g., 'public', 'sales'). "
                    "The generated SQL will only reference tables in this schema. "
                    "Default is 'public'.",
        json_schema_extra={"example": "public"}
    )
