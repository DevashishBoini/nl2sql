"""
API response models for the NL2SQL system.

These models define the structure for all outgoing API responses,
ensuring consistent response formats and type safety.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from .base_enums import QueryStatus, NodeType
from .schema_nodes import TableNode, ColumnNode, RelationshipNode


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status", examples=["healthy", "degraded", "unhealthy"])
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    database_status: str = Field(..., description="Database connection status")
    vector_store_status: str = Field(..., description="Vector store status")
    llm_service_status: str = Field(..., description="LLM service status")
    embedding_service_status: str = Field(..., description="Embedding service status")


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Trace ID for debugging"
    )
    timestamp: datetime = Field(..., description="Error timestamp")


class QueryExecutionResult(BaseModel):
    """Query execution result with metadata."""

    rows: List[Dict[str, Any]] = Field(..., description="Query result rows")
    column_names: List[str] = Field(..., description="Column names in result set")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")
    was_limited: bool = Field(..., description="Whether results were limited due to row limit")


class QueryResponse(BaseModel):
    """Response model for natural language to SQL query conversion."""

    trace_id: str = Field(..., description="Unique trace ID for this query")
    status: QueryStatus = Field(..., description="Query execution status")
    sql: Optional[str] = Field(default=None, description="Generated SQL query")
    results: Optional[QueryExecutionResult] = Field(
        default=None,
        description="Query execution results (if successful)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message (if status is failed)"
    )
    execution_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="SQL execution plan (if requested)"
    )
    retrieved_schema_count: Optional[int] = Field(
        default=None,
        description="Number of schema elements used for generation"
    )
    generation_time_ms: Optional[float] = Field(
        default=None,
        description="SQL generation time in milliseconds"
    )
    total_time_ms: float = Field(..., description="Total request processing time in milliseconds")


class SchemaNodeResult(BaseModel):
    """Schema node search result with similarity score."""

    node_type: NodeType = Field(..., description="Type of schema node")
    content: str = Field(..., description="Schema node content")
    metadata: Dict[str, Any] = Field(..., description="Schema node metadata")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")


class SchemaSearchResponse(BaseModel):
    """Response model for schema search operations."""

    query: str = Field(..., description="Original search query")
    results: List[SchemaNodeResult] = Field(..., description="Matching schema elements")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")


class IndexedSchemaStats(BaseModel):
    """Statistics about indexed schema elements."""

    total_tables: int = Field(..., description="Number of tables indexed")
    total_columns: int = Field(..., description="Number of columns indexed")
    total_relationships: int = Field(..., description="Number of relationships indexed")
    schemas_indexed: List[str] = Field(..., description="List of schema names indexed")


class SchemaIndexResponse(BaseModel):
    """Response model for schema indexing operations."""

    success: bool = Field(..., description="Whether indexing completed successfully")
    message: str = Field(..., description="Human-readable status message")
    stats: Optional[IndexedSchemaStats] = Field(
        default=None,
        description="Indexing statistics (if successful)"
    )
    indexing_time_ms: float = Field(..., description="Total indexing time in milliseconds")
    errors: Optional[List[str]] = Field(
        default=None,
        description="List of errors encountered during indexing"
    )


class SchemaSummaryResponse(BaseModel):
    """Response model for schema summary endpoint."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    schema_name: str = Field(..., description="Name of the database schema")
    table_count: int = Field(..., description="Total number of tables in the schema")
    relationship_count: int = Field(..., description="Total number of foreign key relationships")
    tables: List[TableNode] = Field(..., description="List of all tables in the schema")
    relationships: List[RelationshipNode] = Field(..., description="List of all relationships in the schema")
    sample_table: Optional[str] = Field(
        default=None,
        description="Name of the sample table (first table with columns shown)"
    )
    sample_columns: List[ColumnNode] = Field(
        default_factory=list,
        description="Sample columns from the first table (includes sample values)"
    )


# -------------------------
# Vector Store Response Models
# -------------------------

class VectorSearchResult(BaseModel):
    """
    Single result from a vector similarity search.

    Represents a document retrieved from the vector store,
    including its content, metadata, and similarity score.
    Used internally by vector operations and as part of search responses.
    """
    content: str = Field(..., description="Document content text")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")


class IndexSchemaResponse(BaseModel):
    """Response model for vector store schema indexing."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    schema_name: str = Field(..., description="Name of schema that was indexed")
    tables_indexed: int = Field(..., description="Number of tables indexed")
    columns_indexed: int = Field(..., description="Number of columns indexed")
    relationships_indexed: int = Field(..., description="Number of relationships indexed")
    total_documents: int = Field(..., description="Total documents added to vector store")


class SchemaSearchResult(BaseModel):
    """Single schema search result with similarity score."""

    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")


class VectorSchemaSearchResponse(BaseModel):
    """Response model for vector store schema search."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    query: str = Field(..., description="Original search query")
    results: List[SchemaSearchResult] = Field(..., description="Search results")
    result_count: int = Field(..., description="Number of results returned")


class VectorStatsResponse(BaseModel):
    """Response model for vector store statistics."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    total_documents: int = Field(..., description="Total documents in vector store")
    node_type_counts: Dict[str, int] = Field(..., description="Document counts by node type")
    schema_counts: Dict[str, int] = Field(..., description="Document counts by schema")


class DropCollectionResponse(BaseModel):
    """Response model for dropping vector collection."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    action: str = Field(..., description="Action performed (dropped_table or dropped_index)")
    table_name: Optional[str] = Field(None, description="Name of table that was dropped")
    index_name: Optional[str] = Field(None, description="Name of index that was dropped")
    indexes_dropped: Optional[str] = Field(None, description="Description of indexes dropped")
    success: bool = Field(default=True, description="Whether operation succeeded")


# -------------------------
# Service Layer Models (used by services, no trace_id)
# -------------------------

class IndexSchemaStats(BaseModel):
    """Statistics returned by VectorService.index_schema()."""

    schema_name: str = Field(..., description="Name of schema that was indexed")
    tables_indexed: int = Field(..., description="Number of tables indexed")
    columns_indexed: int = Field(..., description="Number of columns indexed")
    relationships_indexed: int = Field(..., description="Number of relationships indexed")
    total_documents: int = Field(..., description="Total documents added to vector store")


class DropCollectionResult(BaseModel):
    """Result returned by VectorService.drop_collection()."""

    action: str = Field(..., description="Action performed (dropped_table or dropped_index)")
    table_name: Optional[str] = Field(None, description="Name of table that was dropped")
    index_name: Optional[str] = Field(None, description="Name of index that was dropped")
    indexes_dropped: Optional[str] = Field(None, description="Description of indexes dropped")


class VectorCollectionStats(BaseModel):
    """Statistics returned by VectorService.get_collection_stats()."""

    total_documents: int = Field(..., description="Total documents in vector store")
    node_type_counts: Dict[str, int] = Field(..., description="Document counts by node type")
    schema_counts: Dict[str, int] = Field(..., description="Document counts by schema")