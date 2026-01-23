"""
API response models for the NL2SQL system.

These models define the structure for all outgoing API responses,
ensuring consistent response formats and type safety.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from .base_enums import QueryStatus, NodeType


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