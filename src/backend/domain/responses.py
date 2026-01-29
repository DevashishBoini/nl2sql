"""
API response models for the NL2SQL system.

These models define the structure for all outgoing API responses,
ensuring consistent response formats and type safety.
"""

import json
import logging
import re
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, Self
from datetime import datetime
from .base_enums import QueryStatus, NodeType
from .schema_nodes import TableNode, ColumnNode, RelationshipNode

logger = logging.getLogger(__name__)


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


class LLMGenerationResult(BaseModel):
    """
    Result of LLM SQL generation attempt.

    Represents the parsed response from the LLM, indicating whether
    SQL generation was successful or failed with a reason.
    """

    success: bool = Field(..., description="Whether SQL generation succeeded")
    sql: Optional[str] = Field(default=None, description="Generated SQL (if success=True)")
    reason: Optional[str] = Field(default=None, description="Failure reason (if success=False)")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate that successful results have SQL and failed results have reason."""
        if self.success and not self.sql:
            # Allow success=True with no SQL only if explicitly set
            pass  # Will be caught in generation retry loop
        return self

    @classmethod
    def from_llm_response(cls, response: str) -> "LLMGenerationResult":
        """
        Parse LLM response string into result object.

        Handles:
        - Empty/whitespace response: returns failure with clear message
        - JSON format: {"success": true, "sql": "..."} or {"success": false, "reason": "..."}
        - Markdown-wrapped JSON: ```json ... ```
        - Raw SQL fallback: SELECT ... (treated as success)

        Args:
            response: Raw LLM response string

        Returns:
            LLMGenerationResult with parsed data
        """
        # Handle empty or whitespace-only responses
        if not response or not response.strip():
            logger.warning("LLM returned empty response")
            return cls(
                success=False,
                reason="LLM returned empty response - may indicate rate limiting or model unavailability"
            )

        # Initialize cleaned before try block to satisfy type checker
        cleaned: str = ""

        try:
            cleaned = cls._strip_markdown(response)

            # Check if cleaned content is empty (e.g., just markdown fences with no content)
            if not cleaned:
                logger.warning("LLM response contained only markdown fences with no content")
                return cls(
                    success=False,
                    reason="LLM response was empty after removing markdown formatting"
                )

            # Try direct JSON parsing first
            try:
                data = json.loads(cleaned)
                return cls(
                    success=data.get("success", False),
                    sql=data.get("sql"),
                    reason=data.get("reason"),
                )
            except json.JSONDecodeError:
                # Try to extract JSON object from the response
                json_obj = cls._extract_json_object(cleaned)
                if json_obj:
                    return cls(
                        success=json_obj.get("success", False),
                        sql=json_obj.get("sql"),
                        reason=json_obj.get("reason"),
                    )
                raise  # Re-raise to fall through to raw SQL extraction

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse LLM response as JSON, attempting raw SQL extraction",
                extra={"error": str(e), "cleaned_preview": cleaned[:200] if cleaned else ""},
            )
            sql = cls._extract_raw_sql(response)
            if sql:
                return cls(success=True, sql=sql)

            # Provide more helpful error message with cleaned content
            preview = cleaned[:150] if cleaned else response[:150] if response else "(empty)"
            return cls(
                success=False,
                reason=f"LLM response was not valid JSON or SQL. Preview: {preview}"
            )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Strip markdown code fence from text.

        Handles various formats:
        - ```json\n{...}\n```
        - ```\n{...}\n```
        - ``` json\n{...}```
        - Mixed whitespace and newlines
        """
        cleaned = text.strip()

        # Use regex to handle various markdown fence formats
        # Pattern matches: ```json or ``` followed by optional whitespace/newline
        # and captures the content until closing ```
        markdown_pattern = re.compile(
            r'^```(?:json)?\s*\n?(.*?)\n?```$',
            re.DOTALL | re.IGNORECASE
        )

        match = markdown_pattern.match(cleaned)
        if match:
            return match.group(1).strip()

        # Fallback: manual stripping for edge cases
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract a JSON object from text that may contain extra content.

        Finds the first { and last } and tries to parse what's between.
        """
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1 or end <= start:
            return None

        json_str = text[start:end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_raw_sql(response: str) -> Optional[str]:
        """Fallback: extract SQL from non-JSON response."""
        sql = response.strip()

        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        sql = sql.strip()

        if sql.upper().startswith("SELECT"):
            return sql
        return None


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


# -------------------------
# NL2SQL Response Models
# -------------------------

class RetrievedNode(BaseModel):
    """A schema node retrieved from vector search."""

    node_type: NodeType = Field(..., description="Type of node (column, relationship, table)")
    content: str = Field(..., description="Node content used for embedding")
    table_name: Optional[str] = Field(None, description="Table name")
    column_name: Optional[str] = Field(None, description="Column name (for column nodes)")
    similarity_score: float = Field(..., description="Similarity score from vector search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full node metadata")


class FilteredColumn(BaseModel):
    """A column selected after deterministic filtering."""

    table_name: str = Field(..., description="Table containing the column")
    column_name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="PostgreSQL data type")
    description: Optional[str] = Field(None, description="Column description")
    is_primary_key: bool = Field(default=False, description="Whether column is primary key")
    is_foreign_key: bool = Field(default=False, description="Whether column is foreign key")


class FilteredRelationship(BaseModel):
    """A relationship (FK) selected after deterministic filtering."""

    from_table: str = Field(..., description="Source table (child with FK)")
    from_column: str = Field(..., description="Column in source table")
    to_table: str = Field(..., description="Target table (parent with PK)")
    to_column: str = Field(..., description="Column in target table")
    constraint_name: Optional[str] = Field(None, description="FK constraint name")


class FilteredTable(BaseModel):
    """A table derived from filtered columns."""

    table_name: str = Field(..., description="Table name")
    description: Optional[str] = Field(None, description="Table description")
    column_count: int = Field(..., description="Number of columns selected from this table")


class SchemaGrounding(BaseModel):
    """
    Grounding information showing exactly which schema elements were used.

    This provides transparency into what the LLM was allowed to use.
    """

    tables: List[FilteredTable] = Field(
        default_factory=list,
        description="Tables derived from selected columns"
    )
    columns: List[FilteredColumn] = Field(
        default_factory=list,
        description="Columns selected for SQL generation"
    )
    relationships: List[FilteredRelationship] = Field(
        default_factory=list,
        description="FK relationships between selected tables"
    )


class ValidationStep(BaseModel):
    """A single validation step result."""

    step_name: str = Field(..., description="Name of validation step")
    passed: bool = Field(..., description="Whether validation passed")
    message: Optional[str] = Field(None, description="Validation message or error")
    sql_attempted: Optional[str] = Field(None, description="SQL that was validated")


class SchemaProvenance(BaseModel):
    """
    Provenance information tracking the full pipeline execution.

    This provides auditability and debugging information.
    Note: filtered_schema is available at the top-level 'grounding' field.
    """

    retrieved_nodes: List[RetrievedNode] = Field(
        default_factory=list,
        description="Raw nodes retrieved from vector search (before filtering)"
    )
    validation_steps: List[ValidationStep] = Field(
        default_factory=list,
        description="Validation steps executed"
    )
    retries: int = Field(default=0, description="Number of SQL generation retries")


class NL2SQLQueryResponse(BaseModel):
    """Response model for NL2SQL query endpoint."""

    trace_id: str = Field(..., description="Unique trace ID for this request")
    status: QueryStatus = Field(..., description="Query execution status")

    # Original query
    query: str = Field(..., description="Original natural language query from user")

    # The generated SQL
    sql: Optional[str] = Field(
        None,
        description="Final generated SQL query (if successful)"
    )

    # Execution results
    results: Optional[QueryExecutionResult] = Field(
        None,
        description="SQL execution results (if successful)"
    )

    # Error information
    error_message: Optional[str] = Field(
        None,
        description="Error message if query failed"
    )
    error_step: Optional[str] = Field(
        None,
        description="Pipeline step where error occurred"
    )

    # Grounding: What schema elements were used
    grounding: SchemaGrounding = Field(
        default_factory=SchemaGrounding,
        description="Schema elements used for SQL generation"
    )

    # Provenance: Full audit trail
    provenance: SchemaProvenance = Field(
        default_factory=SchemaProvenance,
        description="Full provenance of pipeline execution"
    )

    # Timing
    total_time_ms: float = Field(..., description="Total request processing time in milliseconds")