"""
API request models for the NL2SQL system.

These models define the structure for all incoming API requests,
ensuring type safety and validation at API boundaries.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from .base_enums import NodeType


class QueryRequest(BaseModel):
    """Request model for natural language to SQL query conversion."""

    query: str = Field(..., description="Natural language query to convert to SQL", min_length=1)
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of results to return",
        ge=1,
        le=1000
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

    query: str = Field(..., description="Search query for schema elements", min_length=1)
    top_k: int = Field(
        default=15,
        description="Number of top similar schema elements to return",
        ge=1,
        le=50
    )
    node_types: Optional[List[NodeType]] = Field(
        default=None,
        description="Filter results by node types (if None, includes all types)"
    )
    min_similarity_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    schema_name: Optional[str] = Field(
        default=None,
        description="Filter results by specific schema name"
    )