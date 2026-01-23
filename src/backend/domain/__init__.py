"""
Domain package for the NL2SQL system.

This package contains all domain models, entities, and value objects
used throughout the application for type safety and validation.
"""

from .base_enums import (
    NodeType,
    QueryStatus,
    SQLOperationType,
    PipelineStepName,
    PipelineStepStatus
)
from .schema_nodes import BaseSchemaNode, TableNode, ColumnNode, RelationshipNode
from .requests import QueryRequest, SchemaIndexRequest, SchemaSearchRequest
from .responses import (
    HealthResponse,
    ErrorResponse,
    QueryResponse,
    SchemaSearchResponse,
    SchemaIndexResponse,
    QueryExecutionResult,
    SchemaNodeResult,
    IndexedSchemaStats
)
from .query_trace import QueryTrace, PipelineStep

__all__ = [
    # Enums
    "NodeType",
    "QueryStatus",
    "SQLOperationType",
    "PipelineStepName",
    "PipelineStepStatus",

    # Schema Nodes
    "BaseSchemaNode",
    "TableNode",
    "ColumnNode",
    "RelationshipNode",

    # Requests
    "QueryRequest",
    "SchemaIndexRequest",
    "SchemaSearchRequest",

    # Responses
    "HealthResponse",
    "ErrorResponse",
    "QueryResponse",
    "SchemaSearchResponse",
    "SchemaIndexResponse",
    "QueryExecutionResult",
    "SchemaNodeResult",
    "IndexedSchemaStats",

    # Tracing
    "QueryTrace",
    "PipelineStep"
]