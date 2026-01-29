"""
Pipeline state models for the NL2SQL system.

These models represent the mutable state that flows through
the NL2SQL pipeline stages.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .responses import (
    FilteredColumn,
    FilteredRelationship,
    FilteredTable,
    QueryExecutionResult,
    ValidationStep,
    VectorSearchResult,
)
from .types import TableDescriptionsMap


@dataclass
class PipelineState:
    """
    Mutable state passed through NL2SQL pipeline steps.

    This dataclass tracks all intermediate results as the query
    flows through retrieval, filtering, generation, and execution.
    """

    # Input
    user_query: str
    schema_name: str
    row_limit: int
    timeout_seconds: int

    # Retrieved data (from vector search)
    retrieved_nodes: List[VectorSearchResult] = field(default_factory=list)

    # Filtered data (after deterministic filtering)
    filtered_columns: List[FilteredColumn] = field(default_factory=list)
    filtered_relationships: List[FilteredRelationship] = field(default_factory=list)
    filtered_tables: List[FilteredTable] = field(default_factory=list)

    # Table context (fetched by exact name)
    table_descriptions: TableDescriptionsMap = field(default_factory=dict)

    # SQL generation
    generated_sql: Optional[str] = None
    validation_steps: List[ValidationStep] = field(default_factory=list)
    retries: int = 0

    # Execution results
    execution_result: Optional[QueryExecutionResult] = None

    # Error tracking
    error_message: Optional[str] = None
    error_step: Optional[str] = None
