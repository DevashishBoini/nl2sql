"""
Typed metadata structures for vector store documents.

Provides Pydantic BaseModel definitions for metadata attached to
vectorized schema elements (tables, columns, relationships).

## Type Design

Pydantic BaseModel provides:
- **Compile-time type checking**: Static type safety
- **Runtime validation**: Validates data at creation time
- **Type coercion**: Automatic type conversions
- **Serialization**: Easy conversion to dict/JSON via .model_dump()
- **Consistency**: Matches all other domain models in the codebase

These models work with vector store operations through .model_dump():

- **Service layer**: Creates Pydantic model instances with validation
- **Repository/Client layers**: Accept Sequence[Mapping[str, Any]]
- **Boundary conversion**: Call .model_dump() to get dict for external libraries
"""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from backend.domain.base_enums import NodeType


class TableMetadata(BaseModel):
    """
    Metadata for table documents in vector store.

    Attached to vectorized table elements for filtering and retrieval.
    """

    node_type: str = Field(..., description="Node type identifier")
    table_name: str = Field(..., description="Name of the table")
    schema_name: str = Field(..., description="Schema name")
    table_type: str = Field(..., description="Type of table (e.g., BASE TABLE, VIEW)")

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Ensure node_type is 'table'."""
        if v != NodeType.TABLE.value:
            raise ValueError(f"node_type must be '{NodeType.TABLE.value}', got '{v}'")
        return v


class ColumnMetadata(BaseModel):
    """
    Metadata for column documents in vector store.

    Attached to vectorized column elements for filtering and retrieval.
    Includes constraint flags and optional sample values.
    """

    node_type: str = Field(..., description="Node type identifier")
    table_name: str = Field(..., description="Name of the table")
    column_name: str = Field(..., description="Name of the column")
    schema_name: str = Field(..., description="Schema name")
    data_type: str = Field(..., description="PostgreSQL data type")
    is_primary_key: bool = Field(..., description="Whether column is a primary key")
    is_foreign_key: bool = Field(..., description="Whether column is a foreign key")
    is_unique: bool = Field(..., description="Whether column has unique constraint")
    is_nullable: bool = Field(..., description="Whether column can be null")
    sample_values: Optional[List[Any]] = Field(
        default=None,
        description="Sample values from the column (max 5)"
    )

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Ensure node_type is 'column'."""
        if v != NodeType.COLUMN.value:
            raise ValueError(f"node_type must be '{NodeType.COLUMN.value}', got '{v}'")
        return v


class RelationshipMetadata(BaseModel):
    """
    Metadata for relationship documents in vector store.

    Attached to vectorized foreign key relationships for filtering and retrieval.
    """

    node_type: str = Field(..., description="Node type identifier")
    constraint_name: str = Field(..., description="Name of the foreign key constraint")
    from_table: str = Field(..., description="Source table (child with FK)")
    from_column: str = Field(..., description="Column in source table")
    to_table: str = Field(..., description="Target table (parent with PK)")
    to_column: str = Field(..., description="Column in target table")
    schema_name: str = Field(..., description="Schema name")

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Ensure node_type is 'relationship'."""
        if v != NodeType.RELATIONSHIP.value:
            raise ValueError(f"node_type must be '{NodeType.RELATIONSHIP.value}', got '{v}'")
        return v


# Union type for all metadata types
SchemaNodeMetadata = Union[TableMetadata, ColumnMetadata, RelationshipMetadata]
