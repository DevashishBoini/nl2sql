from pydantic import BaseModel, Field
from .base_enums import NodeType
from typing import Any, Dict, List, Optional
from abc import ABC


class BaseSchemaNode(BaseModel, ABC):
    """Base class for schema nodes."""

    node_type: NodeType = Field(..., description="Type of the schema node")
    content: Optional[str] = Field(default=None, description="Raw content or definition of the schema node")
    metadata : Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the schema node used for filtering and retrieval purposes")

class TableNode(BaseSchemaNode):
    """Represents a database table schema node."""

    node_type : NodeType = Field(default=NodeType.TABLE, frozen=True, description="should be 'table'")
    table_name : str = Field(..., description="Name of the table")
    schema_name : str = Field(..., description="Schema to which the table belongs")
    description : Optional[str] = Field(default=None, description="Description of the table")

class ColumnNode(BaseSchemaNode):
    """Represents a database column schema node."""

    node_type : NodeType = Field(default=NodeType.COLUMN, frozen=True, description="Should be 'column'")
    column_name : str = Field(..., description="Name of the column")
    table_name : str = Field(..., description="Name of the table to which the column belongs")
    schema_name : str = Field(..., description="Schema to which the table belongs")
    data_type : str = Field(..., description="Data type of the column")
    description : Optional[str] = Field(default=None, description="Description of the column")
    sample_values : Optional[list[Any]] = Field(default=None, max_length=5, description="Sample values from the column")

    is_primary_key : bool = Field(default=False, description="Indicates if the column is a primary key")
    is_nullable : bool = Field(default=True, description="Indicates if the column can contain null values")
    is_unique : bool = Field(default=False, description="Indicates if the column has unique values")
    is_foreign_key : bool = Field(default=False, description="Indicates if the column is a foreign key")

class RelationshipNode(BaseSchemaNode):
    """Represents a relationship between two tables."""

    node_type : NodeType = Field(default=NodeType.RELATIONSHIP, frozen=True, description="Should be 'relationship'")
    from_table : str = Field(..., description="Name of the source table in the relationship")
    to_table : str = Field(..., description="Name of the target table in the relationship")
    from_column : str = Field(..., description="Column in the source table")
    to_column : str = Field(..., description="Column in the target table")
    schema_name : str = Field(..., description="Schema to which the tables belong")
    description : Optional[str] = Field(default=None, description="Description of the relationship")


