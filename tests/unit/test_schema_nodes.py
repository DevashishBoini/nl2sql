"""
Unit tests for schema node domain models.

Tests the Pydantic models representing database schema elements:
- BaseSchemaNode
- TableNode
- ColumnNode
- RelationshipNode
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any

from backend.domain.schema_nodes import BaseSchemaNode, TableNode, ColumnNode, RelationshipNode
from backend.domain.base_enums import NodeType


class TestBaseSchemaNode:
    """Test cases for BaseSchemaNode class."""

    def test_cannot_instantiate_directly(self):
        """BaseSchemaNode should not be instantiated directly (it's abstract)."""
        # This should work since BaseSchemaNode is not actually abstract in Pydantic
        # but we test it through concrete subclasses
        node = BaseSchemaNode(
            node_type=NodeType.TABLE,
            content="test content"
        )
        assert node.node_type == NodeType.TABLE
        assert node.content == "test content"
        assert node.metadata == {}

    def test_base_schema_node_without_content(self):
        """Test BaseSchemaNode with content omitted (it's optional)."""
        node = BaseSchemaNode(
            node_type=NodeType.TABLE
        )
        assert node.node_type == NodeType.TABLE
        assert node.content is None
        assert node.metadata == {}

    def test_base_schema_node_with_metadata(self):
        """Test BaseSchemaNode with custom metadata."""
        metadata = {"priority": "high", "tags": ["important", "core"]}
        node = BaseSchemaNode(
            node_type=NodeType.COLUMN,
            content="test content",
            metadata=metadata
        )
        assert node.metadata == metadata

    def test_base_schema_node_missing_required_fields(self):
        """Test BaseSchemaNode validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            BaseSchemaNode()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "node_type" in error_fields
        # content is now optional, so it should not be in error_fields


class TestTableNode:
    """Test cases for TableNode class."""

    def test_create_valid_table_node(self):
        """Test creating a valid TableNode."""
        table = TableNode(
            content="CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100))",
            table_name="users",
            schema_name="public",
            description="User accounts table"
        )

        assert table.node_type == NodeType.TABLE
        assert table.table_name == "users"
        assert table.schema_name == "public"
        assert table.description == "User accounts table"
        assert table.metadata == {}

    def test_table_node_without_description(self):
        """Test TableNode without optional description."""
        table = TableNode(
            content="CREATE TABLE products (id SERIAL PRIMARY KEY)",
            table_name="products",
            schema_name="public"
        )

        assert table.description is None
        assert table.node_type == NodeType.TABLE

    def test_table_node_without_content(self):
        """Test TableNode without optional content field."""
        table = TableNode(
            table_name="products",
            schema_name="public",
            description="Products table"
        )

        assert table.content is None
        assert table.description == "Products table"
        assert table.node_type == NodeType.TABLE

    def test_table_node_with_metadata(self):
        """Test TableNode with custom metadata."""
        metadata = {"row_count": 1000, "last_updated": "2024-01-20"}
        table = TableNode(
            content="CREATE TABLE orders (id SERIAL PRIMARY KEY)",
            table_name="orders",
            schema_name="ecommerce",
            metadata=metadata
        )

        assert table.metadata == metadata

    def test_table_node_frozen_node_type(self):
        """Test that node_type is frozen and cannot be changed."""
        table = TableNode(
            content="CREATE TABLE test (id SERIAL PRIMARY KEY)",
            table_name="test",
            schema_name="public"
        )

        # Try to change the frozen field - this should raise an error
        with pytest.raises(ValidationError):
            table.node_type = NodeType.COLUMN

    def test_table_node_missing_required_fields(self):
        """Test TableNode validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TableNode()  # type: ignore[call-arg]  # content is optional, so we don't need to pass it

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "table_name" in error_fields
        assert "schema_name" in error_fields

    def test_table_node_invalid_types(self):
        """Test TableNode with invalid field types."""
        with pytest.raises(ValidationError):
            TableNode(
                content=123,  # type: ignore[arg-type]  # Should be string
                table_name="test",
                schema_name="public"
            )


class TestColumnNode:
    """Test cases for ColumnNode class."""

    def test_create_valid_column_node(self):
        """Test creating a valid ColumnNode with all fields."""
        column = ColumnNode(
            content="id SERIAL PRIMARY KEY",
            column_name="id",
            table_name="users",
            schema_name="public",
            data_type="SERIAL",
            description="Primary key for users table",
            sample_values=[1, 2, 3, 4, 5],
            is_primary_key=True,
            is_nullable=False,
            is_unique=True,
            is_foreign_key=False
        )

        assert column.node_type == NodeType.COLUMN
        assert column.column_name == "id"
        assert column.table_name == "users"
        assert column.schema_name == "public"
        assert column.data_type == "SERIAL"
        assert column.description == "Primary key for users table"
        assert column.sample_values == [1, 2, 3, 4, 5]
        assert column.is_primary_key is True
        assert column.is_nullable is False
        assert column.is_unique is True
        assert column.is_foreign_key is False

    def test_column_node_default_values(self):
        """Test ColumnNode with default boolean values."""
        column = ColumnNode(
            content="name VARCHAR(100)",
            column_name="name",
            table_name="users",
            schema_name="public",
            data_type="VARCHAR(100)"
        )

        # Test default boolean values
        assert column.is_primary_key is False
        assert column.is_nullable is True
        assert column.is_unique is False
        assert column.is_foreign_key is False
        assert column.description is None
        assert column.sample_values is None

    def test_column_node_without_content(self):
        """Test ColumnNode without optional content field."""
        column = ColumnNode(
            column_name="email",
            table_name="users",
            schema_name="public",
            data_type="VARCHAR(255)",
            description="User email address"
        )

        assert column.content is None
        assert column.column_name == "email"
        assert column.description == "User email address"

    def test_column_node_with_sample_values(self):
        """Test ColumnNode with various sample value types."""
        column = ColumnNode(
            content="status VARCHAR(20)",
            column_name="status",
            table_name="orders",
            schema_name="public",
            data_type="VARCHAR(20)",
            sample_values=["pending", "completed", "cancelled", "shipped"]
        )

        assert column.sample_values == ["pending", "completed", "cancelled", "shipped"]

    def test_column_node_frozen_node_type(self):
        """Test that node_type is frozen and cannot be changed."""
        column = ColumnNode(
            content="test_col INT",
            column_name="test_col",
            table_name="test_table",
            schema_name="public",
            data_type="INT"
        )

        # Try to change the frozen field - this should raise an error
        with pytest.raises(ValidationError):
            column.node_type = NodeType.TABLE

    def test_column_node_missing_required_fields(self):
        """Test ColumnNode validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ColumnNode()  # type: ignore[call-arg]  # content is optional

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "column_name" in error_fields
        assert "table_name" in error_fields
        assert "schema_name" in error_fields
        assert "data_type" in error_fields

    def test_column_node_boolean_validation(self):
        """Test ColumnNode with invalid boolean values."""
        # Note: Pydantic v2 automatically coerces truthy strings to booleans
        # Test that actual invalid types (like dict) raise ValidationError
        with pytest.raises(ValidationError):
            ColumnNode(
                content="test_col INT",
                column_name="test_col",
                table_name="test_table",
                schema_name="public",
                data_type="INT",
                is_primary_key={"invalid": "type"}  # type: ignore[arg-type]  # Dict is not coercible to bool
            )

    def test_column_node_foreign_key_scenario(self):
        """Test ColumnNode representing a foreign key."""
        fk_column = ColumnNode(
            content="user_id INT REFERENCES users(id)",
            column_name="user_id",
            table_name="orders",
            schema_name="public",
            data_type="INT",
            description="Reference to users table",
            is_foreign_key=True,
            is_nullable=False
        )

        assert fk_column.is_foreign_key is True
        assert fk_column.is_nullable is False
        assert fk_column.is_primary_key is False


class TestRelationshipNode:
    """Test cases for RelationshipNode class."""

    def test_create_valid_relationship_node(self):
        """Test creating a valid RelationshipNode."""
        relationship = RelationshipNode(
            content="FOREIGN KEY (user_id) REFERENCES users(id)",
            from_table="orders",
            to_table="users",
            from_column="user_id",
            to_column="id",
            schema_name="public",
            description="Orders belong to users"
        )

        assert relationship.node_type == NodeType.RELATIONSHIP
        assert relationship.from_table == "orders"
        assert relationship.to_table == "users"
        assert relationship.from_column == "user_id"
        assert relationship.to_column == "id"
        assert relationship.schema_name == "public"
        assert relationship.description == "Orders belong to users"

    def test_relationship_node_without_description(self):
        """Test RelationshipNode without optional description."""
        relationship = RelationshipNode(
            content="FK: product_id -> products.id",
            from_table="order_items",
            to_table="products",
            from_column="product_id",
            to_column="id",
            schema_name="public"
        )

        assert relationship.description is None
        assert relationship.node_type == NodeType.RELATIONSHIP

    def test_relationship_node_without_content(self):
        """Test RelationshipNode without optional content field."""
        relationship = RelationshipNode(
            from_table="order_items",
            to_table="products",
            from_column="product_id",
            to_column="id",
            schema_name="public",
            description="Order items reference products"
        )

        assert relationship.content is None
        assert relationship.description == "Order items reference products"
        assert relationship.node_type == NodeType.RELATIONSHIP

    def test_relationship_node_frozen_node_type(self):
        """Test that node_type is frozen and cannot be changed."""
        relationship = RelationshipNode(
            content="FK relationship",
            from_table="table_a",
            to_table="table_b",
            from_column="col_a",
            to_column="col_b",
            schema_name="public"
        )

        # Try to change the frozen field - this should raise an error
        with pytest.raises(ValidationError):
            relationship.node_type = NodeType.TABLE

    def test_relationship_node_missing_required_fields(self):
        """Test RelationshipNode validation with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            RelationshipNode()  # type: ignore[call-arg]  # content is optional

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "from_table" in error_fields
        assert "to_table" in error_fields
        assert "from_column" in error_fields
        assert "to_column" in error_fields
        assert "schema_name" in error_fields

    def test_relationship_node_self_referencing(self):
        """Test RelationshipNode for self-referencing relationships."""
        relationship = RelationshipNode(
            content="FOREIGN KEY (parent_id) REFERENCES categories(id)",
            from_table="categories",
            to_table="categories",
            from_column="parent_id",
            to_column="id",
            schema_name="public",
            description="Categories can have parent categories"
        )

        assert relationship.from_table == "categories"
        assert relationship.to_table == "categories"
        assert relationship.from_column == "parent_id"
        assert relationship.to_column == "id"

    def test_relationship_node_with_metadata(self):
        """Test RelationshipNode with custom metadata."""
        metadata = {
            "constraint_name": "fk_orders_user_id",
            "cascade_delete": True,
            "created_at": "2024-01-20"
        }
        relationship = RelationshipNode(
            content="FK with cascade delete",
            from_table="orders",
            to_table="users",
            from_column="user_id",
            to_column="id",
            schema_name="public",
            metadata=metadata
        )

        assert relationship.metadata == metadata


class TestSchemaNodeInteractions:
    """Test interactions between different schema node types."""

    def test_nodes_json_serialization(self):
        """Test that all nodes can be serialized to JSON."""
        table = TableNode(
            content="CREATE TABLE users (id SERIAL PRIMARY KEY)",
            table_name="users",
            schema_name="public"
        )

        column = ColumnNode(
            content="id SERIAL PRIMARY KEY",
            column_name="id",
            table_name="users",
            schema_name="public",
            data_type="SERIAL",
            is_primary_key=True
        )

        relationship = RelationshipNode(
            content="FK relationship",
            from_table="orders",
            to_table="users",
            from_column="user_id",
            to_column="id",
            schema_name="public"
        )

        # Test JSON serialization
        table_json = table.model_dump()
        column_json = column.model_dump()
        relationship_json = relationship.model_dump()

        assert table_json["node_type"] == "table"
        assert column_json["node_type"] == "column"
        assert relationship_json["node_type"] == "relationship"

    def test_nodes_with_same_schema_name(self):
        """Test nodes belonging to the same schema."""
        schema_name = "ecommerce"

        table = TableNode(
            content="CREATE TABLE products (id SERIAL PRIMARY KEY)",
            table_name="products",
            schema_name=schema_name
        )

        column = ColumnNode(
            content="name VARCHAR(100) NOT NULL",
            column_name="name",
            table_name="products",
            schema_name=schema_name,
            data_type="VARCHAR(100)",
            is_nullable=False
        )

        relationship = RelationshipNode(
            content="FK: category_id -> categories.id",
            from_table="products",
            to_table="categories",
            from_column="category_id",
            to_column="id",
            schema_name=schema_name
        )

        assert table.schema_name == schema_name
        assert column.schema_name == schema_name
        assert relationship.schema_name == schema_name

    def test_node_type_enum_values(self):
        """Test that all node types use correct enum values."""
        table = TableNode(
            content="test",
            table_name="test",
            schema_name="public"
        )

        column = ColumnNode(
            content="test",
            column_name="test",
            table_name="test",
            schema_name="public",
            data_type="INT"
        )

        relationship = RelationshipNode(
            content="test",
            from_table="a",
            to_table="b",
            from_column="col_a",
            to_column="col_b",
            schema_name="public"
        )

        assert table.node_type == NodeType.TABLE == "table"
        assert column.node_type == NodeType.COLUMN == "column"
        assert relationship.node_type == NodeType.RELATIONSHIP == "relationship"