"""
Schema Service for orchestrating schema operations.

This service provides business logic for schema-related operations,
coordinating between the repository layer and API layer.
"""

from typing import Any, Dict, List, Optional

from ..repositories.schema_repository import SchemaRepository
from ..domain.schema_nodes import TableNode, ColumnNode, RelationshipNode
from ..infrastructure.storage_client import StorageClient
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..utils.yaml_loader import YAMLDescriptionLoader, parse_descriptions


logger = get_module_logger()


class SchemaService:
    """
    Service for schema-related business logic.

    This service orchestrates schema operations by coordinating
    between repositories and applying business logic.

    Usage:
        schema_service = SchemaService(schema_repo)
        tables = await schema_service.get_all_tables()
    """

    def __init__(
        self,
        schema_repository: SchemaRepository,
        storage_client: Optional[StorageClient] = None,
        yaml_path: Optional[str] = None,
        yaml_bucket: Optional[str] = None
    ):
        """
        Initialize schema service.

        Args:
            schema_repository: SchemaRepository instance for data access
            storage_client: Optional StorageClient for loading YAML from bucket
            yaml_path: Optional path to YAML descriptions file
            yaml_bucket: Optional bucket name for YAML file in storage
        """
        self.schema_repo = schema_repository
        self.yaml_loader = YAMLDescriptionLoader(storage_client)
        self.yaml_path = yaml_path
        self.yaml_bucket = yaml_bucket

        # Cache for descriptions
        self._table_descriptions: Optional[Dict[str, str]] = None
        self._column_descriptions: Optional[Dict[str, str]] = None

        logger.info("SchemaService initialized")

    async def _load_descriptions(self) -> None:
        """
        Load table and column descriptions from YAML file in storage bucket (lazy loading).

        This method loads the YAML file once and caches the results.
        """
        if self._table_descriptions is not None:
            return  # Already loaded

        trace_id = current_trace_id()

        if not self.yaml_path or not self.yaml_bucket:
            logger.warning(
                "YAML path or bucket not configured, descriptions will not be loaded",
                yaml_path=self.yaml_path,
                yaml_bucket=self.yaml_bucket,
                trace_id=trace_id
            )
            self._table_descriptions = {}
            self._column_descriptions = {}
            return

        try:
            # Load YAML content from storage bucket
            yaml_content = await self.yaml_loader.load(
                bucket=self.yaml_bucket,
                file_path=self.yaml_path
            )

            # Parse descriptions
            table_descs, column_descs = parse_descriptions(yaml_content)

            self._table_descriptions = table_descs
            self._column_descriptions = column_descs

            logger.info(
                "Descriptions loaded successfully from storage",
                bucket=self.yaml_bucket,
                file_path=self.yaml_path,
                table_count=len(table_descs),
                column_count=len(column_descs),
                trace_id=trace_id
            )

        except Exception as e:
            logger.error(
                f"Failed to load descriptions from YAML storage: {e}",
                bucket=self.yaml_bucket,
                yaml_path=self.yaml_path,
                trace_id=trace_id
            )
            # Set empty dicts to avoid repeated load attempts
            self._table_descriptions = {}
            self._column_descriptions = {}

    def _merge_table_descriptions(self, tables: List[TableNode]) -> List[TableNode]:
        """
        Merge YAML descriptions into TableNode objects.

        Args:
            tables: List of TableNode objects from repository

        Returns:
            List of TableNode objects with descriptions merged
        """
        if not self._table_descriptions:
            return tables

        updated_tables = []
        for table in tables:
            description = self._table_descriptions.get(table.table_name)
            if description:
                # Create new TableNode with description
                updated_table = table.model_copy(update={"description": description})
                updated_tables.append(updated_table)
            else:
                updated_tables.append(table)

        return updated_tables

    def _merge_column_descriptions(self, columns: List[ColumnNode]) -> List[ColumnNode]:
        """
        Merge YAML descriptions into ColumnNode objects.

        Args:
            columns: List of ColumnNode objects from repository

        Returns:
            List of ColumnNode objects with descriptions merged
        """
        if not self._column_descriptions:
            return columns

        updated_columns = []
        for column in columns:
            # Column key format in YAML: "table_name.column_name"
            column_key = f"{column.table_name}.{column.column_name}"
            description = self._column_descriptions.get(column_key)

            if description:
                # Create new ColumnNode with description
                updated_column = column.model_copy(update={"description": description})
                updated_columns.append(updated_column)
            else:
                updated_columns.append(column)

        return updated_columns

    async def get_all_tables(self, schema: str = "public") -> List[TableNode]:
        """
        Get all tables in the schema with descriptions from YAML.

        Args:
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of TableNode domain models with descriptions
        """
        trace_id = current_trace_id()
        logger.info("Fetching all tables", schema=schema, trace_id=trace_id)

        # Load descriptions if not already loaded
        await self._load_descriptions()

        # Fetch tables from repository
        tables = await self.schema_repo.get_tables(schema=schema)

        # Merge descriptions
        tables = self._merge_table_descriptions(tables)

        logger.info(
            "Tables fetched via service with descriptions",
            count=len(tables),
            trace_id=trace_id
        )

        return tables

    async def get_table_columns(
        self,
        table_name: str,
        schema: str = "public"
    ) -> List[ColumnNode]:
        """
        Get all columns for a specific table with descriptions from YAML.

        Args:
            table_name: Name of the table
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of ColumnNode domain models with sample values and descriptions
        """
        trace_id = current_trace_id()
        logger.info(
            "Fetching columns for table",
            table_name=table_name,
            schema=schema,
            trace_id=trace_id
        )

        # Load descriptions if not already loaded
        await self._load_descriptions()

        # Fetch columns from repository
        columns = await self.schema_repo.get_columns(
            table_name=table_name,
            schema=schema
        )

        # Merge descriptions
        columns = self._merge_column_descriptions(columns)

        logger.info(
            "Columns fetched via service with descriptions",
            table_name=table_name,
            count=len(columns),
            trace_id=trace_id
        )

        return columns

    async def get_all_relationships(
        self,
        schema: str = "public"
    ) -> List[RelationshipNode]:
        """
        Get all relationships (foreign keys) in the schema.

        Args:
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of RelationshipNode domain models
        """
        trace_id = current_trace_id()
        logger.info("Fetching relationships", schema=schema, trace_id=trace_id)

        relationships = await self.schema_repo.get_relationships(schema=schema)

        logger.info(
            "Relationships fetched via service",
            count=len(relationships),
            trace_id=trace_id
        )

        return relationships

    async def get_schema_summary(
        self,
        schema: str = "public"
    ) -> Dict[str, Any]:
        """
        Get a summary of the schema including tables, columns, and relationships.

        This is a convenience method for getting an overview of the schema.

        Args:
            schema: PostgreSQL schema name (default: "public")

        Returns:
            Dictionary with schema summary information
        """
        trace_id = current_trace_id()
        logger.info("Fetching schema summary", schema=schema, trace_id=trace_id)

        # Fetch all data
        tables = await self.get_all_tables(schema=schema)
        relationships = await self.get_all_relationships(schema=schema)

        # Get sample columns from first table (if exists)
        sample_columns = []
        sample_table_name = None
        if tables:
            sample_table_name = tables[0].table_name
            sample_columns = await self.get_table_columns(
                table_name=sample_table_name,
                schema=schema
            )

        summary = {
            "schema_name": schema,
            "table_count": len(tables),
            "relationship_count": len(relationships),
            "tables": tables,
            "relationships": relationships,
            "sample_table": sample_table_name,
            "sample_columns": sample_columns
        }

        logger.info(
            "Schema summary generated",
            table_count=len(tables),
            relationship_count=len(relationships),
            trace_id=trace_id
        )

        return summary
