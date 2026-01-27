"""
Schema Repository for extracting database schema information.

This repository provides methods to fetch schema metadata from PostgreSQL
information_schema using the DatabaseClient infrastructure layer.
Reuses logic from internal_scripts/fetch_schema_info.py but adapted for
async operations and repository pattern.

All methods return domain models from schema_nodes.py (SSOT) for type safety.
"""

from typing import Any, List

from ..infrastructure.database_client import DatabaseClient
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..domain.errors import DatabaseQueryError
from ..domain.schema_nodes import (
    TableNode,
    ColumnNode,
    RelationshipNode
)


logger = get_module_logger()


class SchemaRepository:
    """
    Repository for schema metadata operations.

    This class provides methods to extract database schema information
    from PostgreSQL's information_schema, including tables, columns,
    primary keys, foreign keys, and other constraints.

    All methods return data structures compatible with schema_nodes.py
    domain models (TableNode, ColumnNode, RelationshipNode).

    Usage:
        db_client = DatabaseClient(config)
        await db_client.connect()

        schema_repo = SchemaRepository(db_client)
        tables = await schema_repo.get_tables()
        columns = await schema_repo.get_columns("actor")
        relationships = await schema_repo.get_relationships()
    """

    def __init__(self, db_client: DatabaseClient):
        """
        Initialize schema repository.

        Args:
            db_client: DatabaseClient instance for database operations
        """
        self.db_client = db_client
        logger.info("SchemaRepository initialized")

    async def _fetch_sample_values(
        self,
        table_name: str,
        column_name: str,
        schema: str = "public",
        limit: int = 5
    ) -> List[Any]:
        """
        Fetch random sample values from a column.

        Args:
            table_name: Name of the table
            column_name: Name of the column
            schema: PostgreSQL schema name
            limit: Maximum number of samples to fetch (default: 5)

        Returns:
            List of sample values (may contain various types)
        """
        trace_id = current_trace_id()

        # Build query with proper quoting for identifiers
        # Note: Using random() without DISTINCT to avoid PostgreSQL error
        # "ORDER BY expressions must appear in select list"
        query = f"""
            SELECT "{column_name}"
            FROM "{schema}"."{table_name}"
            WHERE "{column_name}" IS NOT NULL
            ORDER BY random()
            LIMIT $1
        """

        try:
            results = await self.db_client.execute_query(
                query=query,
                params=[limit],
                schema=schema,
                timeout=5  # Short timeout for sample queries
            )

            # Extract values from result dicts
            sample_values = [row[column_name] for row in results]

            return sample_values

        except Exception as e:
            # Log but don't fail - sample values are optional
            logger.warning(
                f"Failed to fetch sample values for {table_name}.{column_name}: {e}",
                trace_id=trace_id
            )
            return []

    async def get_tables(self, schema: str = "public") -> List[TableNode]:
        """
        Fetch all tables in the specified schema.

        Args:
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of dictionaries containing table information:
            [
                {
                    "table_name": "actor",
                    "table_type": "BASE TABLE"
                },
                ...
            ]

        Raises:
            DatabaseQueryError: If query execution fails
        """
        trace_id = current_trace_id()
        logger.info("Fetching tables", schema=schema, trace_id=trace_id)

        query = """
            SELECT
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = $1
            ORDER BY table_name
        """

        try:
            results = await self.db_client.execute_query(
                query=query,
                params=[schema],
                schema=schema
            )

            # Parse results into domain models (TableNode)
            tables = [
                TableNode(
                    table_name=row["table_name"],
                    table_type=row["table_type"],
                    schema_name=schema
                )
                for row in results
            ]

            logger.info(
                "Tables fetched successfully",
                count=len(tables),
                schema=schema,
                trace_id=trace_id
            )

            return tables

        except Exception as e:
            error_msg = f"Failed to fetch tables from schema '{schema}': {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

    async def get_columns(
        self,
        table_name: str,
        schema: str = "public"
    ) -> List[ColumnNode]:
        """
        Fetch all columns for a specific table.

        Args:
            table_name: Name of the table
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of ColumnNode domain models with:
            - column_name, table_name, schema_name, data_type
            - Constraint flags: is_nullable, is_primary_key, is_unique, is_foreign_key
            - sample_values (up to 5 random values)

        Raises:
            DatabaseQueryError: If query execution fails
        """
        trace_id = current_trace_id()
        logger.info(
            "Fetching columns",
            table_name=table_name,
            schema=schema,
            trace_id=trace_id
        )

        # Optimized query: Single join path with aggregation instead of 3 subqueries
        # This scans table_constraints once instead of three times
        query = """
            SELECT
                c.column_name,
                c.is_nullable,
                c.data_type,
                COALESCE(bool_or(tc.constraint_type = 'PRIMARY KEY'), false) as is_primary_key,
                COALESCE(bool_or(tc.constraint_type = 'UNIQUE'), false) as is_unique,
                COALESCE(bool_or(tc.constraint_type = 'FOREIGN KEY'), false) as is_foreign_key
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
                ON c.column_name = kcu.column_name
                AND c.table_schema = kcu.table_schema
                AND c.table_name = kcu.table_name
            LEFT JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
                AND kcu.table_schema = tc.table_schema
                AND tc.table_schema = $1
                AND tc.table_name = $2
            WHERE c.table_schema = $1 AND c.table_name = $2
            GROUP BY c.column_name, c.is_nullable, c.data_type, c.ordinal_position
            ORDER BY c.ordinal_position
        """

        try:
            results = await self.db_client.execute_query(
                query=query,
                params=[schema, table_name],
                schema=schema
            )

            # Parse results into domain models (ColumnNode) and fetch sample values
            columns = []
            for row in results:
                # Fetch sample values for this column
                sample_values = await self._fetch_sample_values(
                    table_name=table_name,
                    column_name=row["column_name"],
                    schema=schema
                )

                # Create ColumnNode domain model
                column_node = ColumnNode(
                    column_name=row["column_name"],
                    table_name=table_name,
                    schema_name=schema,
                    data_type=row["data_type"],
                    is_nullable=(row["is_nullable"] == "YES"),
                    is_primary_key=row["is_primary_key"],
                    is_unique=row["is_unique"],
                    is_foreign_key=row["is_foreign_key"],
                    sample_values=sample_values if sample_values else None
                )
                columns.append(column_node)

            logger.info(
                "Columns fetched successfully with sample values",
                table_name=table_name,
                count=len(columns),
                schema=schema,
                trace_id=trace_id
            )

            return columns

        except Exception as e:
            error_msg = f"Failed to fetch columns for table '{table_name}': {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

    async def get_primary_keys(
        self,
        table_name: str,
        schema: str = "public"
    ) -> List[str]:
        """
        Fetch primary key columns for a table.

        Args:
            table_name: Name of the table
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of primary key column names: ["actor_id"]

        Raises:
            DatabaseQueryError: If query execution fails
        """
        trace_id = current_trace_id()
        logger.info(
            "Fetching primary keys",
            table_name=table_name,
            schema=schema,
            trace_id=trace_id
        )

        # Optimized query: Filter early in JOIN conditions for better query planning
        query = """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                AND kcu.table_schema = $1
                AND kcu.table_name = $2
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2
            ORDER BY kcu.ordinal_position
        """

        try:
            results = await self.db_client.execute_query(
                query=query,
                params=[schema, table_name],
                schema=schema
            )

            # Extract column names from results
            primary_keys = [row["column_name"] for row in results]

            logger.info(
                "Primary keys fetched successfully",
                table_name=table_name,
                primary_keys=primary_keys,
                schema=schema,
                trace_id=trace_id
            )

            return primary_keys

        except Exception as e:
            error_msg = f"Failed to fetch primary keys for table '{table_name}': {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

    async def get_relationships(self, schema: str = "public") -> List[RelationshipNode]:
        """
        Fetch all foreign key relationships in the schema.

        This method returns all foreign key relationships across all tables
        in the specified schema. Data structure matches RelationshipNode requirements.

        Args:
            schema: PostgreSQL schema name (default: "public")

        Returns:
            List of RelationshipNode domain models with:
            - constraint_name
            - from_table, from_column
            - to_table, to_column
            - schema_name

        Raises:
            DatabaseQueryError: If query execution fails
        """
        trace_id = current_trace_id()
        logger.info("Fetching all relationships", schema=schema, trace_id=trace_id)

        # Optimized query: Filter early in WHERE clause for better query planning
        query = """
            SELECT
                tc.constraint_name,
                tc.table_name AS from_table,
                kcu.column_name AS from_column,
                ccu.table_name AS to_table,
                ccu.column_name AS to_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                AND kcu.table_schema = $1
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
            ORDER BY tc.table_name, tc.constraint_name
        """

        try:
            results = await self.db_client.execute_query(
                query=query,
                params=[schema],
                schema=schema
            )

            # Parse results into domain models (RelationshipNode)
            relationships = [
                RelationshipNode(
                    constraint_name=row["constraint_name"],
                    from_table=row["from_table"],
                    from_column=row["from_column"],
                    to_table=row["to_table"],
                    to_column=row["to_column"],
                    schema_name=schema
                )
                for row in results
            ]

            logger.info(
                "Relationships fetched successfully",
                count=len(relationships),
                schema=schema,
                trace_id=trace_id
            )

            return relationships

        except Exception as e:
            error_msg = f"Failed to fetch relationships from schema '{schema}': {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e
