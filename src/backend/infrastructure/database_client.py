"""
Database client for PostgreSQL/Supabase using asyncpg.

This module provides an async database client for connecting to Supabase
PostgreSQL with schema-specific operations, connection pooling, and
comprehensive error handling.
"""

from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import asyncpg

from ..config import DatabaseConfig
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..domain.errors import DatabaseConnectionError, DatabaseQueryError


logger = get_module_logger()


class DatabaseClient:
    """
    Low-level async PostgreSQL/Supabase database client using asyncpg.

    This is a thin infrastructure layer for database operations.
    Higher-level operations (schema introspection, health checks, etc.)
    should be implemented in Repository or Service layers.

    Features:
    - Connection pooling with asyncpg
    - Query execution
    - Schema-specific operations
    - Structured logging with trace IDs
    - Comprehensive error handling

    Usage:
        client = DatabaseClient(config)
        await client.connect()

        # Execute query
        result = await client.execute_query("SELECT * FROM users LIMIT 10")

        # Execute with parameters
        result = await client.execute_query(
            "SELECT * FROM users WHERE id = $1",
            params=[user_id]
        )

        # Get single value
        count = await client.execute_scalar("SELECT COUNT(*) FROM users")

        await client.close()
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database client with configuration.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._is_connected = False

        logger.info(
            "DatabaseClient initialized",
            default_schema=config.default_schema,
            connection_pool_size=config.connection_pool_max_size,
            query_timeout_seconds=config.query_timeout_seconds,
            application_name=config.application_name
        )

    async def connect(self) -> None:
        """
        Establish connection pool to the database.

        Raises:
            DatabaseConnectionError: If connection fails
        """
        if self._is_connected:
            logger.warning("Database client already connected")
            return

        trace_id = current_trace_id()
        logger.info("Establishing database connection", trace_id=trace_id)

        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                dsn=self.config.database_url,
                min_size=self.config.connection_pool_min_size,
                max_size=self.config.connection_pool_max_size,
                command_timeout=self.config.query_timeout_seconds,
                timeout=self.config.connection_timeout_seconds,
                max_queries=self.config.connection_pool_max_queries,
                max_cached_statement_lifetime=self.config.max_cached_statement_lifetime,
                max_cacheable_statement_size=self.config.max_cacheable_statement_size,
                server_settings={
                    'application_name': self.config.application_name,
                    'search_path': self.config.default_schema,  # Set default schema
                    'jit': 'on' if self.config.jit_enabled else 'off'
                }
            )

            # Test connection
            await self._test_connection(self._pool, "main")

            self._is_connected = True
            logger.info(
                "Database connection established successfully",
                pool_size=self.config.connection_pool_max_size,
                default_schema=self.config.default_schema,
                trace_id=trace_id
            )

        except asyncpg.InvalidCatalogNameError as e:
            error_msg = f"Database does not exist: {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseConnectionError(error_msg) from e

        except asyncpg.InvalidPasswordError as e:
            error_msg = f"Authentication failed: {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseConnectionError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to connect to database: {e}"
            logger.error(error_msg, error_type=type(e).__name__, trace_id=trace_id)
            raise DatabaseConnectionError(error_msg) from e

    async def _test_connection(self, pool: asyncpg.Pool, pool_name: str) -> None:
        """Test database connection."""
        trace_id = current_trace_id()
        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise DatabaseConnectionError(f"Connection test failed for {pool_name} pool")

                # Verify schema access
                current_schema = await conn.fetchval("SELECT current_schema()")
                logger.info(
                    f"Connection test successful for {pool_name} pool",
                    current_schema=current_schema,
                    trace_id=trace_id
                )
        except Exception as e:
            error_msg = f"Connection test failed for {pool_name} pool: {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise DatabaseConnectionError(error_msg) from e

    async def close(self) -> None:
        """Close database connection pool."""
        trace_id = current_trace_id()
        logger.info("Closing database connection", trace_id=trace_id)

        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed", trace_id=trace_id)

        self._is_connected = False
        self._pool = None

        logger.info("Database connection closed", trace_id=trace_id)

    def is_connected(self) -> bool:
        """Check if database client is connected."""
        return self._is_connected and self._pool is not None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database connection.

        Returns:
            Dictionary with status and connection details

        Example:
            {
                "status": "healthy",
                "connected": True,
                "pool_size": 10,
                "current_schema": "public"
            }
        """
        trace_id = current_trace_id()

        if not self.is_connected():
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Database client not connected"
            }

        try:
            async with self.acquire_connection() as conn:
                # Test connection with simple query
                result = await conn.fetchval("SELECT 1")
                current_schema = await conn.fetchval("SELECT current_schema()")

                if result != 1:
                    return {
                        "status": "unhealthy",
                        "connected": True,
                        "error": "Connection test query failed"
                    }

                logger.info("Database health check passed", trace_id=trace_id)

                return {
                    "status": "healthy",
                    "connected": True,
                    "pool_size": self.config.connection_pool_max_size,
                    "current_schema": current_schema
                }

        except Exception as e:
            logger.error(
                "Database health check failed",
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id
            )
            return {
                "status": "unhealthy",
                "connected": True,
                "error": str(e)
            }

    @asynccontextmanager
    async def acquire_connection(self, schema: Optional[str] = None):
        """
        Context manager to acquire a database connection from the pool.

        Args:
            schema: Optional schema to set for this connection

        Yields:
            asyncpg.Connection: Database connection

        Example:
            async with client.acquire_connection(schema="inventory") as conn:
                result = await conn.fetch("SELECT * FROM products")
        """
        if not self.is_connected():
            raise DatabaseConnectionError("Database client is not connected")

        if self._pool is None:
            raise DatabaseConnectionError("Connection pool is not available")

        async with self._pool.acquire() as connection:
            # Set schema if specified
            if schema and schema != self.config.default_schema:
                await connection.execute(f"SET search_path TO {schema}")

            yield connection

            # Reset to default schema after use
            if schema and schema != self.config.default_schema:
                await connection.execute(f"SET search_path TO {self.config.default_schema}")

    async def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        timeout: Optional[int] = None,
        schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dictionaries.

        Args:
            query: SQL query string
            params: Optional query parameters
            timeout: Optional query timeout in seconds
            schema: Optional schema to execute query in

        Returns:
            List of dictionaries containing query results

        Example:
            # Use default schema
            results = await client.execute_query("SELECT * FROM users")

            # Use specific schema
            results = await client.execute_query(
                "SELECT * FROM products",
                schema="inventory"
            )
        """
        if not self.is_connected():
            raise DatabaseConnectionError("Database client is not connected")

        trace_id = current_trace_id()

        if self._pool is None:
            raise DatabaseConnectionError("Connection pool is not available")

        effective_schema = schema or self.config.default_schema

        logger.info(
            "Executing database query",
            query=query[:200],
            schema=effective_schema,
            trace_id=trace_id
        )

        try:
            async with self.acquire_connection(schema=schema) as conn:
                # Set statement timeout if specified
                if timeout:
                    await conn.execute(f"SET statement_timeout = {timeout * 1000}")

                # Execute query
                if params is not None:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)

                # Convert to list of dictionaries
                results = [dict(row) for row in rows]

                logger.info(
                    "Query executed successfully",
                    row_count=len(results),
                    schema=effective_schema,
                    trace_id=trace_id
                )

                return results

        except asyncpg.QueryCanceledError as e:
            error_msg = f"Query timeout exceeded: {e}"
            logger.error(error_msg, query=query[:200], trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

        except asyncpg.PostgresSyntaxError as e:
            error_msg = f"SQL syntax error: {e}"
            logger.error(error_msg, query=query[:200], trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

        except asyncpg.UndefinedTableError as e:
            error_msg = f"Table does not exist: {e}"
            logger.error(error_msg, query=query[:200], schema=effective_schema, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

        except asyncpg.UndefinedColumnError as e:
            error_msg = f"Column does not exist: {e}"
            logger.error(error_msg, query=query[:200], trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

        except asyncpg.InvalidSchemaNameError as e:
            error_msg = f"Invalid schema name: {e}"
            logger.error(error_msg, schema=effective_schema, trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

        except Exception as e:
            error_msg = f"Query execution failed: {e}"
            logger.error(
                error_msg,
                error_type=type(e).__name__,
                query=query[:200],
                trace_id=trace_id
            )
            raise DatabaseQueryError(error_msg) from e

    async def execute_scalar(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        schema: Optional[str] = None
    ) -> Any:
        """Execute a query and return a single scalar value."""
        if not self.is_connected():
            raise DatabaseConnectionError("Database client is not connected")

        trace_id = current_trace_id()

        try:
            async with self.acquire_connection(schema=schema) as conn:
                if params is not None:
                    result = await conn.fetchval(query, *params)
                else:
                    result = await conn.fetchval(query)

                logger.info("Scalar query executed successfully", trace_id=trace_id)
                return result

        except Exception as e:
            error_msg = f"Scalar query execution failed: {e}"
            logger.error(error_msg, query=query[:200], trace_id=trace_id)
            raise DatabaseQueryError(error_msg) from e

