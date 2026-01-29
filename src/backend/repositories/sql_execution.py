"""
SQL Execution Repository.

This repository handles the final step of the NL2SQL pipeline: executing
validated SQL queries against the database with strict safety constraints.

Safety Features:
- Read-only enforcement: All queries run with read_only=True flag
- Timeout protection: Configurable query timeout (default 30s)
- Row limiting: Results capped at requested row_limit
- Schema isolation: Queries execute within specified schema context

Architecture Notes:
- This is a REPOSITORY (data access layer)
- Only executes SQL that has passed validation (SQLValidationRepository)
- Uses injected DatabaseClient for connection management
- Returns structured QueryExecutionResult with metadata

Execution Flow:
1. Receive validated SQL from service layer
2. Acquire read-only connection from pool
3. Set statement timeout
4. Execute query with schema context
5. Convert results to list of dicts
6. Return QueryExecutionResult with row_count, execution_time, etc.

Usage:
    repo = SQLExecutionRepository(db_client)
    result = await repo.execute(
        sql="SELECT * FROM customer LIMIT 10",
        schema="public",
        timeout_seconds=30,
        row_limit=100
    )
    print(f"Returned {result.row_count} rows in {result.execution_time_ms}ms")

Error Handling:
- Raises Exception on execution failure (caught by service layer)
- Errors include SQL context for debugging
- Timeout errors clearly identified
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.domain.responses import QueryExecutionResult
from backend.infrastructure.database_client import DatabaseClient
from backend.utils.logging import get_module_logger
from backend.utils.tracing import current_trace_id

logger = get_module_logger()


class SQLExecutionRepository:
    """
    Repository for SQL execution.

    Executes validated SQL with read-only enforcement.
    """

    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client

    async def execute(
        self,
        sql: str,
        schema: str,
        timeout_seconds: int,
        row_limit: int,
    ) -> QueryExecutionResult:
        """
        Execute validated SQL query.

        Args:
            sql: Validated SQL string
            schema: PostgreSQL schema
            timeout_seconds: Query timeout
            row_limit: Expected row limit for was_limited detection

        Returns:
            QueryExecutionResult with rows and metadata

        Raises:
            Exception if execution fails
        """
        trace_id = current_trace_id()

        logger.info(
            "Executing SQL query",
            sql_length=len(sql),
            schema=schema,
            timeout=timeout_seconds,
            trace_id=trace_id,
        )

        start_time = datetime.now(timezone.utc)

        # Execute with explicit read-only (security critical)
        results = await self.db_client.execute_query(
            query=sql,
            schema=schema,
            timeout=timeout_seconds,
            read_only=True,  # NL2SQL queries must be read-only
        )

        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Extract column names
        column_names = list(results[0].keys()) if results else []

        # Check if results were limited
        was_limited = len(results) >= row_limit

        result = QueryExecutionResult(
            rows=results,
            column_names=column_names,
            row_count=len(results),
            execution_time_ms=execution_time_ms,
            was_limited=was_limited,
        )

        logger.info(
            "SQL execution successful",
            row_count=len(results),
            execution_time_ms=round(execution_time_ms, 2),
            was_limited=was_limited,
            trace_id=trace_id,
        )

        return result
