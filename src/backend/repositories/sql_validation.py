"""
SQL Validation Repository.

This repository provides comprehensive SQL validation for the NL2SQL pipeline,
ensuring generated SQL is safe and correct before execution.

Validation Checks (in order):
1. SELECT-Only Check: Ensures query starts with SELECT
2. Dangerous Keywords Check: Blocks INSERT, UPDATE, DELETE, DROP, ALTER, etc.
3. Allowed Tables Check: Validates only context tables are used
4. Syntax Check: Uses EXPLAIN to validate SQL syntax against real database

Security Philosophy:
- Defense in depth: Multiple validation layers
- Fail-fast: First failing check returns immediately
- Explicit allowlist: Only tables from schema retrieval can be used
- No trusted input: All LLM-generated SQL is treated as untrusted

Architecture Notes:
- This is a REPOSITORY (data access layer)
- EXPLAIN validation requires database connection (injected DatabaseClient)
- Static checks are pure functions (no I/O)
- Returns ValidationStep with pass/fail status and diagnostic message

Usage in Pipeline:
    repo = SQLValidationRepository(db_client)
    result = await repo.validate(
        sql="SELECT * FROM customer LIMIT 10",
        tables=filtered_tables,
        columns=filtered_columns
    )
    if result.passed:
        # Safe to execute
    else:
        # Use result.message for retry feedback

Error Handling:
- ValidationStep.passed=False does NOT raise exceptions
- Exceptions only for infrastructure failures (DB connection, etc.)
- Error messages designed for LLM feedback (human-readable)
"""

import re
from typing import List, Set, Tuple

from backend.domain.responses import FilteredColumn, FilteredTable, ValidationStep
from backend.infrastructure.database_client import DatabaseClient
from backend.utils.logging import get_module_logger
from backend.utils.tracing import get_trace_id

logger = get_module_logger()

# Keywords that indicate potentially dangerous SQL
DANGEROUS_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "EXEC",
}

# SQL keywords that precede table names
TABLE_KEYWORDS = {"FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"}


class SQLValidationRepository:
    """
    Repository for SQL validation.

    Performs static and dynamic validation of generated SQL.
    """

    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client

    async def validate(
        self,
        sql: str,
        tables: List[FilteredTable],
        columns: List[FilteredColumn],
    ) -> ValidationStep:
        """
        Validate SQL with static and dynamic checks.

        Args:
            sql: SQL string to validate
            tables: Allowed tables
            columns: Allowed columns

        Returns:
            ValidationStep with pass/fail status and message
        """
        trace_id = get_trace_id()

        # Check 1: SELECT-only
        select_check = self._check_select_only(sql)
        if not select_check.passed:
            return select_check

        # Check 2: No dangerous keywords
        keyword_check = self._check_dangerous_keywords(sql)
        if not keyword_check.passed:
            return keyword_check

        # Check 3: Only allowed tables
        table_check = self._check_allowed_tables(sql, tables)
        if not table_check.passed:
            return table_check

        # Check 4: Syntax validation via EXPLAIN
        syntax_check = await self._check_syntax(sql, trace_id)
        return syntax_check

    def _check_select_only(self, sql: str) -> ValidationStep:
        """Verify SQL is a SELECT statement."""
        sql_upper = sql.upper().strip()

        if not sql_upper.startswith("SELECT"):
            return ValidationStep(
                step_name="select_only_check",
                passed=False,
                message="Query must be a SELECT statement",
                sql_attempted=sql,
            )

        return ValidationStep(
            step_name="select_only_check",
            passed=True,
            message="SELECT-only check passed",
            sql_attempted=sql,
        )

    def _check_dangerous_keywords(self, sql: str) -> ValidationStep:
        """Check for dangerous SQL keywords."""
        sql_upper = sql.upper()

        found_keywords = []
        for keyword in DANGEROUS_KEYWORDS:
            # Check for keyword as whole word (not part of column name)
            if f" {keyword} " in f" {sql_upper} ":
                found_keywords.append(keyword)

        if found_keywords:
            return ValidationStep(
                step_name="dangerous_keyword_check",
                passed=False,
                message=f"SQL contains dangerous keywords: {', '.join(found_keywords)}",
                sql_attempted=sql,
            )

        return ValidationStep(
            step_name="dangerous_keyword_check",
            passed=True,
            message="No dangerous keywords found",
            sql_attempted=sql,
        )

    def _check_allowed_tables(
        self, sql: str, allowed_tables: List[FilteredTable]
    ) -> ValidationStep:
        """
        Check that SQL only uses allowed tables.

        Extracts table names from FROM and JOIN clauses and validates
        against the allowed list.
        """
        allowed_table_names = {t.table_name.lower() for t in allowed_tables}
        used_tables = self._extract_table_names(sql)

        disallowed = used_tables - allowed_table_names

        if disallowed:
            return ValidationStep(
                step_name="allowed_tables_check",
                passed=False,
                message=f"SQL uses tables not in allowed schema: {', '.join(sorted(disallowed))}",
                sql_attempted=sql,
            )

        return ValidationStep(
            step_name="allowed_tables_check",
            passed=True,
            message=f"All tables are allowed: {', '.join(sorted(used_tables))}",
            sql_attempted=sql,
        )

    def _extract_table_names(self, sql: str) -> Set[str]:
        """
        Extract table names from SQL.

        Handles:
        - FROM table_name
        - FROM table_name alias
        - JOIN table_name
        - JOIN table_name AS alias
        - schema.table_name

        Returns lowercase table names (without schema prefix).
        """
        tables: Set[str] = set()

        # Normalize whitespace
        normalized = " ".join(sql.split())

        # Pattern to match table references after FROM/JOIN keywords
        # Captures: table_name or schema.table_name, optionally followed by alias
        # (?:schema\.)? - optional schema prefix
        # (\w+) - table name
        # (?:\s+(?:AS\s+)?(\w+))? - optional alias
        table_pattern = re.compile(
            r'\b(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?',
            re.IGNORECASE
        )

        for match in table_pattern.finditer(normalized):
            table_name = match.group(1).lower()
            tables.add(table_name)

        return tables

    async def _check_syntax(self, sql: str, trace_id: str) -> ValidationStep:
        """Validate SQL syntax using EXPLAIN."""
        try:
            async with self.db_client.acquire_connection(read_only=True) as conn:
                await conn.execute(f"EXPLAIN {sql}")

            return ValidationStep(
                step_name="syntax_check",
                passed=True,
                message="SQL syntax validated successfully",
                sql_attempted=sql,
            )

        except Exception as e:
            logger.debug(
                "SQL syntax validation failed",
                error=str(e),
                trace_id=trace_id,
            )
            return ValidationStep(
                step_name="syntax_check",
                passed=False,
                message=f"SQL syntax error: {str(e)}",
                sql_attempted=sql,
            )
