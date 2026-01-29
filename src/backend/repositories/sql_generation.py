"""
SQL Generation Repository.

Handles LLM-based SQL generation:
- Prompt building with schema context
- LLM interaction
- Response parsing
"""

from typing import List, Optional

from backend.config import LLMConfig
from backend.domain.responses import (
    FilteredColumn,
    FilteredRelationship,
    FilteredTable,
    LLMGenerationResult,
)
from backend.infrastructure.llm_client import LLMClient
from backend.utils.logging import get_module_logger
from backend.utils.tracing import current_trace_id

logger = get_module_logger()


class SQLGenerationRepository:
    """
    Repository for LLM-based SQL generation.

    Handles prompt construction and LLM interaction.
    """

    def __init__(self, llm_client: LLMClient, config: LLMConfig):
        self.llm_client = llm_client
        self.config = config

    async def generate_sql(
        self,
        user_query: str,
        tables: List[FilteredTable],
        columns: List[FilteredColumn],
        relationships: List[FilteredRelationship],
        row_limit: int,
        last_error: Optional[str] = None,
        last_sql: Optional[str] = None,
    ) -> LLMGenerationResult:
        """
        Generate SQL for user query using LLM.

        Args:
            user_query: Natural language query from user
            tables: Filtered tables with descriptions
            columns: Filtered columns
            relationships: Filtered FK relationships
            row_limit: Max rows for LIMIT clause
            last_error: Error from previous attempt (for retry)
            last_sql: SQL from previous attempt (for retry)

        Returns:
            LLMGenerationResult with success status and SQL or reason
        """
        trace_id = current_trace_id()

        prompt = self._build_prompt(
            user_query=user_query,
            tables=tables,
            columns=columns,
            relationships=relationships,
            row_limit=row_limit,
            last_error=last_error,
            last_sql=last_sql,
        )

        logger.debug(
            "Calling LLM for SQL generation",
            prompt_length=len(prompt),
            trace_id=trace_id,
        )

        # Use JSON mode for structured response (configurable)
        # Note: system_prompt MUST mention "JSON" for json_mode to work
        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=(
                "You are a PostgreSQL SQL expert generating right sql statements from the schema provided. Generate valid, safe SELECT queries. "
                "You MUST respond with a valid JSON object only - no markdown, no explanation."
            ),
            temperature=0.0,
            json_mode=self.config.json_mode,  # From LLMConfig
        )

        return LLMGenerationResult.from_llm_response(response)

    def _build_prompt(
        self,
        user_query: str,
        tables: List[FilteredTable],
        columns: List[FilteredColumn],
        relationships: List[FilteredRelationship],
        row_limit: int,
        last_error: Optional[str] = None,
        last_sql: Optional[str] = None,
    ) -> str:
        """Build the prompt for SQL generation."""
        tables_section = "\n".join(
            f"- {t.table_name}: {t.description or 'No description'}"
            for t in tables
        )

        columns_section = "\n".join(
            f"- {c.table_name}.{c.column_name} ({c.data_type})"
            + (f" [PK]" if c.is_primary_key else "")
            + (f" [FK]" if c.is_foreign_key else "")
            + (f": {c.description}" if c.description else "")
            for c in columns
        )

        relationships_section = "\n".join(
            f"- {r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column}"
            for r in relationships
        ) or "No relationships"

        prompt = f"""Generate a PostgreSQL SELECT query for the user's question.

## REQUIRED JSON RESPONSE FORMAT
You MUST respond with ONLY a valid JSON object matching one of these schemas:

Success case:
{{"success": true, "sql": "<valid PostgreSQL SELECT statement>"}}

Failure case (if schema is insufficient or question is unclear):
{{"success": false, "reason": "<brief explanation>"}}

DO NOT include any text outside the JSON object. No markdown, no explanations, just raw JSON.

## USER QUESTION
{user_query}

## ALLOWED SCHEMA (Use ONLY these - do NOT invent tables, columns, or joins)

### Tables:
{tables_section}

### Columns:
{columns_section}

### Relationships (for JOINs):
{relationships_section}

## SQL GENERATION RULES
1. Use ONLY the tables and columns listed above
2. Use ONLY the relationships listed above for JOINs
3. PostgreSQL dialect only
4. SELECT queries ONLY (no INSERT, UPDATE, DELETE, DDL)
5. ROW LIMIT RULES (MANDATORY):
   - Default: Include LIMIT {row_limit}
   - If user asks for fewer rows: Use their requested limit
   - If user asks for MORE than {row_limit} rows: Ignore user limit request and Use LIMIT {row_limit} (hard cap, NEVER exceed)
   - NEVER generate a query without LIMIT
6. No SQL comments
7. Avoid SELECT * - list specific columns

## STRING MATCHING
- Text/string values: Use ILIKE or LOWER() for case-insensitive matching
- ID columns ([PK], [FK], *_id): Use exact matching

## THINKING PROCESS (internal)
1. What columns to SELECT?
2. Which tables (FROM)?
3. How to JOIN (using relationships)?
4. Filters (WHERE)?
5. Aggregations (GROUP BY)?
6. Ordering (ORDER BY)?
7. Limit (LIMIT {row_limit})?

"""

        if last_error and last_sql:
            prompt += f"""## PREVIOUS ATTEMPT FAILED - FIX REQUIRED
Previous SQL: {last_sql}
Error: {last_error}

Fix the SQL and respond with corrected JSON.

"""

        prompt += "Respond with JSON only:"

        return prompt
