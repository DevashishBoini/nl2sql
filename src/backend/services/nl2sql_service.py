"""
NL2SQL Service - Main orchestrator for natural language to SQL conversion.

This service is a THIN ORCHESTRATOR that coordinates repositories:
1. VectorRepository - Schema retrieval
2. SchemaFilteringRepository - Deterministic filtering
3. SQLGenerationRepository - LLM-based SQL generation
4. SQLValidationRepository - SQL validation
5. SQLExecutionRepository - SQL execution

Key principles:
- Service layer only orchestrates, no business logic
- All logic lives in repositories
- Clear separation of concerns
"""

from datetime import datetime, timezone
from typing import Optional

from backend.config import NL2SQLConfig, Settings
from backend.domain.base_enums import NodeType, PipelineStepName, QueryStatus
from backend.domain.pipeline import PipelineState
from backend.domain.responses import (
    FilteredColumn,
    FilteredTable,
    NL2SQLQueryResponse,
    RetrievedNode,
    SchemaGrounding,
    SchemaProvenance,
    ValidationStep,
)
from backend.repositories.schema_filtering import SchemaFilteringRepository
from backend.repositories.sql_execution import SQLExecutionRepository
from backend.repositories.sql_generation import SQLGenerationRepository
from backend.repositories.sql_validation import SQLValidationRepository
from backend.repositories.vector_repository import VectorRepository
from backend.utils.logging import get_module_logger
from backend.utils.tracing import current_trace_id

logger = get_module_logger()


class NL2SQLService:
    """
    Main orchestrator for NL2SQL pipeline.

    Coordinates repositories to execute the full pipeline:
    1. Retrieval (vector search)
    2. Filtering (deterministic)
    3. Table context (descriptions)
    4. SQL generation (LLM)
    5. Validation (static + EXPLAIN)
    6. Execution (read-only)
    """

    def __init__(
        self,
        vector_repository: VectorRepository,
        schema_filtering_repository: SchemaFilteringRepository,
        sql_generation_repository: SQLGenerationRepository,
        sql_validation_repository: SQLValidationRepository,
        sql_execution_repository: SQLExecutionRepository,
        config: NL2SQLConfig,
    ):
        self.vector_repo = vector_repository
        self.filtering_repo = schema_filtering_repository
        self.generation_repo = sql_generation_repository
        self.validation_repo = sql_validation_repository
        self.execution_repo = sql_execution_repository
        self.config = config

        logger.info(
            "NL2SQLService initialized",
            retrieval_top_k=config.retrieval_top_k,
            max_tables=config.max_tables,
            max_retries=config.max_retries,
        )

    async def execute_query(
        self,
        user_query: str,
        schema_name: str = "public",
        row_limit: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> NL2SQLQueryResponse:
        """
        Execute the full NL2SQL pipeline.

        Args:
            user_query: Natural language query from user
            schema_name: PostgreSQL schema to query
            row_limit: Max rows to return
            timeout_seconds: SQL execution timeout

        Returns:
            NL2SQLQueryResponse with SQL, results, grounding, and provenance
        """
        trace_id = current_trace_id()
        start_time = datetime.now(timezone.utc)

        logger.info(
            "Starting NL2SQL pipeline",
            query_length=len(user_query),
            schema_name=schema_name,
            trace_id=trace_id,
        )

        # Validate and cap row limit
        requested_limit = row_limit if row_limit is not None else self.config.default_row_limit
        effective_row_limit = min(requested_limit, self.config.max_row_limit)

        if row_limit is not None and row_limit > self.config.max_row_limit:
            logger.warning(
                "Row limit capped at max_row_limit",
                requested=row_limit,
                max_allowed=self.config.max_row_limit,
                trace_id=trace_id,
            )

        # Initialize pipeline state
        state = PipelineState(
            user_query=user_query,
            schema_name=schema_name,
            row_limit=effective_row_limit,
            timeout_seconds=timeout_seconds or self.config.query_timeout_seconds,
        )

        try:
            # Step 1: Retrieval
            await self._step_retrieval(state)

            # Step 2: Filtering
            self._step_filtering(state)

            # Step 3: Table context
            await self._step_table_context(state)

            # Step 4-6: Generation + Validation + Retry
            await self._step_generate_and_validate(state)

            # Step 7: Execution
            if state.generated_sql:
                await self._step_execution(state)

            return self._build_response(state, start_time, QueryStatus.COMPLETED)

        except Exception as e:
            logger.error(
                "NL2SQL pipeline failed",
                error=str(e),
                error_step=state.error_step,
                trace_id=trace_id,
                exc_info=True,
            )
            state.error_message = str(e)
            return self._build_response(state, start_time, QueryStatus.FAILED)

    # =========================================================================
    # Pipeline Steps (thin - delegate to repositories)
    # =========================================================================

    async def _step_retrieval(self, state: PipelineState) -> None:
        """Step 1: Retrieve schema nodes via vector search."""
        trace_id = current_trace_id()
        state.error_step = PipelineStepName.SCHEMA_RETRIEVAL.value

        logger.info("Step 1: Schema retrieval", trace_id=trace_id)

        # Retrieve columns
        column_results = await self.vector_repo.search_similar(
            query=state.user_query,
            k=self.config.retrieval_top_k,
            filters={"node_type": NodeType.COLUMN.value},
            min_similarity=None,
        )

        # Retrieve relationships
        relationship_results = await self.vector_repo.search_similar(
            query=state.user_query,
            k=self.config.retrieval_top_k,
            filters={"node_type": NodeType.RELATIONSHIP.value},
            min_similarity=None,
        )

        state.retrieved_nodes = column_results + relationship_results

        logger.info(
            "Schema retrieval complete",
            columns=len(column_results),
            relationships=len(relationship_results),
            trace_id=trace_id,
        )

    def _step_filtering(self, state: PipelineState) -> None:
        """Step 2: Filter schema via repository."""
        state.error_step = PipelineStepName.SCHEMA_FILTERING.value

        columns, relationships, tables, final_tables = self.filtering_repo.filter_schema(
            retrieved_nodes=state.retrieved_nodes
        )

        state.filtered_columns = columns
        state.filtered_relationships = relationships
        state.filtered_tables = tables
        # Store final_tables for table context step
        state._final_table_names = final_tables  # type: ignore

    async def _step_table_context(self, state: PipelineState) -> None:
        """Step 3: Fetch table descriptions (deterministic, no embeddings)."""
        trace_id = current_trace_id()
        state.error_step = PipelineStepName.TABLE_CONTEXT.value

        logger.info("Step 3: Fetching table context", trace_id=trace_id)

        final_tables = getattr(state, '_final_table_names', set())

        for table in state.filtered_tables:
            # Fetch table description via exact match (deterministic, no embeddings)
            results = await self.vector_repo.get_by_exact_match(
                filters={
                    "node_type": NodeType.TABLE.value,
                    "table_name": table.table_name,
                },
                limit=1,
            )

            if results:
                description = results[0].metadata.get("description")
                if description:
                    state.table_descriptions[table.table_name] = description
                    table.description = description

            # Fetch additional columns for tables with few columns
            existing_cols = {
                c.column_name for c in state.filtered_columns
                if c.table_name == table.table_name
            }
            if len(existing_cols) < 3:
                await self._fetch_additional_columns(state, table.table_name, existing_cols)

        # Update column counts
        for table in state.filtered_tables:
            table.column_count = sum(
                1 for c in state.filtered_columns if c.table_name == table.table_name
            )

        logger.info(
            "Table context fetched",
            tables_with_descriptions=len(state.table_descriptions),
            trace_id=trace_id,
        )

    async def _fetch_additional_columns(
        self, state: PipelineState, table_name: str, existing_cols: set
    ) -> None:
        """Fetch additional columns for tables with few retrieved columns (deterministic)."""
        # Use exact match - no embeddings, deterministic retrieval
        results = await self.vector_repo.get_by_exact_match(
            filters={
                "node_type": NodeType.COLUMN.value,
                "table_name": table_name,
            },
            limit=10,  # Fetch more, filter below
        )

        added_count = 0
        for result in results:
            if added_count >= 5:  # Cap at 5 additional columns
                break

            col_name = result.metadata.get("column_name")
            if col_name and col_name not in existing_cols:
                state.filtered_columns.append(FilteredColumn(
                    table_name=table_name,
                    column_name=col_name,
                    data_type=result.metadata.get("data_type", "unknown"),
                    description=result.metadata.get("description"),
                    is_primary_key=result.metadata.get("is_primary_key", False),
                    is_foreign_key=result.metadata.get("is_foreign_key", False),
                ))
                existing_cols.add(col_name)
                added_count += 1

    async def _step_generate_and_validate(self, state: PipelineState) -> None:
        """Steps 4-6: Generate SQL, validate, retry if needed."""
        trace_id = current_trace_id()
        state.error_step = PipelineStepName.SQL_GENERATION.value

        if not state.filtered_columns:
            state.error_message = "No relevant schema columns found for query"
            logger.warning(state.error_message, trace_id=trace_id)
            return

        last_error: Optional[str] = None
        last_sql: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            logger.info(
                f"SQL generation attempt {attempt + 1}/{self.config.max_retries + 1}",
                trace_id=trace_id,
            )

            # Generate SQL via repository
            try:
                generation_result = await self.generation_repo.generate_sql(
                    user_query=state.user_query,
                    tables=state.filtered_tables,
                    columns=state.filtered_columns,
                    relationships=state.filtered_relationships,
                    row_limit=state.row_limit,
                    last_error=last_error,
                    last_sql=last_sql,
                )
            except Exception as e:
                logger.error(
                    "LLM call failed after internal retries",
                    error=str(e),
                    trace_id=trace_id,
                )
                state.error_message = f"LLM service error: {e}"
                state.error_step = PipelineStepName.LLM_CALL.value
                return

            # Check if LLM indicated it cannot generate SQL
            if not generation_result.success:
                state.validation_steps.append(ValidationStep(
                    step_name="generation_feasibility_check",
                    passed=False,
                    message=generation_result.reason or "LLM could not generate SQL",
                    sql_attempted=None,
                ))
                state.error_message = generation_result.reason or "Unable to generate SQL"
                logger.warning(
                    "LLM indicated cannot generate SQL",
                    reason=generation_result.reason,
                    trace_id=trace_id,
                )
                return

            sql = generation_result.sql
            if not sql:
                last_error = "LLM returned success but no SQL"
                state.retries += 1
                continue

            # Validate SQL via repository
            validation_result = await self.validation_repo.validate(
                sql=sql,
                tables=state.filtered_tables,
                columns=state.filtered_columns,
            )
            state.validation_steps.append(validation_result)

            if validation_result.passed:
                state.generated_sql = sql
                logger.info(
                    "SQL generation and validation successful",
                    attempt=attempt + 1,
                    trace_id=trace_id,
                )
                return

            # Validation failed - prepare for retry
            last_error = validation_result.message
            last_sql = sql
            state.retries += 1

        state.error_message = f"SQL generation failed after {self.config.max_retries + 1} attempts: {last_error}"
        logger.warning(state.error_message, trace_id=trace_id)

    async def _step_execution(self, state: PipelineState) -> None:
        """Step 7: Execute validated SQL."""
        trace_id = current_trace_id()
        state.error_step = PipelineStepName.SQL_EXECUTION.value

        if not state.generated_sql:
            return

        logger.info("Step 7: Executing SQL", trace_id=trace_id)

        try:
            state.execution_result = await self.execution_repo.execute(
                sql=state.generated_sql,
                schema=state.schema_name,
                timeout_seconds=state.timeout_seconds,
                row_limit=state.row_limit,
            )
        except Exception as e:
            state.error_message = f"SQL execution failed: {str(e)}"
            logger.error(state.error_message, trace_id=trace_id, exc_info=True)

    # =========================================================================
    # Response Building
    # =========================================================================

    def _build_response(
        self,
        state: PipelineState,
        start_time: datetime,
        status: QueryStatus,
    ) -> NL2SQLQueryResponse:
        """Build the final API response."""
        trace_id = current_trace_id() or "unknown"
        end_time = datetime.now(timezone.utc)
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        # Build retrieved nodes for provenance
        retrieved_nodes = [
            RetrievedNode(
                node_type=NodeType(node.metadata.get("node_type", NodeType.COLUMN.value)),
                content=node.content,
                table_name=node.metadata.get("table_name"),
                column_name=node.metadata.get("column_name"),
                similarity_score=node.similarity_score,
                metadata=node.metadata,
            )
            for node in state.retrieved_nodes
        ]

        # Build grounding
        grounding = SchemaGrounding(
            tables=state.filtered_tables,
            columns=state.filtered_columns,
            relationships=state.filtered_relationships,
        )

        # Build provenance
        provenance = SchemaProvenance(
            retrieved_nodes=retrieved_nodes,
            validation_steps=state.validation_steps,
            retries=state.retries,
        )

        return NL2SQLQueryResponse(
            trace_id=trace_id,
            status=status,
            query=state.user_query,
            sql=state.generated_sql,
            results=state.execution_result,
            error_message=state.error_message,
            error_step=state.error_step,
            grounding=grounding,
            provenance=provenance,
            total_time_ms=total_time_ms,
        )
