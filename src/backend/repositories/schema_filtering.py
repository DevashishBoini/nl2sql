"""
Schema Filtering Repository.

Handles deterministic filtering of retrieved schema nodes:
- Separates columns and relationships
- Detects query type (single-table vs multi-table)
- Applies structural gates for table selection
- Ensures FK connectivity
"""

from typing import Dict, List, Optional, Set

from backend.config import NL2SQLConfig
from backend.domain.base_enums import NodeType
from backend.domain.responses import (
    FilteredColumn,
    FilteredRelationship,
    FilteredTable,
    VectorSearchResult,
)
from backend.domain.types import TableColumnsMap, TableScoresMap
from backend.utils.logging import get_module_logger
from backend.utils.tracing import get_trace_id

logger = get_module_logger()


class SchemaFilteringRepository:
    """
    Repository for deterministic schema filtering.

    Principles:
    - Use STRUCTURE to gate inclusion (binary decisions)
    - Use SIMILARITY only to break ties (secondary signal)
    - FK existence is deterministic truth, not weighted by similarity
    """

    def __init__(self, config: NL2SQLConfig):
        self.config = config

    def filter_schema(
        self,
        retrieved_nodes: List[VectorSearchResult],
    ) -> tuple[
        List[FilteredColumn],
        List[FilteredRelationship],
        List[FilteredTable],
        Set[str],  # final_tables for context fetching
    ]:
        """
        Filter retrieved nodes into structured schema components.

        Args:
            retrieved_nodes: Raw nodes from vector search

        Returns:
            Tuple of (filtered_columns, filtered_relationships, filtered_tables, final_table_names)
        """
        trace_id = get_trace_id()

        logger.info(
            "Filtering schema nodes",
            input_nodes=len(retrieved_nodes),
            trace_id=trace_id,
        )

        # Phase 1: Separate and sort nodes
        raw_columns, raw_relationships = self._separate_nodes(retrieved_nodes)

        # Phase 2: Identify structural sets
        column_tables = self._extract_tables_from_columns(raw_columns)
        connected_tables = self._extract_tables_from_relationships(raw_relationships)

        logger.debug(
            "Structural analysis",
            column_tables=list(column_tables),
            connected_tables=list(connected_tables),
            trace_id=trace_id,
        )

        # Phase 3: Detect query type
        is_single_table_query = self._is_single_table_query(column_tables)

        if is_single_table_query:
            logger.debug(
                "Detected single-table query",
                table=list(column_tables)[0] if column_tables else None,
                trace_id=trace_id,
            )

        # Phase 4: Build candidate tables
        candidate_tables = self._build_candidate_tables(
            column_tables, raw_relationships, is_single_table_query
        )

        # Phase 5: Select columns
        selected_columns = raw_columns[:self.config.max_columns]

        # Phase 6: Select tables
        if len(candidate_tables) <= self.config.max_tables:
            final_tables = candidate_tables
        else:
            final_tables = self._select_tables_structurally(
                candidate_tables=candidate_tables,
                column_tables=column_tables,
                connected_tables=connected_tables,
                selected_columns=selected_columns,
                raw_relationships=raw_relationships[:self.config.max_relationships],
                is_single_table_query=is_single_table_query,
                trace_id=trace_id,
            )
            # Re-filter columns
            selected_columns = [
                node for node in selected_columns
                if node.metadata.get("table_name") in final_tables
            ]

        # Phase 7: Filter relationships
        selected_relationships = [
            node for node in raw_relationships
            if (
                node.metadata.get("from_table") in final_tables
                and node.metadata.get("to_table") in final_tables
            )
        ][:self.config.max_relationships]

        # Convert to domain models
        filtered_columns = self._convert_to_filtered_columns(selected_columns)
        filtered_relationships = self._convert_to_filtered_relationships(selected_relationships)

        # Add FK columns from relationships
        filtered_columns = self._add_fk_columns(filtered_columns, filtered_relationships)

        # Build filtered tables
        filtered_tables = self._build_filtered_tables(filtered_columns, final_tables)

        logger.info(
            "Schema filtering complete",
            tables=len(filtered_tables),
            columns=len(filtered_columns),
            relationships=len(filtered_relationships),
            trace_id=trace_id,
        )

        return filtered_columns, filtered_relationships, filtered_tables, final_tables

    def _separate_nodes(
        self, nodes: List[VectorSearchResult]
    ) -> tuple[List[VectorSearchResult], List[VectorSearchResult]]:
        """Separate nodes into columns and relationships, sorted by similarity."""
        columns: List[VectorSearchResult] = []
        relationships: List[VectorSearchResult] = []

        for node in nodes:
            node_type = node.metadata.get("node_type")
            if node_type == NodeType.COLUMN.value:
                columns.append(node)
            elif node_type == NodeType.RELATIONSHIP.value:
                relationships.append(node)

        columns.sort(key=lambda x: x.similarity_score, reverse=True)
        relationships.sort(key=lambda x: x.similarity_score, reverse=True)

        return columns, relationships

    def _extract_tables_from_columns(self, columns: List[VectorSearchResult]) -> Set[str]:
        """Extract unique table names from column nodes."""
        tables: Set[str] = set()
        for node in columns:
            table_name = node.metadata.get("table_name")
            if table_name:
                tables.add(table_name)
        return tables

    def _extract_tables_from_relationships(self, relationships: List[VectorSearchResult]) -> Set[str]:
        """Extract unique table names from relationship nodes."""
        tables: Set[str] = set()
        for node in relationships:
            from_t = node.metadata.get("from_table")
            to_t = node.metadata.get("to_table")
            if from_t:
                tables.add(from_t)
            if to_t:
                tables.add(to_t)
        return tables

    def _is_single_table_query(self, column_tables: Set[str]) -> bool:
        """
        Detect if this is a single-table query (no joins needed).

        Only checks if columns reference a single table.
        FK enforcement later decides if joins are actually required.

        NOTE: We don't check connected_tables here because:
        - Missing relationships != no joins required
        - Sparse FK retrieval could misclassify multi-table queries
        """
        return len(column_tables) == 1

    def _build_candidate_tables(
        self,
        column_tables: Set[str],
        raw_relationships: List[VectorSearchResult],
        is_single_table_query: bool,
    ) -> Set[str]:
        """Build set of candidate tables for selection."""
        candidates = column_tables.copy()

        if not is_single_table_query and raw_relationships:
            for node in raw_relationships[:self.config.max_relationships]:
                from_t = node.metadata.get("from_table")
                to_t = node.metadata.get("to_table")
                if from_t:
                    candidates.add(from_t)
                if to_t:
                    candidates.add(to_t)

        return candidates

    def _select_tables_structurally(
        self,
        candidate_tables: Set[str],
        column_tables: Set[str],
        connected_tables: Set[str],
        selected_columns: List[VectorSearchResult],
        raw_relationships: List[VectorSearchResult],
        is_single_table_query: bool,
        trace_id: str,
    ) -> Set[str]:
        """
        Select tables using STRUCTURAL priority with similarity as tiebreaker.

        Scoring:
        - FK relationship: +3.0 (fixed structural truth)
        - Column presence: +1.0 (structural evidence)
        - Similarity: min(0.3 * avg, 0.15) - capped to prevent overpowering structure
        """
        table_scores: TableScoresMap = {}

        # Calculate average similarity per table
        table_similarities: Dict[str, List[float]] = {}
        for node in selected_columns:
            t = node.metadata.get("table_name")
            if t:
                if t not in table_similarities:
                    table_similarities[t] = []
                table_similarities[t].append(node.similarity_score)

        # Score 1: FK relationship (+3.0 fixed)
        for node in raw_relationships:
            from_t = node.metadata.get("from_table")
            to_t = node.metadata.get("to_table")
            if from_t and from_t in candidate_tables:
                table_scores[from_t] = table_scores.get(from_t, 0) + 3.0
            if to_t and to_t in candidate_tables:
                table_scores[to_t] = table_scores.get(to_t, 0) + 3.0

        # Score 2: Column presence (+1.0)
        for node in selected_columns:
            t = node.metadata.get("table_name")
            if t and t in candidate_tables:
                table_scores[t] = table_scores.get(t, 0) + 1.0

        # Score 3: Similarity tiebreaker (capped at 0.15 to prevent overpowering structure)
        # Without cap: a single high-similarity column could overpower structural signals
        # With cap: similarity remains a tie-breaker, not a decider
        for t in candidate_tables:
            if t in table_similarities and table_similarities[t]:
                avg_sim = sum(table_similarities[t]) / len(table_similarities[t])
                similarity_contribution = min(0.3 * avg_sim, 0.15)
                table_scores[t] = table_scores.get(t, 0) + similarity_contribution

        logger.debug(
            "Structural table scores",
            scores={t: round(s, 2) for t, s in sorted(table_scores.items(), key=lambda x: -x[1])},
            trace_id=trace_id,
        )

        sorted_tables = sorted(
            candidate_tables,
            key=lambda t: table_scores.get(t, 0),
            reverse=True,
        )

        return set(sorted_tables[:self.config.max_tables])

    def _convert_to_filtered_columns(
        self, nodes: List[VectorSearchResult]
    ) -> List[FilteredColumn]:
        """Convert VectorSearchResult nodes to FilteredColumn models."""
        return [
            FilteredColumn(
                table_name=node.metadata.get("table_name", ""),
                column_name=node.metadata.get("column_name", ""),
                data_type=node.metadata.get("data_type", "unknown"),
                description=node.metadata.get("description"),
                is_primary_key=node.metadata.get("is_primary_key", False),
                is_foreign_key=node.metadata.get("is_foreign_key", False),
            )
            for node in nodes
        ]

    def _convert_to_filtered_relationships(
        self, nodes: List[VectorSearchResult]
    ) -> List[FilteredRelationship]:
        """Convert VectorSearchResult nodes to FilteredRelationship models."""
        return [
            FilteredRelationship(
                from_table=node.metadata.get("from_table", ""),
                from_column=node.metadata.get("from_column", ""),
                to_table=node.metadata.get("to_table", ""),
                to_column=node.metadata.get("to_column", ""),
                constraint_name=node.metadata.get("constraint_name"),
            )
            for node in nodes
        ]

    def _add_fk_columns(
        self,
        columns: List[FilteredColumn],
        relationships: List[FilteredRelationship],
    ) -> List[FilteredColumn]:
        """Add FK columns from relationships if not already present."""
        existing_keys = {(c.table_name, c.column_name) for c in columns}
        result = list(columns)

        for rel in relationships:
            if (rel.from_table, rel.from_column) not in existing_keys:
                result.append(FilteredColumn(
                    table_name=rel.from_table,
                    column_name=rel.from_column,
                    data_type="integer",
                    description=f"Foreign key to {rel.to_table}",
                    is_primary_key=False,
                    is_foreign_key=True,
                ))
                existing_keys.add((rel.from_table, rel.from_column))

            if (rel.to_table, rel.to_column) not in existing_keys:
                result.append(FilteredColumn(
                    table_name=rel.to_table,
                    column_name=rel.to_column,
                    data_type="integer",
                    description=f"Primary key of {rel.to_table}",
                    is_primary_key=True,
                    is_foreign_key=False,
                ))
                existing_keys.add((rel.to_table, rel.to_column))

        return result

    def _build_filtered_tables(
        self,
        columns: List[FilteredColumn],
        final_tables: Set[str],
    ) -> List[FilteredTable]:
        """Build FilteredTable list from columns and final table set."""
        table_column_counts: Dict[str, int] = {}
        for col in columns:
            table_column_counts[col.table_name] = (
                table_column_counts.get(col.table_name, 0) + 1
            )

        for table_name in final_tables:
            if table_name not in table_column_counts:
                table_column_counts[table_name] = 0

        return [
            FilteredTable(
                table_name=table_name,
                description=None,
                column_count=count,
            )
            for table_name, count in table_column_counts.items()
        ]
