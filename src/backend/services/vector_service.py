"""
Vector Service for orchestrating vector store operations.

This service provides business logic for vector store operations,
coordinating schema indexing and semantic search.
"""

from typing import Any, Dict, List, Optional, Tuple

from backend.domain.base_enums import NodeType
from backend.domain.errors import VectorStoreError
from backend.domain.responses import (
    VectorSearchResult,
    IndexSchemaStats,
    DropCollectionResult,
    VectorCollectionStats,
)
from backend.domain.schema_nodes import TableNode, ColumnNode, RelationshipNode
from backend.domain.vector_metadata import (
    TableMetadata,
    ColumnMetadata,
    RelationshipMetadata,
    SchemaNodeMetadata,
)
from backend.repositories.vector_repository import VectorRepository
from backend.services.schema_service import SchemaService
from backend.utils.logging import get_module_logger
from backend.utils.tracing import current_trace_id

logger = get_module_logger()


class VectorService:
    """
    Service for vector store business logic.

    Orchestrates schema indexing and search by transforming schema nodes
    into vector documents and coordinating with the vector repository.
    """

    def __init__(
        self,
        vector_repository: VectorRepository,
        schema_service: SchemaService,
        batch_size: int = 100,
    ):
        """
        Initialize vector service.

        Args:
            vector_repository: VectorRepository for data access
            schema_service: SchemaService for fetching schema with descriptions
            batch_size: Batch size for adding documents
        """
        self.vector_repo = vector_repository
        self.schema_service = schema_service
        self.batch_size = batch_size

        logger.info(
            "VectorService initialized",
            batch_size=batch_size,
            trace_id=current_trace_id(),
        )

    async def index_schema(
        self,
        schema: str = "public",
        replace_existing: bool = True,
    ) -> IndexSchemaStats:
        """
        Index schema into vector store.

        Fetches all schema elements (tables, columns, relationships),
        transforms them into documents, and adds to vector store.

        **Replacement Strategy**:
        1. If replace_existing=False: Additive indexing (keeps existing vectors)
        2. If replace_existing=True: Complete replacement:
           - Fetch and transform all data FIRST
           - Drop entire vector store table
           - Reconnect to recreate table and indexes
           - Add all new documents in batches

        **Note**: With replace_existing=True, ALL vectors are dropped (not just
        the specified schema). This ensures a clean slate for re-indexing.

        Args:
            schema: PostgreSQL schema name to index
            replace_existing: If True, drop and recreate vector store before indexing

        Returns:
            IndexSchemaStats with indexing statistics

        Raises:
            VectorStoreError: If indexing fails
        """
        trace_id = current_trace_id()

        logger.info(
            "Starting schema indexing",
            schema=schema,
            replace_existing=replace_existing,
            trace_id=trace_id,
        )

        all_texts: List[str] = []
        total_added = 0

        try:
            # Step 1: Fetch all schema elements FIRST (before any drop operation)
            # This ensures we have all data ready before modifying the database
            logger.info("Fetching schema elements", schema=schema, trace_id=trace_id)

            tables = await self.schema_service.get_all_tables(schema=schema)
            logger.info(f"Fetched {len(tables)} tables", trace_id=trace_id)

            relationships = await self.schema_service.get_all_relationships(schema=schema)
            logger.info(f"Fetched {len(relationships)} relationships", trace_id=trace_id)

            # Fetch columns for all tables
            all_columns: List[ColumnNode] = []
            for table in tables:
                columns = await self.schema_service.get_table_columns(
                    table_name=table.table_name,
                    schema=schema,
                )
                all_columns.extend(columns)

            logger.info(f"Fetched {len(all_columns)} columns", trace_id=trace_id)

            # Step 2: Transform schema elements to documents
            # Prepare all data before touching the database
            logger.info("Transforming schema elements to documents", trace_id=trace_id)

            all_texts: List[str] = []
            all_metadatas: List[SchemaNodeMetadata] = []

            # Transform tables
            for table in tables:
                text, metadata = self._transform_table_to_document(table, schema)
                all_texts.append(text)
                all_metadatas.append(metadata)

            # Transform columns
            for column in all_columns:
                text, metadata = self._transform_column_to_document(column, schema)
                all_texts.append(text)
                all_metadatas.append(metadata)

            # Transform relationships
            for relationship in relationships:
                text, metadata = self._transform_relationship_to_document(relationship, schema)
                all_texts.append(text)
                all_metadatas.append(metadata)

            logger.info(
                "Schema elements transformed",
                total_documents=len(all_texts),
                trace_id=trace_id,
            )

            # Step 3: Drop and recreate collection ONLY if replacement is requested
            # AND we have successfully prepared new data
            if replace_existing:
                logger.info(
                    "Dropping vector collection (new data ready)",
                    trace_id=trace_id
                )
                await self.vector_repo.drop_collection(drop_table=True)

                logger.info(
                    "Vector collection dropped (will be recreated on next add)",
                    trace_id=trace_id,
                )

            # Step 4: Add documents to vector store in batches (ATOMIC transaction)
            # All batches succeed or all fail together. If any batch fails, entire operation is rolled back.
            logger.info(
                "Adding documents to vector store (atomic transaction)",
                total_documents=len(all_texts),
                batch_size=self.batch_size,
                trace_id=trace_id,
            )

            # Ensure setup before starting transaction
            await self.vector_repo.ensure_setup()

            # Use a single transaction for all batches (atomic all-or-nothing)
            total_added = 0
            batch_number = 0
            async with self.vector_repo.db.acquire_connection() as conn:
                async with conn.transaction():
                    try:
                        for i in range(0, len(all_texts), self.batch_size):
                            batch_number += 1
                            batch_texts = all_texts[i : i + self.batch_size]
                            batch_metadatas = all_metadatas[i : i + self.batch_size]

                            logger.debug(
                                f"Processing batch {batch_number}",
                                batch_size=len(batch_texts),
                                trace_id=trace_id,
                            )

                            # Add vectors using the same connection (within transaction)
                            await self.vector_repo._add_vectors_with_connection(
                                conn=conn,
                                texts=batch_texts,
                                metadatas=batch_metadatas,
                            )

                            total_added += len(batch_texts)
                            logger.info(
                                f"Batch {batch_number} added: {total_added}/{len(all_texts)} documents",
                                trace_id=trace_id,
                            )

                    except Exception as e:
                        # Transaction will auto-rollback on exception
                        logger.error(
                            f"Batch {batch_number} failed - rolling back entire transaction",
                            batch_number=batch_number,
                            total_batches=len(range(0, len(all_texts), self.batch_size)),
                            documents_attempted=total_added,
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )
                        raise VectorStoreError(
                            f"Schema indexing failed at batch {batch_number}/{len(range(0, len(all_texts), self.batch_size))}: {e}. "
                            f"Transaction rolled back, no documents were indexed."
                        ) from e

            # Step 5: Return statistics
            stats = IndexSchemaStats(
                schema_name=schema,
                tables_indexed=len(tables),
                columns_indexed=len(all_columns),
                relationships_indexed=len(relationships),
                total_documents=total_added,
            )

            logger.info(
                "Schema indexing completed",
                schema_name=stats.schema_name,
                tables_indexed=stats.tables_indexed,
                columns_indexed=stats.columns_indexed,
                relationships_indexed=stats.relationships_indexed,
                total_documents=stats.total_documents,
                trace_id=trace_id,
            )

            return stats

        except VectorStoreError:
            raise
        except Exception as e:
            # Log detailed error with context
            logger.error(
                "Failed to index schema",
                error=str(e),
                schema=schema,
                documents_prepared=len(all_texts) if 'all_texts' in locals() else 0,
                documents_added=total_added,
                trace_id=trace_id,
                exc_info=True,
            )

            raise VectorStoreError(f"Failed to index schema: {e}") from e

    async def search_schema(
        self,
        query: str,
        k: int = 5,
        schema_name: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """
        Search schema using vector similarity.

        Args:
            query: Natural language query
            k: Number of results to return
            schema_name: Optional filter by schema name
            node_types: Optional filter by node types (e.g., ["table", "column"])
            min_similarity: Optional minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of VectorSearchResult objects

        Raises:
            VectorStoreError: If search fails
        """
        trace_id = current_trace_id()

        logger.info(
            "Searching schema",
            query_length=len(query),
            k=k,
            schema_name=schema_name,
            node_types=node_types,
            min_similarity=min_similarity,
            trace_id=trace_id,
        )

        try:
            # Build metadata filter
            filters: Dict[str, Any] = {}
            if schema_name:
                filters["schema_name"] = schema_name
            if node_types:
                # Note: LangChain PGVector supports basic equality filters
                # For multiple node_types, we'd need to search multiple times
                # or use raw SQL. For now, support single node_type filter.
                if len(node_types) == 1:
                    filters["node_type"] = node_types[0]
                else:
                    logger.warning(
                        "Multiple node_types not supported, ignoring filter",
                        node_types=node_types,
                        trace_id=trace_id,
                    )

            # Search vector store
            results = await self.vector_repo.search_similar(
                query=query,
                k=k,
                filters=filters if filters else None,
                min_similarity=min_similarity,
            )

            logger.info(
                "Schema search completed",
                results_found=len(results),
                trace_id=trace_id,
            )

            return results

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to search schema",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to search schema: {e}") from e

    async def drop_collection(self, drop_table: bool = True) -> DropCollectionResult:
        """
        Drop vector collection (table and/or indexes).

        **Admin operation**: This permanently deletes the vector store.

        Args:
            drop_table: If True, drop entire table (including all indexes).
                       If False, only drop HNSW index (table remains).

        Returns:
            DropCollectionResult with cleanup details

        Raises:
            VectorStoreError: If cleanup fails
        """
        trace_id = current_trace_id()

        logger.info(
            "Dropping vector collection",
            drop_table=drop_table,
            trace_id=trace_id,
        )

        try:
            result_dict = await self.vector_repo.drop_collection(drop_table=drop_table)

            result = DropCollectionResult(**result_dict)

            logger.info(
                "Vector collection dropped successfully",
                action=result.action,
                table_name=result.table_name,
                index_name=result.index_name,
                indexes_dropped=result.indexes_dropped,
                trace_id=trace_id,
            )

            return result

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to drop vector collection",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to drop vector collection: {e}") from e

    async def get_collection_stats(self) -> VectorCollectionStats:
        """
        Get vector collection statistics.

        Returns:
            VectorCollectionStats with collection statistics

        Raises:
            VectorStoreError: If query fails
        """
        trace_id = current_trace_id()

        logger.info("Fetching collection stats", trace_id=trace_id)

        try:
            stats_dict = await self.vector_repo.get_stats()

            stats = VectorCollectionStats(**stats_dict)

            logger.info(
                "Collection stats fetched",
                total_documents=stats.total_documents,
                trace_id=trace_id,
            )

            return stats

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to fetch collection stats",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to fetch collection stats: {e}") from e

    # -------------------------
    # Private helper methods for document transformation
    # -------------------------

    def _transform_table_to_document(
        self, table: TableNode, schema: str
    ) -> Tuple[str, TableMetadata]:
        """
        Transform TableNode to document for vectorization.

        Args:
            table: TableNode to transform
            schema: Schema name

        Returns:
            Tuple of (content, table metadata)
        """
        # Content for embedding
        content_parts = [
            f"Table: {table.table_name}",
        ]

        if table.description:
            content_parts.append(f"Description: {table.description}")

        content = "\n".join(content_parts)

        # Metadata (not embedded)
        metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name=table.table_name,
            schema_name=schema,
            table_type=table.metadata.get("table_type", "unknown"),
        )

        return content, metadata

    def _transform_column_to_document(
        self, column: ColumnNode, schema: str
    ) -> Tuple[str, ColumnMetadata]:
        """
        Transform ColumnNode to document for vectorization.

        Args:
            column: ColumnNode to transform
            schema: Schema name

        Returns:
            Tuple of (content, column metadata)
        """
        # Content for embedding (minimal - just name and description)
        # Note: Type and constraints are in metadata, not needed for semantic search
        content_parts = [f"Column: {column.table_name}.{column.column_name}"]

        if column.description:
            content_parts.append(f"Description: {column.description}")

        content = "\n".join(content_parts)

        # Metadata (not embedded, but stored)
        metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name=column.table_name,
            column_name=column.column_name,
            schema_name=schema,
            data_type=column.data_type,
            is_primary_key=column.is_primary_key,
            is_foreign_key=column.is_foreign_key,
            is_unique=column.is_unique,
            is_nullable=column.is_nullable,
            sample_values=column.metadata.get("sample_values"),  # Optional field
        )

        return content, metadata

    def _transform_relationship_to_document(
        self, relationship: RelationshipNode, schema: str
    ) -> Tuple[str, RelationshipMetadata]:
        """
        Transform RelationshipNode to document for vectorization.

        Args:
            relationship: RelationshipNode to transform
            schema: Schema name

        Returns:
            Tuple of (content, relationship metadata)
        """
        # Content for embedding
        # Describes foreign key relationship: child_table.fk_column -> parent_table.pk_column
        content_parts = [
            f"Foreign key: {relationship.from_table}.{relationship.from_column} "
            f"references {relationship.to_table}.{relationship.to_column}"
        ]

        if relationship.description:
            content_parts.append(f"Description: {relationship.description}")

        content = "\n".join(content_parts)

        # Metadata (not embedded)
        metadata = RelationshipMetadata(
            node_type=NodeType.RELATIONSHIP.value,
            constraint_name=relationship.constraint_name,
            from_table=relationship.from_table,
            from_column=relationship.from_column,
            to_table=relationship.to_table,
            to_column=relationship.to_column,
            schema_name=schema,
        )

        return content, metadata
