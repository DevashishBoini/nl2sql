"""
Vector repository for data access operations.

Handles both setup (tables, indexes) and runtime operations (add, search, stats).
Uses shared DatabaseClient for connection pooling.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel

from backend.config import VectorStoreConfig
from backend.config_constants import PGVECTOR_OPS_MAP, DistanceStrategy
from backend.domain.errors import VectorStoreError
from backend.domain.responses import VectorSearchResult
from backend.infrastructure.database_client import DatabaseClient
from backend.infrastructure.embedding_client import EmbeddingClient
from backend.utils.logging import get_module_logger
from backend.utils.tracing import current_trace_id

logger = get_module_logger()


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects to be JSON serializable.

    Converts:
    - datetime/date objects to ISO format strings
    - Decimal to float
    - UUID to string
    - Pydantic models to dict
    - Other non-serializable types to string representation

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None

    # Handle datetime and date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Handle Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # Handle UUID
    if isinstance(obj, UUID):
        return str(obj)

    # Handle Pydantic models
    if isinstance(obj, BaseModel):
        return sanitize_for_json(obj.model_dump())

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}

    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [sanitize_for_json(item) for item in obj]

    # Primitives and already serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: convert to string
    return str(obj)


class VectorRepository:
    """
    Repository for vector store data access operations.

    Handles setup (extension, tables, indexes) and runtime operations (add, search).
    Uses shared DatabaseClient for connection pooling.
    """

    def __init__(
        self,
        db_client: DatabaseClient,
        embedding_client: EmbeddingClient,
        config: VectorStoreConfig,
    ):
        """
        Initialize vector repository.

        Args:
            db_client: Shared database client for connection pooling
            embedding_client: Embedding client for generating embeddings
            config: Vector store configuration
        """
        self.db = db_client
        self.embeddings = embedding_client
        self.config = config
        self._setup_done = False

        logger.info(
            "VectorRepository initialized",
            table_name=config.table_name,
            use_hnsw=config.use_hnsw,
            distance_strategy=str(config.distance_strategy),
            trace_id=current_trace_id(),
        )

    async def ensure_setup(self) -> None:
        """
        Ensure vector store setup is complete (idempotent).

        Creates extension, tables, and indexes if they don't exist.
        Safe to call multiple times.

        Raises:
            VectorStoreError: If setup fails
        """
        if self._setup_done:
            return

        trace_id = current_trace_id()

        try:
            logger.info("Ensuring vector store setup", trace_id=trace_id)

            # Step 1: Enable pgvector extension
            await self._ensure_pgvector_extension()

            # Step 2: Create tables if not exist
            await self._ensure_tables_exist()

            # Step 3: Create HNSW index if configured
            if self.config.use_hnsw:
                await self._ensure_hnsw_index()

            # Step 4: Create metadata indexes
            await self._ensure_metadata_indexes()

            self._setup_done = True
            logger.info("Vector store setup complete", trace_id=trace_id)

        except Exception as e:
            logger.error(
                "Failed to setup vector store",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to setup vector store: {e}") from e

    async def add_vectors(
        self,
        texts: List[str],
        metadatas: Sequence[Union[Mapping[str, Any], BaseModel]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to the store (auto-commit).

        Uses LangChain for embedding generation, asyncpg for database insertion.

        Args:
            texts: List of text content to embed and store
            metadatas: List of metadata (accepts Dict, TypedDict, or Pydantic BaseModel)
            ids: Optional list of document IDs (auto-generated UUIDs if not provided)

        Returns:
            List of document IDs

        Raises:
            VectorStoreError: If validation fails or operation fails
        """
        # Ensure setup is done
        await self.ensure_setup()

        # Acquire connection with write access and insert (auto-commit)
        async with self.db.acquire_connection(read_only=False) as conn:
            return await self._add_vectors_with_connection(conn, texts, metadatas, ids)

    async def _add_vectors_with_connection(
        self,
        conn,
        texts: List[str],
        metadatas: Sequence[Union[Mapping[str, Any], BaseModel]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors using an existing connection (for use within transactions).

        This method does NOT manage transactions - caller is responsible.

        Args:
            conn: asyncpg connection
            texts: List of text content to embed and store
            metadatas: List of metadata (accepts Dict, TypedDict, or Pydantic BaseModel)
            ids: Optional list of document IDs (auto-generated UUIDs if not provided)

        Returns:
            List of document IDs

        Raises:
            VectorStoreError: If validation fails or operation fails
        """
        trace_id = current_trace_id()

        # Validate inputs
        if len(texts) != len(metadatas):
            raise VectorStoreError(
                f"Length mismatch: {len(texts)} texts but {len(metadatas)} metadatas"
            )

        if ids is not None and len(ids) != len(texts):
            raise VectorStoreError(
                f"Length mismatch: {len(texts)} texts but {len(ids)} ids"
            )

        if not texts:
            logger.warning("No texts provided to add_vectors", trace_id=trace_id)
            return []

        logger.info(
            "Adding vectors to repository",
            count=len(texts),
            has_ids=ids is not None,
            trace_id=trace_id,
        )

        try:
            # Step 1: Generate embeddings using LangChain
            logger.debug("Generating embeddings", count=len(texts), trace_id=trace_id)
            embeddings = await self.embeddings.embed_batch(texts)

            # Step 2: Prepare metadata (convert Pydantic to dict if needed)
            metadatas_list = [
                metadata.model_dump() if isinstance(metadata, BaseModel) else dict(metadata)
                for metadata in metadatas
            ]

            # Step 3: Insert documents into database using provided connection
            query = f"""
                INSERT INTO {self.config.table_name}
                    (content, embedding, node_type, schema_name, table_name,
                     column_name, constraint_name, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id::text;
            """

            doc_ids: List[str] = []
            for text, embedding, metadata in zip(texts, embeddings, metadatas_list):
                # Extract structured fields from metadata
                node_type = metadata.get("node_type")
                schema_name = metadata.get("schema_name")
                table_name = metadata.get("table_name")
                column_name = metadata.get("column_name")
                constraint_name = metadata.get("constraint_name")

                # Sanitize and store full metadata as JSONB
                # (converts datetime, Decimal, UUID to JSON-serializable types)
                sanitized_metadata = sanitize_for_json(metadata)
                metadata_json = json.dumps(sanitized_metadata)

                # Convert embedding list to PostgreSQL vector format
                # pgvector expects a string like '[0.1, 0.2, 0.3]'
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                # Insert and get generated ID
                row = await conn.fetchrow(
                    query,
                    text,
                    embedding_str,
                    node_type,
                    schema_name,
                    table_name,
                    column_name,
                    constraint_name,
                    metadata_json,
                )
                doc_ids.append(row["id"])

            logger.info(
                "Vectors added successfully",
                count=len(doc_ids),
                trace_id=trace_id,
            )

            return doc_ids

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to add vectors",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add vectors: {e}") from e

    async def search_similar(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Uses LangChain for query embedding generation, asyncpg for database query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional metadata filters (e.g., {"node_type": "table"})
            min_similarity: Optional minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of VectorSearchResult objects

        Raises:
            VectorStoreError: If validation fails or search fails
        """
        trace_id = current_trace_id()

        # Validate inputs
        if not query or not query.strip():
            raise VectorStoreError("Query cannot be empty")

        if k < 1:
            raise VectorStoreError(f"k must be >= 1, got {k}")

        if min_similarity is not None and not (0.0 <= min_similarity <= 1.0):
            raise VectorStoreError(
                f"min_similarity must be between 0.0 and 1.0, got {min_similarity}"
            )

        # Ensure setup is done
        await self.ensure_setup()

        logger.info(
            "Searching similar vectors",
            query_length=len(query),
            k=k,
            has_filters=filters is not None,
            min_similarity=min_similarity,
            trace_id=trace_id,
        )

        try:
            # Step 1: Generate query embedding using LangChain
            logger.debug("Generating query embedding", trace_id=trace_id)
            query_embedding = await self.embeddings.embed_text(query)

            # Step 2: Convert embedding to PostgreSQL vector format
            query_embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Step 3: Build SQL query with vector similarity
            distance_operator = self._get_distance_operator()

            # Build WHERE clause for filters
            where_clauses = []
            params = [query_embedding_str]  # $1
            param_idx = 2

            if filters:
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ${param_idx}")
                    params.append(value)
                    param_idx += 1

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            # Step 4: Execute similarity search query
            sql = f"""
                SELECT
                    id::text,
                    content,
                    embedding,
                    node_type,
                    schema_name,
                    table_name,
                    column_name,
                    constraint_name,
                    metadata,
                    embedding {distance_operator} $1 AS distance
                FROM {self.config.table_name}
                {where_sql}
                ORDER BY distance ASC
                LIMIT {k};
            """

            async with self.db.acquire_connection() as conn:
                rows = await conn.fetch(sql, *params)

            # Step 5: Convert results to VectorSearchResult objects
            search_results: List[VectorSearchResult] = []

            for row in rows:
                # Convert distance to similarity score
                distance = float(row["distance"])
                similarity = self._distance_to_similarity(distance)

                # Apply minimum similarity threshold if specified
                if min_similarity is not None and similarity < min_similarity:
                    continue

                # Parse metadata JSONB
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                # Add structured fields to metadata if not present
                if "node_type" not in metadata and row["node_type"]:
                    metadata["node_type"] = row["node_type"]
                if "schema_name" not in metadata and row["schema_name"]:
                    metadata["schema_name"] = row["schema_name"]
                if "table_name" not in metadata and row["table_name"]:
                    metadata["table_name"] = row["table_name"]
                if "column_name" not in metadata and row["column_name"]:
                    metadata["column_name"] = row["column_name"]
                if "constraint_name" not in metadata and row["constraint_name"]:
                    metadata["constraint_name"] = row["constraint_name"]

                search_results.append(
                    VectorSearchResult(
                        content=row["content"],
                        metadata=metadata,
                        similarity_score=similarity
                    )
                )

            logger.info(
                "Similar vectors found",
                result_count=len(search_results),
                trace_id=trace_id,
            )

            return search_results

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to search similar vectors",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to search similar vectors: {e}") from e

    async def drop_collection(self, drop_table: bool = True) -> Dict[str, Any]:
        """
        Drop vector collection.

        Args:
            drop_table: If True, drop entire table. If False, only drop HNSW index.

        Returns:
            Dictionary with cleanup details

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
            async with self.db.acquire_connection(read_only=False) as conn:
                if drop_table:
                    # Drop entire table (cascade drops all indexes and triggers)
                    await conn.execute(f"DROP TABLE IF EXISTS {self.config.table_name} CASCADE;")
                    logger.info(
                        "Table dropped successfully",
                        table_name=self.config.table_name,
                        trace_id=trace_id,
                    )
                    result = {
                        "action": "dropped_table",
                        "table_name": self.config.table_name,
                        "indexes_dropped": "all (cascaded)",
                    }
                    # Reset setup flag
                    self._setup_done = False
                else:
                    # Drop only HNSW index
                    index_name = f"{self.config.table_name}_hnsw_idx"
                    await conn.execute(f"DROP INDEX IF EXISTS {index_name};")
                    logger.info(
                        "HNSW index dropped successfully",
                        index_name=index_name,
                        trace_id=trace_id,
                    )
                    result = {
                        "action": "dropped_index",
                        "index_name": index_name,
                        "table_name": self.config.table_name,
                    }

            logger.info(
                "Vector collection dropped",
                action=result.get("action"),
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

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection statistics

        Raises:
            VectorStoreError: If query fails
        """
        trace_id = current_trace_id()

        # Ensure setup is done
        await self.ensure_setup()

        logger.info("Fetching vector collection stats", trace_id=trace_id)

        try:
            async with self.db.acquire_connection() as conn:
                # Get total document count
                total_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.config.table_name};"
                )

                # Get counts by node type
                node_type_counts = await conn.fetch(
                    f"""
                    SELECT node_type, COUNT(*) as count
                    FROM {self.config.table_name}
                    GROUP BY node_type;
                    """
                )

                # Get counts by schema
                schema_counts = await conn.fetch(
                    f"""
                    SELECT schema_name, COUNT(*) as count
                    FROM {self.config.table_name}
                    WHERE schema_name IS NOT NULL
                    GROUP BY schema_name;
                    """
                )

                stats = {
                    "total_documents": total_count,
                    "node_type_counts": {row["node_type"]: row["count"] for row in node_type_counts},
                    "schema_counts": {row["schema_name"]: row["count"] for row in schema_counts},
                }

            logger.info(
                "Vector collection stats fetched",
                total_documents=total_count,
                trace_id=trace_id,
            )

            return stats

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(
                "Failed to fetch vector collection stats",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to fetch collection stats: {e}") from e

    # -------------------------
    # Private setup methods
    # -------------------------

    async def _ensure_pgvector_extension(self) -> None:
        """Enable pgvector extension if not already enabled."""
        async with self.db.acquire_connection(read_only=False) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")

    async def _ensure_tables_exist(self) -> None:
        """Create vector store tables if they don't exist."""
        async with self.db.acquire_connection(read_only=False) as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = $1
                );
                """,
                self.config.table_name,
            )

            if not table_exists:
                logger.info("Creating vector store tables", table_name=self.config.table_name)

                # Create table with schema_embeddings structure
                create_table_sql = f"""
                CREATE TABLE {self.config.table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    embedding vector({self.embeddings.config.embedding_dimension}) NOT NULL,
                    node_type TEXT NOT NULL CHECK (node_type IN ('table', 'column', 'relationship')),
                    schema_name TEXT,
                    table_name TEXT,
                    column_name TEXT,
                    constraint_name TEXT,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
                await conn.execute(create_table_sql)

                # Create updated_at trigger
                trigger_sql = f"""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';

                CREATE TRIGGER update_{self.config.table_name}_updated_at
                    BEFORE UPDATE ON {self.config.table_name}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
                """
                await conn.execute(trigger_sql)

                logger.info("Vector store tables created successfully")
            else:
                logger.info("Vector store tables already exist")

    async def _ensure_hnsw_index(self) -> None:
        """Create HNSW index if it doesn't exist."""
        async with self.db.acquire_connection(read_only=False) as conn:
            index_name = f"{self.config.table_name}_hnsw_idx"

            # Check if index exists
            index_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = $1
                );
                """,
                index_name,
            )

            if not index_exists:
                logger.info("Creating HNSW index", index_name=index_name)

                # Map distance strategy to PostgreSQL operator class
                ops_class = PGVECTOR_OPS_MAP[self.config.distance_strategy]

                hnsw_index_sql = f"""
                CREATE INDEX {index_name}
                ON {self.config.table_name}
                USING hnsw (embedding {ops_class})
                WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction});
                """
                await conn.execute(hnsw_index_sql)

                logger.info("HNSW index created successfully")
            else:
                logger.info("HNSW index already exists")

    async def _ensure_metadata_indexes(self) -> None:
        """Create metadata indexes if they don't exist."""
        async with self.db.acquire_connection(read_only=False) as conn:
            # Define indexes to create
            index_definitions = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_node_type ON {self.config.table_name}(node_type);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_schema_name ON {self.config.table_name}(schema_name);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_table_name ON {self.config.table_name}(table_name);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_metadata ON {self.config.table_name} USING gin(metadata);",
            ]

            logger.info("Creating metadata indexes")
            for index_sql in index_definitions:
                await conn.execute(index_sql)

            logger.info("Metadata indexes created successfully")

    # -------------------------
    # Private helper methods
    # -------------------------

    def _get_distance_operator(self) -> str:
        """
        Get the pgvector distance operator for the configured strategy.

        Returns:
            SQL operator string (e.g., '<->', '<=>', '<#>')
        """
        if self.config.distance_strategy == DistanceStrategy.COSINE:
            return "<=>"  # Cosine distance
        elif self.config.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "<->"  # L2 distance
        else:  # MAX_INNER_PRODUCT
            return "<#>"  # Negative inner product

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score.

        For cosine distance: similarity = 1 - distance
        For L2 distance: similarity = 1 / (1 + distance)
        For inner product: similarity = -distance (pgvector returns negative)
        """
        if self.config.distance_strategy == DistanceStrategy.COSINE:
            return 1.0 - distance
        elif self.config.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return 1.0 / (1.0 + distance)
        else:  # MAX_INNER_PRODUCT
            return -distance  # pgvector uses negative inner product for ordering
