"""
Integration tests for VectorRepository.

This module verifies:
- Basic connectivity to vector store (Supabase with pgvector)
- Extension creation
- Document indexing
- Similarity search
- Collection management

Usage:
    # Run all vector store tests
    pytest tests/integration/test_vector_store_connection.py -v

    # Run specific test
    pytest tests/integration/test_vector_store_connection.py::TestVectorStoreConnection::test_basic_setup -v

    # Run with output (see logs)
    pytest tests/integration/test_vector_store_connection.py -v -s

    # Run with coverage
    pytest tests/integration/test_vector_store_connection.py --cov=backend.repositories.vector_repository -v

    # Run all integration tests (vector store + atomic transactions)
    pytest tests/integration/ -v

Requirements:
    - DATABASE__DATABASE_URL must be set in .env
    - EMBEDDING__OPENROUTER_API_KEY must be set in .env
    - pgvector extension must be available in database
"""

import pytest

from backend.config import get_settings
from backend.domain.base_enums import NodeType
from backend.domain.vector_metadata import TableMetadata, ColumnMetadata
from backend.repositories.vector_repository import VectorRepository
from backend.infrastructure.database_client import DatabaseClient
from backend.infrastructure.embedding_client import EmbeddingClient


@pytest.fixture
def vector_config():
    """Get vector store configuration from settings."""
    settings = get_settings()
    return settings.vector_store


@pytest.fixture
def database_config():
    """Get database configuration from settings."""
    settings = get_settings()
    return settings.database


@pytest.fixture
def embedding_config():
    """Get embedding configuration from settings."""
    settings = get_settings()
    return settings.embedding


@pytest.fixture
async def db_client(database_config):
    """Create and connect database client."""
    client = DatabaseClient(database_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.fixture
async def embedding_client(embedding_config):
    """Create and connect embedding client."""
    client = EmbeddingClient(embedding_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.fixture
async def vector_repo(vector_config, db_client, embedding_client):
    """Create vector repository."""
    repo = VectorRepository(
        db_client=db_client,
        embedding_client=embedding_client,
        config=vector_config
    )
    yield repo
    # Cleanup: drop test collection after tests
    try:
        await repo.drop_collection(drop_table=True)
    except Exception:
        pass  # Ignore cleanup errors


@pytest.mark.integration
class TestVectorStoreConnection:
    """Integration tests for vector store connectivity."""

    @pytest.mark.asyncio
    async def test_basic_setup(self, vector_config, db_client, embedding_client):
        """Test basic vector repository setup."""
        repo = VectorRepository(
            db_client=db_client,
            embedding_client=embedding_client,
            config=vector_config
        )

        # Test setup (idempotent)
        await repo.ensure_setup()
        assert repo._setup_done is True, "Setup should be marked as done"

        # Try to add a simple document to verify setup works
        try:
            metadata = TableMetadata(
                node_type=NodeType.TABLE.value,
                table_name="test_setup",
                schema_name="public",
                table_type="BASE TABLE",
            )
            doc_ids = await repo.add_vectors(
                texts=["test document"],
                metadatas=[metadata]
            )
            assert len(doc_ids) == 1
            assert doc_ids[0] is not None
        except Exception as e:
            pytest.fail(f"Vector repository setup not working: {e}")

        # Cleanup
        await repo.drop_collection(drop_table=True)

    @pytest.mark.asyncio
    async def test_pgvector_extension_exists(self, vector_repo):
        """Test that pgvector extension is created."""
        # Try to add a simple document to verify extension works
        try:
            metadata = TableMetadata(
                node_type=NodeType.TABLE.value,
                table_name="test_extension",
                schema_name="public",
                table_type="BASE TABLE",
            )
            doc_ids = await vector_repo.add_vectors(
                texts=["test document"],
                metadatas=[metadata]
            )
            assert len(doc_ids) == 1
            assert doc_ids[0] is not None
        except Exception as e:
            pytest.fail(f"pgvector extension not working: {e}")

    @pytest.mark.asyncio
    async def test_add_and_search_documents(self, vector_repo):
        """Test adding documents and performing similarity search."""
        # Sample schema-like documents
        table_metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="customers",
            schema_name="public",
            table_type="BASE TABLE",
        )

        column_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="customers",
            column_name="email",
            schema_name="public",
            data_type="varchar",
            is_primary_key=False,
            is_foreign_key=False,
            is_unique=True,
            is_nullable=False,
        )

        texts = [
            "Table: customers\nDescription: Customer information",
            "Column: customers.email\nDescription: Customer email address",
        ]

        metadatas = [table_metadata, column_metadata]

        # Add documents
        doc_ids = await vector_repo.add_vectors(
            texts=texts,
            metadatas=metadatas
        )

        assert len(doc_ids) == 2, "Should return 2 document IDs"
        assert all(doc_id is not None for doc_id in doc_ids), "All IDs should be non-null"

        # Search for similar documents
        results = await vector_repo.search_similar(
            query="customer email",
            k=2,
            filters=None,
            min_similarity=0.0
        )

        assert len(results) > 0, "Should return search results"
        assert results[0].content is not None, "Result should have content"
        assert results[0].metadata is not None, "Result should have metadata"
        assert results[0].similarity_score is not None, "Result should have similarity score"
        assert 0.0 <= results[0].similarity_score <= 1.0, "Similarity should be between 0 and 1"

        # The email column should be the most relevant result
        assert "email" in results[0].content.lower(), "Top result should be about email"

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, vector_repo):
        """Test similarity search with metadata filtering."""
        # Add multiple documents with different node types
        table_metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="users",
            schema_name="public",
            table_type="BASE TABLE",
        )

        column_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="users",
            column_name="username",
            schema_name="public",
            data_type="varchar",
            is_primary_key=False,
            is_foreign_key=False,
            is_unique=True,
            is_nullable=False,
        )

        texts = [
            "Table: users",
            "Column: users.username",
        ]

        metadatas = [table_metadata, column_metadata]

        await vector_repo.add_vectors(texts=texts, metadatas=metadatas)

        # Search with filter for only tables
        results = await vector_repo.search_similar(
            query="users",
            k=10,
            filters={"node_type": NodeType.TABLE.value},
            min_similarity=0.0
        )

        # Should only return table results
        assert len(results) > 0, "Should find table results"
        for result in results:
            assert result.metadata.get("node_type") == NodeType.TABLE.value, \
                "All results should be tables when filtered"

    @pytest.mark.asyncio
    async def test_search_with_min_similarity(self, vector_repo):
        """Test similarity search with minimum similarity threshold on specific column."""
        # Create a table with 3 columns with detailed descriptions
        table_metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="users",
            schema_name="public",
            table_type="BASE TABLE",
        )

        email_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="users",
            column_name="email",
            schema_name="public",
            data_type="varchar",
            is_primary_key=False,
            is_foreign_key=False,
            is_unique=True,
            is_nullable=False,
        )

        phone_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="users",
            column_name="phone_number",
            schema_name="public",
            data_type="varchar",
            is_primary_key=False,
            is_foreign_key=False,
            is_unique=False,
            is_nullable=True,
        )

        created_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="users",
            column_name="created_at",
            schema_name="public",
            data_type="timestamp",
            is_primary_key=False,
            is_foreign_key=False,
            is_unique=False,
            is_nullable=False,
        )

        # Add documents with rich descriptions
        texts = [
            "Table: users\nDescription: User accounts and authentication information",
            "Column: users.email\nDescription: User email address for login and communication. Must be unique and valid format.",
            "Column: users.phone_number\nDescription: User mobile or landline phone number for SMS notifications and two-factor authentication.",
            "Column: users.created_at\nDescription: Timestamp when the user account was created in the system.",
        ]

        metadatas = [table_metadata, email_metadata, phone_metadata, created_metadata]

        # Index all documents
        doc_ids = await vector_repo.add_vectors(texts=texts, metadatas=metadatas)
        assert len(doc_ids) == 4, "Should add 4 documents"

        # Test 1: Search for email-related content with low threshold (semantic search typically 0.4-0.6)
        email_results = await vector_repo.search_similar(
            query="email address for user login",
            k=10,
            filters=None,
            min_similarity=0.35  # Realistic threshold for cosine similarity
        )

        assert len(email_results) > 0, "Should find email-related results with 0.35 threshold"
        assert len(email_results) >= 2, "Should find at least 2 relevant results (email column + table)"

        # Verify the top result is the email column
        top_result = email_results[0]
        assert "email" in top_result.content.lower(), "Top result should be about email"
        assert top_result.metadata.get("column_name") == "email", \
            f"Top result should be the email column, got: {top_result.metadata.get('column_name')}"

        # Verify similarity score is reasonable (0.4-0.7 range for good semantic matches)
        assert 0.4 <= top_result.similarity_score <= 1.0, \
            f"Email column similarity should be in reasonable range, got {top_result.similarity_score:.4f}"

        # Test 2: Search for phone-related content
        phone_results = await vector_repo.search_similar(
            query="phone number for SMS and authentication",
            k=10,
            filters=None,
            min_similarity=0.30
        )

        assert len(phone_results) > 0, "Should find phone-related results"
        # Verify phone column is in top results
        phone_found = any("phone" in r.content.lower() for r in phone_results[:3])
        assert phone_found, "Phone column should be in top 3 results"

        # Test 3: Search with higher threshold (should return fewer results)
        high_threshold_results = await vector_repo.search_similar(
            query="email address for user login",
            k=10,
            filters=None,
            min_similarity=0.50  # Higher threshold - should filter out less relevant results
        )

        # With high threshold, we should get fewer results
        assert len(high_threshold_results) <= len(email_results), \
            "High threshold (0.50) should return same or fewer results than low threshold (0.35)"
        assert len(high_threshold_results) >= 1, "Should still find the best match (email column)"

        # Top result should still be email column
        assert high_threshold_results[0].metadata.get("column_name") == "email", \
            "With higher threshold, top result should still be email column"

        # Test 4: Completely unrelated query with moderate threshold
        unrelated_results = await vector_repo.search_similar(
            query="quantum physics nuclear fusion reactor core temperature dynamics",
            k=10,
            filters=None,
            min_similarity=0.40
        )

        # Unrelated query should return very few or no results with moderate threshold
        # (User table/columns are not related to physics)
        assert len(unrelated_results) <= 2, \
            "Unrelated query should return very few results with 0.40 threshold"

    @pytest.mark.asyncio
    async def test_collection_stats(self, vector_repo):
        """Test getting collection statistics."""
        # Add some test documents
        table_metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="orders",
            schema_name="public",
            table_type="BASE TABLE",
        )

        column_metadata = ColumnMetadata(
            node_type=NodeType.COLUMN.value,
            table_name="orders",
            column_name="order_id",
            schema_name="public",
            data_type="integer",
            is_primary_key=True,
            is_foreign_key=False,
            is_unique=True,
            is_nullable=False,
        )

        await vector_repo.add_vectors(
            texts=["Table: orders", "Column: orders.order_id"],
            metadatas=[table_metadata, column_metadata]
        )

        # Get stats
        stats = await vector_repo.get_stats()

        assert "total_documents" in stats, "Stats should include total_documents"
        assert "node_type_counts" in stats, "Stats should include node_type_counts"
        assert "schema_counts" in stats, "Stats should include schema_counts"

        assert stats["total_documents"] >= 2, "Should have at least 2 documents"
        assert NodeType.TABLE.value in stats["node_type_counts"], "Should have table count"
        assert NodeType.COLUMN.value in stats["node_type_counts"], "Should have column count"
        assert "public" in stats["schema_counts"], "Should have public schema count"

    @pytest.mark.asyncio
    async def test_drop_collection(self, vector_config, db_client, embedding_client):
        """Test dropping vector collection."""
        repo = VectorRepository(
            db_client=db_client,
            embedding_client=embedding_client,
            config=vector_config
        )

        # Add some data
        metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="test_table",
            schema_name="public",
            table_type="BASE TABLE",
        )

        await repo.add_vectors(
            texts=["Test document"],
            metadatas=[metadata]
        )

        # Drop collection
        result = await repo.drop_collection(drop_table=True)

        assert "action" in result, "Result should include action"
        assert result["action"] == "dropped_table", "Action should be dropped_table"
        assert repo._setup_done is False, "Setup flag should be reset after drop"

        # Add data again (should recreate table automatically)
        doc_ids = await repo.add_vectors(
            texts=["Test document 2"],
            metadatas=[metadata]
        )
        assert len(doc_ids) == 1, "Should be able to add data after drop (table recreated)"
        assert repo._setup_done is True, "Setup should be done after adding data"

        # Final cleanup
        await repo.drop_collection(drop_table=True)

    @pytest.mark.asyncio
    async def test_batch_document_insertion(self, vector_repo):
        """Test adding multiple documents in batch."""
        # Create 10 test documents
        texts = []
        metadatas = []

        for i in range(10):
            metadata = TableMetadata(
                node_type=NodeType.TABLE.value,
                table_name=f"test_table_{i}",
                schema_name="public",
                table_type="BASE TABLE",
            )
            texts.append(f"Table: test_table_{i}")
            metadatas.append(metadata)

        # Add all at once
        doc_ids = await vector_repo.add_vectors(
            texts=texts,
            metadatas=metadatas
        )

        assert len(doc_ids) == 10, "Should return 10 document IDs"
        assert len(set(doc_ids)) == 10, "All IDs should be unique"

        # Verify all documents are searchable
        results = await vector_repo.search_similar(
            query="test_table",
            k=10,
            filters=None,
            min_similarity=0.0
        )

        assert len(results) >= 10, "Should find all inserted documents"


@pytest.mark.integration
class TestVectorStoreEdgeCases:
    """Edge case tests for vector store."""

    @pytest.mark.asyncio
    async def test_empty_search(self, vector_repo):
        """Test search on empty collection."""
        # Drop to ensure empty (will be recreated on next operation)
        await vector_repo.drop_collection(drop_table=True)

        # Search should return empty results
        results = await vector_repo.search_similar(
            query="anything",
            k=5,
            filters=None,
            min_similarity=0.0
        )

        assert len(results) == 0, "Empty collection should return no results"

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, vector_repo):
        """Test handling of special characters in document content."""
        metadata = TableMetadata(
            node_type=NodeType.TABLE.value,
            table_name="test_table",
            schema_name="public",
            table_type="BASE TABLE",
        )

        # Content with special characters
        special_text = "Table: test\nSpecial chars: @#$%^&*(){}[]|\\\"'<>?/~`"

        doc_ids = await vector_repo.add_vectors(
            texts=[special_text],
            metadatas=[metadata]
        )

        assert len(doc_ids) == 1, "Should handle special characters"

        # Should be able to search and retrieve
        results = await vector_repo.search_similar(
            query="special chars",
            k=1,
            filters=None,
            min_similarity=0.0
        )

        assert len(results) > 0, "Should find document with special characters"
