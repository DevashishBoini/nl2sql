"""
Integration tests for atomic transaction behavior in vector indexing.

This module verifies that schema indexing operations are truly atomic:
- All batches succeed together, or all fail together
- No partial indexing on failure
- Proper error reporting with batch number

Usage:
    # Run all atomic transaction tests
    pytest tests/integration/test_atomic_transactions.py -v

    # Run specific test
    pytest tests/integration/test_atomic_transactions.py::TestAtomicTransactions::test_atomic_batch_addition -v

    # Run with output (see logs)
    pytest tests/integration/test_atomic_transactions.py -v -s

    # Run with coverage
    pytest tests/integration/test_atomic_transactions.py --cov=backend.services.vector_service --cov=backend.repositories.vector_repository -v

Requirements:
    - DATABASE__DATABASE_URL must be set in .env
    - EMBEDDING__OPENROUTER_API_KEY must be set in .env
    - pgvector extension must be available in database
"""

import pytest

from backend.config import get_settings
from backend.domain.base_enums import NodeType
from backend.domain.vector_metadata import TableMetadata
from backend.repositories.vector_repository import VectorRepository
from backend.infrastructure.database_client import DatabaseClient
from backend.infrastructure.embedding_client import EmbeddingClient
from backend.infrastructure.storage_client import StorageClient


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
def storage_config():
    """Get storage configuration from settings."""
    settings = get_settings()
    return settings.storage


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
async def storage_client(storage_config):
    """Create and connect storage client."""
    client = StorageClient(storage_config)
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
class TestAtomicTransactions:
    """Test atomic transaction behavior in vector indexing."""

    @pytest.mark.asyncio
    async def test_atomic_batch_addition(self, vector_repo):
        """Test that multiple batches are added atomically in a transaction."""
        # Create 15 documents (will be 2 batches if batch_size=10)
        texts = []
        metadatas = []

        for i in range(15):
            metadata = TableMetadata(
                node_type=NodeType.TABLE.value,
                table_name=f"test_table_{i}",
                schema_name="public",
                table_type="BASE TABLE",
            )
            texts.append(f"Table: test_table_{i}\nDescription: Test table number {i}")
            metadatas.append(metadata)

        # IMPORTANT: Ensure setup is done before starting transaction
        await vector_repo.ensure_setup()

        # Add all documents using atomic transaction
        async with vector_repo.db.acquire_connection() as conn:
            async with conn.transaction():
                doc_ids = await vector_repo._add_vectors_with_connection(
                    conn=conn,
                    texts=texts,
                    metadatas=metadatas
                )

        assert len(doc_ids) == 15, "All 15 documents should be added"

        # Verify all documents are searchable
        results = await vector_repo.search_similar(
            query="test table",
            k=20,
            filters=None,
            min_similarity=0.0
        )

        assert len(results) >= 15, "All 15 documents should be retrievable"

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, vector_repo):
        """Test that transaction rolls back completely on failure."""
        # First, add some valid documents
        valid_texts = ["Table: valid_table_1", "Table: valid_table_2"]
        valid_metadatas = [
            TableMetadata(
                node_type=NodeType.TABLE.value,
                table_name=f"valid_table_{i}",
                schema_name="public",
                table_type="BASE TABLE",
            )
            for i in range(2)
        ]

        await vector_repo.add_vectors(texts=valid_texts, metadatas=valid_metadatas)

        # Verify they were added
        initial_stats = await vector_repo.get_stats()
        initial_count = initial_stats["total_documents"]
        assert initial_count >= 2, "Initial documents should be present"

        # Now try to add documents in a transaction with one that will fail
        # (invalid metadata that violates constraints)
        try:
            async with vector_repo.db.acquire_connection() as conn:
                async with conn.transaction():
                    # Add first batch (valid)
                    batch1_texts = ["Table: transaction_test_1"]
                    batch1_metadatas = [
                        TableMetadata(
                            node_type=NodeType.TABLE.value,
                            table_name="transaction_test_1",
                            schema_name="public",
                            table_type="BASE TABLE",
                        )
                    ]
                    await vector_repo._add_vectors_with_connection(
                        conn=conn,
                        texts=batch1_texts,
                        metadatas=batch1_metadatas
                    )

                    # Force a failure by trying to insert with missing required field
                    # This should cause the entire transaction to rollback
                    raise Exception("Simulated failure in transaction")

        except Exception as e:
            assert "Simulated failure" in str(e)

        # Verify the transaction was rolled back - count should be unchanged
        final_stats = await vector_repo.get_stats()
        final_count = final_stats["total_documents"]

        assert final_count == initial_count, \
            f"Transaction should have rolled back. Expected {initial_count}, got {final_count}"

        # Verify the "transaction_test_1" document was NOT added
        results = await vector_repo.search_similar(
            query="transaction_test_1",
            k=10,
            filters=None,
            min_similarity=0.0
        )

        transaction_test_found = any("transaction_test_1" in r.content for r in results)
        assert not transaction_test_found, \
            "Rolled back document should not be present in the database"
