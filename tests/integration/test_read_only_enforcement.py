"""
Integration tests for read-only enforcement at database connection level.

This module verifies that the read-only flag properly enforces read-only
transactions using PostgreSQL's SET TRANSACTION READ ONLY.

Usage:
    # Run all read-only enforcement tests
    pytest tests/integration/test_read_only_enforcement.py -v

    # Run with output (see logs)
    pytest tests/integration/test_read_only_enforcement.py -v -s

Requirements:
    - DATABASE__DATABASE_URL must be set in .env
    - DATABASE__ENFORCE_READ_ONLY_DEFAULT=true (default)
"""

import pytest
import asyncpg

from backend.config import get_settings
from backend.infrastructure.database_client import DatabaseClient


@pytest.fixture
def database_config():
    """Get database configuration from settings."""
    settings = get_settings()
    return settings.database


@pytest.fixture
async def db_client(database_config):
    """Create and connect database client."""
    client = DatabaseClient(database_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.mark.integration
class TestReadOnlyEnforcement:
    """Test read-only enforcement at connection level."""

    @pytest.mark.asyncio
    async def test_read_operations_work_in_read_only_mode(self, db_client):
        """Test that read operations work when read_only=True (default)."""
        # Use default read_only behavior (should be True from config)
        async with db_client.acquire_connection() as conn:
            # Simple read query should work
            result = await conn.fetchval("SELECT 1")
            assert result == 1

            # Query from actual table should work
            tables = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                LIMIT 1
                """
            )
            assert tables is not None

    @pytest.mark.asyncio
    async def test_write_operations_blocked_in_read_only_mode(self, db_client):
        """Test that write operations fail when read_only=True."""
        # Explicitly set read_only=True
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            async with db_client.acquire_connection(read_only=True) as conn:
                # Try to create a temporary table (should fail)
                await conn.execute("CREATE TEMP TABLE test_read_only (id INT);")

    @pytest.mark.asyncio
    async def test_write_operations_work_with_read_only_false(self, db_client):
        """Test that write operations work when read_only=False."""
        # Explicitly allow writes
        async with db_client.acquire_connection(read_only=False) as conn:
            # Create temporary table (should succeed)
            await conn.execute("CREATE TEMP TABLE test_write_allowed (id INT);")

            # Insert data (should succeed)
            await conn.execute("INSERT INTO test_write_allowed VALUES (1), (2), (3);")

            # Verify data
            count = await conn.fetchval("SELECT COUNT(*) FROM test_write_allowed;")
            assert count == 3

            # Cleanup (temp table auto-drops at end of session)

    @pytest.mark.asyncio
    async def test_config_default_enforces_read_only(self, db_client):
        """Test that config default (enforce_read_only_default=True) is applied."""
        # Don't specify read_only parameter, should use config default
        if db_client.config.enforce_read_only_default:
            # Should fail because default is read-only
            with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
                async with db_client.acquire_connection() as conn:
                    await conn.execute("CREATE TEMP TABLE test_default (id INT);")
        else:
            # Should succeed if default is not read-only
            async with db_client.acquire_connection() as conn:
                await conn.execute("CREATE TEMP TABLE test_default (id INT);")
                # Cleanup
                await conn.execute("DROP TABLE test_default;")

    @pytest.mark.asyncio
    async def test_transaction_respects_read_only_flag(self, db_client):
        """Test that explicit transactions respect the read_only flag."""
        # Read-only transaction should block writes
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            async with db_client.acquire_connection(read_only=True) as conn:
                async with conn.transaction():
                    await conn.execute("CREATE TEMP TABLE test_txn_readonly (id INT);")

        # Write-enabled transaction should allow writes
        async with db_client.acquire_connection(read_only=False) as conn:
            async with conn.transaction():
                await conn.execute("CREATE TEMP TABLE test_txn_write (id INT);")
                await conn.execute("INSERT INTO test_txn_write VALUES (1);")
                count = await conn.fetchval("SELECT COUNT(*) FROM test_txn_write;")
                assert count == 1

    @pytest.mark.asyncio
    async def test_execute_query_with_default_read_only(self, db_client):
        """Test that execute_query uses default read-only behavior."""
        # Read operations should work
        result = await db_client.execute_query("SELECT 1 AS value")
        assert len(result) == 1
        assert result[0]["value"] == 1

        # Query from information_schema should work
        tables = await db_client.execute_query(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            LIMIT 5
            """
        )
        assert isinstance(tables, list)
