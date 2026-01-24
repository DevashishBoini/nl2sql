"""
Integration tests for DatabaseClient connection.

This module verifies basic connectivity to your Supabase database.

Usage:
    # Run all database connection tests
    pytest tests/integration/test_db_connection.py -v

    # Run specific test
    pytest tests/integration/test_db_connection.py::TestDatabaseConnection::test_basic_connection -v

    # Run with output
    pytest tests/integration/test_db_connection.py -v -s
"""

import pytest

from backend.config import get_settings
from backend.infrastructure.database_client import DatabaseClient


@pytest.fixture
def db_config():
    """Get database configuration from settings."""
    settings = get_settings()
    return settings.database


@pytest.fixture
async def db_client(db_config):
    """Create and connect database client."""
    client = DatabaseClient(db_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.mark.integration
class TestDatabaseConnection:
    """Integration tests for database connectivity."""

    @pytest.mark.asyncio
    async def test_basic_connection(self, db_config):
        """Test basic database connection and disconnection."""
        client = DatabaseClient(db_config)

        # Test connection
        await client.connect()
        assert client.is_connected()

        # Test disconnection
        await client.close()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_simple_query(self, db_client):
        """Test simple query execution."""
        result = await db_client.execute_scalar("SELECT 1")
        assert result == 1

    @pytest.mark.asyncio
    async def test_connection_check(self, db_client):
        """Test connection status check."""
        assert db_client.is_connected()

    @pytest.mark.asyncio
    async def test_config_applied(self, db_config, db_client):
        """Test that configuration is properly applied."""
        assert db_client.config == db_config
        assert db_client.config.default_schema == db_config.default_schema
        assert db_client.config.connection_pool_max_size == db_config.connection_pool_max_size
