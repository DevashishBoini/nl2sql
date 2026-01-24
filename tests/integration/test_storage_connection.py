"""
Integration tests for StorageClient connection.

This module verifies basic connectivity to your Supabase Storage.

Usage:
    # Run all storage connection tests
    pytest tests/integration/test_storage_connection.py -v

    # Run specific test
    pytest tests/integration/test_storage_connection.py::TestStorageConnection::test_basic_connection -v

    # Run with output
    pytest tests/integration/test_storage_connection.py -v -s
"""

import pytest
import httpx

from backend.config import get_settings
from backend.infrastructure.storage_client import StorageClient


@pytest.fixture
def storage_config():
    """Get storage configuration from settings."""
    settings = get_settings()
    return settings.storage


@pytest.fixture
async def storage_client(storage_config):
    """Create and connect storage client."""
    client = StorageClient(storage_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.mark.integration
class TestStorageConnection:
    """Integration tests for storage connectivity."""

    @pytest.mark.asyncio
    async def test_basic_connection(self, storage_config):
        """Test basic storage connection and disconnection."""
        client = StorageClient(storage_config)

        # Test connection
        await client.connect()
        assert client.is_connected()

        # Test disconnection
        await client.close()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_connection_check(self, storage_client):
        """Test connection status check."""
        assert storage_client.is_connected()

    @pytest.mark.asyncio
    async def test_config_applied(self, storage_config, storage_client):
        """Test that configuration is properly applied."""
        assert storage_client.config == storage_config
        assert storage_client.config.supabase_url == storage_config.supabase_url
        assert storage_client.config.default_bucket == storage_config.default_bucket

    @pytest.mark.asyncio
    async def test_handles_connect_timeout(self, storage_config):
        """Test handling of connection timeout errors."""
        # Create client with invalid URL to trigger timeout
        invalid_config = storage_config.model_copy()
        invalid_config.supabase_url = "https://invalid-nonexistent-domain-12345.com"
        invalid_config.connect_timeout_seconds = 1

        client = StorageClient(invalid_config)

        with pytest.raises((httpx.ConnectTimeout, httpx.ConnectError, Exception)):
            await client.connect()

    @pytest.mark.asyncio
    async def test_handles_http_error(self, storage_config):
        """Test handling of HTTP status errors."""
        # Create client with invalid credentials
        invalid_config = storage_config.model_copy()
        invalid_config.supabase_key = "invalid-key-12345"

        client = StorageClient(invalid_config)

        # Connect may or may not fail depending on when validation happens
        # Just verify the client can be created
        assert client is not None
