#!/usr/bin/env python3
"""
Minimal Supabase Storage connection test.

This script verifies basic connectivity to your Supabase Storage.

Usage:
    poetry run python tests/integration/test_storage_connection.py
"""

import asyncio
import sys
import httpx

from backend.config import get_settings
from backend.infrastructure.storage_client import StorageClient


async def test_storage_connection():
    """Test Supabase Storage connection."""
    print("Testing Supabase Storage connection...")
    print()

    # Load configuration
    try:
        settings = get_settings()
        storage_config = settings.storage
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Show connection info
    print(f"Supabase URL: {storage_config.supabase_url}")
    print(f"Default bucket: {storage_config.default_bucket}")
    print()

    client = None

    try:
        # Create client
        print("Creating storage client...")
        client = StorageClient(storage_config)
        print("✓ Client created")
        print()

        # Connect
        print("Connecting to Supabase Storage...")
        await client.connect()
        print("✓ Connected successfully")
        print()

        # Test connection with a simple ping
        print("Testing connection...")
        if client.is_connected():
            print("✓ Connection active")
        else:
            print("❌ Connection check failed")
            sys.exit(1)
        print()

        # Success
        print("=" * 50)
        print("✅ STORAGE CONNECTION WORKING!")
        print("=" * 50)

    except httpx.ConnectTimeout as e:
        print("❌ CONNECTION FAILED: Connection timeout")
        print(f"Error: {e}")
        sys.exit(1)

    except httpx.ConnectError as e:
        print("❌ CONNECTION FAILED: Cannot connect to host")
        print(f"Error: {e}")
        sys.exit(1)

    except httpx.HTTPStatusError as e:
        print("❌ CONNECTION FAILED: HTTP error")
        print(f"Status: {e.response.status_code}")
        print(f"Error: {e}")
        sys.exit(1)

    except Exception as e:
        print("❌ CONNECTION FAILED")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)

    finally:
        if client and client.is_connected():
            await client.close()
            print()
            print("Connection closed")


if __name__ == "__main__":
    asyncio.run(test_storage_connection())
