#!/usr/bin/env python3
"""
Minimal database connection test using DatabaseClient.

This script verifies basic connectivity to your Supabase database.

Usage:
    poetry run python tests/integration/test_db_connection.py
"""

import asyncio
import sys

from backend.config import get_settings
from backend.infrastructure.database_client import DatabaseClient


async def test_connection():
    """Test database connection with DatabaseClient."""
    print("Testing database connection...")
    print()

    # Load configuration
    try:
        settings = get_settings()
        db_config = settings.database
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Show connection info
    print(f"Default schema: {db_config.default_schema}")
    print(f"Pool size: {db_config.connection_pool_max_size}")
    print(f"Application name: {db_config.application_name}")
    print()

    client = None

    try:
        # Create client
        print("Creating database client...")
        client = DatabaseClient(db_config)
        print("✓ Client created")
        print()

        # Connect
        print("Connecting to database...")
        await client.connect()
        print("✓ Connected successfully")
        print()

        # Test connection with a simple query
        print("Testing query...")
        if client.is_connected():
            result = await client.execute_scalar("SELECT 1")
            print(f"✓ Query successful: {result}")
        else:
            print("❌ Connection check failed")
            sys.exit(1)
        print()

        # Success
        print("=" * 50)
        print("✅ DATABASE CONNECTION WORKING!")
        print("=" * 50)

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
    asyncio.run(test_connection())
