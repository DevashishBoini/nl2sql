#!/usr/bin/env python3
"""
Development server runner for the NL2SQL API.

This script starts the FastAPI development server with hot reloading
and proper environment variable loading.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv

env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment variables from {env_file}")
else:
    print(f"⚠ No .env file found at {env_file}")
    print("  Some features may not work without proper configuration")

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from backend.config import get_settings

    # Get configuration
    settings = get_settings()
    server_config = settings.server

    print("🚀 Starting NL2SQL API development server...")
    print(f"📊 API Documentation: http://{server_config.host}:{server_config.port}/docs")
    print(f"🔍 Health Check: http://{server_config.host}:{server_config.port}/health")
    print(f"🏠 Root Endpoint: http://{server_config.host}:{server_config.port}/")
    print(f"⚙️  Server: {server_config.app_module} on {server_config.host}:{server_config.port}")
    print()

    uvicorn.run(
        server_config.app_module,
        host=server_config.host,
        port=server_config.port,
        reload=server_config.reload,
        workers=server_config.workers,
        reload_dirs=[str(src_path)],
        log_config=None,  # Use our structured logging
        access_log=False  # We handle access logging via middleware
    )