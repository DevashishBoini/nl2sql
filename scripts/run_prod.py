#!/usr/bin/env python3
"""
Production server runner for the NL2SQL API.

This script starts the FastAPI production server with optimized settings
for performance, security, and reliability.
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
    print("  Ensure environment variables are set via your deployment system")

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from backend.config import get_settings

    # Get configuration
    settings = get_settings()
    server_config = settings.server

    # Override development settings for production
    production_config = {
        "app": server_config.app_module,
        "host": server_config.host,
        "port": server_config.port,
        "workers": max(server_config.workers, 2),  # Minimum 2 workers
        "reload": False,  # Never reload in production
        "log_config": None,  # Use our structured logging
        "access_log": False,  # We handle access logging via middleware
        "server_header": False,  # Security: don't reveal server info
        "date_header": False,  # Security: don't reveal server time
    }

    print("🚀 Starting NL2SQL API production server...")
    print(f"⚙️  Configuration: {server_config.app_module}")
    print(f"🌐 Listening: {server_config.host}:{server_config.port}")
    print(f"👥 Workers: {production_config['workers']}")
    print(f"📊 API Documentation: http://{server_config.host}:{server_config.port}/docs")
    print(f"🔍 Health Check: http://{server_config.host}:{server_config.port}/health")
    print()

    uvicorn.run(**production_config)