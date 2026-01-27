"""
Utility for loading and parsing schema description YAML files.

This module handles loading YAML files containing table and column descriptions
from storage buckets (Supabase).
"""

import yaml
from typing import Dict, Any, Optional

from ..infrastructure.storage_client import StorageClient
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id


logger = get_module_logger()


class YAMLDescriptionLoader:
    """
    Loader for schema description YAML files from storage bucket.

    Loads YAML files from Supabase storage bucket only.
    """

    def __init__(self, storage_client: Optional[StorageClient] = None):
        """
        Initialize YAML loader.

        Args:
            storage_client: StorageClient for loading from bucket
        """
        self.storage_client = storage_client

    async def load(
        self,
        bucket: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Load YAML file from storage bucket.

        Args:
            bucket: Storage bucket name
            file_path: Path to YAML file in bucket

        Returns:
            Parsed YAML content as dictionary

        Raises:
            RuntimeError: If storage client not configured
            Exception: If file download or parse fails
        """
        if not self.storage_client:
            raise RuntimeError("Storage client not configured")

        trace_id = current_trace_id()
        logger.info(
            "Loading YAML from storage bucket",
            bucket=bucket,
            file_path=file_path,
            trace_id=trace_id
        )

        try:
            # Download file content from storage
            file_bytes = await self.storage_client.download_file(
                bucket=bucket,
                file_path=file_path
            )

            # Parse YAML from bytes
            yaml_content = yaml.safe_load(file_bytes)

            logger.info(
                "YAML loaded from storage successfully",
                bucket=bucket,
                file_path=file_path,
                trace_id=trace_id
            )

            return yaml_content

        except Exception as e:
            logger.error(
                f"Failed to load YAML from storage: {e}",
                bucket=bucket,
                file_path=file_path,
                trace_id=trace_id
            )
            raise


def parse_descriptions(
    yaml_content: Dict[str, Any]
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse table and column descriptions from YAML content.

    Args:
        yaml_content: Parsed YAML dictionary with structure:
            {
                "tables": {
                    "table_name": {
                        "description": "table desc",
                        "columns": {
                            "table.column": {"description": "col desc"}
                        }
                    }
                }
            }

    Returns:
        Tuple of (table_descriptions, column_descriptions) where:
        - table_descriptions: {"table_name": "description"}
        - column_descriptions: {"table_name.column_name": "description"}
    """
    table_descriptions: Dict[str, str] = {}
    column_descriptions: Dict[str, str] = {}

    tables = yaml_content.get("tables", {})

    for table_name, table_data in tables.items():
        # Extract table description
        if "description" in table_data:
            table_descriptions[table_name] = table_data["description"]

        # Extract column descriptions
        columns = table_data.get("columns", {})
        for column_key, column_data in columns.items():
            if "description" in column_data:
                column_descriptions[column_key] = column_data["description"]

    logger.info(
        "Parsed descriptions from YAML",
        table_count=len(table_descriptions),
        column_count=len(column_descriptions)
    )

    return table_descriptions, column_descriptions
