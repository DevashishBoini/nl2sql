"""
Supabase Storage client for file operations.

This module provides a minimal async storage client for Supabase Storage
with core support for uploading and downloading files.
"""

from typing import Any, Dict, Optional
import httpx

from ..config import StorageConfig
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..domain.errors import StorageConnectionError, StorageFileError


logger = get_module_logger()


class StorageClient:
    """
    Minimal async Supabase Storage client for core file operations.

    This is a thin infrastructure layer providing only essential storage operations.
    All higher-level operations (listing, deletion, validation, metadata, etc.)
    should be implemented in Repository or Service layers.

    Core Features:
    - Download files from storage
    - Upload files to storage
    - Generate public URLs
    - Connection management
    - Structured logging with trace IDs
    - Async-safe for concurrent operations

    Usage:
        client = StorageClient(config)
        await client.connect()

        # Upload file
        await client.upload_file("my-bucket", "path/to/file.txt", file_bytes)

        # Download file
        content = await client.download_file("my-bucket", "path/to/file.txt")

        # Get public URL
        url = await client.get_public_url("my-bucket", "path/to/file.txt")

        await client.close()
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize storage client with configuration.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._is_connected = False

        # Build base URL for storage API
        self.storage_url = f"{config.supabase_url}/storage/v1"

        logger.info(
            "StorageClient initialized",
            supabase_url=config.supabase_url,
            default_bucket=config.default_bucket
        )

    async def connect(self) -> None:
        """
        Initialize HTTP client for storage operations.

        Raises:
            StorageConnectionError: If connection initialization fails
        """
        if self._is_connected:
            logger.warning("Storage client already connected")
            return

        trace_id = current_trace_id()
        logger.info("Initializing storage client", trace_id=trace_id)

        try:
            # Create async HTTP client with connection pooling
            # httpx.AsyncClient is async-safe for concurrent requests
            self._client = httpx.AsyncClient(
                base_url=self.storage_url,
                headers={
                    "Authorization": f"Bearer {self.config.supabase_key}",
                    "apikey": self.config.supabase_key
                },
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_seconds,
                    read=self.config.download_timeout_seconds,
                    write=self.config.write_timeout_seconds,
                    pool=self.config.pool_timeout_seconds
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections
                )
            )

            # Test connection by listing buckets
            await self._test_connection()

            self._is_connected = True
            logger.info("Storage client initialized successfully", trace_id=trace_id)

        except Exception as e:
            error_msg = f"Failed to initialize storage client: {e}"
            logger.error(error_msg, error_type=type(e).__name__, trace_id=trace_id)
            raise StorageConnectionError(error_msg) from e

    async def _test_connection(self) -> None:
        """Test storage connection."""
        trace_id = current_trace_id()
        try:
            if not self._client:
                raise StorageConnectionError("HTTP client not initialized")

            # Simple health check endpoint
            response = await self._client.get("/")
            if response.status_code not in (200, 404):  # 404 is ok for root endpoint
                raise StorageConnectionError(f"Storage connection test failed: {response.text}")

            logger.info("Storage connection test successful", trace_id=trace_id)
        except Exception as e:
            error_msg = f"Storage connection test failed: {e}"
            logger.error(error_msg, trace_id=trace_id)
            raise StorageConnectionError(error_msg) from e

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        trace_id = current_trace_id()
        logger.info("Closing storage client", trace_id=trace_id)

        if self._client:
            await self._client.aclose()
            logger.info("Storage client closed", trace_id=trace_id)

        self._is_connected = False
        self._client = None

    def is_connected(self) -> bool:
        """Check if storage client is connected."""
        return self._is_connected and self._client is not None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on storage connection.

        Returns:
            Dictionary with status and connection details

        Example:
            {
                "status": "healthy",
                "connected": True,
                "storage_url": "https://example.supabase.co/storage/v1",
                "default_bucket": "documents"
            }
        """
        trace_id = current_trace_id()

        if not self.is_connected():
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Storage client not connected"
            }

        try:
            if not self._client:
                raise StorageConnectionError("HTTP client not initialized")

            # Test connection with simple request
            response = await self._client.get("/")

            # Accept both 200 and 404 as valid responses (404 is ok for root)
            if response.status_code not in (200, 404):
                return {
                    "status": "unhealthy",
                    "connected": True,
                    "error": f"Unexpected status code: {response.status_code}"
                }

            logger.info("Storage health check passed", trace_id=trace_id)

            return {
                "status": "healthy",
                "connected": True,
                "storage_url": self.storage_url,
                "default_bucket": self.config.default_bucket
            }

        except Exception as e:
            logger.error(
                "Storage health check failed",
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id
            )
            return {
                "status": "unhealthy",
                "connected": True,
                "error": str(e)
            }

    async def download_file(
        self,
        bucket: Optional[str],
        file_path: str
    ) -> bytes:
        """
        Download a file from storage.

        Args:
            bucket: Bucket name (uses default if not specified)
            file_path: Path to file in bucket

        Returns:
            File content as bytes

        Raises:
            StorageConnectionError: If client is not connected
            StorageFileError: If download fails (e.g., file not found, permission denied)

        Note: This is async-safe and can be called concurrently.
        Multiple downloads can run simultaneously without blocking.

        Example:
            # Download file
            content = await client.download_file("documents", "report.pdf")

            # Save to local file
            with open("local_report.pdf", "wb") as f:
                f.write(content)

            # Download from default bucket
            content = await client.download_file(None, "file.txt")
        """
        if not self.is_connected():
            raise StorageConnectionError("Storage client is not connected")

        bucket_name = bucket or self.config.default_bucket
        trace_id = current_trace_id()

        logger.info(
            "Downloading file from storage",
            bucket=bucket_name,
            file_path=file_path,
            trace_id=trace_id
        )

        try:
            if not self._client:
                raise StorageConnectionError("HTTP client not initialized")

            # Remove leading slash if present
            clean_path = file_path.lstrip("/")

            response = await self._client.get(f"/object/{bucket_name}/{clean_path}")
            response.raise_for_status()

            content = response.content
            logger.info(
                "File downloaded successfully",
                bucket=bucket_name,
                file_path=file_path,
                size=len(content),
                trace_id=trace_id
            )
            return content

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to download file: {e.response.text if e.response else str(e)}"
            logger.error(
                error_msg,
                status_code=e.response.status_code if e.response else None,
                bucket=bucket_name,
                file_path=file_path,
                trace_id=trace_id
            )
            raise StorageFileError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to download file: {e}"
            logger.error(error_msg, bucket=bucket_name, file_path=file_path, trace_id=trace_id)
            raise StorageFileError(error_msg) from e

    async def upload_file(
        self,
        bucket: Optional[str],
        file_path: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        upsert: bool = True
    ) -> Dict[str, Any]:
        """
        Upload a file to storage.

        Args:
            bucket: Bucket name (uses default if not specified)
            file_path: Path where file will be stored in bucket
            file_data: File content as bytes
            content_type: MIME type (e.g., "image/png", "application/pdf"). Auto-detected if not provided.
            upsert: If True, overwrite existing file. If False, fail if file exists.

        Returns:
            Dictionary with upload response metadata

        Raises:
            StorageConnectionError: If client is not connected
            StorageFileError: If upload fails

        Example:
            # Upload text file
            with open("document.txt", "rb") as f:
                data = f.read()
            result = await client.upload_file("documents", "reports/doc.txt", data, "text/plain")

            # Upload image with upsert
            with open("image.png", "rb") as f:
                data = f.read()
            result = await client.upload_file("images", "logo.png", data, "image/png", upsert=True)
        """
        if not self.is_connected():
            raise StorageConnectionError("Storage client is not connected")

        bucket_name = bucket or self.config.default_bucket
        clean_path = file_path.lstrip("/")

        trace_id = current_trace_id()
        logger.info(
            "Uploading file to storage",
            bucket=bucket_name,
            file_path=clean_path,
            size_bytes=len(file_data),
            upsert=upsert,
            trace_id=trace_id
        )

        try:
            if not self._client:
                raise StorageConnectionError("HTTP client not initialized")

            # Construct upload URL (base_url already includes /storage/v1)
            url = f"/object/{bucket_name}/{clean_path}"

            # Prepare headers
            headers = {}
            if content_type:
                headers["Content-Type"] = content_type
            if upsert:
                headers["x-upsert"] = "true"

            # Upload file
            response = await self._client.post(
                url,
                content=file_data,
                headers=headers,
                timeout=self.config.upload_timeout_seconds
            )
            response.raise_for_status()

            result = response.json()
            logger.info(
                "File uploaded successfully",
                bucket=bucket_name,
                file_path=clean_path,
                trace_id=trace_id
            )
            return result

        except httpx.HTTPStatusError as e:
            error_msg = f"Failed to upload file: {e.response.text if e.response else str(e)}"
            logger.error(
                error_msg,
                status_code=e.response.status_code if e.response else None,
                bucket=bucket_name,
                file_path=clean_path,
                trace_id=trace_id
            )
            raise StorageFileError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to upload file: {e}"
            logger.error(error_msg, bucket=bucket_name, file_path=clean_path, trace_id=trace_id)
            raise StorageFileError(error_msg) from e



