"""
Custom exception hierarchy for NL2SQL system.

This module defines a comprehensive exception hierarchy with:
- Consistent error codes for API responses
- HTTP status code mappings for FastAPI
- Detailed error messages for debugging

Exception Categories:
- 4xx Client Errors: ValidationError, NotFoundError, BadRequestError
- 5xx Server Errors: DatabaseError, LLMError, VectorStoreError, etc.

Usage:
    raise DatabaseConnectionError("Failed to connect to database")
    raise ValidationError("Invalid query", details={"field": "query", "issue": "empty"})
"""

from typing import Any, Dict, Optional


class NL2SQLException(Exception):
    """
    Base exception for all NL2SQL errors.

    All custom exceptions inherit from this class, providing:
    - error_code: Machine-readable error identifier
    - http_status: Suggested HTTP status code for API responses
    - details: Optional structured data for debugging

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error code (e.g., "DATABASE_CONNECTION_ERROR")
        http_status: HTTP status code to return (default: 500)
        details: Optional dictionary with additional error context
    """

    error_code: str = "INTERNAL_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        http_status: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if error_code:
            self.error_code = error_code
        if http_status:
            self.http_status = http_status

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result: Dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# =============================================================================
# Client Errors (4xx)
# =============================================================================


class ValidationError(NL2SQLException):
    """
    Raised when input validation fails.

    HTTP Status: 422 Unprocessable Entity

    Examples:
        - Invalid query format
        - Missing required fields
        - Value out of allowed range
    """

    error_code = "VALIDATION_ERROR"
    http_status = 422


class BadRequestError(NL2SQLException):
    """
    Raised when the request is malformed or invalid.

    HTTP Status: 400 Bad Request

    Examples:
        - Malformed JSON
        - Invalid content type
        - Missing confirmation for destructive operations
    """

    error_code = "BAD_REQUEST"
    http_status = 400


class NotFoundError(NL2SQLException):
    """
    Raised when a requested resource is not found.

    HTTP Status: 404 Not Found

    Examples:
        - Schema not found
        - Table not found
        - Trace ID not found
    """

    error_code = "NOT_FOUND"
    http_status = 404


class RateLimitError(NL2SQLException):
    """
    Raised when rate limits are exceeded.

    HTTP Status: 429 Too Many Requests

    Examples:
        - Too many API requests
        - LLM rate limiting
    """

    error_code = "RATE_LIMIT_EXCEEDED"
    http_status = 429


# =============================================================================
# Configuration Errors (5xx)
# =============================================================================


class ConfigurationError(NL2SQLException):
    """
    Raised when configuration is invalid or missing.

    HTTP Status: 500 Internal Server Error

    Examples:
        - Missing environment variables
        - Invalid configuration values
        - Failed to load settings
    """

    error_code = "CONFIGURATION_ERROR"
    http_status = 500


# =============================================================================
# Database Errors (5xx)
# =============================================================================


class DatabaseError(NL2SQLException):
    """
    Base class for database-related errors.

    HTTP Status: 503 Service Unavailable
    """

    error_code = "DATABASE_ERROR"
    http_status = 503


class DatabaseConnectionError(DatabaseError):
    """
    Raised when database connection fails.

    HTTP Status: 503 Service Unavailable

    Examples:
        - Connection timeout
        - Authentication failure
        - Network unreachable
    """

    error_code = "DATABASE_CONNECTION_ERROR"
    http_status = 503


class DatabaseQueryError(DatabaseError):
    """
    Raised when database query execution fails.

    HTTP Status: 500 Internal Server Error

    Examples:
        - SQL syntax error
        - Table/column not found
        - Query timeout
    """

    error_code = "DATABASE_QUERY_ERROR"
    http_status = 500


# =============================================================================
# Schema Errors (5xx)
# =============================================================================


class SchemaError(NL2SQLException):
    """
    Raised when schema operations fail.

    HTTP Status: 500 Internal Server Error

    Examples:
        - Schema extraction failure
        - Invalid schema structure
    """

    error_code = "SCHEMA_ERROR"
    http_status = 500


# =============================================================================
# Vector Store Errors (5xx)
# =============================================================================


class VectorStoreError(NL2SQLException):
    """
    Raised when vector store operations fail.

    HTTP Status: 503 Service Unavailable

    Examples:
        - Embedding storage failure
        - Vector search failure
        - Index creation failure
    """

    error_code = "VECTOR_STORE_ERROR"
    http_status = 503


# =============================================================================
# LLM Errors (5xx)
# =============================================================================


class LLMError(NL2SQLException):
    """
    Raised when LLM operations fail.

    HTTP Status: 503 Service Unavailable

    Examples:
        - LLM API unreachable
        - Invalid response format
        - Context length exceeded
    """

    error_code = "LLM_ERROR"
    http_status = 503


class EmbeddingError(NL2SQLException):
    """
    Raised when embedding operations fail.

    HTTP Status: 503 Service Unavailable

    Examples:
        - Embedding API failure
        - Dimension mismatch
    """

    error_code = "EMBEDDING_ERROR"
    http_status = 503


# =============================================================================
# SQL Errors (4xx/5xx)
# =============================================================================


class SQLValidationError(NL2SQLException):
    """
    Raised when SQL validation fails.

    HTTP Status: 422 Unprocessable Entity

    Examples:
        - Unsafe SQL detected (INSERT/UPDATE/DELETE)
        - SQL syntax error
        - EXPLAIN validation failure
    """

    error_code = "SQL_VALIDATION_ERROR"
    http_status = 422


class SQLGenerationError(NL2SQLException):
    """
    Raised when SQL generation fails after all retries.

    HTTP Status: 500 Internal Server Error

    Examples:
        - LLM failed to generate valid SQL
        - All retry attempts exhausted
    """

    error_code = "SQL_GENERATION_ERROR"
    http_status = 500


class QueryExecutionError(NL2SQLException):
    """
    Raised when query execution fails.

    HTTP Status: 500 Internal Server Error

    Examples:
        - Query timeout
        - Runtime SQL error
    """

    error_code = "QUERY_EXECUTION_ERROR"
    http_status = 500


# =============================================================================
# Storage Errors (5xx)
# =============================================================================


class StorageError(NL2SQLException):
    """
    Raised when storage operations fail.

    HTTP Status: 503 Service Unavailable
    """

    error_code = "STORAGE_ERROR"
    http_status = 503


class StorageConnectionError(StorageError):
    """
    Raised when storage connection fails.

    HTTP Status: 503 Service Unavailable

    Examples:
        - Supabase storage unreachable
        - Authentication failure
    """

    error_code = "STORAGE_CONNECTION_ERROR"
    http_status = 503


class StorageFileError(StorageError):
    """
    Raised when file operations fail (download, list, etc.).

    HTTP Status: 500 Internal Server Error

    Examples:
        - File not found
        - Upload failure
        - Permission denied
    """

    error_code = "STORAGE_FILE_ERROR"
    http_status = 500


# =============================================================================
# Service Unavailable (5xx)
# =============================================================================


class ServiceUnavailableError(NL2SQLException):
    """
    Raised when a required service is not available.

    HTTP Status: 503 Service Unavailable

    Examples:
        - Database client not connected
        - LLM service not initialized
        - Embedding service not ready
    """

    error_code = "SERVICE_UNAVAILABLE"
    http_status = 503
