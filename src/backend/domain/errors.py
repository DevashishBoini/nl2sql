"""Custom exception hierarchy for NL2SQL system."""


class NL2SQLException(Exception):
    """Base exception for all NL2SQL errors."""
    pass


class ConfigurationError(NL2SQLException):
    """Raised when configuration is invalid or missing."""
    pass


class DatabaseError(NL2SQLException):
    """Raised when database operations fail."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails."""
    pass


class SchemaError(NL2SQLException):
    """Raised when schema operations fail."""
    pass


class VectorStoreError(NL2SQLException):
    """Raised when vector store operations fail."""
    pass


class LLMError(NL2SQLException):
    """Raised when LLM operations fail."""
    pass

class EmbeddingError(NL2SQLException):
    """Raised when embedding operations fail."""
    pass

class SQLValidationError(NL2SQLException):
    """Raised when SQL validation fails."""
    pass


class QueryExecutionError(NL2SQLException):
    """Raised when query execution fails."""
    pass


class StorageError(NL2SQLException):
    """Raised when storage operations fail."""
    pass


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""
    pass


class StorageFileError(StorageError):
    """Raised when file operations fail (download, list, etc.)."""
    pass
