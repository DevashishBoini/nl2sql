"""
Configuration module for the NL2SQL application.

This module defines all configuration classes using Pydantic BaseModel and BaseSettings.
Configuration is loaded from environment variables with nested delimiter "__".

Example .env:
    DATABASE__DATABASE_URL=postgresql://user:pass@host:5432/db
    LLM__OPENROUTER_API_KEY=sk-xxx
    NL2SQL__MAX_TABLES=6

Usage:
    from backend.config import get_settings
    settings = get_settings()
    print(settings.database.database_url)
"""

from functools import lru_cache

from backend.config_constants import (
    LogLevel,
    OPENROUTER_EMBEDDING_MODELS,
    OPENROUTER_LLM_MODELS,
    OPEN_ROUTER_API_URL,
    DistanceStrategy,
)

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig(BaseModel):
    """
    PostgreSQL database connection and pool configuration.

    Used by DatabaseClient for all database operations including
    schema extraction, SQL execution, and vector storage.
    """

    # Connection string for PostgreSQL database
    # Format: postgresql://user:password@host:port/database
    # For Supabase: Use pooler URL for runtime
    database_url: str

    # PostgreSQL schema to use for all operations (e.g., "public", "myschema")
    # Tables/columns will be queried from this schema
    default_schema: str = "public"

    # Maximum time (seconds) to wait when establishing a new connection
    # Increase for slow networks or distant databases
    connection_timeout_seconds: int = 10

    # Maximum time (seconds) a query can run before being cancelled
    # Protects against runaway queries; user queries should complete faster
    query_timeout_seconds: int = 30

    # Minimum number of connections to keep open in the pool
    # Higher values reduce latency for first requests after idle period
    connection_pool_min_size: int = 1

    # Maximum number of concurrent connections in the pool
    # Should match expected concurrent request load; don't exceed DB limits
    connection_pool_max_size: int = 5

    # Maximum number of queries a single connection can execute before recycling
    # Helps prevent memory leaks in long-running connections
    connection_pool_max_queries: int = 50000

    # Seconds to cache prepared statements before invalidating
    # Lower values = more recompilation; higher values = stale plans possible
    max_cached_statement_lifetime: int = 300

    # Maximum size (bytes) of SQL statements to cache as prepared statements
    # Larger queries won't be cached; 15KB is reasonable for complex queries
    max_cacheable_statement_size: int = 15360  # 1024 * 15 bytes

    # Application name sent to PostgreSQL (visible in pg_stat_activity)
    # Useful for monitoring and identifying connections from this app
    application_name: str = "nl2sql"

    # Enable PostgreSQL JIT (Just-In-Time) compilation for queries
    # Can speed up complex queries but adds overhead; disabled for typical NL2SQL
    jit_enabled: bool = False

    # If True, all user queries run in read-only transaction mode
    # CRITICAL for security: prevents INSERT/UPDATE/DELETE even if SQL validation fails
    enforce_read_only_default: bool = True

    # Number of times to retry failed database operations
    # Helps with transient network issues; exponential backoff applied
    max_retries: int = 3

    # Base delay (seconds) between retry attempts
    # Actual delay = retry_backoff_seconds * (2 ^ attempt_number)
    retry_backoff_seconds: float = 2.0


# =============================================================================
# STORAGE CONFIGURATION (Supabase)
# =============================================================================

class StorageConfig(BaseModel):
    """
    Supabase Storage configuration for YAML schema descriptions.

    The YAML file contains human-written descriptions for tables and columns,
    which are merged into embeddings for better semantic search.
    """

    # Supabase project URL (e.g., https://abcdef.supabase.co)
    # Found in Supabase Dashboard > Project Settings > API
    supabase_url: str

    # Supabase API key (service_role key recommended for server-side access)
    # Use service_role for full access; anon key has RLS restrictions
    supabase_key: str

    # Storage bucket name where schema YAML file is stored
    # Create this bucket in Supabase Dashboard > Storage
    default_bucket: str = "descriptions"

    # Path to the YAML file within the bucket
    # Contains table and column descriptions for semantic enrichment
    schema_yaml_path: str = "schema_descriptions.yaml"

    # Maximum time (seconds) to establish HTTP connection to Supabase
    connect_timeout_seconds: int = 10

    # Maximum time (seconds) for file upload operations
    upload_timeout_seconds: int = 60

    # Maximum time (seconds) for file download operations
    # YAML files are small, but network latency varies
    download_timeout_seconds: int = 60

    # Maximum time (seconds) for write operations (create/update)
    write_timeout_seconds: int = 30

    # Maximum time (seconds) to wait for a connection from the pool
    pool_timeout_seconds: int = 5

    # Maximum total HTTP connections to Supabase
    # Higher values support more concurrent operations
    max_connections: int = 100

    # Maximum idle connections to keep alive
    # Reduces latency for subsequent requests
    max_keepalive_connections: int = 20


# =============================================================================
# DATABASE QUERY LIMITS
# =============================================================================

class DBQueryConfig(BaseModel):
    """
    Row limits for SQL query results.

    These limits protect against accidentally returning massive result sets
    that could overwhelm the client or exhaust server memory.
    """

    # Default LIMIT applied to queries when user doesn't specify
    # Prevents unbounded result sets; users can request up to max_limit
    default_limit: int = 100

    # Absolute maximum rows that can be returned, regardless of user request
    # Hard cap for safety; exceeding this is always blocked
    max_limit: int = 1000


# =============================================================================
# SCHEMA INDEXING CONFIGURATION
# =============================================================================

class SchemaIndexingConfig(BaseModel):
    """
    Configuration for schema extraction and vector indexing.

    Controls how sample values are collected and stored in column metadata.
    Sample values help LLM understand data patterns (e.g., date formats, categories).
    """

    # Maximum number of sample values to store per column
    # More samples = better LLM context but larger embeddings
    max_sample_values_count: int = 5

    # Maximum characters per sample value (truncated if longer)
    # Prevents huge text blobs from dominating the embedding
    max_sample_value_length: int = 100


# =============================================================================
# NL2SQL PIPELINE CONFIGURATION
# =============================================================================

class NL2SQLConfig(BaseModel):
    """
    Configuration for the NL2SQL query pipeline.

    Controls retrieval, filtering, generation, and execution steps.
    These settings balance quality vs. cost/latency.
    """

    # Number of schema nodes (columns + relationships) to retrieve via vector search
    # Higher values = more context for LLM but slower and more expensive
    # Recommended: 10-20 for complex schemas, 5-10 for simple schemas
    retrieval_top_k: int = 12

    # Maximum tables allowed in generated SQL (hard cap, enforced deterministically)
    # Prevents overly complex queries; 6 supports most reasonable JOINs
    max_tables: int = 6

    # Maximum columns allowed in generated SQL context
    # Includes SELECT columns, WHERE filters, JOIN keys
    max_columns: int = 15

    # Maximum foreign key relationships to include in context
    # Guides LLM on valid JOINs; 6 supports multi-hop relationships
    max_relationships: int = 6

    # Number of retry attempts when SQL validation fails
    # Each retry includes the error message for LLM self-correction
    # 2-3 retries usually sufficient; more = higher latency/cost
    max_retries: int = 2

    # Default LIMIT clause value when user doesn't specify
    # Overridden by user request (up to max_row_limit)
    default_row_limit: int = 100

    # Maximum LIMIT clause value allowed (hard cap)
    # Protects against "give me all data" attacks
    max_row_limit: int = 1000

    # SQL execution timeout (seconds)
    # Queries exceeding this are cancelled; prevents runaway queries
    query_timeout_seconds: int = 30


# =============================================================================
# LLM CONFIGURATION (OpenRouter)
# =============================================================================

class LLMConfig(BaseModel):
    """
    LLM client configuration for SQL generation.

    Uses OpenRouter API to access various LLM providers (Claude, GPT-4, etc.).
    Temperature and top_p are set low for deterministic SQL output.
    """

    # OpenRouter API key (get from https://openrouter.ai/keys)
    # Required for all LLM operations
    openrouter_api_key: str

    # Default model for SQL generation
    # Claude models recommended for SQL; GPT-4 also works well
    # Format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4")
    default_model: str = OPENROUTER_LLM_MODELS.ANTHROPIC_SONNET_45

    # Sampling temperature (0.0-1.0)
    # Lower = more deterministic; 0.0-0.2 recommended for SQL generation
    temperature: float = 0.1

    # Nucleus sampling parameter (0.0-1.0)
    # Lower = less random; combined with low temperature for consistency
    top_p: float = 0.1

    # Maximum tokens in LLM response
    # SQL queries rarely exceed 500 tokens; 2048 allows for complex queries
    max_tokens: int = 2048

    # Maximum characters allowed in LLM input (prompt + system prompt)
    # Protects against context window overflow; adjust per model limits
    max_input_chars: int = 50000

    # If True, enforce JSON response format from LLM
    # Recommended: Makes parsing reliable; prompt must mention "JSON"
    json_mode: bool = True

    # OpenRouter API base URL (don't change unless using proxy)
    base_url: str = OPEN_ROUTER_API_URL

    # Maximum time (seconds) to wait for LLM response
    # Complex queries may take 30-60s; increase for slow models
    timeout_seconds: int = 60

    # Number of retry attempts on transient LLM errors (rate limits, timeouts)
    # Exponential backoff applied between retries
    max_retries: int = 3

    # Maximum prompts per batch request
    # Helps avoid rate limits and timeouts for batch operations
    batch_size: int = 10


# =============================================================================
# EMBEDDING CONFIGURATION (OpenRouter)
# =============================================================================

class EmbeddingConfig(BaseModel):
    """
    Embedding client configuration for vector search.

    Uses OpenRouter API to access embedding models (OpenAI, etc.).
    Embeddings are stored in pgvector for semantic schema retrieval.
    """

    # OpenRouter API key (same as LLM key, or separate for billing)
    openrouter_api_key: str

    # Embedding model to use
    # text-embedding-3-small: Good balance of quality/cost (1536 dims)
    # text-embedding-3-large: Higher quality, more expensive (3072 dims)
    embedding_model: str = OPENROUTER_EMBEDDING_MODELS.OPENAI_TEXT_EMBEDDING_MODEL_3_SMALL

    # Embedding vector dimension (must match model output)
    # text-embedding-3-small: 1536
    # text-embedding-3-large: 3072
    # MUST match vector column size in pgvector table
    embedding_dimension: int = 1536

    # Maximum characters allowed in text to embed
    # Model-dependent; text-embedding-3 supports ~8K tokens
    max_input_chars: int = 20000

    # OpenRouter API base URL
    base_url: str = OPEN_ROUTER_API_URL

    # Maximum time (seconds) to wait for embedding response
    # Batch embedding may take longer; 60s usually sufficient
    timeout_seconds: int = 60

    # Number of retry attempts on transient errors
    max_retries: int = 3

    # Maximum texts to embed in a single API request
    # Higher = more efficient but risks timeouts; 100 is safe default
    batch_size: int = 100


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

class RetrievalConfig(BaseModel):
    """
    Vector similarity search configuration.

    Controls how many results to retrieve and minimum similarity threshold.
    Used by schema search endpoint (debugging) and internal retrieval.
    """

    # Number of results to return from vector search
    # Lower than NL2SQL retrieval_top_k for focused debugging
    top_k: int = 5

    # Minimum cosine similarity score (0.0-1.0) to include in results
    # 0.5-0.7 = moderate relevance; 0.7+ = high relevance
    # Set to 0.75 to filter noise; lower for more recall
    min_similarity_threshold: float = 0.75


# =============================================================================
# VECTOR STORE CONFIGURATION (pgvector)
# =============================================================================

class VectorStoreConfig(BaseModel):
    """
    pgvector configuration for schema embeddings.

    Controls the PostgreSQL table structure, HNSW index parameters,
    and similarity search behavior.
    """

    # PostgreSQL table name for storing embeddings
    # Created automatically on first schema indexing
    table_name: str = "schema_embeddings"

    # Logical collection name (for future multi-tenant support)
    collection_name: str = "schema_vectors"

    # If True, create HNSW index for fast approximate nearest neighbor search
    # Recommended for tables with >1000 vectors; exact search used otherwise
    use_hnsw: bool = True

    # HNSW M parameter: max connections per node in the graph
    # Higher = better recall but slower indexing; 16 is good default
    hnsw_m: int = 16

    # HNSW ef_construction: search width during index building
    # Higher = better index quality but slower build; 64-200 typical
    hnsw_ef_construction: int = 64

    # HNSW ef_search: search width during queries
    # Higher = better recall but slower search; 40-100 typical
    hnsw_ef_search: int = 40

    # Distance metric for similarity calculation
    # COSINE: normalized, good for text embeddings (recommended)
    # EUCLIDEAN: L2 distance, sensitive to magnitude
    # MAX_INNER_PRODUCT: dot product, requires normalized vectors
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE

    # Number of vectors to insert per database transaction
    # Higher = faster bulk insert but more memory; 100 is safe
    batch_size: int = 100

    # Default number of results for similarity search
    default_k: int = 5

    # Default minimum similarity threshold (0.0 = no threshold)
    # Set low (0.0) for maximum recall; adjust per use case
    # Cosine similarity: 0.3-0.5 = moderate, 0.5-0.7 = good, 0.7+ = high
    default_min_similarity: float = 0.00  # Retrieval must maximize recall, not precision.


# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

class ServerConfig(BaseModel):
    """
    FastAPI/Uvicorn server configuration.

    Used by run_dev.py and run_prod.py scripts.
    """

    # Network interface to bind (0.0.0.0 = all interfaces)
    # Use 127.0.0.1 for local-only access
    host: str = "0.0.0.0"

    # Port number to listen on
    # Default 8000; change if port is in use
    port: int = 8000

    # Python module path for FastAPI app
    # Format: "package.module:app_variable"
    app_module: str = "backend.main:app"

    # Enable hot reload on code changes (development only)
    # Disable in production for stability
    reload: bool = True

    # Number of worker processes (production only, ignored with reload=True)
    # Set to CPU count for production; 1 for development
    workers: int = 1


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

class AppConfig(BaseModel):
    """
    General application settings.

    Controls logging verbosity and other app-wide behavior.
    """

    # Logging level: DEBUG, INFO, WARNING, ERROR
    # DEBUG: verbose, includes SQL and prompts (development)
    # INFO: normal operation logging (production)
    # WARNING: only warnings and errors
    # ERROR: only errors
    log_level: LogLevel = LogLevel.INFO


# =============================================================================
# ROOT SETTINGS (Environment Loading)
# =============================================================================

class Settings(BaseSettings):
    """
    Root settings class that loads all configuration from environment.

    Environment variables use "__" (double underscore) as nested delimiter.
    Example: DATABASE__DATABASE_URL sets settings.database.database_url

    Required environment variables (no defaults):
    - DATABASE__DATABASE_URL
    - STORAGE__SUPABASE_URL
    - STORAGE__SUPABASE_KEY
    - LLM__OPENROUTER_API_KEY
    - EMBEDDING__OPENROUTER_API_KEY
    """

    # Database connection and pool settings
    database: DatabaseConfig

    # Supabase Storage for YAML descriptions
    storage: StorageConfig

    # SQL query row limits
    db_query: DBQueryConfig = DBQueryConfig()

    # Schema extraction settings
    schema_indexing: SchemaIndexingConfig = SchemaIndexingConfig()

    # NL2SQL pipeline settings
    nl2sql: NL2SQLConfig = NL2SQLConfig()

    # LLM client settings (OpenRouter)
    llm: LLMConfig

    # Embedding client settings (OpenRouter)
    embedding: EmbeddingConfig

    # Vector similarity search defaults
    retrieval: RetrievalConfig = RetrievalConfig()

    # pgvector table and index settings
    vector_store: VectorStoreConfig = VectorStoreConfig()

    # FastAPI server settings
    server: ServerConfig = ServerConfig()

    # Application-wide settings
    app: AppConfig = AppConfig()

    model_config = SettingsConfigDict(
        env_file=".env",            # Load from .env file in project root
        env_file_encoding="utf-8",  # UTF-8 encoding for .env file
        case_sensitive=False,       # ENV_VAR and env_var are equivalent
        env_nested_delimiter="__",  # Use __ for nested config (DATABASE__URL)
    )


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    """
    Get cached Settings instance (singleton pattern).

    Settings are loaded once and cached for the lifetime of the application.
    Use dependency injection in FastAPI routes:

        @app.get("/")
        def root(settings: Settings = Depends(get_settings)):
            return {"db": settings.database.database_url}

    Returns:
        Settings instance with all configuration loaded from environment
    """
    return Settings()  # type: ignore[call-arg]
