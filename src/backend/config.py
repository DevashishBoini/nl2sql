# src/nl2sql/config.py

from functools import lru_cache
from backend.config_constants import LogLevel, OPENROUTER_EMBEDDING_MODELS, OPENROUTER_LLM_MODELS

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------------------------
# Database
# -------------------------

class DatabaseConfig(BaseModel):
    database_url: str
    default_schema: str = "public"  # Default PostgreSQL schema to use
    connection_timeout_seconds: int = 10
    query_timeout_seconds: int = 30

    # Connection pool settings
    connection_pool_min_size: int = 1
    connection_pool_max_size: int = 5
    connection_pool_max_queries: int = 50000
    max_cached_statement_lifetime: int = 300
    max_cacheable_statement_size: int = 15360  # 1024 * 15 bytes

    # Application settings
    application_name: str = "nl2sql"
    jit_enabled: bool = False  # PostgreSQL JIT compilation

    # Retry settings
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


class StorageConfig(BaseModel):
    supabase_url: str  # e.g., https://xxxxx.supabase.co
    supabase_key: str  # Service role key or anon key
    default_bucket: str = "descriptions"  # Default storage bucket
    schema_yaml_path: str = "config/schema_descriptions.yaml"  # Path to schema YAML in storage

    # HTTP timeout settings
    connect_timeout_seconds: int = 10
    upload_timeout_seconds: int = 60
    download_timeout_seconds: int = 60
    write_timeout_seconds: int = 30
    pool_timeout_seconds: int = 5

    # Connection pool limits
    max_connections: int = 100
    max_keepalive_connections: int = 20


class DBQueryConfig(BaseModel):
    default_limit: int = 100
    max_limit: int = 1000


# -------------------------
# LLM / Embeddings
# -------------------------

class LLMConfig(BaseModel):
    openrouter_api_key: str
    default_model: str = OPENROUTER_LLM_MODELS.ANTHROPIC_SONNET_45
    temperature: float = 0.1
    top_p: float = 0.1
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    openrouter_api_key: str
    embedding_model: str = OPENROUTER_EMBEDDING_MODELS.OPENAI_TEXT_EMBEDDING_MODEL_3_SMALL
    embedding_dimension: int = 1536


# -------------------------
# Retrieval
# -------------------------

class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_similarity_threshold: float = 0.75


# -------------------------
# Server
# -------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    app_module: str = "backend.main:app"
    reload: bool = True
    workers: int = 1


# -------------------------
# App
# -------------------------

class AppConfig(BaseModel):
    log_level: LogLevel = LogLevel.INFO




# -------------------------
# Root Settings (ONLY BaseSettings)
# -------------------------

class Settings(BaseSettings):
    database: DatabaseConfig
    storage: StorageConfig
    db_query: DBQueryConfig = DBQueryConfig()
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    server: ServerConfig = ServerConfig()
    app: AppConfig = AppConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


# -------------------------
# Singleton accessor
# -------------------------

@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore[call-arg]
