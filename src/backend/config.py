# src/nl2sql/config.py

from functools import lru_cache
from backend.constants import LogLevel, OPENROUTER_EMBEDDING_MODELS, OPENROUTER_LLM_MODELS

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------------------------
# Database
# -------------------------

class DatabaseConfig(BaseModel):
    database_url: str
    database_readonly_url: str
    database_schema_yaml_path: str
    connection_timeout_seconds: int = 10
    query_timeout_seconds: int = 30
    connection_pool_size: int = 5
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


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
# App
# -------------------------

class AppConfig(BaseModel):
    log_level: LogLevel = LogLevel.INFO




# -------------------------
# Root Settings (ONLY BaseSettings)
# -------------------------

class Settings(BaseSettings):
    database: DatabaseConfig
    db_query: DBQueryConfig = DBQueryConfig()
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    app: AppConfig

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
