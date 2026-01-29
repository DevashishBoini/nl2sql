from enum import Enum
from langchain_postgres.vectorstores import DistanceStrategy


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OPENROUTER_LLM_MODELS(str, Enum):
    # OpenAI models
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"

    # Anthropic Claude models
    ANTHROPIC_SONNET_45 = "anthropic/claude-4.5-sonnet"
    ANTHROPIC_HAIKU_45 = "anthropic/claude-haiku-4.5"

    # Google Gemini models
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"

class OPENROUTER_EMBEDDING_MODELS(str, Enum):
    OPENAI_TEXT_EMBEDDING_MODEL_3_SMALL = "openai/text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_MODEL_3_LARGE = "openai/text-embedding-3-large"

OPEN_ROUTER_API_URL = "https://openrouter.ai/api/v1"

# -------------------------
# Vector Store Constants
# -------------------------

# Mapping from DistanceStrategy to PostgreSQL pgvector operator classes
# These operator class names are PostgreSQL pgvector extension constants
# used in CREATE INDEX statements, not Python library constants
PGVECTOR_OPS_MAP = {
    DistanceStrategy.COSINE: "vector_cosine_ops",
    DistanceStrategy.EUCLIDEAN: "vector_l2_ops",
    DistanceStrategy.MAX_INNER_PRODUCT: "vector_ip_ops",
}