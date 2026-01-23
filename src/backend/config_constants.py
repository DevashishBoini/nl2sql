from typing import Literal
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OPENROUTER_LLM_MODELS(str, Enum):
    GPT_4O = "openai/gpt-4o"
    ANTHROPIC_SONNET_45 = "anthropic/claude-4.5-sonnet"

class OPENROUTER_EMBEDDING_MODELS(str, Enum):
    OPENAI_TEXT_EMBEDDING_MODEL_3_SMALL = "openai/text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_MODEL_3_LARGE = "openai/text-embedding-3-large"