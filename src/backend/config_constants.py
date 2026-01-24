from typing import Literal
from enum import Enum

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
# Input Character Limits
# -------------------------

EMBEDDING_MAX_INPUT_CHARS = 20000

# Maximum characters for LLM input (prompts + system prompts)
LLM_MAX_INPUT_CHARS = 50000