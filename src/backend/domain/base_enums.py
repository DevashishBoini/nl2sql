from enum import Enum


class NodeType(str, Enum):
    TABLE = "table"
    COLUMN = "column"
    RELATIONSHIP = "relationship"


class QueryStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SQLOperationType(str, Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    TRUNCATE = "truncate"


class PipelineStepName(str, Enum):
    """Pipeline step names for the NL2SQL processing pipeline."""
    SCHEMA_RETRIEVAL = "schema_retrieval"
    SQL_GENERATION = "sql_generation"
    SQL_VALIDATION = "sql_validation"
    SQL_EXECUTION = "sql_execution"
    TRACE_PERSISTENCE = "trace_persistence"
    QUERY_INITIALIZATION = "query_initialization"
    EMBEDDING_GENERATION = "embedding_generation"
    SIMILARITY_SEARCH = "similarity_search"
    PROMPT_BUILDING = "prompt_building"
    LLM_CALL = "llm_call"
    SQL_PARSING = "sql_parsing"
    SAFETY_CHECK = "safety_check"
    SYNTAX_CHECK = "syntax_check"
    RESULT_FORMATTING = "result_formatting"


class PipelineStepStatus(str, Enum):
    """Status values for pipeline step execution."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
