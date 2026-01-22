import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable to store trace ID across async operations
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


def generate_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context."""
    trace_id_var.set(trace_id)


def current_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    return trace_id_var.get()


def get_trace_id() -> str:
    """Get existing trace ID or create a new one."""
    trace_id = current_trace_id()
    if trace_id is None:
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
    return trace_id
