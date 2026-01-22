import pytest
from backend.utils.logging import configure_logging, get_logger
from backend.utils.tracing import current_trace_id, generate_trace_id, set_trace_id, get_trace_id


def test_logger_configuration():
    configure_logging()
    logger = get_logger("test")
    assert logger is not None


def test_trace_id_generation():
    trace_id = generate_trace_id()
    assert len(trace_id) == 36  # UUID format
    assert '-' in trace_id


def test_trace_id_context():
    test_id = "test-trace-123"
    set_trace_id(test_id)
    assert current_trace_id() == test_id
