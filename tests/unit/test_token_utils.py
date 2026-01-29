"""Unit tests for token_utils module."""

import pytest
from backend.utils.token_utils import (
    truncate_sample_value,
    truncate_sample_values,
    InputValidator,
)
from backend.config import SchemaIndexingConfig

# Get default values from config for testing
_default_config = SchemaIndexingConfig()
DEFAULT_MAX_SAMPLE_VALUE_LENGTH = _default_config.max_sample_value_length
DEFAULT_MAX_SAMPLE_VALUES_COUNT = _default_config.max_sample_values_count


class TestTruncateSampleValue:
    """Tests for truncate_sample_value function."""

    def test_none_value_unchanged(self):
        """None values should be returned as-is."""
        assert truncate_sample_value(None, max_length=100) is None

    def test_non_string_types_unchanged(self):
        """Non-string types should be returned as-is."""
        assert truncate_sample_value(123, max_length=100) == 123
        assert truncate_sample_value(45.67, max_length=100) == 45.67
        assert truncate_sample_value(True, max_length=100) is True
        assert truncate_sample_value(False, max_length=100) is False

    def test_short_string_unchanged(self):
        """Strings within limit should be returned as-is."""
        short_str = "Hello, World!"
        assert truncate_sample_value(short_str, max_length=100) == short_str

    def test_exact_length_string_unchanged(self):
        """Strings exactly at limit should be returned as-is."""
        exact_str = "a" * 100
        assert truncate_sample_value(exact_str, max_length=100) == exact_str

    def test_long_string_truncated(self):
        """Strings exceeding limit should be truncated with ellipsis."""
        long_str = "a" * 150
        result = truncate_sample_value(long_str, max_length=100)

        assert len(result) == 100
        assert "..." in result
        assert result.startswith("a")
        assert result.endswith("aaa")

    def test_truncation_preserves_end_chars(self):
        """Truncation should preserve some ending characters for context."""
        long_str = "start" + "x" * 100 + "end"
        result = truncate_sample_value(long_str, max_length=50)

        assert len(result) == 50
        assert result.startswith("start")
        assert result.endswith("end")
        assert "..." in result

    def test_very_short_max_length(self):
        """Very short max_length should still work."""
        result = truncate_sample_value("Hello World", max_length=8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_with_config_default_value(self):
        """Test using the config default value for max_length."""
        long_str = "x" * (DEFAULT_MAX_SAMPLE_VALUE_LENGTH + 50)
        result = truncate_sample_value(long_str, max_length=DEFAULT_MAX_SAMPLE_VALUE_LENGTH)

        assert len(result) == DEFAULT_MAX_SAMPLE_VALUE_LENGTH


class TestTruncateSampleValues:
    """Tests for truncate_sample_values function."""

    def test_empty_list(self):
        """Empty list should be returned as-is."""
        assert truncate_sample_values([], max_length=100, max_count=5) == []

    def test_none_values_in_list(self):
        """None values in list should be preserved."""
        result = truncate_sample_values([None, "hello", None], max_length=100, max_count=5)
        assert result == [None, "hello", None]

    def test_mixed_types_handled(self):
        """Mixed types should be handled correctly."""
        values = ["short", 123, 45.67, True, None]
        result = truncate_sample_values(values, max_length=100, max_count=10)
        assert result == ["short", 123, 45.67, True, None]

    def test_long_strings_truncated(self):
        """Long strings in list should be truncated."""
        long_str = "x" * 200
        values = ["short", long_str, 123]
        result = truncate_sample_values(values, max_length=50, max_count=10)

        assert result[0] == "short"
        assert len(result[1]) == 50
        assert result[2] == 123

    def test_list_count_limited(self):
        """List should be limited to max_count."""
        values = list(range(10))
        result = truncate_sample_values(values, max_length=100, max_count=5)

        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_with_config_default_values(self):
        """Test using config default values."""
        values = list(range(20))
        result = truncate_sample_values(
            values,
            max_length=DEFAULT_MAX_SAMPLE_VALUE_LENGTH,
            max_count=DEFAULT_MAX_SAMPLE_VALUES_COUNT
        )

        assert len(result) == DEFAULT_MAX_SAMPLE_VALUES_COUNT

    def test_combined_truncation(self):
        """Both count limiting and string truncation should work together."""
        long_str = "x" * 200
        values = [long_str] * 10

        result = truncate_sample_values(values, max_length=50, max_count=3)

        assert len(result) == 3
        for v in result:
            assert len(v) == 50


class TestInputValidator:
    """Tests for InputValidator class."""

    def test_validate_char_limit_passes(self):
        """Validation should pass for text within limit."""
        # Should not raise
        InputValidator.validate_char_limit("Hello", max_chars=100)

    def test_validate_char_limit_fails(self):
        """Validation should fail for text exceeding limit."""
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_char_limit("a" * 150, max_chars=100)

        assert "150 characters" in str(exc_info.value)
        assert "maximum allowed: 100" in str(exc_info.value)

    def test_validate_char_limit_custom_message(self):
        """Custom error message should be used when provided."""
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_char_limit(
                "a" * 150,
                max_chars=100,
                error_message="Custom error"
            )

        assert str(exc_info.value) == "Custom error"

    def test_validate_total_chars_passes(self):
        """Total validation should pass when within limit."""
        # Should not raise
        InputValidator.validate_total_chars(
            prompt="Hello",
            system_prompt="System",
            max_chars=100
        )

    def test_validate_total_chars_fails(self):
        """Total validation should fail when combined exceeds limit."""
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_total_chars(
                prompt="a" * 60,
                system_prompt="b" * 60,
                max_chars=100
            )

        assert "120 characters" in str(exc_info.value)

    def test_validate_batch_chars_passes(self):
        """Batch validation should pass when all texts within limit."""
        # Should not raise
        InputValidator.validate_batch_chars(
            texts=["short", "also short"],
            max_chars_per_text=100
        )

    def test_validate_batch_chars_fails(self):
        """Batch validation should fail when any text exceeds limit."""
        with pytest.raises(ValueError) as exc_info:
            InputValidator.validate_batch_chars(
                texts=["short", "a" * 150, "ok"],
                max_chars_per_text=100
            )

        assert "index 1" in str(exc_info.value)
        assert "150 characters" in str(exc_info.value)
