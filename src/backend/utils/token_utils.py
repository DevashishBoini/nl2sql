"""
Input validation utilities for LLM and embedding operations.

This module provides simple character-based validation to ensure
inputs stay within API limits. Uses hard character limits for simplicity.
"""

from typing import Any, List, Optional


def truncate_sample_value(value: Any, max_length: int) -> Any:
    """
    Truncate a sample value if it exceeds the maximum length.

    Only truncates string values. Other types (int, float, bool, None) are returned as-is.

    Args:
        value: The sample value to potentially truncate
        max_length: Maximum allowed character length for strings

    Returns:
        Original value if not a string or within limit, truncated string with "..." if exceeded

    Example:
        >>> truncate_sample_value("short")
        'short'
        >>> truncate_sample_value("a" * 150, max_length=100)
        'aaaa...aaa'  # 97 chars + "..."
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    if len(value) <= max_length:
        return value

    # Truncate and add ellipsis
    # Keep some chars from end for context (e.g., file extensions, IDs)
    if max_length <= 10:
        return value[:max_length - 3] + "..."

    # For longer limits, show beginning and end
    prefix_len = max_length - 6  # "..." in middle takes 3, keep some end chars
    suffix_len = 3
    return value[:prefix_len] + "..." + value[-suffix_len:]


def truncate_sample_values(
    values: List[Any],
    max_length: int,
    max_count: int,
) -> List[Any]:
    """
    Truncate a list of sample values.

    Limits both the number of values and the length of each string value.

    Args:
        values: List of sample values
        max_length: Maximum character length per string value
        max_count: Maximum number of values to keep

    Returns:
        List of truncated sample values

    Example:
        >>> truncate_sample_values(["short", "a" * 200, 123, None])
        ['short', 'aaaa...aaa', 123, None]
    """
    if not values:
        return values

    # Limit count first
    limited_values = values[:max_count]

    # Truncate each value
    return [truncate_sample_value(v, max_length) for v in limited_values]


class InputValidator:
    """
    Input validation utility for checking character limits.

    Uses simple character count checks against hard limits.
    """

    @staticmethod
    def validate_char_limit(
        text: str,
        max_chars: int,
        error_message: Optional[str] = None
    ) -> None:
        """
        Validate that text does not exceed maximum character limit.

        Args:
            text: Text to validate
            max_chars: Maximum allowed characters
            error_message: Optional custom error message

        Raises:
            ValueError: If text exceeds character limit

        Example:
            >>> InputValidator.validate_char_limit("Hello", max_chars=100)  # OK
            >>> InputValidator.validate_char_limit("A" * 1000, max_chars=100)  # Raises ValueError
        """
        char_count = len(text)

        if char_count > max_chars:
            if error_message:
                raise ValueError(error_message)
            else:
                raise ValueError(
                    f"Input too large: {char_count} characters, "
                    f"maximum allowed: {max_chars}"
                )

    @staticmethod
    def validate_total_chars(
        prompt: str,
        system_prompt: Optional[str] = None,
        max_chars: int = 0
    ) -> None:
        """
        Validate total character count for an LLM request.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            max_chars: Maximum allowed total characters

        Raises:
            ValueError: If total exceeds character limit

        Example:
            >>> InputValidator.validate_total_chars("Hello", system_prompt="Hi", max_chars=1000)  # OK
        """
        total_chars = len(prompt)
        if system_prompt:
            total_chars += len(system_prompt)

        if total_chars > max_chars:
            raise ValueError(
                f"Total input too large: {total_chars} characters, "
                f"maximum allowed: {max_chars}"
            )

    @staticmethod
    def validate_batch_chars(
        texts: List[str],
        max_chars_per_text: int,
        error_message: Optional[str] = None
    ) -> None:
        """
        Validate that all texts in a batch are within character limits.

        Args:
            texts: List of texts to validate
            max_chars_per_text: Maximum characters allowed per individual text
            error_message: Optional custom error message prefix

        Raises:
            ValueError: If any text exceeds character limit

        Example:
            >>> InputValidator.validate_batch_chars(["Hello", "World"], max_chars_per_text=100)  # OK
            >>> InputValidator.validate_batch_chars(["A" * 1000], max_chars_per_text=100)  # Raises
        """
        for i, text in enumerate(texts):
            char_count = len(text)
            if char_count > max_chars_per_text:
                prefix = error_message or "Text in batch"
                raise ValueError(
                    f"{prefix} at index {i} too large: "
                    f"{char_count} characters, maximum allowed: {max_chars_per_text}"
                )
