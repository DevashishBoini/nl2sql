"""
Input validation utilities for LLM and embedding operations.

This module provides simple character-based validation to ensure
inputs stay within API limits. Uses hard character limits for simplicity.
"""

from typing import List, Optional


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
