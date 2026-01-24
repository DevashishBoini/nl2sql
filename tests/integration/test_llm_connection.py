"""
Integration tests for LLMClient connection and functionality.

This module verifies connectivity to OpenRouter API and tests basic
LLM generation capabilities using LangChain.

Usage:
    # Run all LLM connection tests
    pytest tests/integration/test_llm_connection.py -v

    # Run specific test
    pytest tests/integration/test_llm_connection.py::TestLLMConnection::test_basic_connection -v

    # Run with output
    pytest tests/integration/test_llm_connection.py -v -s
"""

import pytest

from backend.config import get_settings
from backend.infrastructure.llm_client import LLMClient


@pytest.fixture
def llm_config():
    """Get LLM configuration from settings."""
    settings = get_settings()
    return settings.llm


@pytest.fixture
async def llm_client(llm_config):
    """Create and connect LLM client."""
    client = LLMClient(llm_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.mark.integration
class TestLLMConnection:
    """Integration tests for LLM client connectivity."""

    @pytest.mark.asyncio
    async def test_basic_connection(self, llm_config):
        """Test basic LLM client connection and disconnection."""
        client = LLMClient(llm_config)

        # Test connection
        await client.connect()
        assert client.is_connected()

        # Test disconnection
        await client.close()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_connection_check(self, llm_client):
        """Test connection status check."""
        assert llm_client.is_connected()

    @pytest.mark.asyncio
    async def test_config_applied(self, llm_config, llm_client):
        """Test that configuration is properly applied."""
        assert llm_client.config == llm_config
        assert llm_client.config.default_model == llm_config.default_model
        assert llm_client.config.temperature == llm_config.temperature
        assert llm_client.config.max_tokens == llm_config.max_tokens


@pytest.mark.integration
class TestSimpleGeneration:
    """Integration tests for simple text generation."""

    @pytest.mark.asyncio
    async def test_generate_simple_text(self, llm_client):
        """Test simple text generation."""
        test_prompt = "What is 2 + 2? Answer with just the number."

        response = await llm_client.generate(test_prompt)

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_generate_short_response(self, llm_client):
        """Test generation with short prompt and response."""
        test_prompt = "Say 'Hello'"

        response = await llm_client.generate(test_prompt)

        assert "hello" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, llm_client):
        """Test generation with custom parameters."""
        test_prompt = "Count from 1 to 3."

        response = await llm_client.generate(
            test_prompt,
            temperature=0.1,
            max_tokens=100
        )

        assert response is not None
        assert len(response) > 0


@pytest.mark.integration
class TestSystemPrompt:
    """Integration tests for generation with system prompts."""

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm_client):
        """Test generation with system prompt."""
        system_prompt = "You are a SQL expert. Answer questions about databases concisely."
        test_prompt = "What does SELECT do in SQL?"

        response = await llm_client.generate(test_prompt, system_prompt=system_prompt)

        assert response is not None
        assert len(response) > 0
        # Response should be about SQL SELECT
        assert any(word in response.lower() for word in ["select", "retrieve", "query", "data"])

    @pytest.mark.asyncio
    async def test_system_prompt_influences_response(self, llm_client):
        """Test that system prompt influences response style."""
        system_prompt = "You are a technical expert. Answer in exactly one short sentence."
        test_prompt = "What is Python?"

        response = await llm_client.generate(test_prompt, system_prompt=system_prompt)

        assert response is not None
        # With temperature=0.1, should follow instructions fairly consistently
        assert len(response) < 500  # Should be concise


@pytest.mark.integration
class TestBatchGeneration:
    """Integration tests for batch text generation."""

    @pytest.mark.asyncio
    async def test_generate_batch(self, llm_client):
        """Test batch generation with multiple prompts."""
        prompts = [
            "What is Python? Answer in one sentence.",
            "What is SQL? Answer in one sentence.",
            "What is a database? Answer in one sentence."
        ]

        responses = await llm_client.generate_batch(prompts)

        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 0 for r in responses)

    @pytest.mark.asyncio
    async def test_generate_small_batch(self, llm_client):
        """Test batch generation with small number of prompts."""
        prompts = ["Say 'A'", "Say 'B'"]

        responses = await llm_client.generate_batch(prompts)

        assert len(responses) == 2
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_batch_with_system_prompt(self, llm_client):
        """Test batch generation with system prompt."""
        system_prompt = "You are a technical expert. Answer in exactly 10 words or less."
        prompts = [
            "What is an API?",
            "What is REST?",
            "What is JSON?",
            "What is a webhook?"
        ]

        responses = await llm_client.generate_batch(prompts, system_prompt=system_prompt)

        assert len(responses) == len(prompts)
        assert all(isinstance(r, str) for r in responses)
        # With low temperature, should follow word limit fairly consistently
        assert all(len(r.split()) < 30 for r in responses)  # Generous limit

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, llm_client):
        """Test that batch generation preserves input order."""
        prompts = [
            "Say exactly: First",
            "Say exactly: Second",
            "Say exactly: Third"
        ]

        responses = await llm_client.generate_batch(prompts)

        # Check that responses correspond to prompts (allowing for variations)
        assert "first" in responses[0].lower()
        assert "second" in responses[1].lower()
        assert "third" in responses[2].lower()

    @pytest.mark.asyncio
    async def test_batch_with_custom_parameters(self, llm_client):
        """Test batch generation with custom parameters."""
        prompts = ["Count: 1", "Count: 2", "Count: 3"]

        responses = await llm_client.generate_batch(
            prompts,
            temperature=0.1,
            max_tokens=50
        )

        assert len(responses) == 3
        assert all(r is not None for r in responses)


@pytest.mark.integration
class TestModelVariants:
    """Integration tests for different model configurations."""

    @pytest.mark.asyncio
    async def test_generate_with_different_model(self, llm_client):
        """Test generation with model override."""
        test_prompt = "Say 'Hello'"

        # Use same model family but test parameter override works
        response = await llm_client.generate(
            test_prompt,
            model=llm_client.config.default_model
        )

        assert response is not None
        assert "hello" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_with_temperature_override(self, llm_client):
        """Test generation with temperature override."""
        test_prompt = "What is Python?"

        response = await llm_client.generate(
            test_prompt,
            temperature=0.0  # Very deterministic
        )

        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens_override(self, llm_client):
        """Test generation with max tokens override."""
        test_prompt = "Count from 1 to 100"

        response = await llm_client.generate(
            test_prompt,
            max_tokens=50  # Limited output
        )

        assert response is not None
        # With limited tokens, response should be truncated
        assert len(response) < 500
