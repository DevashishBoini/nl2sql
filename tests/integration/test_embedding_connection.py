"""
Integration tests for EmbeddingClient connection and functionality.

This module verifies connectivity to OpenRouter API and tests embedding
generation capabilities using LangChain.

Usage:
    # Run all embedding connection tests
    pytest tests/integration/test_embedding_connection.py -v

    # Run specific test
    pytest tests/integration/test_embedding_connection.py::TestEmbeddingConnection::test_basic_connection -v

    # Run with output
    pytest tests/integration/test_embedding_connection.py -v -s
"""

import pytest
import math

from backend.config import get_settings
from backend.infrastructure.embedding_client import EmbeddingClient


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0


@pytest.fixture
def embedding_config():
    """Get embedding configuration from settings."""
    settings = get_settings()
    return settings.embedding


@pytest.fixture
async def embedding_client(embedding_config):
    """Create and connect embedding client."""
    client = EmbeddingClient(embedding_config)
    await client.connect()
    yield client
    if client.is_connected():
        await client.close()


@pytest.mark.integration
class TestEmbeddingConnection:
    """Integration tests for embedding client connectivity."""

    @pytest.mark.asyncio
    async def test_basic_connection(self, embedding_config):
        """Test basic embedding client connection and disconnection."""
        client = EmbeddingClient(embedding_config)

        # Test connection
        await client.connect()
        assert client.is_connected()

        # Test disconnection
        await client.close()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_connection_check(self, embedding_client):
        """Test connection status check."""
        assert embedding_client.is_connected()

    @pytest.mark.asyncio
    async def test_config_applied(self, embedding_config, embedding_client):
        """Test that configuration is properly applied."""
        assert embedding_client.config == embedding_config
        assert embedding_client.config.embedding_model == embedding_config.embedding_model
        assert embedding_client.config.embedding_dimension == embedding_config.embedding_dimension


@pytest.mark.integration
class TestSingleTextEmbedding:
    """Integration tests for single text embedding."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedding_client, embedding_config):
        """Test embedding generation for single text."""
        test_text = "What is a database?"

        vector = await embedding_client.embed_text(test_text)

        assert vector is not None
        assert len(vector) == embedding_config.embedding_dimension
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_short_text(self, embedding_client, embedding_config):
        """Test embedding generation for very short text."""
        test_text = "Hi"

        vector = await embedding_client.embed_text(test_text)

        assert len(vector) == embedding_config.embedding_dimension

    @pytest.mark.asyncio
    async def test_embed_long_text(self, embedding_client, embedding_config):
        """Test embedding generation for longer text."""
        test_text = " ".join(["This is a test sentence."] * 50)

        vector = await embedding_client.embed_text(test_text)

        assert len(vector) == embedding_config.embedding_dimension


@pytest.mark.integration
class TestBatchEmbedding:
    """Integration tests for batch embedding."""

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_client, embedding_config):
        """Test batch embedding generation."""
        texts = [
            "Python is a programming language",
            "SQL is used for database queries",
            "A database stores data",
            "Python and SQL work together"
        ]

        vectors = await embedding_client.embed_batch(texts)

        assert len(vectors) == len(texts)
        assert all(len(v) == embedding_config.embedding_dimension for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_small_batch(self, embedding_client, embedding_config):
        """Test batch embedding with small number of texts."""
        texts = ["Text 1", "Text 2"]

        vectors = await embedding_client.embed_batch(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == embedding_config.embedding_dimension

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, embedding_client):
        """Test that batch embedding preserves input order."""
        texts = ["First", "Second", "Third"]

        vectors = await embedding_client.embed_batch(texts)

        # Generate individual embeddings to compare
        vector_1 = await embedding_client.embed_text(texts[0])
        vector_2 = await embedding_client.embed_text(texts[1])
        vector_3 = await embedding_client.embed_text(texts[2])

        # Vectors should be very similar (allowing for small API variations)
        assert cosine_similarity(vectors[0], vector_1) > 0.99
        assert cosine_similarity(vectors[1], vector_2) > 0.99
        assert cosine_similarity(vectors[2], vector_3) > 0.99


@pytest.mark.integration
class TestSemanticSimilarity:
    """Integration tests for semantic similarity using embeddings."""

    @pytest.mark.asyncio
    async def test_similar_texts_high_similarity(self, embedding_client):
        """Test that semantically similar texts have high cosine similarity."""
        text1 = "Python is a programming language"
        text2 = "Python is used for programming"

        vector1 = await embedding_client.embed_text(text1)
        vector2 = await embedding_client.embed_text(text2)

        similarity = cosine_similarity(vector1, vector2)
        assert similarity > 0.7  # Similar texts should have high similarity

    @pytest.mark.asyncio
    async def test_different_texts_lower_similarity(self, embedding_client):
        """Test that semantically different texts have lower similarity."""
        text1 = "Python is a programming language"
        text2 = "The weather is sunny today"

        vector1 = await embedding_client.embed_text(text1)
        vector2 = await embedding_client.embed_text(text2)

        similarity = cosine_similarity(vector1, vector2)
        assert similarity < 0.5  # Different texts should have lower similarity

    @pytest.mark.asyncio
    async def test_query_document_similarity(self, embedding_client):
        """Test query vs document similarity ranking."""
        query = "programming languages"
        documents = [
            "Python is a programming language",
            "SQL is a query language",
            "JavaScript is for web development"
        ]

        query_vector = await embedding_client.embed_text(query)
        doc_vectors = await embedding_client.embed_batch(documents)

        similarities = [
            cosine_similarity(query_vector, doc_vec)
            for doc_vec in doc_vectors
        ]

        # First document should be most similar to query
        assert similarities[0] > similarities[1]
        assert similarities[0] > similarities[2]

    @pytest.mark.asyncio
    async def test_pairwise_similarity_computation(self, embedding_client):
        """Test computing pairwise similarities in a batch."""
        texts = [
            "Python is a programming language",
            "SQL is used for database queries",
            "A database stores data",
            "Python and SQL work together"
        ]

        vectors = await embedding_client.embed_batch(texts)

        # Compute all pairwise similarities
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = cosine_similarity(vectors[i], vectors[j])
                similarities.append((i, j, similarity))

        # All similarities should be between 0 and 1
        assert all(0 <= sim <= 1 for _, _, sim in similarities)

        # At least some pairs should have reasonable similarity
        assert any(sim > 0.3 for _, _, sim in similarities)
