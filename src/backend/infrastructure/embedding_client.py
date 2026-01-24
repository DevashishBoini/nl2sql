
"""
Embedding client for OpenRouter using LangChain.

This module provides an async embedding client that uses LangChain's OpenAIEmbeddings
with OpenRouter API for text embedding tasks.
"""

from typing import List, Optional
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings

from ..config import EmbeddingConfig
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..utils.token_utils import InputValidator
from ..domain.errors import EmbeddingError


logger = get_module_logger()


class EmbeddingClient:
    """
    Embedding client using LangChain's OpenAIEmbeddings with OpenRouter.

    This is a thin infrastructure layer for embedding operations using OpenRouter API.
    Higher-level vector operations and indexing should be implemented in
    Repository or Service layers.

    Features:
    - OpenRouter API integration via LangChain
    - Support for multiple embedding models (OpenAI, etc.)
    - Batch embedding support
    - Configurable dimensions
    - Structured logging with trace IDs
    - Automatic retry on transient failures
    - Input size validation against model token limits

    Usage:
        client = EmbeddingClient(config)
        await client.connect()

        # Embed single text
        vector = await client.embed_text("What is a database?")

        # Embed multiple texts
        vectors = await client.embed_batch([
            "What is Python?",
            "What is SQL?"
        ])

        await client.close()
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding client with configuration.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._is_connected = False

        logger.info(
            "EmbeddingClient initialized",
            embedding_model=config.embedding_model,
            base_url=config.base_url,
            embedding_dimension=config.embedding_dimension,
            batch_size=config.batch_size
        )

    async def connect(self) -> None:
        """
        Initialize LangChain OpenAIEmbeddings client.

        Note: This creates the client configuration but doesn't make any API calls.
        Validation happens on first actual use.

        Raises:
            EmbeddingError: If initialization fails
        """
        if self._is_connected:
            logger.warning("Embedding client already connected")
            return

        trace_id = current_trace_id()
        logger.info("Initializing embedding client", trace_id=trace_id)

        try:
            # Create LangChain OpenAIEmbeddings client configured for OpenRouter
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=SecretStr(self.config.openrouter_api_key),
                base_url=self.config.base_url,
                dimensions=self.config.embedding_dimension,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                chunk_size=self.config.batch_size
            )

            self._is_connected = True
            logger.info("Embedding client initialized successfully", trace_id=trace_id)

        except Exception as e:
            error_msg = f"Failed to initialize embedding client: {e}"
            logger.error(error_msg, error_type=type(e).__name__, trace_id=trace_id)
            raise EmbeddingError(error_msg) from e

    async def close(self) -> None:
        """Close embedding client and release resources."""
        trace_id = current_trace_id()
        logger.info("Closing embedding client", trace_id=trace_id)

        # LangChain OpenAIEmbeddings doesn't need explicit cleanup
        self._is_connected = False
        self._embeddings = None

        logger.info("Embedding client closed", trace_id=trace_id)

    def is_connected(self) -> bool:
        """Check if embedding client is connected."""
        return self._is_connected and self._embeddings is not None

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails or text exceeds token limit

        Example:
            vector = await client.embed_text("What is a database?")
            print(f"Vector dimension: {len(vector)}")
        """
        if not self.is_connected():
            raise EmbeddingError("Embedding client is not connected")

        # Validate character limit before API call
        try:
            InputValidator.validate_char_limit(text, max_chars=self.config.max_input_chars)
        except ValueError as e:
            raise EmbeddingError(str(e)) from e

        trace_id = current_trace_id()

        logger.info(
            "Generating embedding for text",
            text_length=len(text),
            max_input_chars=self.config.max_input_chars,
            trace_id=trace_id
        )

        try:
            if not self._embeddings:
                raise EmbeddingError("Embedding client not initialized")

            # Generate embedding
            vector = await self._embeddings.aembed_query(text)

            if not vector or len(vector) != self.config.embedding_dimension:
                raise EmbeddingError(
                    f"Invalid embedding dimension: expected {self.config.embedding_dimension}, "
                    f"got {len(vector) if vector else 0}"
                )

            logger.info(
                "Embedding generated successfully",
                dimension=len(vector),
                trace_id=trace_id
            )

            return vector

        except Exception as e:
            error_msg = f"Embedding generation failed: {e}"
            logger.error(
                error_msg,
                error_type=type(e).__name__,
                text_length=len(text),
                trace_id=trace_id
            )
            raise EmbeddingError(error_msg) from e

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in batch.

        Note: LangChain's OpenAIEmbeddings automatically chunks based on
        config.batch_size (chunk_size parameter). Chunks are processed
        SEQUENTIALLY to respect rate limits. Within each chunk, the API
        provider handles concurrency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input texts)

        Raises:
            EmbeddingError: If batch embedding fails or any text exceeds token limit

        Example:
            texts = [
                "What is Python?",
                "What is SQL?",
                "What is a database?"
            ]
            vectors = await client.embed_batch(texts)
            print(f"Generated {len(vectors)} vectors")
        """
        if not self.is_connected():
            raise EmbeddingError("Embedding client is not connected")

        # Validate all texts before API call
        try:
            InputValidator.validate_batch_chars(
                texts,
                max_chars_per_text=self.config.max_input_chars,
                error_message="Text in batch"
            )
        except ValueError as e:
            raise EmbeddingError(str(e)) from e

        trace_id = current_trace_id()

        logger.info(
            "Generating batch embeddings",
            num_texts=len(texts),
            batch_size=self.config.batch_size,
            trace_id=trace_id
        )

        try:
            if not self._embeddings:
                raise EmbeddingError("Embedding client not initialized")

            # Generate embeddings in batch
            # Note: LangChain automatically chunks into batches of config.batch_size
            # and processes them sequentially
            vectors = await self._embeddings.aembed_documents(texts)

            # Validate dimensions
            for i, vector in enumerate(vectors):
                if not vector or len(vector) != self.config.embedding_dimension:
                    raise EmbeddingError(
                        f"Invalid embedding dimension at index {i}: "
                        f"expected {self.config.embedding_dimension}, "
                        f"got {len(vector) if vector else 0}"
                    )

            logger.info(
                "Batch embeddings generated successfully",
                num_vectors=len(vectors),
                dimension=len(vectors[0]) if vectors else 0,
                trace_id=trace_id
            )

            return vectors

        except Exception as e:
            error_msg = f"Batch embedding generation failed: {e}"
            logger.error(
                error_msg,
                error_type=type(e).__name__,
                num_texts=len(texts),
                trace_id=trace_id
            )
            raise EmbeddingError(error_msg) from e
