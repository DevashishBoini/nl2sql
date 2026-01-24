"""
LLM client for OpenRouter using LangChain.

This module provides an async LLM client that uses LangChain's ChatOpenAI
with OpenRouter API for text generation tasks.
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import SecretStr

from ..config import LLMConfig
from ..utils.logging import get_module_logger
from ..utils.tracing import current_trace_id
from ..utils.token_utils import InputValidator
from ..domain.errors import LLMError


logger = get_module_logger()


class LLMClient:
    """
    LLM client using LangChain's ChatOpenAI with OpenRouter.

    This is a thin infrastructure layer for LLM operations using OpenRouter API.
    Higher-level prompt engineering and chain orchestration should be
    implemented in Service layers.

    Features:
    - OpenRouter API integration via LangChain
    - Support for multiple models (Claude, GPT-4, etc.)
    - Configurable temperature, top_p, max_tokens
    - Structured logging with trace IDs
    - Automatic retry on transient failures
    - Input size validation against model context windows

    Usage:
        client = LLMClient(config)
        await client.connect()

        # Simple text generation
        response = await client.generate("What is the capital of France?")

        # With system prompt
        response = await client.generate(
            "List 3 French cheeses",
            system_prompt="You are a French cuisine expert."
        )

        await client.close()
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with configuration.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._llm: Optional[ChatOpenAI] = None
        self._is_connected = False

        logger.info(
            "LLMClient initialized",
            default_model=config.default_model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    async def connect(self) -> None:
        """
        Initialize LangChain ChatOpenAI client.

        Note: This creates the client configuration but doesn't make any API calls.
        Validation happens on first actual use.

        Raises:
            LLMError: If initialization fails
        """
        if self._is_connected:
            logger.warning("LLM client already connected")
            return

        trace_id = current_trace_id()
        logger.info("Initializing LLM client", trace_id=trace_id)

        try:
            # Create LangChain ChatOpenAI client configured for OpenRouter
            # Pass standard OpenAI parameters explicitly (not via model_kwargs)
            # Note: Use max_completion_tokens (not deprecated max_tokens)
            self._llm = ChatOpenAI(
                model=self.config.default_model,
                api_key=SecretStr(self.config.openrouter_api_key),
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries
            )

            self._is_connected = True
            logger.info("LLM client initialized successfully", trace_id=trace_id)

        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {e}"
            logger.error(error_msg, error_type=type(e).__name__, trace_id=trace_id)
            raise LLMError(error_msg) from e

    async def close(self) -> None:
        """Close LLM client and release resources."""
        trace_id = current_trace_id()
        logger.info("Closing LLM client", trace_id=trace_id)

        # LangChain ChatOpenAI doesn't need explicit cleanup
        self._is_connected = False
        self._llm = None

        logger.info("LLM client closed", trace_id=trace_id)

    def is_connected(self) -> bool:
        """Check if LLM client is connected."""
        return self._is_connected and self._llm is not None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate text response from LLM.

        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override (0.0-1.0)
            max_tokens: Optional max tokens override
            model: Optional model override

        Returns:
            Generated text response

        Raises:
            LLMError: If generation fails or input exceeds context window

        Example:
            # Simple generation
            response = await client.generate("What is Python?")

            # With system prompt
            response = await client.generate(
                "Write a haiku about databases",
                system_prompt="You are a technical poet."
            )

            # Override settings
            response = await client.generate(
                "Tell me a story",
                temperature=0.8,
                max_tokens=500
            )
        """
        if not self.is_connected():
            raise LLMError("LLM client is not connected")

        # Validate input character limit before API call
        try:
            InputValidator.validate_total_chars(
                prompt=prompt,
                system_prompt=system_prompt,
                max_chars=self.config.max_input_chars
            )
        except ValueError as e:
            raise LLMError(str(e)) from e

        trace_id = current_trace_id()

        logger.info(
            "Generating LLM response",
            prompt_length=len(prompt),
            system_prompt_length=len(system_prompt) if system_prompt else 0,
            max_input_chars=self.config.max_input_chars,
            has_system_prompt=system_prompt is not None,
            temperature=temperature or self.config.temperature,
            trace_id=trace_id
        )

        try:
            if not self._llm:
                raise LLMError("LLM client not initialized")

            # Build messages
            messages: List[BaseMessage] = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # Use existing client or create new one only if model/params change
            llm = self._llm

            # If any parameters need override, use bind() to create modified client
            if model is not None or temperature is not None or max_tokens is not None:
                bind_kwargs = {}
                if model is not None:
                    bind_kwargs["model"] = model
                if temperature is not None:
                    bind_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    bind_kwargs["max_completion_tokens"] = max_tokens
                llm = llm.bind(**bind_kwargs)

            # Generate response
            response = await llm.ainvoke(messages)

            if not response or not response.content:
                raise LLMError("LLM returned empty response")

            content = str(response.content)

            logger.info(
                "LLM response generated successfully",
                response_length=len(content),
                trace_id=trace_id
            )

            return content

        except Exception as e:
            error_msg = f"LLM generation failed: {e}"
            logger.error(
                error_msg,
                error_type=type(e).__name__,
                prompt_length=len(prompt),
                trace_id=trace_id
            )
            raise LLMError(error_msg) from e

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch.

        Automatically chunks large batches based on config.batch_size to avoid
        rate limits and timeouts. Processes chunks sequentially.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt for all prompts
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            List of generated responses (same order as prompts)

        Raises:
            LLMError: If batch generation fails or any prompt exceeds context window

        Example:
            prompts = [
                "What is Python?",
                "What is SQL?",
                "What is a database?"
            ]
            responses = await client.generate_batch(prompts)
        """
        if not self.is_connected():
            raise LLMError("LLM client is not connected")

        # Validate all prompts before API calls
        for i, prompt in enumerate(prompts):
            try:
                InputValidator.validate_total_chars(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_chars=self.config.max_input_chars
                )
            except ValueError as e:
                raise LLMError(f"Prompt at index {i}: {str(e)}") from e

        trace_id = current_trace_id()

        logger.info(
            "Generating batch LLM responses",
            num_prompts=len(prompts),
            batch_size=self.config.batch_size,
            max_input_chars=self.config.max_input_chars,
            trace_id=trace_id
        )

        try:
            if not self._llm:
                raise LLMError("LLM client not initialized")

            # Chunk prompts into batches based on config.batch_size
            all_results: List[str] = []
            for i in range(0, len(prompts), self.config.batch_size):
                chunk_prompts = prompts[i:i + self.config.batch_size]

                logger.info(
                    "Processing batch chunk",
                    chunk_index=i // self.config.batch_size + 1,
                    chunk_size=len(chunk_prompts),
                    trace_id=trace_id
                )

                # Build message batches (create system message once, reuse for all prompts)
                base_messages = [SystemMessage(content=system_prompt)] if system_prompt else []
                message_batches = [
                    base_messages + [HumanMessage(content=prompt)]
                    for prompt in chunk_prompts
                ]

                # Always use existing client from connect()
                # Note: LangChain's abatch doesn't support runtime parameter overrides
                # If overrides are needed, we bind them to the model
                llm = self._llm

                # Apply parameter overrides using bind if specified
                if temperature is not None or max_tokens is not None:
                    bind_kwargs = {}
                    if temperature is not None:
                        bind_kwargs["temperature"] = temperature
                    if max_tokens is not None:
                        bind_kwargs["max_completion_tokens"] = max_tokens
                    llm = llm.bind(**bind_kwargs)

                # Batch generate
                # Note: Type checker complains about List[List[BaseMessage]] vs LanguageModelInput
                # but this is the correct usage per LangChain docs
                responses = await llm.abatch(message_batches)  # type: ignore[arg-type]

                # Extract content
                chunk_results = [str(resp.content) for resp in responses]
                all_results.extend(chunk_results)

            logger.info(
                "Batch LLM responses generated successfully",
                num_responses=len(all_results),
                num_chunks=len(range(0, len(prompts), self.config.batch_size)),
                trace_id=trace_id
            )

            return all_results

        except Exception as e:
            error_msg = f"Batch LLM generation failed: {e}"
            logger.error(
                error_msg,
                error_type=type(e).__name__,
                num_prompts=len(prompts),
                trace_id=trace_id
            )
            raise LLMError(error_msg) from e
