"""
LLM Client — Single wrapper around the local Ollama gateway.

Uses Ollama's OpenAI-compatible API with the ``gemma2:9b`` model
(``base_url=http://localhost:11434/v1``).

Provides two call types:
  1. ``extract`` — structured extraction via ``with_structured_output``
     (Component 2).  Cached by ``(source_id, prompt_version)``.
  2. ``explain`` — free-text explanation generation (Component 8).
     Not cached (each request gets a unique explanation).

All calls are logged through the shared ``UsageTracker`` instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .usage_tracker import UsageTracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Bump this whenever the extraction prompt changes to invalidate cache.
PROMPT_VERSION = "v1"


class LLMClient:
    """
    Singleton-style LLM client wrapping the local Ollama gateway.
    """

    def __init__(
        self,
        model: str = "gemma2:9b",
        temperature: float = 0.0,
    ) -> None:
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
        self._cache: Dict[Tuple[str, str], Any] = {}
        self.tracker = UsageTracker(model=model)

    # ------------------------------------------------------------------
    # Extraction (Component 2)
    # ------------------------------------------------------------------

    def extract(
        self,
        messages: List[BaseMessage],
        schema: Type[T],
        target_id: str,
        cache_key: Optional[Tuple[str, str]] = None,
    ) -> Optional[T]:
        """
        Structured extraction LLM call.

        Parameters
        ----------
        messages:
            Chat messages to send (system + human).
        schema:
            Pydantic model class for structured output parsing.
        target_id:
            The ``message_id`` or ``image_id`` being processed.
        cache_key:
            Optional ``(source_id, prompt_version)`` tuple.  When provided,
            results are cached so reruns skip reprocessing.

        Returns
        -------
        Parsed Pydantic model instance, or ``None`` on parsing failure.
        """
        if cache_key and cache_key in self._cache:
            logger.debug("Cache hit for extraction: %s", cache_key)
            return self._cache[cache_key]

        try:
            structured_llm = self.llm.with_structured_output(
                schema, include_raw=True
            )
            result = structured_llm.invoke(messages)
        except Exception:
            logger.exception(
                "LLM extraction call failed for target_id=%s", target_id
            )
            return None

        raw_message = result.get("raw")
        parsed: Optional[T] = result.get("parsed")
        parsing_error = result.get("parsing_error")

        if parsing_error is not None:
            logger.warning(
                "Structured output parsing error for %s: %s",
                target_id,
                parsing_error,
            )

        # Extract token usage from the raw AIMessage.
        input_tokens = 0
        output_tokens = 0
        if raw_message is not None:
            usage = getattr(raw_message, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        self.tracker.log_call(
            call_type="extraction",
            target_id=target_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if cache_key and parsed is not None:
            self._cache[cache_key] = parsed
            logger.debug("Cached extraction result: %s", cache_key)

        return parsed

    # ------------------------------------------------------------------
    # Explanation (Component 8)
    # ------------------------------------------------------------------

    def explain(
        self,
        messages: List[BaseMessage],
        target_id: str,
    ) -> str:
        """
        Explanation generation LLM call.

        Parameters
        ----------
        messages:
            Chat messages to send (system + human with structured decision summary).
        target_id:
            The ``request_id`` being explained.

        Returns
        -------
        The explanation text string.
        """
        try:
            result = self.llm.invoke(messages)
        except Exception:
            logger.exception(
                "LLM explanation call failed for target_id=%s", target_id
            )
            return "Unable to generate explanation."

        input_tokens = 0
        output_tokens = 0
        usage = getattr(result, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        self.tracker.log_call(
            call_type="explanation",
            target_id=target_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result.content

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached extraction results."""
        return len(self._cache)
