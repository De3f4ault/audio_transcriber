"""Google Gemini REST client for LLM inference.

Communicates with Gemini API using google-genai.
"""

from __future__ import annotations

from collections.abc import Iterator

from audiobench.core.error_types import AudioBenchError
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("ai.gemini")


class AIError(AudioBenchError):
    """AI/LLM operation failure."""


class GeminiClient:
    """REST client for Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
    ) -> None:
        self.settings = get_settings()
        self._model = model
        self._client = None
        self._is_available = False

        try:
            from google import genai

            if self.settings.gemini_api_key:
                self._client = genai.Client(api_key=self.settings.gemini_api_key)
                self._is_available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._is_available

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        if not self._is_available:
            raise AIError(
                "Gemini not available", "google-genai SDK not installed or missing API key."
            )

        model_name = model or self._model

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        logger.info("Generating with %s (%.2f°C)", model_name, temperature)

        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            raise AIError("Gemini API call failed", str(e)) from e

    def stream(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Stream response tokens one at a time."""
        if not self._is_available:
            raise AIError(
                "Gemini not available", "google-genai SDK not installed or missing API key."
            )

        model_name = model or self._model

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        logger.info("Streaming with %s", model_name)

        try:
            response_stream = self._client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise AIError("Gemini API stream failed", str(e)) from e
