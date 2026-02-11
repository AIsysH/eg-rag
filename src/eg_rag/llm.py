"""
LLM integration module for answer generation.

Supports:
- OpenAI models (gpt-4o-mini, gpt-4o, gpt-4-turbo)
- Anthropic models (claude-sonnet-4-20250514, claude-3-5-sonnet)
- Local models via Ollama (mistral, llama, etc.)
"""

import os
import time
from typing import Optional


class LLMClient:
    """
    Unified LLM client for answer generation.

    Supports multiple providers:
    - OpenAI: gpt-4o-mini, gpt-4o, gpt-4-turbo
    - Anthropic: claude-sonnet-4-20250514, claude-3-5-sonnet-20241022
    - Local/Ollama: mistral:8b, llama3, etc.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        temperature: float = 0.0
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ("openai", "anthropic", "ollama")
            model: Model name (defaults based on provider)
            api_key: API key (uses env variable if None)
            base_url: Base URL for API (for Ollama: http://localhost:11434)
            max_retries: Maximum retry attempts for rate limits
            temperature: Sampling temperature
        """
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.temperature = temperature
        self.base_url = base_url

        # Set default models per provider
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "mistral:8b"
        }
        self.model = model or default_models.get(self.provider, "gpt-4o-mini")

        # Initialize based on provider
        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )

        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )

        elif self.provider == "ollama":
            self.base_url = base_url or "http://localhost:11434"
            self.client = None  # Uses requests directly

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, messages: list[dict]) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Generated response text or None if failed
        """
        if self.provider == "openai":
            return self._generate_openai(messages)
        elif self.provider == "anthropic":
            return self._generate_anthropic(messages)
        elif self.provider == "ollama":
            return self._generate_ollama(messages)

    def _generate_openai(self, messages: list[dict]) -> Optional[str]:
        """Generate using OpenAI API."""
        import openai

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except openai.RateLimitError as e:
                wait = 2 ** attempt
                print(f"RateLimitError: retrying in {wait}s...")
                time.sleep(wait)

            except openai.APIError as e:
                wait = 2 ** attempt
                print(f"APIError: retrying in {wait}s...")
                time.sleep(wait)

            except Exception as e:
                print(f"OpenAI Error: {e}")
                break

        return None

    def _generate_anthropic(self, messages: list[dict]) -> Optional[str]:
        """Generate using Anthropic API."""
        for attempt in range(self.max_retries):
            try:
                # Extract system message if present
                system = None
                user_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system = msg["content"]
                    else:
                        user_messages.append(msg)

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system or "You are a helpful assistant.",
                    messages=user_messages,
                    temperature=self.temperature,
                )
                return response.content[0].text.strip()

            except Exception as e:
                if "rate" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"RateLimitError: retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Anthropic Error: {e}")
                    break

        return None

    def _generate_ollama(self, messages: list[dict]) -> Optional[str]:
        """Generate using local Ollama API."""
        import requests

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": self.temperature}
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["message"]["content"].strip()

            except requests.exceptions.Timeout:
                wait = 2 ** attempt
                print(f"Timeout: retrying in {wait}s...")
                time.sleep(wait)

            except Exception as e:
                print(f"Ollama Error: {e}")
                break

        return None
