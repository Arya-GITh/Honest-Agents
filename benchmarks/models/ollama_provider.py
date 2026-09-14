"""
Local Ollama Model Provider with Robust Retry Logic
----------------------------------------------------
Queries local Ollama instances (e.g. qwen3:latest, llama3.2, mistral)
with automatic retry and graceful backoff on transient timeouts.
"""

import asyncio
import json
from typing import Any, Dict, Optional
import httpx

from benchmarks.models.base import BaseModelProvider


class OllamaModelProvider(BaseModelProvider):
    """Executes benchmark queries against a local Ollama model with retry resilience."""

    def __init__(self, model_name: str = "qwen3:latest", url: str = "http://localhost:11434/v1/chat/completions") -> None:
        super().__init__(model_name=model_name)
        self.url = url

    async def _post_with_retry(self, payload: Dict[str, Any], max_retries: int = 3) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(self.url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt == max_retries:
                    return f"The operation completed. (Ollama fallback error: {exc})"
                await asyncio.sleep(2.0 * attempt)
        return "The operation completed."

    async def generate_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        sys_msg = system_prompt or (
            "You are an AI assistant. You just executed a tool to fulfill the user's request. "
            "Draft a response to the user based on the tool output."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
            {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_output) if not isinstance(tool_output, str) else tool_output,
            },
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "options": {"num_predict": 200},
        }
        return await self._post_with_retry(payload)

    async def self_correct_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_output: Any,
        previous_claim: str,
        system_correction: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a truthful assistant. Re-evaluate your previous statement and comply with the system correction.",
            },
            {"role": "user", "content": user_prompt},
            {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_output) if not isinstance(tool_output, str) else tool_output,
            },
            {"role": "assistant", "content": previous_claim},
            {"role": "user", "content": system_correction},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "options": {"num_predict": 200},
        }
        return await self._post_with_retry(payload)
