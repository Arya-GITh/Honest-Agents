"""
Abstract Model Provider Interface
---------------------------------
Standard interface for AI model endpoints evaluated by DeceptionBench.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModelProvider(ABC):
    """Abstract interface for querying models in DeceptionBench."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    async def generate_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Given the user prompt and tool execution return, model synthesizes its response.
        """
        pass

    @abstractmethod
    async def self_correct_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_output: Any,
        previous_claim: str,
        system_correction: str,
    ) -> str:
        """
        Given private scratchpad feedback, model re-evaluates and drafts a corrected response.
        """
        pass
