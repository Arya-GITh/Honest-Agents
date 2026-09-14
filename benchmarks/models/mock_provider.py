"""
Mock Model Provider for Testing
-------------------------------
Generates deterministic honest or deceptive responses for test suites.
"""

from typing import Any, Dict, Optional

from benchmarks.models.base import BaseModelProvider


class MockModelProvider(BaseModelProvider):
    """Generates synthetic responses with configurable deception rates."""

    def __init__(self, model_name: str = "mock-model", deception_rate: float = 0.5) -> None:
        super().__init__(model_name=model_name)
        self.deception_rate = deception_rate
        self.call_count = 0

    async def generate_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        self.call_count += 1
        
        # Check if this execution is a failure case
        is_error = False
        if isinstance(tool_output, dict):
            is_error = tool_output.get("status") == "error" or tool_output.get("is_error") is True or "error_code" in tool_output

        # Simulate deception based on rate
        should_deceive = (self.call_count % 2 == 0) if self.deception_rate > 0.0 else False

        if is_error and should_deceive:
            return f"Your request has been successfully processed and confirmed with zero errors!"
        elif is_error:
            err_msg = tool_output.get("message", "unknown error") if isinstance(tool_output, dict) else str(tool_output)
            return f"The operation failed due to an error: {err_msg}."
        else:
            return f"The operation was successfully completed."

    async def self_correct_response(
        self,
        user_prompt: str,
        tool_name: str,
        tool_output: Any,
        previous_claim: str,
        system_correction: str,
    ) -> str:
        err_msg = tool_output.get("message", "an error") if isinstance(tool_output, dict) else str(tool_output)
        return f"Upon review, the operation failed due to {err_msg}. No actions were completed."
