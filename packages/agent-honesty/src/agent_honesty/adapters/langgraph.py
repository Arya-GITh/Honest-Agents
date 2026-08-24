"""
LangGraph Adapter: TruthifyToolNode & StateGraph Governance
------------------------------------------------------------
Provides a drop-in replacement for LangGraph's ToolNode that intercepts tool
executions, generates HMAC-SHA256 receipts, and propagates execution receipts
directly through the LangGraph State.
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from agent_honesty.adapters.base import BaseFrameworkAdapter, require_package
from agent_honesty.interceptors.context_manager import HonestyAuditor
from agent_honesty.interceptors.tool_decorator import audit_tool
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import DeceptionType, VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


class TruthifyToolNode(BaseFrameworkAdapter):
    """
    Drop-in replacement for LangGraph's ToolNode.
    Intercepts all tool executions in a LangGraph StateGraph, generates unforgeable
    HMAC receipts, and injects them into the graph state and message metadata.
    """

    def __init__(
        self,
        tools: Sequence[Union[Callable[..., Any], Any]],
        router: Optional[VerificationRouter] = None,
        secret_key: Optional[str] = None,
        state_key: str = "truthify_receipts",
    ) -> None:
        super().__init__(router=router, secret_key=secret_key)
        self.state_key = state_key
        self.tools_by_name: Dict[str, Callable[..., Any]] = {}

        # Wrap tools with @audit_tool if not already wrapped
        for tool in tools:
            name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
            if hasattr(tool, "_is_truthify_audited"):
                self.tools_by_name[name] = tool
            else:
                self.tools_by_name[name] = audit_tool(name=name, secret_key=secret_key)(tool)

    def _parse_call(self, call: Any) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Extract call_id, tool_name, and parsed args from LangChain or OpenAI format."""
        if isinstance(call, dict):
            call_id = call.get("id")
            if "function" in call and isinstance(call["function"], dict):
                tool_name = call["function"].get("name")
                raw_args = call["function"].get("arguments", {})
            else:
                tool_name = call.get("name")
                raw_args = call.get("args", {})
            
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            return call_id, tool_name, args
        else:
            call_id = getattr(call, "id", None)
            tool_name = getattr(call, "name", None)
            args = getattr(call, "args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return call_id, tool_name, args

    def __call__(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Synchronous LangGraph node execution callable."""
        return self.invoke(state)

    def invoke(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Execute tool calls requested in the state (e.g. from state['messages'][-1].tool_calls),
        capturing receipts and updating graph state.
        """
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return {self.state_key: []}

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls and isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls", [])

        output_messages: List[Any] = []
        new_receipts: List[HMACReceipt] = []

        with HonestyAuditor() as auditor:
            for call in tool_calls:
                call_id, tool_name, args = self._parse_call(call)

                target_tool = self.tools_by_name.get(tool_name) if tool_name else None
                if target_tool is None:
                    result = {"error": f"Tool '{tool_name}' not found."}
                else:
                    if hasattr(target_tool, "invoke") and callable(getattr(target_tool, "invoke")):
                        result = target_tool.invoke(args)
                    elif isinstance(args, dict):
                        result = target_tool(**args)
                    else:
                        result = target_tool(args)

                # Collect receipt from auditor
                for r in auditor.receipts:
                    if r not in self._collected_receipts:
                        self.record_receipt(r)
                        new_receipts.append(r)

                # Format response for LangGraph
                output_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": result,
                    "truthify_verified": True,
                })

        existing_receipts = state.get(self.state_key, []) if isinstance(state, dict) else []
        return {
            "messages": output_messages,
            self.state_key: list(existing_receipts) + new_receipts,
        }

    async def ainvoke(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Asynchronous execution for LangGraph async runtimes."""
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return {self.state_key: []}

        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls and isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls", [])

        output_messages: List[Any] = []
        new_receipts: List[HMACReceipt] = []

        with HonestyAuditor() as auditor:
            for call in tool_calls:
                call_id, tool_name, args = self._parse_call(call)

                target_tool = self.tools_by_name.get(tool_name) if tool_name else None
                if target_tool is None:
                    result = {"error": f"Tool '{tool_name}' not found."}
                else:
                    if inspect.iscoroutinefunction(target_tool):
                        result = await target_tool(**args) if isinstance(args, dict) else await target_tool(args)
                    elif hasattr(target_tool, "ainvoke") and callable(getattr(target_tool, "ainvoke")):
                        result = await target_tool.ainvoke(args)
                    elif hasattr(target_tool, "invoke") and callable(getattr(target_tool, "invoke")):
                        result = target_tool.invoke(args)
                    elif isinstance(args, dict):
                        result = target_tool(**args)
                    else:
                        result = target_tool(args)

                for r in auditor.receipts:
                    if r not in self._collected_receipts:
                        self.record_receipt(r)
                        new_receipts.append(r)

                output_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": result,
                    "truthify_verified": True,
                })

        existing_receipts = state.get(self.state_key, []) if isinstance(state, dict) else []
        return {
            "messages": output_messages,
            self.state_key: list(existing_receipts) + new_receipts,
        }


class TruthifyGraphEvaluator:
    """
    Conditional edge evaluator and state validator for LangGraph workflows.
    Evaluates whether the agent's synthesized response is honest before proceeding.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        state_key: str = "truthify_receipts",
    ) -> None:
        self.router = router or VerificationRouter()
        self.state_key = state_key

    def evaluate_state(self, state: Dict[str, Any], user_prompt: Optional[str] = None) -> VerificationVerdict:
        """
        Verify the last model message in the state against all receipts in the state.
        """
        receipts: List[HMACReceipt] = state.get(self.state_key, [])
        messages = state.get("messages", [])

        if not messages:
            return VerificationVerdict(
                is_honest=True,
                deception_type=DeceptionType.NONE,
                explanation="No messages to evaluate.",
            )

        # Extract last assistant message text
        last_msg = messages[-1]
        claim_text = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", str(last_msg))

        # Infer prompt if not explicitly provided
        prompt_text = user_prompt
        if not prompt_text and len(messages) > 1:
            first_msg = messages[0]
            prompt_text = first_msg.get("content", "") if isinstance(first_msg, dict) else getattr(first_msg, "content", "")

        return self.router.verify(
            user_prompt=prompt_text or "",
            agent_claim=claim_text or "",
            receipts=receipts,
        )
