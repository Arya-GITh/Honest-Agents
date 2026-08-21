"""
AutoGen Adapter: TruthifyAgentInterceptor & ConversableAgent Governance
-----------------------------------------------------------------------
Provides pre/post-reply interceptors and tool execution hooks for AutoGen (AG2)
agents, verifying tool returns and message integrity across conversational turns.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agent_honesty.actions.models import ExecutionIntegrityError
from agent_honesty.adapters.base import BaseFrameworkAdapter
from agent_honesty.interceptors.context_manager import HonestyAuditor
from agent_honesty.interceptors.tool_decorator import audit_tool
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


class TruthifyAgentInterceptor(BaseFrameworkAdapter):
    """
    Hook and interceptor for Microsoft AutoGen / AG2 ConversableAgent instances.
    Registers tool wrappers and reply-verification filters.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        secret_key: Optional[str] = None,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(router=router, secret_key=secret_key)
        self.strict_mode = strict_mode

    def wrap_function(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Wrap a single tool function for AutoGen function_map and record its receipts.
        """
        audited_fn = audit_tool(name=name, secret_key=self.secret_key)(func)
        def audited_caller(*args: Any, **kwargs: Any) -> Any:
            with HonestyAuditor() as auditor:
                res = audited_fn(*args, **kwargs)
                for r in auditor.receipts:
                    self.record_receipt(r)
                return res
        audited_caller._is_truthify_audited = True  # type: ignore[attr-defined]
        return audited_caller

    def register(self, *agents: Any) -> None:
        """
        Register Truthify auditing hooks and wrap function maps on AutoGen agents.
        """
        for agent in agents:
            # 1. Wrap function map if present
            if hasattr(agent, "function_map") and isinstance(agent.function_map, dict):
                for name, func in list(agent.function_map.items()):
                    if not getattr(func, "_is_truthify_audited", False):
                        agent.function_map[name] = self.wrap_function(name, func)

            # 2. Register AutoGen hooks if supported
            if hasattr(agent, "register_hook") and callable(getattr(agent, "register_hook")):
                agent.register_hook(
                    hookable_method="process_message_before_send",
                    hook=self._process_message_before_send_hook,
                )

    def _process_message_before_send_hook(
        self,
        sender: Any,
        message: Union[Dict[str, Any], str],
        recipient: Any,
        silent: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        AutoGen hook executed before an agent sends a reply.
        Cross-examines claim against receipts if tools were executed.
        """
        claim_text = message.get("content", "") if isinstance(message, dict) else str(message)

        # Retrieve conversation history from recipient or sender
        chat_history = getattr(recipient, "chat_messages", {}).get(sender, [])
        user_prompt = ""
        for msg in chat_history:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

        if self.receipts and claim_text:
            verdict = self.verify_claim(
                user_prompt=user_prompt,
                agent_claim=claim_text,
                receipts=self.receipts,
            )

            if not verdict.is_honest:
                if self.strict_mode:
                    raise ExecutionIntegrityError(
                        f"AutoGen execution integrity violation: {verdict.explanation}",
                        verdict=verdict,
                    )
                # Append honesty correction to content
                correction_note = f"\n\n[Truthify Integrity Notice: Tool execution returned non-success. Ground Truth: {verdict.explanation}]"
                if isinstance(message, dict):
                    message["content"] = message.get("content", "") + correction_note
                else:
                    message = str(message) + correction_note

        return message
