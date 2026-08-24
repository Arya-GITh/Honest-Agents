"""
CrewAI Adapter: TruthifyCrewCallback & Multi-Agent Delegation Governance
------------------------------------------------------------------------
Provides lifecycle hooks and task callbacks for CrewAI multi-agent crews,
preventing false-success hallucinations and tool errors from propagating across
agent delegations.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from agent_honesty.actions.models import ExecutionIntegrityError
from agent_honesty.adapters.base import BaseFrameworkAdapter
from agent_honesty.interceptors.context_manager import HonestyAuditor
from agent_honesty.interceptors.tool_decorator import audit_tool
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


def wrap_crew_tool(
    tool: Any,
    name: Optional[str] = None,
    secret_key: Optional[str] = None,
    callback: Optional["TruthifyCrewCallback"] = None,
) -> Any:
    """
    Wrap a CrewAI BaseTool or custom tool instance with @audit_tool interception.
    """
    tool_name = name or getattr(tool, "name", getattr(tool, "__name__", str(tool)))

    if hasattr(tool, "_run") and callable(getattr(tool, "_run")):
        orig_run = tool._run
        audited_fn = audit_tool(name=tool_name, secret_key=secret_key)(orig_run)
        def run_with_audit(*args: Any, **kwargs: Any) -> Any:
            with HonestyAuditor() as auditor:
                res = audited_fn(*args, **kwargs)
                if callback:
                    for r in auditor.receipts:
                        callback.record_receipt(r)
                return res
        setattr(tool, "_run", run_with_audit)
        return tool

    if callable(tool):
        audited_fn = audit_tool(name=tool_name, secret_key=secret_key)(tool)
        def call_with_audit(*args: Any, **kwargs: Any) -> Any:
            with HonestyAuditor() as auditor:
                res = audited_fn(*args, **kwargs)
                if callback:
                    for r in auditor.receipts:
                        callback.record_receipt(r)
                return res
        return call_with_audit

    return tool


class TruthifyCrewCallback(BaseFrameworkAdapter):
    """
    Callback handler for CrewAI agents and tasks.
    Can be passed directly into `step_callback` and `task_callback` of a `Crew` or `Agent`.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        secret_key: Optional[str] = None,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(router=router, secret_key=secret_key)
        self.strict_mode = strict_mode
        self.step_history: List[Dict[str, Any]] = []

    def wrap_tool(self, tool: Any, name: Optional[str] = None) -> Any:
        """Convenience method to wrap a tool and register it directly with this callback."""
        return wrap_crew_tool(tool=tool, name=name, secret_key=self.secret_key, callback=self)

    def __call__(self, step_output: Any) -> None:
        """Called by CrewAI on step_callback."""
        self.on_step(step_output)

    def on_step(self, step_output: Any) -> None:
        """
        Record agent step output and track intermediate tool outputs.
        """
        step_dict = {
            "output": getattr(step_output, "text", str(step_output)),
            "tool": getattr(step_output, "tool", None),
            "tool_input": getattr(step_output, "tool_input", None),
        }
        self.step_history.append(step_dict)

    def on_task_completed(self, task_output: Any, task_description: Optional[str] = None) -> VerificationVerdict:
        """
        Evaluate task output against all receipts generated during the crew's execution.
        """
        claim_text = getattr(task_output, "raw", str(task_output))
        prompt_text = task_description or ""

        verdict = self.verify_claim(
            user_prompt=prompt_text,
            agent_claim=claim_text,
            receipts=self.receipts,
        )

        if not verdict.is_honest and self.strict_mode:
            raise ExecutionIntegrityError(
                f"CrewAI execution integrity violation detected in task output: {verdict.explanation}",
                verdict=verdict,
            )

        return verdict
