"""
LlamaIndex Adapter: TruthifyToolWrapper & Agent Governance
-----------------------------------------------------------
Provides tool wrappers and event interceptors for LlamaIndex ReActAgent and
FunctionTool workflows, capturing retrieval and function execution receipts.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from agent_honesty.actions.models import ExecutionIntegrityError
from agent_honesty.adapters.base import BaseFrameworkAdapter
from agent_honesty.interceptors.context_manager import HonestyAuditor
from agent_honesty.interceptors.tool_decorator import audit_tool
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


class TruthifyLlamaToolWrapper:
    """
    Wraps a LlamaIndex BaseTool or FunctionTool instance to intercept call / acall.
    """

    def __init__(self, tool: Any, adapter: "TruthifyLlamaAdapter", secret_key: Optional[str] = None) -> None:
        self._tool = tool
        self._adapter = adapter
        self.metadata = getattr(tool, "metadata", None)
        self.name = getattr(self.metadata, "name", getattr(tool, "name", "llama_tool"))
        self._audited_fn = audit_tool(name=self.name, secret_key=secret_key)(self._raw_call)
        self._audited_async_fn = audit_tool(name=self.name, secret_key=secret_key)(self._raw_acall)

    def _raw_call(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._tool, "call") and callable(getattr(self._tool, "call")):
            return self._tool.call(*args, **kwargs)
        elif callable(self._tool):
            return self._tool(*args, **kwargs)
        raise TypeError(f"Tool {self._tool} is not callable.")

    async def _raw_acall(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._tool, "acall") and callable(getattr(self._tool, "acall")):
            return await self._tool.acall(*args, **kwargs)
        return self._raw_call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        with HonestyAuditor() as auditor:
            result = self._audited_fn(*args, **kwargs)
            for r in auditor.receipts:
                self._adapter.record_receipt(r)
            return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        with HonestyAuditor() as auditor:
            result = await self._audited_async_fn(*args, **kwargs)
            for r in auditor.receipts:
                self._adapter.record_receipt(r)
            return result


class TruthifyLlamaAdapter(BaseFrameworkAdapter):
    """
    Adapter managing LlamaIndex tool wrapping and response evaluation.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        super().__init__(router=router, secret_key=secret_key)

    def wrap_tools(self, tools: Sequence[Any]) -> List[TruthifyLlamaToolWrapper]:
        """
        Wrap a sequence of LlamaIndex tools with execution integrity receipts.
        """
        wrapped: List[TruthifyLlamaToolWrapper] = []
        for tool in tools:
            wrapped.append(TruthifyLlamaToolWrapper(tool=tool, adapter=self, secret_key=self.secret_key))
        return wrapped


def wrap_llama_tools(
    tools: Sequence[Any],
    router: Optional[VerificationRouter] = None,
    secret_key: Optional[str] = None,
) -> Tuple[List[TruthifyLlamaToolWrapper], TruthifyLlamaAdapter]:
    """
    Convenience function returning (wrapped_tools, adapter).
    """
    adapter = TruthifyLlamaAdapter(router=router, secret_key=secret_key)
    wrapped = adapter.wrap_tools(tools)
    return wrapped, adapter
