"""
Base Framework Adapter Architecture
-----------------------------------
Provides lazy-import validation, common framework execution state tracking,
and receipt bridging across LangGraph, CrewAI, AutoGen, and LlamaIndex.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


def require_package(package_name: str, extra_name: str, import_name: Optional[str] = None) -> Any:
    """
    Safely import an optional framework dependency.
    If the package is not installed, raises a user-friendly ImportError with installation instructions.
    """
    target = import_name or package_name
    try:
        return importlib.import_module(target)
    except ImportError as e:
        raise ImportError(
            f"The '{package_name}' framework is required to use this Truthify adapter.\n"
            f"👉 Please install it with: pip install \"agent-honesty[{extra_name}]\""
        ) from e


class BaseFrameworkAdapter(ABC):
    """
    Abstract base class for all framework adapters.
    Manages receipt collectors, verification routing, and state injection.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.router = router or VerificationRouter()
        self.secret_key = secret_key
        self._collected_receipts: List[HMACReceipt] = []

    @property
    def receipts(self) -> List[HMACReceipt]:
        """Return all HMAC receipts collected across the framework execution cycle."""
        return list(self._collected_receipts)

    def record_receipt(self, receipt: HMACReceipt) -> None:
        """Record an HMAC receipt from an audited tool execution."""
        self._collected_receipts.append(receipt)

    def clear_receipts(self) -> None:
        """Clear the collected receipts buffer."""
        self._collected_receipts.clear()

    def verify_claim(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: Optional[List[HMACReceipt]] = None,
    ) -> VerificationVerdict:
        """
        Cross-examine an agent's claim against collected execution receipts.
        """
        active_receipts = receipts if receipts is not None else self._collected_receipts
        return self.router.verify(
            user_prompt=user_prompt,
            agent_claim=agent_claim,
            receipts=active_receipts,
        )
