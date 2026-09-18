import abc
import hashlib
import json
from typing import Any, Dict, Optional

from agent_honesty.sandbox.models import StateDelta


class BaseSandboxEnvironment(abc.ABC):
    """
    Abstract interface for copy-on-write sandbox environments.
    """

    @abc.abstractmethod
    def enter_speculation(self) -> None:
        """Initialize the speculative fork / savepoint."""
        pass

    @abc.abstractmethod
    def compute_delta(self) -> StateDelta:
        """Compute the StateDelta generated since enter_speculation()."""
        pass

    @abc.abstractmethod
    def commit(self) -> None:
        """Persist changes to the live production state."""
        pass

    @abc.abstractmethod
    def rollback(self) -> None:
        """Revert all changes and restore original pre-speculation state."""
        pass

    @abc.abstractmethod
    def get_state_hash(self) -> str:
        """Compute a cryptographic hash of current state."""
        pass

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Utility helper to produce deterministic SHA-256 hex digest of a dict."""
        try:
            canonical = json.dumps(data, sort_keys=True, default=str)
        except Exception:
            canonical = str(data)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
