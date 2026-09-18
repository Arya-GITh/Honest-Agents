import copy
from typing import Any, Dict, List, Optional

from agent_honesty.sandbox.environments.base import BaseSandboxEnvironment
from agent_honesty.sandbox.models import StateDelta


class DictStateSandbox(BaseSandboxEnvironment):
    """
    In-memory state sandbox for dictionary-backed databases, cache stores, and state registries.
    """

    def __init__(self, state_dict: Dict[str, Any]) -> None:
        self.state_dict = state_dict
        self._pre_state: Dict[str, Any] = {}
        self._pre_hash: str = ""

    def get_state_hash(self) -> str:
        """Compute SHA-256 fingerprint of the current dictionary state."""
        return self._hash_dict(self.state_dict)

    def enter_speculation(self) -> None:
        """Deep-copy state dictionary to isolate speculative modifications."""
        self._pre_state = copy.deepcopy(self.state_dict)
        self._pre_hash = self.get_state_hash()

    def compute_delta(self) -> StateDelta:
        """Inspect key-level modifications, insertions, deletions, and financial balance shifts."""
        current = self.state_dict
        pre = self._pre_state

        keys_inserted = [k for k in current if k not in pre]
        keys_deleted = [k for k in pre if k not in current]
        keys_updated = [k for k in current if k in pre and current[k] != pre[k]]

        all_mutated_keys = list(set(keys_inserted + keys_deleted + keys_updated))
        total_mutations = len(all_mutated_keys)

        # Detect potential financial delta from balance fields
        financial_delta = 0.0
        for k in keys_updated:
            old_val = pre[k]
            new_val = current[k]
            # Check if dict values contain 'balance' or numeric delta
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                financial_delta += abs(float(new_val) - float(old_val))
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                for subk in ("balance", "amount", "funds"):
                    if subk in old_val and subk in new_val:
                        try:
                            financial_delta += abs(float(new_val[subk]) - float(old_val[subk]))
                        except (ValueError, TypeError):
                            pass

        return StateDelta(
            rows_inserted=len(keys_inserted),
            rows_updated=len(keys_updated),
            rows_deleted=len(keys_deleted),
            total_mutations=total_mutations,
            keys_mutated=all_mutated_keys,
            financial_delta=financial_delta,
            metadata={
                "keys_inserted": keys_inserted,
                "keys_deleted": keys_deleted,
                "keys_updated": keys_updated,
            },
        )

    def commit(self) -> None:
        """Acknowledge modifications, discarding the pre-state backup."""
        self._pre_state = {}

    def rollback(self) -> None:
        """Revert the target dictionary to the exact deep-copied pre-state."""
        if self._pre_state:
            self.state_dict.clear()
            self.state_dict.update(copy.deepcopy(self._pre_state))
            self._pre_state = {}
