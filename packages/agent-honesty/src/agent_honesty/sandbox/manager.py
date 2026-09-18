import time
from typing import Any, Optional

from agent_honesty.sandbox.environments.base import BaseSandboxEnvironment
from agent_honesty.sandbox.inspector import StateDeltaInspector
from agent_honesty.sandbox.models import (
    DryRunReceipt,
    InvariantViolationError,
    SandboxPolicy,
    SandboxVerdict,
    StateDelta,
)


class SpeculativeSandbox:
    """
    Context manager for orchestrating speculative execution against a sandbox environment.
    Guarantees atomic commit on success and instant rollback on invariant violation or exception.
    """

    def __init__(
        self,
        environment: BaseSandboxEnvironment,
        policy: Optional[SandboxPolicy] = None,
        tool_name: str = "speculative_action",
        auto_commit: bool = True,
        dry_run_only: bool = False,
    ) -> None:
        self.environment = environment
        self.policy = policy or SandboxPolicy()
        self.tool_name = tool_name
        self.auto_commit = auto_commit
        self.dry_run_only = dry_run_only

        self.pre_state_hash: str = ""
        self.post_state_hash: str = ""
        self.delta: StateDelta = StateDelta()
        self.verdict: Optional[SandboxVerdict] = None
        self.receipt: Optional[DryRunReceipt] = None
        self._is_active: bool = False

    def __enter__(self) -> "SpeculativeSandbox":
        self.pre_state_hash = self.environment.get_state_hash()
        self.environment.enter_speculation()
        self._is_active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if not self._is_active:
            return False

        try:
            if exc_type is not None:
                # An unhandled exception occurred in the block; rollback immediately
                self.environment.rollback()
                self._is_active = False
                return False

            # Compute mutations
            self.delta = self.environment.compute_delta()
            self.post_state_hash = self.environment.get_state_hash()
            self.verdict = StateDeltaInspector.evaluate(self.delta, self.policy)

            # Build receipt
            self.receipt = DryRunReceipt(
                tool_name=self.tool_name,
                pre_state_hash=self.pre_state_hash,
                post_state_hash=self.post_state_hash,
                state_delta=self.delta,
                verdict=self.verdict,
                committed=False,
            )

            if not self.verdict.is_safe:
                self.environment.rollback()
                self._is_active = False
                raise InvariantViolationError(
                    f"Speculative execution aborted: {self.verdict.explanation}",
                    verdict=self.verdict,
                    delta=self.delta,
                )

            if self.auto_commit and not self.dry_run_only:
                self.environment.commit()
                self.receipt.committed = True
            else:
                self.environment.rollback()

            self._is_active = False
            return True
        finally:
            if self._is_active:
                self.environment.rollback()
                self._is_active = False

    async def __aenter__(self) -> "SpeculativeSandbox":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if not self._is_active:
            return False

        try:
            if exc_type is not None:
                self.environment.rollback()
                self._is_active = False
                return False

            self.delta = self.environment.compute_delta()
            self.post_state_hash = self.environment.get_state_hash()
            self.verdict = await StateDeltaInspector.evaluate_async(self.delta, self.policy)

            self.receipt = DryRunReceipt(
                tool_name=self.tool_name,
                pre_state_hash=self.pre_state_hash,
                post_state_hash=self.post_state_hash,
                state_delta=self.delta,
                verdict=self.verdict,
                committed=False,
            )

            if not self.verdict.is_safe:
                self.environment.rollback()
                self._is_active = False
                raise InvariantViolationError(
                    f"Speculative execution aborted: {self.verdict.explanation}",
                    verdict=self.verdict,
                    delta=self.delta,
                )

            if self.auto_commit and not self.dry_run_only:
                self.environment.commit()
                self.receipt.committed = True
            else:
                self.environment.rollback()

            self._is_active = False
            return True
        finally:
            if self._is_active:
                self.environment.rollback()
                self._is_active = False
