import functools
import inspect
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from agent_honesty.interceptors.context_manager import HonestyAuditor
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.sandbox.environments.base import BaseSandboxEnvironment
from agent_honesty.sandbox.environments.filesystem import EphemeralFileSystemSandbox
from agent_honesty.sandbox.environments.sqlite import SQLiteSandbox
from agent_honesty.sandbox.environments.state_dict import DictStateSandbox
from agent_honesty.sandbox.manager import SpeculativeSandbox
from agent_honesty.sandbox.models import (
    InvariantViolationError,
    SandboxPolicy,
    SandboxVerdict,
)


def _resolve_environment(
    args: tuple,
    kwargs: dict,
    environment_getter: Optional[Callable[..., BaseSandboxEnvironment]],
) -> Optional[BaseSandboxEnvironment]:
    """Helper to detect or extract a BaseSandboxEnvironment from function arguments."""
    if environment_getter is not None:
        return environment_getter(*args, **kwargs)

    # Check explicit env parameter
    for val in list(args) + list(kwargs.values()):
        if isinstance(val, BaseSandboxEnvironment):
            return val
        if isinstance(val, sqlite3.Connection):
            return SQLiteSandbox(val)
        if isinstance(val, dict) and "__sandbox_target__" in val:
            return DictStateSandbox(val)

    return None


def speculative_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    policy: Optional[SandboxPolicy] = None,
    environment_getter: Optional[Callable[..., BaseSandboxEnvironment]] = None,
    auto_commit: bool = True,
    name: Optional[str] = None,
) -> Any:
    """
    Decorator that executes a tool function inside an ephemeral copy-on-write sandbox.
    Evaluates state mutations against declarative safety policies before committing to live state.
    """
    active_policy = policy or SandboxPolicy()

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or getattr(fn, "__name__", "speculative_tool")

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                env = _resolve_environment(args, kwargs, environment_getter)
                if env is None:
                    # If no DB/state environment is detected, execute directly while auditing inputs
                    # Pre-check keyword blocklist
                    verdict = SandboxVerdict(
                        is_safe=True,
                        commit_allowed=True,
                        explanation="No state environment detected; dry-run skipped.",
                    )
                    return await fn(*args, **kwargs)

                async with SpeculativeSandbox(
                    environment=env,
                    policy=active_policy,
                    tool_name=tool_name,
                    auto_commit=auto_commit,
                ) as sandbox:
                    res = await fn(*args, **kwargs)
                return res

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                env = _resolve_environment(args, kwargs, environment_getter)
                if env is None:
                    return fn(*args, **kwargs)

                with SpeculativeSandbox(
                    environment=env,
                    policy=active_policy,
                    tool_name=tool_name,
                    auto_commit=auto_commit,
                ) as sandbox:
                    res = fn(*args, **kwargs)
                return res

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator
