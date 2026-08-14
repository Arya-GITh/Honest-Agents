import functools
import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextvars import ContextVar
from pydantic import BaseModel, Field

from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.receipts.normalizer import PayloadNormalizer

# Context variable to hold active audit logs for context manager listening
_ACTIVE_AUDIT_LOG: ContextVar[Optional[List["ToolExecutionRecord"]]] = ContextVar("_ACTIVE_AUDIT_LOG", default=None)


class ToolExecutionRecord(BaseModel):
    """Immutable ground-truth record of a single tool invocation."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    start_time: float
    end_time: float
    duration_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str  # "success" | "error"
    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    receipt: Optional[HMACReceipt] = None


F = TypeVar("F", bound=Callable[..., Any])


def get_current_audit_log() -> Optional[List[ToolExecutionRecord]]:
    """Retrieve the current thread/task context's audit log list if active."""
    return _ACTIVE_AUDIT_LOG.get()


def register_execution_record(record: ToolExecutionRecord) -> None:
    """Register a tool execution record into the active context auditor if present."""
    log = get_current_audit_log()
    if log is not None:
        log.append(record)


def _serialize_arg(arg: Any) -> Any:
    """Best-effort JSON-friendly serialization of tool arguments and outputs."""
    if isinstance(arg, (int, float, str, bool, type(None))):
        return arg
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(item) for item in arg]
    if isinstance(arg, dict):
        return {str(k): _serialize_arg(v) for k, v in arg.items()}
    if hasattr(arg, "model_dump"):  # Pydantic v2
        try:
            return arg.model_dump()
        except Exception:
            pass
    if hasattr(arg, "dict"):  # Pydantic v1
        try:
            return arg.dict()
        except Exception:
            pass
    return repr(arg)


def audit_tool(
    fn_or_name: Union[F, str, None] = None,
    *,
    name: Optional[str] = None,
    error_keypaths: Optional[List[str]] = None,
    secret_key: Optional[Union[str, bytes]] = None,
    raise_exceptions: bool = True,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to wrap agent tool functions and capture ground-truth execution metadata and HMAC receipts.

    Supports usage as:
    @audit_tool
    def my_tool(...): ...

    @audit_tool(name="custom_name", error_keypaths=["data.err_code"])
    async def my_async_tool(...): ...
    """
    custom_name: Optional[str] = None

    if isinstance(fn_or_name, str):
        custom_name = fn_or_name
        target_fn = None
    elif callable(fn_or_name):
        target_fn = fn_or_name
    else:
        custom_name = name
        target_fn = None

    normalizer = PayloadNormalizer(custom_error_keypaths=error_keypaths)

    def decorator(fn: F) -> F:
        tool_name = custom_name or name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_t = time.time()
                serial_args = [_serialize_arg(a) for a in args]
                serial_kwargs = {k: _serialize_arg(v) for k, v in kwargs.items()}
                exec_id = str(uuid.uuid4())
                iso_ts = datetime.now(timezone.utc).isoformat()

                try:
                    result = await fn(*args, **kwargs)
                    end_t = time.time()
                    duration = (end_t - start_t) * 1000.0

                    receipt = HMACReceipt.from_execution(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="success",
                        result=result,
                        normalizer=normalizer,
                        secret_key=secret_key,
                    )

                    record = ToolExecutionRecord(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="success",
                        result=_serialize_arg(result),
                        receipt=receipt,
                    )
                    register_execution_record(record)
                    return result
                except Exception as exc:
                    end_t = time.time()
                    duration = (end_t - start_t) * 1000.0

                    receipt = HMACReceipt.from_execution(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        normalizer=normalizer,
                        secret_key=secret_key,
                    )

                    record = ToolExecutionRecord(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        receipt=receipt,
                    )
                    register_execution_record(record)
                    if raise_exceptions:
                        raise
                    return None

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_t = time.time()
                serial_args = [_serialize_arg(a) for a in args]
                serial_kwargs = {k: _serialize_arg(v) for k, v in kwargs.items()}
                exec_id = str(uuid.uuid4())
                iso_ts = datetime.now(timezone.utc).isoformat()

                try:
                    result = fn(*args, **kwargs)
                    end_t = time.time()
                    duration = (end_t - start_t) * 1000.0

                    receipt = HMACReceipt.from_execution(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="success",
                        result=result,
                        normalizer=normalizer,
                        secret_key=secret_key,
                    )

                    record = ToolExecutionRecord(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="success",
                        result=_serialize_arg(result),
                        receipt=receipt,
                    )
                    register_execution_record(record)
                    return result
                except Exception as exc:
                    end_t = time.time()
                    duration = (end_t - start_t) * 1000.0

                    receipt = HMACReceipt.from_execution(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        normalizer=normalizer,
                        secret_key=secret_key,
                    )

                    record = ToolExecutionRecord(
                        execution_id=exec_id,
                        tool_name=tool_name,
                        args=serial_args,
                        kwargs=serial_kwargs,
                        start_time=start_t,
                        end_time=end_t,
                        duration_ms=duration,
                        timestamp=iso_ts,
                        status="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        receipt=receipt,
                    )
                    register_execution_record(record)
                    if raise_exceptions:
                        raise
                    return None

            return sync_wrapper  # type: ignore[return-value]

    if target_fn is not None:
        return decorator(target_fn)
    return decorator
