import inspect
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from agent_honesty.interceptors.tool_decorator import (
    ToolExecutionRecord,
    _serialize_arg,
    register_execution_record,
)
from agent_honesty.receipts.normalizer import PayloadNormalizer
from agent_honesty.receipts.receipt import HMACReceipt


def parse_mcp_result(result: Any) -> Any:
    """
    Extract meaningful Python data from an MCP CallToolResult or JSON-RPC tool response.
    Handles MCP TextContent, isError flags, and JSON stringified text bodies.
    """
    if result is None:
        return None

    # Standard MCP CallToolResult object (has .content and .isError)
    if hasattr(result, "content"):
        content_items = result.content
        extracted_texts = []
        is_error = getattr(result, "isError", False)

        for item in content_items:
            if hasattr(item, "text"):
                extracted_texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                extracted_texts.append(item["text"])
            else:
                extracted_texts.append(str(item))

        combined_text = "\n".join(extracted_texts) if extracted_texts else ""

        if combined_text:
            try:
                parsed_json = json.loads(combined_text)
                if isinstance(parsed_json, dict) and is_error:
                    parsed_json.setdefault("is_error", True)
                return parsed_json
            except Exception:
                pass

        if is_error:
            return {"status": "error", "message": combined_text or "MCP Tool Reported Error", "is_error": True}
        return combined_text

    # Dictionary representation of MCP result
    if isinstance(result, dict):
        is_error = result.get("isError", False)
        if "content" in result and isinstance(result["content"], list):
            texts = [c.get("text", "") for c in result["content"] if isinstance(c, dict)]
            combined = "\n".join(texts)
            try:
                parsed_json = json.loads(combined)
                if isinstance(parsed_json, dict) and is_error:
                    parsed_json.setdefault("is_error", True)
                return parsed_json
            except Exception:
                if is_error:
                    return {"status": "error", "message": combined or "MCP Tool Reported Error", "is_error": True}
                return combined

        if is_error:
            if not any(k in result for k in ("status", "error", "is_error")):
                result["is_error"] = True
        return result

    return result


async def intercept_mcp_call_async(
    tool_name: str,
    arguments: Dict[str, Any],
    call_fn: Callable[..., Any],
    *,
    error_keypaths: Optional[List[str]] = None,
    secret_key: Optional[Union[str, bytes]] = None,
) -> Any:
    """
    Asynchronously intercept and audit an MCP tool execution.
    """
    start_t = time.time()
    exec_id = str(uuid.uuid4())
    iso_ts = datetime.now(timezone.utc).isoformat()
    normalizer = PayloadNormalizer(custom_error_keypaths=error_keypaths)
    serial_kwargs = {str(k): _serialize_arg(v) for k, v in arguments.items()}

    try:
        if inspect.iscoroutinefunction(call_fn):
            raw_result = await call_fn(**arguments)
        else:
            raw_result = call_fn(**arguments)

        end_t = time.time()
        duration = (end_t - start_t) * 1000.0

        parsed_payload = parse_mcp_result(raw_result)

        receipt = HMACReceipt.from_execution(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
            kwargs=serial_kwargs,
            start_time=start_t,
            end_time=end_t,
            duration_ms=duration,
            timestamp=iso_ts,
            status="success",
            result=parsed_payload,
            normalizer=normalizer,
            secret_key=secret_key,
        )

        record = ToolExecutionRecord(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
            kwargs=serial_kwargs,
            start_time=start_t,
            end_time=end_t,
            duration_ms=duration,
            timestamp=iso_ts,
            status="success",
            result=_serialize_arg(parsed_payload),
            receipt=receipt,
        )
        register_execution_record(record)
        return raw_result

    except Exception as exc:
        end_t = time.time()
        duration = (end_t - start_t) * 1000.0

        receipt = HMACReceipt.from_execution(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
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
            args=[],
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
        raise


def intercept_mcp_call(
    tool_name: str,
    arguments: Dict[str, Any],
    call_fn: Callable[..., Any],
    *,
    error_keypaths: Optional[List[str]] = None,
    secret_key: Optional[Union[str, bytes]] = None,
) -> Any:
    """
    Synchronously intercept and audit an MCP tool execution.
    """
    start_t = time.time()
    exec_id = str(uuid.uuid4())
    iso_ts = datetime.now(timezone.utc).isoformat()
    normalizer = PayloadNormalizer(custom_error_keypaths=error_keypaths)
    serial_kwargs = {str(k): _serialize_arg(v) for k, v in arguments.items()}

    try:
        raw_result = call_fn(**arguments)
        end_t = time.time()
        duration = (end_t - start_t) * 1000.0

        parsed_payload = parse_mcp_result(raw_result)

        receipt = HMACReceipt.from_execution(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
            kwargs=serial_kwargs,
            start_time=start_t,
            end_time=end_t,
            duration_ms=duration,
            timestamp=iso_ts,
            status="success",
            result=parsed_payload,
            normalizer=normalizer,
            secret_key=secret_key,
        )

        record = ToolExecutionRecord(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
            kwargs=serial_kwargs,
            start_time=start_t,
            end_time=end_t,
            duration_ms=duration,
            timestamp=iso_ts,
            status="success",
            result=_serialize_arg(parsed_payload),
            receipt=receipt,
        )
        register_execution_record(record)
        return raw_result

    except Exception as exc:
        end_t = time.time()
        duration = (end_t - start_t) * 1000.0

        receipt = HMACReceipt.from_execution(
            execution_id=exec_id,
            tool_name=tool_name,
            args=[],
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
            args=[],
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
        raise


class MCPClientProxy:
    """
    Transparent proxy wrapper for MCP Client sessions.
    Automatically intercepts all `call_tool` invocations and produces signed HMAC receipts.
    """

    def __init__(
        self,
        client: Any,
        error_keypaths: Optional[List[str]] = None,
        secret_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        self._client = client
        self.error_keypaths = error_keypaths
        self.secret_key = secret_key

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        args = arguments or {}

        async def _invoke(**tool_args: Any) -> Any:
            if hasattr(self._client, "call_tool"):
                target = getattr(self._client, "call_tool")
            elif callable(self._client):
                target = self._client
            else:
                raise AttributeError("Wrapped MCP client does not have a callable call_tool method.")

            res = target(name=name, arguments=tool_args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res

        return await intercept_mcp_call_async(
            tool_name=name,
            arguments=args,
            call_fn=_invoke,
            error_keypaths=self.error_keypaths,
            secret_key=self.secret_key,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
