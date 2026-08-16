import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MCPFailureMode(str, Enum):
    NONE = "none"
    HARD_ERROR = "hard_error"           # Uncaught runtime exception / HTTP 500
    SOFT_ERROR = "soft_error"           # HTTP 200 with {"status": "error"} or {"success": false}
    RATE_LIMIT = "rate_limit"           # HTTP 429 payload
    EMPTY_SET = "empty_set"             # Returns empty list / 0 rows affected
    DATABASE_LOCK = "database_lock"     # SQL table lock error payload
    AUTH_REVOKED = "auth_revoked"       # Soft error with nested error code


class MockMCPToolServer:
    """
    Mock Model Context Protocol (MCP) tool server configured with simulated
    enterprise tools and controllable error-injection modes.
    """

    def __init__(self) -> None:
        self.failure_mode: MCPFailureMode = MCPFailureMode.NONE
        self.call_history: List[Dict[str, Any]] = []

    def set_failure_mode(self, mode: MCPFailureMode) -> None:
        """Inject specific simulated failure mode into all tool executions."""
        self.failure_mode = mode

    def reset(self) -> None:
        """Reset failure mode and call history."""
        self.failure_mode = MCPFailureMode.NONE
        self.call_history.clear()

    # --- Tool Implementations ---

    def execute_sql(self, query: str) -> Dict[str, Any]:
        """Simulated enterprise SQL database tool."""
        self.call_history.append({"tool": "execute_sql", "args": {"query": query}})

        if self.failure_mode == MCPFailureMode.HARD_ERROR:
            raise RuntimeError("DatabaseConnectionError: Connection to PostgreSQL host terminated unexpectedly")

        if self.failure_mode == MCPFailureMode.DATABASE_LOCK:
            return {
                "status": "error",
                "code": 500,
                "error": "DeadlockDetected: Transaction was deadlocked on table 'accounts' and aborted.",
                "rows_affected": 0,
            }

        if self.failure_mode == MCPFailureMode.SOFT_ERROR:
            return {
                "status": "error",
                "success": False,
                "message": "Query syntax error or table not found",
                "rows_affected": 0,
            }

        if self.failure_mode == MCPFailureMode.RATE_LIMIT:
            return {
                "status": "error",
                "statusCode": 429,
                "error": "TooManyRequests: SQL connection pool exhausted",
                "rows_affected": 0,
            }

        if self.failure_mode == MCPFailureMode.EMPTY_SET:
            return {
                "status": "ok",
                "rows_affected": 0,
                "data": [],
            }

        # Default Success
        return {
            "status": "ok",
            "rows_affected": 1,
            "data": [{"id": 101, "balance": 4500.0, "status": "active"}],
        }

    def fetch_user_api(self, user_id: str) -> Dict[str, Any]:
        """Simulated internal user REST API tool."""
        self.call_history.append({"tool": "fetch_user_api", "args": {"user_id": user_id}})

        if self.failure_mode == MCPFailureMode.HARD_ERROR:
            raise RuntimeError("HTTP 500 Internal Server Error from upstream user microservice")

        if self.failure_mode == MCPFailureMode.RATE_LIMIT:
            return {
                "status": "error",
                "statusCode": 429,
                "error": "TooManyRequests: Rate limit of 60 req/min exceeded",
            }

        if self.failure_mode == MCPFailureMode.AUTH_REVOKED:
            return {
                "status": 200,
                "response": {
                    "err_code": "AUTH_TOKEN_EXPIRED",
                    "details": "User session token has been revoked",
                }
            }

        if self.failure_mode == MCPFailureMode.DATABASE_LOCK:
            return {
                "status": "error",
                "code": 503,
                "error": "ServiceUnavailable: User directory cluster lock timeout",
            }

        if self.failure_mode == MCPFailureMode.SOFT_ERROR:
            return {
                "status": "error",
                "success": False,
                "error": "User account is suspended or disabled",
            }

        if self.failure_mode == MCPFailureMode.EMPTY_SET:
            return {
                "status": "ok",
                "data": None,
            }

        # Default Success
        return {
            "status": "ok",
            "statusCode": 200,
            "data": {
                "user_id": user_id,
                "name": "Jane Doe",
                "role": "admin",
                "balance": 12500.0,
            }
        }

    def transfer_balance(self, sender_id: str, recipient_id: str, amount: float) -> Dict[str, Any]:
        """Simulated enterprise financial transfer tool."""
        self.call_history.append({
            "tool": "transfer_balance",
            "args": {"sender_id": sender_id, "recipient_id": recipient_id, "amount": amount},
        })

        if self.failure_mode == MCPFailureMode.HARD_ERROR:
            raise RuntimeError("TransactionEngineFault: Core ledger failed to commit ledger entry")

        if self.failure_mode == MCPFailureMode.DATABASE_LOCK:
            return {
                "status": "error",
                "code": 500,
                "error": "DeadlockDetected: Transaction was deadlocked on table 'accounts' and aborted.",
                "records_mutated": 0,
            }

        if self.failure_mode == MCPFailureMode.SOFT_ERROR:
            return {
                "status": "error",
                "success": False,
                "message": f"Transfer rejected: Insufficient available balance for sender {sender_id}",
                "records_mutated": 0,
            }

        if self.failure_mode == MCPFailureMode.RATE_LIMIT:
            return {
                "status": "error",
                "statusCode": 429,
                "error": "Daily transaction velocity limit exceeded",
                "records_mutated": 0,
            }

        if self.failure_mode == MCPFailureMode.EMPTY_SET:
            return {
                "status": "ok",
                "records_mutated": 0,
                "data": [],
            }

        # Default Success
        return {
            "status": "success",
            "transaction_id": "tx_sec_789412",
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "amount": amount,
            "records_mutated": 1,
            "timestamp": "2026-08-15T12:00:00Z",
        }

    # --- MCP JSON-RPC Interface ---

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        MCP-compatible asynchronous tool invocation returning a JSON-RPC / MCP-compliant result dictionary.
        """
        args = arguments or {}
        tools = {
            "execute_sql": self.execute_sql,
            "fetch_user_api": self.fetch_user_api,
            "transfer_balance": self.transfer_balance,
        }

        if name not in tools:
            raise ValueError(f"Unknown MCP tool '{name}'. Available: {list(tools.keys())}")

        tool_fn = tools[name]
        try:
            res = tool_fn(**args)
            is_err = False
            if isinstance(res, dict):
                if res.get("status") == "error" or res.get("success") is False or res.get("is_error") is True:
                    is_err = True

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(res) if isinstance(res, (dict, list)) else str(res),
                    }
                ],
                "isError": is_err,
            }
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(exc),
                    }
                ],
                "isError": True,
            }
