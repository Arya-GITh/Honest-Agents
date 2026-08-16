import re
from typing import Any, Dict, List, Optional, Sequence
from agent_honesty.receipts.fact_matrix import FactMatrix


def resolve_keypath(data: Any, keypath: str) -> Any:
    """
    Resolve a dot-notation keypath (e.g. 'data.user.id', 'items[0].name') in nested structures.
    Returns None if keypath cannot be resolved.
    """
    if not keypath or data is None:
        return None

    # Split by dots or bracket indices: e.g. "a.b[0].c" -> ["a", "b", "0", "c"]
    tokens = re.findall(r"[^\.\[\]]+", keypath)
    current = data

    for token in tokens:
        if current is None:
            return None
        if isinstance(current, dict):
            if token in current:
                current = current[token]
            else:
                return None
        elif isinstance(current, (list, tuple)):
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        elif hasattr(current, token):
            current = getattr(current, token)
        else:
            return None

    return current


class PayloadNormalizer:
    """
    Normalizes arbitrary tool return payloads, detecting soft errors, status codes,
    emptiness, CLI return codes, and mutation counts into a canonical FactMatrix.
    """

    # Common keys indicating soft errors when present / truthy
    DEFAULT_ERROR_KEYS: Sequence[str] = (
        "error",
        "errors",
        "err",
        "error_code",
        "err_code",
        "errorCode",
        "errorMessage",
        "error_message",
        "error_msg",
        "sql_error",
        "db_error",
        "sqlite_error",
        "pg_error",
        "exception",
    )

    # Common keys indicating record counts / mutations
    MUTATION_KEYS: Sequence[str] = (
        "records_mutated",
        "rows_affected",
        "affected_rows",
        "rows_updated",
        "rowcount",
        "count",
        "updated",
        "deleted",
        "inserted",
    )

    # Common data container keys in API envelopes
    DATA_CONTAINER_KEYS: Sequence[str] = (
        "data",
        "items",
        "results",
        "rows",
        "records",
        "payload",
    )

    def __init__(
        self,
        custom_error_keypaths: Optional[List[str]] = None,
        custom_success_keypaths: Optional[List[str]] = None,
    ) -> None:
        self.custom_error_keypaths = custom_error_keypaths or []
        self.custom_success_keypaths = custom_success_keypaths or []

    def normalize(
        self,
        payload: Any,
        status: str = "success",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> FactMatrix:
        """
        Produce a normalized FactMatrix from raw tool output or error.
        """
        # If execution already failed with an unhandled exception or exception object passed
        if status == "error" or error is not None:
            return FactMatrix(
                is_error=True,
                status_code=500,
                error_type=error_type or "ToolExecutionError",
                error_message=error,
                is_empty=True,
                payload_sha256=FactMatrix.compute_sha256(error),
            )

        if isinstance(payload, BaseException):
            return FactMatrix(
                is_error=True,
                status_code=500,
                error_type=type(payload).__name__,
                error_message=str(payload) or repr(payload),
                is_empty=True,
                payload_sha256=FactMatrix.compute_sha256(str(payload)),
            )

        # Check for HTTP Response objects (httpx, requests, etc.)
        status_code: Optional[int] = None
        extracted_body: Any = payload

        if hasattr(payload, "status_code"):
            status_code = getattr(payload, "status_code")
            if hasattr(payload, "json") and callable(payload.json):
                try:
                    extracted_body = payload.json()
                except Exception:
                    extracted_body = getattr(payload, "text", str(payload))
            elif hasattr(payload, "text"):
                extracted_body = getattr(payload, "text")

        # Check for Subprocess / CLI exit codes (CompletedProcess)
        is_error = False
        extracted_error_msg: Optional[str] = None
        extracted_error_type: Optional[str] = None

        for exit_attr in ("returncode", "exit_code", "exitcode"):
            if hasattr(payload, exit_attr):
                code = getattr(payload, exit_attr)
                if isinstance(code, int) and code != 0:
                    is_error = True
                    status_code = code
                    extracted_error_type = "NonZeroExitCode"
                    stderr = getattr(payload, "stderr", "")
                    extracted_error_msg = f"Process exited with code {code}: {stderr}"

        # Emptiness check on extracted body
        is_empty = False
        if extracted_body is None:
            is_empty = True
        elif isinstance(extracted_body, (list, dict, str, tuple, set)) and len(extracted_body) == 0:
            is_empty = True

        data_keys: List[str] = []
        records_mutated: Optional[int] = None

        # Check HTTP status code failure
        if status_code is not None and isinstance(status_code, int) and status_code >= 400:
            is_error = True
            extracted_error_type = f"HTTP_{status_code}_Error"
            extracted_error_msg = f"HTTP Status {status_code}"

        # Inspect dictionary bodies for soft errors, status codes, and metadata
        if isinstance(extracted_body, dict):
            data_keys = list(str(k) for k in extracted_body.keys())

            # Check status code inside payload (int or string representation)
            for sc_key in ("status_code", "statusCode", "code", "http_status"):
                if sc_key in extracted_body:
                    val = extracted_body[sc_key]
                    parsed_code = None
                    if isinstance(val, int):
                        parsed_code = val
                    elif isinstance(val, str) and val.isdigit():
                        parsed_code = int(val)

                    if parsed_code is not None:
                        if status_code is None:
                            status_code = parsed_code
                        if parsed_code >= 400:
                            is_error = True
                            extracted_error_type = extracted_error_type or f"HTTP_{parsed_code}"

            # Check for custom error keypaths
            for keypath in self.custom_error_keypaths:
                val = resolve_keypath(extracted_body, keypath)
                if val is not None and val is not False and val != 0 and val != "" and str(val).lower() != "false":
                    is_error = True
                    extracted_error_msg = f"Soft failure detected at keypath '{keypath}': {val}"
                    extracted_error_type = extracted_error_type or "SoftError"

            # Check for standard soft-failure indicators
            if not is_error:
                # 1. status / success / ok field checks
                if "status" in extracted_body:
                    status_val = str(extracted_body["status"]).lower()
                    if status_val in ("error", "failed", "failure", "fatal", "rejected", "err", "500", "503", "404", "400", "429"):
                        is_error = True
                        extracted_error_type = extracted_error_type or "SoftError"
                        extracted_error_msg = extracted_body.get("message") or extracted_body.get("msg") or f"Status: {status_val}"

                if "success" in extracted_body:
                    succ_val = extracted_body["success"]
                    if succ_val is False or str(succ_val).lower() in ("false", "0"):
                        is_error = True
                        extracted_error_type = extracted_error_type or "SoftError"
                        extracted_error_msg = extracted_body.get("message") or extracted_body.get("error") or "success is false"

                if "ok" in extracted_body:
                    ok_val = extracted_body["ok"]
                    if ok_val is False or str(ok_val).lower() in ("false", "0"):
                        is_error = True
                        extracted_error_type = extracted_error_type or "SoftError"
                        extracted_error_msg = extracted_body.get("message") or extracted_body.get("error") or "ok is false"

                # 2. Error key checks
                for err_k in self.DEFAULT_ERROR_KEYS:
                    if err_k in extracted_body:
                        err_v = extracted_body[err_k]
                        if err_v is not None and err_v is not False and err_v != "" and err_v != [] and str(err_v).lower() != "none":
                            is_error = True
                            extracted_error_type = extracted_error_type or "SoftError"
                            extracted_error_msg = str(err_v)
                            break

            # Extract mutation counts
            for mut_k in self.MUTATION_KEYS:
                if mut_k in extracted_body and isinstance(extracted_body[mut_k], (int, float)):
                    records_mutated = int(extracted_body[mut_k])
                    break

            # Check nested data containers for emptiness: e.g. {"status": 200, "data": []}
            if not is_empty and not is_error:
                for container_k in self.DATA_CONTAINER_KEYS:
                    if container_k in extracted_body:
                        container_val = extracted_body[container_k]
                        if container_val is None or (isinstance(container_val, (list, dict, set, tuple)) and len(container_val) == 0):
                            is_empty = True
                            break

        elif isinstance(extracted_body, list):
            data_keys = [f"item[{i}]" for i in range(min(len(extracted_body), 5))]

        payload_hash = FactMatrix.compute_sha256(extracted_body)

        return FactMatrix(
            is_error=is_error,
            status_code=status_code or (200 if not is_error else 500),
            error_type=extracted_error_type,
            error_message=extracted_error_msg,
            is_empty=is_empty,
            records_mutated=records_mutated,
            data_keys=data_keys,
            payload_sha256=payload_hash,
        )
