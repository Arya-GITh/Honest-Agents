import hashlib
import hmac
import json
import os
import secrets
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from agent_honesty.receipts.fact_matrix import FactMatrix
from agent_honesty.receipts.normalizer import PayloadNormalizer

# Default session key generated once per Python process if AGENT_HONESTY_SECRET_KEY is not set
_DEFAULT_SECRET_KEY: bytes = os.environ.get("AGENT_HONESTY_SECRET_KEY", secrets.token_hex(32)).encode("utf-8")


def get_default_secret_key() -> bytes:
    """Get the active signing secret key."""
    return _DEFAULT_SECRET_KEY


def set_default_secret_key(key: Union[str, bytes]) -> None:
    """Override the default signing secret key."""
    global _DEFAULT_SECRET_KEY
    if isinstance(key, str):
        _DEFAULT_SECRET_KEY = key.encode("utf-8")
    else:
        _DEFAULT_SECRET_KEY = key


class HMACReceipt(BaseModel):
    """
    Unforgeable, cryptographically signed JSON execution receipt.
    Guarantees runtime ground-truth authenticity for any tool invocation.
    """
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    tool_name: str
    timestamp: str
    duration_ms: float
    args_hash: str
    kwargs_hash: str
    fact_matrix: FactMatrix
    signature: str = ""

    def canonical_payload(self) -> bytes:
        """
        Generate deterministic canonical bytes representing the receipt contents to sign/verify.
        Uses Pydantic v2 mode='json' for cross-platform serialization.
        """
        data = {
            "receipt_id": self.receipt_id,
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 3),
            "args_hash": self.args_hash,
            "kwargs_hash": self.kwargs_hash,
            "fact_matrix": self.fact_matrix.model_dump(mode="json"),
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def sign(self, secret_key: Optional[Union[str, bytes]] = None) -> "HMACReceipt":
        """Compute and attach HMAC-SHA256 signature."""
        key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else (secret_key or get_default_secret_key())
        mac = hmac.new(key_bytes, self.canonical_payload(), hashlib.sha256)
        self.signature = mac.hexdigest()
        return self

    def verify(self, secret_key: Optional[Union[str, bytes]] = None) -> bool:
        """
        Verify that the receipt signature matches the canonical payload and has not been tampered with.
        """
        if not self.signature:
            return False
        key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else (secret_key or get_default_secret_key())
        mac = hmac.new(key_bytes, self.canonical_payload(), hashlib.sha256)
        expected_sig = mac.hexdigest()
        return hmac.compare_digest(self.signature, expected_sig)

    @classmethod
    def from_execution(
        cls,
        execution_id: str,
        tool_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        start_time: float,
        end_time: float,
        duration_ms: float,
        timestamp: str,
        status: str,
        result: Any = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        normalizer: Optional[PayloadNormalizer] = None,
        secret_key: Optional[Union[str, bytes]] = None,
    ) -> "HMACReceipt":
        """
        Factory method to normalize payload and generate a signed HMACReceipt directly from execution records.
        """
        norm = normalizer or PayloadNormalizer()
        fact_matrix = norm.normalize(
            payload=result,
            status=status,
            error=error,
            error_type=error_type,
        )

        args_hash = FactMatrix.compute_sha256(args)
        kwargs_hash = FactMatrix.compute_sha256(kwargs)

        receipt = cls(
            execution_id=execution_id,
            tool_name=tool_name,
            timestamp=timestamp,
            duration_ms=duration_ms,
            args_hash=args_hash,
            kwargs_hash=kwargs_hash,
            fact_matrix=fact_matrix,
        )
        receipt.sign(secret_key)
        return receipt
