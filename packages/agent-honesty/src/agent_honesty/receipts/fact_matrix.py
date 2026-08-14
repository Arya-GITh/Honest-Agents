import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FactMatrix(BaseModel):
    """
    Lightweight, immutable execution truth matrix extracted from tool execution payloads.
    Used by Tier-1 deterministic rules and Tier-2 SLM auditors for claim verification.
    """
    is_error: bool = False
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    is_empty: bool = False
    records_mutated: Optional[int] = None
    data_keys: List[str] = Field(default_factory=list)
    payload_sha256: str = ""
    extra_facts: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def compute_sha256(cls, data: Any) -> str:
        """Compute deterministic SHA-256 hash of any JSON-serializable data."""
        try:
            canonical = json.dumps(data, sort_keys=True, default=str)
        except Exception:
            canonical = str(data)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
