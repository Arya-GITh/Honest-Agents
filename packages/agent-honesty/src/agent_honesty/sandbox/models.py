import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field




class StateDelta(BaseModel):
    """
    Normalized measurement of state changes produced during a speculative dry-run.
    """
    rows_inserted: int = Field(default=0, ge=0)
    rows_updated: int = Field(default=0, ge=0)
    rows_deleted: int = Field(default=0, ge=0)
    total_mutations: int = Field(default=0, ge=0)
    tables_touched: List[str] = Field(default_factory=list)
    keys_mutated: List[str] = Field(default_factory=list)
    financial_delta: float = Field(default=0.0)
    files_modified: List[str] = Field(default_factory=list)
    bytes_written: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SandboxPolicy(BaseModel):
    """
    Declarative safety policy defining allowed state mutations in a speculative sandbox.
    """
    max_records_mutated: Optional[int] = Field(
        default=50,
        ge=0,
        description="Maximum rows/records allowed to be inserted, updated, or deleted in a single execution."
    )
    max_financial_delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum absolute financial amount delta allowed per transaction."
    )
    allowed_tables: Optional[List[str]] = Field(
        default=None,
        description="Whitelist of database tables allowed to be mutated. None allows all except blocked."
    )
    blocked_tables: Optional[List[str]] = Field(
        default_factory=lambda: ["auth_tokens", "credentials", "api_keys", "master_secrets"],
        description="Blacklist of critical system tables forbidden from modification."
    )
    blocked_keywords: List[str] = Field(
        default_factory=lambda: [
            "DROP TABLE", "DROP DATABASE", "TRUNCATE TABLE", "TRUNCATE",
            "DELETE FROM users WHERE 1=1", "DELETE FROM accounts WHERE 1=1",
            "rm -rf /", "rm -rf ~"
        ],
        description="SQL or command substrings that trigger an instant pre-execution block."
    )
    read_only: bool = Field(
        default=False,
        description="If True, any state modification whatsoever is rejected."
    )
    allow_file_writes: bool = Field(
        default=True,
        description="Whether writing or modifying files is permitted."
    )
    max_file_bytes: Optional[int] = Field(
        default=10_000_000, # 10MB
        description="Maximum allowed bytes written to files during execution."
    )
    # Custom callable hooks (excluded from pydantic serialization)
    custom_validator: Optional[Any] = Field(default=None, exclude=True)
    semantic_judge_fn: Optional[Any] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SandboxVerdict(BaseModel):
    """
    Safety certification produced by the StateDeltaInspector.
    """
    is_safe: bool = Field(description="True if the execution satisfied all policy invariants.")
    commit_allowed: bool = Field(description="True if changes may be committed to production.")
    violations: List[str] = Field(default_factory=list, description="List of policy invariant violations found.")
    state_delta: StateDelta = Field(default_factory=StateDelta)
    latency_ms: float = Field(default=0.0, description="Inspection latency in milliseconds.")
    explanation: str = Field(default="Evaluated by Speculative Sandbox Inspector.")


class DryRunReceipt(BaseModel):
    """
    Cryptographically verifiable receipt capturing the outcome of a speculative dry-run.
    """
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    pre_state_hash: str
    post_state_hash: str
    state_delta: StateDelta
    verdict: SandboxVerdict
    timestamp: float = Field(default_factory=time.time)
    committed: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvariantViolationError(Exception):
    """
    Raised when a tool action violates one or more invariants in SandboxPolicy during speculative execution.
    """
    def __init__(self, message: str, verdict: Optional[SandboxVerdict] = None, delta: Optional[StateDelta] = None) -> None:
        super().__init__(message)
        self.verdict = verdict
        self.delta = delta
