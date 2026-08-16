from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from agent_honesty.verifiers.models import VerificationVerdict


class ActionPolicy(str, Enum):
    """
    Policy deciding how to handle an agent output when a verification failure / deception is detected.
    """
    REPROMPT = "reprompt"          # (Default) Injects correction feedback into scratchpad with N=2 hard cap
    AUTO_CORRECT = "auto_correct"  # Immediately rewrites response using verified FactMatrix values ($0 LLM tokens)
    BLOCK = "block"                # Immediately halts execution and raises ExecutionIntegrityError


class ExecutionIntegrityError(Exception):
    """
    Raised when execution integrity is violated and policy is set to BLOCK,
    or when an irrecoverable deception threshold is exceeded.
    """
    def __init__(self, message: str, verdict: Optional[VerificationVerdict] = None) -> None:
        super().__init__(message)
        self.verdict = verdict


class ActionResult(BaseModel):
    """
    Final standardized execution result returned to downstream callers.
    """
    delivered_claim: str = Field(description="The authoritative, honest text response delivered to the user")
    policy_applied: ActionPolicy
    reprompt_count: int = Field(default=0, ge=0, description="Number of internal self-correction attempts executed")
    verdict: VerificationVerdict
    overridden: bool = Field(default=False, description="True if LLM was bypassed by deterministic fallback override")
    metadata: Dict[str, Any] = Field(default_factory=dict)
