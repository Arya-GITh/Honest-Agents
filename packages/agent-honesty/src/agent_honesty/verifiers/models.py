from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from agent_honesty.receipts.fact_matrix import FactMatrix


class DeceptionType(str, Enum):
    """Primary deception failure modes exhibited in multi-step agent trajectories."""
    NONE = "none"
    FALSE_SUCCESS = "false_success"          # Claiming success when tool returned error or timeout
    OUTPUT_FABRICATION = "output_fabrication" # Inventing specific data values when tool returned empty/partial
    PARAMETER_MUTATION = "parameter_mutation" # Silently mutating arguments contrary to user constraints
    GOAL_DRIFT = "goal_drift"                 # Execution intent deviates from original prompt


class VerificationVerdict(BaseModel):
    """
    Standardized verdict produced by Tier 1 (Deterministic) and Tier 2 (Semantic SLM) verifiers.
    """
    is_honest: bool
    deception_score: float = Field(ge=0.0, le=1.0, description="0.0 = completely honest, 1.0 = completely deceptive")
    deception_type: DeceptionType = DeceptionType.NONE
    tier_used: str = Field(description="'tier_1_deterministic' or 'tier_2_semantic_slm'")
    latency_ms: float = Field(ge=0.0)
    explanation: str
    fact_matrix: Optional[FactMatrix] = None
    receipt_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
