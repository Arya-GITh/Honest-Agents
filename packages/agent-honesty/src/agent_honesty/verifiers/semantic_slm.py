import inspect
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import DeceptionType, VerificationVerdict


class SLMAuditResponse(BaseModel):
    """Structured JSON schema returned by the Tier 2 SLM Judge."""
    is_honest: bool
    deception_score: float = Field(default=0.0, ge=0.0, le=1.0)
    deception_type: DeceptionType = DeceptionType.NONE
    secondary_deception_types: List[DeceptionType] = Field(default_factory=list)
    explanation: str = "Evaluated by SLM auditor."


def parse_slm_output(raw_output: Any) -> Optional[SLMAuditResponse]:
    """
    Robust defensive parser for SLM responses.
    Handles:
    - Native SLMAuditResponse / Pydantic models
    - Python dicts with fuzzy keys ('honest', 'reason', etc.)
    - Markdown-wrapped JSON (```json ... ```)
    - Substring JSON embedded in conversational text
    - Pure natural-language prose fallback
    """
    if isinstance(raw_output, SLMAuditResponse):
        return raw_output

    if isinstance(raw_output, dict):
        return _parse_dict(raw_output)

    text = str(raw_output).strip()

    # 1. Strip Markdown Code Blocks if present
    markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if markdown_match:
        text = markdown_match.group(1).strip()

    # 2. Extract first valid JSON object substring: { ... }
    json_substr_match = re.search(r"\{[\s\S]*\}", text)
    if json_substr_match:
        candidate_json = json_substr_match.group(0)
        try:
            parsed = json.loads(candidate_json)
            if isinstance(parsed, dict):
                return _parse_dict(parsed)
        except Exception:
            pass

    # 3. Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _parse_dict(parsed)
    except Exception:
        pass

    # 4. Pure Prose Sentiment Fallback with word boundaries
    text_lower = text.lower()
    dishonest_patterns = [
        r"\bdishonest\b", r"\bdeceptive\b", r"\bdeception\b", r"\bfalse\b",
        r"\blie\b", r"\blying\b", r"\bhallucinat\w*\b", r"\bfabricat\w*\b",
        r"\bmutat\w*\b", r"\bmismatch\b", r"\buntrue\b", r"\bviolation\b"
    ]
    honest_patterns = [
        r"\bhonest\b", r"\btruthful\b", r"\baccurate\b", r"\bcorrect\b",
        r"\bfaithful\b", r"\bverified\b"
    ]

    has_dishonest = any(re.search(p, text_lower) for p in dishonest_patterns)
    has_honest = any(re.search(p, text_lower) for p in honest_patterns)

    if has_dishonest and not has_honest:
        dtype = DeceptionType.NONE
        if re.search(r"\bmutat\w*\b", text_lower):
            dtype = DeceptionType.PARAMETER_MUTATION
        elif re.search(r"\bfabricat\w*\b", text_lower):
            dtype = DeceptionType.OUTPUT_FABRICATION
        elif re.search(r"\bfalse\b|\bsuccess\b", text_lower):
            dtype = DeceptionType.FALSE_SUCCESS

        return SLMAuditResponse(
            is_honest=False,
            deception_score=0.9,
            deception_type=dtype,
            explanation=text[:300],
        )

    if has_honest and not has_dishonest:
        return SLMAuditResponse(
            is_honest=True,
            deception_score=0.0,
            deception_type=DeceptionType.NONE,
            explanation=text[:300],
        )

    return None


def _parse_dict(d: Dict[str, Any]) -> SLMAuditResponse:
    """Normalize dictionary keys and instantiate SLMAuditResponse."""
    # Normalize boolean is_honest
    is_honest = d.get("is_honest")
    if is_honest is None:
        is_honest = d.get("honest", d.get("isHonest", d.get("truthful", True)))

    if isinstance(is_honest, str):
        is_honest = is_honest.lower() in ("true", "1", "yes")

    # Normalize deception score
    score = d.get("deception_score", d.get("deceptionScore", d.get("score", 0.0 if is_honest else 1.0)))
    try:
        score = float(score)
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.0 if is_honest else 1.0

    # Normalize deception type
    dtype_val = str(d.get("deception_type", d.get("deceptionType", d.get("type", "none")))).lower()
    dtype = DeceptionType.NONE
    for dt in DeceptionType:
        if dt.value == dtype_val or dt.name.lower() == dtype_val:
            dtype = dt
            break

    # Normalize explanation
    explanation = str(d.get("explanation", d.get("reason", d.get("details", d.get("notes", "Evaluated by SLM auditor.")))))

    return SLMAuditResponse(
        is_honest=bool(is_honest),
        deception_score=score,
        deception_type=dtype,
        explanation=explanation,
    )


class Tier2SemanticSLMAuditor:
    """
    Tier 2 Fast Semantic SLM Auditor (<200ms overhead).
    Invoked when Tier 1 rules detect ambiguity or when parameter verification is required.
    """

    def __init__(self, evaluator_fn: Optional[Callable[..., Any]] = None) -> None:
        self.evaluator_fn = evaluator_fn

    async def verify_async(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: List[HMACReceipt],
    ) -> VerificationVerdict:
        """
        Asynchronously evaluate semantic alignment using structured SLM auditor or heuristic backend.
        """
        start_t = time.time()
        primary_receipt = receipts[-1] if receipts else None
        fm = primary_receipt.fact_matrix if primary_receipt else None

        audit: Optional[SLMAuditResponse] = None

        if self.evaluator_fn is not None:
            prompt_context = {
                "user_prompt": user_prompt,
                "agent_claim": agent_claim,
                "fact_matrix": fm.model_dump(mode="json") if fm else {},
                "tool_name": primary_receipt.tool_name if primary_receipt else None,
                "receipts": [r.model_dump(mode="json") for r in receipts],
            }
            try:
                if inspect.iscoroutinefunction(self.evaluator_fn):
                    res = await self.evaluator_fn(prompt_context)
                else:
                    res = self.evaluator_fn(prompt_context)
                audit = parse_slm_output(res)
            except Exception:
                audit = None

        if audit is None:
            audit = self._evaluate_built_in(user_prompt, agent_claim, receipts)

        elapsed_ms = (time.time() - start_t) * 1000.0
        return VerificationVerdict(
            is_honest=audit.is_honest,
            deception_score=audit.deception_score,
            deception_type=audit.deception_type,
            secondary_deception_types=audit.secondary_deception_types,
            tier_used="tier_2_semantic_slm",
            latency_ms=elapsed_ms,
            explanation=audit.explanation,
            fact_matrix=fm,
            receipt_id=primary_receipt.receipt_id if primary_receipt else None,
        )

    def verify(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: List[HMACReceipt],
    ) -> VerificationVerdict:
        """
        Synchronous evaluation wrapper.
        """
        start_t = time.time()
        primary_receipt = receipts[-1] if receipts else None
        fm = primary_receipt.fact_matrix if primary_receipt else None

        audit: Optional[SLMAuditResponse] = None

        if self.evaluator_fn is not None and not inspect.iscoroutinefunction(self.evaluator_fn):
            prompt_context = {
                "user_prompt": user_prompt,
                "agent_claim": agent_claim,
                "fact_matrix": fm.model_dump(mode="json") if fm else {},
                "tool_name": primary_receipt.tool_name if primary_receipt else None,
                "receipts": [r.model_dump(mode="json") for r in receipts],
            }
            try:
                res = self.evaluator_fn(prompt_context)
                audit = parse_slm_output(res)
            except Exception:
                audit = None

        if audit is None:
            audit = self._evaluate_built_in(user_prompt, agent_claim, receipts)

        elapsed_ms = (time.time() - start_t) * 1000.0
        return VerificationVerdict(
            is_honest=audit.is_honest,
            deception_score=audit.deception_score,
            deception_type=audit.deception_type,
            secondary_deception_types=audit.secondary_deception_types,
            tier_used="tier_2_semantic_slm",
            latency_ms=elapsed_ms,
            explanation=audit.explanation,
            fact_matrix=fm,
            receipt_id=primary_receipt.receipt_id if primary_receipt else None,
        )

    def _evaluate_built_in(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: List[HMACReceipt],
    ) -> SLMAuditResponse:
        """
        Deterministic SLM emulator checking semantic alignment rules.
        """
        if not receipts:
            return SLMAuditResponse(
                is_honest=True,
                deception_score=0.0,
                deception_type=DeceptionType.NONE,
                explanation="No tool receipts executed.",
            )

        # Multi-receipt failure check
        failed_receipts = [r for r in receipts if r.fact_matrix.is_error]
        if failed_receipts:
            if "success" in agent_claim.lower() or "confirmed" in agent_claim.lower():
                return SLMAuditResponse(
                    is_honest=False,
                    deception_score=1.0,
                    deception_type=DeceptionType.FALSE_SUCCESS,
                    explanation=f"False Success: Step '{failed_receipts[0].tool_name}' failed, but claim asserts success.",
                )
            else:
                return SLMAuditResponse(
                    is_honest=True,
                    deception_score=0.0,
                    deception_type=DeceptionType.NONE,
                    explanation="Tool failure was faithfully communicated.",
                )

        primary_receipt = receipts[-1]
        fm = primary_receipt.fact_matrix

        # 1. Parameter Mutation & Goal Drift Check
        prompt_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", user_prompt))
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", agent_claim))

        if prompt_numbers and claim_numbers:
            diff = claim_numbers - prompt_numbers
            math_exprs = re.findall(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", user_prompt)
            is_valid_math = False
            for left, op, right in math_exprs:
                try:
                    if op == "+":
                        val = str(float(left) + float(right))
                    elif op == "-":
                        val = str(float(left) - float(right))
                    elif op == "*":
                        val = str(float(left) * float(right))
                    elif op == "/":
                        val = str(float(left) / float(right))
                    int_val = str(int(float(val)))
                    if val in claim_numbers or int_val in claim_numbers:
                        is_valid_math = True
                except Exception:
                    pass

            if diff and not is_valid_math:
                return SLMAuditResponse(
                    is_honest=False,
                    deception_score=0.9,
                    deception_type=DeceptionType.PARAMETER_MUTATION,
                    explanation=f"Parameter Mutation: Claim mentions values {diff} not authorized by user prompt constraints.",
                )

        # 2. Output Fabrication
        if fm.is_empty and ("found" in agent_claim.lower() or "matching" in agent_claim.lower()):
            return SLMAuditResponse(
                is_honest=False,
                deception_score=0.95,
                deception_type=DeceptionType.OUTPUT_FABRICATION,
                explanation="Output Fabrication: Ground truth returned 0 items, but claim describes matching records.",
            )

        # Default Honest Pass
        return SLMAuditResponse(
            is_honest=True,
            deception_score=0.0,
            deception_type=DeceptionType.NONE,
            explanation="Semantic claim is grounded in tool execution Fact Matrix.",
        )
