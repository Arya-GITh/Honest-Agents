import time
from typing import Any, Callable, List, Optional
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.deterministic import Tier1DeterministicEngine
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.semantic_slm import Tier2SemanticSLMAuditor


class VerificationRouter:
    """
    Verification Router orchestrating the Two-Tier verification waterfall.
    - Tier 1 Deterministic Engine (<10ms, $0 token cost) runs first.
    - Tier 2 Fast Semantic SLM Auditor (<200ms) runs if Tier 1 detects ambiguity or force_tier_2=True.
    """

    def __init__(
        self,
        tier1_engine: Optional[Tier1DeterministicEngine] = None,
        tier2_auditor: Optional[Tier2SemanticSLMAuditor] = None,
        slm_evaluator_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.tier1 = tier1_engine or Tier1DeterministicEngine()
        self.tier2 = tier2_auditor or Tier2SemanticSLMAuditor(evaluator_fn=slm_evaluator_fn)

    def verify(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: List[HMACReceipt],
        force_tier_2: bool = False,
    ) -> VerificationVerdict:
        """
        Synchronously verify an agent's claim against execution receipts.
        """
        start_t = time.time()

        if not force_tier_2:
            verdict = self.tier1.verify(user_prompt, agent_claim, receipts)
            if verdict is not None:
                verdict.latency_ms = (time.time() - start_t) * 1000.0
                return verdict

        # Escalate to Tier 2
        verdict = self.tier2.verify(user_prompt, agent_claim, receipts)
        verdict.latency_ms = (time.time() - start_t) * 1000.0
        return verdict

    async def verify_async(
        self,
        user_prompt: str,
        agent_claim: str,
        receipts: List[HMACReceipt],
        force_tier_2: bool = False,
    ) -> VerificationVerdict:
        """
        Asynchronously verify an agent's claim against execution receipts.
        """
        start_t = time.time()

        if not force_tier_2:
            verdict = self.tier1.verify(user_prompt, agent_claim, receipts)
            if verdict is not None:
                verdict.latency_ms = (time.time() - start_t) * 1000.0
                return verdict

        # Escalate to Tier 2
        verdict = await self.tier2.verify_async(user_prompt, agent_claim, receipts)
        verdict.latency_ms = (time.time() - start_t) * 1000.0
        return verdict
