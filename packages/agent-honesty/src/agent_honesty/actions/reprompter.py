import inspect
from typing import Any, Callable, List, Optional
from agent_honesty.actions.models import ActionPolicy, ActionResult, ExecutionIntegrityError
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.models import VerificationVerdict
from agent_honesty.verifiers.router import VerificationRouter


class SelfCorrectionLoop:
    """
    Manages in-scratchpad self-correction loops and deterministic fallback overrides.
    Enforces a strict N=2 hard reprompt cap to eliminate infinite loops and preserve context window budget.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        max_reprompts: int = 2,
    ) -> None:
        self.router = router or VerificationRouter()
        self.max_reprompts = max_reprompts

    @staticmethod
    def format_reprompt_message(verdict: VerificationVerdict) -> str:
        """
        Format a structured system correction to inject into the agent's private scratchpad/history.
        """
        return (
            f"[System Honesty Correction: Your previous statement is inconsistent with the execution ground truth. "
            f"Issue: {verdict.explanation} "
            f"Please re-evaluate your tool execution facts and provide an accurate, grounded response to the user.]"
        )

    @staticmethod
    def generate_deterministic_fallback(
        user_prompt: str,
        receipts: List[HMACReceipt],
        verdict: VerificationVerdict,
    ) -> str:
        """
        Authoritative system-generated truth summary extracted directly from FactMatrix.
        Delivered when N=2 reprompt cap is exceeded or when AUTO_CORRECT policy is active.
        """
        if not receipts:
            return "System Notice: The action could not be verified against any tool execution receipts."

        primary_receipt = receipts[-1]
        fm = primary_receipt.fact_matrix
        tool_name = primary_receipt.tool_name

        if fm.is_error or (fm.status_code is not None and fm.status_code >= 400):
            err_detail = fm.error_message or fm.error_type or f"status code {fm.status_code}"
            return f"System Notice: The requested action '{tool_name}' failed with {err_detail}. Zero state mutations were confirmed."

        if fm.is_empty:
            return f"System Notice: The query '{tool_name}' executed successfully but returned 0 matching records."

        if fm.records_mutated is not None:
            return f"System Notice: Action '{tool_name}' completed with {fm.records_mutated} records mutated."

        return f"System Notice: Action '{tool_name}' completed with status {fm.status_code}."

    async def execute_policy_async(
        self,
        user_prompt: str,
        initial_claim: str,
        receipts: List[HMACReceipt],
        reprompt_callback: Optional[Callable[[str], Any]] = None,
        policy: ActionPolicy = ActionPolicy.REPROMPT,
        force_tier_2: bool = False,
    ) -> ActionResult:
        """
        Asynchronously evaluate claim and apply configured action policy (REPROMPT, AUTO_CORRECT, BLOCK).
        """
        verdict = await self.router.verify_async(
            user_prompt=user_prompt,
            agent_claim=initial_claim,
            receipts=receipts,
            force_tier_2=force_tier_2,
        )

        # 1. Honest on first try -> Instant Release
        if verdict.is_honest:
            return ActionResult(
                delivered_claim=initial_claim,
                policy_applied=policy,
                reprompt_count=0,
                verdict=verdict,
                overridden=False,
            )

        # 2. Policy: BLOCK -> Immediate Halt
        if policy == ActionPolicy.BLOCK:
            raise ExecutionIntegrityError(
                f"Execution Integrity Violation: {verdict.explanation}",
                verdict=verdict,
            )

        # 3. Policy: AUTO_CORRECT -> Instant $0 Token Fallback
        if policy == ActionPolicy.AUTO_CORRECT:
            fallback = self.generate_deterministic_fallback(user_prompt, receipts, verdict)
            return ActionResult(
                delivered_claim=fallback,
                policy_applied=ActionPolicy.AUTO_CORRECT,
                reprompt_count=0,
                verdict=verdict,
                overridden=True,
            )

        # 4. Policy: REPROMPT -> Self-Correction Loop with N=2 Hard Cap
        current_verdict = verdict
        current_claim = initial_claim

        if reprompt_callback is not None:
            for attempt in range(1, self.max_reprompts + 1):
                system_correction = self.format_reprompt_message(current_verdict)

                # Invoke the agent's LLM generation callback with the private correction
                if inspect.iscoroutinefunction(reprompt_callback):
                    new_claim = await reprompt_callback(system_correction)
                else:
                    new_claim = reprompt_callback(system_correction)

                current_claim = str(new_claim or "")

                # Verify the newly drafted response
                new_verdict = await self.router.verify_async(
                    user_prompt=user_prompt,
                    agent_claim=current_claim,
                    receipts=receipts,
                    force_tier_2=force_tier_2,
                )

                if new_verdict.is_honest:
                    return ActionResult(
                        delivered_claim=current_claim,
                        policy_applied=ActionPolicy.REPROMPT,
                        reprompt_count=attempt,
                        verdict=new_verdict,
                        overridden=False,
                    )

                current_verdict = new_verdict

        # Max reprompts exceeded without self-correction -> Deterministic Fallback Override!
        fallback_claim = self.generate_deterministic_fallback(user_prompt, receipts, current_verdict)
        return ActionResult(
            delivered_claim=fallback_claim,
            policy_applied=ActionPolicy.REPROMPT,
            reprompt_count=self.max_reprompts,
            verdict=current_verdict,
            overridden=True,
        )

    def execute_policy(
        self,
        user_prompt: str,
        initial_claim: str,
        receipts: List[HMACReceipt],
        reprompt_callback: Optional[Callable[[str], Any]] = None,
        policy: ActionPolicy = ActionPolicy.REPROMPT,
        force_tier_2: bool = False,
    ) -> ActionResult:
        """
        Synchronous evaluation and action policy execution wrapper.
        """
        verdict = self.router.verify(
            user_prompt=user_prompt,
            agent_claim=initial_claim,
            receipts=receipts,
            force_tier_2=force_tier_2,
        )

        if verdict.is_honest:
            return ActionResult(
                delivered_claim=initial_claim,
                policy_applied=policy,
                reprompt_count=0,
                verdict=verdict,
                overridden=False,
            )

        if policy == ActionPolicy.BLOCK:
            raise ExecutionIntegrityError(
                f"Execution Integrity Violation: {verdict.explanation}",
                verdict=verdict,
            )

        if policy == ActionPolicy.AUTO_CORRECT:
            fallback = self.generate_deterministic_fallback(user_prompt, receipts, verdict)
            return ActionResult(
                delivered_claim=fallback,
                policy_applied=ActionPolicy.AUTO_CORRECT,
                reprompt_count=0,
                verdict=verdict,
                overridden=True,
            )

        # Policy: REPROMPT (Sync)
        current_verdict = verdict
        current_claim = initial_claim

        if reprompt_callback is not None:
            for attempt in range(1, self.max_reprompts + 1):
                system_correction = self.format_reprompt_message(current_verdict)

                if inspect.iscoroutinefunction(reprompt_callback):
                    raise RuntimeError("Async reprompt_callback passed to synchronous execute_policy. Use execute_policy_async.")
                new_claim = reprompt_callback(system_correction)
                current_claim = str(new_claim or "")

                new_verdict = self.router.verify(
                    user_prompt=user_prompt,
                    agent_claim=current_claim,
                    receipts=receipts,
                    force_tier_2=force_tier_2,
                )

                if new_verdict.is_honest:
                    return ActionResult(
                        delivered_claim=current_claim,
                        policy_applied=ActionPolicy.REPROMPT,
                        reprompt_count=attempt,
                        verdict=new_verdict,
                        overridden=False,
                    )

                current_verdict = new_verdict

        fallback_claim = self.generate_deterministic_fallback(user_prompt, receipts, current_verdict)
        return ActionResult(
            delivered_claim=fallback_claim,
            policy_applied=ActionPolicy.REPROMPT,
            reprompt_count=self.max_reprompts,
            verdict=current_verdict,
            overridden=True,
        )
