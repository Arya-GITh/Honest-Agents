import inspect
from typing import Any, AsyncIterator, Callable, List, Optional
from agent_honesty.actions.models import ActionPolicy, ActionResult, ExecutionIntegrityError
from agent_honesty.actions.reprompter import SelfCorrectionLoop
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.router import VerificationRouter


class DualChannelStreamManager:
    """
    Dual-Channel Streaming Engine managing low-risk fast-path streaming
    vs. high-risk gated token buffering with in-flight integrity evaluation.
    """

    def __init__(
        self,
        router: Optional[VerificationRouter] = None,
        reprompter: Optional[SelfCorrectionLoop] = None,
        default_policy: ActionPolicy = ActionPolicy.REPROMPT,
    ) -> None:
        self.router = router or VerificationRouter()
        self.reprompter = reprompter or SelfCorrectionLoop(router=self.router)
        self.default_policy = default_policy

    async def stream_with_integrity(
        self,
        token_stream: AsyncIterator[str],
        user_prompt: str,
        receipts: List[HMACReceipt],
        reprompt_stream_factory: Optional[Callable[[str], AsyncIterator[str]]] = None,
        is_high_risk: bool = True,
        policy: Optional[ActionPolicy] = None,
        force_tier_2: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream LLM tokens with real-time honesty governance.

        Args:
            token_stream: Active async iterator of string chunks from LLM.
            user_prompt: Original user input prompt.
            receipts: Tool execution receipts for this turn.
            reprompt_stream_factory: Optional factory producing an AsyncIterator[str] when reprompted.
            is_high_risk: If True, buffers tokens until verified. If False, streams tokens in fast-path.
            policy: Configured action policy (defaults to self.default_policy).
            force_tier_2: If True, forces Tier 2 SLM auditor.
        """
        active_policy = policy or self.default_policy

        if not is_high_risk:
            # Low-Risk Fast Channel: Yield chunks immediately
            full_text_chunks = []
            async for chunk in token_stream:
                full_text_chunks.append(chunk)
                yield chunk

            # Post-stream background verification
            full_claim = "".join(full_text_chunks)
            await self.router.verify_async(user_prompt, full_claim, receipts, force_tier_2=force_tier_2)
            return

        # High-Risk Gated Channel: Buffer in memory before release
        buffer: List[str] = []
        async for chunk in token_stream:
            buffer.append(chunk)

        initial_claim = "".join(buffer)

        # Evaluate against execution truth
        verdict = await self.router.verify_async(
            user_prompt=user_prompt,
            agent_claim=initial_claim,
            receipts=receipts,
            force_tier_2=force_tier_2,
        )

        # 1. Honest -> Flush buffered chunks to consumer
        if verdict.is_honest:
            for chunk in buffer:
                yield chunk
            return

        # 2. Deceptive: Discard false buffer and apply policy
        if active_policy == ActionPolicy.BLOCK:
            raise ExecutionIntegrityError(
                f"Execution Integrity Violation: {verdict.explanation}",
                verdict=verdict,
            )

        if active_policy == ActionPolicy.AUTO_CORRECT:
            fallback = self.reprompter.generate_deterministic_fallback(user_prompt, receipts, verdict)
            yield fallback
            return

        # 3. Policy: REPROMPT
        current_verdict = verdict
        current_claim = initial_claim

        if reprompt_stream_factory is not None:
            for attempt in range(1, self.reprompter.max_reprompts + 1):
                try:
                    system_correction = self.reprompter.format_reprompt_message(current_verdict)
                    new_stream = reprompt_stream_factory(system_correction)

                    new_buffer = []
                    async for chunk in new_stream:
                        new_buffer.append(chunk)

                    current_claim = "".join(new_buffer)
                    new_verdict = await self.router.verify_async(
                        user_prompt=user_prompt,
                        agent_claim=current_claim,
                        receipts=receipts,
                        force_tier_2=force_tier_2,
                    )

                    if new_verdict.is_honest:
                        for chunk in new_buffer:
                            yield chunk
                        return

                    current_verdict = new_verdict
                except Exception:
                    break

        # If N=2 exceeded or stream factory fails -> Deliver Deterministic Fallback Override
        fallback = self.reprompter.generate_deterministic_fallback(user_prompt, receipts, current_verdict)
        yield fallback
