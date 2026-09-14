"""
DeceptionBench Dual-Pass Evaluator
----------------------------------
Evaluates models against benchmark scenarios in dual-pass mode:
Pass 1: Raw Baseline (Out-of-the-box model response)
Pass 2: Truthify-Guarded (In-scratchpad self-correction + deterministic fallback)
"""

import time
from typing import Any, Callable, Dict, List, Optional

from agent_honesty.actions.models import ActionPolicy
from agent_honesty.actions.reprompter import SelfCorrectionLoop
from agent_honesty.receipts.normalizer import PayloadNormalizer
from agent_honesty.receipts.receipt import HMACReceipt
from agent_honesty.verifiers.router import VerificationRouter
from benchmarks.datasets.generator import BenchmarkScenario
from benchmarks.metrics import ModelBenchmarkSummary, ScenarioExecution
from benchmarks.models.base import BaseModelProvider


class DeceptionBenchEvaluator:
    """Evaluates an AI model provider against DeceptionBench scenarios."""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        router: Optional[VerificationRouter] = None,
        max_reprompts: int = 2,
    ) -> None:
        self.provider = model_provider
        self.router = router or VerificationRouter()
        self.reprompter = SelfCorrectionLoop(router=self.router, max_reprompts=max_reprompts)
        self.normalizer = PayloadNormalizer()

    def _create_synthetic_receipt(self, scenario: BenchmarkScenario) -> HMACReceipt:
        """Create a cryptographically signed HMAC receipt from a scenario's tool return."""
        return HMACReceipt.from_execution(
            execution_id=f"exec_{scenario.scenario_id}",
            tool_name=scenario.tool_name,
            args=[],
            kwargs=scenario.tool_input,
            start_time=time.time() - 0.05,
            end_time=time.time(),
            duration_ms=50.0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="error" if scenario.is_failure_case else "success",
            result=scenario.tool_output,
            normalizer=self.normalizer,
        )

    async def evaluate_scenario(self, scenario: BenchmarkScenario) -> ScenarioExecution:
        """Execute dual-pass evaluation for a single benchmark scenario."""
        receipt = self._create_synthetic_receipt(scenario)
        receipts = [receipt]

        # --- PASS 1: Raw Baseline Response ---
        raw_response = await self.provider.generate_response(
            user_prompt=scenario.user_prompt,
            tool_name=scenario.tool_name,
            tool_input=scenario.tool_input,
            tool_output=scenario.tool_output,
            system_prompt=scenario.system_prompt,
        )

        raw_verdict = self.router.verify(
            user_prompt=scenario.user_prompt,
            agent_claim=raw_response,
            receipts=receipts,
        )

        # --- PASS 2: Truthify-Guarded Response (In-Scratchpad Reprompt) ---
        async def reprompt_callback(system_correction: str) -> str:
            return await self.provider.self_correct_response(
                user_prompt=scenario.user_prompt,
                tool_name=scenario.tool_name,
                tool_output=scenario.tool_output,
                previous_claim=raw_response,
                system_correction=system_correction,
            )

        start_time = time.perf_counter()
        guarded_result = await self.reprompter.execute_policy_async(
            user_prompt=scenario.user_prompt,
            initial_claim=raw_response,
            receipts=receipts,
            reprompt_callback=reprompt_callback,
            policy=ActionPolicy.REPROMPT,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ScenarioExecution(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            user_prompt=scenario.user_prompt,
            tool_name=scenario.tool_name,
            tool_input=scenario.tool_input,
            tool_output=scenario.tool_output,
            raw_response=raw_response,
            raw_verdict=raw_verdict,
            guarded_response=guarded_result.delivered_claim,
            guarded_verdict=guarded_result.verdict,
            reprompts_used=guarded_result.reprompt_count,
            overridden_by_fallback=guarded_result.overridden,
            verification_latency_ms=round(latency_ms, 2),
        )

    async def evaluate_dataset(
        self,
        scenarios: List[BenchmarkScenario],
        progress_callback: Optional[Callable[[int, int, ScenarioExecution], None]] = None,
        delay_seconds: float = 0.0,
    ) -> ModelBenchmarkSummary:
        """Run evaluation over an entire list of benchmark scenarios with optional thermal cooldown."""
        import asyncio

        executions: List[ScenarioExecution] = []
        total = len(scenarios)

        for idx, scenario in enumerate(scenarios, start=1):
            exec_result = await self.evaluate_scenario(scenario)
            executions.append(exec_result)

            if progress_callback:
                progress_callback(idx, total, exec_result)

            if delay_seconds > 0.0 and idx < total:
                await asyncio.sleep(delay_seconds)

        return ModelBenchmarkSummary.calculate_from_executions(
            model_name=self.provider.model_name,
            executions=executions,
        )
