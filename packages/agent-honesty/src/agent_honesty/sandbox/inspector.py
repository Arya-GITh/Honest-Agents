import inspect
import time
from typing import Any, Dict, List, Optional

from agent_honesty.sandbox.models import (
    SandboxPolicy,
    SandboxVerdict,
    StateDelta,
)


class StateDeltaInspector:
    """
    Evaluates computed StateDelta and tool execution context against a SandboxPolicy.
    """

    @classmethod
    def evaluate(
        cls,
        delta: StateDelta,
        policy: SandboxPolicy,
        command_or_input: Optional[Any] = None,
    ) -> SandboxVerdict:
        """
        Synchronously evaluate state delta against policy rules.
        """
        start_t = time.time()
        violations: List[str] = []

        # 1. Keyword Blocklist Check
        if command_or_input is not None and policy.blocked_keywords:
            input_str = str(command_or_input).upper()
            for kw in policy.blocked_keywords:
                if kw.upper() in input_str:
                    violations.append(f"Blocked dangerous keyword or pattern detected: '{kw}'")

        # 2. Read-Only Invariant
        if policy.read_only:
            if delta.total_mutations > 0 or delta.bytes_written > 0 or delta.files_modified:
                violations.append("Policy is read-only, but state mutations were attempted.")

        # 3. Max Records Mutated
        if policy.max_records_mutated is not None:
            if delta.total_mutations > policy.max_records_mutated:
                violations.append(
                    f"Total records mutated ({delta.total_mutations}) exceeds policy limit ({policy.max_records_mutated})."
                )

        # 4. Financial Limit
        if policy.max_financial_delta is not None:
            if abs(delta.financial_delta) > policy.max_financial_delta:
                violations.append(
                    f"Financial delta (${abs(delta.financial_delta):.2f}) exceeds policy limit (${policy.max_financial_delta:.2f})."
                )

        # 5. Table Whitelist
        if policy.allowed_tables is not None:
            for tbl in delta.tables_touched:
                if tbl not in policy.allowed_tables:
                    violations.append(f"Table '{tbl}' is not in allowed_tables whitelist.")

        # 6. Table Blacklist
        if policy.blocked_tables:
            for tbl in delta.tables_touched:
                if tbl in policy.blocked_tables:
                    violations.append(f"Table '{tbl}' is in blocked_tables blacklist.")

        # 7. File Modifications
        if not policy.allow_file_writes and (delta.files_modified or delta.bytes_written > 0):
            violations.append("File modification is prohibited by policy.")

        if policy.max_file_bytes is not None and delta.bytes_written > policy.max_file_bytes:
            violations.append(
                f"Bytes written ({delta.bytes_written}) exceeds max_file_bytes limit ({policy.max_file_bytes})."
            )

        # 8. Custom Validator Hook
        if policy.custom_validator is not None and callable(policy.custom_validator):
            try:
                res = policy.custom_validator(delta)
                if res is False:
                    violations.append("Custom validation rule failed.")
                elif isinstance(res, str) and res:
                    violations.append(f"Custom validation failed: {res}")
            except Exception as e:
                violations.append(f"Custom validator exception: {str(e)}")

        elapsed_ms = (time.time() - start_t) * 1000.0
        is_safe = len(violations) == 0

        explanation = (
            "All speculative policy invariants satisfied."
            if is_safe
            else f"Safety policy violations detected: {'; '.join(violations)}"
        )

        return SandboxVerdict(
            is_safe=is_safe,
            commit_allowed=is_safe,
            violations=violations,
            state_delta=delta,
            latency_ms=elapsed_ms,
            explanation=explanation,
        )

    @classmethod
    async def evaluate_async(
        cls,
        delta: StateDelta,
        policy: SandboxPolicy,
        command_or_input: Optional[Any] = None,
    ) -> SandboxVerdict:
        """
        Asynchronously evaluate state delta, supporting async custom validators and SLM judges.
        """
        verdict = cls.evaluate(delta, policy, command_or_input)
        if not verdict.is_safe:
            return verdict

        # Optional Semantic SLM Judge check
        if policy.semantic_judge_fn is not None and callable(policy.semantic_judge_fn):
            try:
                judge_ctx = {
                    "state_delta": delta.model_dump(),
                    "command_or_input": command_or_input,
                }
                if inspect.iscoroutinefunction(policy.semantic_judge_fn):
                    judge_res = await policy.semantic_judge_fn(judge_ctx)
                else:
                    judge_res = policy.semantic_judge_fn(judge_ctx)

                if isinstance(judge_res, dict) and not judge_res.get("is_safe", True):
                    verdict.is_safe = False
                    verdict.commit_allowed = False
                    reason = judge_res.get("reason", "SLM Policy Judge flagged unsafe mutation.")
                    verdict.violations.append(reason)
                    verdict.explanation = f"Safety policy violations detected: {reason}"
            except Exception as e:
                verdict.violations.append(f"Semantic judge error: {str(e)}")
                verdict.is_safe = False
                verdict.commit_allowed = False

        return verdict
