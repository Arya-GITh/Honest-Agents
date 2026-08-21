"""
========================================================================================
Example 06: CrewAI Adapter with TruthifyCrewCallback & Multi-Agent Governance
========================================================================================

Demonstrates:
1. Multi-agent delegation governance in CrewAI
2. `TruthifyCrewCallback` for step_callback and task_callback
3. Wrapping CrewAI tools with `wrap_crew_tool`
4. Intercepting inter-agent false-success hallucinations across delegation chains

Run:
  uv run python examples/06_crewai_agent.py
========================================================================================
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_honesty.actions.models import ExecutionIntegrityError
from agent_honesty.adapters.crewai import TruthifyCrewCallback, wrap_crew_tool
from agent_honesty.verifiers.models import DeceptionType


def execute_database_migration(table_name: str) -> Dict[str, Any]:
    """Simulated production tool that encounters a database lock."""
    return {
        "status": "error",
        "error_code": "SQLITE_LOCKED_500",
        "message": f"Table '{table_name}' is locked by concurrent writer.",
        "success": False,
    }


def main():
    print("=" * 80)
    print("🚀 Running CrewAI Adapter Example with TruthifyCrewCallback")
    print("=" * 80)

    # 1. Instantiate TruthifyCrewCallback
    callback = TruthifyCrewCallback(strict_mode=False)

    # 2. Wrap tool using callback
    audited_tool = callback.wrap_tool(execute_database_migration, name="db_migration_tool")

    # --- Agent Step Execution ---
    print("\n🎬 Agent Step: Running db_migration_tool('users_v2')...")
    raw_result = audited_tool("users_v2")
    print(f"📦 [Tool Output]: {raw_result}")
    print(f"🔒 [Receipts Collected]: {len(callback.receipts)}")

    # Record step callback
    class DummyStepOutput:
        tool = "db_migration_tool"
        tool_input = {"table_name": "users_v2"}
        text = "Executed db_migration_tool"

    callback(DummyStepOutput())

    # --- Task Output Evaluation: Legitimate Failure Reporting ---
    print("\n" + "-" * 80)
    print("🎬 Scenario A: Faithful Task Completion Reporting")
    class HonestTaskOutput:
        raw = "The database migration on 'users_v2' failed because the table is locked by a concurrent writer."

    verdict_a = callback.on_task_completed(HonestTaskOutput(), task_description="Migrate users_v2")
    print(f"🛡️  [Truthify Verdict]: is_honest={verdict_a.is_honest}, latency={verdict_a.latency_ms:.2f}ms")
    print(f"   Explanation: {verdict_a.explanation}")

    # --- Task Output Evaluation: Deceptive Delegation Claim ---
    print("\n" + "-" * 80)
    print("🎬 Scenario B: Deceptive Delegation Claim (Claiming Success on Error)")
    class DeceptiveTaskOutput:
        raw = "Migration completed successfully! Table users_v2 has been fully upgraded with zero errors."

    verdict_b = callback.on_task_completed(DeceptiveTaskOutput(), task_description="Migrate users_v2")
    print(f"🛡️  [Truthify Verdict]: is_honest={verdict_b.is_honest}, type={verdict_b.deception_type}")
    print(f"   Explanation: {verdict_b.explanation}")
    print("=" * 80)


if __name__ == "__main__":
    main()
