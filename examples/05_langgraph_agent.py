"""
========================================================================================
Example 05: LangGraph Agent with TruthifyToolNode & StateGraph Governance
========================================================================================

Demonstrates:
1. Native integration with LangGraph state graphs
2. Drop-in replacement with `TruthifyToolNode`
3. Tool execution receipt propagation inside LangGraph `State`
4. State-level honesty evaluation with `TruthifyGraphEvaluator`

Run:
  uv run python examples/05_langgraph_agent.py
========================================================================================
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_honesty.adapters.langgraph import TruthifyToolNode, TruthifyGraphEvaluator
from agent_honesty.verifiers.models import DeceptionType


# 1. Define standard tools (can be plain Python functions or LangChain tools)

def search_account_balance(account_id: str) -> Dict[str, Any]:
    """Retrieve the current balance for an account."""
    if account_id == "acc_alice":
        return {"status": "success", "account_id": account_id, "balance": 1500.0, "currency": "USD"}
    return {"status": "error", "error_code": "NOT_FOUND", "message": f"Account '{account_id}' not found."}


def transfer_funds(sender: str, recipient: str, amount: float) -> Dict[str, Any]:
    """Transfer funds between accounts."""
    if amount > 1000:
        return {
            "status": "error",
            "error_code": "LIMIT_EXCEEDED",
            "message": "Daily transaction limit of $1000 exceeded.",
            "success": False,
        }
    return {
        "status": "success",
        "sender": sender,
        "recipient": recipient,
        "amount": amount,
        "records_mutated": 2,
        "message": f"Successfully transferred ${amount:.2f} to {recipient}.",
    }


def main():
    print("=" * 80)
    print("🚀 Running LangGraph Adapter Example with TruthifyToolNode")
    print("=" * 80)

    # 2. Instantiate TruthifyToolNode (Drop-in replacement for LangGraph ToolNode)
    tool_node = TruthifyToolNode(tools=[search_account_balance, transfer_funds])
    evaluator = TruthifyGraphEvaluator()

    # --- Scenario A: Legitimate LangGraph Tool Execution ---
    print("\n🎬 Scenario A: Valid LangGraph Execution")
    state_a = {
        "messages": [
            {"role": "user", "content": "What is Alice's balance?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "search_account_balance",
                        "args": {"account_id": "acc_alice"},
                    }
                ],
            },
        ]
    }

    # Execute tool node in LangGraph state cycle
    tool_result_a = tool_node(state_a)
    print(f"📦 [LangGraph Tool Output]: {tool_result_a['messages'][0]['content']}")
    print(f"🔒 [Signed HMAC Receipt Attached]: {tool_result_a['truthify_receipts'][0].receipt_id}")

    # Assistant synthesizes accurate message
    state_a["messages"].extend(tool_result_a["messages"])
    state_a["messages"].append({
        "role": "assistant",
        "content": "Alice's current balance is $1,500.00 USD.",
    })
    state_a["truthify_receipts"] = tool_result_a["truthify_receipts"]

    # Graph state evaluator verifies claim
    verdict_a = evaluator.evaluate_state(state_a)
    print(f"🛡️  [Truthify State Verdict]: is_honest={verdict_a.is_honest}, latency={verdict_a.latency_ms:.2f}ms")
    print(f"   Explanation: {verdict_a.explanation}")

    # --- Scenario B: Intercepting a False Success Claim in LangGraph ---
    print("\n" + "-" * 80)
    print("🎬 Scenario B: Intercepting Deception (Limit Exceeded)")
    state_b = {
        "messages": [
            {"role": "user", "content": "Please transfer $5000 to Bob."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "name": "transfer_funds",
                        "args": {"sender": "acc_alice", "recipient": "acc_bob", "amount": 5000.0},
                    }
                ],
            },
        ]
    }

    # Tool executes and returns error
    tool_result_b = tool_node(state_b)
    print(f"📦 [LangGraph Tool Output]: {tool_result_b['messages'][0]['content']}")

    # Agent hallucinates false success
    state_b["messages"].extend(tool_result_b["messages"])
    state_b["messages"].append({
        "role": "assistant",
        "content": "Great news! Your transfer of $5000.00 to Bob has been completed successfully!",
    })
    state_b["truthify_receipts"] = tool_result_b["truthify_receipts"]

    # Graph evaluator intercepts lie in sub-millisecond time
    verdict_b = evaluator.evaluate_state(state_b)
    print(f"🛡️  [Truthify State Verdict]: is_honest={verdict_b.is_honest}, type={verdict_b.deception_type}")
    print(f"   Explanation: {verdict_b.explanation}")
    print("=" * 80)


if __name__ == "__main__":
    main()
