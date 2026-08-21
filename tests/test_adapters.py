"""
Tests for Milestone 2 Framework Adapters:
- LangGraph (TruthifyToolNode, TruthifyGraphEvaluator)
- CrewAI (TruthifyCrewCallback, wrap_crew_tool)
- AutoGen (TruthifyAgentInterceptor)
- LlamaIndex (TruthifyLlamaAdapter, wrap_llama_tools)
- Lazy Loading & require_package
"""

import asyncio
import pytest
from typing import Any, Dict

from agent_honesty.actions.models import ExecutionIntegrityError
from agent_honesty.adapters import (
    require_package,
    TruthifyToolNode,
    TruthifyGraphEvaluator,
    TruthifyCrewCallback,
    wrap_crew_tool,
    TruthifyAgentInterceptor,
    TruthifyLlamaAdapter,
    wrap_llama_tools,
)
from agent_honesty.verifiers.models import DeceptionType


# --- 1. Test Lazy Loading & require_package ---

def test_require_package_missing_raises_actionable_error():
    with pytest.raises(ImportError) as excinfo:
        require_package("nonexistent_framework_xyz", extra_name="xyz")
    
    msg = str(excinfo.value)
    assert "nonexistent_framework_xyz" in msg
    assert "pip install \"agent-honesty[xyz]\"" in msg


# --- 2. Test LangGraph Adapter ---

def test_langgraph_tool_node_sync_execution():
    def sample_transfer(sender: str, recipient: str, amount: float) -> Dict[str, Any]:
        return {
            "status": "success",
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "records_mutated": 2,
        }

    tool_node = TruthifyToolNode(tools=[sample_transfer])

    state = {
        "messages": [
            {
                "role": "user",
                "content": "Transfer 100 to bob",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "sample_transfer",
                        "args": {"sender": "alice", "recipient": "bob", "amount": 100},
                    }
                ],
            },
        ]
    }

    result = tool_node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["truthify_verified"] is True
    assert "truthify_receipts" in result
    assert len(result["truthify_receipts"]) == 1
    assert result["truthify_receipts"][0].tool_name == "sample_transfer"
    assert result["truthify_receipts"][0].verify() is True


@pytest.mark.asyncio
async def test_langgraph_tool_node_async_execution():
    async def async_fetch_data(query: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"status": "success", "rows": ["record_1", "record_2"], "row_count": 2}

    tool_node = TruthifyToolNode(tools=[async_fetch_data])

    state = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_async",
                        "name": "async_fetch_data",
                        "args": {"query": "SELECT *"},
                    }
                ],
            }
        ]
    }

    result = await tool_node.ainvoke(state)
    assert len(result["messages"]) == 1
    assert len(result["truthify_receipts"]) == 1
    assert result["truthify_receipts"][0].fact_matrix.status_code == 200
    assert result["truthify_receipts"][0].fact_matrix.is_error is False


def test_langgraph_graph_evaluator_detects_false_success():
    def failing_tool(action: str) -> Dict[str, Any]:
        return {"status": "error", "error_code": "DB_DEADLOCK_500", "message": "Deadlock encountered."}

    tool_node = TruthifyToolNode(tools=[failing_tool])

    state = {
        "messages": [
            {"role": "user", "content": "Execute critical transfer"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "failing_tool", "args": {"action": "pay"}}],
            },
        ]
    }

    step_result = tool_node(state)

    # Next agent node crafts a false success claim
    eval_state = {
        "messages": [
            {"role": "user", "content": "Execute critical transfer"},
            {"role": "assistant", "content": "The critical transfer was successfully processed!"},
        ],
        "truthify_receipts": step_result["truthify_receipts"],
    }

    evaluator = TruthifyGraphEvaluator()
    verdict = evaluator.evaluate_state(eval_state)

    assert verdict.is_honest is False
    assert verdict.deception_type == DeceptionType.FALSE_SUCCESS
    assert "failing_tool" in verdict.explanation


# --- 3. Test CrewAI Adapter ---

def test_crewai_callback_and_tool_wrapping():
    class DummyCrewTool:
        name = "crew_database_tool"
        def _run(self, query: str) -> Dict[str, Any]:
            return {"status": "success", "records_mutated": 1, "message": "Updated"}

    raw_tool = DummyCrewTool()
    audited_tool = wrap_crew_tool(raw_tool)
    callback = TruthifyCrewCallback()

    # Tool execution
    res = audited_tool._run("UPDATE users SET active=1")
    assert res["status"] == "success"

    # Step callback
    class DummyStepOutput:
        tool = "crew_database_tool"
        tool_input = {"query": "UPDATE users SET active=1"}
        text = "Executed query successfully"

    callback(DummyStepOutput())
    assert len(callback.step_history) == 1

    # Task completion
    class DummyTaskOutput:
        raw = "User account was updated successfully."

    verdict = callback.on_task_completed(DummyTaskOutput(), task_description="Update user account")
    assert verdict.is_honest is True


def test_crewai_strict_mode_raises_on_deception():
    def failing_payment(amount: float) -> Dict[str, Any]:
        return {"status": "error", "error_code": "INSUFFICIENT_FUNDS", "message": "Balance too low."}

    callback = TruthifyCrewCallback(strict_mode=True)
    audited_fn = wrap_crew_tool(failing_payment, name="payment_tool", callback=callback)

    audited_fn(500.0)

    class DeceptiveTaskOutput:
        raw = "Payment of $500.00 was successfully processed and confirmed."

    with pytest.raises(ExecutionIntegrityError) as excinfo:
        callback.on_task_completed(DeceptiveTaskOutput(), task_description="Send $500")

    assert "CrewAI execution integrity violation" in str(excinfo.value)


# --- 4. Test AutoGen Adapter ---

def test_autogen_interceptor_and_notice_injection():
    class DummyAutoGenAgent:
        def __init__(self):
            self.function_map = {}
            self.hooks = {}
            self.chat_messages = {}

        def register_hook(self, hookable_method: str, hook: Any):
            self.hooks[hookable_method] = hook

    agent = DummyAutoGenAgent()
    user_proxy = DummyAutoGenAgent()
    user_proxy.chat_messages[agent] = [{"role": "user", "content": "Transfer $1000"}]

    def failing_api(amount: float) -> Dict[str, Any]:
        return {"status": "error", "error_code": "NETWORK_TIMEOUT", "message": "Gateway timeout."}

    agent.function_map["transfer_api"] = failing_api

    interceptor = TruthifyAgentInterceptor(strict_mode=False)
    interceptor.register(agent)

    # Execute wrapped function
    agent.function_map["transfer_api"](1000.0)

    # Agent drafts false message
    outgoing_msg = {"content": "Your transfer of $1000 has been completed."}
    hook = agent.hooks["process_message_before_send"]
    filtered_msg = hook(sender=agent, message=outgoing_msg, recipient=user_proxy, silent=False)

    assert "Truthify Integrity Notice" in filtered_msg["content"]
    assert "Deterministic False Success" in filtered_msg["content"]


def test_autogen_strict_mode_raises_on_deception():
    class DummyAutoGenAgent:
        def __init__(self):
            self.function_map = {}
            self.hooks = {}
            self.chat_messages = {}
        def register_hook(self, hookable_method: str, hook: Any):
            self.hooks[hookable_method] = hook

    agent = DummyAutoGenAgent()
    user_proxy = DummyAutoGenAgent()
    user_proxy.chat_messages[agent] = [{"role": "user", "content": "Run action err_tool"}]

    def error_tool(x: int):
        return {"status": "error", "error_code": "CRASH", "message": "Failed"}

    agent.function_map["err_tool"] = error_tool
    interceptor = TruthifyAgentInterceptor(strict_mode=True)
    interceptor.register(agent)

    agent.function_map["err_tool"](42)

    hook = agent.hooks["process_message_before_send"]
    with pytest.raises(ExecutionIntegrityError) as excinfo:
        hook(sender=agent, message="Action err_tool was successfully completed!", recipient=user_proxy, silent=False)

    assert "AutoGen execution integrity violation" in str(excinfo.value)


# --- 5. Test LlamaIndex Adapter ---

def test_llamaindex_tool_wrapping_and_verification():
    class DummyLlamaTool:
        name = "query_db_tool"
        def call(self, query: str) -> Dict[str, Any]:
            return {"status": "success", "rows": [{"id": 1, "name": "Alice"}], "row_count": 1}

    raw_tool = DummyLlamaTool()
    wrapped_tools, adapter = wrap_llama_tools([raw_tool])

    assert len(wrapped_tools) == 1
    tool = wrapped_tools[0]

    res = tool.call("SELECT * FROM users LIMIT 1")
    assert res["status"] == "success"
    assert len(adapter.receipts) == 1
    assert adapter.receipts[0].fact_matrix.status_code == 200
    assert adapter.receipts[0].fact_matrix.is_error is False

    verdict = adapter.verify_claim(
        user_prompt="Get user 1",
        agent_claim="Found user record for Alice.",
    )
    assert verdict.is_honest is True


@pytest.mark.asyncio
async def test_llamaindex_async_tool_wrapping():
    class DummyAsyncLlamaTool:
        name = "async_llama_tool"
        async def acall(self, prompt: str) -> Dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"status": "error", "error_code": "403_FORBIDDEN", "message": "Access Denied."}

    raw_tool = DummyAsyncLlamaTool()
    wrapped_tools, adapter = wrap_llama_tools([raw_tool])
    tool = wrapped_tools[0]

    res = await tool.acall("get_admin_data")
    assert res["status"] == "error"
    assert len(adapter.receipts) == 1

    verdict = adapter.verify_claim(
        user_prompt="Get admin data",
        agent_claim="Here is the admin data: all settings were successfully completed.",
    )
    assert verdict.is_honest is False
    assert verdict.deception_type == DeceptionType.FALSE_SUCCESS
