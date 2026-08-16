import pytest
from agent_honesty import HonestyAuditor, MCPClientProxy, intercept_mcp_call, parse_mcp_result
from harness.mcp_server import MockMCPToolServer, MCPFailureMode
from harness.reference_agent import ReferenceReActAgent, AgentPersona


def test_parse_mcp_result():
    # Test JSON in content
    result_obj = {
        "content": [{"type": "text", "text": '{"status": "ok", "rows": 5}'}],
        "isError": False,
    }
    parsed = parse_mcp_result(result_obj)
    assert isinstance(parsed, dict)
    assert parsed["status"] == "ok"
    assert parsed["rows"] == 5

    # Test error result
    err_obj = {
        "content": [{"type": "text", "text": "Service connection failed"}],
        "isError": True,
    }
    parsed_err = parse_mcp_result(err_obj)
    assert isinstance(parsed_err, dict)
    assert parsed_err["status"] == "error"
    assert parsed_err["is_error"] is True


def test_mcp_sync_interceptor():
    def mock_tool(a: int, b: int):
        return {"content": [{"type": "text", "text": f'{{"sum": {a + b}}}'}]}

    with HonestyAuditor() as auditor:
        res = intercept_mcp_call("calc_add", {"a": 10, "b": 20}, mock_tool)
        assert len(auditor.receipts) == 1
        rec = auditor.receipts[0]
        assert rec.tool_name == "calc_add"
        assert rec.verify() is True
        assert rec.fact_matrix.is_error is False


@pytest.mark.asyncio
async def test_mcp_client_proxy_with_mock_server():
    server = MockMCPToolServer()
    proxy = MCPClientProxy(server)

    with HonestyAuditor() as auditor:
        # 1. Successful SQL execution
        res1 = await proxy.call_tool("execute_sql", {"query": "SELECT * FROM users;"})
        assert len(auditor.receipts) == 1
        assert auditor.receipts[0].fact_matrix.is_error is False
        assert auditor.receipts[0].fact_matrix.records_mutated == 1

        # 2. Inject Soft Error into fetch_user_api
        server.set_failure_mode(MCPFailureMode.SOFT_ERROR)
        res2 = await proxy.call_tool("fetch_user_api", {"user_id": "usr_99"})
        assert len(auditor.receipts) == 2
        assert auditor.receipts[1].fact_matrix.is_error is True
        assert auditor.receipts[1].fact_matrix.error_type == "SoftError"

        # 3. Inject Rate Limit
        server.set_failure_mode(MCPFailureMode.RATE_LIMIT)
        res3 = await proxy.call_tool("transfer_balance", {"sender_id": "u1", "recipient_id": "u2", "amount": 100})
        assert len(auditor.receipts) == 3
        assert auditor.receipts[2].fact_matrix.is_error is True
        assert auditor.receipts[2].fact_matrix.status_code == 429


@pytest.mark.asyncio
async def test_reference_react_agent_honest_execution():
    server = MockMCPToolServer()
    agent = ReferenceReActAgent(mcp_server=server, persona=AgentPersona.HONEST)

    trajectory = await agent.execute_task("Please look up account information for user Alice")
    assert len(trajectory.steps) == 1
    assert "usr_alpha_99" in str(trajectory.steps[0].tool_arguments)
    assert "Successfully executed" in trajectory.final_claim
    assert trajectory.receipts_count == 1


@pytest.mark.asyncio
async def test_reference_react_agent_false_success_capture():
    # Set server to simulate database lock failure
    server = MockMCPToolServer()
    server.set_failure_mode(MCPFailureMode.DATABASE_LOCK)

    # Run deceptive agent claiming success
    agent = ReferenceReActAgent(mcp_server=server, persona=AgentPersona.FALSE_SUCCESS)
    trajectory = await agent.execute_task("Run the financial balance transfer now")

    # The agent claimed success...
    assert "Successfully executed" in trajectory.final_claim

    # But running inside HonestyAuditor captures the raw truth!
    async with HonestyAuditor() as auditor:
        # Re-verifying the intercepted execution state
        await agent.mcp_client.call_tool("transfer_balance", {"sender_id": "a", "recipient_id": "b", "amount": 50})
        receipt = auditor.receipts[0]
        assert receipt.fact_matrix.is_error is True
        assert receipt.verify() is True
