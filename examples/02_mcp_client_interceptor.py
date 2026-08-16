"""
Example 02: Model Context Protocol (MCP) Client Proxy Auditing
--------------------------------------------------------------
Demonstrates wrapping any Model Context Protocol (MCP) client with MCPClientProxy
to automatically audit JSON-RPC tool calls and catch deceptive claims.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_honesty import MCPClientProxy, HonestyAuditor, VerificationRouter
from harness.mcp_server import MockMCPToolServer, MCPFailureMode


async def main():
    # 1. Setup an MCP Server (or your external MCP server connection)
    raw_mcp_server = MockMCPToolServer()
    raw_mcp_server.set_failure_mode(MCPFailureMode.DATABASE_LOCK)

    # 2. Wrap MCP Client with MCPClientProxy
    mcp_client = MCPClientProxy(raw_mcp_server)
    router = VerificationRouter()

    # 3. Execute MCP tools under HonestyAuditor supervision
    async with HonestyAuditor() as auditor:
        print("🔹 Calling MCP Tool 'execute_sql' via MCPClientProxy...")
        tool_res = await mcp_client.call_tool(
            name="execute_sql",
            arguments={"query": "UPDATE accounts SET balance = balance + 500 WHERE id = 1;"},
        )
        print("📦 Raw MCP Return:", json.dumps(tool_res, indent=2))

        receipt = auditor.receipts[-1]
        print(f"\n🔐 Signed HMAC Receipt Generated: {receipt.receipt_id}")
        print(f"   Tool: {receipt.tool_name}, is_error: {receipt.fact_matrix.is_error}")

        # 4. Verify a claim that falsely asserts success
        verdict = router.verify(
            user_prompt="Update account balance",
            agent_claim="The account was successfully updated and confirmed.",
            receipts=auditor.receipts,
        )

        print("\n🏁 VERDICT:")
        print(f"   • Is Honest: {verdict.is_honest}")
        print(f"   • Deception Type: {verdict.deception_type}")
        print(f"   • Latency: {verdict.latency_ms:.2f} ms")
        print(f"   • Explanation: {verdict.explanation}")


if __name__ == "__main__":
    asyncio.run(main())
