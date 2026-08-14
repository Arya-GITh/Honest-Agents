import pytest
import asyncio
from agent_honesty import audit_tool, HonestyAuditor, ToolExecutionRecord

@audit_tool
def sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@audit_tool(name="custom_async_multiply")
async def async_multiply(x: float, y: float) -> float:
    """Multiply two numbers asynchronously."""
    await asyncio.sleep(0.01)
    return x * y

@audit_tool
def failing_sync_tool():
    raise ValueError("Database connection refused")

@audit_tool
async def failing_async_tool():
    await asyncio.sleep(0.01)
    raise RuntimeError("API Rate Limit Exceeded")


def test_sync_tool_execution():
    with HonestyAuditor() as auditor:
        res = sync_add(3, 4)
        assert res == 7
        assert len(auditor.records) == 1
        
        rec: ToolExecutionRecord = auditor.records[0]
        assert rec.tool_name == "sync_add"
        assert rec.args == [3, 4]
        assert rec.kwargs == {}
        assert rec.status == "success"
        assert rec.result == 7
        assert rec.duration_ms > 0


@pytest.mark.asyncio
async def test_async_tool_execution():
    async with HonestyAuditor() as auditor:
        res = await async_multiply(2.5, 4.0)
        assert res == 10.0
        assert len(auditor.records) == 1
        
        rec: ToolExecutionRecord = auditor.records[0]
        assert rec.tool_name == "custom_async_multiply"
        assert rec.args == [2.5, 4.0]
        assert rec.status == "success"
        assert rec.result == 10.0
        assert rec.duration_ms > 0


def test_failing_sync_tool_recording():
    with HonestyAuditor() as auditor:
        with pytest.raises(ValueError, match="Database connection refused"):
            failing_sync_tool()
        
        assert len(auditor.records) == 1
        rec = auditor.records[0]
        assert rec.tool_name == "failing_sync_tool"
        assert rec.status == "error"
        assert rec.error_type == "ValueError"
        assert "Database connection refused" in rec.error


@pytest.mark.asyncio
async def test_failing_async_tool_recording():
    async with HonestyAuditor() as auditor:
        with pytest.raises(RuntimeError, match="API Rate Limit Exceeded"):
            await failing_async_tool()
            
        assert len(auditor.records) == 1
        rec = auditor.records[0]
        assert rec.tool_name == "failing_async_tool"
        assert rec.status == "error"
        assert rec.error_type == "RuntimeError"
        assert "API Rate Limit Exceeded" in rec.error


def test_docstring_and_metadata_preservation():
    assert sync_add.__name__ == "sync_add"
    assert sync_add.__doc__ == "Add two numbers."
