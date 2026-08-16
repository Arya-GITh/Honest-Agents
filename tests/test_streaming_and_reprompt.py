import pytest
from typing import AsyncIterator, List
from agent_honesty import (
    ActionPolicy,
    ActionResult,
    ExecutionIntegrityError,
    SelfCorrectionLoop,
    DualChannelStreamManager,
    VerificationRouter,
    HMACReceipt,
    FactMatrix,
)


def create_mock_receipt(
    is_error: bool = False,
    status_code: int = 200,
    error_message: str = None,
    tool_name: str = "transfer_balance",
) -> HMACReceipt:
    fm = FactMatrix(
        is_error=is_error,
        status_code=status_code,
        error_message=error_message,
        payload_sha256="test-sha",
    )
    receipt = HMACReceipt(
        execution_id="exec-stream-1",
        tool_name=tool_name,
        timestamp="2026-08-16T12:00:00Z",
        duration_ms=10.0,
        args_hash="args-sha",
        kwargs_hash="kwargs-sha",
        fact_matrix=fm,
    )
    receipt.sign()
    return receipt


@pytest.mark.asyncio
async def test_self_correction_loop_honest_pass():
    loop = SelfCorrectionLoop()
    receipt = create_mock_receipt(is_error=False)

    result = await loop.execute_policy_async(
        user_prompt="Run status check",
        initial_claim="The status check completed successfully.",
        receipts=[receipt],
    )
    assert result.verdict.is_honest is True
    assert result.reprompt_count == 0
    assert result.overridden is False
    assert result.delivered_claim == "The status check completed successfully."


@pytest.mark.asyncio
async def test_self_correction_loop_reprompts_and_succeeds():
    loop = SelfCorrectionLoop()
    receipt = create_mock_receipt(is_error=True, status_code=500, error_message="Deadlock on accounts")

    # Mock agent callback: corrects itself on reprompt 1
    async def mock_agent_reprompt(feedback: str) -> str:
        assert "[System Honesty Correction:" in feedback
        return "I apologize, the transfer failed due to a database deadlock."

    result = await loop.execute_policy_async(
        user_prompt="Transfer $100 to Bob",
        initial_claim="Successfully processed the transfer of $100 to Bob.",
        receipts=[receipt],
        reprompt_callback=mock_agent_reprompt,
        policy=ActionPolicy.REPROMPT,
    )
    assert result.verdict.is_honest is True
    assert result.reprompt_count == 1
    assert result.overridden is False
    assert "transfer failed" in result.delivered_claim


@pytest.mark.asyncio
async def test_self_correction_loop_n2_hard_cap_and_deterministic_override():
    loop = SelfCorrectionLoop(max_reprompts=2)
    receipt = create_mock_receipt(is_error=True, status_code=500, error_message="Deadlock on accounts")

    call_count = 0

    # Stubborn agent callback: refuses to admit failure
    async def stubborn_agent_reprompt(feedback: str) -> str:
        nonlocal call_count
        call_count += 1
        return "No, everything was definitely confirmed and completed successfully."

    result = await loop.execute_policy_async(
        user_prompt="Transfer $100 to Bob",
        initial_claim="Successfully processed the transfer of $100 to Bob.",
        receipts=[receipt],
        reprompt_callback=stubborn_agent_reprompt,
        policy=ActionPolicy.REPROMPT,
    )
    # Enforced hard limit: called exactly 2 times, then bypassed!
    assert call_count == 2
    assert result.reprompt_count == 2
    assert result.overridden is True
    assert "System Notice: The requested action 'transfer_balance' failed with Deadlock on accounts" in result.delivered_claim


@pytest.mark.asyncio
async def test_auto_correct_policy():
    loop = SelfCorrectionLoop()
    receipt = create_mock_receipt(is_error=True, status_code=503, error_message="Service Unavailable")

    result = await loop.execute_policy_async(
        user_prompt="Fetch records",
        initial_claim="Successfully fetched all user records.",
        receipts=[receipt],
        policy=ActionPolicy.AUTO_CORRECT,
    )
    assert result.reprompt_count == 0
    assert result.overridden is True
    assert "Service Unavailable" in result.delivered_claim
    assert result.policy_applied == ActionPolicy.AUTO_CORRECT


@pytest.mark.asyncio
async def test_block_policy_raises_exception():
    loop = SelfCorrectionLoop()
    receipt = create_mock_receipt(is_error=True, status_code=500, error_message="Internal Error")

    with pytest.raises(ExecutionIntegrityError) as exc_info:
        await loop.execute_policy_async(
            user_prompt="Execute SQL",
            initial_claim="Successfully executed and updated rows.",
            receipts=[receipt],
            policy=ActionPolicy.BLOCK,
        )
    assert "Execution Integrity Violation" in str(exc_info.value)
    assert exc_info.value.verdict.is_honest is False


@pytest.mark.asyncio
async def test_dual_channel_stream_manager_gated_reprompt():
    stream_mgr = DualChannelStreamManager()
    receipt = create_mock_receipt(is_error=True, status_code=500, error_message="Database Lock")

    # Initial deceptive token stream
    async def mock_deceptive_stream() -> AsyncIterator[str]:
        for token in ["Your ", "transfer ", "was ", "successfully ", "completed."]:
            yield token

    # Corrected reprompt stream factory
    def mock_reprompt_stream_factory(feedback: str) -> AsyncIterator[str]:
        async def stream() -> AsyncIterator[str]:
            for token in ["The ", "transfer ", "failed ", "due ", "to ", "an ", "error."]:
                yield token
        return stream()

    delivered_chunks: List[str] = []
    async for chunk in stream_mgr.stream_with_integrity(
        token_stream=mock_deceptive_stream(),
        user_prompt="Transfer $50",
        receipts=[receipt],
        reprompt_stream_factory=mock_reprompt_stream_factory,
        is_high_risk=True,
    ):
        delivered_chunks.append(chunk)

    full_delivered = "".join(delivered_chunks)
    assert "The transfer failed due to an error." == full_delivered
    assert "successfully" not in full_delivered


@pytest.mark.asyncio
async def test_dual_channel_stream_manager_fast_path():
    stream_mgr = DualChannelStreamManager()
    receipt = create_mock_receipt(is_error=False)

    async def mock_read_stream() -> AsyncIterator[str]:
        for token in ["Here ", "is ", "the ", "weather: ", "Sunny."]:
            yield token

    delivered_chunks: List[str] = []
    async for chunk in stream_mgr.stream_with_integrity(
        token_stream=mock_read_stream(),
        user_prompt="Get weather",
        receipts=[receipt],
        is_high_risk=False,
    ):
        delivered_chunks.append(chunk)

    assert "".join(delivered_chunks) == "Here is the weather: Sunny."


@pytest.mark.asyncio
async def test_self_correction_loop_callback_exception_failsafe():
    """Edge Case: If the agent's LLM callback crashes during reprompt, deliver fallback without crashing."""
    loop = SelfCorrectionLoop()
    receipt = create_mock_receipt(is_error=True, status_code=500, error_message="Database Lock")

    def crashing_reprompt_callback(feedback: str):
        raise ConnectionError("LLM API endpoint timeout")

    result = await loop.execute_policy_async(
        user_prompt="Transfer $100",
        initial_claim="Transfer confirmed successfully.",
        receipts=[receipt],
        reprompt_callback=crashing_reprompt_callback,
    )
    assert result.overridden is True
    assert "System Notice:" in result.delivered_claim

