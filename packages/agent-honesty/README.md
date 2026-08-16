# agent-honesty

[![PyPI Version](https://img.shields.io/pypi/v/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://github.com/Arya-GITh/Truthify/actions/workflows/ci.yml/badge.svg)](https://github.com/Arya-GITh/Truthify/actions)

`agent-honesty` is a lightweight, zero-heavy-dependency Python SDK providing execution integrity and trajectory authenticity governance for autonomous AI agents.

It captures cryptographic machine execution receipts from raw tool and API outputs, verifies model claims via a sub-millisecond two-tier engine, and enforces in-scratchpad self-correction before deceptive outputs reach users.

---

## Installation

```bash
pip install agent-honesty
```

---

## Core Features

* **Universal Tool Interception**: Decorate any Python function with `@audit_tool` or wrap any Model Context Protocol server with `MCPClientProxy`.
* **Cryptographic HMAC Receipts**: Signs raw execution state (`is_error`, `status_code`, `records_mutated`, `is_empty`) with HMAC-SHA256.
* **Deep Soft-Error Normalization**: Detects application-layer failures disguised inside HTTP 200 responses.
* **Two-Tier Verification Waterfall**:
  * **Tier 1 (Deterministic, <0.5 ms, $0 Token Cost)**: Evaluates mathematical and execution invariants.
  * **Tier 2 (Semantic SLM Auditor, ~50 ms)**: Evaluates complex prompt arithmetic, entity references, and paraphrasing.
* **Gated Streaming & Self-Correction**: Buffers high-risk output tokens and prompts models to self-correct in private reasoning scratchpads with a strict $N=2$ safety cap and deterministic fallback.

---

## Usage

### 1. In-Process Function Auditing

```python
from agent_honesty import audit_tool, HonestyAuditor, VerificationRouter

@audit_tool(name="execute_payment")
def execute_payment(account_id: str, amount: float):
    # Simulating a backend failure:
    return {"status": "error", "error_code": "DB_DEADLOCK_500", "message": "Transaction aborted."}

with HonestyAuditor() as auditor:
    # Tool executes and generates an unalterable HMAC-SHA256 receipt
    execute_payment("acc_1001", 250.0)
    
    # Model drafts response
    agent_claim = "Payment of $250.00 was successfully processed."
    
    # Verify claim against machine ground truth
    router = VerificationRouter()
    verdict = router.verify(
        user_prompt="Send $250 to account 1001",
        agent_claim=agent_claim,
        receipts=auditor.receipts,
    )
    
    print(verdict.is_honest)       # False
    print(verdict.deception_type)  # DeceptionType.FALSE_SUCCESS
    print(verdict.explanation)     # "Deterministic False Success: Step 'execute_payment' failed..."
```

---

### 2. Model Context Protocol (MCP) Client Auditing

```python
from agent_honesty import MCPClientProxy, HonestyAuditor, VerificationRouter

raw_mcp_client = get_mcp_client()
client = MCPClientProxy(raw_mcp_client)

async with HonestyAuditor() as auditor:
    result = await client.call_tool("execute_sql", {"query": "UPDATE accounts SET balance = 500;"})
```

---

### 3. In-Scratchpad Self-Correction ($N=2$ Hard Cap)

```python
from agent_honesty import SelfCorrectionLoop, ActionPolicy

reprompter = SelfCorrectionLoop(max_reprompts=2)

async def agent_llm_callback(system_feedback: str) -> str:
    # Injects private system correction into LLM reasoning scratchpad
    chat_history.append({"role": "system", "content": system_feedback})
    return await call_model(chat_history)

result = await reprompter.execute_policy_async(
    user_prompt="Transfer $250 to acc_1001",
    initial_claim=initial_draft,
    receipts=auditor.receipts,
    reprompt_callback=agent_llm_callback,
    policy=ActionPolicy.REPROMPT,
)

# Verified, honest claim delivered to user:
print(result.delivered_claim)
```

---

## Deception Modes Detected

| Deception Type | Description |
| :--- | :--- |
| **`FALSE_SUCCESS`** | Asserting an action succeeded when the tool returned an error, deadlock, timeout, or soft-failure. |
| **`OUTPUT_FABRICATION`** | Hallucinating records, entity identifiers, or values when the query returned empty results (`[]`). |
| **`PARAMETER_MUTATION`** | Modifying tool arguments contrary to user constraints (e.g., executing with modified recipient IDs or amounts). |
| **`GOAL_DRIFT`** | Executing actions completely unrelated to prompt constraints. |

---

## License

Apache License 2.0. See [LICENSE](https://github.com/Arya-GITh/Truthify/blob/main/LICENSE) for details.
