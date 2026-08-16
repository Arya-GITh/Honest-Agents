# agent-honesty

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://github.com/Arya-GITh/Truthify/actions/workflows/ci.yml/badge.svg)](https://github.com/Arya-GITh/Truthify/actions)

**Execution Integrity & Trajectory Authenticity Governance Middleware for Autonomous AI Agents**

*Stop AI agents from hallucinating false success, fabricating data on empty returns, or concealing tool failures.*

</div>

---

## 💡 Why agent-honesty?

When autonomous AI agents call tools, databases, and APIs:
- **The database fails with an HTTP 500 error** $\rightarrow$ The LLM hallucinates: *"Your transaction is confirmed!"*
- **The search returns 0 matching rows** $\rightarrow$ The LLM invents fake accounts or numbers.
- **The API returns a soft-failure (`{"status": "error"}`)** $\rightarrow$ The LLM misreads it as success.

Traditional text guardrails are **blind** to execution reality because the LLM's English response looks polite, fluent, and convincing. 

`agent-honesty` operates at the **machine execution boundary**: capturing cryptographic HMAC-SHA256 receipts from raw OS/network returns, cross-examining agent claims via a Two-Tier verification engine, and forcing in-scratchpad self-corrections live before deceptive tokens reach the user.

---

## ⚡ Quickstart

### 1. Installation

```bash
pip install agent-honesty
```

---

### 2. Wrap Any Python Tool with `@audit_tool`

```python
from agent_honesty import audit_tool, HonestyAuditor, VerificationRouter

# Wrap your tool functions:
@audit_tool(name="transfer_funds")
def transfer_funds(sender: str, recipient: str, amount: float):
    # If this tool returns an error, soft-failure, or empty payload...
    return {"status": "error", "error_code": "DB_DEADLOCK_500", "message": "Deadlock detected."}

# Run under HonestyAuditor supervision:
with HonestyAuditor() as auditor:
    tool_res = transfer_funds("alice", "bob", 150.0)
    
    # What the agent tries to say to the user:
    agent_claim = "Your transfer of $150 was successfully completed and confirmed!"
    
    # Verify the claim against the cryptographic machine receipt (<0.1ms):
    router = VerificationRouter()
    verdict = router.verify(
        user_prompt="Transfer $150 from Alice to Bob",
        agent_claim=agent_claim,
        receipts=auditor.receipts,
    )
    
    print(verdict.is_honest)       # False
    print(verdict.deception_type)  # DeceptionType.FALSE_SUCCESS
    print(verdict.explanation)     # "Deterministic False Success: Step 'transfer_funds' failed with DB_DEADLOCK_500..."
```

---

### 3. Model Context Protocol (MCP) Interception

Audit any Model Context Protocol tool client seamlessly:

```python
from agent_honesty import MCPClientProxy, HonestyAuditor, VerificationRouter

raw_mcp_client = get_my_mcp_client()
audited_client = MCPClientProxy(raw_mcp_client)

async with HonestyAuditor() as auditor:
    # Automatically intercepts MCP JSON-RPC calls and creates signed HMAC receipts
    result = await audited_client.call_tool("execute_sql", {"query": "UPDATE accounts SET balance = 500;"})
```

---

### 4. Automatic In-Scratchpad Self-Correction ($N=2$ Hard Cap)

Automatically prompt the agent in its private reasoning scratchpad to correct its wording before the user ever sees a lie:

```python
from agent_honesty import SelfCorrectionLoop, ActionPolicy

reprompter = SelfCorrectionLoop(max_reprompts=2)

async def agent_llm_callback(system_feedback: str) -> str:
    # Private reprompt injected directly into the LLM context:
    chat_history.append({"role": "system", "content": system_feedback})
    return await call_my_llm(chat_history)

result = await reprompter.execute_policy_async(
    user_prompt="Transfer $150 from Alice to Bob",
    initial_claim=initial_agent_draft,
    receipts=auditor.receipts,
    reprompt_callback=agent_llm_callback,
    policy=ActionPolicy.REPROMPT,
)

# Honest final answer delivered to the user:
print(result.delivered_claim)
```

---

## 🏛️ Architecture

```
                               THE AGENT-HONESTY PIPELINE
                               
   1. Execution Boundary   2. Ground Truth       3. Two-Tier Verification       4. Gated Streaming & Action
┌─────────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│ • @audit_tool       │  │ • Normalized     │  │ • Tier 1: Invariant Rules │  │ • Gated Token Buffer      │
│ • MCPClientProxy    │─>│   FactMatrix     │─>│   (<0.1ms, $0 token cost) │─>│ • In-Scratchpad Reprompt  │
│ (Catches raw return)│  │ • HMAC-SHA256    │  │ • Tier 2: Semantic SLM   │  │   (N=2 Hard Limit)        │
│                     │  │   Signed Receipt │  │   (~50ms, Language/Math)  │  │ • Deterministic Fallback  │
└─────────────────────┘  └──────────────────┘  └───────────────────────────┘  └───────────────────────────┘
```

### Two-Tier Verification Waterfall
1. **Tier 1 (Deterministic Engine)**: Evaluates strict mathematical invariants directly against the `FactMatrix` in **$<0.1\text{ms}$ at $\$0$ token cost** (resolving ~80% of calls instantly).
2. **Tier 2 (Fast Semantic SLM Auditor)**: Evaluates natural language nuances, prompt arithmetic (`5300-2000 = 3300`), and entity alias mappings using a fast local SLM (Qwen2.5 / Phi-3.5) or cloud endpoint (Claude Haiku / GPT-4o-mini).

---

## 🛡️ Deception Modes Detected

| Deception Mode | Description |
| :--- | :--- |
| **`FALSE_SUCCESS`** | Claiming an action succeeded when the tool returned an HTTP error, deadlock, timeout, or soft-error. |
| **`OUTPUT_FABRICATION`** | Inventing specific entity names or accounts when the query returned 0 rows (`[]`). |
| **`PARAMETER_MUTATION`** | Mutating tool input arguments contrary to user prompt constraints (e.g. transferring $5000 instead of $50). |
| **`GOAL_DRIFT`** | Executing actions completely unrelated to prompt constraints. |

---

## 📦 Examples
 
 Runnable real-world examples are available in the [`examples/`](https://github.com/Arya-GITh/Truthify/tree/main/examples) directory:
 - [`01_basic_audit_tool.py`](https://github.com/Arya-GITh/Truthify/blob/main/examples/01_basic_audit_tool.py): Basic function wrapping with `@audit_tool`, soft-error detection, and HMAC receipt verification.
 - [`02_mcp_client_interceptor.py`](https://github.com/Arya-GITh/Truthify/blob/main/examples/02_mcp_client_interceptor.py): Model Context Protocol (MCP) JSON-RPC tool server proxy auditing.
 - [`03_live_local_agent.py`](https://github.com/Arya-GITh/Truthify/blob/main/examples/03_live_local_agent.py): 100% Local Autonomous Agent using Ollama (`qwen3:latest` primary + `qwen2.5:0.5b` SLM judge) with a real SQLite database on disk.
 - [`04_live_gemini_agent.py`](https://github.com/Arya-GITh/Truthify/blob/main/examples/04_live_gemini_agent.py): Cloud Autonomous Agent using Google Gemini (`gemini-flash-latest`) + Local 8B SLM Judge (`qwen3:latest`) with a real SQLite database on disk.

---

## 📄 License

Apache 2.0 License. See [LICENSE](https://github.com/Arya-GITh/Truthify/blob/main/LICENSE) for details.
