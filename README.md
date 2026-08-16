# Truthify: Execution Integrity & Trajectory Authenticity Platform

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://github.com/Arya-GITh/Truthify/actions/workflows/ci.yml/badge.svg)](https://github.com/Arya-GITh/Truthify/actions)

**Open-source governance middleware that guarantees AI agents never lie, fabricate data on empty returns, or conceal tool execution failures.**

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [Examples](#-runnable-examples) • [Roadmap](#-milestone-roadmap) • [Documentation](packages/agent-honesty/README.md)

</div>

---

## 🌟 Overview

Autonomous AI agents frequently experience silent runtime failures when calling tools, APIs, and databases:
* **The database deadlocks (500 Error)** $\rightarrow$ The LLM hallucinates: *"Your transaction is confirmed!"*
* **The search query returns 0 rows (`[]`)** $\rightarrow$ The LLM invents fake accounts or numbers.
* **The API returns a soft-failure (`{"status": "error"}`)** $\rightarrow$ The LLM misinterprets it as success.

Traditional text guardrails are **blind** to execution reality because the LLM's English response looks polite, fluent, and convincing.

**Truthify (`agent-honesty`)** operates at the **machine execution boundary**: capturing cryptographic HMAC-SHA256 receipts from raw OS/network returns, cross-examining agent claims via a Two-Tier verification engine, and forcing in-scratchpad self-corrections live before deceptive tokens reach the user.

---

## ⚡ Quickstart

### 1. Installation

```bash
pip install agent-honesty
```

### 2. Basic Example

```python
from agent_honesty import audit_tool, HonestyAuditor, VerificationRouter

@audit_tool(name="transfer_funds")
def transfer_funds(sender: str, recipient: str, amount: float):
    # Simulated API failure:
    return {"status": "error", "error_code": "DB_DEADLOCK_500", "message": "Deadlock detected."}

with HonestyAuditor() as auditor:
    # Tool executes and generates an unforgeable HMAC-SHA256 receipt:
    transfer_funds("alice", "bob", 150.0)
    
    # What the agent tries to say:
    agent_claim = "Your transfer was successfully completed and confirmed!"
    
    # Cross-examine against machine truth in <0.1ms:
    router = VerificationRouter()
    verdict = router.verify("Transfer $150 from Alice to Bob", agent_claim, auditor.receipts)
    
    print(verdict.is_honest)       # False
    print(verdict.deception_type)  # DeceptionType.FALSE_SUCCESS
    print(verdict.explanation)     # "Deterministic False Success: Step 'transfer_funds' failed with DB_DEADLOCK_500..."
```

---

## 🏛️ Architecture & Verification Waterfall

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

1. **Tier 1 (Deterministic Rule Engine)**: Sub-millisecond rule matcher evaluating strict mathematical invariants directly against the `FactMatrix` in **$<0.1\text{ms}$ at $\$0$ token cost** (resolving ~80% of calls instantly).
2. **Tier 2 (Fast Semantic SLM Auditor)**: Evaluates natural language nuances, prompt arithmetic (`5300-2000 = 3300`), and entity alias mappings using a fast local SLM (Qwen2.5 / Phi-3.5) or cloud endpoint (Claude Haiku / GPT-4o-mini).

---

## 📂 Repository Layout

```
Truthify/
├── packages/
│   └── agent-honesty/             <- Core publishable Python SDK package
│       ├── src/agent_honesty/
│       │   ├── interceptors/      <- @audit_tool, HonestyAuditor, MCPClientProxy
│       │   ├── receipts/          <- FactMatrix, PayloadNormalizer, HMACReceipt
│       │   ├── verifiers/         <- Tier 1 Deterministic, Tier 2 Semantic SLM, Router
│       │   ├── actions/           <- SelfCorrectionLoop, Fallback Override, Policies
│       │   └── streaming/         <- DualChannelStreamManager
│       ├── pyproject.toml         <- Package configuration & PyPI metadata
│       └── README.md              <- SDK documentation
├── examples/                      <- Runnable real-world scripts
│   ├── 01_basic_audit_tool.py     <- Basic tool wrapping & receipt verification
│   ├── 02_mcp_client_interceptor.py <- Model Context Protocol (MCP) tool auditing
│   └── 03_live_ollama_agent.py    <- Live ReAct agent with local Ollama & live self-correction
├── harness/                       <- Mock MCP server & Reference ReAct agent
├── tests/                         <- 40-test automated verification test suite
└── .github/workflows/             <- Automated CI/CD & PyPI publishing
```

---

## 📦 Runnable Examples

Check out the [`examples/`](examples/) directory:

- **Run Basic Tool Auditing**:
  ```bash
  uv run python examples/01_basic_audit_tool.py
  ```
- **Run MCP Client Interceptor**:
  ```bash
  uv run python examples/02_mcp_client_interceptor.py
  ```
- **Run Live Local Ollama Agent with Real-Time Self-Correction**:
  ```bash
  uv run python examples/03_live_ollama_agent.py
  ```

---

## 🗺️ Milestone Roadmap

- [x] **Milestone 1: Core Engine & SDK (`agent-honesty v0.1.0`)**
  - Tool Decorator & Context Isolation (`@audit_tool`, `HonestyAuditor`)
  - Payload Schema Normalizer & HMAC-SHA256 Receipt Engine
  - MCP Interceptor Proxy & Multi-Persona ReAct Harness
  - Two-Tier Verification Engine (Tier 1 Deterministic + Tier 2 Semantic SLM)
  - Dual-Channel Streaming & In-Scratchpad Self-Correction with $N=2$ Hard Cap
- [ ] **Milestone 2: Multi-Framework Adapters**
  - LangGraph, AutoGen, CrewAI, and LlamaIndex middleware hooks.
- [ ] **Milestone 3: `DeceptionBench` & Industry Evaluation Suite**
  - 100+ flaky, soft-failing, and malicious tool benchmark testbed.
- [ ] **Milestone 4: Speculative Execution Sandbox (`agent-honesty-sandbox`)**
  - Isolated ephemeral copy-on-write containers for pre-executing risky tools.
- [ ] **Milestone 5: Mechanistic Interpretability Probes (`agent-honesty-interp`)**
  - Neural hidden-state probing for internal deception detection.

---

## 📄 License

Apache 2.0 License. See [LICENSE](LICENSE) for details.
