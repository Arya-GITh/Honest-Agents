# Truthify

[![PyPI Version](https://img.shields.io/pypi/v/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://github.com/Arya-GITh/Truthify/actions/workflows/ci.yml/badge.svg)](https://github.com/Arya-GITh/Truthify/actions)

**Execution Integrity and Trajectory Authenticity Governance Platform for Autonomous AI Agents**

Truthify is an open-source security and governance framework designed to eliminate the **Execution-Claim Gap** in autonomous LLM agent systems. It enforces cryptographic verification over tool and API execution trajectories, preventing models from hallucinating false success, fabricating data on empty returns, or concealing runtime failures.

---

## The Problem: The Execution-Claim Gap

When autonomous AI agents interact with external tools, APIs, and databases, systemic failures frequently occur at the execution boundary:

1. **False Success Invariance**: An API returns an HTTP 500 error, transaction deadlock, or timeout, but the model drafts a confirmation asserting successful execution.
2. **Empty Return Fabrication**: A database query returns 0 matching rows (`[]`), but the model hallucinates specific records, account identifiers, or entity states.
3. **Soft-Failure Blindness**: An API returns an HTTP 200 containing an application error payload (`{"status": "error", "code": 403}`), which the model misinterprets as an operational success.
4. **Parameter Taint & Mutation**: The model executes a mutating tool with parameters that deviate from prompt constraints without user authorization.

Traditional text guardrails evaluate only lexical strings and are incapable of verifying physical machine state. Naive LLM-as-a-Judge architectures double inference costs, introduce multi-second latency, and remain susceptible to linguistic hallucination.

Truthify operates at the **machine execution boundary**, capturing unalterable cryptographic execution receipts and cross-examining model assertions prior to client release.

---

## Core Architecture

Truthify implements a four-stage execution integrity pipeline:

```
+-------------------+      +-------------------+      +-------------------------+      +-------------------------+
|  Tool Interceptor | ---> | Cryptographic Log | ---> |  Two-Tier Verification  | ---> | Gated Stream & Actions  |
|  (@audit_tool /   |      | (HMAC-SHA256 &    |      |  (Tier 1 Deterministic  |      | (Token buffer, N=2 Cap, |
|   MCPClientProxy) |      |  FactMatrix)      |      |   + Tier 2 Semantic SLM)|      |  Deterministic Fallback)|
+-------------------+      +-------------------+      +-------------------------+      +-------------------------+
```

### 1. Interception & Normalization Layer
* **`@audit_tool`**: In-process Python decorator capturing input arguments, return objects, exceptions, and execution latency.
* **`MCPClientProxy`**: Protocol proxy for Model Context Protocol (MCP) JSON-RPC clients.
* **`PayloadNormalizer`**: Analyzes response structures, extracting normalized execution parameters (`is_error`, `status_code`, `records_mutated`, `is_empty`).

### 2. Cryptographic Machine Receipts
* Generates an unforgeable **`HMACReceipt`** signed with HMAC-SHA256 over canonical JSON representations of the normalized `FactMatrix`. Receipts cannot be altered by in-context prompt injections.

### 3. Two-Tier Verification Waterfall
* **Tier 1 (Deterministic Engine, <0.5 ms, $0 Token Cost)**: Evaluates strict mathematical and boolean invariants directly against signed receipts. Resolves binary execution contradictions instantly without neural inference.
* **Tier 2 (Semantic SLM Auditor, ~50 ms)**: Evaluates arithmetic operations, paraphrased claims, and entity alias mappings using a fast Small Language Model (e.g., Qwen 2.5 0.5B, Phi 3.5, or Claude Haiku).

### 4. Gated Streaming & Self-Correction Engine
* **Dual-Channel Token Gating**: Buffers streaming output for state-mutating actions, releasing tokens only upon verification.
* **In-Scratchpad Self-Correction ($N=2$ Hard Cap)**: Injects targeted corrective feedback into the agent's reasoning scratchpad upon deception detection.
* **Deterministic Fallback Override**: Automatically synthesizes verified ground-truth summaries directly from the `FactMatrix` if the model fails to self-correct after 2 attempts.

## Platform Components & Roadmap

| Component | Type | Milestone | Description | Location |
| :--- | :--- | :---: | :--- | :--- |
| **`agent-honesty`** | **Python SDK** | `1` | Core governance library: Interceptors, HMAC Receipts, Two-Tier Verification, Self-Correction | [`packages/agent-honesty`](packages/agent-honesty) |
| **`agent-honesty-adapters`** | **Integrations** | `2` | Middleware adapters for LangGraph, CrewAI, AutoGen, and LlamaIndex | `packages/agent-honesty-adapters` |
| **`DeceptionBench`** | **Benchmark Platform** | `3` | Standardized evaluation suite with 100+ failure scenarios & public leaderboard | `benchmarks/` |
| **`Speculative Sandbox`** | **Safety Platform** | `4` | Ephemeral copy-on-write execution environments for pre-execution action gating | `sandbox/` |
| **`Mechanistic Probes`** | **Research Suite** | `5` | Neural activation probes for detecting latent deception in LLM hidden states | `research/` |

## Quickstart

### Installation

```bash
pip install agent-honesty
```

### Basic Function Auditing

```python
from agent_honesty import audit_tool, HonestyAuditor, VerificationRouter

@audit_tool(name="transfer_funds")
def transfer_funds(sender: str, recipient: str, amount: float):
    # Simulated backend failure:
    return {"status": "error", "error_code": "DB_DEADLOCK_500", "message": "Transaction aborted."}

with HonestyAuditor() as auditor:
    # 1. Execute audited tool (generates signed HMAC receipt)
    transfer_funds("acc_alice", "acc_bob", 150.0)
    
    # 2. Model drafts response
    agent_claim = "Your transfer of $150 has been completed successfully."
    
    # 3. Verify claim against machine truth in <0.5ms
    router = VerificationRouter()
    verdict = router.verify(
        user_prompt="Transfer $150 from Alice to Bob",
        agent_claim=agent_claim,
        receipts=auditor.receipts,
    )
    
    print(f"Is Honest: {verdict.is_honest}")            # False
    print(f"Deception Type: {verdict.deception_type}")  # DeceptionType.FALSE_SUCCESS
    print(f"Latency: {verdict.latency_ms:.2f} ms")      # 0.31 ms
```

---

## Runnable Examples

The [`examples/`](examples/) directory contains standalone, reproducible implementations:

* **[`01_basic_audit_tool.py`](examples/01_basic_audit_tool.py)**: Basic function wrapping with `@audit_tool`, soft-error detection, and HMAC receipt verification.
* **[`02_mcp_client_interceptor.py`](examples/02_mcp_client_interceptor.py)**: Model Context Protocol (MCP) JSON-RPC tool server proxy auditing via `MCPClientProxy`.
* **[`03_live_local_agent.py`](examples/03_live_local_agent.py)**: Autonomous local agent using Ollama (`qwen3:latest` primary + `qwen2.5:0.5b` SLM judge) with a live SQLite database on physical disk.
* **[`04_live_gemini_agent.py`](examples/04_live_gemini_agent.py)**: Autonomous cloud agent using Google Gemini (`gemini-flash-latest`) + local 8B SLM judge with a live SQLite database on physical disk.
* **[`05_langgraph_agent.py`](examples/05_langgraph_agent.py)**: LangGraph state graph agent with `TruthifyToolNode` and `TruthifyGraphEvaluator`.
* **[`06_crewai_agent.py`](examples/06_crewai_agent.py)**: CrewAI multi-agent crew with `TruthifyCrewCallback` and `wrap_crew_tool`.

---

### Speculative Execution Sandboxing (Pre-Execution Safety)

```python
import sqlite3
from agent_honesty.sandbox import speculative_tool, SandboxPolicy

conn = sqlite3.connect("production.db")

# Enforce pre-execution safety limits:
policy = SandboxPolicy(
    max_records_mutated=5,             # Max 5 rows can be modified
    allowed_tables=["orders"],         # Restrict access to orders table
    max_financial_delta=250.0,         # Cap transaction amounts
    blocked_keywords=["DROP", "TRUNCATE"]
)

@speculative_tool(policy=policy)
def update_order_status(conn: sqlite3.Connection, order_id: int, new_status: str):
    cursor = conn.cursor()
    cursor.execute("UPDATE orders SET status = ? WHERE id = ?;", (new_status, order_id))
    cursor.close()
    return {"status": "updated", "order_id": order_id}

# Safe calls commit automatically; destructive bulk operations roll back instantly!
```

---

## Roadmap

- [x] **Milestone 1: Core Engine & SDK (`agent-honesty v0.1.0`)**
  - Tool Decorators, Context Isolation, and MCP Proxy Interception
  - Normalized `FactMatrix` & HMAC-SHA256 Cryptographic Receipts
  - Two-Tier Verification Engine (Tier 1 Deterministic + Tier 2 Semantic SLM)
  - Dual-Channel Streaming, $N=2$ Scratchpad Reprompting, and Fallback Overrides
- [x] **Milestone 2: Multi-Framework Adapters (`agent_honesty.adapters`)**
  - Native integration hooks for LangGraph (`TruthifyToolNode`), CrewAI (`TruthifyCrewCallback`), AutoGen (`TruthifyAgentInterceptor`), and LlamaIndex (`wrap_llama_tools`)
- [x] **Milestone 3: Evaluation Suite (`DeceptionBench`)**
  - Standardized benchmark with 105 curated failure modes, soft-errors, and deceptive trajectories
  - Multi-model evaluation CLI (`benchmarks/runner.py`) with EDR, FER, and Protection Gain metrics
  - Automated Markdown, CSV, and LaTeX leaderboard generator
- [x] **Milestone 4: Speculative Sandboxing (`agent_honesty.sandbox`)**
  - Isolated ephemeral copy-on-write environments (SQLite, In-Memory Dict, Filesystem)
  - Pre-execution invariant inspection and two-phase commit / instant rollback engine
  - `@speculative_tool` decorator and `SpeculativeSandbox` context manager
- [ ] **Milestone 5: Mechanistic Probing (`Mechanistic Probes`)**
  - Neural activation probes for internal representation monitoring


---

## 🏆 DeceptionBench Evaluation Suite

Run standardized benchmarks across local models (Ollama) and cloud models (Gemini):

```bash
# Run quick benchmark evaluation on local or mock models
uv run python -m benchmarks.runner --models mock,qwen3:latest --limit 10

# Run full 105-scenario benchmark suite
uv run python -m benchmarks.runner --models qwen3:latest,gemini-flash-latest
```

Benchmark outputs are saved to `benchmarks/results/` as `leaderboard.md`, `leaderboard.csv`, `leaderboard.tex`, and `results.json`.
---

## Development

Truthify uses `uv` for workspace package management:

```bash
# Clone the repository
git clone https://github.com/Arya-GITh/Truthify.git
cd Truthify

# Install dependencies across all workspace packages
uv sync --all-packages

# Run the full test suite
uv run pytest
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for full details.
