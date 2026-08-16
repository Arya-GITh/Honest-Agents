# Truthify

[![PyPI Version](https://img.shields.io/pypi/v/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agent-honesty.svg)](https://pypi.org/project/agent-honesty/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://github.com/Arya-GITh/Truthify/actions/workflows/ci.yml/badge.svg)](https://github.com/Arya-GITh/Truthify/actions)

**The Execution Integrity and Trajectory Authenticity Platform for Autonomous AI Agents**

Truthify is an open-source platform and research initiative designed to eliminate the **Execution-Claim Gap** in autonomous AI agents. By integrating machine-level execution proofs, runtime verification, speculative sandboxing, and mechanistic interpretability probes, Truthify provides comprehensive governance over what AI agents do and what they report.

---

## The Problem: The Execution-Claim Gap

When autonomous AI agents interact with real-world tools, APIs, and databases, critical failures frequently occur at the execution boundary:

1. **False Success Invariance**: An API returns an HTTP 500 error, transaction deadlock, or timeout, but the model drafts a confirmation asserting successful execution.
2. **Empty Return Fabrication**: A database query returns 0 matching rows (`[]`), but the model hallucinates specific records, account identifiers, or entity states.
3. **Soft-Failure Blindness**: An API returns an HTTP 200 containing an application error payload (`{"status": "error", "code": 403}`), which the model misinterprets as an operational success.
4. **Parameter Taint & Mutation**: The model executes a mutating tool with parameters that deviate from user constraints without authorization.

Traditional text guardrails evaluate only lexical strings and cannot inspect physical machine state. Naive LLM-as-a-Judge architectures double inference costs, introduce multi-second latency, and remain susceptible to linguistic hallucination.

Truthify operates at the **machine execution boundary**, capturing unalterable cryptographic receipts and cross-examining model assertions prior to client delivery.

---

## Platform Architecture & Pipeline

Truthify provides an end-to-end execution integrity lifecycle:

```
+-------------------+      +-------------------+      +-------------------------+      +-------------------------+
|  Tool Interceptor | ---> | Cryptographic Log | ---> |  Two-Tier Verification  | ---> | Gated Stream & Actions  |
|  (@audit_tool /   |      | (HMAC-SHA256 &    |      |  (Tier 1 Deterministic  |      | (Token buffer, N=2 Cap, |
|   MCPClientProxy) |      |  FactMatrix)      |      |   + Tier 2 Semantic SLM)|      |  Deterministic Fallback)|
+-------------------+      +-------------------+      +-------------------------+      +-------------------------+
```

### 1. Interception & Normalization Layer
* **`@audit_tool`**: In-process Python decorator capturing input arguments, return objects, exceptions, and execution duration.
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
* **Deterministic Fallback Override**: Automatically delivers verified ground-truth summaries directly from the `FactMatrix` if the model fails to self-correct after 2 attempts.

---

## Platform Roadmap & Research Initiatives

Truthify is being developed across five core milestones spanning SDK engineering, framework integrations, benchmark evaluations, sandboxing, and mechanistic interpretability research:

### Milestone 1: Core Engine & SDK (`agent-honesty v0.1.0`)
* Universal tool interception (`@audit_tool`, `HonestyAuditor`, `MCPClientProxy`).
* Schema normalizer for soft-errors & HMAC-SHA256 cryptographic receipt engine.
* Two-Tier Verification Engine (Tier 1 Deterministic + Tier 2 Semantic SLM).
* Dual-Channel Streaming, in-scratchpad reprompt loop ($N=2$ hard cap), and deterministic fallback.

### Milestone 2: Multi-Framework Adapters
* Native middleware adapters for leading agent orchestration frameworks:
  * **LangGraph**: StateGraph node wrappers and pre/post-execution hooks.
  * **CrewAI**: Multi-agent task callback handlers and hierarchical tool proxies.
  * **AutoGen**: ConversableAgent filter intercepting tool calls between user proxies and assistants.
  * **LlamaIndex**: QueryEngine and FunctionTool integration layers.

### Milestone 3: Evaluation Suite & Benchmark Platform (`DeceptionBench`)
* Comprehensive open-source evaluation benchmark measuring deception rates, soft-failure blindness, and parameter mutations across frontier LLM architectures (GPT-4o, Claude 3.5, Gemini 2.5, Llama 3, Qwen 2.5).
* Public leaderboard comparing agent robustness under flaky APIs, transaction deadlocks, and empty result sets.

### Milestone 4: Speculative Execution Sandboxing
* Ephemeral, copy-on-write execution sandboxes for high-risk, state-mutating tools.
* Pre-execution gating: High-risk actions are executed speculatively in isolation; mutations are committed only after the trajectory is verified honest and compliant.

### Milestone 5: Mechanistic Interpretability Probes
* Neural representation research: Probing internal hidden states and sparse autoencoder (SAE) activations of models during reasoning to detect latent deception *before* token emission.

---

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
