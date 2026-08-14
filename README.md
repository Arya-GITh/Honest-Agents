# agent-honesty

> **Execution Integrity & Trajectory Authenticity Platform for Autonomous AI Agents**

`agent-honesty` is an open-source Python infrastructure and governance toolkit for verifying, auditing, and enforcing ground-truth honesty and execution integrity across autonomous AI agent tool interactions in real time.

---

## 🎯 The Core Mission

As AI models evolve into multi-step autonomous agents interacting directly with external APIs, databases, and systems, traditional text-based guardrails (e.g. toxicity or PII filtering) are insufficient. 

Agents can exhibit **Action-Claim Misalignment**:
* **False Success Claims:** Reporting that an action succeeded when an API returned a 500 error, rate limit, or timeout.
* **Soft Error Hallucinations:** Ignoring error payloads inside HTTP 200 OK responses and fabricating success metrics.
* **Data & Parameter Mutation:** Inventing response data or silently modifying arguments mid-sequence without authorization.

`agent-honesty` enforces a strict execution invariant: **verifying that what an agent claims strictly matches what it actually executed in the environment.**

---

## 🏗 Ecosystem Architecture

The repository is structured as a `uv` monorepo containing standalone PyPI packages:

* **`agent-honesty`**: Core zero-heavy-dependency runtime middleware, HMAC tool interceptors, Tier 1/2 verification engines, dual-channel streaming buffers, and in-scratchpad reprompt handlers.
* **`agent-honesty-sandbox`**: Ephemeral WASM/Docker speculative execution sandboxes, state entropy tracking, and transactional rollbacks.
* **`agent-honesty-interp`**: Mechanistic interpretability research layer with PyTorch residual stream hooks and SAE latent feature probes.

---

## 🚀 Quickstart

### Prerequisites
* Python 3.11+
* [`uv`](https://github.com/astral-sh/uv) package manager

### Installation

```bash
uv pip install agent-honesty
```

### Basic Usage

```python
from agent_honesty.interceptors import audit_tool

@audit_tool
async def fetch_user_data(user_id: str) -> dict:
    # Tool execution automatically generates cryptographic HMAC receipts
    # and extracts Fact Matrices for real-time verification.
    ...
```

---

## 📜 License

Apache-2.0 / MIT
