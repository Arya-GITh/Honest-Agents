# agent-honesty

> **Execution Integrity & Trajectory Authenticity Middleware for Autonomous AI Agents**

`agent-honesty` is a lightweight, zero-heavy-dependency Python SDK that captures ground-truth tool execution receipts and verifies raw API outputs against agent claims in real time.

## Installation

```bash
pip install agent-honesty
```

## Basic Usage

```python
from agent_honesty import audit_tool, HonestyAuditor

@audit_tool
def execute_query(query: str):
    return {"status": "ok", "rows": 42}

with HonestyAuditor() as auditor:
    execute_query("SELECT * FROM users")
    print(auditor.records)
```
