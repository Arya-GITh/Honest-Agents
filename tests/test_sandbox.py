import os
import sqlite3
import tempfile
import pytest
from pathlib import Path

from agent_honesty.sandbox import (
    DictStateSandbox,
    EphemeralFileSystemSandbox,
    InvariantViolationError,
    SandboxPolicy,
    SpeculativeSandbox,
    SQLiteSandbox,
    StateDeltaInspector,
    speculative_tool,
)


@pytest.fixture
def sqlite_db():
    """Fixture providing an initialized in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, balance REAL);")
    cursor.execute("INSERT INTO users (id, name, balance) VALUES (1, 'Alice', 500.0);")
    cursor.execute("INSERT INTO users (id, name, balance) VALUES (2, 'Bob', 250.0);")
    cursor.execute("INSERT INTO users (id, name, balance) VALUES (3, 'Charlie', 100.0);")
    conn.commit()
    cursor.close()
    yield conn
    conn.close()


def test_sqlite_sandbox_safe_commit(sqlite_db):
    """Verify that safe mutations within policy are successfully committed."""
    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(max_records_mutated=5)

    with SpeculativeSandbox(environment=env, policy=policy, tool_name="update_user"):
        cursor = sqlite_db.cursor()
        cursor.execute("UPDATE users SET balance = 600.0 WHERE id = 1;")
        cursor.close()

    cursor = sqlite_db.cursor()
    cursor.execute("SELECT balance FROM users WHERE id = 1;")
    balance = cursor.fetchone()[0]
    cursor.close()

    assert balance == 600.0


def test_sqlite_sandbox_unsafe_rollback(sqlite_db):
    """Verify that policy violations roll back changes completely, leaving DB untouched."""
    env = SQLiteSandbox(sqlite_db)
    # Policy allows max 1 mutation, but we will delete 3 rows
    policy = SandboxPolicy(max_records_mutated=1)

    with pytest.raises(InvariantViolationError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy, tool_name="bulk_delete"):
            cursor = sqlite_db.cursor()
            cursor.execute("DELETE FROM users;")
            cursor.close()

    assert "exceeds policy limit" in str(exc_info.value)

    # Verify all 3 rows are intact in the database
    cursor = sqlite_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM users;")
    count = cursor.fetchone()[0]
    cursor.close()

    assert count == 3


def test_sqlite_blocked_keyword(sqlite_db):
    """Verify that dangerous SQL keywords trigger an immediate block."""
    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(blocked_keywords=["DROP TABLE"])

    delta = env.compute_delta()
    verdict = StateDeltaInspector.evaluate(
        delta=delta,
        policy=policy,
        command_or_input="DROP TABLE users;",
    )

    assert not verdict.is_safe
    assert any("DROP TABLE" in v for v in verdict.violations)


def test_sqlite_table_whitelist(sqlite_db):
    """Verify that touching non-whitelisted tables is blocked."""
    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(allowed_tables=["orders"]) # users table not allowed

    with pytest.raises(InvariantViolationError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy, tool_name="insert_user"):
            cursor = sqlite_db.cursor()
            cursor.execute("INSERT INTO users (id, name, balance) VALUES (4, 'Dave', 50.0);")
            cursor.close()

    assert "not in allowed_tables" in str(exc_info.value)


def test_dict_state_sandbox_safe_and_unsafe():
    """Verify in-memory dictionary state isolation and financial limit policy."""
    state = {
        "acc_1": {"name": "Alice", "balance": 1000.0},
        "acc_2": {"name": "Bob", "balance": 500.0},
    }
    env = DictStateSandbox(state)
    policy = SandboxPolicy(max_financial_delta=200.0)

    # 1. Safe transfer of $100
    with SpeculativeSandbox(environment=env, policy=policy, tool_name="transfer_funds"):
        state["acc_1"]["balance"] -= 100.0
        state["acc_2"]["balance"] += 100.0

    assert state["acc_1"]["balance"] == 900.0
    assert state["acc_2"]["balance"] == 600.0

    # 2. Excessive transfer of $500 (exceeds $200 limit)
    with pytest.raises(InvariantViolationError):
        with SpeculativeSandbox(environment=env, policy=policy, tool_name="transfer_funds"):
            state["acc_1"]["balance"] -= 500.0
            state["acc_2"]["balance"] += 500.0

    # Rolled back to previous state!
    assert state["acc_1"]["balance"] == 900.0
    assert state["acc_2"]["balance"] == 600.0


def test_filesystem_sandbox():
    """Verify ephemeral filesystem sandbox tracks and rolls back file modifications."""
    temp_dir = tempfile.mkdtemp(prefix="truthify_test_fs_")
    try:
        ws = Path(temp_dir)
        (ws / "existing.txt").write_text("initial content")

        env = EphemeralFileSystemSandbox(str(ws))
        # Policy forbids file creation
        policy = SandboxPolicy(allow_file_writes=False)

        with pytest.raises(InvariantViolationError):
            with SpeculativeSandbox(environment=env, policy=policy, tool_name="write_file"):
                (ws / "unauthorized.txt").write_text("malicious payload")
                (ws / "existing.txt").write_text("corrupted content")

        # Verify unauthorized.txt was deleted and existing.txt was restored
        assert not (ws / "unauthorized.txt").exists()
        assert (ws / "existing.txt").read_text() == "initial content"
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_speculative_tool_decorator(sqlite_db):
    """Verify @speculative_tool decorator automatically inspects and commits/rolls back."""
    policy = SandboxPolicy(max_records_mutated=2)

    @speculative_tool(policy=policy)
    def update_user_balance(conn: sqlite3.Connection, user_id: int, new_balance: float):
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET balance = ? WHERE id = ?;", (new_balance, user_id))
        cursor.close()
        return {"status": "success", "user_id": user_id, "balance": new_balance}

    # Safe call
    res = update_user_balance(sqlite_db, 1, 999.0)
    assert res["status"] == "success"

    cursor = sqlite_db.cursor()
    cursor.execute("SELECT balance FROM users WHERE id = 1;")
    assert cursor.fetchone()[0] == 999.0
    cursor.close()

    @speculative_tool(policy=policy)
    def delete_all_users(conn: sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users;")
        cursor.close()
        return {"status": "deleted"}

    # Unsafe call
    with pytest.raises(InvariantViolationError):
        delete_all_users(sqlite_db)

    # DB remains unharmed
    cursor = sqlite_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM users;")
    assert cursor.fetchone()[0] == 3
    cursor.close()


@pytest.mark.asyncio
async def test_async_speculative_tool(sqlite_db):
    """Verify asynchronous @speculative_tool decorator."""
    policy = SandboxPolicy(max_records_mutated=2)

    @speculative_tool(policy=policy)
    async def async_update(conn: sqlite3.Connection, user_id: int, name: str):
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET name = ? WHERE id = ?;", (name, user_id))
        cursor.close()
        return {"status": "updated"}

    res = await async_update(sqlite_db, 2, "Robert")
    assert res["status"] == "updated"

    cursor = sqlite_db.cursor()
    cursor.execute("SELECT name FROM users WHERE id = 2;")
    assert cursor.fetchone()[0] == "Robert"
    cursor.close()


@pytest.mark.asyncio
async def test_async_semantic_judge_hook():
    """Verify custom async semantic judge integration."""
    async def custom_judge(ctx):
        # Reject if total mutations >= 2
        delta = ctx.get("state_delta", {})
        if delta.get("total_mutations", 0) >= 2:
            return {"is_safe": False, "reason": "SLM Judge flagged suspicious multi-record change."}
        return {"is_safe": True}

    state = {"a": 1, "b": 2, "c": 3}
    env = DictStateSandbox(state)
    policy = SandboxPolicy(semantic_judge_fn=custom_judge)

    with pytest.raises(InvariantViolationError) as exc_info:
        async with SpeculativeSandbox(environment=env, policy=policy):
            state["a"] = 10
            state["b"] = 20

    assert "SLM Judge flagged suspicious multi-record change" in str(exc_info.value)
    # Rolled back
    assert state["a"] == 1
    assert state["b"] == 2


def test_unhandled_exception_rollback(sqlite_db):
    """Verify that an unhandled Python exception inside the sandbox triggers immediate rollback."""
    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(max_records_mutated=10)

    with pytest.raises(RuntimeError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy):
            cursor = sqlite_db.cursor()
            cursor.execute("UPDATE users SET balance = 0.0 WHERE id = 1;")
            cursor.close()
            raise RuntimeError("Database connection interrupted during action!")

    assert "Database connection interrupted" in str(exc_info.value)

    # Verify Alice's balance was not left at 0.0
    cursor = sqlite_db.cursor()
    cursor.execute("SELECT balance FROM users WHERE id = 1;")
    balance = cursor.fetchone()[0]
    cursor.close()

    assert balance == 500.0


def test_blocked_table_blacklist(sqlite_db):
    """Verify that attempting to touch a blacklisted table is blocked."""
    cursor = sqlite_db.cursor()
    cursor.execute("CREATE TABLE auth_tokens (id INTEGER PRIMARY KEY, token TEXT);")
    cursor.execute("INSERT INTO auth_tokens (id, token) VALUES (1, 'secret_token');")
    sqlite_db.commit()
    cursor.close()

    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(blocked_tables=["auth_tokens"])

    with pytest.raises(InvariantViolationError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy):
            cursor = sqlite_db.cursor()
            cursor.execute("UPDATE auth_tokens SET token = 'compromised' WHERE id = 1;")
            cursor.close()

    assert "is in blocked_tables blacklist" in str(exc_info.value)

    # Verify token remained untouched
    cursor = sqlite_db.cursor()
    cursor.execute("SELECT token FROM auth_tokens WHERE id = 1;")
    token = cursor.fetchone()[0]
    cursor.close()

    assert token == "secret_token"


def test_read_only_policy(sqlite_db):
    """Verify that a read-only policy strictly forbids any state change."""
    env = SQLiteSandbox(sqlite_db)
    policy = SandboxPolicy(read_only=True)

    with pytest.raises(InvariantViolationError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy):
            cursor = sqlite_db.cursor()
            cursor.execute("INSERT INTO users (id, name, balance) VALUES (99, 'Test', 10.0);")
            cursor.close()

    assert "Policy is read-only" in str(exc_info.value)


def test_custom_validator_string_error():
    """Verify custom validator returning error message string."""
    def only_even_values(delta):
        if delta.financial_delta % 2 != 0:
            return "Financial transfer amount must be an even integer."
        return True

    state = {"balance": 100.0}
    env = DictStateSandbox(state)
    policy = SandboxPolicy(custom_validator=only_even_values)

    with pytest.raises(InvariantViolationError) as exc_info:
        with SpeculativeSandbox(environment=env, policy=policy):
            state["balance"] += 3.0 # Odd delta = 3.0

    assert "Financial transfer amount must be an even integer" in str(exc_info.value)
    assert state["balance"] == 100.0

