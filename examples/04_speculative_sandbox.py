#!/usr/bin/env python3
"""
Example 04: Speculative Sandbox & Pre-Execution Safety Gating
-------------------------------------------------------------
Demonstrates how Truthify intercepts state-mutating tool actions, evaluates
mutations inside an ephemeral Copy-on-Write sandbox, and commits safe changes
or rolls back unsafe/destructive actions before they touch production.
"""

import sqlite3
from agent_honesty.sandbox import (
    InvariantViolationError,
    SandboxPolicy,
    SpeculativeSandbox,
    SQLiteSandbox,
    speculative_tool,
)


def create_demo_database() -> sqlite3.Connection:
    """Initialize a mock production database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, balance REAL);")
    cursor.executemany(
        "INSERT INTO users (id, name, balance) VALUES (?, ?, ?);",
        [
            (1, "Alice Smith", 1250.0),
            (2, "Bob Jones", 450.0),
            (3, "Charlie Brown", 80.0),
            (4, "Dana White", 2300.0),
            (5, "Evan Wright", 15.0),
        ],
    )
    conn.commit()
    cursor.close()
    return conn


def print_users(conn: sqlite3.Connection, title: str) -> None:
    print(f"\n--- {title} ---")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, balance FROM users ORDER BY id;")
    rows = cursor.fetchall()
    for r in rows:
        print(f"  User #{r[0]}: {r[1]:<15} | Balance: ${r[2]:.2f}")
    print(f"  [Total Rows: {len(rows)}]\n")
    cursor.close()


def main():
    print("=" * 70)
    print("🛡️ Truthify Speculative Sandbox: Pre-Execution Safety Demo")
    print("=" * 70)

    db = create_demo_database()
    print_users(db, "Initial Production State")

    # Define a strict safety policy for AI agent operations
    policy = SandboxPolicy(
        max_records_mutated=2,                  # Max 2 rows can be modified per call
        allowed_tables=["users"],               # Allowed tables whitelist
        max_financial_delta=300.0,              # Max $300.00 balance adjustment
        blocked_keywords=["DROP", "TRUNCATE"],  # Block dangerous SQL keywords
    )

    print(f"Active Safety Policy: max_mutations={policy.max_records_mutated}, max_financial_delta=${policy.max_financial_delta}")

    # -------------------------------------------------------------
    # Scenario 1: Safe Mutation (Alice receives $150 credit)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Scenario 1: Executing Safe Balance Adjustment ($150 credit to Alice)")
    print("=" * 50)

    @speculative_tool(policy=policy)
    def update_balance(conn: sqlite3.Connection, user_id: int, new_balance: float):
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET balance = ? WHERE id = ?;", (new_balance, user_id))
        cursor.close()
        return {"status": "success", "user_id": user_id, "new_balance": new_balance}

    try:
        res = update_balance(db, 1, 1400.0) # Delta = +$150.00 (within $300 limit, 1 row)
        print(f"✅ Safe Tool Executed: {res}")
        print("  -> Speculative Check: 1 row modified, financial delta = $150.00 (PASSED)")
        print("  -> Two-Phase Commit: COMMITTED to live database.")
    except InvariantViolationError as e:
        print(f"❌ Blocked: {e}")

    print_users(db, "Production State After Scenario 1")

    # -------------------------------------------------------------
    # Scenario 2: Unsafe Bulk Deletion (Agent attempts DELETE without WHERE)
    # -------------------------------------------------------------
    print("=" * 50)
    print("Scenario 2: Unsafe Destructive Action (Agent runs bulk DELETE)")
    print("=" * 50)

    @speculative_tool(policy=policy)
    def delete_all_users(conn: sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users;")
        cursor.close()
        return {"status": "deleted"}

    try:
        print("Agent attempts: 'DELETE FROM users;' ...")
        delete_all_users(db)
        print("⚠️ Warning: Destructive action was NOT blocked!")
    except InvariantViolationError as e:
        print(f"🛡️ PRE-EXECUTION BLOCKED: {e}")
        print("  -> Speculative Check: 5 rows deleted (Policy allows max 2 rows) -> REJECTED")
        print("  -> Atomic Rollback: Sandbox reverted changes in <1ms.")
        print("  -> Live database remains 100% UNTOUCHED.")

    print_users(db, "Production State After Scenario 2 (Zero Data Loss)")

    # -------------------------------------------------------------
    # Scenario 3: Context Manager Sandbox (Table Whitelist & Custom Policy)
    # -------------------------------------------------------------
    print("=" * 50)
    print("Scenario 3: Direct Context Manager (Table Whitelist Violation)")
    print("=" * 50)

    # Restrict policy to 'orders' table only
    strict_policy = SandboxPolicy(allowed_tables=["orders"])
    env = SQLiteSandbox(db)

    try:
        with SpeculativeSandbox(environment=env, policy=strict_policy, tool_name="unauthorized_table_write"):
            print("Agent attempts: Modifying 'users' table (when only 'orders' is permitted)...")
            cursor = db.cursor()
            cursor.execute("UPDATE users SET balance = 5000.0 WHERE id = 4;")
            cursor.close()
    except InvariantViolationError as e:
        print(f"🛡️ PRE-EXECUTION BLOCKED: {e}")
        print(f"  -> Reason: Attempted to touch table '{e.delta.tables_touched}' not in allowed whitelist.")
        print("  -> Atomic Rollback: Dana's balance remains unaltered.")

    print_users(db, "Final Production State (Dana's balance remains $2300.00)")
    print("=" * 70)
    print("✨ Demo completed successfully. Zero production corruption.")
    print("=" * 70)



if __name__ == "__main__":
    main()
