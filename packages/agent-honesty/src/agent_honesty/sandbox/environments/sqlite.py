import hashlib
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from agent_honesty.sandbox.environments.base import BaseSandboxEnvironment
from agent_honesty.sandbox.models import StateDelta


class SQLiteSandbox(BaseSandboxEnvironment):
    """
    Copy-on-Write sandbox for SQLite databases using transactional savepoints.
    Enables instant microsecond speculative execution, mutation auditing, and atomic rollback.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection
        self._savepoint_id: Optional[str] = None
        self._pre_counts: Dict[str, int] = {}
        self._pre_total_changes: int = 0
        self._pre_hash: str = ""

    def _get_tables(self) -> List[str]:
        """Fetch all user table names in the database."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def _snapshot_table_counts(self) -> Dict[str, int]:
        """Snapshot row counts for each user table."""
        counts = {}
        cursor = self.conn.cursor()
        try:
            tables = self._get_tables()
            for tbl in tables:
                cursor.execute(f"SELECT COUNT(*) FROM \"{tbl}\";")
                counts[tbl] = cursor.fetchone()[0]
            return counts
        finally:
            cursor.close()

    def get_state_hash(self) -> str:
        """Compute SHA-256 fingerprint over table counts and schemas."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;")
            schema_data = cursor.fetchall()
            counts = self._snapshot_table_counts()
            combined = {"schema": schema_data, "counts": counts}
            return self._hash_dict(combined)
        finally:
            cursor.close()

    def enter_speculation(self) -> None:
        """Capture baseline and establish a transactional savepoint."""
        self._pre_hash = self.get_state_hash()
        self._pre_counts = self._snapshot_table_counts()
        self._pre_total_changes = self.conn.total_changes
        self._savepoint_id = f"truthify_sp_{uuid.uuid4().hex[:12]}"
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SAVEPOINT {self._savepoint_id};")
        finally:
            cursor.close()

    def compute_delta(self) -> StateDelta:
        """Compute mutations by comparing current table states to pre-speculation snapshot."""
        post_counts = self._snapshot_table_counts()
        post_total_changes = self.conn.total_changes
        total_db_changes = max(0, post_total_changes - self._pre_total_changes)

        rows_inserted = 0
        rows_deleted = 0
        tables_touched = []

        all_tables = set(self._pre_counts.keys()) | set(post_counts.keys())
        for tbl in all_tables:
            pre_c = self._pre_counts.get(tbl, 0)
            post_c = post_counts.get(tbl, 0)
            diff = post_c - pre_c

            if pre_c != post_c:
                tables_touched.append(tbl)

            if diff > 0:
                rows_inserted += diff
            elif diff < 0:
                rows_deleted += abs(diff)

        # In SQLite, total_changes counts all INSERT, UPDATE, and DELETE operations.
        # If total_changes > (inserted + deleted), the remainder represents UPDATE operations.
        known_mutations = rows_inserted + rows_deleted
        rows_updated = max(0, total_db_changes - known_mutations)
        
        if rows_updated > 0 and not tables_touched:
            # Table counts remained identical, but rows were updated
            tables_touched = list(self._pre_counts.keys())

        total_mutations = rows_inserted + rows_updated + rows_deleted

        return StateDelta(
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
            rows_deleted=rows_deleted,
            total_mutations=total_mutations,
            tables_touched=tables_touched,
            metadata={"sqlite_total_changes": total_db_changes},
        )

    def commit(self) -> None:
        """Release savepoint to commit changes to the active database."""
        if not self._savepoint_id:
            return
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"RELEASE SAVEPOINT {self._savepoint_id};")
        finally:
            cursor.close()
            self._savepoint_id = None

    def rollback(self) -> None:
        """Rollback to savepoint and release it, restoring exact initial state."""
        if not self._savepoint_id:
            return
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint_id};")
            cursor.execute(f"RELEASE SAVEPOINT {self._savepoint_id};")
        finally:
            cursor.close()
            self._savepoint_id = None
