import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent_honesty.sandbox.environments.base import BaseSandboxEnvironment
from agent_honesty.sandbox.models import StateDelta


class EphemeralFileSystemSandbox(BaseSandboxEnvironment):
    """
    Copy-on-Write overlay sandbox for file system modifications.
    Tracks file additions, modifications, and deletions in a workspace directory.
    """

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self._backup_dir: Optional[Path] = None
        self._pre_files: Dict[str, Tuple[int, float]] = {} # rel_path -> (size, mtime)
        self._pre_hash: str = ""

    def _scan_directory(self) -> Dict[str, Tuple[int, float]]:
        """Scan workspace for relative file paths, sizes, and mtimes."""
        if not self.workspace_dir.exists():
            return {}
        scanned = {}
        for root, _, files in os.walk(self.workspace_dir):
            for f in files:
                full_p = Path(root) / f
                try:
                    rel_p = str(full_p.relative_to(self.workspace_dir))
                    stat = full_p.stat()
                    scanned[rel_p] = (stat.st_size, stat.st_mtime)
                except Exception:
                    pass
        return scanned

    def get_state_hash(self) -> str:
        """Hash file paths and sizes."""
        scanned = self._scan_directory()
        return self._hash_dict(scanned)

    def enter_speculation(self) -> None:
        """Create a temporary backup directory of the current workspace."""
        self._pre_files = self._scan_directory()
        self._pre_hash = self.get_state_hash()
        
        # Create temp backup
        self._backup_dir = Path(tempfile.mkdtemp(prefix="truthify_fs_bak_"))
        if self.workspace_dir.exists():
            for rel_p in self._pre_files:
                src = self.workspace_dir / rel_p
                dst = self._backup_dir / rel_p
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def compute_delta(self) -> StateDelta:
        """Inspect file additions, modifications, deletions, and total bytes written."""
        current_files = self._scan_directory()
        created = [f for f in current_files if f not in self._pre_files]
        deleted = [f for f in self._pre_files if f not in current_files]
        modified = [
            f for f in current_files
            if f in self._pre_files and current_files[f] != self._pre_files[f]
        ]

        total_bytes_written = sum(current_files[f][0] for f in created + modified)
        all_touched = list(set(created + modified + deleted))

        return StateDelta(
            rows_inserted=len(created),
            rows_updated=len(modified),
            rows_deleted=len(deleted),
            total_mutations=len(all_touched),
            files_modified=all_touched,
            bytes_written=total_bytes_written,
            metadata={
                "created_files": created,
                "modified_files": modified,
                "deleted_files": deleted,
            },
        )

    def commit(self) -> None:
        """Acknowledge changes and remove backup directory."""
        if self._backup_dir and self._backup_dir.exists():
            shutil.rmtree(self._backup_dir, ignore_errors=True)
            self._backup_dir = None

    def rollback(self) -> None:
        """Revert directory to the exact backed up pre-state."""
        if not self._backup_dir or not self._backup_dir.exists():
            return

        current_files = self._scan_directory()
        # 1. Delete new files created in workspace
        for f in current_files:
            if f not in self._pre_files:
                (self.workspace_dir / f).unlink(missing_ok=True)

        # 2. Restore modified/deleted files from backup
        for rel_p in self._pre_files:
            src = self._backup_dir / rel_p
            dst = self.workspace_dir / rel_p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        shutil.rmtree(self._backup_dir, ignore_errors=True)
        self._backup_dir = None
