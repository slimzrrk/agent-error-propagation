"""Resumable JSONL logs and run metadata for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("openai", "datasets", "numpy", "pandas", "scipy", "PyYAML")
ROOT = Path(__file__).resolve().parent.parent
# Everything that changes what a run produces: results made with a different version
# of any of these files must not be mixed in the same log. The faults enter through the
# hash of their messages (passed by the caller), not the raw faults.json.
PROTOCOL_FILES = (
    "src/prompts.py", "src/parser.py", "src/agents.py", "src/topologies.py", "src/faults.py",
    "src/defenses.py", "data/splits.json",
)


def protocol_fingerprint(*extra: str | None) -> str:
    h = hashlib.sha256()
    for name in PROTOCOL_FILES:
        path = ROOT / name
        if path.exists():
            h.update(name.encode())
            h.update(path.read_bytes())
    for part in extra:
        h.update(str(part).encode())
    return h.hexdigest()[:12]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def package_versions(names=PACKAGES) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "git_commit": git_commit(),
    }


class JsonlLog:
    """Append-only JSONL file shared by worker threads.

    Two record types: "meta" (one per run session: config, seeds, versions) and
    "run" (one per (config_id, question_id)).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.session_id = uuid.uuid4().hex[:12]

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def write_meta(self, meta: dict) -> None:
        self.write({"record_type": "meta", "session_id": self.session_id, "timestamp": now(), **meta})

    def write_run(self, record: dict) -> None:
        self.write({"record_type": "run", "session_id": self.session_id, "timestamp": now(), **record})

    def done_keys(self) -> set[tuple[str, str]]:
        return {(r["config_id"], r["question_id"]) for r in read_records(self.path, "run")}


def read_records(path: str | Path, record_type: str | None = "run") -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from an interrupted run
            if record_type is None or rec.get("record_type") == record_type:
                records.append(rec)
    return records
