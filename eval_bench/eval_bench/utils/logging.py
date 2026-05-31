"""Small logging helpers used by CLI scripts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    """Emit a human-readable progress line."""
    print(message, flush=True)


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def now_iso() -> str:
    """Return a UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def fail(message: str, code: int = 1) -> None:
    """Print an error message and terminate a CLI process."""
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)
