"""Path and reproducibility helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit hash if the working directory is a repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def environment_info(argv: list[str] | None = None, cwd: str | Path | None = None) -> dict[str, Any]:
    """Capture lightweight run metadata for reproducibility."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path(cwd or Path.cwd()).resolve()),
        "argv": argv if argv is not None else sys.argv,
        "git_commit": find_git_commit(cwd or Path.cwd()),
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
