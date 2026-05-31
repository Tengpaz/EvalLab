"""Configuration loading and resolution helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    data.setdefault("_config_path", str(path.resolve()))
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a YAML file with stable key ordering disabled for readability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config_refs(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    """Resolve optional dataset/model config references inside a run config.

    A run may inline ``dataset`` and ``model`` sections, or use:

    dataset_config: configs/datasets/foo.yaml
    model_config: configs/models/bar.yaml

    Inline values override referenced files.
    """
    resolved = deepcopy(config)
    cfg_path = config.get("_config_path")
    if base_dir is None:
        base = Path(cfg_path).resolve().parent if cfg_path else Path.cwd()
    else:
        base = Path(base_dir)

    for section, ref_key in (("dataset", "dataset_config"), ("model", "model_config")):
        if ref_key not in config:
            continue
        ref_path = Path(config[ref_key]).expanduser()
        if not ref_path.is_absolute():
            ref_path = (base / ref_path).resolve()
        ref_cfg = load_yaml(ref_path)
        inline_cfg = config.get(section, {})
        if inline_cfg and not isinstance(inline_cfg, dict):
            raise ValueError(f"Run config section '{section}' must be a mapping.")
        resolved[section] = deep_merge(ref_cfg, inline_cfg or {})
    return resolved


def path_from_config(value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a filesystem path from config without requiring it to exist."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) / path if base_dir else path).resolve()
