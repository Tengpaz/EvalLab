"""Split parsing and generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval_bench.io.transforms import read_json, write_json


def normalize_id_list(values: Any, key: str) -> list[int]:
    """Normalize a split id list to unique integer ids preserving order."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"Split key '{key}' must be a list of frame ids.")
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        frame_id = int(value)
        if frame_id not in seen:
            result.append(frame_id)
            seen.add(frame_id)
    return result


def read_split_file(path: str | Path) -> dict[str, Any]:
    """Read a split json in common benchmark formats."""
    data = read_json(path)
    if isinstance(data, dict) and "input_ids" in data:
        return {
            "input_ids": normalize_id_list(data.get("input_ids"), "input_ids"),
            "target_ids": normalize_id_list(
                data.get("target_ids", data.get("test_ids")), "target_ids"
            ),
            "metadata": {k: v for k, v in data.items() if k not in {"input_ids", "target_ids"}},
        }
    if isinstance(data, dict) and "train_ids" in data:
        return {
            "input_ids": normalize_id_list(data.get("train_ids"), "train_ids"),
            "target_ids": normalize_id_list(data.get("test_ids"), "test_ids"),
            "metadata": {k: v for k, v in data.items() if k not in {"train_ids", "test_ids"}},
        }
    if isinstance(data, dict) and "splits" in data:
        splits = data["splits"]
        if not isinstance(splits, list) or not splits:
            raise ValueError(f"Split file {path} has empty 'splits'.")
        return {
            "input_ids": normalize_id_list(splits[0].get("input_ids"), "input_ids"),
            "target_ids": normalize_id_list(splits[0].get("target_ids"), "target_ids"),
            "metadata": data,
        }
    raise ValueError(
        f"Unsupported split format in {path}. Expected input_ids/target_ids, "
        "train_ids/test_ids, or a non-empty splits list."
    )


def generate_split(frame_ids: list[int], spec: dict[str, Any]) -> dict[str, Any]:
    """Generate a simple split from frame ids.

    Supported strategies:
    - first_k_as_input: first k frames are inputs, remaining frames are targets.
    - fixed_input_ids: configured input_ids are inputs, all other frames are targets.
    - every_n: every nth frame is input, all other frames are targets.
    """
    if not frame_ids:
        raise ValueError("Cannot generate a split without frame ids.")
    strategy = spec.get("strategy", "first_k_as_input")
    if strategy == "first_k_as_input":
        k = int(spec.get("k", spec.get("num_inputs", 1)))
        input_ids = frame_ids[:k]
        target_ids = [x for x in frame_ids if x not in set(input_ids)]
    elif strategy == "fixed_input_ids":
        input_ids = normalize_id_list(spec.get("input_ids"), "input_ids")
        target_ids = normalize_id_list(spec.get("target_ids"), "target_ids")
        if not target_ids:
            target_ids = [x for x in frame_ids if x not in set(input_ids)]
    elif strategy == "every_n":
        n = max(1, int(spec.get("n", 3)))
        input_ids = [frame_id for idx, frame_id in enumerate(frame_ids) if idx % n == 0]
        target_ids = [frame_id for frame_id in frame_ids if frame_id not in set(input_ids)]
    else:
        raise ValueError(
            f"Unknown split strategy {strategy!r}. Use first_k_as_input, fixed_input_ids, or every_n."
        )
    return {"input_ids": input_ids, "target_ids": target_ids, "metadata": {"strategy": strategy}}


def write_split_file(path: str | Path, input_ids: list[int], target_ids: list[int]) -> None:
    """Write a canonical split json."""
    write_json(path, {"input_ids": input_ids, "target_ids": target_ids})
