"""Factories for dataset and model adapters."""

from __future__ import annotations

from typing import Any


def build_dataset(config: dict[str, Any]):
    """Instantiate a dataset adapter from a config mapping."""
    dataset_type = config.get("type")
    if dataset_type == "seva_benchmark":
        from eval_bench.datasets.seva_benchmark import SevaBenchmarkDataset

        return SevaBenchmarkDataset(config)
    if dataset_type == "generic_transforms":
        from eval_bench.datasets.generic_transforms import GenericTransformsDataset

        return GenericTransformsDataset(config)
    raise ValueError(
        f"Unknown dataset type {dataset_type!r}. Expected one of: seva_benchmark, generic_transforms."
    )


def build_model(config: dict[str, Any]):
    """Instantiate a model adapter from a config mapping."""
    model_type = config.get("type")
    if model_type == "python":
        from eval_bench.models.python_adapter import PythonAdapter

        return PythonAdapter(config)
    if model_type == "command":
        from eval_bench.models.command_adapter import CommandAdapter

        return CommandAdapter(config)
    raise ValueError(f"Unknown model type {model_type!r}. Expected one of: python, command.")
