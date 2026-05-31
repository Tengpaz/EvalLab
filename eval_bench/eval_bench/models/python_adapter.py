"""Python-file model adapter loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from eval_bench.models.base import BaseModelAdapter


def load_symbol(spec: str):
    """Load ``/path/to/file.py:ClassName`` and return the symbol."""
    if ":" not in spec:
        raise ValueError("Python adapter must be written as /path/to/file.py:ClassName")
    path_text, symbol_name = spec.split(":", 1)
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Python adapter file not found: {path}")
    module_name = f"_eval_bench_adapter_{path.stem}_{abs(hash(path))}"
    import_spec = importlib.util.spec_from_file_location(module_name, path)
    if import_spec is None or import_spec.loader is None:
        raise ImportError(f"Could not load Python adapter module from {path}")
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"{path} does not define {symbol_name}")
    return getattr(module, symbol_name)


class PythonAdapter(BaseModelAdapter):
    """Adapter that delegates setup/predict to a user-provided Python class."""

    def setup(self) -> None:
        cls = load_symbol(str(self.config["adapter"]))
        self.impl = cls()
        if not hasattr(self.impl, "setup") or not hasattr(self.impl, "predict"):
            raise TypeError("Python model adapter must implement setup(self, model_config) and predict(self, batch).")
        self.impl.setup(self.config)

    def predict(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        return self.impl.predict(batch)
