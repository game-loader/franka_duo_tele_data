"""Small YAML/JSON loader shared by the MCAP recorder and evaluator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - base dependency
            raise RuntimeError("YAML configuration requires PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config {path} must contain a mapping")
    return value
