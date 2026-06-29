"""Tiny YAML config loader. Keeps all hyperparameters/paths out of the code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(data).__name__}")
    return data


def project_root() -> Path:
    """Repo root (two levels above this file's package dir)."""
    return Path(__file__).resolve().parents[2]
