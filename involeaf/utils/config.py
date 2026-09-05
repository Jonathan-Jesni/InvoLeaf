"""YAML config loading with single-parent inheritance and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    """Parse a CLI override value using YAML rules, so 7, true and 1e-4 all work."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a config, resolving ``_base_`` relative to the config's own directory.

    ``overrides`` are ``key=value`` strings from the command line and win over the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    cfg: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        parent = load_config((path.parent / base).resolve())
        parent.update(cfg)
        cfg = parent

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be of the form key=value")
        key, value = item.split("=", 1)
        cfg[key.strip()] = _coerce(value.strip())

    cfg.setdefault("name", path.stem)
    return cfg
