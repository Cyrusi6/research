"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import deep_merge, expand_env_vars, read_yaml, repo_root


DEFAULT_CONFIG_PATH = Path(".auto-research/config.yaml")


def load_root_config() -> dict[str, Any]:
    _load_env_files()
    path = repo_root() / DEFAULT_CONFIG_PATH
    config = read_yaml(path, default={}) or {}
    return expand_env_vars(config)


def load_project_config(project_root: Path) -> dict[str, Any]:
    root_config = load_root_config()
    project_config_path = project_root / "meta" / "project_config.yaml"
    if project_config_path.exists():
        project_config = read_yaml(project_config_path, default={}) or {}
        return expand_env_vars(deep_merge(root_config, project_config))
    return root_config


def apply_runtime_overrides(config: dict[str, Any], *, simulate: bool | None = None) -> dict[str, Any]:
    merged = dict(config)
    if simulate is not None:
        merged.setdefault("experiment", {})
        merged["experiment"] = dict(merged["experiment"])
        merged["experiment"]["simulate"] = simulate
    return merged


def _load_env_files() -> None:
    root = repo_root()
    for candidate in [root / ".env.local", root / ".env"]:
        _load_env_file(candidate)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
