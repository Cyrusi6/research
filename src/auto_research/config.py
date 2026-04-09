"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import deep_merge, expand_env_vars, read_yaml, repo_root


DEFAULT_CONFIG_PATH = Path(".auto-research/config.yaml")


def load_root_config() -> dict[str, Any]:
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
