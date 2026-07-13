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
        return apply_orchestration_profile(expand_env_vars(deep_merge(root_config, project_config)))
    return apply_orchestration_profile(root_config)


def apply_runtime_overrides(config: dict[str, Any], *, simulate: bool | None = None) -> dict[str, Any]:
    merged = dict(config)
    if simulate is not None:
        merged.setdefault("experiment", {})
        merged["experiment"] = dict(merged["experiment"])
        merged["experiment"]["simulate"] = simulate
    return merged


def orchestration_profile(config: dict[str, Any] | None) -> str:
    orchestration = (config or {}).get("orchestration") or {}
    if not isinstance(orchestration, dict):
        return "standard"
    profile = str(orchestration.get("profile") or "standard").strip().lower()
    return profile if profile in {"standard", "bootstrap"} else "standard"


def bootstrap_profile_enabled(config: dict[str, Any] | None) -> bool:
    return orchestration_profile(config) == "bootstrap"


def bootstrap_profile_options(config: dict[str, Any] | None) -> dict[str, Any]:
    orchestration = (config or {}).get("orchestration") or {}
    if not isinstance(orchestration, dict):
        return {}
    options = orchestration.get("bootstrap") or {}
    return options if isinstance(options, dict) else {}


def bootstrap_proxy_only_enabled(config: dict[str, Any] | None) -> bool:
    return bootstrap_profile_enabled(config) and bool(bootstrap_profile_options(config).get("proxy_only", True))


def apply_orchestration_profile(config: dict[str, Any]) -> dict[str, Any]:
    if not bootstrap_profile_enabled(config):
        return config
    return deep_merge(
        config,
        {
            "review": {"max_iterations": 1},
            "orchestration": {"stop_after_stage": "S3_experiment"},
            "ideation": {
                "contract_quality": {
                    "reject_placeholder_evidence": False,
                    "require_novelty_audit": False,
                }
            },
            "code_patch": {
                "validation": {
                    "gate_mode": "discovery",
                    "require_py_compile": True,
                    "require_targeted_tests": True,
                    "require_config_activation": False,
                    "runtime_smoke": {"enabled": False},
                    "mechanism_self_review": {"enabled": False},
                }
            },
            "c2c": {
                "small_loop": {
                    "max_candidates": 1,
                    "proxy_screen": {
                        "enabled": True,
                        "activation_smoke": {"enabled": False},
                    },
                }
            },
        },
    )


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
