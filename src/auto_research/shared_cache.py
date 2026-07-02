"""Shared cache path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, repo_root


def shared_cache_root(project_root: Path, config: dict[str, Any], *, create: bool = True) -> Path:
    project_cfg = config.get("project") if isinstance(config, dict) else {}
    configured = (project_cfg or {}).get("shared_cache_root") if isinstance(project_cfg, dict) else None
    if configured:
        root = Path(str(configured)).expanduser()
        if not root.is_absolute():
            root = repo_root() / root
    else:
        root = project_root.parent / "_shared_cache" / "auto_research"
    return ensure_dir(root) if create else root
