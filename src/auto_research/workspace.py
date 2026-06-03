"""Workspace creation and path helpers."""

from __future__ import annotations

import platform
import uuid
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactManager
from .constants import REQUIRED_STAGE_DIRS, STAGE_LABELS, STAGE_ORDER
from .orchestration_state import OrchestrationStateManager
from .registry import default_registry
from .stage_contracts import StageContractManager
from .utils import ensure_dir, now_utc, read_yaml, slugify, write_json, write_text, write_yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def references_dir(self) -> Path:
        return self.root / "references"

    @property
    def papers_dir(self) -> Path:
        return self.references_dir / "papers"

    @property
    def bib_dir(self) -> Path:
        return self.references_dir / "bib"

    @property
    def meta_dir(self) -> Path:
        return self.root / "meta"

    def stage_dir(self, stage_key: str) -> Path:
        return self.root / STAGE_LABELS[stage_key]


def project_paths(workspace_root: Path, project_id: str) -> ProjectPaths:
    return ProjectPaths(root=workspace_root / project_id)


def build_project_id(topic: str) -> str:
    return f"{now_utc()[:10]}_{slugify(topic, max_length=32)}_{uuid.uuid4().hex[:8]}"


def init_workspace(config: dict, topic: str, *, project_id: str | None = None, simulate: bool | None = None) -> ProjectPaths:
    workspace_root = Path(config["project"]["workspace_root"])
    ensure_dir(workspace_root)
    project_id = project_id or build_project_id(topic)
    paths = project_paths(workspace_root, project_id)

    ensure_dir(paths.root)
    for name in REQUIRED_STAGE_DIRS:
        ensure_dir(paths.root / name)
    ensure_dir(paths.papers_dir)
    ensure_dir(paths.bib_dir)
    artifacts = ArtifactManager(paths.root)

    for stage_key in STAGE_ORDER:
        stage_dir = paths.stage_dir(stage_key)
        ensure_dir(stage_dir / "logs")
        ensure_dir(stage_dir / "_tmp")
        ensure_dir(stage_dir / "failed")
        artifacts.initialize_stage_manifest(stage_key, force=True)

    write_json(paths.papers_dir / "manifest.json", {"updated_at": now_utc(), "papers": []})
    write_text(paths.meta_dir / "session_log.jsonl", "")
    write_yaml(paths.meta_dir / "codex_sessions.yaml", {"sessions": {}})
    write_yaml(
        paths.meta_dir / "project_config.yaml",
        {
            "project": {"research_topic": topic},
            "experiment": {"simulate": bool(simulate)} if simulate is not None else {},
        },
    )
    registry = default_registry(project_id=project_id, topic=topic, config=config)
    write_yaml(paths.meta_dir / "registry.yaml", registry)
    OrchestrationStateManager(paths.root).initialize(registry, force=True)
    StageContractManager(paths.root).initialize_all(force=True, config=config, iteration=registry.get("iteration"))
    write_yaml(
        paths.meta_dir / "environment.yaml",
        {
            "created_at": now_utc(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    )
    return paths


def load_registry(project_root: Path) -> dict:
    return read_yaml(project_root / "meta" / "registry.yaml")
