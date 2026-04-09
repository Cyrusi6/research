"""Shared agent context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactManager
from ..llm import ModelClient


@dataclass
class AgentContext:
    project_root: Path
    config: dict[str, Any]
    artifacts: ArtifactManager
    llm: ModelClient
