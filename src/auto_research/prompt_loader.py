"""Prompt file loading."""

from __future__ import annotations

from pathlib import Path

from .utils import read_text, repo_root


def load_agent_prompt(agent_name: str) -> str:
    path = repo_root() / ".claude" / "agents" / f"{agent_name}.md"
    return read_text(path)
