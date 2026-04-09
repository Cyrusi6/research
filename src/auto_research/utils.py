"""Utility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str, *, max_length: int = 60) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:max_length] or "project"


def sanitize_filename(value: str, *, max_length: int = 80) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^\w\-.]+", "_", value, flags=re.ASCII)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:max_length] or "artifact"


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_text(path: Path, *, default: str | None = None) -> str:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_yaml(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [item.strip() for item in sentences if item.strip()]


def compact_markdown(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
