"""Artifact tracking and atomic writes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .constants import STAGE_LABELS
from .utils import ensure_dir, now_utc, read_json, sha256_file, write_json


STAGE_MANIFEST_SCHEMA_VERSION = "stage_manifest_v1"
ARTIFACT_RECORD_SCHEMA_VERSION = "artifact_record_v1"

_STAGE_CREATED_BY = {
    "S0_intake": "intake-agent",
    "S1_literature": "literature-agent",
    "S2_plan": "plan-agent",
    "S3_experiment": "experiment-agent",
    "S4_writing": "writing-agent",
    "S5_review": "review-agent",
}

_STAGE_VALIDATOR_PREFIX = {
    "S0_intake": "s0",
    "S1_literature": "s1",
    "S2_plan": "s2",
    "S3_experiment": "s3",
    "S4_writing": "s4",
    "S5_review": "s5",
}


class ArtifactManager:
    """Owns stage output registration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def stage_dir(self, stage_key: str) -> Path:
        return self.project_root / STAGE_LABELS[stage_key]

    def manifest_path(self, stage_key: str) -> Path:
        return self.stage_dir(stage_key) / "stage_manifest.json"

    def load_manifest(self, stage_key: str) -> dict[str, Any]:
        manifest = read_json(self.manifest_path(stage_key), default=self._empty_manifest(stage_key))
        return self._normalize_manifest(stage_key, manifest)

    def initialize_stage_manifest(self, stage_key: str, *, force: bool = False) -> dict[str, Any]:
        """Create or upgrade a stage manifest to the current schema."""

        if force or not self.manifest_path(stage_key).exists():
            manifest = self._empty_manifest(stage_key)
        else:
            manifest = self.load_manifest(stage_key)
        self._save_manifest(stage_key, manifest)
        return manifest

    def _save_manifest(self, stage_key: str, manifest: dict[str, Any]) -> None:
        manifest = self._normalize_manifest(stage_key, manifest)
        manifest["updated_at"] = now_utc()
        manifest["artifact_count"] = len(manifest.get("artifacts", []))
        write_json(self.manifest_path(stage_key), manifest)

    def reserve_path(self, stage_key: str, relative_path: str) -> Path:
        final_path = self.stage_dir(stage_key) / relative_path
        ensure_dir(final_path.parent)
        return final_path

    def _tmp_path(self, stage_key: str, relative_path: str) -> Path:
        tmp_dir = self.stage_dir(stage_key) / "_tmp"
        ensure_dir(tmp_dir)
        return tmp_dir / f"{uuid.uuid4().hex}_{Path(relative_path).name}"

    def _commit(self, tmp_path: Path, final_path: Path) -> None:
        ensure_dir(final_path.parent)
        tmp_path.replace(final_path)

    def register_artifact(
        self,
        stage_key: str,
        final_path: Path,
        *,
        artifact_type: str,
        summary: str = "",
        source_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.load_manifest(stage_key)
        rel_path = final_path.relative_to(self.project_root).as_posix()
        artifact_type = artifact_type or "artifact"
        entry = {
            "schema_version": ARTIFACT_RECORD_SCHEMA_VERSION,
            "path": rel_path,
            "type": artifact_type,
            "artifact_type": artifact_type,
            "summary": summary,
            "source_paths": source_paths or [],
            "metadata": metadata or {},
            "sha256": sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "created_by": created_by or _default_created_by(stage_key, artifact_type),
            "created_at": now_utc(),
            "status": "committed",
            "validator": validator or _default_validator(stage_key, artifact_type),
        }
        artifacts = [item for item in manifest.get("artifacts", []) if item.get("path") != rel_path]
        artifacts.append(entry)
        manifest["artifacts"] = sorted(artifacts, key=lambda item: item["path"])
        self._save_manifest(stage_key, manifest)
        return entry

    def write_text(
        self,
        stage_key: str,
        relative_path: str,
        content: str,
        *,
        artifact_type: str,
        summary: str = "",
        source_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        tmp_path = self._tmp_path(stage_key, relative_path)
        tmp_path.write_text(content, encoding="utf-8")
        final_path = self.reserve_path(stage_key, relative_path)
        self._commit(tmp_path, final_path)
        return self.register_artifact(
            stage_key,
            final_path,
            artifact_type=artifact_type,
            summary=summary,
            source_paths=source_paths,
            metadata=metadata,
            created_by=created_by,
            validator=validator,
        )

    def write_json(
        self,
        stage_key: str,
        relative_path: str,
        payload: Any,
        *,
        artifact_type: str,
        summary: str = "",
        source_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        import json

        return self.write_text(
            stage_key,
            relative_path,
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            artifact_type=artifact_type,
            summary=summary,
            source_paths=source_paths,
            metadata=metadata,
            created_by=created_by,
            validator=validator,
        )

    def write_yaml(
        self,
        stage_key: str,
        relative_path: str,
        payload: Any,
        *,
        artifact_type: str,
        summary: str = "",
        source_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        import yaml

        return self.write_text(
            stage_key,
            relative_path,
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            artifact_type=artifact_type,
            summary=summary,
            source_paths=source_paths,
            metadata=metadata,
            created_by=created_by,
            validator=validator,
        )

    def write_bytes(
        self,
        stage_key: str,
        relative_path: str,
        data: bytes,
        *,
        artifact_type: str,
        summary: str = "",
        source_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        tmp_path = self._tmp_path(stage_key, relative_path)
        tmp_path.write_bytes(data)
        final_path = self.reserve_path(stage_key, relative_path)
        self._commit(tmp_path, final_path)
        return self.register_artifact(
            stage_key,
            final_path,
            artifact_type=artifact_type,
            summary=summary,
            source_paths=source_paths,
            metadata=metadata,
            created_by=created_by,
            validator=validator,
        )

    def copy_into_stage(
        self,
        stage_key: str,
        source_path: Path,
        relative_path: str,
        *,
        artifact_type: str,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        validator: str | None = None,
    ) -> dict[str, Any]:
        tmp_path = self._tmp_path(stage_key, relative_path)
        ensure_dir(tmp_path.parent)
        shutil.copy2(source_path, tmp_path)
        final_path = self.reserve_path(stage_key, relative_path)
        self._commit(tmp_path, final_path)
        try:
            source_ref = source_path.relative_to(self.project_root).as_posix()
        except ValueError:
            source_ref = str(source_path)
        return self.register_artifact(
            stage_key,
            final_path,
            artifact_type=artifact_type,
            summary=summary,
            source_paths=[source_ref],
            metadata=metadata,
            created_by=created_by,
            validator=validator,
        )

    def list_stage_artifacts(self, stage_key: str) -> list[str]:
        manifest = self.load_manifest(stage_key)
        return [item["path"] for item in manifest.get("artifacts", [])]

    def reference_manifest_path(self) -> Path:
        return self.project_root / "references" / "papers" / "manifest.json"

    def load_reference_manifest(self) -> dict[str, Any]:
        return read_json(self.reference_manifest_path(), default={"updated_at": now_utc(), "papers": []})

    def register_reference_paper(self, record: dict[str, Any]) -> None:
        manifest = self.load_reference_manifest()
        papers = [item for item in manifest.get("papers", []) if item.get("paper_id") != record.get("paper_id")]
        papers.append(record)
        manifest["updated_at"] = now_utc()
        manifest["papers"] = sorted(papers, key=lambda item: item.get("title", ""))
        write_json(self.reference_manifest_path(), manifest)

    def write_reference_pdf(self, filename: str, data: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        papers_dir = self.project_root / "references" / "papers"
        ensure_dir(papers_dir)
        tmp_path = papers_dir / f".tmp_{uuid.uuid4().hex}_{filename}"
        tmp_path.write_bytes(data)
        final_path = papers_dir / filename
        tmp_path.replace(final_path)
        record = dict(metadata)
        record["local_pdf_path"] = final_path.relative_to(self.project_root).as_posix()
        record["downloaded_at"] = now_utc()
        record["sha256"] = sha256_file(final_path)
        self.register_reference_paper(record)
        return record

    def _empty_manifest(self, stage_key: str) -> dict[str, Any]:
        return {
            "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
            "project_id": self.project_root.name,
            "stage_key": stage_key,
            "stage": STAGE_LABELS[stage_key],
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "artifact_count": 0,
            "artifacts": [],
        }

    def _normalize_manifest(self, stage_key: str, manifest: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
            "project_id": manifest.get("project_id") or self.project_root.name,
            "stage_key": manifest.get("stage_key") or stage_key,
            "stage": manifest.get("stage") or STAGE_LABELS[stage_key],
            "created_at": manifest.get("created_at") or now_utc(),
            "updated_at": manifest.get("updated_at") or now_utc(),
            "artifacts": [],
        }
        deduped: dict[str, dict[str, Any]] = {}
        for item in manifest.get("artifacts", []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            entry = self._normalize_artifact_entry(stage_key, item)
            deduped[entry["path"]] = entry
        normalized["artifacts"] = sorted(deduped.values(), key=lambda item: item["path"])
        normalized["artifact_count"] = len(normalized["artifacts"])
        return normalized

    def _normalize_artifact_entry(self, stage_key: str, item: dict[str, Any]) -> dict[str, Any]:
        artifact_type = item.get("type") or item.get("artifact_type") or "artifact"
        entry = dict(item)
        entry["schema_version"] = item.get("schema_version") or ARTIFACT_RECORD_SCHEMA_VERSION
        entry["type"] = artifact_type
        entry["artifact_type"] = artifact_type
        entry["source_paths"] = list(item.get("source_paths") or [])
        entry["metadata"] = dict(item.get("metadata") or {})
        entry["created_by"] = item.get("created_by") or _default_created_by(stage_key, artifact_type)
        entry["created_at"] = item.get("created_at") or now_utc()
        entry["status"] = item.get("status") or "committed"
        entry["validator"] = item.get("validator") or _default_validator(stage_key, artifact_type)
        path = self.project_root / str(entry["path"])
        if path.exists() and path.is_file():
            if not entry.get("sha256"):
                entry["sha256"] = sha256_file(path)
            if not entry.get("size_bytes"):
                entry["size_bytes"] = path.stat().st_size
        return entry


def _default_created_by(stage_key: str, artifact_type: str) -> str:
    if artifact_type.startswith(("c2c_patch", "c2c_code_patch", "c2c_frozen_patch")):
        return "codex-code-patch-agent"
    return _STAGE_CREATED_BY.get(stage_key, "auto-research")


def _default_validator(stage_key: str, artifact_type: str) -> str:
    prefix = _STAGE_VALIDATOR_PREFIX.get(stage_key, "stage")
    safe_type = "".join(char if char.isalnum() else "_" for char in artifact_type.lower()).strip("_")
    safe_type = "_".join(part for part in safe_type.split("_") if part)
    return f"{prefix}_{safe_type or 'artifact'}_schema_v1"
