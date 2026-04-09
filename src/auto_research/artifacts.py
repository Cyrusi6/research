"""Artifact tracking and atomic writes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .constants import STAGE_LABELS
from .utils import ensure_dir, now_utc, read_json, sha256_file, write_json


class ArtifactManager:
    """Owns stage output registration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def stage_dir(self, stage_key: str) -> Path:
        return self.project_root / STAGE_LABELS[stage_key]

    def manifest_path(self, stage_key: str) -> Path:
        return self.stage_dir(stage_key) / "stage_manifest.json"

    def load_manifest(self, stage_key: str) -> dict[str, Any]:
        return read_json(self.manifest_path(stage_key), default={"stage": STAGE_LABELS[stage_key], "artifacts": []})

    def _save_manifest(self, stage_key: str, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = now_utc()
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
    ) -> dict[str, Any]:
        manifest = self.load_manifest(stage_key)
        rel_path = final_path.relative_to(self.project_root).as_posix()
        entry = {
            "path": rel_path,
            "artifact_type": artifact_type,
            "summary": summary,
            "source_paths": source_paths or [],
            "metadata": metadata or {},
            "sha256": sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "created_at": now_utc(),
            "status": "committed",
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
