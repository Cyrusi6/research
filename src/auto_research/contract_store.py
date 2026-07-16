"""Secure content-addressed storage for immutable contract bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator

CONTRACT_BLOB_SCHEMA_VERSION = "auto_research_contract_blob_v1"
CONTRACT_REF_SCHEMA_VERSION = "auto_research_contract_ref_v1"
SAMPLE_MANIFEST_SCHEMA_VERSION = "auto_research_sample_manifest_v4"
EVALUATOR_MANIFEST_SCHEMA_VERSION = "auto_research_evaluator_manifest_v2"
CONTRACT_STORE_ROOT = PurePosixPath("meta/contracts/sha256")

_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def canonical_contract_bytes(payload: Any) -> bytes:
    """Encode JSON without relying on a mutable file or object hash."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def contract_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def content_addressed_contract_path(digest: str) -> str:
    _validate_digest(digest)
    return str(CONTRACT_STORE_ROOT / digest[:2] / f"{digest}.json")


def validate_schema(payload: Mapping[str, Any], schema_file: str) -> None:
    errors = sorted(_schema_validator(schema_file).iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path) or "<root>"
        raise ValueError(f"{schema_file} validation failed at {location}: {error.message}")


@lru_cache(maxsize=None)
def _schema_validator(schema_file: str) -> Draft202012Validator:
    schema_path = Path(__file__).resolve().parent / "schemas" / schema_file
    return Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))


def validate_sample_manifest(payload: Mapping[str, Any], *, store: ContractStore | None = None) -> None:
    validate_schema(payload, "sample_manifest_v4.schema.json")
    if store is not None:
        store._validate_known_contract(payload, "sample_manifest_v4.schema.json")


def validate_evaluator_manifest(payload: Mapping[str, Any], *, store: ContractStore | None = None) -> None:
    validate_schema(payload, "evaluator_manifest_v2.schema.json")
    if store is not None:
        store._validate_known_contract(payload, "evaluator_manifest_v2.schema.json")


class ContractStore:
    """Fail-closed CAS whose digest always proves the persisted bytes."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.root = self.project_root / CONTRACT_STORE_ROOT

    def put_bytes(self, raw: bytes, *, expected_digest: str | None = None) -> dict[str, Any]:
        if not isinstance(raw, bytes):
            raise TypeError("contract bytes must be bytes")
        digest = contract_digest(raw)
        if expected_digest is not None:
            _validate_digest(expected_digest)
            if digest != expected_digest:
                raise ValueError("contract bytes do not match expected digest")
        reference = self.reference(digest, size_bytes=len(raw))
        self.write_bytes(reference, raw)
        return reference

    def put_json(
        self,
        payload: Mapping[str, Any],
        *,
        schema_file: str | None = None,
        expected_digest: str | None = None,
    ) -> dict[str, Any]:
        if schema_file is not None:
            validate_schema(payload, schema_file)
            self._validate_known_contract(payload, schema_file)
        return self.put_bytes(canonical_contract_bytes(payload), expected_digest=expected_digest)

    def put_contract(
        self,
        payload: Mapping[str, Any],
        *,
        contract_kind: str,
        schema_file: str,
    ) -> dict[str, Any]:
        blob = self.put_json(payload, schema_file=schema_file)
        reference = {
            "schema_version": CONTRACT_REF_SCHEMA_VERSION,
            "contract_kind": contract_kind,
            "blob": blob,
        }
        validate_schema(reference, "contract_ref_v1.schema.json")
        return reference

    def read_contract(
        self,
        reference: Mapping[str, Any],
        *,
        contract_kind: str,
        schema_file: str,
    ) -> dict[str, Any]:
        validate_schema(reference, "contract_ref_v1.schema.json")
        if reference["contract_kind"] != contract_kind:
            raise ValueError("contract reference kind mismatch")
        return self.read_json(reference["blob"], schema_file=schema_file)

    def write_bytes(self, reference: Mapping[str, Any], raw: bytes) -> None:
        normalized = self._normalize_reference(reference)
        actual_digest = contract_digest(raw)
        if actual_digest != normalized["digest"]:
            raise ValueError("contract bytes do not match reference digest")
        if len(raw) != normalized["size_bytes"]:
            raise ValueError("contract bytes do not match reference size")
        relative = PurePosixPath(normalized["relative_path"])
        parent_fd: int | None = None
        temporary: str | None = None
        try:
            parent_fd = self._open_parent(relative, create=True)
            try:
                existing_fd = os.open(relative.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                try:
                    self._validate_open_regular_file(existing_fd)
                    existing = self._read_fd(existing_fd)
                finally:
                    os.close(existing_fd)
                if existing != raw:
                    raise ValueError("content-addressed contract collision")
                return
            temporary = f".{relative.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            file_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                self._validate_open_regular_file(file_fd)
                view = memoryview(raw)
                while view:
                    written = os.write(file_fd, view)
                    view = view[written:]
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.rename(temporary, relative.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temporary = None
            os.fsync(parent_fd)
        except OSError as exc:
            raise ValueError("contract path contains a symlink or is unavailable") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if parent_fd is not None:
                os.close(parent_fd)

    def read_bytes(self, reference: Mapping[str, Any] | str) -> bytes:
        normalized = self._normalize_reference(reference)
        relative = PurePosixPath(normalized["relative_path"])
        parent_fd: int | None = None
        try:
            parent_fd = self._open_parent(relative, create=False)
            file_fd = os.open(relative.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            try:
                self._validate_open_regular_file(file_fd)
                raw = self._read_fd(file_fd)
            finally:
                os.close(file_fd)
        except OSError as exc:
            raise ValueError("contract path contains a symlink or is unavailable") from exc
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
        if len(raw) != normalized["size_bytes"]:
            raise ValueError("persisted contract size does not match reference")
        if contract_digest(raw) != normalized["digest"]:
            raise ValueError("persisted contract bytes do not match reference digest")
        return raw

    def read_json(
        self,
        reference: Mapping[str, Any] | str,
        *,
        schema_file: str | None = None,
    ) -> dict[str, Any]:
        raw = self.read_bytes(reference)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("contract blob is not canonical JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("contract JSON root must be an object")
        if canonical_contract_bytes(payload) != raw:
            raise ValueError("contract JSON bytes are not canonical")
        if schema_file is not None:
            validate_schema(payload, schema_file)
            self._validate_known_contract(payload, schema_file)
        return payload

    def verify(self, reference: Mapping[str, Any] | str) -> dict[str, Any]:
        normalized = self._normalize_reference(reference)
        self.read_bytes(normalized)
        return normalized

    def reference(self, digest: str, *, size_bytes: int) -> dict[str, Any]:
        _validate_digest(digest)
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("contract size_bytes must be a non-negative integer")
        return {
            "schema_version": CONTRACT_BLOB_SCHEMA_VERSION,
            "algorithm": "sha256",
            "digest": digest,
            "size_bytes": size_bytes,
            "relative_path": content_addressed_contract_path(digest),
        }

    def digest_referenced_bytes(self, references: list[Mapping[str, Any]]) -> str:
        digest = hashlib.sha256()
        for reference in references:
            digest.update(self.read_bytes(reference))
        return digest.hexdigest()

    def _validate_known_contract(self, payload: Mapping[str, Any], schema_file: str) -> None:
        if schema_file == "sample_manifest_v4.schema.json":
            dataset_ids: set[str] = set()
            for dataset in payload["datasets"]:
                if dataset["dataset_id"] in dataset_ids:
                    raise ValueError("sample manifest contains duplicate dataset_id")
                dataset_ids.add(dataset["dataset_id"])
                if dataset["sample_count"] != len(dataset["ordered_sample_ids"]):
                    raise ValueError("sample_count does not match ordered_sample_ids")
                if len(set(dataset["ordered_sample_ids"])) != len(dataset["ordered_sample_ids"]):
                    raise ValueError("sample manifest contains duplicate ordered sample identity")
                if [item["digest"] for item in dataset["raw_sample_refs"]] != dataset["ordered_sample_ids"]:
                    raise ValueError("ordered sample identities do not match raw sample refs")
                actual = self.digest_referenced_bytes(dataset["raw_sample_refs"])
                if actual != dataset["content_digest"]:
                    raise ValueError("sample content_digest does not prove source blob bytes")
        elif schema_file == "evaluator_manifest_v2.schema.json":
            source_digest = self.digest_referenced_bytes(payload["source_blobs"])
            dependency_digest = self.digest_referenced_bytes(payload["dependency_blobs"])
            config_digest = self.digest_referenced_bytes([payload["config_blob"]])
            if source_digest != payload["source_digest"]:
                raise ValueError("evaluator source_digest does not prove source blob bytes")
            if dependency_digest != payload["dependency_digest"]:
                raise ValueError("evaluator dependency_digest does not prove dependency blob bytes")
            if config_digest != payload["config_digest"]:
                raise ValueError("evaluator config_digest does not prove config blob bytes")
        elif schema_file == "phase_run_receipt_v4.schema.json":
            for output in payload["outputs"]:
                reference = output["contract_ref"]
                self.verify(reference)
                if reference["digest"] != output["content_hash"]:
                    raise ValueError("receipt output content hash does not match ContractRef")

    store_bytes = put_bytes
    store_json = put_json

    def _normalize_reference(self, reference: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(reference, str):
            _validate_digest(reference)
            relative = PurePosixPath(content_addressed_contract_path(reference))
            path = self.project_root / relative
            try:
                size_bytes = path.stat(follow_symlinks=False).st_size
            except OSError as exc:
                raise ValueError("contract blob is unavailable") from exc
            reference = self.reference(reference, size_bytes=size_bytes)
        required = {"schema_version", "algorithm", "digest", "size_bytes", "relative_path"}
        if set(reference) != required:
            raise ValueError("contract reference fields are invalid")
        normalized = dict(reference)
        if normalized["schema_version"] != CONTRACT_BLOB_SCHEMA_VERSION or normalized["algorithm"] != "sha256":
            raise ValueError("unsupported contract reference")
        _validate_digest(normalized["digest"])
        expected_path = content_addressed_contract_path(normalized["digest"])
        if normalized["relative_path"] != expected_path:
            raise ValueError("contract reference path is not content-addressed")
        if not isinstance(normalized["size_bytes"], int) or isinstance(normalized["size_bytes"], bool) or normalized["size_bytes"] < 0:
            raise ValueError("contract reference size is invalid")
        return normalized

    def _open_parent(self, relative: PurePosixPath, *, create: bool) -> int:
        if relative.is_absolute() or not relative.parts or "." in relative.parts or ".." in relative.parts:
            raise ValueError("contract path traversal is forbidden")
        expected_prefix = CONTRACT_STORE_ROOT.parts
        if relative.parts[: len(expected_prefix)] != expected_prefix:
            raise ValueError("contract path is outside the authoritative contract root")
        root_fd = os.open(self.project_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        current_fd = root_fd
        try:
            for part in relative.parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    part,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _read_fd(file_fd: int) -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    @staticmethod
    def _validate_open_regular_file(file_fd: int) -> None:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("contract blob must be a regular file")
        if metadata.st_nlink != 1:
            raise ValueError("contract blob hard links are forbidden")


def _validate_digest(digest: str) -> None:
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("contract digest must be a lowercase sha256 hex string")
