"""Canonical scientific evidence decoding from immutable attempt-scoped bytes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator

EXECUTION_OBSERVATION_SCHEMA_VERSION = "auto_research_execution_observation_v4"
EVIDENCE_MANIFEST_SCHEMA_VERSION = "auto_research_evidence_manifest_v4"
COMPLETION_EVIDENCE_SCHEMA_VERSION = "auto_research_completion_evidence_v3"
QUANTITATIVE_EVIDENCE_SCHEMA_VERSIONS = {
    "main_results": "auto_research_main_results_v3",
    "proxy_results": "auto_research_proxy_results_v1",
    "ablation_results": "auto_research_ablation_results_v3",
    "coverage_results": "auto_research_coverage_results_v3",
    "matched_control_results": "auto_research_matched_control_results_v3",
}
EVIDENCE_SCHEMA_VERSIONS = {
    **QUANTITATIVE_EVIDENCE_SCHEMA_VERSIONS,
    "activation_evidence": "auto_research_activation_evidence_v3",
    "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v3",
    "proxy_cache_report": "auto_research_proxy_cache_report_v3",
    "full_s3_readiness": "auto_research_full_s3_readiness_v3",
    "bootstrap_completion": "auto_research_bootstrap_completion_v3",
}

TRANSACTION_EVIDENCE_SCHEMA_VERSIONS = {
    "failure_evidence": "auto_research_failure_evidence_v5",
    "resource_probe": "auto_research_resource_probe_evidence_v4",
    "resume_evidence": "auto_research_resume_evidence_v5",
}

CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS = {
    **EVIDENCE_SCHEMA_VERSIONS,
    **TRANSACTION_EVIDENCE_SCHEMA_VERSIONS,
}

_SCHEMA_FILES = {
    kind: f"{kind}_v{version.rsplit('_v', 1)[1]}.schema.json"
    for kind, version in CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS.items()
}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_EXECUTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")


class EvidenceStore:
    """Single fail-closed reader for attempt-scoped immutable evidence."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.root = self.project_root / "experiment" / "attempts"

    def read_entry(self, entry: Mapping[str, Any], attempt: Mapping[str, Any]) -> bytes:
        _validate_content_addressed_path(entry, attempt)
        relative = PurePosixPath(str(entry["relative_path"]))
        expected_prefix = PurePosixPath("experiment") / "attempts"
        if relative.parts[:2] != expected_prefix.parts:
            raise ValueError("evidence path is outside the authoritative evidence root")
        return self._read_relative(relative)

    def read_staged_source(self, relative_path: str) -> bytes:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or not relative.parts or "." in relative.parts or ".." in relative.parts:
            raise ValueError("staged evidence path traversal is forbidden")
        return self._read_relative(relative)

    def write_entry(self, entry: Mapping[str, Any], attempt: Mapping[str, Any], raw: bytes) -> None:
        _validate_content_addressed_path(entry, attempt)
        if hashlib.sha256(raw).hexdigest() != entry["content_hash"]:
            raise ValueError("immutable evidence bytes do not match content hash")
        relative = PurePosixPath(str(entry["relative_path"]))
        parent_fd = self._open_parent(relative, create=True)
        try:
            try:
                existing_fd = os.open(relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                try:
                    existing = self._read_fd(existing_fd)
                finally:
                    os.close(existing_fd)
                if existing != raw:
                    raise ValueError("content-addressed evidence collision")
                return
            temporary = f".{relative.parts[-1]}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            file_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(file_fd, view)
                    view = view[written:]
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.rename(temporary, relative.parts[-1], src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise ValueError("immutable evidence path contains a symlink or is unavailable") from exc
        finally:
            os.close(parent_fd)

    def _read_relative(self, relative: PurePosixPath) -> bytes:
        parent_fd: int | None = None
        try:
            parent_fd = self._open_parent(relative, create=False)
            file_fd = os.open(relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            try:
                return self._read_fd(file_fd)
            finally:
                os.close(file_fd)
        except OSError as exc:
            raise ValueError("evidence path contains a symlink or is unavailable") from exc
        finally:
            if parent_fd is not None:
                os.close(parent_fd)

    def _open_parent(self, relative: PurePosixPath, *, create: bool) -> int:
        current_fd = os.open(self.project_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in relative.parts[:-1]:
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except Exception:
            os.close(current_fd)
            raise

    @staticmethod
    def _read_fd(file_fd: int) -> bytes:
        stat = os.fstat(file_fd)
        if not __import__("stat").S_ISREG(stat.st_mode) or stat.st_nlink != 1:
            raise ValueError("evidence artifact must be a unique regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def validate_execution_id(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _SAFE_EXECUTION_ID.fullmatch(value):
        raise ValueError(f"{field} must use safe execution identity characters")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def decode_evidence_inventory(
    *,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evidence_bytes: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate immutable bytes once and deterministically decode observations.

    ``evidence_bytes`` is keyed by evidence_id. Callers must pass the exact bytes
    that were content-addressed; this function never reopens an artifact path.
    """

    _validate_schema(manifest, "evidence_manifest_v4.schema.json")
    _validate_manifest_identity(attempt, trial_spec, manifest)
    entries = manifest["entries"]
    entry_ids = [entry["evidence_id"] for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("duplicate evidence manifest identity")
    if set(evidence_bytes) != set(entry_ids):
        raise ValueError("evidence bytes must exactly match the manifest inventory")

    observations: list[dict[str, Any]] = []
    decoded: dict[str, dict[str, Any]] = {}
    for entry in entries:
        evidence_id = entry["evidence_id"]
        raw = evidence_bytes[evidence_id]
        if not isinstance(raw, bytes):
            raise ValueError("evidence inventory values must be immutable bytes")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["content_hash"]:
            raise ValueError(f"evidence content hash mismatch: {evidence_id}")
        _validate_content_addressed_path(entry, attempt)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"evidence is not canonical JSON: {evidence_id}") from exc
        if canonical_json(payload).encode("utf-8") != raw:
            raise ValueError(f"evidence bytes are not canonical JSON: {evidence_id}")
        expected_version = EVIDENCE_SCHEMA_VERSIONS[entry["kind"]]
        if entry["schema_version"] != expected_version:
            raise ValueError(f"evidence manifest schema version mismatch: {evidence_id}")
        _validate_schema(payload, _SCHEMA_FILES[entry["kind"]])
        _validate_evidence_identity(payload, entry, attempt, trial_spec)
        _validate_cross_references(payload, entry, manifest)
        _validate_evidence_semantics(payload)
        decoded[evidence_id] = deepcopy(payload)
        if entry["kind"] in QUANTITATIVE_EVIDENCE_SCHEMA_VERSIONS:
            observations.extend(_decode_quantitative_rows(payload, entry))

    identities = [_observation_identity(item) for item in observations]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate quantitative measurement row identity")
    return observations, decoded


def decode_receipt_bound_evidence_inventory(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
):
    """Decode evidence only after proving completed command receipt lineage.

    Imported lazily to keep the low-level byte decoder independent from ledger
    projections while exposing one production integration point.
    """

    from .evidence_lineage import validate_receipt_bound_evidence

    return validate_receipt_bound_evidence(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        manifest=manifest,
        phase_commands=phase_commands,
        phase=phase,
    )


def stage_completion_evidence(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    inventory: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stage raw phase outputs without accepting caller-derived conclusions.

    The returned CompletionEvidence v3 contains only immutable evidence facts.
    Command, receipt, observations, constraints, outcome, and summary are not
    caller fields; the authoritative transaction derives them later.
    """

    if not inventory:
        raise ValueError("completion evidence inventory is empty")
    first_raw = EvidenceStore(project_root).read_staged_source(str(inventory[0]["source_path"]))
    try:
        first_payload = json.loads(first_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("staged evidence is not canonical JSON") from exc
    phase = str(first_payload.get("phase") or "") if isinstance(first_payload, dict) else ""
    phase_execution = (attempt.get("phase_executions") or {}).get(phase)
    if phase not in {"proxy", "full"} or not isinstance(phase_execution, Mapping):
        raise ValueError("completion evidence requires an authoritative running phase")
    expected_kinds = {
        str(requirement["kind"])
        for requirement in trial_spec.get("evidence_requirements", [])
        if phase in requirement.get("applicable_phases", []) or "always" in requirement.get("applicable_phases", [])
    }
    supplied_kinds = [str(item.get("kind") or "") for item in inventory]
    if len(supplied_kinds) != len(set(supplied_kinds)) or set(supplied_kinds) != expected_kinds:
        raise ValueError("completion evidence kinds must exactly match the frozen phase contract")

    store = EvidenceStore(project_root)
    entries: list[dict[str, Any]] = []
    for item in inventory:
        kind = str(item["kind"])
        raw = store.read_staged_source(str(item["source_path"]))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("staged evidence is not canonical JSON") from exc
        if not isinstance(payload, dict) or canonical_json(payload).encode("utf-8") != raw:
            raise ValueError("staged evidence bytes are not canonical JSON")
        _validate_schema(payload, _SCHEMA_FILES[kind])
        digest = hashlib.sha256(raw).hexdigest()
        entry = {
            "evidence_id": str(payload["evidence_id"]),
            "kind": kind,
            "relative_path": content_addressed_evidence_path(
                attempt_id=str(attempt["attempt_id"]),
                producer_run_id=str(payload["producer_run_id"]),
                evidence_kind=kind,
                content_hash=digest,
            ),
            "content_hash": digest,
            "schema_version": str(payload["schema_version"]),
            "attempt_id": str(attempt["attempt_id"]),
            "producer_run_id": str(payload["producer_run_id"]),
            "direction_semantic_hash": str(attempt["direction_semantic_hash"]),
            "direction_spec_hash": str(attempt["direction_spec_hash"]),
            "variant_semantic_hash": str(attempt["variant_semantic_hash"]),
            "variant_spec_hash": str(attempt["variant_spec_hash"]),
            "trial_spec_hash": str(attempt["trial_spec_hash"]),
            "protocol_hash": str(attempt["protocol_hash"]),
            "sample_manifest_hash": str(attempt["sample_manifest_hash"]),
            "evaluator_hash": str(attempt["evaluator_hash"]),
            "lifecycle_generation": attempt["lifecycle_generation"],
            "implementation_hash": str(attempt["implementation_hash"]),
            "attempt_input_hash": str(attempt["attempt_input_hash"]),
            "phase": phase,
            "phase_execution_id": str(phase_execution["phase_execution_id"]),
            "phase_start_event_id": str(phase_execution["phase_start_event_id"]),
        }
        identity_entry = {**entry, "cross_references": deepcopy(payload.get("cross_references", {}))}
        _validate_evidence_identity(payload, identity_entry, attempt, trial_spec)
        _validate_evidence_semantics(payload)
        store.write_entry(entry, attempt, raw)
        entries.append(entry)

    completion = {
        "schema_version": COMPLETION_EVIDENCE_SCHEMA_VERSION,
        "attempt_id": str(attempt["attempt_id"]),
        "trial_spec_hash": str(attempt["trial_spec_hash"]),
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": str(attempt["implementation_hash"]),
        "attempt_input_hash": str(attempt["attempt_input_hash"]),
        "phase": phase,
        "phase_execution_id": str(phase_execution["phase_execution_id"]),
        "producer_run_id": str(phase_execution["producer_run_id"]),
        "command_plan_hash": str(phase_execution["command_plan_hash"]),
        "entries": entries,
    }
    _validate_schema(completion, "completion_evidence_v3.schema.json")
    return completion


def encode_canonical_evidence(payload: Mapping[str, Any]) -> bytes:
    """Return the only accepted byte representation for evidence JSON."""

    return canonical_json(payload).encode("utf-8")


def evidence_content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(encode_canonical_evidence(payload)).hexdigest()


def content_addressed_evidence_path(
    *, attempt_id: str, producer_run_id: str, evidence_kind: str, content_hash: str
) -> str:
    if evidence_kind not in CONTENT_ADDRESSED_EVIDENCE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported evidence kind: {evidence_kind}")
    if not _SHA256.fullmatch(content_hash):
        raise ValueError("content hash must be lowercase SHA-256")
    if not attempt_id or "/" in attempt_id or "\\" in attempt_id or attempt_id in {".", ".."}:
        raise ValueError("unsafe attempt_id")
    validate_execution_id(producer_run_id, field="producer_run_id")
    return f"experiment/attempts/{attempt_id}/{producer_run_id}/{evidence_kind}/{content_hash}.json"


def _decode_quantitative_rows(payload: Mapping[str, Any], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed_roles = {
        "main_results": {"baseline", "candidate"},
        "proxy_results": {"baseline", "candidate"},
        "ablation_results": {"ablation"},
        "coverage_results": {"coverage"},
        "matched_control_results": {"matched_control"},
    }[entry["kind"]]
    result = []
    for row in payload["rows"]:
        if row["role"] not in allowed_roles:
            raise ValueError(f"role {row['role']} is not permitted by {entry['kind']}")
        value = row["metric_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("quantitative metric_value must be a finite non-boolean number")
        identity = {
            "phase": row["phase"],
            "phase_execution_id": row["phase_execution_id"],
            "role": row["role"],
            "dataset_id": row["dataset_id"],
            "metric_id": row["metric_id"],
            "seed": row["seed"],
        }
        observation = {
            "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
            "observation_id": f"obs:{canonical_hash({'evidence_id': entry['evidence_id'], **identity})}",
            **identity,
            "command_status": row["command_status"],
            "metric_value": value,
            "attempt_id": row["attempt_id"],
            "variant_semantic_hash": row["variant_semantic_hash"],
            "variant_spec_hash": row["variant_spec_hash"],
            "trial_spec_hash": row["trial_spec_hash"],
            "sample_manifest_hash": row["sample_manifest_hash"],
            "evaluator_hash": row["evaluator_hash"],
            "producer_run_id": row["producer_run_id"],
            "lifecycle_generation": row["lifecycle_generation"],
            "implementation_hash": row["implementation_hash"],
            "attempt_input_hash": row["attempt_input_hash"],
            "phase_start_event_id": row["phase_start_event_id"],
            "evidence_id": entry["evidence_id"],
            "evidence_kind": entry["kind"],
            "raw_artifact_path": entry["relative_path"],
            "raw_artifact_hash": entry["content_hash"],
        }
        _validate_schema(observation, "execution_observation_v4.schema.json")
        result.append(observation)
    return result


def _validate_manifest_identity(
    attempt: Mapping[str, Any], trial_spec: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    expected = {
        "attempt_id": attempt["attempt_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"evidence manifest {key} mismatch")
    if canonical_hash(trial_spec) != attempt["trial_spec_hash"]:
        raise ValueError("frozen TrialSpec hash mismatch")


def _validate_evidence_identity(
    payload: Mapping[str, Any],
    entry: Mapping[str, Any],
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": entry["schema_version"],
        "evidence_kind": entry["kind"],
        "evidence_id": entry["evidence_id"],
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": entry["producer_run_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase": entry["phase"],
        "phase_execution_id": entry["phase_execution_id"],
        "phase_start_event_id": entry["phase_start_event_id"],
    }
    for key, value in expected.items():
        entry_value = entry.get("kind") if key == "evidence_kind" else entry.get(key)
        if payload.get(key) != value or entry_value != value:
            raise ValueError(f"evidence {key} mismatch: {entry['evidence_id']}")
    for row in payload.get("rows", []):
        for key in (
            "attempt_id",
            "producer_run_id",
            "variant_semantic_hash",
            "variant_spec_hash",
            "trial_spec_hash",
            "sample_manifest_hash",
            "evaluator_hash",
            "lifecycle_generation",
            "implementation_hash",
            "attempt_input_hash",
            "phase",
            "phase_execution_id",
            "phase_start_event_id",
        ):
            if row.get(key) != expected[key]:
                raise ValueError(f"quantitative row {key} mismatch: {entry['evidence_id']}")
    if canonical_hash(trial_spec) != payload["trial_spec_hash"]:
        raise ValueError("evidence does not bind the frozen TrialSpec")
    validate_execution_id(entry["producer_run_id"], field="producer_run_id")
    validate_execution_id(entry["phase_execution_id"], field="phase_execution_id")


def _validate_content_addressed_path(entry: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    path = PurePosixPath(entry["relative_path"])
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("evidence path traversal is forbidden")
    expected = content_addressed_evidence_path(
        attempt_id=attempt["attempt_id"],
        producer_run_id=entry["producer_run_id"],
        evidence_kind=entry["kind"],
        content_hash=entry["content_hash"],
    )
    if entry["relative_path"] != expected:
        raise ValueError("evidence path is not attempt-scoped and content-addressed")


def _validate_cross_references(
    payload: Mapping[str, Any], entry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    entry_hashes = {item["kind"]: item["content_hash"] for item in manifest["entries"]}
    for key, expected_hash in entry["cross_references"].items():
        if not key.endswith("_hash"):
            raise ValueError("evidence cross-reference keys must end in _hash")
        referenced_kind = key.removesuffix("_hash")
        if entry_hashes.get(referenced_kind) != expected_hash:
            raise ValueError(f"evidence cross-reference mismatch: {key}")
        if payload.get("cross_references", {}).get(key) != expected_hash:
            raise ValueError(f"evidence payload cross-reference mismatch: {key}")
    if payload.get("cross_references", {}) != entry["cross_references"]:
        raise ValueError("evidence cross-references differ from manifest")


def _validate_evidence_semantics(payload: Mapping[str, Any]) -> None:
    kind = payload["evidence_kind"]
    if kind == "proxy_baseline_fingerprint":
        if payload["baseline_hash"] != canonical_hash(payload["fingerprint_inputs"]):
            raise ValueError("proxy baseline fingerprint is not reproducible")
        inputs = payload["fingerprint_inputs"]
        for key in ("sample_manifest_hash", "evaluator_hash", "protocol_hash"):
            if inputs[key] != payload[key]:
                raise ValueError(f"proxy baseline {key} cross-reference mismatch")
    elif kind == "activation_evidence":
        passed = payload["status"] == "passed"
        if passed != (payload["command_status"] == "completed" and payload["exit_code"] == 0):
            raise ValueError("activation status contradicts command evidence")


def _observation_identity(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        observation["phase"],
        observation["phase_execution_id"],
        observation["role"],
        observation["dataset_id"],
        observation["metric_id"],
        observation["seed"],
    )


def _validate_schema(payload: Any, schema_name: str) -> None:
    errors = sorted(_schema_validator(schema_name).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        messages = []
        for error in errors[:20]:
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            messages.append(f"{location}: {error.message}")
        raise ValueError("; ".join(messages))


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema = json.loads((_schema_dir() / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"
