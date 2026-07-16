"""Pure validation for authoritative failure, resource-pause, and resume evidence."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .domain_contracts import (
    FAILURE_EVIDENCE_SCHEMA_VERSION,
    RESOURCE_PROBE_SCHEMA_VERSION,
    RESUME_EVIDENCE_SCHEMA_VERSION,
)

FAILURE_EVIDENCE_SCHEMA = f"{FAILURE_EVIDENCE_SCHEMA_VERSION.removeprefix('auto_research_')}.schema.json"
RESOURCE_PROBE_SCHEMA = f"{RESOURCE_PROBE_SCHEMA_VERSION.removeprefix('auto_research_').replace('_evidence_', '_')}.schema.json"
RESUME_EVIDENCE_SCHEMA = f"{RESUME_EVIDENCE_SCHEMA_VERSION.removeprefix('auto_research_')}.schema.json"

_RESOURCE_FAILURE_CLASSES = {"resource_pause", "oom_retry"}
_IDENTITY_FIELDS = (
    "attempt_id",
    "direction_semantic_hash",
    "direction_spec_hash",
    "variant_semantic_hash",
    "variant_spec_hash",
    "trial_spec_hash",
    "protocol_hash",
    "sample_manifest_hash",
    "evaluator_hash",
    "lifecycle_generation",
    "implementation_hash",
    "attempt_input_hash",
    "phase",
    "phase_execution_id",
    "phase_start_event_id",
    "producer_run_id",
)
_STABLE_RESUME_FIELDS = (
    "attempt_id",
    "direction_semantic_hash",
    "direction_spec_hash",
    "variant_semantic_hash",
    "variant_spec_hash",
    "trial_spec_hash",
    "protocol_hash",
    "sample_manifest_hash",
    "evaluator_hash",
    "implementation_hash",
    "attempt_input_hash",
)
_RESOURCE_FIELDS = (
    "resource_type",
    "resource_id",
    "required_capacity",
    "observed_capacity",
    "unit",
    "probe_status",
)


def validate_failure_evidence(
    expected_identity: Mapping[str, Any],
    failure_raw: bytes,
    *,
    phase_run_receipt_raw: bytes | None = None,
    resource_probe_raw: bytes | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate one failure from canonical bytes against exact phase authority."""

    expected = _require_identity(expected_identity)
    if expected["phase"] == "resume":
        raise ValueError("failure authority phase cannot be resume")
    failure = _decode_canonical(failure_raw, FAILURE_EVIDENCE_SCHEMA, label="failure evidence")
    _match_identity(failure, expected, label="failure evidence")
    if failure["source_phase"] != failure["phase"]:
        raise ValueError("failure source_phase does not match authoritative phase")

    if failure["failure_class"] in _RESOURCE_FAILURE_CLASSES:
        if phase_run_receipt_raw is not None:
            raise ValueError("resource failure must not substitute a command receipt for a resource probe")
        if resource_probe_raw is None:
            raise ValueError("resource failure requires canonical resource probe bytes")
        probe = _decode_canonical(resource_probe_raw, RESOURCE_PROBE_SCHEMA, label="resource probe")
        _match_identity(probe, expected, label="resource probe")
        _match_digest(
            failure["cross_references"]["resource_probe_hash"],
            resource_probe_raw,
            label="resource probe",
        )
        _validate_insufficient_probe(probe)
        if failure["command_status"] != "resource_paused" or failure["exit_code"] == 0:
            raise ValueError("resource failure must be paused with a non-zero exit code")
        return {"failure_evidence": deepcopy(failure), "resource_probe": deepcopy(probe)}

    if resource_probe_raw is not None:
        raise ValueError("non-resource failure must not substitute a resource probe for a command receipt")
    if phase_run_receipt_raw is None:
        raise ValueError("non-resource failure requires canonical PhaseRunReceipt bytes")
    receipt = _decode_canonical(
        phase_run_receipt_raw,
        "phase_run_receipt_v4.schema.json",
        label="PhaseRunReceipt",
    )
    _match_receipt_identity(receipt, expected)
    _match_digest(
        failure["cross_references"]["phase_run_receipt_hash"],
        phase_run_receipt_raw,
        label="PhaseRunReceipt",
    )
    _validate_failed_receipt(receipt)
    expected_status = {
        "implementation_failure": "failed",
        "activation_failure": "failed",
        "integrity_failure": "integrity_blocked",
        "safety_failure": "integrity_blocked",
    }[failure["failure_class"]]
    if failure["command_status"] != expected_status:
        raise ValueError("failure class contradicts command status")
    if failure["exit_code"] != receipt["exit_code"]:
        raise ValueError("failure exit_code does not match canonical command receipt")
    if failure["log_hash"] != receipt["stderr_hash"]:
        raise ValueError("failure log_hash does not match canonical command stderr")
    if failure["receipt_hash"] != evidence_bytes_hash(phase_run_receipt_raw):
        raise ValueError("failure receipt_hash does not match canonical PhaseRunReceipt")
    return {"failure_evidence": deepcopy(failure), "phase_run_receipt": deepcopy(receipt)}


def _match_receipt_identity(receipt: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field_name in (
        "attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash",
        "phase_execution_id", "phase_start_event_id", "producer_run_id",
    ):
        if receipt.get(field_name) != expected[field_name]:
            raise ValueError(f"PhaseRunReceipt {field_name} mismatch")


def validate_resume_evidence(
    expected_identity: Mapping[str, Any],
    resume_raw: bytes,
    *,
    resource_probe_raw: bytes,
    expected_pause_identity: Mapping[str, Any] | None = None,
    pause_failure_raw: bytes | None = None,
    pause_resource_probe_raw: bytes | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate a resume decision and its canonical available-resource probe."""

    expected = _require_identity(expected_identity)
    if expected["phase"] != "resume":
        raise ValueError("resume authority phase must be resume")
    resume = _decode_canonical(resume_raw, RESUME_EVIDENCE_SCHEMA, label="resume evidence")
    probe = _decode_canonical(resource_probe_raw, RESOURCE_PROBE_SCHEMA, label="resource probe")
    _match_identity(resume, expected, label="resume evidence")
    _match_identity(probe, expected, label="resource probe")
    _match_digest(
        resume["cross_references"]["resource_probe_hash"],
        resource_probe_raw,
        label="resource probe",
    )
    _validate_available_probe(probe)
    for field_name in _RESOURCE_FIELDS:
        if resume[field_name] != probe[field_name]:
            raise ValueError(f"resume {field_name} does not match canonical resource probe")

    pause_expected = None
    if expected_pause_identity is not None:
        pause_expected = _require_identity(expected_pause_identity)
        if pause_expected["phase"] == "resume":
            raise ValueError("pause authority phase cannot be resume")
        for field_name in _STABLE_RESUME_FIELDS:
            if pause_expected[field_name] != expected[field_name]:
                raise ValueError(f"resume lineage does not match pause {field_name}")
        if resume["pause_phase"] != pause_expected["phase"]:
            raise ValueError("resume pause_phase does not match pause authority")
        if resume["pause_phase_execution_id"] != pause_expected["phase_execution_id"]:
            raise ValueError("resume pause_phase_execution_id does not match pause authority")
        if resume["pause_producer_run_id"] != pause_expected["producer_run_id"]:
            raise ValueError("resume pause_producer_run_id does not match pause authority")

    result = {"resume_evidence": deepcopy(resume), "resource_probe": deepcopy(probe)}
    if (pause_failure_raw is None) != (pause_resource_probe_raw is None):
        raise ValueError("resume validation requires both pause failure and pause resource probe bytes")
    if pause_failure_raw is not None and pause_resource_probe_raw is not None:
        pause = _decode_canonical(pause_failure_raw, FAILURE_EVIDENCE_SCHEMA, label="pause failure evidence")
        pause_probe = _decode_canonical(
            pause_resource_probe_raw,
            RESOURCE_PROBE_SCHEMA,
            label="pause resource probe",
        )
        _match_digest(resume["pause_evidence_hash"], pause_failure_raw, label="pause failure evidence")
        if pause["failure_class"] not in _RESOURCE_FAILURE_CLASSES:
            raise ValueError("resume pause evidence is not a resource failure")
        _match_digest(
            pause["cross_references"]["resource_probe_hash"],
            pause_resource_probe_raw,
            label="pause resource probe",
        )
        _validate_insufficient_probe(pause_probe)
        if pause_expected is not None:
            _match_identity(pause, pause_expected, label="pause failure evidence")
            _match_identity(pause_probe, pause_expected, label="pause resource probe")
        for field_name in _STABLE_RESUME_FIELDS:
            if pause[field_name] != resume[field_name]:
                raise ValueError(f"resume lineage does not match pause evidence {field_name}")
        if resume["pause_phase"] != pause["phase"]:
            raise ValueError("resume pause_phase does not match pause evidence")
        if resume["pause_phase_execution_id"] != pause["phase_execution_id"]:
            raise ValueError("resume pause_phase_execution_id does not match pause evidence")
        if resume["pause_producer_run_id"] != pause["producer_run_id"]:
            raise ValueError("resume pause_producer_run_id does not match pause evidence")
        for field_name in ("resource_type", "resource_id", "required_capacity", "unit"):
            if resume[field_name] != pause_probe[field_name]:
                raise ValueError(f"resume {field_name} does not match pause resource probe")
        result["pause_failure_evidence"] = deepcopy(pause)
        result["pause_resource_probe"] = deepcopy(pause_probe)
    return result


def canonical_evidence_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode the sole accepted JSON representation for these evidence types."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def evidence_bytes_hash(raw: bytes) -> str:
    """Hash immutable evidence bytes without decoding or normalizing them."""

    if not isinstance(raw, bytes):
        raise TypeError("evidence must be supplied as immutable bytes")
    return hashlib.sha256(raw).hexdigest()


def _decode_canonical(raw: bytes, schema_name: str, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise TypeError(f"{label} must be supplied as immutable bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to an object")
    try:
        canonical = canonical_evidence_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-canonical JSON values") from exc
    if canonical != raw:
        raise ValueError(f"{label} bytes are not canonical JSON")
    _validate_schema(payload, schema_name, label=label)
    return payload


def _validate_schema(payload: Mapping[str, Any], schema_name: str, *, label: str) -> None:
    errors = sorted(_schema_validator(schema_name).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    messages = []
    for error in errors[:20]:
        location = ".".join(str(item) for item in error.absolute_path) or "$"
        messages.append(f"{location}: {error.message}")
    raise ValueError(f"{label} schema violation: {'; '.join(messages)}")


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Draft202012Validator:
    schema_path = Path(__file__).resolve().parent / "schemas" / schema_name
    return Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))


def _require_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("expected identity must be a mapping")
    missing = [field_name for field_name in _IDENTITY_FIELDS if field_name not in identity]
    if missing:
        raise ValueError(f"expected identity is missing fields: {missing}")
    return {field_name: identity[field_name] for field_name in _IDENTITY_FIELDS}


def _match_identity(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for field_name in _IDENTITY_FIELDS:
        if payload.get(field_name) != expected[field_name]:
            raise ValueError(f"{label} {field_name} does not match authoritative identity")


def _match_digest(expected_hash: str, raw: bytes, *, label: str) -> None:
    if evidence_bytes_hash(raw) != expected_hash:
        raise ValueError(f"{label} raw bytes do not match referenced hash")


def _validate_failed_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt["exit_code"] == 0:
        raise ValueError("canonical PhaseRunReceipt does not prove command failure")
    started = _parse_datetime(receipt["started_at"], label="command started_at")
    finished = _parse_datetime(receipt["completed_at"], label="command completed_at")
    if finished < started:
        raise ValueError("PhaseRunReceipt completed_at precedes started_at")


def _validate_insufficient_probe(probe: Mapping[str, Any]) -> None:
    _validate_finite_capacity(probe)
    if probe["probe_status"] != "insufficient":
        raise ValueError("resource pause requires an insufficient probe")
    if not probe["observed_capacity"] < probe["required_capacity"]:
        raise ValueError("resource pause requires observed_capacity < required_capacity")


def _validate_available_probe(probe: Mapping[str, Any]) -> None:
    _validate_finite_capacity(probe)
    if probe["probe_status"] != "available":
        raise ValueError("resume requires an available probe")
    if not probe["observed_capacity"] >= probe["required_capacity"]:
        raise ValueError("resume requires observed_capacity >= required_capacity")


def _validate_finite_capacity(probe: Mapping[str, Any]) -> None:
    for field_name in ("required_capacity", "observed_capacity"):
        value = probe[field_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"resource probe {field_name} must be finite")


def _parse_datetime(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed
