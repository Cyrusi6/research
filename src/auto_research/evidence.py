"""Canonical scientific evidence decoding from immutable attempt-scoped bytes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft202012Validator

EXECUTION_OBSERVATION_SCHEMA_VERSION = "auto_research_execution_observation_v3"
EVIDENCE_MANIFEST_SCHEMA_VERSION = "auto_research_evidence_manifest_v2"
QUANTITATIVE_EVIDENCE_SCHEMA_VERSIONS = {
    "main_results": "auto_research_main_results_v2",
    "ablation_results": "auto_research_ablation_results_v2",
    "coverage_results": "auto_research_coverage_results_v2",
    "matched_control_results": "auto_research_matched_control_results_v2",
}
EVIDENCE_SCHEMA_VERSIONS = {
    **QUANTITATIVE_EVIDENCE_SCHEMA_VERSIONS,
    "activation_evidence": "auto_research_activation_evidence_v2",
    "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v2",
    "proxy_cache_report": "auto_research_proxy_cache_report_v2",
    "effective_proxy_policy": "auto_research_effective_proxy_policy_v2",
    "proxy_calibration_policy": "auto_research_proxy_calibration_policy_v2",
    "proxy_decision_report": "auto_research_proxy_decision_report_v2",
    "full_s3_readiness": "auto_research_full_s3_readiness_v2",
    "bootstrap_completion": "auto_research_bootstrap_completion_v2",
    "failure_evidence": "auto_research_failure_evidence_v2",
    "resource_probe": "auto_research_resource_probe_evidence_v1",
    "resume_evidence": "auto_research_resume_evidence_v2",
}

_SCHEMA_FILES = {
    kind: f"{kind}_v{version.rsplit('_v', 1)[1]}.schema.json"
    for kind, version in EVIDENCE_SCHEMA_VERSIONS.items()
}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


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

    _validate_schema(manifest, "evidence_manifest_v2.schema.json")
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


def encode_canonical_evidence(payload: Mapping[str, Any]) -> bytes:
    """Return the only accepted byte representation for evidence JSON."""

    return canonical_json(payload).encode("utf-8")


def evidence_content_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(encode_canonical_evidence(payload)).hexdigest()


def content_addressed_evidence_path(
    *, attempt_id: str, producer_run_id: str, evidence_kind: str, content_hash: str
) -> str:
    if evidence_kind not in EVIDENCE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported evidence kind: {evidence_kind}")
    if not _SHA256.fullmatch(content_hash):
        raise ValueError("content hash must be lowercase SHA-256")
    for value, label in ((attempt_id, "attempt_id"), (producer_run_id, "producer_run_id")):
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError(f"unsafe {label}")
    return f"experiment/attempts/{attempt_id}/{producer_run_id}/{evidence_kind}/{content_hash}.json"


def _decode_quantitative_rows(payload: Mapping[str, Any], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed_roles = {
        "main_results": {"baseline", "candidate"},
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
            "evidence_id": entry["evidence_id"],
            "evidence_kind": entry["kind"],
            "raw_artifact_path": entry["relative_path"],
            "raw_artifact_hash": entry["content_hash"],
        }
        _validate_schema(observation, "execution_observation_v3.schema.json")
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
        ):
            if row.get(key) != expected[key]:
                raise ValueError(f"quantitative row {key} mismatch: {entry['evidence_id']}")
    if canonical_hash(trial_spec) != payload["trial_spec_hash"]:
        raise ValueError("evidence does not bind the frozen TrialSpec")


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
    elif kind == "effective_proxy_policy":
        body = {
            "required_phases": payload["required_phases"],
            "proxy_terminal_allowed": payload["proxy_terminal_allowed"],
            "decision_threshold": payload["decision_threshold"],
        }
        if payload["policy_hash"] != canonical_hash(body):
            raise ValueError("effective proxy policy hash is not reproducible")
    elif kind == "proxy_calibration_policy":
        body = {
            "status": payload["status"],
            "calibration_metric": payload["calibration_metric"],
            "calibration_value": payload["calibration_value"],
            "cross_references": payload["cross_references"],
        }
        if payload["calibration_hash"] != canonical_hash(body):
            raise ValueError("proxy calibration hash is not reproducible")
    elif kind == "activation_evidence":
        passed = payload["status"] == "passed"
        if passed != (payload["command_status"] == "completed" and payload["exit_code"] == 0):
            raise ValueError("activation status contradicts command evidence")
    elif kind == "failure_evidence":
        expected_status = {
            "implementation_failure": "failed",
            "activation_failure": "failed",
            "resource_pause": "resource_paused",
            "oom_retry": "resource_paused",
            "integrity_failure": "integrity_blocked",
            "safety_failure": "integrity_blocked",
        }[payload["failure_class"]]
        if payload["command_status"] != expected_status:
            raise ValueError("failure class contradicts command status")
        if payload["exit_code"] == 0:
            raise ValueError("failure evidence cannot report a successful exit code")
    elif kind in {"resource_probe", "resume_evidence"}:
        available = payload["observed_capacity"] >= payload["required_capacity"]
        if (payload["probe_status"] == "available") != available:
            raise ValueError("resource probe status contradicts observed capacity")


def _observation_identity(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        observation["phase"],
        observation["role"],
        observation["dataset_id"],
        observation["metric_id"],
        observation["seed"],
    )


def _validate_schema(payload: Any, schema_name: str) -> None:
    schema = json.loads((_schema_dir() / schema_name).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        messages = []
        for error in errors[:20]:
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            messages.append(f"{location}: {error.message}")
        raise ValueError("; ".join(messages))


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"
