"""Read-only immutable derivation and receipt lineage validation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contract_store import validate_schema
from .derivation_validation import ReceiptBoundSources, validate_immutable_derivation
from .evidence import EVIDENCE_MANIFEST_SCHEMA_VERSION, EvidenceStore, decode_evidence_inventory


_LINEAGE_FIELDS = {
    "command_id",
    "command_hash",
    "command_plan_hash",
    "receipt_ref",
    "receipt_hash",
    "output_ref",
    "completed_event_id",
    "derivation_ref",
    "derivation_hash",
}


@dataclass(frozen=True)
class ReceiptEvidenceLineage:
    evidence_id: str
    evidence_kind: str
    command_id: str
    command_hash: str
    completed_event_id: str
    receipt_ref: dict[str, Any]
    receipt_hash: str
    output_ref: dict[str, Any]
    derivation_ref: dict[str, Any]
    derivation_hash: str


@dataclass(frozen=True)
class ReceiptBoundEvidence:
    manifest: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    decoded_evidence: dict[str, dict[str, Any]]
    evidence_bytes: dict[str, bytes]
    lineage: dict[str, ReceiptEvidenceLineage]
    receipt_bound_sources: ReceiptBoundSources


def manifest_from_completion_evidence(
    *,
    attempt: Mapping[str, Any],
    completion_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build staged manifest facts without accepting caller-authored lineage."""

    expected_top_level = {
        "attempt_id": attempt["attempt_id"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
    }
    for key, value in expected_top_level.items():
        if completion_evidence.get(key) != value:
            raise ValueError(f"CompletionEvidence {key} mismatch")
    phase = completion_evidence.get("phase")
    phase_execution = (attempt.get("phase_executions") or {}).get(phase)
    if not isinstance(phase_execution, Mapping):
        raise ValueError("CompletionEvidence phase is not authoritative")
    for key in ("phase_execution_id", "producer_run_id", "command_plan_hash"):
        if completion_evidence.get(key) != phase_execution.get(key):
            raise ValueError(f"CompletionEvidence {key} mismatch")
    entries = completion_evidence.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("CompletionEvidence entries are required")
    return {
        "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt["attempt_id"],
        "direction_semantic_hash": attempt["direction_semantic_hash"],
        "direction_spec_hash": attempt["direction_spec_hash"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "phase": phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "producer_run_id": phase_execution["producer_run_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "entries": deepcopy(entries),
    }


def validate_receipt_bound_evidence(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
) -> ReceiptBoundEvidence:
    """Validate the one frozen physical→derive receipt→evidence chain."""

    if phase not in {"proxy", "full"}:
        raise ValueError("receipt evidence phase must be proxy or full")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("evidence manifest entries are required")
    expected_kinds = _expected_phase_kinds(trial_spec, phase)
    supplied_kinds = [
        str(entry.get("kind"))
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("phase") == phase
    ]
    if supplied_kinds != expected_kinds or len(supplied_kinds) != len(entries):
        raise ValueError("receipt evidence order/exact-set differs from frozen phase contract")

    validated = validate_immutable_derivation(
        project_root=project_root,
        attempt=attempt,
        trial_spec=trial_spec,
        phase_commands=phase_commands,
        phase=phase,
        evidence_manifest=manifest,
    )
    derive_record = validated.derive_record
    derive_command = derive_record["command"]
    derive_receipt_ref = derive_record["receipt_ref"]
    completed_event_id = derive_record.get("completed_event_id")
    if not isinstance(completed_event_id, str) or not completed_event_id:
        raise ValueError("completed derive command lacks event identity")

    evidence_store = EvidenceStore(project_root)
    evidence_bytes: dict[str, bytes] = {}
    lineage: dict[str, ReceiptEvidenceLineage] = {}
    canonical_entries: list[dict[str, Any]] = []
    for supplied_entry in entries:
        entry = deepcopy(dict(supplied_entry))
        kind = str(entry["kind"])
        output_ref = validated.output_refs[kind]
        normalized_raw = validated.output_bytes[kind]
        staged_raw = evidence_store.read_entry(entry, attempt)
        if staged_raw != normalized_raw:
            raise ValueError(f"attempt-scoped evidence differs from deterministic derive output: {kind}")
        try:
            payload = json.loads(normalized_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"normalized evidence is not JSON: {kind}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"normalized evidence root is not an object: {kind}")
        entry["cross_references"] = deepcopy(payload.get("cross_references") or {})
        derived_lineage = {
            "command_id": derive_command["command_id"],
            "command_hash": derive_command["command_hash"],
            "command_plan_hash": derive_command["command_plan_hash"],
            "receipt_ref": deepcopy(dict(derive_receipt_ref)),
            "receipt_hash": str(derive_receipt_ref["digest"]),
            "output_ref": deepcopy(output_ref),
            "completed_event_id": completed_event_id,
            "derivation_ref": deepcopy(validated.derivation_ref),
            "derivation_hash": validated.derivation_hash,
        }
        supplied_lineage = {
            key: entry[key]
            for key in _LINEAGE_FIELDS
            if key in entry
        }
        if supplied_lineage and supplied_lineage != derived_lineage:
            raise ValueError(f"caller-supplied evidence lineage is not canonical: {kind}")
        entry.update(derived_lineage)
        canonical_entries.append(entry)
        evidence_id = str(entry["evidence_id"])
        evidence_bytes[evidence_id] = normalized_raw
        lineage[evidence_id] = ReceiptEvidenceLineage(
            evidence_id=evidence_id,
            evidence_kind=kind,
            command_id=str(derive_command["command_id"]),
            command_hash=str(derive_command["command_hash"]),
            completed_event_id=completed_event_id,
            receipt_ref=deepcopy(dict(derive_receipt_ref)),
            receipt_hash=str(derive_receipt_ref["digest"]),
            output_ref=deepcopy(output_ref),
            derivation_ref=deepcopy(validated.derivation_ref),
            derivation_hash=validated.derivation_hash,
        )

    canonical_manifest = {
        key: deepcopy(value)
        for key, value in manifest.items()
        if key != "entries"
    }
    canonical_manifest["schema_version"] = EVIDENCE_MANIFEST_SCHEMA_VERSION
    canonical_manifest["phase"] = phase
    canonical_manifest["phase_execution_id"] = attempt["phase_executions"][phase]["phase_execution_id"]
    canonical_manifest["producer_run_id"] = attempt["phase_executions"][phase]["producer_run_id"]
    canonical_manifest["derivation_ref"] = deepcopy(validated.derivation_ref)
    canonical_manifest["derivation_hash"] = validated.derivation_hash
    canonical_manifest["derive_receipt_ref"] = deepcopy(dict(derive_receipt_ref))
    canonical_manifest["derive_receipt_hash"] = str(derive_receipt_ref["digest"])
    canonical_manifest["entries"] = canonical_entries
    validate_schema(canonical_manifest, _manifest_schema_file())
    observations, decoded = decode_evidence_inventory(
        attempt=attempt,
        trial_spec=trial_spec,
        manifest=canonical_manifest,
        evidence_bytes=evidence_bytes,
    )
    if set(decoded) != set(lineage):
        raise ValueError("decoded evidence and receipt lineage exact-set mismatch")
    return ReceiptBoundEvidence(
        manifest=deepcopy(canonical_manifest),
        observations=tuple(deepcopy(observations)),
        decoded_evidence=deepcopy(decoded),
        evidence_bytes=dict(evidence_bytes),
        lineage=lineage,
        receipt_bound_sources=validated.receipt_bound_sources,
    )


def _expected_phase_kinds(trial_spec: Mapping[str, Any], phase: str) -> list[str]:
    contracts = [item for item in trial_spec.get("phase_contracts", []) if item.get("phase") == phase]
    if len(contracts) != 1:
        raise ValueError(f"frozen TrialSpec must contain exactly one {phase} phase contract")
    outputs = contracts[0]["derivation_plan"]["expected_normalized_outputs"]
    kinds = [str(item["kind"]) for item in outputs]
    if len(kinds) != len(set(kinds)):
        raise ValueError("frozen derivation plan contains duplicate normalized evidence kind")
    if set(kinds) != set(contracts[0]["evidence_kinds"]):
        raise ValueError("frozen derivation and phase evidence exact-sets disagree")
    return kinds


def _manifest_schema_file() -> str:
    try:
        suffix = EVIDENCE_MANIFEST_SCHEMA_VERSION.rsplit("_v", 1)[1]
    except (IndexError, AttributeError) as error:
        raise ValueError("unsupported EvidenceManifest schema version") from error
    return f"evidence_manifest_v{suffix}.schema.json"


__all__ = [
    "ReceiptBoundEvidence",
    "ReceiptEvidenceLineage",
    "manifest_from_completion_evidence",
    "validate_receipt_bound_evidence",
]
