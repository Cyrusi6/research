"""Pure receipt-to-evidence lineage validation.

This module does not mutate the ledger or projections. It derives the only
canonical EvidenceManifest from staged evidence facts plus completed command
receipts, then decodes the exact receipt-linked immutable bytes.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest, validate_schema
from .evidence import EvidenceStore, decode_evidence_inventory

_RECEIPT_SCHEMA_FILE = "phase_run_receipt_v3.schema.json"
_LINEAGE_FIELDS = {
    "command_id",
    "command_hash",
    "command_plan_hash",
    "receipt_ref",
    "receipt_hash",
    "output_ref",
    "completed_event_id",
    "derivation",
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


@dataclass(frozen=True)
class ReceiptBoundEvidence:
    manifest: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    decoded_evidence: dict[str, dict[str, Any]]
    evidence_bytes: dict[str, bytes]
    lineage: dict[str, ReceiptEvidenceLineage]


def manifest_from_completion_evidence(
    *,
    attempt: Mapping[str, Any],
    completion_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the staged manifest facts that receipt validation canonicalizes."""

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
        "schema_version": "auto_research_evidence_manifest_v4",
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
    """Derive and validate canonical receipt-bound evidence.

    ``manifest`` may be the staged inventory produced by CompletionEvidence v3
    or an already canonical EvidenceManifest v4. Caller-supplied lineage is
    never trusted: it must exactly equal the lineage derived from completed
    PhaseCommand receipts.
    """

    if phase not in {"proxy", "full"}:
        raise ValueError("receipt evidence phase must be proxy or full")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("evidence manifest entries are required")
    phase_entries = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("phase") == phase]
    if len(phase_entries) != len(entries):
        raise ValueError("evidence manifest mixes phases or contains invalid entries")

    expected_kinds = _expected_phase_kinds(trial_spec, phase)
    manifest_kinds = [str(entry.get("kind")) for entry in phase_entries]
    if len(manifest_kinds) != len(set(manifest_kinds)):
        raise ValueError("duplicate evidence kind in authoritative manifest")
    if set(manifest_kinds) != expected_kinds:
        missing = sorted(expected_kinds - set(manifest_kinds))
        extra = sorted(set(manifest_kinds) - expected_kinds)
        raise ValueError(f"receipt evidence exact-set mismatch; missing={missing}, extra={extra}")

    command_outputs = _completed_command_outputs(
        project_root=project_root,
        attempt=attempt,
        phase_commands=phase_commands,
        phase=phase,
        manifest=manifest,
    )
    if set(command_outputs) != expected_kinds:
        missing = sorted(expected_kinds - set(command_outputs))
        extra = sorted(set(command_outputs) - expected_kinds)
        raise ValueError(f"completed command output exact-set mismatch; missing={missing}, extra={extra}")

    evidence_bytes: dict[str, bytes] = {}
    lineage: dict[str, ReceiptEvidenceLineage] = {}
    canonical_entries: list[dict[str, Any]] = []
    evidence_store = EvidenceStore(project_root)
    for supplied_entry in phase_entries:
        entry = deepcopy(dict(supplied_entry))
        kind = str(entry["kind"])
        output = command_outputs[kind]
        output_ref = deepcopy(output["output_ref"])
        raw = output["raw"]
        if output_ref["digest"] != entry["content_hash"] or contract_digest(raw) != entry["content_hash"]:
            raise ValueError(f"receipt output hash does not match evidence entry: {kind}")
        staged_raw = evidence_store.read_entry(entry, attempt)
        if staged_raw != raw:
            raise ValueError(f"receipt output bytes differ from staged evidence bytes: {kind}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"receipt output is not canonical JSON: {kind}") from exc
        entry["cross_references"] = deepcopy(payload.get("cross_references") or {})

        derived = {
            "command_id": output["command_id"],
            "command_hash": output["command_hash"],
            "command_plan_hash": output["command_plan_hash"],
            "receipt_ref": deepcopy(output["receipt_ref"]),
            "receipt_hash": output["receipt_hash"],
            "output_ref": output_ref,
            "completed_event_id": output["completed_event_id"],
        }
        supplied_lineage = {key: entry[key] for key in _LINEAGE_FIELDS if key in entry}
        if supplied_lineage:
            expected_lineage = dict(derived)
            if "derivation" in supplied_lineage:
                expected_lineage["derivation"] = None
            if supplied_lineage != expected_lineage:
                raise ValueError(f"caller-supplied evidence lineage is not canonical: {kind}")
        entry.update(derived)
        entry.pop("derivation", None)
        canonical_entries.append(entry)

        evidence_id = str(entry["evidence_id"])
        evidence_bytes[evidence_id] = raw
        lineage[evidence_id] = ReceiptEvidenceLineage(
            evidence_id=evidence_id,
            evidence_kind=kind,
            command_id=output["command_id"],
            command_hash=output["command_hash"],
            completed_event_id=output["completed_event_id"],
            receipt_ref=deepcopy(output["receipt_ref"]),
            receipt_hash=output["receipt_hash"],
            output_ref=output_ref,
        )

    canonical_manifest = {key: deepcopy(value) for key, value in manifest.items() if key != "entries"}
    canonical_manifest["schema_version"] = "auto_research_evidence_manifest_v4"
    canonical_manifest["entries"] = canonical_entries
    validate_schema(canonical_manifest, "evidence_manifest_v4.schema.json")
    observations, decoded = decode_evidence_inventory(
        attempt=attempt,
        trial_spec=trial_spec,
        manifest=canonical_manifest,
        evidence_bytes=evidence_bytes,
    )
    _validate_receipt_derived_status(decoded, lineage, command_outputs)
    return ReceiptBoundEvidence(
        manifest=deepcopy(canonical_manifest),
        observations=tuple(deepcopy(observations)),
        decoded_evidence=deepcopy(decoded),
        evidence_bytes=dict(evidence_bytes),
        lineage=lineage,
    )


def _expected_phase_kinds(trial_spec: Mapping[str, Any], phase: str) -> set[str]:
    contracts = trial_spec.get("phase_contracts")
    if not isinstance(contracts, Sequence) or isinstance(contracts, (str, bytes, bytearray)):
        raise ValueError("frozen TrialSpec phase contracts are missing")
    matches = [item for item in contracts if isinstance(item, Mapping) and item.get("phase") == phase]
    if len(matches) != 1:
        raise ValueError(f"frozen TrialSpec must contain exactly one {phase} phase contract")
    kinds = matches[0].get("evidence_kinds")
    if not isinstance(kinds, Sequence) or isinstance(kinds, (str, bytes, bytearray)) or not kinds:
        raise ValueError(f"{phase} phase evidence kinds are missing")
    normalized = [str(kind) for kind in kinds]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{phase} phase contract contains duplicate evidence kinds")
    return set(normalized)


def _completed_command_outputs(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(phase_commands, Mapping):
        raise ValueError("authoritative phase command state is required for evidence lineage")
    execution_ids = {str(entry["phase_execution_id"]) for entry in manifest["entries"]}
    producer_ids = {str(entry["producer_run_id"]) for entry in manifest["entries"]}
    phase_start_ids = {str(entry["phase_start_event_id"]) for entry in manifest["entries"]}
    if len(execution_ids) != 1 or len(producer_ids) != 1 or len(phase_start_ids) != 1:
        raise ValueError("evidence inventory mixes phase execution or producer identity")
    execution_id = next(iter(execution_ids))
    producer_id = next(iter(producer_ids))
    phase_start_id = next(iter(phase_start_ids))

    relevant: list[Mapping[str, Any]] = []
    for record in phase_commands.values():
        if not isinstance(record, Mapping):
            continue
        command = record.get("command")
        if not isinstance(command, Mapping):
            continue
        if (
            command.get("attempt_id") == attempt.get("attempt_id")
            and command.get("lifecycle_generation") == attempt.get("lifecycle_generation")
            and command.get("implementation_hash") == attempt.get("implementation_hash")
            and command.get("attempt_input_hash") == attempt.get("attempt_input_hash")
            and command.get("phase") == phase
            and command.get("phase_execution_id") == execution_id
            and command.get("phase_start_event_id") == phase_start_id
            and command.get("producer_run_id") == producer_id
        ):
            relevant.append(record)
    if not relevant:
        raise ValueError("evidence has no completed command receipt lineage")

    outputs_by_kind: dict[str, dict[str, Any]] = {}
    store = ContractStore(project_root)
    decoded_records: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for record in relevant:
        if record.get("status") != "completed":
            raise ValueError(f"phase command is not completed: {record.get('status')}")
        completed_event_id = record.get("completed_event_id")
        receipt_ref = record.get("receipt_ref")
        command = record["command"]
        if not isinstance(completed_event_id, str) or not completed_event_id or not isinstance(receipt_ref, Mapping):
            raise ValueError("completed phase command lacks receipt lineage")
        receipt = store.read_json(receipt_ref, schema_file=_RECEIPT_SCHEMA_FILE)
        _validate_receipt_identity(receipt, record, attempt, phase)
        decoded_records.append((record, receipt))
    successful_specs = {
        str(record["command"]["command_spec_id"])
        for record, receipt in decoded_records
        if receipt.get("exit_code") == 0
    }
    for record, receipt in decoded_records:
        command = record["command"]
        command_spec = command.get("command_spec") or {}
        if receipt.get("exit_code") != 0:
            condition = (command_spec.get("condition") or {}).get("kind")
            superseded = any(
                condition_record["command"].get("command_spec", {}).get("condition", {}).get("kind") != "always"
                and str(command_spec.get("command_spec_id"))
                in (condition_record["command"].get("command_spec", {}).get("dependencies") or [])
                and str(condition_record["command"].get("command_spec_id")) in successful_specs
                for condition_record, _ in decoded_records
            )
            if condition == "always" and not superseded:
                raise ValueError("failed command receipt cannot authorize scientific evidence")
            if receipt.get("outputs"):
                raise ValueError("failed command receipt cannot carry scientific outputs")
            continue
        expected_outputs = command.get("command_spec", {}).get("expected_outputs")
        receipt_outputs = receipt.get("outputs")
        if not isinstance(expected_outputs, list) or not isinstance(receipt_outputs, list) or len(expected_outputs) != len(receipt_outputs):
            raise ValueError("command receipt outputs differ from frozen expected outputs")
        receipt_hash = _blob_reference(receipt_ref)["digest"]
        for expected, output in zip(expected_outputs, receipt_outputs, strict=True):
            if not isinstance(output, Mapping) or output.get("kind") != expected.get("kind"):
                raise ValueError("command receipt output order/kind differs from frozen command spec")
            kind = str(output["kind"])
            if kind in outputs_by_kind:
                raise ValueError(f"duplicate completed command output kind: {kind}")
            if output.get("schema_version") != expected.get("schema_version"):
                raise ValueError(f"command receipt output schema differs from frozen command spec: {kind}")
            output_ref = output.get("contract_ref")
            if not isinstance(output_ref, Mapping):
                raise ValueError("command receipt output lacks immutable contract reference")
            raw = store.read_bytes(output_ref)
            if output.get("content_hash") != output_ref.get("digest") or contract_digest(raw) != output_ref.get("digest"):
                raise ValueError(f"command receipt output content hash mismatch: {kind}")
            if output.get("producer_run_id") != command["producer_run_id"]:
                raise ValueError(f"command receipt output producer mismatch: {kind}")
            if output.get("phase") != phase or output.get("lifecycle_generation") != attempt["lifecycle_generation"]:
                raise ValueError(f"command receipt output execution identity mismatch: {kind}")
            outputs_by_kind[kind] = {
                "command_id": command["command_id"],
                "command_hash": command["command_hash"],
                "command_plan_hash": command["command_plan_hash"],
                "completed_event_id": completed_event_id,
                "receipt_ref": deepcopy(dict(receipt_ref)),
                "receipt_hash": receipt_hash,
                "receipt": receipt,
                "output_ref": deepcopy(dict(output_ref)),
                "raw": raw,
            }
    return outputs_by_kind


def _blob_reference(reference: Mapping[str, Any]) -> Mapping[str, Any]:
    blob = reference.get("blob")
    return blob if isinstance(blob, Mapping) else reference


def _validate_receipt_identity(
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    attempt: Mapping[str, Any],
    phase: str,
) -> None:
    command = record["command"]
    expected = {
        "command_id": command.get("command_id"),
        "command_hash": command.get("command_hash"),
        "command_spec_id": command.get("command_spec_id"),
        "command_plan_hash": command.get("command_plan_hash"),
        "started_event_id": record.get("started_event_id"),
        "started_event_hash": record.get("started_event_hash"),
        "attempt_id": attempt.get("attempt_id"),
        "lifecycle_generation": attempt.get("lifecycle_generation"),
        "phase": phase,
        "phase_execution_id": command.get("phase_execution_id"),
        "phase_start_event_id": command.get("phase_start_event_id"),
        "producer_run_id": command.get("producer_run_id"),
        "implementation_hash": attempt.get("implementation_hash"),
        "attempt_input_hash": attempt.get("attempt_input_hash"),
        "provenance_mode": command.get("provenance_mode"),
        "started_at": record.get("created_at"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"command receipt {key} mismatch")


def _validate_receipt_derived_status(
    decoded: Mapping[str, Mapping[str, Any]],
    lineage: Mapping[str, ReceiptEvidenceLineage],
    command_outputs: Mapping[str, Mapping[str, Any]],
) -> None:
    for evidence_id, payload in decoded.items():
        item = lineage[evidence_id]
        receipt = command_outputs[item.evidence_kind]["receipt"]
        completed = receipt["exit_code"] == 0
        if "command_status" in payload and payload["command_status"] != ("completed" if completed else "failed"):
            raise ValueError(f"producer command_status contradicts authoritative receipt: {evidence_id}")
        if "exit_code" in payload and payload["exit_code"] != receipt["exit_code"]:
            raise ValueError(f"producer exit_code contradicts authoritative receipt: {evidence_id}")
        if payload.get("evidence_kind") == "activation_evidence":
            derived_status = "passed" if completed else "failed"
            if payload.get("status") != derived_status:
                raise ValueError("producer activation status contradicts authoritative receipt")
        if payload.get("evidence_kind") == "full_s3_readiness" and payload.get("ready") is True and not completed:
            raise ValueError("producer readiness cannot override failed authoritative command")


__all__ = [
    "ReceiptBoundEvidence",
    "ReceiptEvidenceLineage",
    "manifest_from_completion_evidence",
    "validate_receipt_bound_evidence",
]
