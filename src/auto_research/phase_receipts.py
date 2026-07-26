"""Shared pure validation for authoritative phase command receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest
from .phase_command_plan import validate_phase_command_plan


PHASE_RUN_RECEIPT_SCHEMA_FILE = "phase_run_receipt_v5.schema.json"
PHASE_COMMAND_PLAN_SCHEMA_FILE = "phase_command_plan_v3.schema.json"
EVIDENCE_DERIVATION_PLAN_SCHEMA_FILE = "evidence_derivation_plan_v1.schema.json"
EVIDENCE_DERIVATION_MANIFEST_SCHEMA_FILE = "evidence_derivation_manifest_v2.schema.json"
C2C_RAW_OUTPUT_SPECS_ENV = "AUTO_RESEARCH_C2C_RAW_OUTPUT_SPECS"


def validate_phase_run_receipt(
    project_root: Path,
    record: Mapping[str, Any],
    receipt_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable receipt against its Started event and frozen plan."""

    store = ContractStore(project_root)
    receipt_blob = store.verify(receipt_ref)
    receipt = store.read_json(receipt_ref, schema_file=PHASE_RUN_RECEIPT_SCHEMA_FILE)
    if receipt_blob["digest"] != receipt_ref.get("digest"):
        raise ValueError("PhaseRunReceipt locator hash mismatch")
    command = record.get("command") if isinstance(record.get("command"), Mapping) else record
    if not isinstance(command, Mapping):
        raise ValueError("PhaseRunReceipt command identity is missing")
    started_event_id = record.get("started_event_id") or record.get("event_id")
    started_event_hash = record.get("started_event_hash") or record.get("event_hash")
    started_at = record.get("created_at")
    expected = {
        "command_id": command.get("command_id"),
        "command_hash": command.get("command_hash"),
        "command_spec_id": command.get("command_spec_id"),
        "command_plan_hash": command.get("command_plan_hash"),
        "started_event_id": started_event_id,
        "started_event_hash": started_event_hash,
        "attempt_id": command.get("attempt_id"),
        "lifecycle_generation": command.get("lifecycle_generation"),
        "phase": command.get("phase"),
        "phase_execution_id": command.get("phase_execution_id"),
        "phase_start_event_id": command.get("phase_start_event_id"),
        "producer_run_id": command.get("producer_run_id"),
        "implementation_hash": command.get("implementation_hash"),
        "attempt_input_hash": command.get("attempt_input_hash"),
        "provenance_mode": command.get("provenance_mode"),
        "started_at": started_at,
    }
    for field_name, value in expected.items():
        if receipt.get(field_name) != value:
            raise ValueError(f"PhaseRunReceipt {field_name} mismatch")

    plan_ref = command.get("command_plan_ref")
    if not isinstance(plan_ref, Mapping) or plan_ref.get("digest") != command.get("command_plan_hash"):
        raise ValueError("PhaseRunReceipt command plan reference mismatch")
    plan = store.read_json(plan_ref, schema_file=PHASE_COMMAND_PLAN_SCHEMA_FILE)
    validate_phase_command_plan(plan)
    matches = [item for item in plan["commands"] if item["command_spec_id"] == command.get("command_spec_id")]
    if len(matches) != 1 or canonical_contract_bytes(matches[0]) != canonical_contract_bytes(command.get("command_spec")):
        raise ValueError("PhaseRunReceipt command spec differs from frozen plan")
    if contract_digest(canonical_contract_bytes(matches[0])) != command.get("command_hash"):
        raise ValueError("PhaseRunReceipt command hash differs from frozen plan")

    _validate_derivation_binding(store, plan, command, receipt)

    for prefix in ("stdout", "stderr"):
        reference = receipt.get(f"{prefix}_ref")
        if not isinstance(reference, Mapping):
            raise ValueError(f"PhaseRunReceipt {prefix} reference is missing")
        blob = store.verify(reference)
        if blob["digest"] != receipt.get(f"{prefix}_hash"):
            raise ValueError(f"PhaseRunReceipt {prefix} hash mismatch")

    expected_outputs = list(command["command_spec"]["expected_outputs"])
    outputs = list(receipt["outputs"])
    if receipt["exit_code"] != 0 and not outputs:
        _validate_raw_outputs(store, command, receipt)
        return receipt
    if len(outputs) != len(expected_outputs):
        raise ValueError("PhaseRunReceipt output count differs from frozen command plan")
    for expected_output, output in zip(expected_outputs, outputs, strict=True):
        if output.get("output_id") != expected_output.get("output_id"):
            raise ValueError("PhaseRunReceipt output_id differs from frozen command plan")
        if output.get("kind") != expected_output.get("kind") or output.get("schema_version") != expected_output.get("schema_version"):
            raise ValueError("PhaseRunReceipt output differs from frozen command plan")
        if output.get("producer_run_id") != command.get("producer_run_id"):
            raise ValueError("PhaseRunReceipt output producer mismatch")
        if output.get("phase") != command.get("phase") or output.get("lifecycle_generation") != command.get("lifecycle_generation"):
            raise ValueError("PhaseRunReceipt output execution identity mismatch")
        reference = output.get("contract_ref")
        if not isinstance(reference, Mapping):
            raise ValueError("PhaseRunReceipt output reference is missing")
        blob = store.verify(reference)
        if blob["digest"] != output.get("content_hash"):
            raise ValueError("PhaseRunReceipt output content hash mismatch")
    _validate_raw_outputs(store, command, receipt)
    return receipt


def _validate_derivation_binding(
    store: ContractStore,
    phase_plan: Mapping[str, Any],
    command: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    command_spec = command.get("command_spec")
    if not isinstance(command_spec, Mapping):
        raise ValueError("PhaseRunReceipt frozen command spec is missing")
    is_derive = command_spec.get("authority_role") == "derivation"
    plan_ref = phase_plan["derivation_plan_ref"]
    plan_hash = phase_plan["derivation_plan_hash"]
    derivation_ref = receipt.get("derivation_ref")
    derivation_hash = receipt.get("derivation_hash")
    if not is_derive:
        if command_spec.get("derivation_plan_ref") is not None or command_spec.get("derivation_plan_hash") is not None:
            raise ValueError("physical command cannot carry derivation plan authority")
        if derivation_ref is not None or derivation_hash is not None:
            raise ValueError("non-derive PhaseRunReceipt cannot publish a derivation manifest")
        return
    if receipt.get("exit_code") != 0:
        if derivation_ref is not None or derivation_hash is not None:
            raise ValueError("failed derive PhaseRunReceipt cannot publish a derivation manifest")
        return
    if command_spec.get("derivation_plan_ref") != plan_ref or command_spec.get("derivation_plan_hash") != plan_hash:
        raise ValueError("derive command spec differs from frozen phase derivation plan")
    if not isinstance(plan_ref, Mapping) or not isinstance(plan_hash, str):
        raise ValueError("frozen derivation plan identity is malformed")
    plan_blob = store.verify(plan_ref)
    if plan_blob["digest"] != plan_hash:
        raise ValueError("frozen derivation plan hash mismatch")
    store.read_json(plan_ref, schema_file=EVIDENCE_DERIVATION_PLAN_SCHEMA_FILE)

    if not isinstance(derivation_ref, Mapping) or not isinstance(derivation_hash, str):
        raise ValueError("derive PhaseRunReceipt must publish its structured derivation reference")
    manifest_blob = store.verify(derivation_ref)
    if manifest_blob["digest"] != derivation_hash:
        raise ValueError("PhaseRunReceipt derivation hash mismatch")
    manifest = store.read_json(
        derivation_ref,
        schema_file=EVIDENCE_DERIVATION_MANIFEST_SCHEMA_FILE,
    )
    if manifest.get("derivation_plan_ref") != plan_ref or manifest.get("derivation_plan_hash") != plan_hash:
        raise ValueError("derivation manifest differs from the frozen derivation plan")
    expected_identity = {
        "attempt_id": command.get("attempt_id"),
        "lifecycle_generation": command.get("lifecycle_generation"),
        "phase": command.get("phase"),
        "phase_execution_id": command.get("phase_execution_id"),
        "producer_run_id": command.get("producer_run_id"),
        "implementation_hash": command.get("implementation_hash"),
        "attempt_input_hash": command.get("attempt_input_hash"),
    }
    for field_name, value in expected_identity.items():
        if manifest.get(field_name) != value:
            raise ValueError(f"derivation manifest {field_name} mismatch")
    if receipt.get("raw_outputs"):
        raise ValueError("derive PhaseRunReceipt cannot relabel physical raw outputs")
    normalized_outputs = manifest.get("normalized_outputs")
    receipt_outputs = receipt.get("outputs")
    if not isinstance(normalized_outputs, list) or not isinstance(receipt_outputs, list):
        raise ValueError("derive PhaseRunReceipt normalized outputs are missing")
    canonical_manifest_outputs = [
        {
            "output_id": item.get("output_id"),
            "kind": item.get("kind"),
            "schema_version": item.get("schema_version"),
            "content_hash": item.get("content_hash"),
            "contract_ref": item.get("contract_ref"),
        }
        for item in normalized_outputs
    ]
    canonical_receipt_outputs = [
        {
            "output_id": item.get("output_id"),
            "kind": item.get("kind"),
            "schema_version": item.get("schema_version"),
            "content_hash": item.get("content_hash"),
            "contract_ref": item.get("contract_ref"),
        }
        for item in receipt_outputs
    ]
    if canonical_receipt_outputs != canonical_manifest_outputs:
        raise ValueError("derive PhaseRunReceipt outputs differ from its derivation manifest")


def _validate_raw_outputs(
    store: ContractStore,
    command: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    raw_outputs = receipt.get("raw_outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError("PhaseRunReceipt raw_outputs are missing")
    if receipt.get("exit_code") != 0:
        if raw_outputs:
            raise ValueError("failed PhaseRunReceipt cannot publish raw scientific outputs")
        return
    specs = (command.get("command_spec") or {}).get("physical_raw_outputs")
    if not specs:
        if raw_outputs:
            raise ValueError("PhaseRunReceipt contains unregistered raw outputs")
        return
    if not isinstance(specs, list) or any(not isinstance(item, dict) for item in specs):
        raise ValueError("frozen physical raw output specs must be an object array")
    required = [item for item in specs if item.get("required", True)]
    expected_ids = [str(item.get("output_id") or "") for item in required]
    actual_ids = [str(item.get("output_id") or "") for item in raw_outputs]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("PhaseRunReceipt raw output set differs from the frozen command")
    by_id = {str(item["output_id"]): item for item in specs}
    for output in raw_outputs:
        spec = by_id[output["output_id"]]
        expected = {
            "kind": spec.get("kind"),
            "schema_version": spec.get("schema_version"),
            "locator": spec.get("locator"),
            "locator_type": spec.get("locator_type"),
            "dataset_id": spec.get("dataset_id"),
            "role": spec.get("role"),
            "command_spec_id": command.get("command_spec_id"),
            "producer_run_id": command.get("producer_run_id"),
            "phase": command.get("phase"),
            "lifecycle_generation": command.get("lifecycle_generation"),
        }
        for field_name, value in expected.items():
            if output.get(field_name) != value:
                raise ValueError(f"PhaseRunReceipt raw output {field_name} mismatch")
        reference = output.get("contract_ref")
        if not isinstance(reference, Mapping):
            raise ValueError("PhaseRunReceipt raw output ContractRef is missing")
        blob = store.verify(reference)
        if blob["digest"] != output.get("content_hash"):
            raise ValueError("PhaseRunReceipt raw output content hash mismatch")


__all__ = ["C2C_RAW_OUTPUT_SPECS_ENV", "validate_phase_run_receipt"]
