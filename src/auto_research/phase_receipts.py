"""Shared pure validation for authoritative phase command receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest
from .phase_command_plan import validate_phase_command_plan


PHASE_RUN_RECEIPT_SCHEMA_FILE = "phase_run_receipt_v4.schema.json"
PHASE_COMMAND_PLAN_SCHEMA_FILE = "phase_command_plan_v2.schema.json"
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
    environment = (command.get("command_spec") or {}).get("environment") or {}
    encoded = environment.get(C2C_RAW_OUTPUT_SPECS_ENV)
    if encoded is None:
        if raw_outputs:
            raise ValueError("PhaseRunReceipt contains unregistered raw outputs")
        return
    try:
        specs = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("frozen C2C raw output specs are malformed") from error
    if not isinstance(specs, list) or any(not isinstance(item, dict) for item in specs):
        raise ValueError("frozen C2C raw output specs must be an object array")
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
