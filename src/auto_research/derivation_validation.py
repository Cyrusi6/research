"""Deterministic physical-receipt to canonical-evidence derivation.

The functions in this module are shared by the constrained derive executor and
all authority readers.  They never mutate the contract store, evidence store,
ledger, or projections.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest, validate_schema
from .derivation_contracts import (
    derivation_plan_for_phase,
    readiness_check_plan_for_phase,
    validate_evidence_derivation_plan,
)
from .evidence import EVIDENCE_SCHEMA_VERSIONS
from .phase_receipts import validate_phase_run_receipt


DERIVATION_PLAN_SCHEMA_FILE = "evidence_derivation_plan_v1.schema.json"
DERIVATION_MANIFEST_SCHEMA_FILE = "evidence_derivation_manifest_v2.schema.json"


@dataclass(frozen=True)
class PhysicalRawOutput:
    source_ordinal: int
    output_id: str
    kind: str
    schema_version: str
    role: str | None
    dataset_id: str | None
    normalized_kinds: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: tuple[str, ...]
    contract_ref: dict[str, Any]
    content_hash: str
    raw_bytes: bytes


@dataclass(frozen=True)
class PhysicalReceiptInput:
    ordinal: int
    phase: str
    command_id: str
    command_spec_id: str
    command_hash: str
    command_plan_hash: str
    completed_event_id: str
    completed_event_hash: str
    receipt_ref: dict[str, Any]
    receipt_hash: str
    receipt: dict[str, Any]
    raw_outputs: tuple[PhysicalRawOutput, ...]


@dataclass(frozen=True)
class DerivedEvidence:
    output_id: str
    kind: str
    schema_version: str
    raw_bytes: bytes

    @property
    def content_hash(self) -> str:
        return contract_digest(self.raw_bytes)


@dataclass(frozen=True)
class ReceiptBoundSources:
    raw_facts: dict[tuple[str, str, str], dict[str, Any]]
    raw_fact_lineage: dict[tuple[str, str, str], dict[str, Any]]
    surface_checks: tuple[dict[str, Any], ...]
    physical_inputs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DeterministicDerivation:
    normalized_outputs: tuple[DerivedEvidence, ...]
    manifest_facts: dict[str, Any]
    receipt_bound_sources: ReceiptBoundSources


@dataclass(frozen=True)
class ValidatedEvidenceDerivation:
    derivation_ref: dict[str, Any]
    derivation_hash: str
    derivation_manifest: dict[str, Any]
    derive_record: dict[str, Any]
    derive_receipt: dict[str, Any]
    normalized_outputs: tuple[DerivedEvidence, ...]
    output_refs: dict[str, dict[str, Any]]
    output_bytes: dict[str, bytes]
    receipt_bound_sources: ReceiptBoundSources


Decoder = Callable[
    [
        Mapping[str, Any],
        Mapping[str, Any],
        str,
        Mapping[str, Any],
        Sequence[PhysicalReceiptInput],
        ReceiptBoundSources,
    ],
    tuple[DerivedEvidence, ...],
]


def derive_evidence_deterministically(
    *,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase: str,
    derivation_plan: Mapping[str, Any],
    physical_inputs: Sequence[PhysicalReceiptInput],
    decoder_implementation_bytes: bytes,
) -> DeterministicDerivation:
    """Run the single constrained pure decoder over validated physical bytes."""

    validate_evidence_derivation_plan(derivation_plan)
    _validate_derivation_identity(attempt, trial_spec, phase, derivation_plan)
    decoder = derivation_plan["decoder_descriptor"]
    _validate_decoder_implementation(
        derivation_plan,
        decoder_implementation_bytes,
    )
    decoder_key = (str(decoder["decoder_id"]), str(decoder["decoder_version"]))
    implementation = _DECODER_REGISTRY.get(decoder_key)
    if implementation is None:
        raise ValueError(f"unsupported frozen evidence decoder: {decoder_key[0]}@{decoder_key[1]}")
    sources = _receipt_bound_sources(physical_inputs)
    outputs = implementation(
        attempt,
        trial_spec,
        phase,
        derivation_plan,
        physical_inputs,
        sources,
    )
    expected_outputs = list(derivation_plan["expected_normalized_outputs"])
    actual_identity = [(item.output_id, item.kind, item.schema_version) for item in outputs]
    expected_identity = [
        (str(item["output_id"]), str(item["kind"]), str(item["schema_version"]))
        for item in expected_outputs
    ]
    if actual_identity != expected_identity:
        raise ValueError("deterministic decoder output order differs from frozen derivation plan")
    for output in outputs:
        payload = _canonical_json_object(output.raw_bytes, label=f"normalized {output.kind}")
        expected_version = EVIDENCE_SCHEMA_VERSIONS.get(output.kind)
        if expected_version is None or output.schema_version != expected_version:
            raise ValueError(f"unsupported normalized evidence kind/version: {output.kind}")
        if payload.get("schema_version") != output.schema_version or payload.get("evidence_kind") != output.kind:
            raise ValueError(f"deterministic normalized evidence identity mismatch: {output.kind}")
        version_suffix = output.schema_version.rsplit("_v", 1)[-1]
        validate_schema(payload, f"{output.kind}_v{version_suffix}.schema.json")

    derivation_identity_hash = contract_digest(canonical_contract_bytes({
        "plan_id": derivation_plan["plan_id"],
        "phase_execution_id": attempt["phase_executions"][phase]["phase_execution_id"],
    }))
    facts = {
        "derivation_id": f"derive:{derivation_identity_hash[:24]}",
        "attempt_id": str(attempt["attempt_id"]),
        "lifecycle_generation": attempt["lifecycle_generation"],
        "phase": phase,
        "phase_execution_id": str(attempt["phase_executions"][phase]["phase_execution_id"]),
        "producer_run_id": str(attempt["phase_executions"][phase]["producer_run_id"]),
        "implementation_hash": str(attempt["implementation_hash"]),
        "attempt_input_hash": str(attempt["attempt_input_hash"]),
        "trial_spec_hash": str(attempt["trial_spec_hash"]),
        "protocol_hash": str(attempt["protocol_hash"]),
        "sample_manifest_hash": str(attempt["sample_manifest_hash"]),
        "evaluator_hash": str(attempt["evaluator_hash"]),
        "source_commands": sorted(
            (
                _source_output_fact(item, output)
                for item in physical_inputs
                for output in item.raw_outputs
            ),
            key=lambda item: item["source_ordinal"],
        ),
        "decoder_descriptor": deepcopy(dict(decoder)),
    }
    return DeterministicDerivation(
        normalized_outputs=tuple(outputs),
        manifest_facts=facts,
        receipt_bound_sources=sources,
    )


def validate_immutable_derivation(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
    evidence_manifest: Mapping[str, Any],
) -> ValidatedEvidenceDerivation:
    """Read and byte-compare one already committed phase derivation.

    This function is deliberately read-only.  It obtains authority exclusively
    from the completed derive receipt and the frozen TrialSpec contract.
    """

    store = ContractStore(project_root)
    plan, plan_ref, plan_hash = _frozen_derivation_plan(store, trial_spec, phase)
    derive_record, derive_receipt = _completed_derive_receipt(
        project_root=project_root,
        attempt=attempt,
        phase_commands=phase_commands,
        phase=phase,
        plan_ref=plan_ref,
        plan_hash=plan_hash,
    )
    derivation_ref = derive_receipt.get("derivation_ref")
    derivation_hash = derive_receipt.get("derivation_hash")
    if not isinstance(derivation_ref, Mapping) or not isinstance(derivation_hash, str):
        raise ValueError("completed derive receipt lacks structured derivation authority")
    derivation_blob = store.verify(derivation_ref)
    if derivation_blob["digest"] != derivation_hash:
        raise ValueError("derive receipt derivation hash mismatch")
    derivation_manifest = store.read_json(
        derivation_ref,
        schema_file=DERIVATION_MANIFEST_SCHEMA_FILE,
    )
    if derivation_manifest.get("derivation_plan_ref") != plan_ref:
        raise ValueError("derivation manifest plan reference mismatch")
    if derivation_manifest.get("derivation_plan_hash") != plan_hash:
        raise ValueError("derivation manifest plan hash mismatch")

    physical_inputs = validated_physical_receipt_inputs(
        project_root=project_root,
        attempt=attempt,
        phase_commands=phase_commands,
        phase=phase,
        derivation_plan=plan,
    )
    deterministic = derive_evidence_deterministically(
        attempt=attempt,
        trial_spec=trial_spec,
        phase=phase,
        derivation_plan=plan,
        physical_inputs=physical_inputs,
        decoder_implementation_bytes=store.read_bytes(plan["decoder_descriptor"]["immutable_ref"]),
    )
    receipt_outputs = list(derive_receipt.get("outputs") or [])
    output_refs: dict[str, dict[str, Any]] = {}
    output_bytes: dict[str, bytes] = {}
    normalized_facts: list[dict[str, Any]] = []
    normalized_receipt_outputs = receipt_outputs
    if len(normalized_receipt_outputs) != len(deterministic.normalized_outputs):
        raise ValueError("derive receipt normalized output count mismatch")
    for ordinal, (expected, receipt_output) in enumerate(
        zip(deterministic.normalized_outputs, normalized_receipt_outputs, strict=True)
    ):
        reference = receipt_output.get("contract_ref")
        if not isinstance(reference, Mapping):
            raise ValueError("derive receipt normalized output lacks immutable reference")
        raw = store.read_bytes(reference)
        if raw != expected.raw_bytes:
            raise ValueError(f"derive receipt normalized bytes differ from deterministic decoder: {expected.kind}")
        if receipt_output.get("output_id") != expected.output_id:
            raise ValueError("derive receipt normalized output_id mismatch")
        if receipt_output.get("kind") != expected.kind or receipt_output.get("schema_version") != expected.schema_version:
            raise ValueError("derive receipt normalized output identity mismatch")
        if receipt_output.get("content_hash") != expected.content_hash or reference.get("digest") != expected.content_hash:
            raise ValueError("derive receipt normalized output hash mismatch")
        if expected.kind in output_refs:
            raise ValueError("derive receipt contains duplicate normalized evidence kind")
        output_refs[expected.kind] = deepcopy(dict(reference))
        output_bytes[expected.kind] = raw
        normalized_facts.append({
            "ordinal": ordinal,
            "output_id": expected.output_id,
            "kind": expected.kind,
            "schema_version": expected.schema_version,
            "contract_ref": deepcopy(dict(reference)),
            "content_hash": expected.content_hash,
        })
    expected_manifest = {
        "schema_version": derivation_manifest["schema_version"],
        **deepcopy(deterministic.manifest_facts),
        "derivation_plan_ref": deepcopy(dict(plan_ref)),
        "derivation_plan_hash": plan_hash,
        "normalized_outputs": normalized_facts,
    }
    if canonical_contract_bytes(derivation_manifest) != canonical_contract_bytes(expected_manifest):
        raise ValueError("EvidenceDerivationManifest differs from deterministic receipt-derived facts")

    entries = evidence_manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("EvidenceManifest entries are required for derivation validation")
    entry_kinds = [str(item.get("kind")) for item in entries if isinstance(item, Mapping)]
    if entry_kinds != [item.kind for item in deterministic.normalized_outputs]:
        raise ValueError("EvidenceManifest entry order differs from frozen normalized output order")
    for entry in entries:
        kind = str(entry["kind"])
        if entry.get("derivation_ref") not in (None, derivation_ref):
            raise ValueError("EvidenceManifest points to a different derivation manifest")
        if entry.get("derivation_hash") not in (None, derivation_hash):
            raise ValueError("EvidenceManifest derivation hash differs from derive receipt")
        if entry.get("content_hash") != output_refs[kind]["digest"]:
            raise ValueError("EvidenceManifest content hash differs from derive output")
        supplied_output_ref = entry.get("output_ref")
        if supplied_output_ref is not None and supplied_output_ref != output_refs[kind]:
            raise ValueError("EvidenceManifest output reference differs from derive receipt")
    return ValidatedEvidenceDerivation(
        derivation_ref=deepcopy(dict(derivation_ref)),
        derivation_hash=derivation_hash,
        derivation_manifest=deepcopy(derivation_manifest),
        derive_record=deepcopy(dict(derive_record)),
        derive_receipt=deepcopy(derive_receipt),
        normalized_outputs=deterministic.normalized_outputs,
        output_refs=output_refs,
        output_bytes=output_bytes,
        receipt_bound_sources=deterministic.receipt_bound_sources,
    )


def validated_physical_receipt_inputs(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
    derivation_plan: Mapping[str, Any],
) -> tuple[PhysicalReceiptInput, ...]:
    """Read the frozen ordered physical exact-set from completed receipts."""

    if not isinstance(phase_commands, Mapping):
        raise ValueError("authoritative phase command state is required")
    source_bindings = derivation_plan.get("source_bindings")
    if not isinstance(source_bindings, Sequence) or isinstance(source_bindings, (str, bytes, bytearray)) or not source_bindings:
        raise ValueError("EvidenceDerivationPlan source_bindings are required")
    bindings = [deepcopy(dict(item)) for item in source_bindings]
    existing_binding_ids = {
        (item["source_phase"], item["command_spec_id"], item["output_id"])
        for item in bindings
    }
    for cross_binding in derivation_plan.get("cross_phase_bindings") or []:
        identity = (
            cross_binding["source_phase"],
            cross_binding["command_spec_id"],
            cross_binding["output_id"],
        )
        if identity in existing_binding_ids:
            continue
        bindings.append({
            "source_ordinal": len(bindings),
            "source_phase": cross_binding["source_phase"],
            "command_spec_id": cross_binding["command_spec_id"],
            "output_id": cross_binding["output_id"],
            "output_kind": cross_binding["output_kind"],
            "output_schema_version": cross_binding["output_schema_version"],
            "normalized_kinds": list(cross_binding["normalized_kinds"]),
            "role": None,
            "dataset_id": None,
            "seeds": list(derivation_plan["coverage_contract"]["seeds"]),
            "metrics": list(derivation_plan["coverage_contract"]["metrics"]),
        })
        existing_binding_ids.add(identity)
    store = ContractStore(project_root)
    record_cache: dict[tuple[str, str], tuple[Mapping[str, Any], dict[str, Any]]] = {}
    grouped: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    used_outputs: set[tuple[str, str]] = set()
    for ordinal, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise ValueError("EvidenceDerivationPlan source binding is malformed")
        if binding.get("source_ordinal") != ordinal:
            raise ValueError("EvidenceDerivationPlan source binding order is not canonical")
        source_phase = str(binding.get("source_phase") or "")
        command_spec_id = str(binding.get("command_spec_id") or "")
        phase_execution = (attempt.get("phase_executions") or {}).get(source_phase)
        if not command_spec_id or not isinstance(phase_execution, Mapping):
            raise ValueError("EvidenceDerivationPlan source phase/command is not authoritative")
        cache_key = (source_phase, command_spec_id)
        cached = record_cache.get(cache_key)
        if cached is None:
            matches: list[Mapping[str, Any]] = []
            for record in phase_commands.values():
                if not isinstance(record, Mapping):
                    continue
                command = record.get("command")
                command_spec = command.get("command_spec") if isinstance(command, Mapping) else None
                if not isinstance(command, Mapping) or not isinstance(command_spec, Mapping):
                    continue
                if (
                    command.get("attempt_id") == attempt.get("attempt_id")
                    and command.get("lifecycle_generation") == phase_execution.get("lifecycle_generation")
                    and command.get("implementation_hash") == phase_execution.get("implementation_hash")
                    and command.get("attempt_input_hash") == phase_execution.get("attempt_input_hash")
                    and command.get("phase") == source_phase
                    and command.get("phase_execution_id") == phase_execution.get("phase_execution_id")
                    and command.get("phase_start_event_id") == phase_execution.get("phase_start_event_id")
                    and command.get("producer_run_id") == phase_execution.get("producer_run_id")
                    and command.get("command_spec_id") == command_spec_id
                    and command_spec.get("authority_role") == "physical"
                ):
                    matches.append(record)
            if len(matches) != 1:
                raise ValueError(f"physical derivation source must resolve once: {source_phase}/{command_spec_id}")
            record = matches[0]
            if record.get("status") != "completed":
                raise ValueError("physical derivation source command is not completed")
            receipt_ref = record.get("receipt_ref")
            if not isinstance(receipt_ref, Mapping):
                raise ValueError("physical derivation source lacks receipt reference")
            receipt = validate_phase_run_receipt(project_root, record, receipt_ref)
            if receipt.get("exit_code") != 0:
                raise ValueError("failed physical command cannot source normalized evidence")
            if receipt.get("derivation_ref") is not None or receipt.get("derivation_hash") is not None:
                raise ValueError("derive command cannot be a physical derivation source")
            record_cache[cache_key] = (record, receipt)
        else:
            record, receipt = cached
            receipt_ref = record["receipt_ref"]
        command = record["command"]
        command_id = str(command["command_id"])
        actual_outputs = receipt.get("raw_outputs")
        if not isinstance(actual_outputs, list):
            raise ValueError("physical derivation raw output set is missing")
        matches = [item for item in actual_outputs if item.get("output_id") == binding.get("output_id")]
        if len(matches) != 1:
            raise ValueError("physical derivation output_id must resolve exactly once in its receipt")
        actual = matches[0]
        expected_identity = {
            "kind": binding.get("output_kind"),
            "schema_version": binding.get("output_schema_version"),
        }
        if binding.get("role") is not None:
            expected_identity["role"] = binding["role"]
        if binding.get("dataset_id") is not None:
            expected_identity["dataset_id"] = binding["dataset_id"]
        for field_name, value in expected_identity.items():
            if actual.get(field_name) != value:
                raise ValueError(f"physical derivation raw output {field_name} mismatch")
        output_identity = (command_spec_id, str(actual["output_id"]))
        if output_identity in used_outputs:
            raise ValueError("physical derivation repeats a raw output binding")
        used_outputs.add(output_identity)
        reference = actual.get("contract_ref")
        if not isinstance(reference, Mapping):
            raise ValueError("physical derivation raw output lacks ContractRef")
        raw = store.read_bytes(reference)
        if actual.get("content_hash") != reference.get("digest") or contract_digest(raw) != reference.get("digest"):
            raise ValueError("physical derivation raw output hash mismatch")
        physical_output = PhysicalRawOutput(
            source_ordinal=ordinal,
            output_id=str(actual["output_id"]),
            kind=str(actual["kind"]),
            schema_version=str(actual["schema_version"]),
            role=str(actual["role"]) if actual.get("role") is not None else None,
            dataset_id=str(actual["dataset_id"]) if actual.get("dataset_id") is not None else None,
            normalized_kinds=tuple(str(item) for item in binding["normalized_kinds"]),
            seeds=tuple(int(item) for item in binding["seeds"]),
            metrics=tuple(str(item) for item in binding["metrics"]),
            contract_ref=deepcopy(dict(reference)),
            content_hash=str(actual["content_hash"]),
            raw_bytes=raw,
        )
        if command_id not in grouped:
            completed_event_id = record.get("completed_event_id")
            completed_event_hash = record.get("completed_event_hash")
            if not isinstance(completed_event_id, str) or not isinstance(completed_event_hash, str):
                raise ValueError("physical source lacks completed event identity")
            grouped[command_id] = {
                "ordinal": len(group_order),
                "phase": source_phase,
                "command_id": command_id,
                "command_spec_id": command_spec_id,
                "command_hash": str(command["command_hash"]),
                "command_plan_hash": str(command["command_plan_hash"]),
                "completed_event_id": completed_event_id,
                "completed_event_hash": completed_event_hash,
                "receipt_ref": deepcopy(dict(receipt_ref)),
                "receipt_hash": str(receipt_ref["digest"]),
                "receipt": deepcopy(receipt),
                "raw_outputs": [],
            }
            group_order.append(command_id)
        grouped[command_id]["raw_outputs"].append(physical_output)

    planned_outputs = {
        (str(item["source_phase"]), str(item["command_spec_id"]), str(item["output_id"]))
        for item in bindings
    }
    resolved_outputs = {
        (item.phase, item.command_spec_id, output.output_id)
        for item in (
            PhysicalReceiptInput(**grouped[command_id])
            for command_id in group_order
        )
        for output in item.raw_outputs
    }
    if resolved_outputs != planned_outputs:
        raise ValueError("physical raw output exact-set differs from declared derivation bindings")
    return tuple(
        PhysicalReceiptInput(**grouped[command_id])
        for command_id in group_order
    )


def _frozen_derivation_plan(
    store: ContractStore,
    trial_spec: Mapping[str, Any],
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    contracts = [item for item in trial_spec.get("phase_contracts", []) if item.get("phase") == phase]
    if len(contracts) != 1:
        raise ValueError(f"frozen TrialSpec must contain one {phase} phase contract")
    contract = contracts[0]
    plan = derivation_plan_for_phase(trial_spec, phase)
    plan_ref = contract.get("derivation_plan_ref")
    plan_hash = contract.get("derivation_plan_hash")
    if not isinstance(plan_ref, Mapping) or not isinstance(plan_hash, str):
        raise ValueError("frozen phase derivation plan identity is missing")
    stored = store.read_json(plan_ref, schema_file=DERIVATION_PLAN_SCHEMA_FILE)
    if plan_ref.get("digest") != plan_hash or contract_digest(canonical_contract_bytes(plan)) != plan_hash:
        raise ValueError("frozen phase derivation plan hash mismatch")
    if canonical_contract_bytes(stored) != canonical_contract_bytes(plan):
        raise ValueError("frozen phase derivation plan projection differs from immutable bytes")
    _validate_derivation_identity({}, trial_spec, phase, stored, allow_attempt_missing=True)
    return deepcopy(stored), deepcopy(dict(plan_ref)), plan_hash


def _completed_derive_receipt(
    *,
    project_root: Path,
    attempt: Mapping[str, Any],
    phase_commands: Mapping[str, Mapping[str, Any]],
    phase: str,
    plan_ref: Mapping[str, Any],
    plan_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_execution = (attempt.get("phase_executions") or {}).get(phase)
    if not isinstance(phase_execution, Mapping):
        raise ValueError("derive receipt phase execution is missing")
    matches: list[Mapping[str, Any]] = []
    for record in phase_commands.values():
        if not isinstance(record, Mapping):
            continue
        command = record.get("command")
        command_spec = command.get("command_spec") if isinstance(command, Mapping) else None
        if not isinstance(command, Mapping) or not isinstance(command_spec, Mapping):
            continue
        if (
            command.get("attempt_id") == attempt.get("attempt_id")
            and command.get("lifecycle_generation") == phase_execution.get("lifecycle_generation")
            and command.get("implementation_hash") == phase_execution.get("implementation_hash")
            and command.get("attempt_input_hash") == phase_execution.get("attempt_input_hash")
            and command.get("phase") == phase
            and command.get("phase_execution_id") == phase_execution.get("phase_execution_id")
            and command.get("phase_start_event_id") == phase_execution.get("phase_start_event_id")
            and command.get("producer_run_id") == phase_execution.get("producer_run_id")
            and command_spec.get("authority_role") == "derivation"
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError("phase must contain exactly one frozen derive command")
    record = matches[0]
    if record.get("status") != "completed" or not isinstance(record.get("receipt_ref"), Mapping):
        raise ValueError("frozen derive command is not completed")
    receipt = validate_phase_run_receipt(project_root, record, record["receipt_ref"])
    if receipt.get("exit_code") != 0:
        raise ValueError("failed derive command cannot authorize evidence")
    command_plan_ref = record["command"].get("command_plan_ref")
    if not isinstance(command_plan_ref, Mapping):
        raise ValueError("derive command lacks frozen PhaseCommandPlan reference")
    command_plan = ContractStore(project_root).read_json(
        command_plan_ref,
        schema_file="phase_command_plan_v3.schema.json",
    )
    if command_plan.get("derivation_plan_ref") != plan_ref:
        raise ValueError("derive command PhaseCommandPlan derivation reference mismatch")
    if command_plan.get("derivation_plan_hash") != plan_hash:
        raise ValueError("derive command PhaseCommandPlan derivation hash mismatch")
    return deepcopy(dict(record)), deepcopy(receipt)


def _validate_derivation_identity(
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase: str,
    plan: Mapping[str, Any],
    *,
    allow_attempt_missing: bool = False,
) -> None:
    if phase not in {"proxy", "full"} or plan.get("phase") != phase:
        raise ValueError("EvidenceDerivationPlan phase mismatch")
    del attempt, trial_spec, allow_attempt_missing


def _validate_decoder_implementation(
    derivation_plan: Mapping[str, Any],
    raw: bytes,
) -> None:
    if not isinstance(raw, bytes):
        raise ValueError("frozen decoder implementation must be immutable bytes")
    descriptor = derivation_plan["decoder_descriptor"]
    reference = descriptor["immutable_ref"]
    digest = contract_digest(raw)
    if reference["digest"] != descriptor["implementation_hash"] or digest != descriptor["implementation_hash"]:
        raise ValueError("frozen decoder implementation hash mismatch")
    semantic_contract = {
        "canonicalization": deepcopy(derivation_plan["canonicalization"]),
        "coverage_contract": deepcopy(derivation_plan["coverage_contract"]),
    }
    expected = {
        "decoder_id": descriptor["decoder_id"],
        "decoder_version": descriptor["decoder_version"],
        "semantic_contract": semantic_contract,
    }
    if canonical_contract_bytes(expected) != raw:
        raise ValueError("frozen decoder implementation bytes differ from its semantic contract")
    if contract_digest(canonical_contract_bytes(semantic_contract)) != descriptor["semantic_hash"]:
        raise ValueError("frozen decoder semantic hash mismatch")


def _source_output_fact(
    item: PhysicalReceiptInput,
    output: PhysicalRawOutput,
) -> dict[str, Any]:
    return {
        "source_ordinal": output.source_ordinal,
        "command_id": item.command_id,
        "command_spec_id": item.command_spec_id,
        "command_hash": item.command_hash,
        "completed_event_id": item.completed_event_id,
        "receipt_ref": deepcopy(item.receipt_ref),
        "receipt_hash": item.receipt_hash,
        "output_id": output.output_id,
        "output_kind": output.kind,
        "output_schema_version": output.schema_version,
        "output_ref": deepcopy(output.contract_ref),
        "output_hash": output.content_hash,
    }


def _receipt_bound_sources(inputs: Sequence[PhysicalReceiptInput]) -> ReceiptBoundSources:
    raw_facts: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_fact_lineage: dict[tuple[str, str, str], dict[str, Any]] = {}
    observed_surfaces: list[str] = []
    physical: list[dict[str, Any]] = []
    for item in inputs:
        for output in item.raw_outputs:
            payload = _json_object(output.raw_bytes, label=f"raw output {output.output_id}")
            source_key = (item.phase, item.command_spec_id, output.output_id)
            if source_key in raw_facts:
                raise ValueError("physical raw source identity is duplicate")
            raw_facts[source_key] = deepcopy(payload)
            raw_fact_lineage[source_key] = {
                "source_phase": item.phase,
                "command_spec_id": item.command_spec_id,
                "output_id": output.output_id,
                "output_kind": output.kind,
                "role": output.role,
                "dataset_id": output.dataset_id,
                "command_status": "completed",
                "exit_code": int(item.receipt["exit_code"]),
                "receipt_ref": deepcopy(item.receipt_ref),
                "receipt_hash": item.receipt_hash,
                "completed_event_id": item.completed_event_id,
                "output_ref": deepcopy(output.contract_ref),
                "output_hash": output.content_hash,
            }
            for surface_id in payload.get("observed_surface_ids") or []:
                if not isinstance(surface_id, str) or not surface_id:
                    raise ValueError("physical activation surface identity is invalid")
                if surface_id not in observed_surfaces:
                    observed_surfaces.append(surface_id)
            physical.append({
                "command_id": item.command_id,
                "command_spec_id": item.command_spec_id,
                "receipt_hash": item.receipt_hash,
                "output_id": output.output_id,
                "output_hash": output.content_hash,
            })
    surface_checks = tuple(
        {"surface_id": surface_id, "observed": True}
        for surface_id in observed_surfaces
    )
    return ReceiptBoundSources(
        raw_facts=raw_facts,
        raw_fact_lineage=raw_fact_lineage,
        surface_checks=surface_checks,
        physical_inputs=tuple(physical),
    )


def _canonical_passthrough_decoder(
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase: str,
    plan: Mapping[str, Any],
    inputs: Sequence[PhysicalReceiptInput],
    sources: ReceiptBoundSources,
) -> tuple[DerivedEvidence, ...]:
    del attempt, trial_spec, phase, sources
    indexed: dict[str, bytes] = {}
    for source in inputs:
        for output in source.raw_outputs:
            payload = _canonical_json_object(output.raw_bytes, label=f"raw output {output.output_id}")
            kind = payload.get("evidence_kind")
            if output.normalized_kinds != (kind,):
                raise ValueError("canonical identity source kind differs from frozen normalized_kinds")
            if kind in indexed:
                raise ValueError(f"canonical decoder received duplicate evidence kind: {kind}")
            indexed[str(kind)] = output.raw_bytes
    results: list[DerivedEvidence] = []
    for expected in plan["expected_normalized_outputs"]:
        kind = str(expected["kind"])
        raw = indexed.get(kind)
        if raw is None:
            raise ValueError(f"canonical decoder lacks physical output for {kind}")
        results.append(DerivedEvidence(
            output_id=str(expected["output_id"]),
            kind=kind,
            schema_version=str(expected["schema_version"]),
            raw_bytes=raw,
        ))
    if set(indexed) != {item.kind for item in results}:
        raise ValueError("canonical decoder physical evidence exact-set mismatch")
    return tuple(results)


def _c2c_physical_decoder(
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase: str,
    plan: Mapping[str, Any],
    inputs: Sequence[PhysicalReceiptInput],
    sources: ReceiptBoundSources,
) -> tuple[DerivedEvidence, ...]:
    measurements: dict[tuple[str, str, int, str], float] = {}
    for source in inputs:
        for output in source.raw_outputs:
            if output.role not in {"baseline", "candidate", "ablation", "matched_control", "coverage", "activation_disabled"}:
                continue
            if output.dataset_id is None or len(output.seeds) != 1 or len(output.metrics) != 1:
                raise ValueError("C2C physical measurement lacks frozen dataset/seed/metric identity")
            payload = _json_object(output.raw_bytes, label=f"C2C raw output {output.output_id}")
            declared_dataset = payload.get("dataset")
            if declared_dataset is not None and str(declared_dataset) != output.dataset_id:
                raise ValueError("C2C raw measurement dataset differs from frozen binding")
            identity = (output.role, output.dataset_id, output.seeds[0], output.metrics[0])
            if identity in measurements:
                raise ValueError("C2C physical measurement identity is duplicate")
            measurements[identity] = _metric_value(payload)

    phase_execution = attempt["phase_executions"][phase]
    expected_outputs = list(plan["expected_normalized_outputs"])
    payloads: dict[str, dict[str, Any]] = {}
    for expected in expected_outputs:
        kind = str(expected["kind"])
        if kind in {"proxy_results", "main_results", "ablation_results", "matched_control_results", "coverage_results"}:
            roles = {
                "proxy_results": ("baseline", "candidate"),
                "main_results": ("baseline", "candidate"),
                "ablation_results": ("ablation",),
                "matched_control_results": ("matched_control",),
                "coverage_results": ("coverage",),
            }[kind]
            rows = []
            phase_contract = next(item for item in trial_spec["phase_contracts"] if item["phase"] == phase)
            for dataset_id in phase_contract["datasets"]:
                for seed in phase_contract["seeds"]:
                    for metric_id in phase_contract["metrics"]:
                        for role in roles:
                            identity = (role, dataset_id, seed, metric_id)
                            if identity not in measurements:
                                raise ValueError(f"C2C decoder lacks paired measurement {identity}")
                            rows.append(_measurement_row(
                                attempt=attempt,
                                phase=phase,
                                phase_execution=phase_execution,
                                role=role,
                                dataset_id=dataset_id,
                                metric_id=metric_id,
                                seed=seed,
                                value=measurements[identity],
                            ))
            payloads[kind] = {**_evidence_identity(attempt, phase, phase_execution, kind, str(expected["schema_version"])), "rows": rows}
    if "activation_evidence" in {str(item["kind"]) for item in expected_outputs}:
        payloads["activation_evidence"] = _activation_payload(
            attempt=attempt,
            trial_spec=trial_spec,
            phase_execution=phase_execution,
            measurements=measurements,
            sources=sources,
            physical_inputs=inputs,
            schema_version=next(str(item["schema_version"]) for item in expected_outputs if item["kind"] == "activation_evidence"),
        )
        _bind_activation_readiness_facts(
            trial_spec=trial_spec,
            sources=sources,
            activation_payload=payloads["activation_evidence"],
        )
    _derive_auxiliary_proxy_payloads(
        attempt=attempt,
        trial_spec=trial_spec,
        phase=phase,
        phase_execution=phase_execution,
        plan=plan,
        inputs=inputs,
        sources=sources,
        payloads=payloads,
    )
    return tuple(
        DerivedEvidence(
            output_id=str(expected["output_id"]),
            kind=str(expected["kind"]),
            schema_version=str(expected["schema_version"]),
            raw_bytes=canonical_contract_bytes(payloads[str(expected["kind"])]),
        )
        for expected in expected_outputs
    )


def _derive_auxiliary_proxy_payloads(
    *,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase: str,
    phase_execution: Mapping[str, Any],
    plan: Mapping[str, Any],
    inputs: Sequence[PhysicalReceiptInput],
    sources: ReceiptBoundSources,
    payloads: dict[str, dict[str, Any]],
) -> None:
    if phase != "proxy":
        return
    expected = {
        str(item["kind"]): str(item["schema_version"])
        for item in plan["expected_normalized_outputs"]
    }
    source_hashes = [output.content_hash for item in inputs for output in item.raw_outputs]
    fingerprint_inputs = {
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "protocol_hash": attempt["protocol_hash"],
        "phase_execution_id": phase_execution["phase_execution_id"],
    }
    baseline_hash = contract_digest(canonical_contract_bytes(fingerprint_inputs))
    if "proxy_baseline_fingerprint" in expected:
        payloads["proxy_baseline_fingerprint"] = {
            **_evidence_identity(attempt, phase, phase_execution, "proxy_baseline_fingerprint", expected["proxy_baseline_fingerprint"]),
            "baseline_hash": baseline_hash,
            "dataset_ids": list(next(item for item in trial_spec["phase_contracts"] if item["phase"] == phase)["datasets"]),
            "seeds": list(next(item for item in trial_spec["phase_contracts"] if item["phase"] == phase)["seeds"]),
            "fingerprint_inputs": fingerprint_inputs,
        }
    if "proxy_cache_report" in expected:
        fingerprint = payloads.get("proxy_baseline_fingerprint")
        if fingerprint is None:
            raise ValueError("proxy cache report requires baseline fingerprint output")
        fingerprint_hash = contract_digest(canonical_contract_bytes(fingerprint))
        payloads["proxy_cache_report"] = {
            **_evidence_identity(attempt, phase, phase_execution, "proxy_cache_report", expected["proxy_cache_report"]),
            "cross_references": {"proxy_baseline_fingerprint_hash": fingerprint_hash},
            "cache_key": contract_digest(canonical_contract_bytes({"attempt_input_hash": attempt["attempt_input_hash"], "source_hashes": source_hashes})),
            "baseline_hash": baseline_hash,
            "cache_entry_hash": baseline_hash,
            "status": "created",
        }
    proxy_results = payloads.get("proxy_results")
    activation = payloads.get("activation_evidence")
    cross_references = {}
    if proxy_results is not None:
        cross_references["proxy_results_hash"] = contract_digest(canonical_contract_bytes(proxy_results))
    if activation is not None:
        cross_references["activation_evidence_hash"] = contract_digest(canonical_contract_bytes(activation))
    if "full_s3_readiness" in expected:
        from .proxy_classifier import derive_readiness_from_receipts

        readiness_plan = readiness_check_plan_for_phase(trial_spec, phase)
        if not isinstance(readiness_plan, Mapping):
            raise ValueError("full readiness derivation requires frozen ReadinessCheckPlan")
        derived = derive_readiness_from_receipts(
            readiness_check_plan=readiness_plan,
            receipt_bound_sources=sources,
        )
        phase_contract = next(
            item for item in trial_spec["phase_contracts"] if item["phase"] == phase
        )
        readiness_ref = phase_contract.get("readiness_check_plan_ref")
        readiness_hash = phase_contract.get("readiness_check_plan_hash")
        if not isinstance(readiness_ref, Mapping) or readiness_ref.get("digest") != readiness_hash:
            raise ValueError("frozen ReadinessCheckPlan reference is invalid")
        if derived["readiness_check_plan_hash"] != readiness_hash:
            raise ValueError("receipt-derived readiness used a different frozen plan")
        payloads["full_s3_readiness"] = {
            **_evidence_identity(attempt, phase, phase_execution, "full_s3_readiness", expected["full_s3_readiness"]),
            "cross_references": cross_references,
            "readiness_check_plan_ref": deepcopy(dict(readiness_ref)),
            "readiness_check_plan_hash": str(readiness_hash),
            "ready": derived["ready"],
            "classification": derived["classification"],
            "checks": derived["checks"],
        }
    if "bootstrap_completion" in expected:
        payloads["bootstrap_completion"] = {
            **_evidence_identity(attempt, phase, phase_execution, "bootstrap_completion", expected["bootstrap_completion"]),
            "cross_references": cross_references,
            "completion_status": "verified",
        }


def _bind_activation_readiness_facts(
    *,
    trial_spec: Mapping[str, Any],
    sources: ReceiptBoundSources,
    activation_payload: Mapping[str, Any],
) -> None:
    readiness_plan = readiness_check_plan_for_phase(trial_spec, "proxy")
    if not isinstance(readiness_plan, Mapping):
        raise ValueError("activation facts require frozen proxy ReadinessCheckPlan")
    checks = [item for item in readiness_plan["checks"] if item["check_kind"] == "activation_delta"]
    if len(checks) != 1:
        raise ValueError("proxy readiness plan must contain one activation_delta check")
    candidate_values = [
        _metric_value(sources.raw_facts[key])
        for key, lineage in sources.raw_fact_lineage.items()
        if lineage.get("role") == "candidate"
    ]
    disabled_values = [
        _metric_value(sources.raw_facts[key])
        for key, lineage in sources.raw_fact_lineage.items()
        if lineage.get("role") == "activation_disabled"
    ]
    if not candidate_values or len(candidate_values) != len(disabled_values):
        raise ValueError("activation readiness lacks paired physical measurements")
    observed_delta = (
        sum(candidate_values) / len(candidate_values)
        - sum(disabled_values) / len(disabled_values)
    )
    for binding in checks[0]["source_bindings"]:
        key = (
            str(binding["source_phase"]),
            str(binding["command_spec_id"]),
            str(binding["output_id"]),
        )
        raw = sources.raw_facts.get(key)
        if not isinstance(raw, dict):
            raise ValueError("activation readiness source is not receipt-bound")
        raw["surface_measurements"] = {"delta": observed_delta}
        raw["observed_surface_ids"] = list(activation_payload["observed_surface_ids"])


def _activation_payload(
    *,
    attempt: Mapping[str, Any],
    trial_spec: Mapping[str, Any],
    phase_execution: Mapping[str, Any],
    measurements: Mapping[tuple[str, str, int, str], float],
    sources: ReceiptBoundSources,
    physical_inputs: Sequence[PhysicalReceiptInput],
    schema_version: str,
) -> dict[str, Any]:
    readiness_plan = readiness_check_plan_for_phase(trial_spec, "proxy")
    if not isinstance(readiness_plan, Mapping):
        raise ValueError("activation derivation requires frozen proxy ReadinessCheckPlan")
    activation_checks = [
        item
        for item in readiness_plan["checks"]
        if item["check_kind"] == "activation_delta"
    ]
    if len(activation_checks) != 1:
        raise ValueError("proxy ReadinessCheckPlan must contain exactly one activation_delta check")
    activation_check = activation_checks[0]
    predicate = activation_check["predicate"]
    if predicate["comparator"] != "delta_gte":
        raise ValueError("activation_delta check must use delta_gte")
    threshold = predicate["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("activation_delta threshold must be numeric")
    required_surfaces = list(activation_check["required_coverage"]["expected_surface_ids"])
    observed = {item["surface_id"] for item in sources.surface_checks if item.get("observed") is True}
    phase_contract = next(item for item in trial_spec["phase_contracts"] if item["phase"] == "proxy")
    frozen_sources = [
        (item["source_phase"], item["command_spec_id"], item["output_id"])
        for item in activation_check["source_bindings"]
    ]
    actual_sources = {
        (item.phase, item.command_spec_id, output.output_id)
        for item in physical_inputs
        for output in item.raw_outputs
    }
    if len(frozen_sources) != len(set(frozen_sources)) or not set(frozen_sources).issubset(actual_sources):
        raise ValueError("activation_delta source bindings are not receipt-bound")
    paired_values: list[tuple[float, float]] = []
    for dataset_id in phase_contract["datasets"]:
        for seed in phase_contract["seeds"]:
            for metric_id in phase_contract["metrics"]:
                enabled = measurements.get(("candidate", dataset_id, seed, metric_id))
                disabled = measurements.get(("activation_disabled", dataset_id, seed, metric_id))
                if enabled is None or disabled is None:
                    raise ValueError("activation derivation lacks enabled/disabled paired measurements")
                paired_values.append((enabled, disabled))
    if not paired_values:
        raise ValueError("activation derivation produced no paired measurements")
    enabled_value = sum(item[0] for item in paired_values) / len(paired_values)
    disabled_value = sum(item[1] for item in paired_values) / len(paired_values)
    delta = enabled_value - disabled_value
    surface_measurements = [
        {
            "surface_id": surface_id,
            "enabled_value": enabled_value,
            "disabled_value": disabled_value,
            "delta": delta,
            "threshold": float(threshold),
            "status": (
                "ACTIVATED"
                if surface_id in observed and delta >= float(threshold)
                else "NOT_ACTIVATED"
            ),
        }
        for surface_id in required_surfaces
        if surface_id in observed
    ]
    activated = (
        {item["surface_id"] for item in surface_measurements} == set(required_surfaces)
        and all(item["status"] == "ACTIVATED" for item in surface_measurements)
    )
    return {
        **_evidence_identity(attempt, "proxy", phase_execution, "activation_evidence", schema_version),
        "probe_id": str(activation_check["check_id"]),
        "status": "activated" if activated else "not_activated",
        "command_status": "completed",
        "exit_code": 0,
        "expected_surface_ids": required_surfaces,
        "observed_surface_ids": [item for item in required_surfaces if item in observed],
        "activation_delta_threshold": float(threshold),
        "surface_measurements": surface_measurements,
    }


def _evidence_identity(
    attempt: Mapping[str, Any],
    phase: str,
    phase_execution: Mapping[str, Any],
    kind: str,
    schema_version: str,
) -> dict[str, Any]:
    producer = str(phase_execution["producer_run_id"])
    return {
        "schema_version": schema_version,
        "evidence_kind": kind,
        "evidence_id": f"evidence:{kind}:{producer}",
        "attempt_id": attempt["attempt_id"],
        "producer_run_id": producer,
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
        "phase": phase,
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
        "cross_references": {},
    }


def _measurement_row(
    *,
    attempt: Mapping[str, Any],
    phase: str,
    phase_execution: Mapping[str, Any],
    role: str,
    dataset_id: str,
    metric_id: str,
    seed: int,
    value: float,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "role": role,
        "dataset_id": dataset_id,
        "metric_id": metric_id,
        "seed": seed,
        "metric_value": value,
        "command_status": "completed",
        "attempt_id": attempt["attempt_id"],
        "variant_semantic_hash": attempt["variant_semantic_hash"],
        "variant_spec_hash": attempt["variant_spec_hash"],
        "trial_spec_hash": attempt["trial_spec_hash"],
        "sample_manifest_hash": attempt["sample_manifest_hash"],
        "evaluator_hash": attempt["evaluator_hash"],
        "producer_run_id": phase_execution["producer_run_id"],
        "lifecycle_generation": attempt["lifecycle_generation"],
        "implementation_hash": attempt["implementation_hash"],
        "attempt_input_hash": attempt["attempt_input_hash"],
        "phase_execution_id": phase_execution["phase_execution_id"],
        "phase_start_event_id": phase_execution["phase_start_event_id"],
    }


def _metric_value(payload: Mapping[str, Any]) -> float:
    raw_value = next(
        (payload[key] for key in ("metric_value", "accuracy_percent", "overall_accuracy", "accuracy", "mean") if key in payload),
        None,
    )
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
        raise ValueError("C2C raw measurement lacks a finite numeric metric")
    value = float(raw_value)
    if "overall_accuracy" in payload or ("accuracy" in payload and 0.0 <= value <= 1.0):
        value *= 100.0
    return round(value, 10)


def _canonical_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    payload = _json_object(raw, label=label)
    if canonical_contract_bytes(payload) != raw:
        raise ValueError(f"{label} bytes are not a canonical JSON object")
    return payload


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


_DECODER_REGISTRY: dict[tuple[str, str], Decoder] = {
    ("canonical-identity", "1"): _canonical_passthrough_decoder,
    ("c2c-receipt-measurements", "1"): _c2c_physical_decoder,
}


__all__ = [
    "DERIVATION_MANIFEST_SCHEMA_FILE",
    "DERIVATION_PLAN_SCHEMA_FILE",
    "DerivedEvidence",
    "DeterministicDerivation",
    "PhysicalRawOutput",
    "PhysicalReceiptInput",
    "ReceiptBoundSources",
    "ValidatedEvidenceDerivation",
    "derive_evidence_deterministically",
    "validate_immutable_derivation",
    "validated_physical_receipt_inputs",
]
