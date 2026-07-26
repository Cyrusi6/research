"""Frozen declarative decoder, derivation, and readiness contracts."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, validate_schema


DECODER_PROGRAM_SCHEMA_VERSION = "auto_research_decoder_program_v1"
DECODER_IMPLEMENTATION_BUNDLE_SCHEMA_VERSION = "auto_research_decoder_implementation_bundle_v1"
DECODER_DESCRIPTOR_SCHEMA_VERSION = "auto_research_decoder_descriptor_v2"
DECODER_RUNTIME_ABI = "auto_research_declarative_decoder_vm_v1"
DECODER_ENTRYPOINT = "execute_decoder_program"
EVIDENCE_DERIVATION_PLAN_SCHEMA_VERSION = "auto_research_evidence_derivation_plan_v2"
EVIDENCE_DERIVATION_MANIFEST_SCHEMA_VERSION = "auto_research_evidence_derivation_manifest_v3"
READINESS_CHECK_PLAN_SCHEMA_VERSION = "auto_research_readiness_check_plan_v2"

SUPPORTED_DECODER_IDS = frozenset({"canonical-identity", "c2c-receipt-measurements"})

_DECODER_KIND_BY_ID = {
    "canonical-identity": "canonical_identity",
    "c2c-receipt-measurements": "c2c_receipt_measurements",
}
_AUTHORITY_ROLE_ORDER = (
    "normalized_evidence_source",
    "scientific_metric",
    "activation_enabled",
    "activation_disabled",
    "activation_surface",
    "readiness",
)
_DECODER_OPERATIONS = {
    "canonical_identity": (
        "validate_ordered_source_inventory",
        "parse_canonical_json",
        "validate_authority_roles",
        "copy_registered_evidence",
        "validate_exact_output_inventory",
        "emit_canonical_outputs",
    ),
    "c2c_receipt_measurements": (
        "validate_ordered_source_inventory",
        "parse_canonical_json",
        "validate_authority_roles",
        "extract_finite_measurements",
        "normalize_metric_units",
        "validate_exact_cartesian_coverage",
        "derive_measurement_rows",
        "derive_activation_evidence",
        "derive_readiness_evidence",
        "derive_auxiliary_evidence",
        "validate_exact_output_inventory",
        "emit_canonical_outputs",
    ),
}
_DECODER_VM_SEMANTICS = {
    "runtime_abi": DECODER_RUNTIME_ABI,
    "entrypoint": DECODER_ENTRYPOINT,
    "operation_order": "program_order",
    "unknown_operation_policy": "reject",
    "source_inventory_policy": "ordered_exact",
    "readiness_inventory_policy": "ordered_exact",
    "output_inventory_policy": "ordered_exact",
    "json_policy": "canonical_utf8_object",
    "numeric_policy": "finite_non_boolean",
    "operations": sorted({operation for values in _DECODER_OPERATIONS.values() for operation in values}),
}
DECODER_OPERATION_CONTRACT = {
    "source_json": {
        "canonical_bytes_required": True,
        "object_required": True,
        "unknown_field_policy": "schema_reject",
    },
    "measurement": {
        "value_fields": ["metric_value", "accuracy_percent", "overall_accuracy", "accuracy", "mean"],
        "dataset_field": "dataset",
        "fractional_scale_fields": ["overall_accuracy", "accuracy"],
        "fractional_scale_min": 0.0,
        "fractional_scale_max": 1.0,
        "fractional_scale_multiplier": 100.0,
        "pairing_key_fields": ["phase_execution_id", "role", "dataset_id", "metric_id", "seed"],
        "duplicate_policy": "reject",
        "numeric_policy": "finite_non_boolean",
    },
    "activation": {
        "enabled_role": "candidate",
        "disabled_role": "activation_disabled",
        "surface_field": "observed_surface_ids",
        "comparison": "paired_mean_delta_gte",
        "surface_coverage": "ordered_exact",
        "command_status_source": "phase_receipt",
    },
    "readiness": {
        "check_identity_fields": ["check_id", "readiness_checks", "checks"],
        "source_inventory": "ordered_exact",
        "check_inventory": "ordered_exact",
        "predicate_source": "frozen_readiness_check_plan",
        "command_status_source": "phase_receipt",
    },
    "auxiliary": {
        "baseline_fingerprint_fields": ["sample_manifest_hash", "evaluator_hash", "protocol_hash", "phase_execution_id"],
        "cache_key_fields": ["attempt_input_hash", "source_hashes"],
        "cross_reference_hash": "sha256_canonical_json",
    },
    "output_identity_fields": [
        "attempt_id",
        "trial_spec_hash",
        "sample_manifest_hash",
        "evaluator_hash",
        "lifecycle_generation",
        "implementation_hash",
        "attempt_input_hash",
        "phase",
        "phase_execution_id",
        "phase_start_event_id",
        "producer_run_id",
    ],
}


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_contract_bytes(value)).hexdigest()


DECODER_RUNTIME_IMPLEMENTATION_HASH = _hash(_DECODER_VM_SEMANTICS)


def freeze_decoder_descriptor(
    project_root: Path,
    *,
    decoder_id: str,
    decoder_version: str,
    semantic_contract: Mapping[str, Any],
    authority_role_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the complete declarative decoder program as immutable bundle bytes."""

    decoder_kind = _DECODER_KIND_BY_ID.get(decoder_id)
    if decoder_kind is None:
        raise ValueError(f"unsupported authoritative decoder: {decoder_id}")
    program = {
        "schema_version": DECODER_PROGRAM_SCHEMA_VERSION,
        "decoder_id": decoder_id,
        "decoder_version": str(decoder_version),
        "decoder_kind": decoder_kind,
        "semantic_contract": deepcopy(dict(semantic_contract)),
        "operations": list(_DECODER_OPERATIONS[decoder_kind]),
        "operation_contract": deepcopy(DECODER_OPERATION_CONTRACT),
        "authority_role_contract": deepcopy(dict(authority_role_contract)),
        "output_contract": deepcopy(dict(output_contract)),
    }
    validate_schema(program, "decoder_program_v1.schema.json")
    dependencies = [
        {
            "dependency_id": "declarative-decoder-vm",
            "kind": "runtime_abi",
            "version": "1",
            "content_hash": DECODER_RUNTIME_IMPLEMENTATION_HASH,
        },
        {
            "dependency_id": "canonical-contract-json",
            "kind": "canonicalization",
            "version": "1",
            "content_hash": _hash(program["semantic_contract"]["canonicalization"]),
        },
        {
            "dependency_id": "normalized-output-contract",
            "kind": "output_contract",
            "version": "1",
            "content_hash": _hash(program["output_contract"]),
        },
    ]
    bundle = {
        "schema_version": DECODER_IMPLEMENTATION_BUNDLE_SCHEMA_VERSION,
        "runtime_abi": DECODER_RUNTIME_ABI,
        "entrypoint": DECODER_ENTRYPOINT,
        "decoder_program": program,
        "dependencies": dependencies,
    }
    validate_schema(bundle, "decoder_implementation_bundle_v1.schema.json")
    immutable_ref = ContractStore(project_root).put_bytes(canonical_contract_bytes(bundle))
    descriptor = {
        "schema_version": DECODER_DESCRIPTOR_SCHEMA_VERSION,
        "decoder_id": decoder_id,
        "decoder_version": str(decoder_version),
        "runtime_abi": DECODER_RUNTIME_ABI,
        "entrypoint": DECODER_ENTRYPOINT,
        "semantic_hash": _hash(program),
        "implementation_hash": immutable_ref["digest"],
        "immutable_ref": immutable_ref,
    }
    validate_schema(descriptor, "decoder_descriptor_v2.schema.json")
    return descriptor


def build_evidence_derivation_plan(
    *,
    plan_id: str,
    phase: str,
    decoder_descriptor: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    expected_normalized_outputs: Sequence[Mapping[str, Any]],
    canonicalization: Mapping[str, Any],
    coverage_contract: Mapping[str, Any],
    cross_phase_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    plan = {
        "schema_version": EVIDENCE_DERIVATION_PLAN_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "phase": str(phase),
        "decoder_descriptor": deepcopy(dict(decoder_descriptor)),
        "source_bindings": [deepcopy(dict(item)) for item in source_bindings],
        "expected_normalized_outputs": [deepcopy(dict(item)) for item in expected_normalized_outputs],
        "canonicalization": deepcopy(dict(canonicalization)),
        "coverage_contract": deepcopy(dict(coverage_contract)),
        "cross_phase_bindings": [deepcopy(dict(item)) for item in cross_phase_bindings],
    }
    validate_evidence_derivation_plan(plan)
    return plan


def validate_evidence_derivation_plan(plan: Mapping[str, Any]) -> None:
    validate_schema(plan, "evidence_derivation_plan_v2.schema.json")
    descriptor = plan["decoder_descriptor"]
    if descriptor["decoder_id"] not in SUPPORTED_DECODER_IDS:
        raise ValueError("EvidenceDerivationPlan decoder is not supported by the declarative decoder VM")
    sources = list(plan["source_bindings"])
    if [item["source_ordinal"] for item in sources] != list(range(len(sources))):
        raise ValueError("EvidenceDerivationPlan source ordinals must be contiguous and ordered")
    source_ids = [(item["source_phase"], item["command_spec_id"], item["output_id"]) for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("EvidenceDerivationPlan source bindings must be unique")
    if any("derive" in item["command_spec_id"] for item in sources):
        raise ValueError("EvidenceDerivationPlan cannot use its derive command as a physical source")
    for source in sources:
        roles = list(source["authority_roles"])
        if roles != [role for role in _AUTHORITY_ROLE_ORDER if role in roles]:
            raise ValueError("EvidenceDerivationPlan authority roles must use canonical order")
        readiness_ids = list(source["readiness_check_ids"])
        if "readiness" in roles and not readiness_ids:
            raise ValueError("readiness authority role requires frozen readiness_check_ids")
    outputs = list(plan["expected_normalized_outputs"])
    if [item["ordinal"] for item in outputs] != list(range(len(outputs))):
        raise ValueError("EvidenceDerivationPlan normalized output ordinals must be contiguous and ordered")
    output_ids = [item["output_id"] for item in outputs]
    output_kinds = [item["kind"] for item in outputs]
    if len(output_ids) != len(set(output_ids)) or len(output_kinds) != len(set(output_kinds)):
        raise ValueError("EvidenceDerivationPlan normalized outputs must be an exact unique set")
    referenced = {kind for item in sources for kind in item["normalized_kinds"]}
    missing = set(output_kinds) - referenced
    if missing:
        raise ValueError(f"EvidenceDerivationPlan normalized outputs lack physical sources: {sorted(missing)}")
    cross_phase = list(plan["cross_phase_bindings"])
    cross_ids = [item["binding_id"] for item in cross_phase]
    if len(cross_ids) != len(set(cross_ids)):
        raise ValueError("EvidenceDerivationPlan cross-phase binding IDs must be unique")


def store_evidence_derivation_plan(project_root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    validate_evidence_derivation_plan(plan)
    reference = ContractStore(project_root).put_json(plan, schema_file="evidence_derivation_plan_v2.schema.json")
    return reference, str(reference["digest"])


def build_readiness_check_plan(
    *,
    plan_id: str,
    phase: str,
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = {
        "schema_version": READINESS_CHECK_PLAN_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "phase": str(phase),
        "source_inventory_mode": "ordered_exact",
        "derived_check_inventory_mode": "ordered_exact",
        "checks": [deepcopy(dict(item)) for item in checks],
        "route_semantics": {
            "integrity_invalid": "BLOCK_INTEGRITY",
            "resource_insufficient": "PAUSE_RESOURCE",
            "semantic_blocked": "REPAIR_IMPLEMENTATION",
            "all_pass": "CONTINUE_PROXY_DECISION",
        },
    }
    validate_readiness_check_plan(plan)
    return plan


def validate_readiness_check_plan(plan: Mapping[str, Any]) -> None:
    validate_schema(plan, "readiness_check_plan_v2.schema.json")
    checks = list(plan["checks"])
    if [item["ordinal"] for item in checks] != list(range(len(checks))):
        raise ValueError("ReadinessCheckPlan ordinals must be contiguous and ordered")
    if len({item["check_id"] for item in checks}) != len(checks):
        raise ValueError("ReadinessCheckPlan check IDs must be unique")
    for check in checks:
        if check["decoder_descriptor"]["decoder_id"] not in SUPPORTED_DECODER_IDS:
            raise ValueError("ReadinessCheckPlan decoder is not supported")
        bindings = check["source_bindings"]
        if [item["source_ordinal"] for item in bindings] != list(range(len(bindings))):
            raise ValueError("ReadinessCheckPlan source ordinals must be contiguous and ordered")
        identities = [(item["source_phase"], item["command_spec_id"], item["output_id"]) for item in bindings]
        if len(identities) != len(set(identities)):
            raise ValueError("ReadinessCheckPlan source bindings must be unique")
        for binding in bindings:
            if binding["check_id"] != check["check_id"]:
                raise ValueError("ReadinessCheckPlan source binding check identity mismatch")
            required_roles = set(binding["required_authority_roles"])
            if check["check_kind"] == "raw_measurement" and "readiness" not in required_roles:
                raise ValueError("raw readiness bindings must require readiness authority")
            if check["check_kind"] == "activation_delta" and not required_roles.issubset({
                "activation_enabled",
                "activation_disabled",
                "activation_surface",
            }):
                raise ValueError("activation readiness bindings may only require activation authority roles")
        if check["check_kind"] == "activation_delta":
            observed_roles = {role for binding in bindings for role in binding["required_authority_roles"]}
            required_roles = {"activation_enabled", "activation_disabled", "activation_surface"}
            if not required_roles.issubset(observed_roles):
                raise ValueError("activation_delta requires enabled, disabled, and surface authority bindings")
        if check["blocked_classification"] != "IMPLEMENTATION_BLOCKED" or check["blocked_route"] != "REPAIR_IMPLEMENTATION":
            raise ValueError("readiness BLOCKED semantics must route to implementation repair")


def store_readiness_check_plan(project_root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    validate_readiness_check_plan(plan)
    reference = ContractStore(project_root).put_json(plan, schema_file="readiness_check_plan_v2.schema.json")
    return reference, str(reference["digest"])


def derivation_plan_for_phase(trial_spec: Mapping[str, Any], phase: str) -> dict[str, Any]:
    contract = _phase_contract(trial_spec, phase)
    plan = deepcopy(contract["derivation_plan"])
    validate_evidence_derivation_plan(plan)
    return plan


def readiness_check_plan_for_phase(trial_spec: Mapping[str, Any], phase: str) -> dict[str, Any] | None:
    contract = _phase_contract(trial_spec, phase)
    plan = contract.get("readiness_check_plan")
    if plan is None:
        return None
    result = deepcopy(plan)
    validate_readiness_check_plan(result)
    return result


def _phase_contract(trial_spec: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    matches = [item for item in trial_spec.get("phase_contracts", []) if item.get("phase") == phase]
    if len(matches) != 1:
        raise ValueError(f"TrialSpec must contain exactly one {phase} phase contract")
    return matches[0]


__all__ = [
    "DECODER_DESCRIPTOR_SCHEMA_VERSION",
    "DECODER_ENTRYPOINT",
    "DECODER_IMPLEMENTATION_BUNDLE_SCHEMA_VERSION",
    "DECODER_PROGRAM_SCHEMA_VERSION",
    "DECODER_RUNTIME_ABI",
    "DECODER_RUNTIME_IMPLEMENTATION_HASH",
    "DECODER_OPERATION_CONTRACT",
    "EVIDENCE_DERIVATION_MANIFEST_SCHEMA_VERSION",
    "EVIDENCE_DERIVATION_PLAN_SCHEMA_VERSION",
    "READINESS_CHECK_PLAN_SCHEMA_VERSION",
    "SUPPORTED_DECODER_IDS",
    "build_evidence_derivation_plan",
    "build_readiness_check_plan",
    "derivation_plan_for_phase",
    "freeze_decoder_descriptor",
    "readiness_check_plan_for_phase",
    "store_evidence_derivation_plan",
    "store_readiness_check_plan",
    "validate_evidence_derivation_plan",
    "validate_readiness_check_plan",
]
