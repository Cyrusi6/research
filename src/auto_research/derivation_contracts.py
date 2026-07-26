"""Frozen derivation and readiness plans for authoritative phase evidence."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, validate_schema


EVIDENCE_DERIVATION_PLAN_SCHEMA_VERSION = "auto_research_evidence_derivation_plan_v1"
EVIDENCE_DERIVATION_MANIFEST_SCHEMA_VERSION = "auto_research_evidence_derivation_manifest_v2"
READINESS_CHECK_PLAN_SCHEMA_VERSION = "auto_research_readiness_check_plan_v1"
DECODER_DESCRIPTOR_SCHEMA_VERSION = "auto_research_decoder_descriptor_v1"

SUPPORTED_DECODER_IDS = frozenset({"canonical-identity", "c2c-receipt-measurements"})


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_contract_bytes(value)).hexdigest()


def freeze_decoder_descriptor(
    project_root: Path,
    *,
    decoder_id: str,
    decoder_version: str,
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if decoder_id not in SUPPORTED_DECODER_IDS:
        raise ValueError(f"unsupported authoritative decoder: {decoder_id}")
    implementation = {
        "decoder_id": decoder_id,
        "decoder_version": str(decoder_version),
        "semantic_contract": deepcopy(dict(semantic_contract)),
    }
    immutable_ref = ContractStore(project_root).put_bytes(canonical_contract_bytes(implementation))
    descriptor = {
        "schema_version": DECODER_DESCRIPTOR_SCHEMA_VERSION,
        "decoder_id": decoder_id,
        "decoder_version": str(decoder_version),
        "semantic_hash": _hash(semantic_contract),
        "implementation_hash": immutable_ref["digest"],
        "immutable_ref": immutable_ref,
    }
    validate_schema(descriptor, "decoder_descriptor_v1.schema.json")
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
    validate_schema(plan, "evidence_derivation_plan_v1.schema.json")
    if plan["decoder_descriptor"]["decoder_id"] not in SUPPORTED_DECODER_IDS:
        raise ValueError("EvidenceDerivationPlan decoder is not supported by the pure decoder registry")
    sources = list(plan["source_bindings"])
    if [item["source_ordinal"] for item in sources] != list(range(len(sources))):
        raise ValueError("EvidenceDerivationPlan source ordinals must be contiguous and ordered")
    source_ids = [(item["source_phase"], item["command_spec_id"], item["output_id"]) for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("EvidenceDerivationPlan source bindings must be unique")
    if any("derive" in item["command_spec_id"] for item in sources):
        raise ValueError("EvidenceDerivationPlan cannot use its derive command as a physical source")
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
    reference = ContractStore(project_root).put_json(plan, schema_file="evidence_derivation_plan_v1.schema.json")
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
    validate_schema(plan, "readiness_check_plan_v1.schema.json")
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
        identities = [(item["command_spec_id"], item["output_id"]) for item in bindings]
        if len(identities) != len(set(identities)):
            raise ValueError("ReadinessCheckPlan source bindings must be unique")
        if check["blocked_classification"] != "IMPLEMENTATION_BLOCKED" or check["blocked_route"] != "REPAIR_IMPLEMENTATION":
            raise ValueError("readiness BLOCKED semantics must route to implementation repair")


def store_readiness_check_plan(project_root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    validate_readiness_check_plan(plan)
    reference = ContractStore(project_root).put_json(plan, schema_file="readiness_check_plan_v1.schema.json")
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
