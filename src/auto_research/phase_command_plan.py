"""Frozen, content-addressed command plans for authoritative phases."""

from __future__ import annotations

import json
import stat
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .contract_store import ContractStore
from .derivation_contracts import (
    build_evidence_derivation_plan,
    build_readiness_check_plan,
    freeze_decoder_descriptor,
    store_evidence_derivation_plan,
    store_readiness_check_plan,
    validate_evidence_derivation_plan,
    validate_readiness_check_plan,
)


PHASE_COMMAND_PLAN_SCHEMA_VERSION = "auto_research_phase_command_plan_v3"

_PHASES = {"proxy", "full"}
_PROVENANCE_MODES = {"synthetic", "local-external", "production"}
_C2C_RAW_OUTPUT_SPECS_ENV = "AUTO_RESEARCH_C2C_RAW_OUTPUT_SPECS"
_OUTPUT_SCHEMA_VERSIONS = {
    "main_results": "auto_research_main_results_v3",
    "proxy_results": "auto_research_proxy_results_v1",
    "ablation_results": "auto_research_ablation_results_v3",
    "coverage_results": "auto_research_coverage_results_v3",
    "matched_control_results": "auto_research_matched_control_results_v3",
    "activation_evidence": "auto_research_activation_evidence_v4",
    "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v3",
    "proxy_cache_report": "auto_research_proxy_cache_report_v3",
    "full_s3_readiness": "auto_research_full_s3_readiness_v4",
    "bootstrap_completion": "auto_research_bootstrap_completion_v3",
}


def build_phase_command_plan(
    *,
    phase: str,
    adapter_id: str,
    adapter_version: str,
    provenance_mode: str,
    variant_spec_hash: str,
    source_snapshot_hash: str,
    command_values: Sequence[Any],
    expected_evidence: Sequence[Mapping[str, Any]],
    default_cwd: str,
    project_root: Path,
    coverage_contract: Mapping[str, Any],
    readiness_checks: Sequence[Mapping[str, Any]] = (),
    cross_phase_bindings: Sequence[Mapping[str, Any]] = (),
    decoder_id: str = "canonical-identity",
    decoder_version: str = "1",
) -> dict[str, Any]:
    """Build a deterministic command DAG from already selected execution facts."""

    if phase not in _PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    if provenance_mode not in _PROVENANCE_MODES:
        raise ValueError(f"unsupported provenance mode: {provenance_mode}")
    if not command_values:
        raise ValueError(f"{adapter_id} {phase} phase requires at least one frozen command")
    outputs = _normalize_expected_outputs(expected_evidence, output_role="normalized_evidence")
    commands: list[dict[str, Any]] = []
    previous_id: str | None = None
    for ordinal, raw in enumerate(command_values):
        command = _normalize_command(
            raw,
            phase=phase,
            ordinal=ordinal,
            default_cwd=default_cwd,
            source_snapshot_hash=source_snapshot_hash,
        )
        if command["dependencies"] is None:
            command["dependencies"] = [] if previous_id is None else [previous_id]
        if "expected_outputs" not in command:
            command["expected_outputs"] = []
        commands.append(command)
        previous_id = command["command_spec_id"]
    derive_commands = [item for item in commands if item["authority_role"] == "derivation"]
    if not derive_commands:
        physical_ids = [item["command_spec_id"] for item in commands if item["authority_role"] == "physical"]
        depended = {
            dependency
            for item in commands
            if item["authority_role"] == "physical"
            for dependency in item["dependencies"]
        }
        terminal_ids = [command_id for command_id in physical_ids if command_id not in depended]
        derive_command = _normalize_command(
            {
                "command_spec_id": f"{phase}-derive-evidence",
                "authority_role": "derivation",
                "argv": ["auto-research-core", "derive-evidence", phase],
                "cwd": str(project_root),
                "dependencies": terminal_ids,
                "expected_outputs": [],
            },
            phase=phase,
            ordinal=len(commands),
            default_cwd=str(project_root),
            source_snapshot_hash=source_snapshot_hash,
        )
        commands.append(derive_command)
        derive_commands = [derive_command]
    if len(derive_commands) != 1:
        raise ValueError("PhaseCommandPlan requires exactly one constrained derivation command")
    derive_command = derive_commands[0]
    derive_command["expected_outputs"] = outputs
    physical_commands = [item for item in commands if item["authority_role"] == "physical"]
    if (
        adapter_id == "generic-external-adapter"
        and physical_commands
        and not any(item["physical_raw_outputs"] for item in physical_commands)
    ):
        physical_commands[-1]["physical_raw_outputs"] = [
            {
                "output_id": f"raw-{item['kind'].replace('_', '-')}",
                "kind": item["kind"],
                "schema_version": item["schema_version"],
                "locator": f"external-manifest/{item['kind']}.json",
                "locator_type": "file",
                "dataset_id": None,
                "role": None,
                "required": True,
                "normalized_kinds": [item["kind"]],
            }
            for item in outputs
        ]
    source_bindings = _derivation_source_bindings(
        commands,
        outputs,
        phase=phase,
        coverage_contract=coverage_contract,
    )
    decoder = freeze_decoder_descriptor(
        project_root,
        decoder_id=decoder_id,
        decoder_version=decoder_version,
        semantic_contract={
            "canonicalization": {
                "encoding": "utf-8",
                "object_key_order": "lexicographic",
                "row_order": ["phase", "role", "dataset_id", "metric_id", "seed"],
                "duplicate_policy": "reject",
                "numeric_policy": "finite_non_boolean",
            },
            "coverage_contract": dict(coverage_contract),
        },
    )
    derivation_plan = build_evidence_derivation_plan(
        plan_id=f"{adapter_id}:{phase}:derivation",
        phase=phase,
        decoder_descriptor=decoder,
        source_bindings=source_bindings,
        expected_normalized_outputs=[
            {"ordinal": index, "output_id": item["output_id"], "kind": item["kind"], "schema_version": item["schema_version"]}
            for index, item in enumerate(outputs)
        ],
        canonicalization={
            "encoding": "utf-8",
            "object_key_order": "lexicographic",
            "row_order": ["phase", "role", "dataset_id", "metric_id", "seed"],
            "duplicate_policy": "reject",
            "numeric_policy": "finite_non_boolean",
        },
        coverage_contract=coverage_contract,
        cross_phase_bindings=cross_phase_bindings,
    )
    derivation_ref, derivation_hash = store_evidence_derivation_plan(project_root, derivation_plan)
    derive_command["derivation_plan_ref"] = derivation_ref
    derive_command["derivation_plan_hash"] = derivation_hash
    readiness_plan = None
    readiness_ref = None
    readiness_hash = None
    if readiness_checks:
        frozen_checks = _freeze_readiness_checks(
            readiness_checks,
            source_bindings=source_bindings,
            decoder_descriptor=decoder,
        )
        readiness_plan = build_readiness_check_plan(
            plan_id=f"{adapter_id}:{phase}:readiness",
            phase=phase,
            checks=frozen_checks,
        )
        readiness_ref, readiness_hash = store_readiness_check_plan(project_root, readiness_plan)
    plan = {
        "schema_version": PHASE_COMMAND_PLAN_SCHEMA_VERSION,
        "plan_id": f"{adapter_id}:{phase}:commands",
        "phase": phase,
        "adapter_identity": {
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "provenance_mode": provenance_mode,
        },
        "variant_spec_hash": variant_spec_hash,
        "source_snapshot_hash": source_snapshot_hash,
        "derivation_plan": derivation_plan,
        "derivation_plan_ref": derivation_ref,
        "derivation_plan_hash": derivation_hash,
        "readiness_check_plan": readiness_plan,
        "readiness_check_plan_ref": readiness_ref,
        "readiness_check_plan_hash": readiness_hash,
        "commands": commands,
    }
    validate_phase_command_plan(plan, expected_evidence_kinds=[item["kind"] for item in outputs])
    return plan


def store_phase_command_plan(project_root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Persist a validated plan and return its immutable reference and digest."""

    validate_phase_command_plan(plan)
    reference = ContractStore(project_root).put_json(
        dict(plan),
        schema_file="phase_command_plan_v3.schema.json",
    )
    return reference, str(reference["digest"])


def validate_phase_command_plan(
    plan: Mapping[str, Any],
    *,
    expected_evidence_kinds: Iterable[str] | None = None,
) -> None:
    """Validate DAG order, dependency closure, paths, and exact output registration."""

    from .contract_store import validate_schema

    validate_schema(plan, "phase_command_plan_v3.schema.json")
    validate_evidence_derivation_plan(plan["derivation_plan"])
    if plan["derivation_plan_ref"]["digest"] != plan["derivation_plan_hash"]:
        raise ValueError("PhaseCommandPlan derivation plan reference mismatch")
    readiness = plan.get("readiness_check_plan")
    if readiness is None:
        if plan.get("readiness_check_plan_ref") is not None or plan.get("readiness_check_plan_hash") is not None:
            raise ValueError("PhaseCommandPlan readiness plan identity is incomplete")
    else:
        validate_readiness_check_plan(readiness)
        if plan["readiness_check_plan_ref"]["digest"] != plan["readiness_check_plan_hash"]:
            raise ValueError("PhaseCommandPlan readiness plan reference mismatch")
    commands = list(plan["commands"])
    command_ids = [item["command_spec_id"] for item in commands]
    if len(command_ids) != len(set(command_ids)):
        raise ValueError("PhaseCommandPlan command_spec_id values must be unique")
    ordinals = [item["ordinal"] for item in commands]
    if ordinals != list(range(len(commands))):
        raise ValueError("PhaseCommandPlan ordinals must be contiguous and ordered")
    positions = {command_id: index for index, command_id in enumerate(command_ids)}
    derive_commands = [item for item in commands if item["authority_role"] == "derivation"]
    if len(derive_commands) != 1:
        raise ValueError("PhaseCommandPlan must contain exactly one derivation command")
    derive_command = derive_commands[0]
    if derive_command["derivation_plan_ref"] != plan["derivation_plan_ref"] or derive_command["derivation_plan_hash"] != plan["derivation_plan_hash"]:
        raise ValueError("derive command does not bind the top-level EvidenceDerivationPlan identity")
    for index, command in enumerate(commands):
        if command["phase"] != plan["phase"]:
            raise ValueError("PhaseCommandPlan command phase mismatch")
        if command["source_snapshot_hash"] != plan["source_snapshot_hash"]:
            raise ValueError("PhaseCommandPlan command source snapshot mismatch")
        _validate_cwd(command["cwd"])
        dependencies = command["dependencies"]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("PhaseCommandPlan dependencies must be unique")
        for dependency in dependencies:
            if dependency not in positions:
                raise ValueError(f"PhaseCommandPlan dependency is unknown: {dependency}")
            if positions[dependency] >= index:
                raise ValueError("PhaseCommandPlan dependencies must precede their command")
    output_kinds = [output["kind"] for command in commands for output in command["expected_outputs"] if output["output_role"] == "normalized_evidence"]
    if len(output_kinds) != len(set(output_kinds)):
        raise ValueError("PhaseCommandPlan evidence kinds must be produced exactly once")
    if expected_evidence_kinds is not None and set(output_kinds) != set(expected_evidence_kinds):
        raise ValueError("PhaseCommandPlan outputs must exactly match the phase evidence contract")
    frozen_sources = [
        (item["source_phase"], item["command_spec_id"], item["output_id"])
        for item in plan["derivation_plan"]["source_bindings"]
    ]
    physical_sources = [
        (plan["phase"], command["command_spec_id"], output["output_id"])
        for command in commands
        if command["authority_role"] == "physical"
        for output in command["physical_raw_outputs"]
        if output["normalized_kinds"]
    ]
    if frozen_sources != physical_sources:
        raise ValueError("EvidenceDerivationPlan sources differ from ordered physical raw outputs")


def phase_command_plan_for_phase(trial_spec: Mapping[str, Any], phase: str) -> dict[str, Any]:
    contracts = [item for item in trial_spec.get("phase_contracts", []) if item.get("phase") == phase]
    if len(contracts) != 1:
        raise ValueError(f"TrialSpec must contain exactly one {phase} phase contract")
    plan = deepcopy(contracts[0]["command_plan"])
    validate_phase_command_plan(plan, expected_evidence_kinds=contracts[0]["evidence_kinds"])
    return plan


def _normalize_command(
    raw: Any,
    *,
    phase: str,
    ordinal: int,
    default_cwd: str,
    source_snapshot_hash: str,
) -> dict[str, Any]:
    if isinstance(raw, str):
        raise ValueError("PhaseCommandPlan commands must use typed argv, not shell strings")
    elif isinstance(raw, Mapping):
        payload = deepcopy(dict(raw))
        raw_argv = payload.pop("argv", None)
        if isinstance(raw_argv, str):
            raise ValueError("PhaseCommandPlan argv must be a string array")
        argv = list(raw_argv or [])
    else:
        argv = list(raw) if isinstance(raw, Sequence) else []
        payload = {}
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("PhaseCommandPlan argv must be a non-empty string array")
    command_spec_id = str(payload.pop("command_spec_id", f"{phase}-command-{ordinal:03d}"))
    raw_environment = dict(payload.pop("environment", {}))
    raw_physical_outputs = payload.pop("physical_raw_outputs", None)
    command = {
        "command_spec_id": command_spec_id,
        "phase": phase,
        "ordinal": ordinal,
        "authority_role": str(payload.pop("authority_role", "derivation" if "derive-evidence" in command_spec_id else "physical")),
        "dependencies": list(payload.pop("dependencies")) if "dependencies" in payload else None,
        "argv": argv,
        "environment": {
            str(key): str(value)
            for key, value in raw_environment.items()
        },
        "inherited_environment": sorted(
            str(value)
            for value in payload.pop("inherited_environment", ["HOME", "PATH", "TMPDIR"])
        ),
        "cwd": str(payload.pop("cwd", default_cwd)),
        "source_snapshot_hash": str(payload.pop("source_snapshot_hash", source_snapshot_hash)),
        "expected_outputs": (
            _normalize_expected_outputs(payload.pop("expected_outputs"), output_role="physical_raw")
            if "expected_outputs" in payload else None
        ),
        "physical_raw_outputs": _normalize_physical_raw_outputs(
            raw_physical_outputs,
            environment=raw_environment,
            phase=phase,
        ),
        "resource_policy": payload.pop(
            "resource_policy",
            {"resource_class": "cpu", "minimum_capacity": 1, "unit": "count"},
        ),
        "retry_policy": payload.pop(
            "retry_policy",
            {"max_attempts": 1, "retryable_exit_codes": [], "backoff_seconds": 0},
        ),
        "resume_policy": payload.pop(
            "resume_policy",
            {"mode": "receipt_only", "external_job_attach_required": False},
        ),
        "condition": payload.pop("condition", {"kind": "always", "predicate_hash": None}),
    }
    if command["expected_outputs"] is None:
        del command["expected_outputs"]
    if payload:
        raise ValueError(f"unsupported PhaseCommandPlan command fields: {sorted(payload)}")
    return command


def _normalize_expected_outputs(values: Sequence[Mapping[str, Any]], *, output_role: str) -> list[dict[str, Any]]:
    outputs = []
    for value in values:
        kind = str(value.get("kind") or "")
        schema_version = str(value.get("schema_version") or _OUTPUT_SCHEMA_VERSIONS.get(kind) or "")
        if not kind or not schema_version:
            raise ValueError("phase evidence outputs require kind and schema_version")
        output_id = str(value.get("output_id") or f"normalized-{kind.replace('_', '-')}")
        outputs.append({
            "output_id": output_id,
            "kind": kind,
            "schema_version": schema_version,
            "required": bool(value.get("required", True)),
            "output_role": str(value.get("output_role") or output_role),
            "normalized_kinds": list(value.get("normalized_kinds") or ([kind] if output_role == "physical_raw" else [])),
            "role": value.get("role"),
            "dataset_id": value.get("dataset_id"),
        })
    return outputs


def _derivation_source_bindings(
    commands: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    coverage_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    physical = [item for item in commands if item["authority_role"] == "physical"]
    raw_outputs = [
        (command, output)
        for command in physical
        for output in command["physical_raw_outputs"]
    ]
    if not raw_outputs and len(physical) == 1 and physical[0]["argv"][0] == "auto-research-adapter":
        physical[0]["physical_raw_outputs"] = [
            {
                "output_id": f"raw-{item['kind'].replace('_', '-')}",
                "kind": f"raw_{item['kind']}",
                "schema_version": item["schema_version"],
                "locator": f"synthetic/{item['kind']}.json",
                "locator_type": "file",
                "dataset_id": None,
                "role": None,
                "required": True,
                "normalized_kinds": [item["kind"]],
            }
            for item in outputs
        ]
        raw_outputs = [(physical[0], output) for output in physical[0]["physical_raw_outputs"]]
    if not raw_outputs:
        raise ValueError("PhaseCommandPlan physical commands must freeze raw outputs for derivation")
    bindings = []
    expected = {item["kind"] for item in outputs}
    for command, output in raw_outputs:
        normalized_kinds = [item for item in output["normalized_kinds"] if item in expected]
        output["normalized_kinds"] = normalized_kinds
        if not normalized_kinds:
            continue
        bindings.append({
            "source_ordinal": len(bindings),
            "source_phase": phase,
            "command_spec_id": command["command_spec_id"],
            "output_id": output["output_id"],
            "output_kind": output["kind"],
            "output_schema_version": output["schema_version"],
            "normalized_kinds": normalized_kinds,
            "role": output.get("role"),
            "dataset_id": output.get("dataset_id"),
            "seeds": list(coverage_contract["seeds"]),
            "metrics": list(coverage_contract["metrics"]),
        })
    if {kind for item in bindings for kind in item["normalized_kinds"]} != expected:
        raise ValueError("PhaseCommandPlan raw outputs do not exactly cover normalized evidence kinds")
    return bindings


def _normalize_physical_raw_outputs(
    value: Any,
    *,
    environment: Mapping[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    raw = value
    if raw is None and environment.get(_C2C_RAW_OUTPUT_SPECS_ENV) is not None:
        try:
            raw = json.loads(str(environment[_C2C_RAW_OUTPUT_SPECS_ENV]))
        except json.JSONDecodeError as exc:
            raise ValueError("frozen physical raw output specs are malformed") from exc
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("physical_raw_outputs must be an ordered object array")
    outputs = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("physical raw output spec must be an object")
        normalized = deepcopy(dict(item))
        kind = str(normalized.get("kind") or "")
        role = normalized.get("role")
        if kind == "c2c_measurement":
            if phase == "proxy" and role == "baseline":
                candidates = ["proxy_results", "proxy_baseline_fingerprint", "proxy_cache_report"]
            elif phase == "proxy" and role == "candidate":
                candidates = ["proxy_results", "activation_evidence", "bootstrap_completion"]
            else:
                candidates = ["main_results"]
        elif kind == "c2c_activation_measurement":
            candidates = ["activation_evidence", "full_s3_readiness", "bootstrap_completion"]
        elif kind == "c2c_ablation_measurement":
            candidates = ["ablation_results"]
        else:
            candidates = [kind] if kind in _OUTPUT_SCHEMA_VERSIONS else []
        normalized["normalized_kinds"] = list(normalized.get("normalized_kinds") or candidates)
        normalized.setdefault("dataset_id", None)
        normalized.setdefault("role", role)
        normalized.setdefault("required", True)
        outputs.append(normalized)
    return outputs


def _freeze_readiness_checks(
    checks: Sequence[Mapping[str, Any]],
    *,
    source_bindings: Sequence[Mapping[str, Any]],
    decoder_descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for ordinal, value in enumerate(checks):
        check = deepcopy(dict(value))
        check_kind = str(check.get("check_kind") or "raw_measurement")
        normalized_kind = "activation_evidence" if check_kind == "activation_delta" else "full_s3_readiness"
        matching = [item for item in source_bindings if normalized_kind in item["normalized_kinds"]]
        if not matching:
            raise ValueError(f"readiness check lacks a physical source for {normalized_kind}")
        check["ordinal"] = ordinal
        check["check_kind"] = check_kind
        check["source_bindings"] = [
            {
                "source_ordinal": index,
                "source_phase": item["source_phase"],
                "command_spec_id": item["command_spec_id"],
                "output_id": item["output_id"],
                "output_kind": item["output_kind"],
                "output_schema_version": item["output_schema_version"],
            }
            for index, item in enumerate(matching)
        ]
        check["decoder_descriptor"] = deepcopy(dict(decoder_descriptor))
        check["blocked_classification"] = "IMPLEMENTATION_BLOCKED"
        check["blocked_route"] = "REPAIR_IMPLEMENTATION"
        result.append(check)
    return result


def _validate_cwd(value: str) -> None:
    if not value or "\x00" in value:
        raise ValueError("PhaseCommandPlan cwd is invalid")
    path = PurePosixPath(value.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("PhaseCommandPlan cwd cannot traverse parent directories")
    candidate = Path(value)
    if not candidate.is_absolute():
        return
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError("PhaseCommandPlan cwd cannot contain symlink components")


__all__ = [
    "PHASE_COMMAND_PLAN_SCHEMA_VERSION",
    "build_phase_command_plan",
    "phase_command_plan_for_phase",
    "store_phase_command_plan",
    "validate_phase_command_plan",
]
