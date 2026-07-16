"""Frozen, content-addressed command plans for authoritative phases."""

from __future__ import annotations

import stat
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .contract_store import ContractStore


PHASE_COMMAND_PLAN_SCHEMA_VERSION = "auto_research_phase_command_plan_v2"

_PHASES = {"proxy", "full"}
_PROVENANCE_MODES = {"synthetic", "local-external", "production"}
_OUTPUT_SCHEMA_VERSIONS = {
    "main_results": "auto_research_main_results_v3",
    "proxy_results": "auto_research_proxy_results_v1",
    "ablation_results": "auto_research_ablation_results_v3",
    "coverage_results": "auto_research_coverage_results_v3",
    "matched_control_results": "auto_research_matched_control_results_v3",
    "activation_evidence": "auto_research_activation_evidence_v3",
    "proxy_baseline_fingerprint": "auto_research_proxy_baseline_fingerprint_v3",
    "proxy_cache_report": "auto_research_proxy_cache_report_v3",
    "full_s3_readiness": "auto_research_full_s3_readiness_v3",
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
) -> dict[str, Any]:
    """Build a deterministic command DAG from already selected execution facts."""

    if phase not in _PHASES:
        raise ValueError(f"unsupported phase: {phase}")
    if provenance_mode not in _PROVENANCE_MODES:
        raise ValueError(f"unsupported provenance mode: {provenance_mode}")
    if not command_values:
        raise ValueError(f"{adapter_id} {phase} phase requires at least one frozen command")
    outputs = _normalize_expected_outputs(expected_evidence)
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
            command["expected_outputs"] = outputs if ordinal == len(command_values) - 1 else []
        commands.append(command)
        previous_id = command["command_spec_id"]
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
        "commands": commands,
    }
    validate_phase_command_plan(plan, expected_evidence_kinds=[item["kind"] for item in outputs])
    return plan


def store_phase_command_plan(project_root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Persist a validated plan and return its immutable reference and digest."""

    validate_phase_command_plan(plan)
    reference = ContractStore(project_root).put_json(
        dict(plan),
        schema_file="phase_command_plan_v2.schema.json",
    )
    return reference, str(reference["digest"])


def validate_phase_command_plan(
    plan: Mapping[str, Any],
    *,
    expected_evidence_kinds: Iterable[str] | None = None,
) -> None:
    """Validate DAG order, dependency closure, paths, and exact output registration."""

    from .contract_store import validate_schema

    validate_schema(plan, "phase_command_plan_v2.schema.json")
    commands = list(plan["commands"])
    command_ids = [item["command_spec_id"] for item in commands]
    if len(command_ids) != len(set(command_ids)):
        raise ValueError("PhaseCommandPlan command_spec_id values must be unique")
    ordinals = [item["ordinal"] for item in commands]
    if ordinals != list(range(len(commands))):
        raise ValueError("PhaseCommandPlan ordinals must be contiguous and ordered")
    positions = {command_id: index for index, command_id in enumerate(command_ids)}
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
    output_kinds = [output["kind"] for command in commands for output in command["expected_outputs"]]
    if len(output_kinds) != len(set(output_kinds)):
        raise ValueError("PhaseCommandPlan evidence kinds must be produced exactly once")
    if expected_evidence_kinds is not None and set(output_kinds) != set(expected_evidence_kinds):
        raise ValueError("PhaseCommandPlan outputs must exactly match the phase evidence contract")


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
    command = {
        "command_spec_id": command_spec_id,
        "phase": phase,
        "ordinal": ordinal,
        "dependencies": list(payload.pop("dependencies")) if "dependencies" in payload else None,
        "argv": argv,
        "environment": {
            str(key): str(value)
            for key, value in dict(payload.pop("environment", {})).items()
        },
        "inherited_environment": sorted(
            str(value)
            for value in payload.pop("inherited_environment", ["HOME", "PATH", "TMPDIR"])
        ),
        "cwd": str(payload.pop("cwd", default_cwd)),
        "source_snapshot_hash": str(payload.pop("source_snapshot_hash", source_snapshot_hash)),
        "expected_outputs": payload.pop("expected_outputs", None),
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


def _normalize_expected_outputs(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for value in values:
        kind = str(value.get("kind") or "")
        schema_version = str(value.get("schema_version") or _OUTPUT_SCHEMA_VERSIONS.get(kind) or "")
        if not kind or not schema_version:
            raise ValueError("phase evidence outputs require kind and schema_version")
        outputs.append({"kind": kind, "schema_version": schema_version, "required": bool(value.get("required", True))})
    return outputs


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
