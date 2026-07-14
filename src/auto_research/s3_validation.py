"""Shared, side-effect-free S3 pre-commit and post-commit validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain_contracts import (
    canonical_hash,
    contract_errors,
    validate_contract,
    validate_direction_identity,
    validate_trial_result,
    validate_variant_identity,
)
from .utils import read_json, sha256_file


class S3ValidationError(ValueError):
    """Raised when an S3 draft or committed projection is not authoritative."""


_IDENTITY_FIELDS = (
    "direction_id",
    "direction_semantic_hash",
    "direction_spec_hash",
    "variant_id",
    "variant_semantic_hash",
    "variant_spec_hash",
    "attempt_id",
)

_FAILURE_ROUTES = {
    "activation_failed": ("IMPLEMENTATION_REPAIR", "REPAIR_IMPLEMENTATION"),
    "implementation_failed": ("IMPLEMENTATION_REPAIR", "REPAIR_IMPLEMENTATION"),
    "resource_paused": ("RESOURCE_PAUSED", "PAUSE_RESOURCE"),
    "oom_retry": ("RESOURCE_PAUSED", "PAUSE_RESOURCE"),
    "resource_unavailable": ("RESOURCE_PAUSED", "PAUSE_RESOURCE"),
    "integrity": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
    "safety": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
    "identity_mismatch": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
    "artifact_hash_mismatch": ("INTEGRITY_BLOCKED", "BLOCK_INTEGRITY"),
}


def validate_trial_precommit(
    *,
    project_root: Path,
    direction: dict[str, Any],
    variant: dict[str, Any],
    attempt: dict[str, Any],
    trial_spec: dict[str, Any],
    trial_result: dict[str, Any],
    state: dict[str, Any] | None = None,
    allow_pending_full_transition: bool = False,
) -> dict[str, Any]:
    """Validate a TrialResult draft without writing events or projections."""

    errors: list[str] = []
    _capture(errors, validate_direction_identity, direction)
    _capture(errors, validate_variant_identity, direction, variant)
    _capture(errors, validate_trial_result, trial_result)
    _validate_identity(errors, direction, variant, attempt, trial_result)
    _validate_canonical_preregistration(errors, project_root, attempt, trial_spec)
    _validate_attempt_contract(errors, attempt, state=state, terminal=attempt.get("state") == "METHOD_COMPLETED")
    _validate_trial_observations(errors, project_root, attempt, trial_spec, trial_result)
    _validate_execution_contracts(
        errors,
        project_root,
        attempt,
        trial_result,
        allow_pending_full_transition=allow_pending_full_transition,
    )
    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    return {
        "status": "PASS",
        "attempt_id": attempt["attempt_id"],
        "completeness": trial_result["completeness"],
        "validated_artifact_hashes": dict(trial_result["raw_artifacts"]),
    }


def validate_ledger_trial_precommit(
    *,
    project_root: Path,
    state: dict[str, Any],
    trial_result: dict[str, Any],
) -> dict[str, Any]:
    """Adapter for a ledger transaction to invoke the shared validator."""

    attempt = (state.get("attempts") or {}).get(trial_result.get("attempt_id"))
    if not isinstance(attempt, dict):
        raise S3ValidationError("TrialResult attempt does not exist in authoritative state")
    direction_state = (state.get("directions") or {}).get(attempt.get("direction_semantic_hash"))
    direction = direction_state.get("spec") if isinstance(direction_state, dict) else None
    variant = (state.get("variants") or {}).get(attempt.get("variant_spec_hash"))
    trial_spec = read_json(project_root / "plan" / "trial_spec.json", default={}) or {}
    if not isinstance(direction, dict) or not isinstance(variant, dict):
        raise S3ValidationError("attempt direction or variant contract is missing from authoritative state")
    return validate_trial_precommit(
        project_root=project_root,
        direction=direction,
        variant=variant,
        attempt=attempt,
        trial_spec=trial_spec,
        trial_result=trial_result,
        state=state,
    )


def validate_failure_precommit(
    *,
    project_root: Path,
    attempt: dict[str, Any],
    failure_class: str,
    result: dict[str, Any],
    artifact_hashes: dict[str, str],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate structured non-evaluable failure evidence before disposition."""

    errors: list[str] = []
    _validate_attempt_contract(errors, attempt, state=state, terminal=False)
    canonical_trial_spec = read_json(project_root / "plan" / "trial_spec.json", default={}) or {}
    _validate_canonical_preregistration(errors, project_root, attempt, canonical_trial_spec)
    if failure_class not in _FAILURE_ROUTES:
        errors.append(f"unsupported structured failure_class: {failure_class}")
    if result.get("method_evaluable") is True:
        errors.append("failure disposition cannot be method_evaluable")
    _validate_artifact_hashes(errors, project_root, artifact_hashes)
    explicit = result.get("failure_class") or result.get("failure_classification")
    status = str(result.get("status") or "")
    if explicit and explicit != failure_class:
        errors.append("structured failure evidence disagrees with failure_class")
    if failure_class in {"resource_paused", "oom_retry", "resource_unavailable"} and not (
        explicit == failure_class or status in {"resource_paused", "retryable_paused"}
    ):
        errors.append("resource disposition requires structured resource evidence")
    if failure_class in {"integrity", "safety", "identity_mismatch", "artifact_hash_mismatch"} and not (
        explicit == failure_class or status == "integrity_blocked"
    ):
        errors.append("integrity disposition requires structured integrity evidence")
    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    target_state, next_action = _FAILURE_ROUTES[failure_class]
    return {"status": "PASS", "target_state": target_state, "next_action": next_action}


def validate_committed_s3(
    *,
    project_root: Path,
    direction: dict[str, Any],
    variant: dict[str, Any],
    state: dict[str, Any],
    attempt: dict[str, Any],
    route_outcome: dict[str, Any],
    trial_spec: dict[str, Any],
    trial_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Audit reducer-committed S3 state using the same rules as pre-commit."""

    errors: list[str] = []
    _capture(errors, validate_direction_identity, direction)
    _capture(errors, validate_variant_identity, direction, variant)
    for key in ("direction_id", "direction_semantic_hash", "direction_spec_hash"):
        if attempt.get(key) != direction.get(key):
            errors.append(f"committed attempt {key} mismatch")
    for key in ("variant_id", "variant_semantic_hash", "variant_spec_hash"):
        if attempt.get(key) != variant.get(key):
            errors.append(f"committed attempt {key} mismatch")
    _validate_attempt_contract(errors, attempt, state=state, terminal=True)
    _validate_artifact_hashes(errors, project_root, attempt.get("artifact_hashes") or {})
    errors.extend(contract_errors(route_outcome, "route_outcome_v2.schema.json"))
    source = route_outcome.get("source") if isinstance(route_outcome.get("source"), dict) else {}
    if source.get("attempt_id") != attempt.get("attempt_id") or not source.get("event_id"):
        errors.append("RouteOutcome source must bind the committed attempt and event")
    for key in _IDENTITY_FIELDS:
        if route_outcome.get("identity", {}).get(key) != attempt.get(key):
            errors.append(f"RouteOutcome {key} mismatch")
    if route_outcome.get("artifact_hashes") != attempt.get("artifact_hashes"):
        errors.append("RouteOutcome artifact hashes mismatch")
    budget = _direction_budget(state, attempt)
    if route_outcome.get("budget_snapshot") != budget:
        errors.append("RouteOutcome budget snapshot mismatch")
    expected_idempotency_key = canonical_hash(
        {
            "source_event_id": source.get("event_id"),
            "attempt_id": attempt.get("attempt_id"),
            "lifecycle_generation": attempt.get("lifecycle_generation"),
            "next_action": route_outcome.get("next_action"),
            "reason_codes": route_outcome.get("reason_codes"),
            "budget": budget,
            "artifact_hashes": attempt.get("artifact_hashes") or {},
            "variant_spec_hash": attempt.get("variant_spec_hash"),
        }
    )
    if route_outcome.get("idempotency_key") != expected_idempotency_key:
        errors.append("RouteOutcome idempotency key mismatch")

    if attempt.get("method_evaluable"):
        if not isinstance(trial_result, dict):
            errors.append("method-evaluable attempt is missing TrialResult")
        else:
            try:
                validate_trial_precommit(
                    project_root=project_root,
                    direction=direction,
                    variant=variant,
                    attempt=attempt,
                    trial_spec=trial_spec,
                    trial_result=trial_result,
                    state=state,
                    allow_pending_full_transition=False,
                )
            except S3ValidationError as exc:
                errors.append(str(exc))
            if attempt.get("state") != "METHOD_COMPLETED":
                errors.append("method-evaluable attempt must be METHOD_COMPLETED")
            if attempt.get("phases", {}).get(trial_result.get("completeness")) != "COMPLETED":
                errors.append("committed attempt phase does not match TrialResult completeness")
            _validate_success_route(errors, attempt, route_outcome, budget)
    else:
        if trial_result is not None:
            errors.append("non-evaluable attempt cannot own TrialResult")
        failure_class = attempt.get("failure_class")
        expected = _FAILURE_ROUTES.get(str(failure_class))
        if not expected:
            errors.append("non-evaluable attempt lacks a recognized failure classification")
        elif (attempt.get("state"), route_outcome.get("next_action")) != expected:
            errors.append("failure_class, attempt state, and RouteOutcome action disagree")

    if errors:
        raise S3ValidationError("; ".join(dict.fromkeys(errors)))
    return {"status": "PASS", "attempt_id": attempt["attempt_id"], "next_action": route_outcome["next_action"]}


def _validate_identity(errors, direction, variant, attempt, trial_result) -> None:
    for key in ("direction_id", "direction_semantic_hash", "direction_spec_hash"):
        if attempt.get(key) != direction.get(key) or trial_result.get(key) != attempt.get(key):
            errors.append(f"S3 {key} identity mismatch")
    for key in ("variant_id", "variant_semantic_hash", "variant_spec_hash"):
        if attempt.get(key) != variant.get(key) or trial_result.get(key) != attempt.get(key):
            errors.append(f"S3 {key} identity mismatch")
    for key in ("attempt_id", "attempt_input_hash", "protocol_hash"):
        if trial_result.get(key) != attempt.get(key):
            errors.append(f"TrialResult {key} mismatch")


def _validate_attempt_contract(errors, attempt, *, state, terminal) -> None:
    profile = attempt.get("profile")
    kind = attempt.get("attempt_kind")
    if profile == "bootstrap":
        if kind != "bootstrap_proxy" or attempt.get("consumes_direction_budget") is not False or attempt.get("reserved_slot") is not False:
            errors.append("bootstrap attempt profile/kind/budget mapping is invalid")
    elif profile == "standard":
        if kind not in {"proxy", "full", "proxy_full"} or attempt.get("consumes_direction_budget") is not True:
            errors.append("standard attempt profile/kind/budget mapping is invalid")
        expected_reserved = not terminal or not attempt.get("method_evaluable")
        if terminal and attempt.get("method_evaluable"):
            expected_reserved = False
        if attempt.get("reserved_slot") is not expected_reserved:
            errors.append("standard attempt reservation state is invalid")
    else:
        errors.append(f"unsupported attempt profile: {profile}")
    if state is not None:
        budget = _direction_budget(state, attempt)
        if budget.get("target") != 5 or min(int(budget.get("reserved", -1)), int(budget.get("consumed", -1))) < 0:
            errors.append("direction budget values are invalid")
        if int(budget.get("reserved", 0)) + int(budget.get("consumed", 0)) > 5:
            errors.append("direction budget exceeds target")


def _validate_trial_observations(errors, project_root, attempt, trial_spec, trial_result) -> None:
    observations = trial_result.get("observations") or []
    required = set(trial_result.get("required_datasets") or [])
    observed = set(trial_result.get("observed_datasets") or [])
    if required != observed:
        errors.append("required and observed dataset coverage mismatch")
    registered_hashes = set((trial_result.get("raw_artifacts") or {}).values())
    identities = set()
    roles_by_dataset_seed: dict[tuple[str, int], set[str]] = {}
    seen_datasets = set()
    for observation in observations:
        identity = tuple(observation.get(key) for key in ("phase", "role", "dataset_id", "metric_id", "seed"))
        if identity in identities:
            errors.append("duplicate execution observation identity")
        identities.add(identity)
        seen_datasets.add(observation.get("dataset_id"))
        if observation.get("sample_manifest_hash") != attempt.get("sample_manifest_hash"):
            errors.append("observation sample_manifest_hash mismatch")
        if observation.get("evaluator_hash") != attempt.get("evaluator_hash"):
            errors.append("observation evaluator_hash mismatch")
        if observation.get("seed") not in set(attempt.get("seeds") or []):
            errors.append("observation seed was not preregistered")
        if observation.get("command_status") != "completed":
            errors.append("method-evaluable observation command_status must be completed")
        if observation.get("raw_artifact_hash") not in registered_hashes:
            errors.append("observation raw artifact is not registered by TrialResult")
        roles_by_dataset_seed.setdefault((str(observation.get("dataset_id")), int(observation.get("seed", -1))), set()).add(str(observation.get("role")))
    if seen_datasets != required:
        errors.append("observation datasets do not match required coverage")
    require_seed_coverage = bool(attempt.get("require_complete_seed_coverage", False))
    expected_seeds = set(attempt.get("seeds") or []) if require_seed_coverage else {item.get("seed") for item in observations}
    for dataset_id in required:
        for seed in expected_seeds:
            if not {"baseline", "candidate"}.issubset(roles_by_dataset_seed.get((dataset_id, int(seed)), set())):
                errors.append(f"baseline/candidate coverage missing for dataset={dataset_id}, seed={seed}")
    _validate_artifact_hashes(errors, project_root, trial_result.get("raw_artifacts") or {})


def _validate_canonical_preregistration(errors, project_root, attempt, supplied_trial_spec) -> None:
    canonical_path = project_root / "plan" / "trial_spec.json"
    canonical = read_json(canonical_path, default={}) or {}
    if not canonical_path.is_file() or not isinstance(canonical, dict) or not canonical:
        errors.append("canonical plan/trial_spec.json is missing")
        return
    if supplied_trial_spec != canonical:
        errors.append("supplied TrialSpec differs from canonical plan/trial_spec.json")
    protocol = canonical.get("protocol") if isinstance(canonical.get("protocol"), dict) else canonical
    sample_manifest = canonical.get("sample_manifest") if isinstance(canonical.get("sample_manifest"), dict) else {"datasets": canonical.get("datasets") or []}
    if canonical_hash(protocol) != attempt.get("protocol_hash"):
        errors.append("canonical TrialSpec protocol hash mismatch")
    if canonical_hash(sample_manifest) != attempt.get("sample_manifest_hash"):
        errors.append("canonical TrialSpec sample manifest hash mismatch")
    if attempt.get("trial_spec_hash") is not None and attempt.get("trial_spec_hash") != canonical_hash(canonical):
        errors.append("canonical TrialSpec hash mismatch")
    datasets = _dataset_ids(sample_manifest.get("datasets") or canonical.get("datasets") or [])
    default_terminal_phase = "proxy" if attempt.get("attempt_kind") in {"proxy", "bootstrap_proxy"} else "full"
    phases = sorted(str(item) for item in (protocol.get("required_phases") or [default_terminal_phase]))
    roles = sorted(str(item) for item in (protocol.get("required_roles") or ["baseline", "candidate"]))
    terminal_phases = sorted(str(item) for item in (protocol.get("terminal_method_phases") or phases))
    expected = {
        "required_datasets": datasets,
        "required_phases": phases,
        "terminal_method_phases": terminal_phases,
        "required_roles": roles,
        "require_complete_seed_coverage": bool(protocol.get("require_complete_seed_coverage", protocol.get("require_seed_coverage", False))),
    }
    for key, value in expected.items():
        if key in attempt and attempt.get(key) != value:
            errors.append(f"attempt preregistration {key} mismatch")


def _validate_execution_contracts(errors, project_root, attempt, trial_result, *, allow_pending_full_transition) -> None:
    completeness = trial_result.get("completeness")
    phases = attempt.get("phases") or {}
    state = attempt.get("state")
    observed_phases = {item.get("phase") for item in trial_result.get("observations") or []}
    if observed_phases != {completeness}:
        errors.append("observation phases must exactly match TrialResult completeness")
    if completeness == "proxy":
        if attempt.get("attempt_kind") not in {"proxy", "bootstrap_proxy"}:
            errors.append("attempt kind does not permit terminal proxy outcome")
        if state not in {"PROXY_RUNNING", "METHOD_COMPLETED"}:
            errors.append("proxy TrialResult requires proxy execution state")
    elif completeness == "full":
        if attempt.get("attempt_kind") not in {"full", "proxy_full"}:
            errors.append("attempt kind does not permit full outcome")
        if state not in {"FULL_RUNNING", "METHOD_COMPLETED"} and not (allow_pending_full_transition and state == "PROXY_RUNNING"):
            errors.append("full TrialResult requires full execution state")
    if state == "METHOD_COMPLETED" and phases.get(completeness) != "COMPLETED":
        errors.append("METHOD_COMPLETED phase is inconsistent with TrialResult")
    protocol = read_json(project_root / "plan" / "trial_spec.json", default={}).get("protocol") or {}
    if protocol.get("requires_activation_evidence") is True:
        activation_paths = [
            project_root / "experiment" / "results" / "c2c_proxy_decision_report.json",
            project_root / "experiment" / "results" / "full_s3_readiness_report.json",
        ]
        if not any(path.is_file() for path in activation_paths):
            errors.append("protocol requires activation evidence but none was produced")
    if completeness == "full" and protocol.get("requires_full_readiness_evidence") is True:
        if not (project_root / "experiment" / "results" / "full_s3_readiness_report.json").is_file():
            errors.append("protocol requires full-readiness evidence but none was produced")
    _validate_proxy_artifacts(errors, project_root, attempt, completeness)


def _validate_proxy_artifacts(errors, project_root, attempt, completeness) -> None:
    report_path = project_root / "experiment" / "results" / "c2c_proxy_decision_report.json"
    if report_path.exists():
        report = read_json(report_path, default={}) or {}
        try:
            validate_contract(report, "c2c_proxy_decision_report.schema.json")
        except ValueError as exc:
            errors.append(f"proxy decision contract invalid: {exc}")
        if report.get("schema_version") != "c2c_proxy_decision_report_v1":
            errors.append("proxy decision contract schema version mismatch")
        checks = report.get("static_checks") if isinstance(report.get("static_checks"), dict) else {}
        if checks.get("patch_gate_passed") is not True or checks.get("has_executable_change") is not True:
            errors.append("proxy decision lacks patch/readiness evidence")
        if checks.get("activation_smoke_passed") is not True:
            errors.append("proxy decision lacks activation evidence")
        allowed = {"proxy_pass", "neutral_proxy_full_s3"} if completeness == "full" else {"proxy_pass"}
        if report.get("decision") not in allowed:
            errors.append("proxy decision does not authorize method outcome")
    readiness_path = project_root / "experiment" / "results" / "full_s3_readiness_report.json"
    if completeness == "full" and readiness_path.exists():
        readiness = read_json(readiness_path, default={}) or {}
        if readiness.get("status") not in {"ready", "passed"}:
            errors.append("full S3 readiness did not pass")
        activation = readiness.get("activation_smoke") if isinstance(readiness.get("activation_smoke"), dict) else {}
        if activation and activation.get("status") != "passed":
            errors.append("full S3 activation readiness did not pass")
    if attempt.get("profile") == "bootstrap":
        completion_path = project_root / "experiment" / "results" / "bootstrap_proxy_completion.json"
        completion = read_json(completion_path, default={}) or {}
        if not completion_path.exists() or completion.get("bootstrap_proxy_complete") is not True:
            errors.append("bootstrap completion evidence is missing or unverified")
        if completion.get("full_train_executed") is True or completion.get("full_eval_executed") is True:
            errors.append("bootstrap completion cannot include full execution")


def _validate_artifact_hashes(errors, project_root, artifact_hashes) -> None:
    root = project_root.resolve()
    for relative_path, expected_hash in artifact_hashes.items():
        path = (project_root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"raw artifact escapes project root: {relative_path}")
            continue
        if not path.is_file():
            errors.append(f"raw artifact is missing: {relative_path}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"raw artifact hash mismatch: {relative_path}")


def _validate_success_route(errors, attempt, route, budget) -> None:
    action = route.get("next_action")
    if attempt.get("profile") == "bootstrap":
        if action != "FINISH_RUN":
            errors.append("verified bootstrap proxy must route FINISH_RUN")
    elif int(budget.get("consumed", 0)) < 5 and action != "PROPOSE_NEXT_VARIANT":
        errors.append("standard outcome before budget completion must propose next variant")
    elif int(budget.get("consumed", 0)) == 5 and action not in {"FINISH_DIRECTION", "START_NEW_DIRECTION"}:
        errors.append("fifth standard outcome must close or exhaust direction")


def _direction_budget(state, attempt) -> dict[str, Any]:
    return dict(((state.get("directions") or {}).get(attempt.get("direction_semantic_hash")) or {}).get("budget") or {})


def _dataset_ids(items: list[Any]) -> list[str]:
    values = []
    for item in items:
        value = item.get("name") if isinstance(item, dict) else item
        if value:
            values.append(str(value))
    return sorted(values)


def _capture(errors: list[str], function, *args) -> None:
    try:
        function(*args)
    except ValueError as exc:
        errors.append(str(exc))
