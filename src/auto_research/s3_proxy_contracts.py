"""Deterministic C2C S3 proxy decision contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .utils import now_utc, read_json, read_yaml, sha256_file


C2C_PROXY_BASELINE_FINGERPRINT_SCHEMA_VERSION = "c2c_proxy_baseline_fingerprint_v1"
C2C_PROXY_CACHE_REPORT_SCHEMA_VERSION = "c2c_proxy_cache_report_v1"
C2C_EFFECTIVE_PROXY_POLICY_SCHEMA_VERSION = "c2c_effective_proxy_policy_v1"
C2C_PROXY_DECISION_REPORT_SCHEMA_VERSION = "c2c_proxy_decision_report_v1"
C2C_FULL_S3_WORTHINESS_SCHEMA_VERSION = "c2c_full_s3_worthiness_v1"
C2C_PROXY_CALIBRATION_POLICY_SCHEMA_VERSION = "c2c_proxy_calibration_policy_v1"


def build_c2c_proxy_baseline_fingerprint(
    *,
    project_root: Path,
    config: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    proxy_config: dict[str, Any] | None = None,
    run_spec: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    env_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable fingerprint for the paired proxy baseline cache."""

    config = config or {}
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    proxy_config = proxy_config if isinstance(proxy_config, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    run_spec = run_spec if isinstance(run_spec, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}

    snapshot_path = str(c2c_cfg.get("snapshot_path") or project_root / "external" / "c2c_snapshot")
    snapshot_root = Path(snapshot_path)
    if not snapshot_root.is_absolute():
        snapshot_root = project_root / snapshot_root
    dataset_signature = _dataset_signature(config, proxy_config, run_spec)
    evaluator_hashes = _evaluator_file_hashes(snapshot_root, run_spec)
    s2_5_locks = _s2_5_lock_hashes(project_root)
    env_inputs = {
        "python": str(c2c_cfg.get("env_python") or ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_policy_hash": _stable_hash((config.get("experiment") or {}).get("gpu_policy") or {}),
    }
    if isinstance(env_signature, dict):
        env_inputs.update({str(key): value for key, value in env_signature.items()})

    inputs = {
        "snapshot_path": _rel_or_abs(project_root, snapshot_root),
        "snapshot_manifest_hash": _snapshot_manifest_hash(snapshot_root),
        "plan_hash": _plan_hash(project_root, plan),
        "execution_hash": _stable_hash(execution),
        "proxy_config_hash": _stable_hash(proxy_config),
        "eval_recipe_hash": _eval_recipe_hash(run_spec, proxy_config),
        "dataset_signature": dataset_signature,
        "baseline_model_signature": {
            "model_map_hash": _stable_hash(c2c_cfg.get("model_map") or {}),
            "baseline_config_hash": _stable_hash(c2c_cfg.get("baseline") or execution.get("baseline") or {}),
        },
        "evaluator_file_hashes": evaluator_hashes,
        "s2_5_locks": s2_5_locks,
        "env_signature": env_inputs,
        "candidate_signature": {
            "candidate_id": str(candidate.get("id") or candidate.get("variant_id") or ""),
            "variant_fingerprint": str(candidate.get("variant_fingerprint") or ""),
            "config_overrides_hash": _stable_hash((run_spec.get("config_overrides") or {}) or _candidate_config_overrides(candidate)),
        },
    }
    return {
        "schema_version": C2C_PROXY_BASELINE_FINGERPRINT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "fingerprint_hash": _stable_hash(inputs),
        "inputs": inputs,
    }


def build_c2c_proxy_cache_report(
    *,
    expected_fingerprint: dict[str, Any],
    actual_fingerprint: dict[str, Any] | None,
    baseline_cache_path: str | Path,
    baseline_cache_exists: bool | None = None,
    require_cache_fingerprint_match: bool = True,
) -> dict[str, Any]:
    """Decide whether the cached paired proxy baseline may be reused."""

    expected_hash = str((expected_fingerprint or {}).get("fingerprint_hash") or "")
    actual_hash = str((actual_fingerprint or {}).get("fingerprint_hash") or "")
    cache_path = Path(baseline_cache_path)
    cache_exists = cache_path.exists() if baseline_cache_exists is None else bool(baseline_cache_exists)
    if not cache_exists:
        status = "missing"
        reason = "baseline_cache_missing"
        action = "rerun_baseline"
    elif not require_cache_fingerprint_match:
        status = "hit"
        reason = "fingerprint_match_not_required"
        action = "reuse"
    elif not actual_hash:
        status = "invalidated"
        reason = "baseline_fingerprint_missing"
        action = "rerun_baseline"
    elif actual_hash != expected_hash:
        status = "invalidated"
        reason = _fingerprint_mismatch_reason(expected_fingerprint, actual_fingerprint or {})
        action = "rerun_baseline"
    else:
        status = "hit"
        reason = "fingerprint_hash_match"
        action = "reuse"
    return {
        "schema_version": C2C_PROXY_CACHE_REPORT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "cache_status": status,
        "reason": reason,
        "expected_fingerprint_hash": expected_hash,
        "actual_fingerprint_hash": actual_hash or None,
        "baseline_cache_path": str(baseline_cache_path),
        "action": action,
    }


def build_c2c_proxy_calibration_policy(
    *,
    project_root: Path | None = None,
    config: dict[str, Any] | None = None,
    direction_fingerprint: dict[str, Any] | None = None,
    calibration_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic policy recommendation from proxy/full calibration history."""

    config = config or {}
    proxy_cfg = _proxy_cfg(config)
    adaptive_cfg = proxy_cfg.get("adaptive_policy") if isinstance(proxy_cfg.get("adaptive_policy"), dict) else {}
    history = calibration_history if isinstance(calibration_history, dict) else _load_calibration_history(project_root)
    summary = history.get("summary") if isinstance(history.get("summary"), dict) else {}
    method_feedback = summary.get("method_feedback") if isinstance(summary.get("method_feedback"), dict) else {}
    direction_fingerprint = direction_fingerprint if isinstance(direction_fingerprint, dict) else {}
    history_count = int(summary.get("candidate_count") or _history_candidate_count(history))
    min_history = int(adaptive_cfg.get("min_history_for_adjustment", 3) or 0)
    fp_threshold = float(adaptive_cfg.get("false_positive_rate_threshold", 0.5) or 0.0)
    mechanism_rate = _matched_false_positive_rate(
        method_feedback.get("risky_mechanisms") or [],
        direction_fingerprint,
        keys=("mechanism_type", "mechanism_axis"),
        fallback=float(summary.get("proxy_false_positive_rate") or 0.0),
    )
    integration_rate = _matched_false_positive_rate(
        method_feedback.get("risky_integration_points") or [],
        direction_fingerprint,
        keys=("integration_point",),
        fallback=0.0,
    )
    dataset_risk = _dataset_misprediction_risk(summary.get("dataset_error_summary") or {}, method_feedback.get("risky_datasets") or [])
    adjustments_active = bool(adaptive_cfg.get("enabled", True)) and history_count >= min_history and (
        mechanism_rate >= fp_threshold
        or integration_rate >= fp_threshold
        or any(value >= fp_threshold for value in dataset_risk.values())
    )
    reason_codes: list[str] = []
    if history_count < min_history:
        reason_codes.append("insufficient_proxy_calibration_history")
    if mechanism_rate >= fp_threshold:
        reason_codes.append("historical_proxy_full_false_positive_rate_high")
    elif mechanism_rate >= max(0.25, fp_threshold * 0.5):
        reason_codes.append("historical_proxy_full_false_positive_rate_medium")
    for dataset, risk in sorted(dataset_risk.items()):
        if risk >= fp_threshold:
            reason_codes.append(f"{dataset}_proxy_regression_risk")
    recommended_min_delta = proxy_cfg.get("min_proxy_mean_delta")
    recommended_max_regression = proxy_cfg.get("max_proxy_dataset_regression")
    if adjustments_active:
        if adaptive_cfg.get("tighten_min_proxy_delta_to") is not None:
            recommended_min_delta = max(float(proxy_cfg.get("min_proxy_mean_delta") or 0.0), float(adaptive_cfg["tighten_min_proxy_delta_to"]))
        if adaptive_cfg.get("tighten_max_dataset_regression_to") is not None:
            recommended_max_regression = min(
                float(proxy_cfg.get("max_proxy_dataset_regression") or adaptive_cfg["tighten_max_dataset_regression_to"]),
                float(adaptive_cfg["tighten_max_dataset_regression_to"]),
            )
    neutral_allowed = bool(proxy_cfg.get("allow_neutral_proxy_full_s3", True))
    block_fp = ((proxy_cfg.get("full_s3_worthiness") or {}) if isinstance(proxy_cfg.get("full_s3_worthiness"), dict) else {}).get("block_if_false_positive_rate_above", 0.6)
    if mechanism_rate > float(block_fp or 1.0):
        neutral_allowed = False
        reason_codes.append("neutral_proxy_blocked_by_false_positive_history")
    payload = {
        "schema_version": C2C_PROXY_CALIBRATION_POLICY_SCHEMA_VERSION,
        "created_at": now_utc(),
        "history_path": "experiment/results/proxy_calibration.json" if project_root else None,
        "history_count": history_count,
        "min_history_for_adjustment": min_history,
        "adjustments_active": adjustments_active,
        "mechanism_false_positive_rate": round(mechanism_rate, 4),
        "integration_point_false_positive_rate": round(integration_rate, 4),
        "dataset_misprediction_risk": dataset_risk,
        "recommended_min_proxy_mean_delta": recommended_min_delta,
        "recommended_max_proxy_dataset_regression": recommended_max_regression,
        "neutral_proxy_allowed": neutral_allowed,
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }
    payload["policy_hash"] = _stable_hash({key: value for key, value in payload.items() if key not in {"created_at", "policy_hash"}})
    return payload


def build_c2c_effective_proxy_policy(
    *,
    static_proxy_config: dict[str, Any],
    calibration_policy: dict[str, Any] | None = None,
    variant_scorecard: dict[str, Any] | None = None,
    next_variant: dict[str, Any] | None = None,
    patch_gate_report: dict[str, Any] | None = None,
    direction_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine static proxy config and calibration into an executable proxy policy."""

    static_proxy_config = static_proxy_config if isinstance(static_proxy_config, dict) else {}
    calibration_policy = calibration_policy if isinstance(calibration_policy, dict) else {}
    static_policy = {
        "min_proxy_mean_delta": static_proxy_config.get("min_proxy_mean_delta"),
        "soft_proxy_mean_delta": static_proxy_config.get("soft_proxy_mean_delta"),
        "max_proxy_dataset_regression": static_proxy_config.get("max_proxy_dataset_regression"),
        "soft_max_proxy_dataset_regression": static_proxy_config.get("soft_max_proxy_dataset_regression"),
        "allow_neutral_proxy_full_s3": bool(static_proxy_config.get("allow_neutral_proxy_full_s3", True)),
        "neutral_proxy_min_delta": static_proxy_config.get("neutral_proxy_min_delta", -0.1),
        "neutral_proxy_max_dataset_regression": static_proxy_config.get("neutral_proxy_max_dataset_regression", 0.25),
    }
    adjustments = {
        "mechanism_false_positive_rate": calibration_policy.get("mechanism_false_positive_rate", 0.0),
        "integration_point_false_positive_rate": calibration_policy.get("integration_point_false_positive_rate", 0.0),
        "dataset_misprediction_risk": calibration_policy.get("dataset_misprediction_risk") or {},
        "recommended_min_proxy_mean_delta": calibration_policy.get("recommended_min_proxy_mean_delta", static_policy["min_proxy_mean_delta"]),
        "recommended_max_proxy_dataset_regression": calibration_policy.get("recommended_max_proxy_dataset_regression", static_policy["max_proxy_dataset_regression"]),
        "neutral_proxy_allowed": calibration_policy.get("neutral_proxy_allowed", static_policy["allow_neutral_proxy_full_s3"]),
        "adjustments_active": bool(calibration_policy.get("adjustments_active")),
    }
    effective = dict(static_policy)
    if adjustments["adjustments_active"]:
        if adjustments["recommended_min_proxy_mean_delta"] is not None and static_policy["min_proxy_mean_delta"] is not None:
            effective["min_proxy_mean_delta"] = max(float(static_policy["min_proxy_mean_delta"]), float(adjustments["recommended_min_proxy_mean_delta"]))
        if adjustments["recommended_max_proxy_dataset_regression"] is not None and static_policy["max_proxy_dataset_regression"] is not None:
            effective["max_proxy_dataset_regression"] = min(float(static_policy["max_proxy_dataset_regression"]), float(adjustments["recommended_max_proxy_dataset_regression"]))
        effective["allow_neutral_proxy_full_s3"] = bool(static_policy["allow_neutral_proxy_full_s3"] and adjustments["neutral_proxy_allowed"])
    worth_cfg = static_proxy_config.get("full_s3_worthiness") if isinstance(static_proxy_config.get("full_s3_worthiness"), dict) else {}
    effective["full_s3_worthiness_min_score"] = float(worth_cfg.get("min_score", 0.60) or 0.0)
    effective["neutral_proxy_budget_per_direction"] = int(worth_cfg.get("neutral_proxy_budget_per_direction", 1) or 0)
    reason_codes = list(calibration_policy.get("reason_codes") or [])
    if variant_scorecard and isinstance(variant_scorecard, dict):
        reason_codes.append("variant_scorecard_available")
    if patch_gate_report and isinstance(patch_gate_report, dict):
        reason_codes.append("patch_gate_available")
    payload = {
        "schema_version": C2C_EFFECTIVE_PROXY_POLICY_SCHEMA_VERSION,
        "created_at": now_utc(),
        "static_policy": static_policy,
        "calibration_adjustments": adjustments,
        "effective_policy": effective,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "inputs": {
            "selected_variant_id": (variant_scorecard or {}).get("selected_variant_id") if isinstance(variant_scorecard, dict) else None,
            "next_variant_id": (next_variant or {}).get("id") if isinstance(next_variant, dict) else None,
            "patch_gate": (patch_gate_report or {}).get("gate") if isinstance(patch_gate_report, dict) else None,
            "direction_fingerprint": (direction_fingerprint or {}).get("fingerprint_hash") or (direction_fingerprint or {}).get("direction_fingerprint")
            if isinstance(direction_fingerprint, dict)
            else None,
        },
    }
    payload["policy_hash"] = _stable_hash({key: value for key, value in payload.items() if key not in {"created_at", "policy_hash"}})
    return payload


def build_c2c_full_s3_worthiness_score(
    *,
    candidate: dict[str, Any] | None = None,
    proxy_screen: dict[str, Any] | None = None,
    effective_proxy_policy: dict[str, Any] | None = None,
    variant_scorecard: dict[str, Any] | None = None,
    patch_gate_report: dict[str, Any] | None = None,
    novelty_audit: dict[str, Any] | None = None,
    calibration_policy: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    neutral_proxy_budget_remaining: bool = True,
) -> dict[str, Any]:
    """Score whether a neutral proxy is worth spending full S3 budget on."""

    candidate = candidate if isinstance(candidate, dict) else {}
    proxy_screen = proxy_screen if isinstance(proxy_screen, dict) else {}
    effective = (effective_proxy_policy or {}).get("effective_policy") if isinstance(effective_proxy_policy, dict) else {}
    effective = effective if isinstance(effective, dict) else {}
    patch_score = _patch_gate_score(patch_gate_report, candidate)
    activation_score = _activation_smoke_score(proxy_screen.get("activation_smoke") or candidate.get("activation_smoke"))
    mean_delta = _number(proxy_screen.get("proxy_delta_vs_comparison_baseline", proxy_screen.get("proxy_delta_vs_proxy_baseline", proxy_screen.get("proxy_delta_vs_baseline"))))
    worst_regression = _number(proxy_screen.get("proxy_worst_dataset_regression"), 0.0) or 0.0
    proxy_delta_score = _proxy_delta_component(mean_delta, effective)
    no_regression_score = _dataset_regression_component(worst_regression, effective)
    s2_score = _s2_variant_score(variant_scorecard, candidate)
    novelty_score = _novelty_score(novelty_audit, variant_scorecard, candidate)
    false_positive_risk = float((calibration_policy or {}).get("mechanism_false_positive_rate") or 0.0) if isinstance(calibration_policy, dict) else 0.0
    resource_cost_risk = _resource_cost_risk(config or {}, candidate)
    components = {
        "patch_gate_score": patch_score,
        "activation_smoke_score": activation_score,
        "proxy_delta_score": proxy_delta_score,
        "no_dataset_regression_score": no_regression_score,
        "s2_variant_score": s2_score,
        "novelty_score": novelty_score,
        "calibration_false_positive_risk": round(false_positive_risk, 4),
        "resource_cost_risk": round(resource_cost_risk, 4),
    }
    score = (
        0.20 * patch_score
        + 0.20 * activation_score
        + 0.20 * proxy_delta_score
        + 0.15 * no_regression_score
        + 0.10 * s2_score
        + 0.10 * novelty_score
        - 0.15 * false_positive_risk
        - 0.10 * resource_cost_risk
    )
    score = round(_clamp(score), 4)
    min_score = float(effective.get("full_s3_worthiness_min_score", 0.60) or 0.0)
    decision = "run_full_s3" if score >= min_score and neutral_proxy_budget_remaining else "do_not_run_full_s3"
    reason_codes = []
    if mean_delta is not None and mean_delta >= float(effective.get("neutral_proxy_min_delta", -0.1) or 0.0):
        reason_codes.append("neutral_proxy_but_activation_signal_present" if activation_score > 0.0 else "neutral_proxy")
    if no_regression_score >= 0.8:
        reason_codes.append("low_dataset_regression")
    if s2_score >= 0.6:
        reason_codes.append("planner_score_above_threshold")
    if false_positive_risk >= 0.5:
        reason_codes.append("historical_false_positive_risk")
    if not neutral_proxy_budget_remaining:
        reason_codes.append("neutral_proxy_budget_exhausted")
    return {
        "schema_version": C2C_FULL_S3_WORTHINESS_SCHEMA_VERSION,
        "created_at": now_utc(),
        "candidate_id": str(candidate.get("id") or candidate.get("variant_id") or ""),
        "score": score,
        "threshold": min_score,
        "components": components,
        "neutral_proxy_budget_remaining": bool(neutral_proxy_budget_remaining),
        "decision": decision,
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }


def build_c2c_proxy_decision_report(
    *,
    candidate: dict[str, Any] | None = None,
    proxy_screen: dict[str, Any] | None = None,
    baseline_fingerprint: dict[str, Any] | None = None,
    effective_proxy_policy: dict[str, Any] | None = None,
    patch_gate_report: dict[str, Any] | None = None,
    planner_gate_report: dict[str, Any] | None = None,
    variant_scorecard: dict[str, Any] | None = None,
    full_s3_worthiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source-of-truth deterministic routing decision for S3 proxy."""

    candidate = candidate if isinstance(candidate, dict) else {}
    proxy_screen = proxy_screen if isinstance(proxy_screen, dict) else {}
    effective_proxy_policy = effective_proxy_policy if isinstance(effective_proxy_policy, dict) else {}
    effective = effective_proxy_policy.get("effective_policy") if isinstance(effective_proxy_policy.get("effective_policy"), dict) else {}
    proxy_metrics = proxy_screen.get("metrics") if isinstance(proxy_screen.get("metrics"), dict) else {}
    baseline_metrics = (
        proxy_screen.get("proxy_baseline")
        if isinstance(proxy_screen.get("proxy_baseline"), dict)
        else proxy_screen.get("baseline_metrics") if isinstance(proxy_screen.get("baseline_metrics"), dict) else {}
    )
    deltas = _proxy_deltas(proxy_metrics, baseline_metrics, proxy_screen)
    static_checks = {
        "patch_gate_passed": _patch_gate_passed(patch_gate_report, candidate),
        "no_eval_code_change": "evaluation_code_changed" not in set((proxy_screen.get("patch_risk") or {}).get("risk_labels") or []),
        "has_executable_change": bool((proxy_screen.get("signals") or {}).get("has_executable_change", candidate.get("has_executable_change", True))),
        "activation_smoke_passed": _activation_smoke_passed(proxy_screen.get("activation_smoke") or candidate.get("activation_smoke")),
    }
    decision, route_hint, failure_class, reason_codes = _proxy_decision(proxy_screen, deltas, effective, static_checks, full_s3_worthiness)
    payload = {
        "schema_version": C2C_PROXY_DECISION_REPORT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "candidate_id": str(candidate.get("id") or candidate.get("variant_id") or ""),
        "variant_id": str(candidate.get("variant_id") or candidate.get("id") or ""),
        "variant_fingerprint": str(candidate.get("variant_fingerprint") or ((candidate.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(candidate.get("s2_variant"), dict) else "")),
        "baseline_fingerprint_hash": str((baseline_fingerprint or {}).get("fingerprint_hash") or ""),
        "effective_policy_hash": str(effective_proxy_policy.get("policy_hash") or ""),
        "proxy_metrics": _metric_view(proxy_metrics),
        "paired_baseline_metrics": _metric_view(baseline_metrics),
        "deltas": deltas,
        "static_checks": static_checks,
        "decision": decision,
        "failure_class": failure_class,
        "route_hint": route_hint,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "policy_thresholds": {
            "min_proxy_mean_delta": effective.get("min_proxy_mean_delta"),
            "max_proxy_dataset_regression": effective.get("max_proxy_dataset_regression"),
            "neutral_proxy_min_delta": effective.get("neutral_proxy_min_delta"),
            "neutral_proxy_max_dataset_regression": effective.get("neutral_proxy_max_dataset_regression"),
            "full_s3_worthiness_min_score": effective.get("full_s3_worthiness_min_score"),
        },
        "planner_gate": {
            "gate": (planner_gate_report or {}).get("gate") if isinstance(planner_gate_report, dict) else None,
            "selected_variant_id": (planner_gate_report or {}).get("selected_variant_id") if isinstance(planner_gate_report, dict) else None,
        },
        "variant_scorecard": {
            "selected_variant_id": (variant_scorecard or {}).get("selected_variant_id") if isinstance(variant_scorecard, dict) else None,
        },
    }
    if isinstance(full_s3_worthiness, dict) and full_s3_worthiness:
        payload["full_s3_worthiness"] = {
            "path": "experiment/results/c2c_full_s3_worthiness.json",
            "score": full_s3_worthiness.get("score"),
            "decision": full_s3_worthiness.get("decision"),
            "threshold": full_s3_worthiness.get("threshold"),
        }
    return payload


def _proxy_decision(
    proxy_screen: dict[str, Any],
    deltas: dict[str, Any],
    effective: dict[str, Any],
    static_checks: dict[str, Any],
    worthiness: dict[str, Any] | None,
) -> tuple[str, str, str | None, list[str]]:
    status = str(proxy_screen.get("status") or "")
    reason_codes: list[str] = []
    if status in {"resource_retry", "baseline_blocked", "blocked"}:
        return "blocked", "block_resource", "proxy_resource_or_baseline_blocked", [status or "blocked"]
    if status == "rejected":
        return "proxy_rejected", "return_s2", "proxy_threshold_rejected", ["proxy_screen_rejected"]
    if status in {"failed", "repairable_proxy_risk"}:
        route = "repair_s2_5" if "patch" in str(proxy_screen.get("repair_mode") or proxy_screen.get("failure_category") or "") else "return_s2"
        return "proxy_repairable", route, str(proxy_screen.get("failure_category") or proxy_screen.get("repair_mode") or "proxy_repairable"), ["proxy_screen_repairable"]
    if status and status != "passed":
        return "blocked", "block_resource", f"proxy_status_{status}", [f"proxy_status_{status}"]
    mean_delta = _number(deltas.get("mean_delta"))
    worst = _number(deltas.get("worst_dataset_regression"), 0.0) or 0.0
    min_delta = _number(effective.get("min_proxy_mean_delta"), -0.3)
    max_regression = _number(effective.get("max_proxy_dataset_regression"), 1.5)
    if mean_delta is None:
        if proxy_screen.get("require_proxy_metrics") is False:
            return "proxy_pass", "run_full_s3", None, ["proxy_metrics_not_required"]
        return "proxy_rejected", "return_s2", "missing_proxy_mean_delta", ["missing_proxy_mean_delta"]
    if min_delta is not None and mean_delta < min_delta:
        return "proxy_rejected", "return_s2", "proxy_mean_delta_below_effective_policy", ["proxy_mean_delta_below_effective_policy"]
    if max_regression is not None and worst > max_regression:
        return "proxy_rejected", "return_s2", "proxy_dataset_regression_above_effective_policy", ["proxy_dataset_regression_above_effective_policy"]
    if not static_checks.get("patch_gate_passed"):
        return "proxy_repairable", "repair_s2_5", "patch_gate_not_passed", ["patch_gate_not_passed"]
    if not static_checks.get("no_eval_code_change") or not static_checks.get("has_executable_change"):
        return "proxy_repairable", "repair_s2_5", "static_patch_gate_failed", ["static_patch_gate_failed"]
    neutral = _is_neutral_proxy(mean_delta, worst, effective)
    if neutral:
        reason_codes.append("neutral_proxy")
        if not isinstance(worthiness, dict) or not worthiness:
            return "blocked", "return_s2", "neutral_proxy_missing_worthiness", reason_codes + ["neutral_proxy_missing_worthiness"]
        if worthiness.get("decision") == "run_full_s3":
            return "neutral_proxy_full_s3", "run_full_s3", None, reason_codes + ["full_s3_worthiness_passed"]
        failure = "neutral_proxy_budget_exhausted" if worthiness.get("neutral_proxy_budget_remaining") is False else "full_s3_not_worthy"
        return "proxy_rejected", "return_s2", failure, reason_codes + [failure]
    return "proxy_pass", "run_full_s3", None, ["proxy_thresholds_passed"]


def _is_neutral_proxy(mean_delta: float, worst_regression: float, effective: dict[str, Any]) -> bool:
    if not bool(effective.get("allow_neutral_proxy_full_s3", True)):
        return False
    neutral_min = _number(effective.get("neutral_proxy_min_delta"), -0.1)
    neutral_max_regression = _number(effective.get("neutral_proxy_max_dataset_regression"), 0.25)
    return (
        mean_delta < 0.0
        and (neutral_min is None or mean_delta >= neutral_min)
        and (neutral_max_regression is None or worst_regression <= neutral_max_regression)
    )


def _proxy_deltas(metrics: dict[str, Any], baseline: dict[str, Any], proxy_screen: dict[str, Any]) -> dict[str, Any]:
    if proxy_screen.get("proxy_delta_vs_comparison_baseline") is not None:
        mean_delta = proxy_screen.get("proxy_delta_vs_comparison_baseline")
    elif proxy_screen.get("proxy_delta_vs_proxy_baseline") is not None:
        mean_delta = proxy_screen.get("proxy_delta_vs_proxy_baseline")
    elif metrics.get("mean") is not None and baseline.get("mean") is not None:
        mean_delta = round(float(metrics["mean"]) - float(baseline["mean"]), 4)
    else:
        mean_delta = None
    dataset_deltas = proxy_screen.get("proxy_dataset_deltas") if isinstance(proxy_screen.get("proxy_dataset_deltas"), dict) else {}
    if not dataset_deltas:
        metric_datasets = metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {}
        baseline_datasets = baseline.get("datasets") if isinstance(baseline.get("datasets"), dict) else {}
        dataset_deltas = {
            str(dataset): round(float(value) - float(baseline_datasets[dataset]), 4)
            for dataset, value in metric_datasets.items()
            if dataset in baseline_datasets and _is_number(value) and _is_number(baseline_datasets[dataset])
        }
    regressions = proxy_screen.get("proxy_dataset_regressions") if isinstance(proxy_screen.get("proxy_dataset_regressions"), dict) else {}
    if not regressions and dataset_deltas:
        regressions = {dataset: round(max(0.0, -float(delta)), 4) for dataset, delta in dataset_deltas.items()}
    worst = proxy_screen.get("proxy_worst_dataset_regression")
    if worst is None:
        worst = max((float(value) for value in regressions.values()), default=0.0)
    return {
        "mean_delta": mean_delta,
        "worst_dataset_regression": round(float(worst), 4) if _is_number(worst) else None,
        "dataset_deltas": dataset_deltas,
    }


def _metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": metrics.get("mean"),
        "datasets": metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {},
    }


def _patch_gate_passed(patch_gate_report: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    if isinstance(patch_gate_report, dict) and patch_gate_report:
        return str(patch_gate_report.get("gate") or "").lower() == "pass"
    code_patch = candidate.get("code_patch") if isinstance(candidate.get("code_patch"), dict) else {}
    return str(code_patch.get("status") or "ok") == "ok"


def _patch_gate_score(patch_gate_report: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if _patch_gate_passed(patch_gate_report, candidate):
        return 1.0
    if not patch_gate_report:
        return 0.8 if candidate else 0.6
    return 0.0


def _activation_smoke_passed(smoke: Any) -> bool:
    if not isinstance(smoke, dict) or not smoke:
        return True
    return smoke.get("status") in {"passed", "ok", "skipped"} or bool(smoke.get("hard_gate_overridden"))


def _activation_smoke_score(smoke: Any) -> float:
    if not isinstance(smoke, dict) or not smoke:
        return 0.6
    if smoke.get("status") in {"passed", "ok"}:
        return 1.0
    if smoke.get("hard_gate_overridden"):
        return 0.8
    if smoke.get("status") == "skipped":
        return 0.6
    return 0.0


def _proxy_delta_component(mean_delta: float | None, effective: dict[str, Any]) -> float:
    if mean_delta is None:
        return 0.0
    min_delta = _number(effective.get("min_proxy_mean_delta"), -0.3) or -0.3
    neutral_min = _number(effective.get("neutral_proxy_min_delta"), -0.1) or -0.1
    if mean_delta >= 0.1:
        return 1.0
    if mean_delta >= 0.0:
        return 0.8
    if mean_delta >= neutral_min:
        return 0.55
    if mean_delta >= min_delta:
        return 0.35
    return 0.0


def _dataset_regression_component(worst_regression: float, effective: dict[str, Any]) -> float:
    max_regression = _number(effective.get("max_proxy_dataset_regression"), 1.5) or 1.5
    if max_regression <= 0:
        return 1.0 if worst_regression <= 0 else 0.0
    return round(_clamp(1.0 - (worst_regression / max_regression)), 4)


def _s2_variant_score(scorecard: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if isinstance(scorecard, dict):
        candidate_id = str(candidate.get("id") or candidate.get("variant_id") or scorecard.get("selected_variant_id") or "")
        rows = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
        selected = next((item for item in rows if str(item.get("variant_id") or "") == candidate_id), None)
        if selected is None:
            selected = next((item for item in rows if item.get("decision") == "selected"), None)
        if selected and _is_number(selected.get("score")):
            value = float(selected["score"])
            if value > 10:
                value /= 100.0
            return round(_clamp(value), 4)
    return 0.6


def _novelty_score(novelty_audit: dict[str, Any] | None, scorecard: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if isinstance(novelty_audit, dict):
        latest = novelty_audit.get("latest") if isinstance(novelty_audit.get("latest"), dict) else {}
        audit = latest.get("audit") if isinstance(latest.get("audit"), dict) else novelty_audit
        for key in ["novelty_score", "score"]:
            if _is_number(audit.get(key)):
                value = float(audit[key])
                return round(_clamp(value if value <= 1 else value / 10.0), 4)
    if isinstance(scorecard, dict):
        rows = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
        selected = next((item for item in rows if item.get("decision") == "selected"), rows[0] if rows else {})
        components = selected.get("components") if isinstance(selected.get("components"), dict) else {}
        if _is_number(components.get("novelty")):
            return round(_clamp(float(components["novelty"])), 4)
    if _is_number(candidate.get("novelty_score")):
        value = float(candidate["novelty_score"])
        return round(_clamp(value if value <= 1 else value / 10.0), 4)
    return 0.6


def _resource_cost_risk(config: dict[str, Any], candidate: dict[str, Any]) -> float:
    cost = (((config.get("c2c") or {}).get("small_loop") or {}).get("resource_cost_risk")) if isinstance(config.get("c2c"), dict) else None
    if _is_number(cost):
        return _clamp(float(cost))
    budget = candidate.get("resource_budget") if isinstance(candidate.get("resource_budget"), dict) else {}
    gpu_hours = _number(budget.get("gpu_hours"), None)
    if gpu_hours is None:
        return 0.1
    return _clamp(float(gpu_hours) / 24.0)


def _dataset_signature(config: dict[str, Any], proxy_config: dict[str, Any], run_spec: dict[str, Any]) -> dict[str, Any]:
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    proxy_spec = run_spec.get("proxy_screen") if isinstance(run_spec.get("proxy_screen"), dict) else {}
    proxy_spec_config = proxy_spec.get("config") if isinstance(proxy_spec.get("config"), dict) else {}
    return {
        "eval_datasets": list(proxy_spec_config.get("eval_datasets") or proxy_config.get("eval_datasets") or c2c_cfg.get("datasets") or []),
        "eval_limit": proxy_spec_config.get("eval_limit", proxy_config.get("eval_limit")),
        "dataset_root": str(c2c_cfg.get("dataset_root") or c2c_cfg.get("datasets_root") or ""),
    }


def _plan_hash(project_root: Path, plan: dict[str, Any]) -> str:
    for rel in ["plan/plan.yaml", "plan/short_loop_plan.yaml"]:
        path = project_root / rel
        if path.exists() and path.is_file():
            return sha256_file(path)
    return _stable_hash(plan)


def _eval_recipe_hash(run_spec: dict[str, Any], proxy_config: dict[str, Any]) -> str:
    proxy_spec = run_spec.get("proxy_screen") if isinstance(run_spec.get("proxy_screen"), dict) else {}
    paths = []
    train_config = proxy_spec.get("train_config")
    if train_config:
        paths.append(Path(train_config))
    eval_configs = proxy_spec.get("eval_configs") if isinstance(proxy_spec.get("eval_configs"), dict) else {}
    paths.extend(Path(path) for path in eval_configs.values() if path)
    file_hashes = {str(path): sha256_file(path) for path in paths if path.exists() and path.is_file()}
    if file_hashes:
        return _stable_hash(file_hashes)
    return _stable_hash({"proxy_eval_datasets": proxy_config.get("eval_datasets"), "proxy_eval_limit": proxy_config.get("eval_limit")})


def _snapshot_manifest_hash(snapshot_root: Path) -> str:
    candidates = [
        snapshot_root / "manifest.json",
        snapshot_root / "stage_manifest.json",
        snapshot_root / ".git" / "HEAD",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return sha256_file(path)
    return _stable_hash({"snapshot_path": str(snapshot_root), "exists": snapshot_root.exists()})


def _evaluator_file_hashes(snapshot_root: Path, run_spec: dict[str, Any]) -> dict[str, str]:
    roots = [snapshot_root]
    run_root = run_spec.get("run_root")
    if run_root:
        roots.append(Path(run_root).parents[2] if len(Path(run_root).parents) > 2 else Path(run_root))
    hashes = {}
    for root in roots:
        path = root / "script" / "evaluation" / "unified_evaluator.py"
        if path.exists() and path.is_file():
            hashes["script/evaluation/unified_evaluator.py"] = sha256_file(path)
            break
    return hashes


def _s2_5_lock_hashes(project_root: Path) -> dict[str, Any]:
    locks = {
        "patch_manifest_sha256": _sha_or_none(project_root / "plan" / "code_patches" / "patch_manifest.json"),
        "implementation_contract_sha256": _sha_or_none(project_root / "plan" / "code_patches" / "implementation_contract.json"),
        "patch_gate_report_sha256": _sha_or_none(project_root / "plan" / "code_patches" / "patch_gate_report.json"),
        "planner_gate_report_sha256": _sha_or_none(project_root / "plan" / "s2_planner" / "planner_gate_report.json"),
        "variant_scorecard_sha256": _sha_or_none(project_root / "plan" / "s2_planner" / "variant_scorecard.json"),
    }
    manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={})
    selected_patch = manifest.get("selected_patch") if isinstance(manifest, dict) and isinstance(manifest.get("selected_patch"), dict) else {}
    patch_rel = selected_patch.get("patch_json")
    contract_rel = selected_patch.get("implementation_contract")
    locks["selected_patch_sha256"] = _sha_or_none(project_root / str(patch_rel)) if patch_rel else None
    locks["selected_implementation_contract_sha256"] = _sha_or_none(project_root / str(contract_rel)) if contract_rel else None
    return locks


def _sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() and path.is_file() else None


def _candidate_config_overrides(candidate: dict[str, Any]) -> dict[str, Any]:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    return contract.get("config_overrides") if isinstance(contract.get("config_overrides"), dict) else {}


def _fingerprint_mismatch_reason(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    expected_inputs = expected.get("inputs") if isinstance(expected.get("inputs"), dict) else {}
    actual_inputs = actual.get("inputs") if isinstance(actual.get("inputs"), dict) else {}
    for key in [
        "proxy_config_hash",
        "eval_recipe_hash",
        "dataset_signature",
        "baseline_model_signature",
        "s2_5_locks",
        "env_signature",
        "snapshot_manifest_hash",
    ]:
        if expected_inputs.get(key) != actual_inputs.get(key):
            return f"{key}_changed"
    return "fingerprint_hash_changed"


def _load_calibration_history(project_root: Path | None) -> dict[str, Any]:
    if not project_root:
        return {}
    for rel in ["experiment/results/proxy_calibration.json", "experiment/results/c2c_proxy_calibration_history.json"]:
        path = project_root / rel
        if path.exists():
            return read_json(path, default={}) or {}
    return {}


def _history_candidate_count(history: dict[str, Any]) -> int:
    iterations = history.get("iterations") if isinstance(history.get("iterations"), list) else []
    return sum(int(item.get("candidate_count") or len(item.get("candidates") or [])) for item in iterations if isinstance(item, dict))


def _matched_false_positive_rate(items: list[Any], direction_fingerprint: dict[str, Any], *, keys: tuple[str, ...], fallback: float) -> float:
    direction_values = {str(direction_fingerprint.get(key) or "").lower() for key in keys if direction_fingerprint.get(key)}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_values = {str(item.get(key) or "").lower() for key in keys if item.get(key)}
        if direction_values and item_values and direction_values & item_values:
            return float(item.get("false_positive_rate") or fallback or 0.0)
    return float(fallback or 0.0)


def _dataset_misprediction_risk(summary: dict[str, Any], risky_datasets: list[Any]) -> dict[str, float]:
    risk: dict[str, float] = {}
    for dataset, item in summary.items():
        if not isinstance(item, dict):
            continue
        count = int(item.get("count") or 0)
        if count:
            risk[str(dataset)] = round(float(item.get("misprediction_count") or 0) / count, 4)
    for item in risky_datasets:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset") or "")
        if dataset:
            risk[dataset] = round(float(item.get("misprediction_rate") or risk.get(dataset, 0.0)), 4)
    return dict(sorted(risk.items()))


def _proxy_cfg(config: dict[str, Any]) -> dict[str, Any]:
    c2c = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    small_loop = c2c.get("small_loop") if isinstance(c2c.get("small_loop"), dict) else {}
    return small_loop.get("proxy_screen") if isinstance(small_loop.get("proxy_screen"), dict) else {}


def _rel_or_abs(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_number(value: Any) -> bool:
    return _number(value, None) is not None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
