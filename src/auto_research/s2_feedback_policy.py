"""Deterministic feedback policy for C2C S2 variant selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .utils import now_utc, read_json, read_yaml


C2C_S2_FEEDBACK_CONTEXT_SCHEMA_VERSION = "c2c_s2_feedback_context_v1"
C2C_S2_ADAPTIVE_POLICY_SCHEMA_VERSION = "c2c_s2_adaptive_policy_v1"
C2C_S2_SCORE_ADJUSTMENT_REPORT_SCHEMA_VERSION = "c2c_s2_score_adjustment_report_v1"


DEFAULT_S2_ADAPTIVE_POLICY_CONFIG = {
    "enabled": True,
    "min_history_for_penalty": 1,
    "min_history_for_mechanism_penalty": 2,
    "max_proxy_false_positive_penalty": 0.18,
    "max_full_s3_failure_penalty": 0.20,
    "implementation_failure_penalty": 0.03,
    "resource_failure_penalty": 0.0,
    "dataset_regression_penalty": 0.06,
    "reward_addresses_dragging_dataset": 0.06,
    "force_new_integration_after_proxy_failures": 2,
    "force_new_direction_after_full_s3_failures": 1,
    "require_adjustment_report": True,
}

METHOD_FAILURE_CLASSES = {
    "proxy_negative",
    "proxy_threshold_rejected",
    "proxy_mean_delta_below_effective_policy",
    "proxy_dataset_regression_above_effective_policy",
    "proxy_false_positive",
    "full_s3_method_failure",
    "method_failure",
    "full_s3_not_worthy",
}

IMPLEMENTATION_FAILURE_MARKERS = {
    "implementation_failure",
    "no_executable_change",
    "activation_switch_missing",
    "missing_ablation_switch",
    "forbidden_file_touched",
    "patch_generation_failure",
    "patch_gate_failed",
    "static_patch_gate_failed",
    "patch_gate_not_passed",
}

REPAIRABLE_PROXY_FAILURE_MARKERS = {
    "proxy_repairable",
    "effect_first_proxy_repair",
    "repairable_proxy",
    "repairable_proxy_risk_before_full_training",
    "patch_repair",
    "patch_only_repair",
    "proxy_timeout",
}


def build_s2_feedback_context(
    *,
    project_root: Path,
    direction: dict[str, Any],
    config: dict[str, Any] | None = None,
    shared_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build planner feedback only from event-derived method outcomes."""

    direction_id = str(direction.get("direction_id") or "unknown_direction")
    direction_hash = str(direction.get("direction_semantic_hash") or "")
    registry = read_yaml(project_root / "meta" / "registry.yaml", default={}) or {}
    state = read_json(project_root / "meta" / "research_state.json", default={}) or {}
    route_outcome = read_json(project_root / "meta" / "route_outcome.json", default={}) or {}
    proxy_calibration_policy = read_json(project_root / "experiment" / "results" / "c2c_proxy_calibration_policy.json", default={}) or {}
    method_history = [
        item for item in state.get("method_tried_history") or []
        if isinstance(item, dict) and item.get("method_evaluable") and item.get("direction_semantic_hash") == direction_hash
    ]
    attempts = state.get("attempts") if isinstance(state.get("attempts"), dict) else {}
    recent_failures = []
    for item in method_history:
        attempt = attempts.get(item.get("attempt_id")) if isinstance(attempts, dict) else {}
        recent_failures.append({
            "attempt_id": item.get("attempt_id"),
            "variant_id": item.get("variant_id"),
            "variant_spec_hash": item.get("variant_spec_hash"),
            "failure_class": (attempt or {}).get("failure_class") or item.get("outcome_classification"),
            "outcome_classification": item.get("outcome_classification"),
            "method_evaluable": True,
        })
    implementation_failures = [
        item for item in state.get("implementation_history") or []
        if isinstance(item, dict)
    ]
    budget = (((state.get("directions") or {}).get(direction_hash) or {}).get("budget") or {}) if direction_hash else {}
    dragging_datasets = _dragging_datasets({}, {}, {}, proxy_calibration_policy)
    return {
        "schema_version": C2C_S2_FEEDBACK_CONTEXT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "direction_id": direction_id,
        "current_iteration": int(registry.get("iteration") or 1),
        "route_summary": {
            "last_decision": route_outcome.get("next_action"),
            "failure_class": None,
            "reason_codes": route_outcome.get("reason_codes") or [],
        },
        "attempt_counters": {
            "same_direction_proxy_failures": sum(1 for item in recent_failures if "proxy" in str(item.get("failure_class") or "")),
            "same_direction_full_s3_failures": sum(1 for item in recent_failures if item.get("outcome_classification") in {"rejected", "falsified"}),
            "patch_repairs": len(implementation_failures),
            "resource_retries": sum(1 for attempt in attempts.values() if isinstance(attempt, dict) and attempt.get("direction_semantic_hash") == direction_hash and attempt.get("state") == "RESOURCE_PAUSED"),
            "max_same_direction_proxy_failures": 5,
            "max_same_direction_full_s3_failures": 5,
            "max_patch_repair_attempts": None,
            "max_resource_retries": None,
            "direction_budget_consumed": int(budget.get("consumed", 0)),
            "direction_budget_reserved": int(budget.get("reserved", 0)),
        },
        "proxy_calibration": {
            "global_false_positive_rate": _number(_nested(proxy_calibration_policy, ["summary", "proxy_false_positive_rate"]), 0.0),
            "mechanism_false_positive_rate": _number(proxy_calibration_policy.get("mechanism_false_positive_rate"), 0.0),
            "integration_point_false_positive_rate": _number(proxy_calibration_policy.get("integration_point_false_positive_rate"), 0.0),
            "dataset_risks": [],
        },
        "recent_failures": recent_failures[-25:],
        "failed_axes": [],
        "failed_integration_points": [],
        "failed_control_signals": [],
        "implementation_failure_surfaces": [],
        "dragging_datasets": dragging_datasets,
        "shared_method_memory": _shared_memory_summary(shared_memory),
        "source_paths": [
            "meta/research_state.json",
            "meta/research_events",
            "meta/route_outcome.json",
            "experiment/results/trial_result.json",
            "experiment/results/c2c_proxy_calibration_policy.json",
        ],
    }


def build_s2_adaptive_policy(feedback_context: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert route/proxy history into deterministic S2 selection policy."""

    cfg = _policy_config(config or {})
    direction_id = str((feedback_context or {}).get("direction_id") or "unknown_direction")
    counters = (feedback_context.get("attempt_counters") or {}) if isinstance(feedback_context, dict) else {}
    recent = [item for item in (feedback_context.get("recent_failures") or []) if isinstance(item, dict)] if isinstance(feedback_context, dict) else []
    history_count = len(recent) + int(counters.get("same_direction_proxy_failures") or 0) + int(counters.get("same_direction_full_s3_failures") or 0)
    history_sufficient = history_count >= int(cfg.get("min_history_for_penalty") or 1)
    proxy_failures = int(counters.get("same_direction_proxy_failures") or 0)
    full_failures = int(counters.get("same_direction_full_s3_failures") or 0)
    max_proxy = int(counters.get("max_same_direction_proxy_failures") or _route_budget(config or {}, "same_direction_proxy_failures", 5))
    max_full = int(counters.get("max_same_direction_full_s3_failures") or _route_budget(config or {}, "same_direction_full_s3_failures", 1))
    force_new_integration = history_sufficient and proxy_failures >= int(cfg.get("force_new_integration_after_proxy_failures") or 2)
    consumed = int(counters.get("direction_budget_consumed") or 0)
    reserved = int(counters.get("direction_budget_reserved") or 0)
    force_new_direction = False
    same_direction_budget_remaining = consumed + reserved < 5
    penalties = {
        "same_mechanism_proxy_false_positive": -min(0.12, float(cfg.get("max_proxy_false_positive_penalty") or 0.18)),
        "same_integration_point_proxy_false_positive": -min(0.10, float(cfg.get("max_proxy_false_positive_penalty") or 0.18)),
        "same_control_signal_full_s3_failure": -min(0.08, float(cfg.get("max_full_s3_failure_penalty") or 0.20)),
        "repeated_dataset_regression": -float(cfg.get("dataset_regression_penalty") or 0.06),
        "patch_failure_only": -float(cfg.get("implementation_failure_penalty") or 0.03),
        "resource_failure": -float(cfg.get("resource_failure_penalty") or 0.0),
    }
    bonuses = {
        "addresses_dragging_dataset": float(cfg.get("reward_addresses_dragging_dataset") or 0.06),
        "new_integration_point_after_proxy_failure": 0.07,
        "mechanism_with_low_false_positive_rate": 0.06,
        "has_explicit_ablation_for_failed_axis": 0.05,
        "new_control_signal_after_full_failure": 0.04,
    }
    reason_codes = []
    if history_sufficient:
        reason_codes.append("proxy_failure_history_available" if proxy_failures else "route_history_available")
    else:
        reason_codes.append("history_insufficient_no_adaptive_penalty")
    reason_codes.append("same_direction_budget_remaining" if same_direction_budget_remaining else "same_direction_budget_exhausted")
    if force_new_integration:
        reason_codes.append("force_new_integration_after_proxy_failures")
    if force_new_direction:
        reason_codes.append("force_new_direction_after_full_s3_failures")
    policy = {
        "schema_version": C2C_S2_ADAPTIVE_POLICY_SCHEMA_VERSION,
        "created_at": now_utc(),
        "direction_id": direction_id,
        "enabled": bool(cfg.get("enabled", True)),
        "history_sufficient": bool(history_sufficient),
        "history_count": history_count,
        "penalties": penalties,
        "bonuses": bonuses,
        "route_constraints": {
            "same_direction_budget_remaining": bool(same_direction_budget_remaining),
            "force_new_integration_point": bool(force_new_integration),
            "force_new_direction": bool(force_new_direction),
            "failed_integration_points": list(feedback_context.get("failed_integration_points") or []) if isinstance(feedback_context, dict) else [],
            "failed_mechanism_axes": list(feedback_context.get("failed_axes") or []) if isinstance(feedback_context, dict) else [],
            "failed_control_signals": list(feedback_context.get("failed_control_signals") or []) if isinstance(feedback_context, dict) else [],
        },
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "config": {
            key: cfg.get(key)
            for key in [
                "min_history_for_penalty",
                "min_history_for_mechanism_penalty",
                "force_new_integration_after_proxy_failures",
                "force_new_direction_after_full_s3_failures",
                "require_adjustment_report",
            ]
        },
    }
    policy["policy_hash"] = _stable_hash(
        {
            "direction_id": direction_id,
            "enabled": policy["enabled"],
            "history_sufficient": policy["history_sufficient"],
            "penalties": penalties,
            "bonuses": bonuses,
            "route_constraints": policy["route_constraints"],
            "proxy_calibration": (feedback_context or {}).get("proxy_calibration") if isinstance(feedback_context, dict) else {},
            "attempt_counters": counters,
        }
    )
    return policy


def build_s2_variant_failure_prior(
    candidate: dict[str, Any],
    feedback_context: dict[str, Any] | None,
    adaptive_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return route history, patch surface, and budget priors for one variant."""

    if not _adaptive_enabled(adaptive_policy):
        return _prior_result(route_history=0.0, patch_surface=0.0, budget=0.0)
    context = feedback_context if isinstance(feedback_context, dict) else {}
    policy = adaptive_policy if isinstance(adaptive_policy, dict) else {}
    penalties = policy.get("penalties") if isinstance(policy.get("penalties"), dict) else {}
    bonuses = policy.get("bonuses") if isinstance(policy.get("bonuses"), dict) else {}
    candidate_axis = str(candidate.get("mechanism_axis") or "")
    candidate_point = str(candidate.get("integration_point") or "")
    candidate_signal = str(candidate.get("control_signal") or "")
    route_delta = 0.0
    patch_delta = 0.0
    budget_delta = 0.0
    applied: list[dict[str, Any]] = []
    method_failures = [item for item in context.get("recent_failures") or [] if isinstance(item, dict) and _is_method_failure(item.get("failure_class"))]
    implementation_failures = [item for item in context.get("recent_failures") or [] if isinstance(item, dict) and _is_implementation_failure(item.get("failure_class"))]
    for failure in method_failures:
        same_axis = candidate_axis and candidate_axis == str(failure.get("mechanism_axis") or "")
        same_point = candidate_point and candidate_point == str(failure.get("integration_point") or "")
        same_signal = candidate_signal and candidate_signal == str(failure.get("control_signal") or "")
        failure_class = str(failure.get("failure_class") or "")
        if same_axis and ("full_s3" in failure_class or "false_positive" in failure_class):
            delta = float(penalties.get("same_mechanism_proxy_false_positive") or -0.08)
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "same_mechanism_proxy_false_positive"))
        elif same_axis and "proxy" in failure_class:
            delta = max(float(penalties.get("same_mechanism_proxy_false_positive") or -0.08), -0.08)
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "same_mechanism_proxy_negative"))
        if same_point:
            delta = float(penalties.get("same_integration_point_proxy_false_positive") or -0.06)
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "same_integration_point_proxy_false_positive"))
        if same_axis and same_point and same_signal:
            delta = -0.15
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "repeated_full_tuple"))
        if same_axis and candidate_point and failure.get("integration_point") and not same_point:
            delta = float(bonuses.get("new_integration_point_after_proxy_failure") or 0.07)
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "new_integration_after_proxy_failure"))
        if "full_s3" in failure_class and same_axis and candidate_signal and failure.get("control_signal") and not same_signal:
            delta = float(bonuses.get("new_control_signal_after_full_failure") or 0.04)
            route_delta += delta
            applied.append(_applied("route_history_prior", delta, "new_control_signal_after_full_failure"))
    for failure in implementation_failures:
        if _same_patch_surface(candidate, failure):
            delta = float(penalties.get("patch_failure_only") or -0.03)
            patch_delta += delta
            applied.append(_applied("patch_surface_prior", delta, "same_patch_surface_implementation_failure"))
    constraints = policy.get("route_constraints") if isinstance(policy.get("route_constraints"), dict) else {}
    counters = context.get("attempt_counters") if isinstance(context.get("attempt_counters"), dict) else {}
    proxy_failures = int(counters.get("same_direction_proxy_failures") or 0)
    max_proxy = int(counters.get("max_same_direction_proxy_failures") or 2)
    if constraints.get("same_direction_budget_remaining") is False:
        budget_delta -= 0.20
        applied.append(_applied("budget_prior", -0.20, "same_direction_budget_exhausted"))
    elif proxy_failures and proxy_failures + 1 >= max_proxy:
        budget_delta -= 0.05
        applied.append(_applied("budget_prior", -0.05, "same_direction_budget_near_limit"))
    elif method_failures:
        budget_delta += 0.03
        applied.append(_applied("budget_prior", 0.03, "same_direction_budget_remaining"))
    return _prior_result(route_history=route_delta, patch_surface=patch_delta, budget=budget_delta, applied=applied)


def build_s2_proxy_calibration_prior(
    candidate: dict[str, Any],
    feedback_context: dict[str, Any] | None,
    adaptive_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _adaptive_enabled(adaptive_policy):
        return _prior_result(proxy=0.0)
    calibration = (feedback_context or {}).get("proxy_calibration") if isinstance(feedback_context, dict) else {}
    calibration = calibration if isinstance(calibration, dict) else {}
    bonuses = adaptive_policy.get("bonuses") if isinstance(adaptive_policy, dict) and isinstance(adaptive_policy.get("bonuses"), dict) else {}
    penalties = adaptive_policy.get("penalties") if isinstance(adaptive_policy, dict) and isinstance(adaptive_policy.get("penalties"), dict) else {}
    rate = float(calibration.get("mechanism_false_positive_rate") or calibration.get("global_false_positive_rate") or 0.0)
    integration_rate = float(calibration.get("integration_point_false_positive_rate") or 0.0)
    delta = 0.0
    applied: list[dict[str, Any]] = []
    if rate >= 0.5:
        value = max(float(penalties.get("same_mechanism_proxy_false_positive") or -0.12), -0.12)
        delta += value
        applied.append(_applied("proxy_calibration_prior", value, "mechanism_false_positive_rate_high"))
    elif rate > 0.0 and rate <= 0.2:
        value = float(bonuses.get("mechanism_with_low_false_positive_rate") or 0.06)
        delta += value
        applied.append(_applied("proxy_calibration_prior", value, "mechanism_low_false_positive_rate"))
    if integration_rate >= 0.5 and candidate.get("integration_point"):
        value = max(float(penalties.get("same_integration_point_proxy_false_positive") or -0.10), -0.10)
        delta += value
        applied.append(_applied("proxy_calibration_prior", value, "integration_point_false_positive_rate_high"))
    return _prior_result(proxy=delta, applied=applied)


def build_s2_dataset_risk_prior(
    candidate: dict[str, Any],
    feedback_context: dict[str, Any] | None,
    adaptive_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _adaptive_enabled(adaptive_policy):
        return _prior_result(dataset=0.0)
    context = feedback_context if isinstance(feedback_context, dict) else {}
    penalties = adaptive_policy.get("penalties") if isinstance(adaptive_policy, dict) and isinstance(adaptive_policy.get("penalties"), dict) else {}
    bonuses = adaptive_policy.get("bonuses") if isinstance(adaptive_policy, dict) and isinstance(adaptive_policy.get("bonuses"), dict) else {}
    dragging = [str(item) for item in context.get("dragging_datasets") or [] if item]
    if not dragging:
        return _prior_result(dataset=0.0)
    addressed = [dataset for dataset in dragging if _candidate_addresses_dataset(candidate, dataset)]
    delta = 0.0
    applied: list[dict[str, Any]] = []
    if addressed:
        value = float(bonuses.get("addresses_dragging_dataset") or 0.06)
        delta += value
        for dataset in addressed:
            applied.append(_applied("dataset_risk_prior", value, f"addresses_{dataset}_regression"))
    missing = [dataset for dataset in dragging if dataset not in addressed]
    if missing:
        value = float(penalties.get("repeated_dataset_regression") or -0.06)
        delta += value
        applied.append(_applied("dataset_risk_prior", value, "does_not_address_dragging_dataset", {"datasets": missing}))
    return _prior_result(dataset=delta, applied=applied)


def build_s2_score_adjustment_report(
    *,
    direction: dict[str, Any],
    candidate_pool: dict[str, Any],
    scorecard: dict[str, Any],
    adaptive_policy: dict[str, Any],
    feedback_context: dict[str, Any],
) -> dict[str, Any]:
    rows = {str(item.get("variant_id") or ""): item for item in scorecard.get("ranking") or [] if isinstance(item, dict)}
    adjustments = []
    for candidate in candidate_pool.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        variant_id = str(candidate.get("id") or "")
        row = rows.get(variant_id, {})
        adjustments.append(
            {
                "variant_id": variant_id,
                "base_score": row.get("base_score", row.get("score", 0.0)),
                "adjusted_score": row.get("adjusted_score", row.get("score", 0.0)),
                "applied": row.get("adjustments") or [],
            }
        )
    return {
        "schema_version": C2C_S2_SCORE_ADJUSTMENT_REPORT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "direction_id": str(direction.get("direction_id") or direction.get("id") or scorecard.get("direction_id") or ""),
        "policy_hash": adaptive_policy.get("policy_hash"),
        "selected_variant_id": scorecard.get("selected_variant_id"),
        "history_sufficient": adaptive_policy.get("history_sufficient"),
        "adjustment_count": sum(len(item.get("applied") or []) for item in adjustments),
        "adjustments": adjustments,
        "feedback_context_ref": "plan/s2_planner/feedback_context.json",
        "adaptive_policy_ref": "plan/s2_planner/adaptive_policy.json",
    }


def _collect_recent_failures(
    *,
    direction: dict[str, Any],
    route_decision: dict[str, Any],
    attempt_ledger: dict[str, Any],
    proxy_decision: dict[str, Any],
    main_results: dict[str, Any],
    performance_feedback: dict[str, Any],
    iteration_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direction_defaults = {
        "mechanism_axis": direction.get("mechanism_axis"),
        "integration_point": direction.get("integration_point"),
        "control_signal": direction.get("control_signal"),
    }
    if route_decision:
        rows.append(
            _failure_row(
                {
                    **direction_defaults,
                    "variant_id": ((route_decision.get("attempt_record") or {}).get("variant_id") if isinstance(route_decision.get("attempt_record"), dict) else None),
                    "failure_class": route_decision.get("failure_class"),
                    "route_decision": route_decision.get("decision"),
                    "reason_codes": route_decision.get("reason_codes") or [],
                    "consumes_same_direction_attempt": ((route_decision.get("budget_effects") or {}).get("consumes_same_direction_attempt") if isinstance(route_decision.get("budget_effects"), dict) else None),
                },
                source="meta/route_outcome.json",
            )
        )
    if proxy_decision:
        rows.append(
            _failure_row(
                {
                    **direction_defaults,
                    "variant_id": proxy_decision.get("variant_id") or proxy_decision.get("candidate_id"),
                    "variant_fingerprint": proxy_decision.get("variant_fingerprint"),
                    "failure_class": proxy_decision.get("failure_class") or proxy_decision.get("decision"),
                    "route_hint": proxy_decision.get("route_hint"),
                    "reason_codes": proxy_decision.get("reason_codes") or [],
                    "dataset_deltas": (proxy_decision.get("deltas") or {}).get("dataset_deltas") if isinstance(proxy_decision.get("deltas"), dict) else {},
                },
                source="experiment/results/c2c_proxy_decision_report.json",
            )
        )
    for candidate in main_results.get("candidate_results") or []:
        if isinstance(candidate, dict):
            rows.append(_failure_row(_candidate_failure_payload(candidate, direction_defaults), source="experiment/results/main_results.json"))
    for candidate in performance_feedback.get("candidate_results") or []:
        if isinstance(candidate, dict):
            rows.append(_failure_row(_candidate_failure_payload(candidate, direction_defaults), source="plan/performance_feedback.json"))
    for record in attempt_ledger.get("records") or []:
        if isinstance(record, dict):
            rows.append(
                _failure_row(
                    {
                        **direction_defaults,
                        "variant_id": record.get("variant_id"),
                        "failure_class": record.get("failure_class"),
                        "route_decision": record.get("route_decision"),
                        "consumes_same_direction_attempt": record.get("consumes_same_direction_attempt"),
                    },
                    source="meta/research_state.json",
                )
            )
    for event in iteration_trace[-12:]:
        if isinstance(event, dict) and event.get("event") == "route_decision":
            rows.append(
                _failure_row(
                    {
                        **direction_defaults,
                        "variant_id": event.get("variant_id"),
                        "failure_class": event.get("failure_class"),
                        "route_decision": event.get("decision"),
                        "reason_codes": event.get("reason_codes") or [],
                    },
                    source="meta/iteration_trace.jsonl",
                )
            )
    deduped = []
    seen = set()
    for row in rows:
        key = _failure_dedupe_key(row)
        if key in seen or not row.get("failure_class"):
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _failure_dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    identity = str(row.get("variant_fingerprint") or row.get("variant_id") or "").strip()
    if not identity:
        identity = "|".join(
            str(row.get(key) or "")
            for key in ["mechanism_axis", "integration_point", "control_signal"]
        )
    return (identity, str(row.get("failure_class") or ""))


def _candidate_failure_payload(candidate: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    variant = candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {}
    attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    return {
        **defaults,
        "variant_id": candidate.get("id") or candidate.get("candidate_id") or candidate.get("variant_id"),
        "variant_fingerprint": candidate.get("variant_fingerprint") or variant.get("variant_fingerprint"),
        "mechanism_axis": candidate.get("mechanism_axis") or variant.get("mechanism_axis") or defaults.get("mechanism_axis"),
        "integration_point": candidate.get("integration_point") or variant.get("integration_point") or defaults.get("integration_point"),
        "control_signal": candidate.get("control_signal") or variant.get("control_signal") or defaults.get("control_signal"),
        "failure_class": attribution.get("primary_failure") or candidate.get("failure_class") or candidate.get("decision") or proxy.get("status"),
        "dataset_deltas": proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {},
        "dragging_datasets": attribution.get("dragging_datasets") if isinstance(attribution.get("dragging_datasets"), list) else [],
    }


def _failure_row(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    failure_class = _normalized_failure_class(payload.get("failure_class"))
    return {
        "variant_id": str(payload.get("variant_id") or ""),
        "variant_fingerprint": str(payload.get("variant_fingerprint") or ""),
        "mechanism_axis": payload.get("mechanism_axis"),
        "integration_point": payload.get("integration_point"),
        "control_signal": payload.get("control_signal"),
        "failure_class": failure_class,
        "route_decision": payload.get("route_decision"),
        "route_hint": payload.get("route_hint"),
        "reason_codes": payload.get("reason_codes") or [],
        "consumes_same_direction_attempt": payload.get("consumes_same_direction_attempt"),
        "dataset_deltas": payload.get("dataset_deltas") if isinstance(payload.get("dataset_deltas"), dict) else {},
        "dragging_datasets": payload.get("dragging_datasets") if isinstance(payload.get("dragging_datasets"), list) else [],
        "source": source,
    }


def _normalized_failure_class(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in REPAIRABLE_PROXY_FAILURE_MARKERS or any(marker in lowered for marker in ["effect_first_proxy_repair", "repairable_proxy", "patch_repair"]):
        return "implementation_failure"
    if lowered in {"proxy_rejected", "rejected", "cheap_proxy_rejected_before_full_training"}:
        return "proxy_negative"
    if lowered in {"proxy_pass_full_fail", "acceptance_failed"}:
        return "full_s3_method_failure"
    if "resource" in lowered and "retry" in lowered:
        return "resource_retry"
    if "implementation" in lowered or lowered in IMPLEMENTATION_FAILURE_MARKERS:
        return "implementation_failure"
    return text or "unknown"


def _dragging_datasets(
    proxy_decision: dict[str, Any],
    main_results: dict[str, Any],
    performance_feedback: dict[str, Any],
    calibration_policy: dict[str, Any],
) -> list[str]:
    names: list[str] = []
    deltas = (proxy_decision.get("deltas") or {}).get("dataset_deltas") if isinstance(proxy_decision.get("deltas"), dict) else {}
    if isinstance(deltas, dict):
        names.extend(str(dataset) for dataset, delta in deltas.items() if _number(delta, 0.0) < 0)
    for candidate in main_results.get("candidate_results") or []:
        if not isinstance(candidate, dict):
            continue
        proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
        candidate_deltas = proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {}
        names.extend(str(dataset) for dataset, delta in candidate_deltas.items() if _number(delta, 0.0) < 0)
        attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
        names.extend(str(item.get("dataset")) for item in attribution.get("dragging_datasets") or [] if isinstance(item, dict) and item.get("dataset"))
    summary = performance_feedback.get("summary") if isinstance(performance_feedback.get("summary"), dict) else {}
    names.extend(str(item.get("dataset")) for item in summary.get("dragging_datasets") or [] if isinstance(item, dict) and item.get("dataset"))
    dataset_risk = calibration_policy.get("dataset_misprediction_risk") or _nested(calibration_policy, ["calibration_adjustments", "dataset_misprediction_risk"]) or {}
    if isinstance(dataset_risk, dict):
        names.extend(str(dataset) for dataset, risk in dataset_risk.items() if _number(risk, 0.0) >= 0.3)
    return sorted(set(item for item in names if item and item != "None"))


def _dataset_risks(calibration_policy: dict[str, Any], dragging: list[str]) -> dict[str, float]:
    payload = calibration_policy.get("dataset_misprediction_risk") or _nested(calibration_policy, ["calibration_adjustments", "dataset_misprediction_risk"]) or {}
    risks = {str(dataset): round(float(value), 4) for dataset, value in payload.items()} if isinstance(payload, dict) else {}
    for dataset in dragging:
        risks.setdefault(str(dataset), 0.35)
    return risks


def _attempt_counters(ledger: dict[str, Any], direction_id: str) -> dict[str, int]:
    records = [item for item in ledger.get("records") or [] if isinstance(item, dict)] if isinstance(ledger, dict) else []
    if records:
        counters = {"proxy_failures": 0, "full_s3_failures": 0, "patch_repairs": 0, "resource_retries": 0}
        for record in records:
            record_direction = str(record.get("direction_id") or "")
            if direction_id and record_direction and record_direction != direction_id:
                continue
            failure_class = _normalized_failure_class(record.get("failure_class"))
            if record.get("consumes_patch_repair_attempt"):
                counters["patch_repairs"] += 1
            if record.get("consumes_resource_retry"):
                counters["resource_retries"] += 1
            if not record.get("consumes_same_direction_attempt"):
                continue
            if failure_class == "full_s3_method_failure" or "full_s3" in failure_class:
                counters["full_s3_failures"] += 1
            elif _is_method_failure(failure_class):
                counters["proxy_failures"] += 1
        return counters
    by_direction = ((ledger.get("counters") or {}).get("by_direction") or {}) if isinstance(ledger, dict) else {}
    counters = by_direction.get(direction_id)
    if not isinstance(counters, dict):
        counters = next(iter(by_direction.values()), {}) if by_direction else {}
    return {
        "proxy_failures": int(counters.get("proxy_failures") or 0),
        "full_s3_failures": int(counters.get("full_s3_failures") or 0),
        "patch_repairs": int(counters.get("patch_repairs") or 0),
        "resource_retries": int(counters.get("resource_retries") or 0),
    }


def _budget_limits(config: dict[str, Any]) -> dict[str, int]:
    return {
        "same_direction_proxy_failures": _route_budget(config, "same_direction_proxy_failures", 5),
        "same_direction_full_s3_failures": _route_budget(config, "same_direction_full_s3_failures", 1),
        "patch_repair_attempts_per_variant": _route_budget(config, "patch_repair_attempts_per_variant", 2),
        "resource_retries_per_stage": _route_budget(config, "resource_retries_per_stage", 3),
    }


def _route_budget(config: dict[str, Any], key: str, default: int) -> int:
    route_cfg = ((config.get("orchestration") or {}).get("route_policy") or {}) if isinstance(config.get("orchestration"), dict) else {}
    budgets = route_cfg.get("budgets") if isinstance(route_cfg.get("budgets"), dict) else {}
    feedback = ((config.get("orchestration") or {}).get("failure_feedback") or {}) if isinstance(config.get("orchestration"), dict) else {}
    aliases = {
        "same_direction_proxy_failures": ["same_direction_proxy_failures", "max_same_direction_proxy_failures"],
        "same_direction_full_s3_failures": ["same_direction_full_s3_failures"],
        "patch_repair_attempts_per_variant": ["patch_repair_attempts_per_variant", "max_implementation_repair_routes_per_iteration"],
        "resource_retries_per_stage": ["resource_retries_per_stage"],
    }
    for alias in aliases.get(key, [key]):
        if budgets.get(alias) is not None:
            return int(budgets.get(alias) or default)
        if feedback.get(alias) is not None:
            return int(feedback.get(alias) or default)
    return default


def _policy_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_S2_ADAPTIVE_POLICY_CONFIG)
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    user_cfg = c2c_cfg.get("s2_adaptive_policy") if isinstance(c2c_cfg.get("s2_adaptive_policy"), dict) else {}
    cfg.update(user_cfg)
    return cfg


def _adaptive_enabled(adaptive_policy: dict[str, Any] | None) -> bool:
    return bool(isinstance(adaptive_policy, dict) and adaptive_policy.get("enabled", True) and adaptive_policy.get("history_sufficient"))


def _same_patch_surface(candidate: dict[str, Any], failure: dict[str, Any]) -> bool:
    point = str(candidate.get("integration_point") or "")
    if point and point == str(failure.get("integration_point") or ""):
        return True
    files = set(_expected_files(candidate))
    failed_files = set(str(item) for item in failure.get("expected_files") or [] if item)
    return bool(files and failed_files and files.intersection(failed_files))


def _candidate_addresses_dataset(candidate: dict[str, Any], dataset: str) -> bool:
    needle = str(dataset).lower()
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    diagnostics = " ".join(str(item) for item in contract.get("diagnostics_required") or [])
    implementation_plan = candidate.get("implementation_plan") if isinstance(candidate.get("implementation_plan"), dict) else {}
    text = " ".join(
        str(item)
        for item in [
            candidate.get("title"),
            candidate.get("hypothesis"),
            candidate.get("expected_signature"),
            diagnostics,
            json.dumps(implementation_plan, sort_keys=True),
            json.dumps(contract, sort_keys=True),
        ]
        if item
    ).lower()
    return needle in text or "dragging_dataset_probe" in text or "dataset_regression_probe" in text


def _expected_files(candidate: dict[str, Any]) -> list[str]:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    files = contract.get("expected_files") or candidate.get("expected_files") or []
    return [str(item) for item in files if item] if isinstance(files, list) else [str(files)] if files else []


def _is_method_failure(value: Any) -> bool:
    text = str(value or "").lower()
    if "resource" in text or _is_implementation_failure(text):
        return False
    if any(marker in text for marker in REPAIRABLE_PROXY_FAILURE_MARKERS):
        return False
    return any(marker in text for marker in ["proxy", "full_s3", "method"]) or text in METHOD_FAILURE_CLASSES


def _is_implementation_failure(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in IMPLEMENTATION_FAILURE_MARKERS) or any(marker in text for marker in REPAIRABLE_PROXY_FAILURE_MARKERS)


def _prior_result(
    *,
    proxy: float | None = None,
    route_history: float | None = None,
    dataset: float | None = None,
    patch_surface: float | None = None,
    budget: float | None = None,
    applied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    components: dict[str, float] = {}
    if proxy is not None:
        components["proxy_calibration_prior"] = round(float(proxy), 4)
    if route_history is not None:
        components["route_history_prior"] = round(float(route_history), 4)
    if dataset is not None:
        components["dataset_risk_prior"] = round(float(dataset), 4)
    if patch_surface is not None:
        components["patch_surface_prior"] = round(float(patch_surface), 4)
    if budget is not None:
        components["budget_prior"] = round(float(budget), 4)
    return {"components": components, "applied": applied or []}


def _applied(component: str, delta: float, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"component": component, "delta": round(float(delta), 4), "reason": reason}
    if details:
        payload["details"] = details
    return payload


def _shared_memory_summary(shared_memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(shared_memory, dict):
        return {"checked": False, "entry_count": 0}
    return {
        "checked": True,
        "entry_count": int(shared_memory.get("entry_count") or len(shared_memory.get("entries") or [])),
        "retrieved_count": len(shared_memory.get("entries") or shared_memory.get("recent_entries") or []),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _nested(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()
