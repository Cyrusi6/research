"""Deterministic route policy for C2C orchestration failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir, now_utc, read_json, write_json


ROUTE_CONTEXT_SCHEMA_VERSION = "c2c_route_context_v1"
ROUTE_DECISION_SCHEMA_VERSION = "c2c_route_decision_v1"
ATTEMPT_LEDGER_SCHEMA_VERSION = "c2c_attempt_ledger_v1"


ROUTE_DECISIONS = {
    "route_to_s1",
    "route_to_s2",
    "route_to_s2_5",
    "retry_resource",
    "pause",
    "block",
    "continue_stage",
    "complete",
}


def build_route_context(
    project_root: Path,
    registry: dict[str, Any] | None,
    config: dict[str, Any] | None,
    *,
    trigger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect route-relevant contracts into one deterministic snapshot."""

    registry = registry if isinstance(registry, dict) else {}
    config = config if isinstance(config, dict) else {}
    trigger = trigger if isinstance(trigger, dict) else {}
    direction = read_json(project_root / "literature" / "direction.json", default={}) or {}
    evidence_quality = read_json(project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
    direction_fingerprint = read_json(project_root / "literature" / "c2c" / "direction_fingerprint.json", default={}) or {}
    planner_gate = read_json(project_root / "plan" / "s2_planner" / "planner_gate_report.json", default={}) or {}
    variant_scorecard = read_json(project_root / "plan" / "s2_planner" / "variant_scorecard.json", default={}) or {}
    patch_gate = read_json(project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
    proxy_decision = read_json(project_root / "experiment" / "results" / "c2c_proxy_decision_report.json", default={}) or {}
    worthiness = read_json(project_root / "experiment" / "results" / "c2c_full_s3_worthiness.json", default={}) or {}
    calibration_policy = read_json(project_root / "experiment" / "results" / "c2c_proxy_calibration_policy.json", default={}) or {}
    main_results = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    performance_feedback = read_json(project_root / "plan" / "performance_feedback.json", default={}) or {}
    ledger = read_json(project_root / "meta" / "attempt_ledger.json", default={}) or {}
    iteration = int(registry.get("iteration") or 1)
    direction_id = _direction_id(direction, direction_fingerprint, evidence_quality, main_results, performance_feedback)
    variant_id = _variant_id(planner_gate, variant_scorecard, patch_gate, proxy_decision, main_results)
    budgets = _budget_snapshot(
        config=config,
        registry=registry,
        ledger=ledger,
        direction_id=direction_id,
        variant_id=variant_id,
    )
    return {
        "schema_version": ROUTE_CONTEXT_SCHEMA_VERSION,
        "created_at": now_utc(),
        "project_id": str(registry.get("project_id") or project_root.name),
        "iteration": iteration,
        "current_stage": str(trigger.get("stage") or registry.get("current_stage") or ""),
        "trigger": {
            "source": str(trigger.get("source") or ""),
            "status": str(trigger.get("status") or ""),
            "reason": str(trigger.get("reason") or ""),
            "stage": str(trigger.get("stage") or registry.get("current_stage") or ""),
            "gate_status": trigger.get("gate_status"),
        },
        "s1": {
            "direction_id": direction_id,
            "mechanism_axis": direction.get("mechanism_axis") or direction_fingerprint.get("mechanism_axis"),
            "integration_point": direction.get("integration_point") or direction_fingerprint.get("integration_point"),
            "control_signal": direction.get("control_signal") or direction_fingerprint.get("control_signal"),
            "evidence_gate": evidence_quality.get("gate") or evidence_quality.get("status"),
            "novelty_score": evidence_quality.get("novelty_score") or _novelty_from_direction(direction),
        },
        "s2": {
            "variant_id": variant_id,
            "planner_gate": planner_gate.get("gate"),
            "variant_score": _selected_variant_score(variant_scorecard),
            "selected_variant_fingerprint": planner_gate.get("selected_variant_fingerprint") or proxy_decision.get("variant_fingerprint"),
        },
        "s2_5": {
            "patch_gate": patch_gate.get("gate"),
            "failure_class": patch_gate.get("failure_class"),
            "repairable": patch_gate.get("repairable"),
            "selected_variant_matches_planner": ((patch_gate.get("checks") or {}) if isinstance(patch_gate.get("checks"), dict) else {}).get("selected_variant_matches_planner"),
        },
        "s3": {
            "proxy_decision": proxy_decision.get("decision"),
            "route_hint": proxy_decision.get("route_hint"),
            "failure_class": proxy_decision.get("failure_class"),
            "full_s3_worthiness_score": worthiness.get("score") or ((proxy_decision.get("full_s3_worthiness") or {}).get("score") if isinstance(proxy_decision.get("full_s3_worthiness"), dict) else None),
            "full_s3_worthiness_decision": worthiness.get("decision") or ((proxy_decision.get("full_s3_worthiness") or {}).get("decision") if isinstance(proxy_decision.get("full_s3_worthiness"), dict) else None),
            "calibration_policy_hash": calibration_policy.get("policy_hash"),
            "acceptance_passed": (main_results.get("acceptance") or {}).get("passed") if isinstance(main_results.get("acceptance"), dict) else None,
            "best_candidate_decision": ((main_results.get("best_candidate") or {}).get("decision") if isinstance(main_results.get("best_candidate"), dict) else None),
            "has_full_s3_metrics": _has_full_s3_metrics(main_results),
            "candidate_failure_class": _main_results_failure_class(main_results),
        },
        "budgets": budgets,
        "artifacts": {
            "route_context": "meta/route_context.json",
            "route_decision": "meta/route_decision.json",
            "attempt_ledger": "meta/attempt_ledger.json",
            "planner_gate": "plan/s2_planner/planner_gate_report.json" if (project_root / "plan" / "s2_planner" / "planner_gate_report.json").exists() else None,
            "variant_scorecard": "plan/s2_planner/variant_scorecard.json" if (project_root / "plan" / "s2_planner" / "variant_scorecard.json").exists() else None,
            "patch_gate": "plan/code_patches/patch_gate_report.json" if (project_root / "plan" / "code_patches" / "patch_gate_report.json").exists() else None,
            "proxy_decision": "experiment/results/c2c_proxy_decision_report.json" if (project_root / "experiment" / "results" / "c2c_proxy_decision_report.json").exists() else None,
            "full_s3_worthiness": "experiment/results/c2c_full_s3_worthiness.json" if (project_root / "experiment" / "results" / "c2c_full_s3_worthiness.json").exists() else None,
            "main_results": "experiment/results/main_results.json" if (project_root / "experiment" / "results" / "main_results.json").exists() else None,
            "performance_feedback": "plan/performance_feedback.json" if (project_root / "plan" / "performance_feedback.json").exists() else None,
        },
    }


def decide_next_route(route_context: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an auditable route decision from a route context."""

    return build_route_decision(route_context, _decision_core(route_context, config or {}), config=config)


def build_route_decision(
    route_context: dict[str, Any],
    decision_core: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the route_decision.json payload from a selected rule."""

    del config
    context = route_context if isinstance(route_context, dict) else {}
    core = decision_core if isinstance(decision_core, dict) else {"decision": "block", "reason_codes": ["route_policy_missing_decision_core"]}
    decision = str(core.get("decision") or "block")
    if decision not in ROUTE_DECISIONS:
        decision = "block"
    trigger_stage = str((context.get("trigger") or {}).get("stage") or context.get("current_stage") or "")
    next_stage = core.get("next_stage")
    next_iteration = core.get("next_iteration")
    budget_effects = core.get("budget_effects") if isinstance(core.get("budget_effects"), dict) else _default_budget_effects()
    memory_effects = core.get("memory_effects") if isinstance(core.get("memory_effects"), dict) else _default_memory_effects()
    artifact_effects = core.get("artifact_effects") if isinstance(core.get("artifact_effects"), dict) else _artifact_effects_for_decision(decision)
    return {
        "schema_version": ROUTE_DECISION_SCHEMA_VERSION,
        "created_at": now_utc(),
        "trigger_stage": trigger_stage,
        "trigger_source": (context.get("trigger") or {}).get("source"),
        "failure_class": str(core.get("failure_class") or _context_failure_class(context) or "none"),
        "decision": decision,
        "next_stage": next_stage,
        "next_iteration": next_iteration,
        "reason_codes": list(dict.fromkeys([str(item) for item in core.get("reason_codes") or [] if item])),
        "budget_effects": budget_effects,
        "memory_effects": memory_effects,
        "artifact_effects": artifact_effects,
        "orchestrator_action": {
            "registry_current_stage": next_stage or trigger_stage,
            "status": core.get("orchestrator_status") or _orchestrator_status(decision),
        },
        "context_ref": "meta/route_context.json",
        "attempt_ledger_ref": "meta/attempt_ledger.json",
    }


def apply_route_decision_summary(
    project_root: Path,
    route_context: dict[str, Any],
    route_decision: dict[str, Any],
) -> dict[str, Any]:
    """Append attempt ledger and iteration trace records for a route decision."""

    meta_dir = project_root / "meta"
    ensure_dir(meta_dir)
    ledger_path = meta_dir / "attempt_ledger.json"
    ledger = read_json(ledger_path, default={}) or {}
    if not isinstance(ledger, dict) or ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION:
        ledger = {
            "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
            "project_id": (route_context or {}).get("project_id") or project_root.name,
            "records": [],
            "counters": {"by_direction": {}},
        }
    direction_id = str(((route_context.get("s1") or {}).get("direction_id")) or "unknown_direction")
    variant_id = str(((route_context.get("s2") or {}).get("variant_id")) or ((route_context.get("s3") or {}).get("variant_id") or ""))
    patch_id = str(((route_context.get("s2_5") or {}).get("patch_id")) or variant_id or "")
    budget = route_decision.get("budget_effects") if isinstance(route_decision.get("budget_effects"), dict) else {}
    memory = route_decision.get("memory_effects") if isinstance(route_decision.get("memory_effects"), dict) else {}
    record = {
        "timestamp": now_utc(),
        "iteration": route_context.get("iteration"),
        "direction_id": direction_id,
        "variant_id": variant_id,
        "patch_id": patch_id,
        "stage": route_decision.get("trigger_stage") or route_context.get("current_stage"),
        "failure_class": route_decision.get("failure_class"),
        "route_decision": route_decision.get("decision"),
        "consumes_same_direction_attempt": bool(budget.get("consumes_same_direction_attempt")),
        "consumes_patch_repair_attempt": bool(budget.get("consumes_patch_repair_attempt")),
        "consumes_resource_retry": bool(budget.get("consumes_resource_retry")),
        "writes_method_memory": bool(memory.get("write_shared_method_memory")),
        "artifact_refs": {
            "route_context": "meta/route_context.json",
            "route_decision": "meta/route_decision.json",
            "proxy_decision": ((route_context.get("artifacts") or {}).get("proxy_decision")),
            "variant_scorecard": ((route_context.get("artifacts") or {}).get("variant_scorecard")),
            "patch_gate": ((route_context.get("artifacts") or {}).get("patch_gate")),
        },
    }
    ledger.setdefault("records", []).append(record)
    counters = ledger.setdefault("counters", {}).setdefault("by_direction", {}).setdefault(
        direction_id,
        {
            "proxy_failures": 0,
            "full_s3_failures": 0,
            "patch_repairs": 0,
            "resource_retries": 0,
        },
    )
    failure_class = str(route_decision.get("failure_class") or "")
    reason_text = " ".join(str(item) for item in route_decision.get("reason_codes") or [])
    failure_bucket = _same_direction_failure_bucket(failure_class, reason_text)
    if record["consumes_same_direction_attempt"] and failure_bucket == "full_s3":
        counters["full_s3_failures"] = int(counters.get("full_s3_failures") or 0) + 1
    elif record["consumes_same_direction_attempt"] and failure_bucket == "proxy":
        counters["proxy_failures"] = int(counters.get("proxy_failures") or 0) + 1
    if record["consumes_patch_repair_attempt"]:
        counters["patch_repairs"] = int(counters.get("patch_repairs") or 0) + 1
    if record["consumes_resource_retry"]:
        counters["resource_retries"] = int(counters.get("resource_retries") or 0) + 1
    ledger["updated_at"] = now_utc()
    write_json(ledger_path, ledger)
    trace = {
        "timestamp": record["timestamp"],
        "iteration": record["iteration"],
        "event": "route_decision",
        "from_stage": route_decision.get("trigger_stage"),
        "to_stage": route_decision.get("next_stage"),
        "failure_class": route_decision.get("failure_class"),
        "reason_codes": route_decision.get("reason_codes") or [],
        "direction_id": direction_id,
        "variant_id": variant_id,
        "patch_id": patch_id,
        "decision": route_decision.get("decision"),
    }
    with (meta_dir / "iteration_trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def write_route_artifacts(
    project_root: Path,
    route_context: dict[str, Any],
    route_decision: dict[str, Any],
) -> dict[str, Any]:
    """Write route context, decision, ledger, and trace artifacts."""

    meta_dir = project_root / "meta"
    ensure_dir(meta_dir)
    write_json(meta_dir / "route_context.json", route_context)
    write_json(meta_dir / "route_decision.json", route_decision)
    attempt_record = apply_route_decision_summary(project_root, route_context, route_decision)
    route_decision = {**route_decision, "attempt_record": attempt_record}
    archive_rel = _route_decision_archive_rel(route_decision)
    write_json(project_root / archive_rel, route_decision)
    route_decision = {**route_decision, "archive_ref": archive_rel}
    write_json(project_root / archive_rel, route_decision)
    write_json(meta_dir / "route_decision.json", route_decision)
    return {
        "route_context": route_context,
        "route_decision": route_decision,
        "attempt_record": attempt_record,
        "route_context_path": "meta/route_context.json",
        "route_decision_path": "meta/route_decision.json",
        "attempt_ledger_path": "meta/attempt_ledger.json",
    }


def _route_decision_archive_rel(route_decision: dict[str, Any]) -> str:
    created = str(route_decision.get("created_at") or now_utc())
    stage = str(route_decision.get("trigger_stage") or "unknown")
    stamp = "".join(ch if ch.isalnum() else "-" for ch in created).strip("-")
    stage_slug = "".join(ch if ch.isalnum() else "_" for ch in stage).strip("_") or "unknown"
    return f"meta/route_decisions/{stamp}_{stage_slug}.json"


def _decision_core(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    s1 = context.get("s1") if isinstance(context.get("s1"), dict) else {}
    s2 = context.get("s2") if isinstance(context.get("s2"), dict) else {}
    s2_5 = context.get("s2_5") if isinstance(context.get("s2_5"), dict) else {}
    s3 = context.get("s3") if isinstance(context.get("s3"), dict) else {}
    budgets = context.get("budgets") if isinstance(context.get("budgets"), dict) else {}
    trigger = context.get("trigger") if isinstance(context.get("trigger"), dict) else {}
    if s1.get("evidence_gate") and s1.get("evidence_gate") != "pass":
        return _route_to_s1("s1_evidence_gate_failed", consumes_same=False, memory=False)
    if s2.get("planner_gate") and s2.get("planner_gate") != "pass":
        return _route_to_s2("s2_planner_gate_failed", failure_class="planner_contract_failure", consumes_same=False, memory=False)
    if s2_5.get("patch_gate") and s2_5.get("patch_gate") != "pass":
        if s2_5.get("failure_class") == "planner_contract_failure" or s2_5.get("selected_variant_matches_planner") is False:
            return _block("s2_5_selected_variant_mismatch", failure_class="handoff_bug")
        if s2_5.get("failure_class") in {"runtime_smoke_resource_retry", "resource_retry"}:
            return _pause("s2_5_resource_retry", failure_class="resource_retry")
        if s2_5.get("repairable") is True or s2_5.get("failure_class"):
            return _route_to_s2_5("patch_gate_failed_repairable", failure_class=s2_5.get("failure_class") or "patch_gate_failed")
    if _is_resource_retry_context(context):
        return _pause("resource_retry", failure_class="resource_retry")
    route_hint = str(s3.get("route_hint") or "")
    proxy_decision = str(s3.get("proxy_decision") or "")
    s3_failure_class = str(s3.get("failure_class") or s3.get("candidate_failure_class") or "")
    if _is_repairable_proxy_context(s3):
        return _route_to_s2_5("proxy_repairable_patch_only_repair", failure_class=s3_failure_class or "implementation_failure")
    if route_hint == "repair_s2_5":
        return _route_to_s2_5("proxy_decision_report_route_hint_repair_s2_5", failure_class=s3_failure_class or "implementation_failure")
    if route_hint == "return_s1":
        return _route_to_s1("proxy_decision_report_route_hint_return_s1", failure_class=s3_failure_class or "method_failure", memory=True)
    if route_hint == "return_s2":
        if _same_direction_budget_exhausted(budgets):
            return _route_to_s1(
                "same_direction_budget_exhausted",
                failure_class=s3_failure_class or "proxy_negative",
                memory=True,
                extra_reasons=["proxy_decision_report_route_hint_return_s2"],
            )
        reasons = ["proxy_decision_report_route_hint_return_s2", "patch_gate_passed", "same_direction_budget_remaining"]
        if proxy_decision == "neutral_proxy_full_s3" or s3.get("full_s3_worthiness_decision") == "do_not_run_full_s3":
            reasons.append("neutral_proxy_worthiness_low")
        return _route_to_s2(
            *reasons,
            failure_class=s3_failure_class or "proxy_negative",
            consumes_same=True,
            memory=True,
            memory_kind="proxy_rejected_variant",
        )
    if proxy_decision == "proxy_rejected":
        if _same_direction_budget_exhausted(budgets):
            return _route_to_s1("proxy_rejected_same_direction_budget_exhausted", failure_class=s3_failure_class or "proxy_negative", memory=True)
        return _route_to_s2("proxy_rejected", "same_direction_budget_remaining", failure_class=s3_failure_class or "proxy_negative", consumes_same=True, memory=True, memory_kind="proxy_rejected_variant")
    if proxy_decision == "proxy_repairable":
        return _route_to_s2_5("proxy_repairable", failure_class=s3_failure_class or "implementation_failure")
    if proxy_decision in {"proxy_pass", "neutral_proxy_full_s3"} and s3.get("acceptance_passed") is False:
        if _same_direction_full_s3_budget_exhausted(budgets):
            return _route_to_s1("full_s3_failure_budget_exhausted", failure_class="full_s3_method_failure", memory=True)
        return _route_to_s2(
            "proxy_pass_but_full_s3_failed",
            "record_proxy_false_positive",
            failure_class="full_s3_method_failure",
            consumes_same=True,
            memory=True,
            memory_kind="proxy_false_positive_full_s3_failure",
        )
    if str(trigger.get("status") or "").lower() in {"pass", "passed", "ok"}:
        return {
            "decision": "continue_stage",
            "next_stage": context.get("current_stage"),
            "failure_class": None,
            "reason_codes": ["trigger_passed"],
            "budget_effects": _default_budget_effects(),
            "memory_effects": _default_memory_effects(),
            "artifact_effects": _artifact_effects_for_decision("continue_stage"),
        }
    if "implementation" in s3_failure_class:
        return _route_to_s2_5("implementation_failure", failure_class="implementation_failure")
    if str(trigger.get("reason") or "").strip():
        return _block("unclassified_failure", failure_class=s3_failure_class or "unknown")
    return _block("route_policy_no_matching_rule", failure_class="unknown")


def _route_to_s1(reason: str, *, failure_class: str = "method_failure", consumes_same: bool = True, memory: bool = True, extra_reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "decision": "route_to_s1",
        "next_stage": "S1_literature",
        "next_iteration": "increment",
        "failure_class": failure_class,
        "reason_codes": [reason, *(extra_reasons or [])],
        "budget_effects": {
            "consumes_same_direction_attempt": consumes_same,
            "consumes_patch_repair_attempt": False,
            "consumes_resource_retry": False,
            "increments_iteration": True,
        },
        "memory_effects": _memory_effects(memory, "method_failure" if memory else None),
        "artifact_effects": _artifact_effects_for_decision("route_to_s1"),
    }


def _route_to_s2(*reasons: str, failure_class: str, consumes_same: bool, memory: bool, memory_kind: str = "method_failure") -> dict[str, Any]:
    return {
        "decision": "route_to_s2",
        "next_stage": "S2_plan",
        "next_iteration": None,
        "failure_class": failure_class,
        "reason_codes": list(reasons),
        "budget_effects": {
            "consumes_same_direction_attempt": consumes_same,
            "consumes_patch_repair_attempt": False,
            "consumes_resource_retry": False,
            "increments_iteration": False,
        },
        "memory_effects": _memory_effects(memory, memory_kind if memory else None),
        "artifact_effects": _artifact_effects_for_decision("route_to_s2"),
    }


def _route_to_s2_5(reason: str, *, failure_class: str) -> dict[str, Any]:
    return {
        "decision": "route_to_s2_5",
        "next_stage": "S2_plan",
        "next_iteration": None,
        "failure_class": failure_class,
        "reason_codes": [reason, "patch_only_repair"],
        "budget_effects": {
            "consumes_same_direction_attempt": False,
            "consumes_patch_repair_attempt": True,
            "consumes_resource_retry": False,
            "increments_iteration": False,
        },
        "memory_effects": _memory_effects(False, None, skip_reason="implementation_failure_is_not_method_memory"),
        "artifact_effects": _artifact_effects_for_decision("route_to_s2_5"),
    }


def _pause(reason: str, *, failure_class: str) -> dict[str, Any]:
    return {
        "decision": "pause",
        "next_stage": None,
        "next_iteration": None,
        "failure_class": failure_class,
        "reason_codes": [reason, "resource_or_external_retry"],
        "budget_effects": {
            "consumes_same_direction_attempt": False,
            "consumes_patch_repair_attempt": False,
            "consumes_resource_retry": True,
            "increments_iteration": False,
        },
        "memory_effects": _memory_effects(False, None, skip_reason="resource_failure_is_not_method_memory"),
        "artifact_effects": _artifact_effects_for_decision("pause"),
        "orchestrator_status": "retryable_paused",
    }


def _block(reason: str, *, failure_class: str) -> dict[str, Any]:
    return {
        "decision": "block",
        "next_stage": None,
        "next_iteration": None,
        "failure_class": failure_class,
        "reason_codes": [reason],
        "budget_effects": _default_budget_effects(),
        "memory_effects": _memory_effects(False, None, skip_reason="route_policy_blocked"),
        "artifact_effects": _artifact_effects_for_decision("block"),
        "orchestrator_status": "blocked",
    }


def _memory_effects(write: bool, memory_kind: str | None, *, skip_reason: str | None = None) -> dict[str, Any]:
    return {
        "write_shared_method_memory": bool(write),
        "memory_kind": memory_kind,
        "skip_reason": None if write else skip_reason or "not_method_level_failure",
    }


def _default_memory_effects() -> dict[str, Any]:
    return _memory_effects(False, None, skip_reason="not_applicable")


def _default_budget_effects() -> dict[str, Any]:
    return {
        "consumes_same_direction_attempt": False,
        "consumes_patch_repair_attempt": False,
        "consumes_resource_retry": False,
        "increments_iteration": False,
    }


def _artifact_effects_for_decision(decision: str) -> dict[str, Any]:
    if decision == "route_to_s1":
        return {
            "invalidate_from": "S1_literature",
            "preserve_s1_direction": False,
            "preserve_s2_selected_variant": False,
            "preserve_s2_5_patch_lock": False,
        }
    if decision == "route_to_s2":
        return {
            "invalidate_from": "S2_plan",
            "invalidate_artifacts": [
                "plan/s2_planner/feedback_context.json",
                "plan/s2_planner/adaptive_policy.json",
                "plan/s2_planner/variant_scorecard.json",
                "plan/s2_planner/score_adjustment_report.json",
            ],
            "preserve_s1_direction": True,
            "preserve_s2_selected_variant": False,
            "preserve_s2_5_patch_lock": False,
        }
    if decision == "route_to_s2_5":
        return {
            "invalidate_from": "S2_plan",
            "preserve_s1_direction": True,
            "preserve_s2_selected_variant": True,
            "preserve_s2_5_patch_lock": False,
        }
    return {
        "invalidate_from": None,
        "preserve_s1_direction": True,
        "preserve_s2_selected_variant": True,
        "preserve_s2_5_patch_lock": True,
    }


def _orchestrator_status(decision: str) -> str:
    if decision == "pause":
        return "retryable_paused"
    if decision == "block":
        return "blocked"
    if decision == "complete":
        return "completed"
    return "feedback_routed" if decision in {"route_to_s1", "route_to_s2", "route_to_s2_5"} else "running"


def _budget_snapshot(
    *,
    config: dict[str, Any],
    registry: dict[str, Any],
    ledger: dict[str, Any],
    direction_id: str,
    variant_id: str,
) -> dict[str, Any]:
    policy_cfg = _route_policy_cfg(config)
    budgets_cfg = policy_cfg.get("budgets") if isinstance(policy_cfg.get("budgets"), dict) else {}
    feedback_cfg = (config.get("orchestration") or {}).get("failure_feedback") if isinstance(config.get("orchestration"), dict) else {}
    feedback_cfg = feedback_cfg if isinstance(feedback_cfg, dict) else {}
    by_direction = (((ledger.get("counters") or {}).get("by_direction") or {}).get(direction_id) or {}) if isinstance(ledger, dict) else {}
    iteration_key = str(registry.get("iteration") or 1)
    proxy_failures = int(by_direction.get("proxy_failures") or (registry.get("proxy_rejected_routes") or {}).get(iteration_key) or 0)
    patch_repairs = int(by_direction.get("patch_repairs") or _sum_registry_repair_routes(registry, variant_id) or 0)
    resource_retries = int(by_direction.get("resource_retries") or 0)
    full_s3_failures = int(by_direction.get("full_s3_failures") or 0)
    return {
        "same_direction_proxy_failures": proxy_failures,
        "max_same_direction_proxy_failures": int(budgets_cfg.get("same_direction_proxy_failures") or feedback_cfg.get("max_same_direction_proxy_failures") or feedback_cfg.get("max_same_direction_proxy_iterations") or feedback_cfg.get("max_proxy_rejected_routes_per_iteration") or 2),
        "same_direction_full_s3_failures": full_s3_failures,
        "max_same_direction_full_s3_failures": int(budgets_cfg.get("same_direction_full_s3_failures") or 1),
        "patch_repair_attempts": patch_repairs,
        "max_patch_repair_attempts": int(budgets_cfg.get("patch_repair_attempts_per_variant") or feedback_cfg.get("max_implementation_repair_routes_per_iteration") or 2),
        "resource_retries": resource_retries,
        "max_resource_retries": int(budgets_cfg.get("resource_retries_per_stage") or 3),
    }


def _route_policy_cfg(config: dict[str, Any]) -> dict[str, Any]:
    orchestration = config.get("orchestration") if isinstance(config.get("orchestration"), dict) else {}
    cfg = orchestration.get("route_policy") if isinstance(orchestration.get("route_policy"), dict) else {}
    return cfg


def _same_direction_budget_exhausted(budgets: dict[str, Any]) -> bool:
    return int(budgets.get("same_direction_proxy_failures") or 0) >= int(budgets.get("max_same_direction_proxy_failures") or 1)


def _same_direction_full_s3_budget_exhausted(budgets: dict[str, Any]) -> bool:
    return int(budgets.get("same_direction_full_s3_failures") or 0) + 1 > int(budgets.get("max_same_direction_full_s3_failures") or 1)


def _is_repairable_proxy_context(s3: dict[str, Any]) -> bool:
    proxy_decision = str(s3.get("proxy_decision") or "").lower()
    route_hint = str(s3.get("route_hint") or "").lower()
    failure_class = str(s3.get("failure_class") or s3.get("candidate_failure_class") or "").lower()
    if proxy_decision == "proxy_repairable":
        return True
    haystack = " ".join(item for item in [route_hint, failure_class] if item)
    repair_markers = [
        "effect_first_proxy_repair",
        "repairable_proxy",
        "proxy_repairable",
        "patch_gate_not_passed",
        "static_patch_gate_failed",
        "patch_repair",
        "implementation_failure",
    ]
    return any(marker in haystack for marker in repair_markers)


def _same_direction_failure_bucket(failure_class: str, reason_text: str) -> str | None:
    haystack = f"{failure_class} {reason_text}".lower()
    if any(
        marker in haystack
        for marker in [
            "repairable_proxy",
            "proxy_repairable",
            "effect_first_proxy_repair",
            "patch_repair",
            "implementation_failure",
        ]
    ):
        return None
    if "full_s3" in haystack:
        return "full_s3"
    if "proxy" in haystack or "method" in haystack:
        return "proxy"
    return None


def _sum_registry_repair_routes(registry: dict[str, Any], variant_id: str) -> int:
    routes = registry.get("implementation_repair_routes_by_candidate") if isinstance(registry.get("implementation_repair_routes_by_candidate"), dict) else {}
    if not routes:
        return 0
    if variant_id:
        return sum(int(value or 0) for key, value in routes.items() if variant_id in str(key))
    return sum(int(value or 0) for value in routes.values())


def _direction_id(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ["direction_id", "id"]:
            if payload.get(key):
                return str(payload[key])
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        if summary.get("direction_id"):
            return str(summary["direction_id"])
    return "unknown_direction"


def _variant_id(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in ["selected_variant_id", "variant_id", "candidate_id", "selected_candidate_id"]:
            if payload.get(key):
                return str(payload[key])
        selected = payload.get("selected_patch") if isinstance(payload.get("selected_patch"), dict) else {}
        if selected.get("candidate_id"):
            return str(selected["candidate_id"])
    return ""


def _selected_variant_score(scorecard: dict[str, Any]) -> float | None:
    if not isinstance(scorecard, dict):
        return None
    rows = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
    selected = next((item for item in rows if item.get("decision") == "selected"), None)
    if selected and selected.get("score") is not None:
        try:
            return float(selected["score"])
        except (TypeError, ValueError):
            return None
    return None


def _novelty_from_direction(direction: dict[str, Any]) -> float | None:
    value = direction.get("novelty_score") if isinstance(direction, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _has_full_s3_metrics(main_results: dict[str, Any]) -> bool:
    candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)] if isinstance(main_results, dict) else []
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        if metrics.get("mean") is not None or (isinstance(metrics.get("datasets"), dict) and metrics.get("datasets")):
            return True
    best = main_results.get("best_candidate") if isinstance(main_results.get("best_candidate"), dict) else {}
    metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
    return metrics.get("mean") is not None or (isinstance(metrics.get("datasets"), dict) and metrics.get("datasets"))


def _main_results_failure_class(main_results: dict[str, Any]) -> str | None:
    if not isinstance(main_results, dict):
        return None
    candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
    if any(_candidate_resource_retry(item) for item in candidates):
        return "resource_retry"
    if any(_candidate_implementation_failure(item) for item in candidates):
        return "implementation_failure"
    if any((item.get("decision") in {"proxy_rejected", "proxy_repairable"}) for item in candidates):
        return "proxy_negative"
    if main_results.get("acceptance") and (main_results.get("acceptance") or {}).get("passed") is False:
        return "full_s3_method_failure"
    return None


def _candidate_resource_retry(candidate: dict[str, Any]) -> bool:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    return bool(proxy.get("resource_retry") or proxy.get("status") == "resource_retry" or proxy.get("failure_category") in {"s3_proxy_resource_oom", "s3_proxy_gpu_resource_retry"})


def _candidate_implementation_failure(candidate: dict[str, Any]) -> bool:
    if _candidate_resource_retry(candidate):
        return False
    if candidate.get("decision") in {"patch_rejected", "failed_no_metrics"}:
        return True
    if candidate.get("command_status") in {"patch_rejected", "failed"}:
        return True
    attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
    if attribution.get("primary_failure") in {"proxy_eval_output_health_failure", "proxy_activation_smoke_no_effect", "patch_rejected", "failed_no_metrics"}:
        return True
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    return bool(proxy.get("command_failure") or proxy.get("proxy_eval_health_failure"))


def _is_resource_retry_context(context: dict[str, Any]) -> bool:
    s3 = context.get("s3") if isinstance(context.get("s3"), dict) else {}
    trigger = context.get("trigger") if isinstance(context.get("trigger"), dict) else {}
    haystack = " ".join(
        str(item)
        for item in [
            s3.get("route_hint"),
            s3.get("failure_class"),
            s3.get("candidate_failure_class"),
            trigger.get("reason"),
            trigger.get("source"),
        ]
        if item
    ).lower()
    return "resource_retry" in haystack or "resource" in haystack and "retry" in haystack or s3.get("route_hint") == "block_resource"


def _context_failure_class(context: dict[str, Any]) -> str | None:
    for section in ["s3", "s2_5"]:
        payload = context.get(section) if isinstance(context.get(section), dict) else {}
        if payload.get("failure_class"):
            return str(payload["failure_class"])
    return None
