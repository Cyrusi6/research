"""Project report assembly for monitoring active research loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_project_config, load_root_config
from .direction_contracts import direction_planner_seed
from .method_memory import load_shared_method_memory, shared_method_memory_for_prompt
from .registry import load_registry
from .utils import read_json


def build_project_report(project_root: Path) -> dict[str, Any]:
    registry = load_registry(project_root / "meta" / "registry.yaml")
    direction = read_json(project_root / "literature" / "direction.json", default={}) or {}
    ideas = [direction_planner_seed(direction)] if isinstance(direction, dict) and direction.get("direction_id") else []
    direction_scorecard = read_json(project_root / "plan" / "direction_scorecard.json", default={}) or {}
    performance_feedback = read_json(project_root / "plan" / "performance_feedback.json", default={}) or {}
    main_results = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    trial_result = read_json(project_root / "experiment" / "results" / "trial_result.json", default={}) or {}
    research_state = read_json(project_root / "meta" / "research_state.json", default={}) or {}
    readiness = read_json(project_root / "experiment" / "results" / "full_s3_readiness_report.json", default={}) or {}
    patch_manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
    route_decision = read_json(project_root / "meta" / "route_outcome.json", default={}) or {}
    attempt_ledger = research_state
    evidence_quality = read_json(project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
    planner_gate = read_json(project_root / "plan" / "s2_planner" / "planner_gate_report.json", default={}) or {}
    variant_scorecard = read_json(project_root / "plan" / "s2_planner" / "variant_scorecard.json", default={}) or {}
    patch_gate = read_json(project_root / "plan" / "code_patches" / "patch_gate_report.json", default={}) or {}
    proxy_decision = read_json(project_root / "experiment" / "results" / "c2c_proxy_decision_report.json", default={}) or {}
    worthiness = read_json(project_root / "experiment" / "results" / "c2c_full_s3_worthiness.json", default={}) or {}
    e2e_readiness = read_json(project_root / "meta" / "c2c_e2e_readiness_report.json", default={}) or {}
    e2e_audit = read_json(project_root / "meta" / "c2c_artifact_audit_report.json", default={}) or {}
    e2e_manifest = read_json(project_root / "meta" / "c2c_e2e_run_manifest.json", default={}) or {}
    e2e_replay = read_json(project_root / "meta" / "c2c_replay_result.json", default={}) or {}
    e2e_smoke = read_json(project_root / "meta" / "c2c_real_smoke_record.json", default={}) or {}
    selected_idea = _selected_idea(ideas)
    current_direction = _current_direction(selected_idea, direction_scorecard)
    attempts = _direction_attempts(direction_scorecard, main_results)
    best_patch = _best_patch(main_results, patch_manifest)
    stage_states = _stage_states(main_results, readiness, best_patch)
    latest_failure = _latest_failure(performance_feedback, main_results, registry)
    next_route = _next_route(performance_feedback, registry, route_decision)
    return {
        "schema_version": "auto_research_project_report_v1",
        "project_id": registry.get("project_id") or project_root.name,
        "research_topic": registry.get("research_topic"),
        "status": registry.get("status"),
        "current_stage": registry.get("current_stage"),
        "iteration": registry.get("iteration"),
        "blocked_reason": registry.get("blocked_reason"),
        "pause_type": registry.get("pause_type"),
        "resume_instruction": registry.get("resume_instruction"),
        "s1_direction": current_direction,
        "same_direction_attempt": _same_direction_attempt(direction_scorecard, performance_feedback),
        "proxy_attempts": attempts,
        "current_best_patch": best_patch,
        "stage_states": stage_states,
        "latest_failure": latest_failure,
        "next_route": next_route,
        "route": _route_report(route_decision),
        "s1_quality": _s1_quality_report(evidence_quality),
        "s2_planner": _s2_planner_report(planner_gate, variant_scorecard),
        "s2_5_patch": _s2_5_patch_report(patch_gate),
        "s3_proxy": _s3_proxy_report(proxy_decision, worthiness),
        "attempt_ledger": _attempt_ledger_report(attempt_ledger, current_direction),
        "e2e": _e2e_report(e2e_readiness, e2e_audit, e2e_manifest, e2e_replay, e2e_smoke),
        "artifact_paths": {
            "direction": "literature/direction.json" if (project_root / "literature" / "direction.json").exists() else None,
            "variant": "plan/variant.json" if (project_root / "plan" / "variant.json").exists() else None,
            "direction_scorecard": "plan/direction_scorecard.json" if (project_root / "plan" / "direction_scorecard.json").exists() else None,
            "performance_feedback": "plan/performance_feedback.json" if (project_root / "plan" / "performance_feedback.json").exists() else None,
            "main_results": "experiment/results/main_results.json" if (project_root / "experiment" / "results" / "main_results.json").exists() else None,
            "full_s3_readiness_report": "experiment/results/full_s3_readiness_report.json" if (project_root / "experiment" / "results" / "full_s3_readiness_report.json").exists() else None,
            "patch_manifest": "plan/code_patches/patch_manifest.json" if (project_root / "plan" / "code_patches" / "patch_manifest.json").exists() else None,
            "route_outcome": "meta/route_outcome.json" if (project_root / "meta" / "route_outcome.json").exists() else None,
            "research_state": "meta/research_state.json" if (project_root / "meta" / "research_state.json").exists() else None,
            "s1_evidence_quality": "literature/c2c/evidence_quality_score.json" if (project_root / "literature" / "c2c" / "evidence_quality_score.json").exists() else None,
            "planner_gate": "plan/s2_planner/planner_gate_report.json" if (project_root / "plan" / "s2_planner" / "planner_gate_report.json").exists() else None,
            "patch_gate": "plan/code_patches/patch_gate_report.json" if (project_root / "plan" / "code_patches" / "patch_gate_report.json").exists() else None,
            "proxy_decision": "experiment/results/c2c_proxy_decision_report.json" if (project_root / "experiment" / "results" / "c2c_proxy_decision_report.json").exists() else None,
            "c2c_e2e_readiness": "meta/c2c_e2e_readiness_report.json" if (project_root / "meta" / "c2c_e2e_readiness_report.json").exists() else None,
            "c2c_artifact_audit": "meta/c2c_artifact_audit_report.json" if (project_root / "meta" / "c2c_artifact_audit_report.json").exists() else None,
            "c2c_e2e_run_manifest": "meta/c2c_e2e_run_manifest.json" if (project_root / "meta" / "c2c_e2e_run_manifest.json").exists() else None,
            "c2c_replay_result": "meta/c2c_replay_result.json" if (project_root / "meta" / "c2c_replay_result.json").exists() else None,
            "c2c_real_smoke_record": "meta/c2c_real_smoke_record.json" if (project_root / "meta" / "c2c_real_smoke_record.json").exists() else None,
        },
    }


def format_project_report(report: dict[str, Any]) -> str:
    direction = report.get("s1_direction") if isinstance(report.get("s1_direction"), dict) else {}
    same = report.get("same_direction_attempt") if isinstance(report.get("same_direction_attempt"), dict) else {}
    best = report.get("current_best_patch") if isinstance(report.get("current_best_patch"), dict) else {}
    states = report.get("stage_states") if isinstance(report.get("stage_states"), dict) else {}
    failure = report.get("latest_failure") if isinstance(report.get("latest_failure"), dict) else {}
    route = report.get("next_route") if isinstance(report.get("next_route"), dict) else {}
    route_report = report.get("route") if isinstance(report.get("route"), dict) else {}
    e2e = report.get("e2e") if isinstance(report.get("e2e"), dict) else {}
    lines = [
        f"Project: {report.get('project_id')}",
        f"Status: {report.get('status')} | Stage: {report.get('current_stage')} | Iteration: {report.get('iteration')}",
        f"S1 direction: {direction.get('direction_id') or 'unknown'} ({direction.get('title') or direction.get('mechanism_type') or 'n/a'})",
        f"Same-direction attempt: {same.get('count', 0)}/{same.get('budget') or '?'}",
        "",
        "Proxy attempts:",
    ]
    attempts = report.get("proxy_attempts") if isinstance(report.get("proxy_attempts"), list) else []
    if attempts:
        for item in attempts[-8:]:
            lines.append(
                "- "
                f"{item.get('iteration', '?')}:{item.get('candidate_id') or 'candidate'} "
                f"decision={item.get('decision') or 'unknown'} "
                f"proxy_delta={_fmt(item.get('proxy_delta'))} "
                f"full_delta={_fmt(item.get('full_delta'))} "
                f"dragging={','.join(item.get('dragging_datasets') or []) or 'none'}"
            )
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "",
            f"Best patch: {best.get('candidate_id') or 'none'} decision={best.get('decision') or 'n/a'} proxy_delta={_fmt(best.get('proxy_delta'))} full_delta={_fmt(best.get('full_delta'))}",
            f"Activation no-op: {_yes_no(states.get('activation_no_op'))}",
            f"Stage states: proxy={states.get('proxy') or 'n/a'} activation={states.get('activation') or 'n/a'} full={states.get('full') or 'n/a'} ablation={states.get('ablation') or 'n/a'}",
            f"Recent failure: {failure.get('reason') or 'none'}",
            f"Next route: {route.get('action') or states.get('next_route_hint') or 'unknown'} ({route.get('reason') or states.get('next_route_hint') or 'no reason recorded'})",
            f"Route decision: {route_report.get('last_decision') or 'none'} -> {route_report.get('next_stage') or 'n/a'}",
            f"C2C E2E: readiness={e2e.get('readiness_gate') or 'n/a'} audit={e2e.get('artifact_audit_gate') or 'n/a'} replay={e2e.get('replay', {}).get('last_replay_status') if isinstance(e2e.get('replay'), dict) else 'n/a'}",
        ]
    )
    if report.get("status") == "retryable_paused":
        lines.extend(
            [
                f"Pause type: {report.get('pause_type') or 'retryable'}",
                f"Resume: {report.get('resume_instruction') or 'wait for quota recovery, then resume this project'}",
            ]
        )
    return "\n".join(lines)


def build_memory_report(*, config: dict[str, Any] | None = None, project_root: Path | None = None, prompt_limit: int | None = None) -> dict[str, Any]:
    effective_config = config or (load_project_config(project_root) if project_root else load_root_config())
    memory = load_shared_method_memory(effective_config, limit=1_000_000)
    project_memory = shared_method_memory_for_prompt(effective_config, limit=prompt_limit) if project_root else None
    entries = [item for item in memory.get("entries") or [] if isinstance(item, dict)]
    return {
        "schema_version": "auto_research_memory_report_v1",
        "status": "ok" if memory.get("enabled") else "disabled",
        "path": memory.get("path"),
        "entry_count": memory.get("entry_count", len(entries)),
        "method_failure_count": len(entries),
        "top_failed_mechanisms": _top_failed_mechanisms(entries),
        "top_dragging_datasets": _top_dragging_datasets(entries),
        "recent_memory": [_memory_report_entry(item) for item in _recent_memory(entries, limit=8)],
        "project_retrieval": {
            "project_id": project_root.name if project_root else None,
            "prompt_limit": prompt_limit,
            "retrieved_count": len((project_memory or {}).get("recent_entries") or []),
            "memory_ids": [item.get("memory_id") for item in (project_memory or {}).get("recent_entries") or [] if isinstance(item, dict)],
            "entries": (project_memory or {}).get("recent_entries") or [],
        }
        if project_root
        else None,
        "summary": memory.get("summary") or {},
    }


def format_memory_report(report: dict[str, Any]) -> str:
    lines = [
        "Shared Method Memory",
        f"Status: {report.get('status')}",
        f"Path: {report.get('path') or 'n/a'}",
        f"Method failures: {report.get('method_failure_count', 0)}",
        "",
        "Top failed mechanisms:",
    ]
    mechanisms = report.get("top_failed_mechanisms") if isinstance(report.get("top_failed_mechanisms"), list) else []
    if mechanisms:
        for item in mechanisms[:8]:
            lines.append(f"- {item.get('mechanism_type')}: count={item.get('count')} false_positive={item.get('proxy_false_positive_count', 0)}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Top dragging datasets:")
    datasets = report.get("top_dragging_datasets") if isinstance(report.get("top_dragging_datasets"), list) else []
    if datasets:
        for item in datasets[:8]:
            lines.append(f"- {item.get('dataset')}: count={item.get('count')} max_regression={_fmt(item.get('max_regression'))}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Recent memory:")
    recent = report.get("recent_memory") if isinstance(report.get("recent_memory"), list) else []
    if recent:
        for item in recent[:8]:
            lines.append(
                "- "
                f"{item.get('memory_id')} project={item.get('project_id') or 'n/a'} "
                f"priority={_fmt(item.get('priority'))} route={item.get('route') or 'n/a'} "
                f"mechanism={item.get('mechanism_type') or 'unknown'}"
            )
    else:
        lines.append("- none")
    project = report.get("project_retrieval") if isinstance(report.get("project_retrieval"), dict) else None
    if project:
        lines.extend(["", f"Current project retrieval: {project.get('project_id')}", f"Retrieved memories: {project.get('retrieved_count', 0)}"])
        entries = project.get("entries") if isinstance(project.get("entries"), list) else []
        if entries:
            for item in entries[:8]:
                quality = item.get("memory_quality") if isinstance(item.get("memory_quality"), dict) else {}
                signals = ",".join(str(signal) for signal in quality.get("signals") or [])
                lines.append(f"- {item.get('memory_id')} priority={_fmt(quality.get('priority'))} signals={signals or 'none'}")
        else:
            lines.append("- none")
    return "\n".join(lines)


def _read_list(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default=[])
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _selected_idea(ideas: list[dict[str, Any]]) -> dict[str, Any]:
    return next((idea for idea in ideas if idea.get("selected")), ideas[0] if ideas else {})


def _current_direction(selected: dict[str, Any], scorecard: dict[str, Any]) -> dict[str, Any]:
    current = scorecard.get("current_direction") if isinstance(scorecard.get("current_direction"), dict) else {}
    summary = current.get("summary") if isinstance(current.get("summary"), dict) else {}
    selected_id = selected.get("s1_direction_id") or selected.get("direction_id") or selected.get("id")
    selected_title = selected.get("title")
    selected_mechanism = selected.get("mechanism_type")
    return {
        "direction_id": selected_id or current.get("direction_id"),
        "title": selected_title or current.get("title"),
        "mechanism_type": selected_mechanism or current.get("mechanism_type"),
        "quality": summary.get("direction_quality") if current.get("direction_id") == selected_id else None,
        "best_proxy_delta": summary.get("best_proxy_delta") if current.get("direction_id") == selected_id else None,
        "previous_direction_id": current.get("direction_id") if selected_id and current.get("direction_id") != selected_id else None,
    }


def _same_direction_attempt(scorecard: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    current = scorecard.get("current_direction") if isinstance(scorecard.get("current_direction"), dict) else {}
    summary = current.get("summary") if isinstance(current.get("summary"), dict) else {}
    perf_summary = feedback.get("summary") if isinstance(feedback.get("summary"), dict) else {}
    count = summary.get("same_direction_failure_count", perf_summary.get("same_direction_failure_count"))
    budget = summary.get("same_direction_failure_budget", perf_summary.get("same_direction_failure_budget"))
    return {
        "count": _int_or_zero(count),
        "budget": budget,
        "attempt_count": summary.get("attempt_count") or current.get("attempt_count"),
        "quality": summary.get("direction_quality"),
    }


def _direction_attempts(scorecard: dict[str, Any], main_results: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    current = scorecard.get("current_direction") if isinstance(scorecard.get("current_direction"), dict) else {}
    for attempt in current.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        attempts.append(
            {
                "iteration": attempt.get("iteration"),
                "candidate_id": ",".join(str(item) for item in attempt.get("candidate_ids") or [] if item),
                "decision": attempt.get("route"),
                "proxy_delta": attempt.get("best_proxy_delta"),
                "full_delta": None,
                "dragging_datasets": [str(item.get("dataset")) for item in attempt.get("dragging_datasets") or [] if isinstance(item, dict) and item.get("dataset")],
            }
        )
    existing_ids = {item.get("candidate_id") for item in attempts}
    for candidate in main_results.get("candidate_results") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("id") or candidate.get("candidate_id") or "")
        if candidate_id in existing_ids:
            continue
        proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
        attempts.append(
            {
                "iteration": main_results.get("iteration"),
                "candidate_id": candidate_id,
                "decision": candidate.get("decision"),
                "proxy_delta": proxy.get("proxy_delta_vs_comparison_baseline", proxy.get("proxy_delta_vs_proxy_baseline", proxy.get("proxy_delta_vs_baseline"))),
                "full_delta": candidate.get("delta_vs_baseline"),
                "dragging_datasets": _candidate_dragging_datasets(candidate),
            }
        )
    return attempts


def _best_patch(main_results: dict[str, Any], patch_manifest: dict[str, Any]) -> dict[str, Any]:
    candidate = main_results.get("best_candidate") if isinstance(main_results.get("best_candidate"), dict) else {}
    if not candidate:
        candidate = main_results.get("best_proxy_candidate") if isinstance(main_results.get("best_proxy_candidate"), dict) else {}
    selected_patch = patch_manifest.get("selected_patch") if isinstance(patch_manifest.get("selected_patch"), dict) else {}
    selected_candidate_id = patch_manifest.get("selected_candidate_id")
    candidate_id = candidate.get("id") or candidate.get("candidate_id")
    if candidate_id:
        patch_candidate = _patch_manifest_candidate(patch_manifest, candidate_id)
        if not patch_candidate and str(candidate_id) == str(selected_candidate_id or ""):
            patch_candidate = selected_patch
    else:
        patch_candidate = selected_patch or _patch_manifest_candidate(patch_manifest, selected_candidate_id)
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    return {
        "candidate_id": candidate_id or patch_candidate.get("candidate_id") or patch_candidate.get("id") or selected_candidate_id,
        "title": candidate.get("title") or patch_candidate.get("title"),
        "decision": candidate.get("decision") or patch_candidate.get("status"),
        "proxy_delta": proxy.get("proxy_delta_vs_comparison_baseline", proxy.get("proxy_delta_vs_proxy_baseline", proxy.get("proxy_delta_vs_baseline"))),
        "full_delta": candidate.get("delta_vs_baseline"),
        "changed_files": (candidate.get("patch_result") or {}).get("changed_files") if isinstance(candidate.get("patch_result"), dict) else patch_candidate.get("changed_files"),
        "patch_status": patch_candidate.get("status"),
        "patch_json": patch_candidate.get("patch_json"),
        "selected_variant": patch_candidate.get("selected_variant"),
        "quality_score": patch_candidate.get("quality_score"),
    }


def _stage_states(main_results: dict[str, Any], readiness: dict[str, Any], best_patch: dict[str, Any]) -> dict[str, Any]:
    candidate = _report_candidate_for_states(main_results, readiness, best_patch)
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    activation = candidate.get("activation_smoke") if isinstance(candidate.get("activation_smoke"), dict) else {}
    full_readiness = candidate.get("full_s3_readiness") if isinstance(candidate.get("full_s3_readiness"), dict) else readiness if isinstance(readiness, dict) else {}
    ablation = candidate.get("ablation") if isinstance(candidate.get("ablation"), dict) else {}
    full_state = _full_state(candidate, full_readiness, main_results)
    activation_status = activation.get("status") or ((full_readiness.get("activation_smoke") or {}).get("status") if isinstance(full_readiness.get("activation_smoke"), dict) else None)
    activation_no_op = _activation_no_op(activation, full_readiness)
    return {
        "candidate_id": candidate.get("id") or candidate.get("candidate_id") or best_patch.get("candidate_id") or readiness.get("candidate_id"),
        "proxy": proxy.get("status") or ((full_readiness.get("proxy") or {}).get("status") if isinstance(full_readiness.get("proxy"), dict) else None),
        "activation": activation_status,
        "activation_no_op": activation_no_op,
        "full": full_state,
        "ablation": ablation.get("status") or ((main_results.get("ablation_summary") or {}).get("status") if isinstance(main_results.get("ablation_summary"), dict) else None),
        "readiness": full_readiness.get("status"),
        "full_train_allowed": full_readiness.get("full_train_allowed"),
        "next_route_hint": _route_hint_from_states(proxy, activation, full_state, ablation, activation_no_op),
    }


def _report_candidate_for_states(main_results: dict[str, Any], readiness: dict[str, Any], best_patch: dict[str, Any]) -> dict[str, Any]:
    target_ids = [
        best_patch.get("candidate_id"),
        readiness.get("candidate_id") if isinstance(readiness, dict) else None,
        ((main_results.get("best_candidate") or {}).get("id") if isinstance(main_results.get("best_candidate"), dict) else None),
        ((main_results.get("best_proxy_candidate") or {}).get("id") if isinstance(main_results.get("best_proxy_candidate"), dict) else None),
    ]
    candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
    for target in target_ids:
        if target:
            match = next((item for item in candidates if str(item.get("id") or item.get("candidate_id") or "") == str(target)), None)
            if match:
                return match
    for key in ["best_candidate", "best_proxy_candidate"]:
        value = main_results.get(key)
        if isinstance(value, dict):
            return value
    return candidates[-1] if candidates else {}


def _full_state(candidate: dict[str, Any], readiness: dict[str, Any], main_results: dict[str, Any]) -> str | None:
    if (candidate.get("metrics") or {}).get("mean") is not None:
        return "completed"
    decision = candidate.get("decision")
    command_status = candidate.get("command_status")
    if decision in {"proxy_rejected", "proxy_repairable"}:
        return "blocked_before_full"
    if command_status in {"failed", "partial", "blocked"}:
        return str(command_status)
    if readiness.get("full_train_allowed") is True:
        return "ready_or_running"
    if readiness.get("full_train_allowed") is False:
        return "not_ready"
    acceptance = main_results.get("acceptance") if isinstance(main_results.get("acceptance"), dict) else {}
    if acceptance.get("best_mean") is not None:
        return "completed"
    return None


def _activation_no_op(activation: dict[str, Any], readiness: dict[str, Any]) -> bool | None:
    if isinstance(activation, dict) and activation:
        comparison = activation.get("comparison") if isinstance(activation.get("comparison"), dict) else {}
        if activation.get("status") == "failed" and comparison.get("mechanism_observed") is False:
            return True
        if activation.get("status") == "passed":
            return False
    readiness_activation = readiness.get("activation_smoke") if isinstance(readiness.get("activation_smoke"), dict) else {}
    if "no_op" in readiness_activation:
        return bool(readiness_activation.get("no_op"))
    if readiness_activation.get("status") == "passed":
        return False
    return None


def _route_hint_from_states(proxy: dict[str, Any], activation: dict[str, Any], full_state: str | None, ablation: dict[str, Any], activation_no_op: bool | None) -> str:
    if proxy.get("status") == "repairable_proxy_risk" or activation_no_op:
        return "repair"
    if proxy.get("status") == "rejected":
        return "new_variant"
    if full_state in {"failed", "partial", "blocked"}:
        return "repair"
    if full_state == "completed" and ablation.get("status") in {"partial", "failed"}:
        return "repair"
    return "continue"


def _patch_manifest_candidate(patch_manifest: dict[str, Any], candidate_id: Any) -> dict[str, Any]:
    candidates = patch_manifest.get("candidates") or patch_manifest.get("patches") or patch_manifest.get("ideas") or []
    for item in candidates:
        if isinstance(item, dict) and str(item.get("id") or item.get("candidate_id") or "") == str(candidate_id or ""):
            return item
    return next((item for item in candidates if isinstance(item, dict) and item.get("status") == "ok"), {})


def _latest_failure(feedback: dict[str, Any], main_results: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    summary = feedback.get("summary") if isinstance(feedback.get("summary"), dict) else {}
    if summary:
        return {
            "reason": feedback.get("reason") or summary.get("repair_vs_variant_reason") or summary.get("route"),
            "route": summary.get("route"),
            "signals": summary.get("repair_vs_variant_signals") or [],
        }
    candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
    if candidates:
        candidate = candidates[-1]
        attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
        proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
        return {"reason": attribution.get("primary_failure") or proxy.get("reason") or candidate.get("decision"), "route": candidate.get("decision"), "signals": []}
    return {"reason": registry.get("blocked_reason"), "route": None, "signals": []}


def _route_report(route_decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(route_decision, dict) or not route_decision:
        return {"last_decision": None, "next_stage": None, "reason_codes": [], "budget_effects": {}}
    return {
        "last_decision": route_decision.get("next_action"),
        "next_stage": route_decision.get("next_action"),
        "reason_codes": route_decision.get("reason_codes") or [],
        "budget_effects": route_decision.get("budget_snapshot") if isinstance(route_decision.get("budget_snapshot"), dict) else {},
        "idempotency_key": route_decision.get("idempotency_key"),
    }


def _s1_quality_report(evidence_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_gate": evidence_quality.get("gate") or evidence_quality.get("status"),
        "novelty_score": evidence_quality.get("novelty_score"),
        "support_coverage": evidence_quality.get("support_coverage") if isinstance(evidence_quality.get("support_coverage"), dict) else {},
    }


def _s2_planner_report(planner_gate: dict[str, Any], variant_scorecard: dict[str, Any]) -> dict[str, Any]:
    return {
        "planner_gate": planner_gate.get("gate") or planner_gate.get("status"),
        "selected_variant_id": planner_gate.get("selected_variant_id") or variant_scorecard.get("selected_variant_id"),
        "selected_variant_score": _selected_variant_score(variant_scorecard),
    }


def _s2_5_patch_report(patch_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_gate": patch_gate.get("gate") or patch_gate.get("status"),
        "failure_class": patch_gate.get("failure_class"),
        "repairable": patch_gate.get("repairable"),
    }


def _s3_proxy_report(proxy_decision: dict[str, Any], worthiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": proxy_decision.get("decision"),
        "route_hint": proxy_decision.get("route_hint"),
        "failure_class": proxy_decision.get("failure_class"),
        "worthiness_score": worthiness.get("score") or ((proxy_decision.get("full_s3_worthiness") or {}).get("score") if isinstance(proxy_decision.get("full_s3_worthiness"), dict) else None),
    }


def _attempt_ledger_report(attempt_ledger: dict[str, Any], current_direction: dict[str, Any]) -> dict[str, Any]:
    direction_hash = attempt_ledger.get("current_direction_semantic_hash") if isinstance(attempt_ledger, dict) else None
    direction_state = ((attempt_ledger.get("directions") or {}).get(direction_hash) or {}) if direction_hash else {}
    budget = direction_state.get("budget") if isinstance(direction_state.get("budget"), dict) else {}
    attempts = attempt_ledger.get("attempts") if isinstance(attempt_ledger.get("attempts"), dict) else {}
    return {
        "target": budget.get("target", 5),
        "reserved": budget.get("reserved", 0),
        "consumed": budget.get("consumed", 0),
        "record_count": len(attempts),
        "method_tried_count": len(attempt_ledger.get("method_tried_history") or []),
    }


def _e2e_report(
    readiness: dict[str, Any],
    audit: dict[str, Any],
    manifest: dict[str, Any],
    replay: dict[str, Any],
    smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "readiness_gate": readiness.get("gate") if isinstance(readiness, dict) else None,
        "artifact_audit_gate": audit.get("gate") if isinstance(audit, dict) else None,
        "real_run_manifest": {
            "mode": manifest.get("mode"),
            "final_status": manifest.get("final_status"),
            "stage_boundaries": manifest.get("stage_boundaries") if isinstance(manifest.get("stage_boundaries"), dict) else {},
        }
        if isinstance(manifest, dict) and manifest
        else {},
        "artifact_audit_summary": audit.get("summary") if isinstance(audit, dict) and isinstance(audit.get("summary"), dict) else {},
        "replay": {
            "last_replay_status": replay.get("status"),
            "mismatches": replay.get("mismatches") if isinstance(replay.get("mismatches"), list) else [],
        }
        if isinstance(replay, dict) and replay
        else {},
        "real_smoke_record": _real_smoke_report(smoke or {}),
    }


def _real_smoke_report(smoke: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(smoke, dict) or not smoke:
        return {}
    return {
        "readiness_gate": smoke.get("readiness_gate"),
        "run_manifest_final_status": smoke.get("run_manifest_final_status"),
        "artifact_audit_gate": smoke.get("artifact_audit_gate"),
        "replay_status": smoke.get("replay_status"),
        "last_stage": smoke.get("last_stage"),
        "s1_evidence_gate": smoke.get("s1_evidence_gate"),
        "s2_planner_gate": smoke.get("s2_planner_gate"),
        "s2_5_patch_gate": smoke.get("s2_5_patch_gate"),
        "s3_proxy_decision": smoke.get("s3_proxy_decision"),
        "route_decision": smoke.get("route_decision"),
        "blocking_reasons": smoke.get("blocking_reasons") if isinstance(smoke.get("blocking_reasons"), list) else [],
        "warnings": smoke.get("warnings") if isinstance(smoke.get("warnings"), list) else [],
    }


def _selected_variant_score(scorecard: dict[str, Any]) -> float | None:
    if not isinstance(scorecard, dict):
        return None
    rows = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
    selected = next((item for item in rows if item.get("decision") == "selected"), None)
    try:
        return float(selected["score"]) if selected and selected.get("score") is not None else None
    except (TypeError, ValueError):
        return None


def _next_route(feedback: dict[str, Any], registry: dict[str, Any], route_decision: dict[str, Any] | None = None) -> dict[str, Any]:
    if registry.get("status") == "retryable_paused":
        pause_type = registry.get("pause_type") or "retryable_paused"
        action = "resume_after_gpu_resource_available" if pause_type == "runtime_smoke_resource_retry" else "resume_after_quota_recovery"
        return {
            "action": action,
            "reason": registry.get("blocked_reason") or "retryable external quota/rate-limit pause",
            "matched_rule": pause_type,
            "next_action": registry.get("resume_instruction"),
        }
    if isinstance(route_decision, dict) and route_decision.get("next_action"):
        reason_codes = route_decision.get("reason_codes") or []
        return {
            "action": route_decision.get("next_action"),
            "reason": ", ".join(str(item) for item in reason_codes),
            "matched_rule": reason_codes[0] if reason_codes else None,
            "next_action": route_decision.get("next_action"),
        }
    summary = feedback.get("summary") if isinstance(feedback.get("summary"), dict) else {}
    policy = summary.get("s2_action_policy") if isinstance(summary.get("s2_action_policy"), dict) else {}
    action = policy.get("action") or summary.get("recommended_s2_action")
    if not action:
        current_stage = registry.get("current_stage")
        action = "continue_current_stage" if registry.get("status") == "running" else "inspect_blocked_stage"
        reason = f"registry status={registry.get('status')} current_stage={current_stage}"
    else:
        reason = policy.get("reason") or summary.get("repair_vs_variant_reason")
    return {
        "action": action,
        "reason": reason,
        "matched_rule": policy.get("matched_rule"),
        "next_action": summary.get("next_action"),
    }


def _candidate_dragging_datasets(candidate: dict[str, Any]) -> list[str]:
    attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
    dragging = attribution.get("dragging_datasets") if isinstance(attribution.get("dragging_datasets"), list) else []
    names = [str(item.get("dataset")) for item in dragging if isinstance(item, dict) and item.get("dataset")]
    if names:
        return names
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    deltas = proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {}
    return [str(dataset) for dataset, delta in deltas.items() if _float_or_none(delta) is not None and float(delta) < 0]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_failed_mechanisms(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for mechanism in _entry_mechanisms(entry):
            item = stats.setdefault(mechanism, {"mechanism_type": mechanism, "count": 0, "proxy_false_positive_count": 0, "memory_ids": []})
            item["count"] += 1
            item["memory_ids"].append(entry.get("memory_id"))
            proxy = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
            if _entry_proxy_false_positive_count(proxy):
                item["proxy_false_positive_count"] += _entry_proxy_false_positive_count(proxy)
    ranked = sorted(stats.values(), key=lambda item: (item["count"], item["proxy_false_positive_count"], item["mechanism_type"]), reverse=True)
    for item in ranked:
        item["memory_ids"] = [memory_id for memory_id in item["memory_ids"][:8] if memory_id]
    return ranked[:12]


def _top_dragging_datasets(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for dataset, regression in _entry_dragging_datasets(entry).items():
            item = stats.setdefault(dataset, {"dataset": dataset, "count": 0, "max_regression": 0.0, "memory_ids": []})
            item["count"] += 1
            item["max_regression"] = max(float(item["max_regression"]), float(regression or 0.0))
            item["memory_ids"].append(entry.get("memory_id"))
    ranked = sorted(stats.values(), key=lambda item: (item["count"], item["max_regression"], item["dataset"]), reverse=True)
    for item in ranked:
        item["max_regression"] = round(float(item["max_regression"]), 4)
        item["memory_ids"] = [memory_id for memory_id in item["memory_ids"][:8] if memory_id]
    return ranked[:12]


def _recent_memory(entries: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: str(item.get("timestamp") or ""), reverse=True)[:limit]


def _memory_report_entry(entry: dict[str, Any]) -> dict[str, Any]:
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else {}
    return {
        "memory_id": entry.get("memory_id"),
        "timestamp": entry.get("timestamp"),
        "project_id": entry.get("project_id"),
        "route": entry.get("route"),
        "priority": quality.get("priority"),
        "signals": quality.get("signals") or [],
        "mechanism_type": next(iter(_entry_mechanisms(entry)), None),
        "dragging_datasets": sorted(_entry_dragging_datasets(entry)),
        "proxy_false_positive_count": _entry_proxy_false_positive_count(entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}),
    }


def _entry_mechanisms(entry: dict[str, Any]) -> set[str]:
    mechanisms: set[str] = set()
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    if method_context.get("mechanism_type"):
        mechanisms.add(str(method_context["mechanism_type"]))
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    if direction.get("mechanism_type"):
        mechanisms.add(str(direction["mechanism_type"]))
    for candidate in entry.get("entries") or []:
        if isinstance(candidate, dict):
            _add_mechanism_from_dict(mechanisms, candidate)
            for nested in candidate.get("candidate_results") or []:
                if isinstance(nested, dict):
                    _add_mechanism_from_dict(mechanisms, nested)
    proxy = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    summary = proxy.get("summary") if isinstance(proxy.get("summary"), dict) else {}
    for mechanism in (summary.get("mechanism_false_positive_summary") or {}).keys():
        mechanisms.add(str(mechanism))
    method_feedback = summary.get("method_feedback") if isinstance(summary.get("method_feedback"), dict) else {}
    for item in method_feedback.get("risky_mechanisms") or []:
        if isinstance(item, dict) and item.get("mechanism_type"):
            mechanisms.add(str(item["mechanism_type"]))
    for mechanism in proxy.get("risky_mechanisms") or []:
        if mechanism:
            mechanisms.add(str(mechanism))
    for item in proxy.get("false_positive_candidates") or []:
        if isinstance(item, dict) and item.get("mechanism_type"):
            mechanisms.add(str(item["mechanism_type"]))
    return mechanisms or {"unknown"}


def _add_mechanism_from_dict(mechanisms: set[str], value: dict[str, Any]) -> None:
    if value.get("mechanism_type"):
        mechanisms.add(str(value["mechanism_type"]))
    variant = value.get("s2_variant") if isinstance(value.get("s2_variant"), dict) else {}
    if variant.get("mechanism_type"):
        mechanisms.add(str(variant["mechanism_type"]))


def _entry_dragging_datasets(entry: dict[str, Any]) -> dict[str, float]:
    datasets: dict[str, float] = {}
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    for dataset, regression in (summary.get("dataset_regressions") or {}).items():
        _merge_dataset_regression(datasets, str(dataset), regression)
    for item in summary.get("dragging_datasets") or []:
        if isinstance(item, dict) and item.get("dataset"):
            _merge_dataset_regression(datasets, str(item["dataset"]), item.get("regression") or item.get("delta") or 0.0)
    for node in _walk_memory_dicts(entry.get("entries") or []):
        for dataset, regression in (node.get("dataset_regressions") or {}).items():
            _merge_dataset_regression(datasets, str(dataset), regression)
        for item in node.get("dragging_datasets") or []:
            if isinstance(item, dict) and item.get("dataset"):
                _merge_dataset_regression(datasets, str(item["dataset"]), item.get("regression") or item.get("delta") or 0.0)
        proxy_screen = node.get("proxy_screen") if isinstance(node.get("proxy_screen"), dict) else {}
        for dataset, regression in (proxy_screen.get("proxy_dataset_regressions") or {}).items():
            _merge_dataset_regression(datasets, str(dataset), regression)
        for dataset, delta in (proxy_screen.get("proxy_dataset_deltas") or {}).items():
            value = _float_or_none(delta)
            if value is not None and value < 0:
                _merge_dataset_regression(datasets, str(dataset), abs(value))
    proxy = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    for dataset in proxy.get("mispredicted_datasets") or []:
        if dataset:
            _merge_dataset_regression(datasets, str(dataset), 0.0)
    for section_key in ["current_iteration", "summary"]:
        section = proxy.get(section_key) if isinstance(proxy.get(section_key), dict) else {}
        for dataset, stats in (section.get("dataset_error_summary") or {}).items():
            if isinstance(stats, dict) and int(stats.get("misprediction_count") or 0) > 0:
                _merge_dataset_regression(datasets, str(dataset), stats.get("max_abs_proxy_full_delta_error") or stats.get("mean_abs_proxy_full_delta_error") or 0.0)
    return datasets


def _walk_memory_dicts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            result.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return result


def _merge_dataset_regression(target: dict[str, float], dataset: str, regression: Any) -> None:
    value = _float_or_none(regression)
    if value is None:
        value = 0.0
    target[dataset] = max(target.get(dataset, 0.0), abs(float(value)))


def _entry_proxy_false_positive_count(proxy_calibration: dict[str, Any]) -> int:
    for key in ["current_false_positive_count", "overall_false_positive_count", "proxy_false_positive_count"]:
        try:
            count = int(proxy_calibration.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            return count
    for key in ["current_iteration", "summary"]:
        section = proxy_calibration.get(key) if isinstance(proxy_calibration.get(key), dict) else {}
        try:
            count = int(section.get("proxy_false_positive_count") or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            return count
    return 0
