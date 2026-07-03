"""Deterministic S2 planner and S2.5 patch handoff contracts."""

from __future__ import annotations

import fnmatch
from typing import Any


C2C_S2_CANDIDATE_POOL_SCHEMA_VERSION = "c2c_s2_candidate_pool_v1"
C2C_S2_VARIANT_SCORECARD_SCHEMA_VERSION = "c2c_s2_variant_scorecard_v1"
C2C_S2_PLANNER_GATE_REPORT_SCHEMA_VERSION = "c2c_s2_planner_gate_report_v1"
C2C_S2_5_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION = "c2c_s2_5_implementation_contract_v1"
C2C_S2_5_PATCH_GATE_REPORT_SCHEMA_VERSION = "c2c_s2_5_patch_gate_report_v1"


def build_s2_candidate_pool(
    *,
    direction: dict[str, Any],
    candidates: list[dict[str, Any]],
    source: str = "s2_directional_planner",
    used_shared_memory_refs: list[str] | None = None,
) -> dict[str, Any]:
    normalized = [_candidate_for_pool(item) for item in candidates if isinstance(item, dict)]
    return {
        "schema_version": C2C_S2_CANDIDATE_POOL_SCHEMA_VERSION,
        "direction_id": str(direction.get("direction_id") or direction.get("id") or ""),
        "source": source,
        "candidate_count": len(normalized),
        "used_shared_memory_refs": list(used_shared_memory_refs or []),
        "candidates": normalized,
    }


def build_s2_variant_scorecard(
    *,
    direction: dict[str, Any],
    candidate_pool: dict[str, Any],
    selected_variant: dict[str, Any],
    evidence_quality: dict[str, Any] | None = None,
    variant_fingerprint: dict[str, Any] | None = None,
    planner_memory: dict[str, Any] | None = None,
    feedback: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_quality = evidence_quality if isinstance(evidence_quality, dict) else {}
    variant_fingerprint = variant_fingerprint if isinstance(variant_fingerprint, dict) else {}
    history = _history_summary(planner_memory if isinstance(planner_memory, dict) else {})
    selected_id = str(selected_variant.get("id") or selected_variant.get("variant_id") or "")
    ranking = []
    for candidate in candidate_pool.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        components, reasons = _score_components(
            direction=direction,
            candidate=candidate,
            evidence_quality=evidence_quality,
            variant_fingerprint=variant_fingerprint,
            history=history,
            feedback=feedback or [],
            config=config or {},
        )
        score = round(sum(float(value) for value in components.values()), 4)
        candidate_id = str(candidate.get("id") or "")
        ranking.append(
            {
                "variant_id": candidate_id,
                "variant_fingerprint": candidate.get("variant_fingerprint"),
                "score": score,
                "components": components,
                "decision": "selected" if candidate_id == selected_id else "rejected",
                "reasons": reasons,
            }
        )
    ranking.sort(key=lambda item: (item["decision"] == "selected", item["score"]), reverse=True)
    rejected = [
        {
            "variant_id": item["variant_id"],
            "score": item["score"],
            "reasons": item.get("reasons") or ["lower_ranked_variant"],
        }
        for item in ranking
        if item.get("decision") != "selected"
    ]
    if ranking and not any(item.get("decision") == "selected" for item in ranking):
        ranking[0]["decision"] = "selected"
        selected_id = str(ranking[0].get("variant_id") or selected_id)
        rejected = [
            {"variant_id": item["variant_id"], "score": item["score"], "reasons": item.get("reasons") or ["lower_ranked_variant"]}
            for item in ranking[1:]
        ]
    return {
        "schema_version": C2C_S2_VARIANT_SCORECARD_SCHEMA_VERSION,
        "direction_id": str(direction.get("direction_id") or direction.get("id") or ""),
        "selected_variant_id": selected_id,
        "ranking": ranking,
        "rejected_variants": rejected,
    }


def build_s2_planner_gate_report(
    *,
    direction: dict[str, Any],
    candidate_pool: dict[str, Any],
    scorecard: dict[str, Any],
    next_variant: dict[str, Any],
    variant_contract: dict[str, Any],
    variant_fingerprint: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    errors: list[str] = []
    direction_id = str(direction.get("direction_id") or direction.get("id") or "")
    selected_id = str(next_variant.get("id") or next_variant.get("variant_id") or "")
    selected_fp = str(next_variant.get("variant_fingerprint") or "")
    candidates = [item for item in candidate_pool.get("candidates") or [] if isinstance(item, dict)]
    candidate_ids = {str(item.get("id") or "") for item in candidates}
    candidate_fps = {str(item.get("variant_fingerprint") or "") for item in candidates}
    score_rows = [item for item in scorecard.get("ranking") or [] if isinstance(item, dict)]
    selected_rows = [item for item in score_rows if item.get("decision") == "selected"]
    selected_score = float(selected_rows[0].get("score") or 0.0) if selected_rows else 0.0
    if not candidates:
        errors.append("candidate_pool.candidates must be non-empty")
    if selected_id not in candidate_ids:
        errors.append("next_variant.id must exist in candidate_pool")
    expected_fp = str(variant_fingerprint.get("variant_fingerprint") or "")
    if selected_fp and expected_fp and selected_fp != expected_fp:
        errors.append("next_variant.variant_fingerprint must match variant_fingerprint.json")
    if selected_fp and candidate_fps and selected_fp not in candidate_fps:
        errors.append("next_variant.variant_fingerprint must come from candidate_pool")
    for key in ["direction_id", "s1_direction_id"]:
        value = next_variant.get(key)
        if value and direction_id and str(value) != direction_id:
            errors.append(f"next_variant.{key} must match S1 direction_id")
    if str(variant_contract.get("direction_id") or "") != direction_id:
        errors.append("variant_contract.direction_id must match S1 direction_id")
    expected_files = _expected_files(next_variant) or _expected_files(variant_contract)
    if not expected_files:
        errors.append("next_variant.expected_files must be non-empty")
    disallowed = _disallowed_c2c_files(expected_files, config)
    if disallowed:
        errors.append(f"next_variant expected_files outside allowed C2C edit surface: {disallowed[:5]}")
    if not _ablation_switch(next_variant) and not _ablation_switch(variant_contract):
        errors.append("next_variant.ablation_switch must be present")
    implementation_repair_mode = str(variant_fingerprint.get("mode") or "").startswith("implementation_repair")
    if variant_fingerprint.get("is_repeat") is True and not implementation_repair_mode:
        errors.append("variant_fingerprint repeats a previous same-direction variant")
    if scorecard.get("selected_variant_id") and str(scorecard.get("selected_variant_id")) != selected_id:
        errors.append("scorecard.selected_variant_id must match next_variant.id")
    if not selected_rows:
        errors.append("variant_scorecard must mark exactly one selected variant")
    elif len(selected_rows) > 1:
        errors.append("variant_scorecard must not select multiple variants")
    gate_cfg = ((config.get("c2c") or {}).get("s2_planner_gate") or {}) if isinstance(config.get("c2c"), dict) else {}
    min_score = float(gate_cfg.get("min_selected_variant_score", 0.0) or 0.0)
    if min_score and selected_score < min_score:
        errors.append(f"selected variant score {selected_score:.4f} below min_selected_variant_score {min_score:.4f}")
    if gate_cfg.get("require_rejected_variant_reasons") and len(candidates) > 1 and not scorecard.get("rejected_variants"):
        errors.append("variant_scorecard.rejected_variants must include structured reasons")
    return {
        "schema_version": C2C_S2_PLANNER_GATE_REPORT_SCHEMA_VERSION,
        "direction_id": direction_id,
        "gate": "pass" if not errors else "fail",
        "selected_variant_id": selected_id,
        "selected_variant_fingerprint": selected_fp or expected_fp,
        "selected_variant_score": selected_score,
        "selected_variant": next_variant,
        "checks": {
            "candidate_pool_non_empty": bool(candidates),
            "selected_variant_in_pool": selected_id in candidate_ids,
            "fingerprint_matches": bool(not selected_fp or not expected_fp or selected_fp == expected_fp),
            "direction_id_matches": not any("direction_id" in item and "match" in item for item in errors),
            "expected_files_present": bool(expected_files),
            "expected_files_within_allowed_surface": not disallowed,
            "ablation_switch_present": bool(_ablation_switch(next_variant) or _ablation_switch(variant_contract)),
            "not_repeated": not (variant_fingerprint.get("is_repeat") is True) or implementation_repair_mode,
            "scorecard_selected": bool(selected_rows),
        },
        "errors": errors,
        "return_to": "S2_planner" if errors else None,
    }


def build_s2_implementation_contract(
    *,
    direction: dict[str, Any],
    selected_variant: dict[str, Any],
    variant_contract: dict[str, Any],
    planner_gate_report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    forbidden = _forbidden_files(selected_variant)
    if not forbidden:
        forbidden = ["script/evaluation/*", "experiment/results/*", "local/auto_research_runs/*"]
    return {
        "schema_version": C2C_S2_5_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
        "direction_id": str(direction.get("direction_id") or selected_variant.get("s1_direction_id") or ""),
        "variant_id": str(selected_variant.get("id") or variant_contract.get("variant_id") or ""),
        "variant_fingerprint": str(selected_variant.get("variant_fingerprint") or variant_contract.get("variant_fingerprint") or ""),
        "allowed_files": [str(item) for item in c2c_cfg.get("allowed_files") or [] if item],
        "allowed_prefixes": [str(item) for item in c2c_cfg.get("allowed_prefixes") or [] if item],
        "forbidden_files": forbidden,
        "expected_files": _expected_files(selected_variant) or _expected_files(variant_contract),
        "ablation_switch": _ablation_switch(selected_variant) or _ablation_switch(variant_contract),
        "runtime_wiring_requirements": {
            "switch_referenced_in_runtime_code": True,
            "must_change_forward_path": True,
            "must_not_touch_evaluator": True,
        },
        "diagnostics_required": ["activation_check", "mechanism_review", "risk_check"],
        "planner_gate_report": {
            "path": "plan/s2_planner/planner_gate_report.json",
            "gate": planner_gate_report.get("gate"),
            "selected_variant_id": planner_gate_report.get("selected_variant_id"),
        },
    }


def build_s2_5_patch_gate_report(
    *,
    patch_manifest: dict[str, Any],
    implementation_contract: dict[str, Any],
    planner_gate_report: dict[str, Any],
    variant_fingerprint: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant_fingerprint = variant_fingerprint if isinstance(variant_fingerprint, dict) else {}
    expected_fp = str(variant_fingerprint.get("variant_fingerprint") or implementation_contract.get("variant_fingerprint") or "")
    selected_variant_id = str(planner_gate_report.get("selected_variant_id") or implementation_contract.get("variant_id") or "")
    selected_manifest_id = str(patch_manifest.get("selected_candidate_id") or (patch_manifest.get("selected_patch") or {}).get("candidate_id") or "")
    entries = [item for item in (patch_manifest.get("patches") or patch_manifest.get("candidates") or []) if isinstance(item, dict)]
    selected_entries = [item for item in entries if str(item.get("candidate_id") or "") == selected_manifest_id] or entries[:1]
    changed_files = _changed_files(selected_entries)
    checks = {
        "selected_variant_matches_planner": bool(selected_variant_id and selected_manifest_id == selected_variant_id),
        "has_executable_change": any(_patch_has_executable_change(item) for item in selected_entries),
        "changed_files_within_allowed_surface": not _disallowed_contract_files(changed_files, implementation_contract),
        "forbidden_files_untouched": not _forbidden_changed_files(changed_files, implementation_contract),
        "ablation_switch_present": bool(implementation_contract.get("ablation_switch")),
        "runtime_activation_check_present": _activation_check_present(selected_entries, config or {}),
        "mechanism_review_passed": _mechanism_review_passed(selected_entries),
        "risk_check_passed": _risk_check_passed(selected_entries),
    }
    manifest_status = str(patch_manifest.get("status") or "")
    failure_class = None
    repairable = False
    if manifest_status not in {"ok", "disabled"}:
        failure_class = "patch_generation_failure"
        repairable = True
    if not checks["selected_variant_matches_planner"]:
        failure_class = "planner_contract_failure"
    elif not checks["has_executable_change"] and manifest_status != "disabled":
        failure_class = "no_executable_change"
        repairable = True
    elif not checks["forbidden_files_untouched"]:
        failure_class = "forbidden_file_touched"
        repairable = True
    elif not checks["ablation_switch_present"]:
        failure_class = "activation_switch_missing"
        repairable = True
    elif _manifest_has_runtime_resource_retry(patch_manifest):
        failure_class = "runtime_smoke_resource_retry"
        repairable = True
    gate = "pass" if manifest_status == "ok" and all(checks.values()) else "fail"
    if failure_class == "runtime_smoke_resource_retry":
        gate = "retry"
    return {
        "schema_version": C2C_S2_5_PATCH_GATE_REPORT_SCHEMA_VERSION,
        "gate": gate,
        "variant_id": str(implementation_contract.get("variant_id") or selected_variant_id),
        "variant_fingerprint": expected_fp,
        "patch_manifest_status": manifest_status,
        "checks": checks,
        "changed_files": changed_files,
        "failure_class": failure_class,
        "repairable": repairable,
    }


def build_s2_rejected_variant_report(scorecard: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "c2c_s2_rejected_variant_report_v1",
        "direction_id": scorecard.get("direction_id"),
        "rejected_variants": scorecard.get("rejected_variants") or [],
    }


def _candidate_for_pool(candidate: dict[str, Any]) -> dict[str, Any]:
    experiment_contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    implementation_plan = candidate.get("implementation_plan") if isinstance(candidate.get("implementation_plan"), dict) else {}
    s2_variant = candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {}
    score = candidate.get("variant_score") if isinstance(candidate.get("variant_score"), dict) else s2_variant.get("variant_score") if isinstance(s2_variant.get("variant_score"), dict) else {}
    return {
        "id": str(candidate.get("id") or ""),
        "title": str(candidate.get("title") or candidate.get("id") or ""),
        "direction_id": candidate.get("direction_id") or candidate.get("s1_direction_id"),
        "variant_fingerprint": candidate.get("variant_fingerprint") or s2_variant.get("variant_fingerprint"),
        "mechanism_axis": candidate.get("mechanism_axis") or s2_variant.get("mechanism_axis"),
        "integration_point": candidate.get("integration_point") or s2_variant.get("integration_point"),
        "control_signal": candidate.get("control_signal") or s2_variant.get("control_signal"),
        "expected_files": _expected_files(candidate),
        "ablation_switch": _ablation_switch(candidate),
        "experiment_contract": experiment_contract,
        "implementation_plan": implementation_plan,
        "failure_feedback_refs": list(candidate.get("failure_feedback_refs") or []),
        "used_shared_memory_refs": list(candidate.get("used_shared_memory_refs") or []),
        "variant_score": score,
        "risk_budget": candidate.get("risk_budget") if isinstance(candidate.get("risk_budget"), dict) else s2_variant.get("risk_budget") if isinstance(s2_variant.get("risk_budget"), dict) else {},
        "selected_for_s2_5": bool(candidate.get("selected_for_s2_5") or candidate.get("selected")),
    }


def _score_components(
    *,
    direction: dict[str, Any],
    candidate: dict[str, Any],
    evidence_quality: dict[str, Any],
    variant_fingerprint: dict[str, Any],
    history: dict[str, set[str]],
    feedback: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    reasons: list[str] = []
    axis_match = candidate.get("mechanism_axis") == direction.get("mechanism_axis")
    point_present = bool(candidate.get("integration_point"))
    signal_match = candidate.get("control_signal") == direction.get("control_signal")
    alignment = 0.06 + (0.05 if axis_match else 0.0) + (0.04 if point_present else 0.0) + (0.03 if signal_match else 0.0)
    if not axis_match:
        reasons.append("mechanism_axis_diverges_from_s1")
    paper = ((evidence_quality.get("support_coverage") or {}).get("paper") or 0) if isinstance(evidence_quality.get("support_coverage"), dict) else 0
    code = ((evidence_quality.get("support_coverage") or {}).get("code") or 0) if isinstance(evidence_quality.get("support_coverage"), dict) else 0
    evidence = min(0.12, 0.03 * float(paper) + 0.03 * float(code))
    fingerprint = str(candidate.get("variant_fingerprint") or "")
    novelty = 0.15 if fingerprint and fingerprint not in history["fingerprints"] else 0.0
    if not novelty:
        reasons.append("repeated_fingerprint")
    anti_repeat = 0.12 if str(candidate.get("integration_point") or "") not in history["integration_points"] else 0.04
    if anti_repeat < 0.12:
        reasons.append("repeated_integration_point")
    feedback_fit = 0.08 + (0.05 if candidate.get("failure_feedback_refs") or feedback else 0.0)
    files = _expected_files(candidate)
    feasibility = 0.14 if files and not _disallowed_c2c_files(files, config) else 0.03 if files else 0.0
    if files and feasibility < 0.14:
        reasons.append("expected_files_not_supported")
    if not files:
        reasons.append("expected_files_missing")
    ablation = 0.08 if _ablation_switch(candidate) else 0.0
    if not ablation:
        reasons.append("weak_ablation_switch")
    proxy_prior = 0.02
    risk_penalty = -0.10 if _forbidden_files(candidate) else 0.0
    if risk_penalty:
        reasons.append("forbidden_file_risk")
    base_score = candidate.get("variant_score") if isinstance(candidate.get("variant_score"), dict) else {}
    if base_score.get("reasons"):
        reasons.extend(str(item) for item in base_score.get("reasons") or [] if item)
    return (
        {
            "s1_direction_alignment": round(alignment, 4),
            "s1_evidence_support": round(evidence, 4),
            "novelty": round(novelty, 4),
            "anti_repeat": round(anti_repeat, 4),
            "feedback_fit": round(feedback_fit, 4),
            "implementation_feasibility": round(feasibility, 4),
            "ablation_readiness": round(ablation, 4),
            "proxy_calibration_prior": round(proxy_prior, 4),
            "risk_penalty": round(risk_penalty, 4),
        },
        _dedupe(reasons) or ["ranked_by_deterministic_scorecard"],
    )


def _history_summary(planner_memory: dict[str, Any]) -> dict[str, set[str]]:
    history = {"fingerprints": set(), "integration_points": set()}
    for entry in planner_memory.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        for item in [entry.get("selected_candidate"), *(entry.get("candidate_summaries") or [])]:
            if not isinstance(item, dict):
                continue
            if item.get("variant_fingerprint"):
                history["fingerprints"].add(str(item.get("variant_fingerprint")))
            if item.get("integration_point"):
                history["integration_points"].add(str(item.get("integration_point")))
    return history


def _expected_files(payload: dict[str, Any]) -> list[str]:
    contract = payload.get("experiment_contract") if isinstance(payload.get("experiment_contract"), dict) else {}
    files = contract.get("expected_files") or payload.get("expected_files") or []
    if isinstance(files, str):
        return [files]
    return [str(item) for item in files if item] if isinstance(files, list) else []


def _ablation_switch(payload: dict[str, Any]) -> str:
    contract = payload.get("experiment_contract") if isinstance(payload.get("experiment_contract"), dict) else {}
    ablation = payload.get("ablation") if isinstance(payload.get("ablation"), dict) else {}
    ablation_plan = payload.get("ablation_plan") if isinstance(payload.get("ablation_plan"), dict) else {}
    return str(contract.get("ablation_switch") or payload.get("ablation_switch") or ablation.get("switch") or ablation_plan.get("switch") or "")


def _forbidden_files(payload: dict[str, Any]) -> list[str]:
    risk = payload.get("risk_budget") if isinstance(payload.get("risk_budget"), dict) else {}
    forbidden = risk.get("forbidden_files") or payload.get("forbidden_files") or []
    if isinstance(forbidden, str):
        return [forbidden]
    return [str(item) for item in forbidden if item] if isinstance(forbidden, list) else []


def _disallowed_c2c_files(files: list[str], config: dict[str, Any]) -> list[str]:
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    allowed_files = {str(item).strip("/") for item in c2c_cfg.get("allowed_files") or [] if item}
    allowed_prefixes = [str(item).strip("/") for item in c2c_cfg.get("allowed_prefixes") or [] if item]
    if not allowed_files and not allowed_prefixes:
        return []
    disallowed = []
    for file_path in files:
        normalized = str(file_path).strip("/")
        if normalized in allowed_files or any(normalized.startswith(prefix.rstrip("/") + "/") or normalized == prefix for prefix in allowed_prefixes):
            continue
        disallowed.append(file_path)
    return disallowed


def _disallowed_contract_files(files: list[str], implementation_contract: dict[str, Any]) -> list[str]:
    allowed_files = {str(item).strip("/") for item in implementation_contract.get("allowed_files") or [] if item}
    allowed_prefixes = [str(item).strip("/") for item in implementation_contract.get("allowed_prefixes") or [] if item]
    expected_files = {str(item).strip("/") for item in implementation_contract.get("expected_files") or [] if item}
    if not allowed_files and not allowed_prefixes and not expected_files:
        return []
    disallowed = []
    for file_path in files:
        normalized = str(file_path).strip("/")
        if normalized in allowed_files or normalized in expected_files:
            continue
        if any(normalized.startswith(prefix.rstrip("/") + "/") or normalized == prefix for prefix in allowed_prefixes):
            continue
        disallowed.append(file_path)
    return disallowed


def _forbidden_changed_files(files: list[str], implementation_contract: dict[str, Any]) -> list[str]:
    patterns = [str(item) for item in implementation_contract.get("forbidden_files") or [] if item]
    return [path for path in files if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)]


def _patch_has_executable_change(entry: dict[str, Any]) -> bool:
    if entry.get("has_executable_change") is True:
        return True
    code_patch = entry.get("code_patch") if isinstance(entry.get("code_patch"), dict) else {}
    if code_patch.get("has_executable_change") is True:
        return True
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    if validation.get("status") == "PASS" and validation.get("has_executable_change") is True:
        return True
    changed = entry.get("changed_files") or code_patch.get("changed_files") or []
    return any(str(path).endswith(".py") for path in changed)


def _changed_files(entries: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for entry in entries:
        code_patch = entry.get("code_patch") if isinstance(entry.get("code_patch"), dict) else {}
        for path in entry.get("changed_files") or code_patch.get("changed_files") or []:
            if path:
                files.append(str(path))
    return _dedupe(files)


def _activation_check_present(entries: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    runtime_cfg = (((config.get("code_patch") or {}).get("validation") or {}).get("runtime_smoke") or {}) if isinstance(config.get("code_patch"), dict) else {}
    if runtime_cfg.get("enabled") is False:
        return True
    for entry in entries:
        validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
        activation = validation.get("activation_check") or entry.get("activation_check")
        if isinstance(activation, dict):
            return True
    return not entries


def _mechanism_review_passed(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        review = entry.get("mechanism_review") or ((entry.get("validation") or {}).get("mechanism_review") if isinstance(entry.get("validation"), dict) else None)
        if isinstance(review, dict) and str(review.get("status") or "").lower() not in {"", "ok", "pass"}:
            return False
    return True


def _risk_check_passed(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        risk = entry.get("risk_check") or ((entry.get("validation") or {}).get("risk_check") if isinstance(entry.get("validation"), dict) else None)
        if isinstance(risk, dict) and str(risk.get("status") or "").lower() not in {"", "ok", "pass"}:
            return False
    return True


def _manifest_has_runtime_resource_retry(manifest: dict[str, Any]) -> bool:
    if manifest.get("resource_retry") is True:
        return True
    for entry in manifest.get("patches") or manifest.get("candidates") or []:
        if isinstance(entry, dict) and (entry.get("resource_retry") is True or entry.get("failure_category") == "runtime_smoke_resource_retry"):
            return True
    return False


def _dedupe(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
