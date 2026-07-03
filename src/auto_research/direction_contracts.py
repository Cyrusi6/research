"""Direction and S2 variant contract helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .utils import read_json


DIRECTION_SCHEMA_VERSION = "auto_research_direction_v1"
DIRECTION_SCORECARD_SCHEMA_VERSION = "auto_research_direction_scorecard_v1"
NOVELTY_AUDIT_SCHEMA_VERSION = "auto_research_novelty_audit_v1"
PLANNER_DECISION_SCHEMA_VERSION = "auto_research_planner_decision_v1"
VARIANT_CONTRACT_SCHEMA_VERSION = "auto_research_variant_contract_v1"
VARIANT_FINGERPRINT_SCHEMA_VERSION = "auto_research_variant_fingerprint_v1"
C2C_S1_EVIDENCE_QUALITY_SCHEMA_VERSION = "c2c_s1_evidence_quality_v1"
C2C_S1_EVIDENCE_RETRIEVAL_TRACE_SCHEMA_VERSION = "c2c_s1_evidence_retrieval_trace_v1"
C2C_S1_DETERMINISTIC_RETRIEVAL_TRACE_SCHEMA_VERSION = "c2c_s1_deterministic_retrieval_trace_v1"
C2C_S1_DIRECTION_FINGERPRINT_SCHEMA_VERSION = "c2c_s1_direction_fingerprint_v1"


MECHANISM_DEFAULTS: dict[str, dict[str, str]] = {
    "utility_predicted_cache_routing": {
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
    },
    "counterfactual_training_objective": {
        "mechanism_axis": "training_signal",
        "integration_point": "train_loss",
        "control_signal": "counterfactual",
    },
    "semantic_span_graph_alignment": {
        "mechanism_axis": "span_selection",
        "integration_point": "aligner",
        "control_signal": "semantic_similarity",
    },
    "verifier_guided_cache_acceptance": {
        "mechanism_axis": "scoring",
        "integration_point": "projector",
        "control_signal": "confidence",
    },
    "latent_bridge_memory": {
        "mechanism_axis": "normalization",
        "integration_point": "projector",
        "control_signal": "span_agreement",
    },
    "pathology_conditioned_controller": {
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "pathology",
    },
}


def build_direction_contract(
    payload: dict[str, Any],
    *,
    mode: str = "generic",
    used_shared_memory_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build the new S1 direction artifact from legacy S1 payload shapes."""

    direction = payload.get("direction") if isinstance(payload.get("direction"), dict) else {}
    decision = payload.get("direction_decision") if isinstance(payload.get("direction_decision"), dict) else {}
    selected = _selected_idea(payload.get("selected_ideas"))
    evidence_bundle = payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {}
    negative_constraints = payload.get("negative_constraints") if isinstance(payload.get("negative_constraints"), dict) else {}
    refs = _as_list(used_shared_memory_refs)
    if not refs:
        refs = _as_list(payload.get("used_shared_memory_refs")) or _as_list(decision.get("used_shared_memory_refs")) or _as_list(selected.get("used_shared_memory_refs"))

    mechanism_type = str(
        direction.get("mechanism_type")
        or decision.get("mechanism_type")
        or selected.get("mechanism_type")
        or ""
    )
    defaults = MECHANISM_DEFAULTS.get(mechanism_type, {})
    title = (
        direction.get("title")
        or decision.get("title")
        or decision.get("mechanism_direction")
        or selected.get("title")
        or decision.get("direction_id")
        or selected.get("id")
        or "Selected direction"
    )
    direction_id = str(
        direction.get("direction_id")
        or decision.get("direction_id")
        or selected.get("direction_id")
        or selected.get("s1_direction_id")
        or selected.get("id")
        or _snakeish(title)
    )
    expected_files = _as_list(direction.get("expected_files")) or _as_list(decision.get("expected_files")) or _as_list(selected.get("expected_files"))
    implementation_refs = (
        _as_list(direction.get("implementation_surface_refs"))
        or _as_list(decision.get("implementation_surface_refs"))
        or _as_list(selected.get("implementation_surface_refs"))
        or _as_list(selected.get("code_refs"))
        or _surface_refs_from_files(expected_files)
    )
    required_refs = (
        _as_list(direction.get("required_evidence_refs"))
        or _as_list(decision.get("required_evidence_refs"))
        or _as_list(selected.get("evidence_refs"))
        or _refs_from_evidence_bundle(evidence_bundle, want_risk=False)
    )
    counter_refs = (
        _as_list(direction.get("counterevidence_refs"))
        or _as_list(decision.get("counterevidence_refs"))
        or _as_list(selected.get("counterevidence_refs"))
        or _refs_from_evidence_bundle(evidence_bundle, want_risk=True)
    )
    expected_metric_signature = (
        direction.get("expected_metric_signature")
        if isinstance(direction.get("expected_metric_signature"), dict)
        else decision.get("expected_metric_signature")
        if isinstance(decision.get("expected_metric_signature"), dict)
        else selected.get("expected_metric_signature")
        if isinstance(selected.get("expected_metric_signature"), dict)
        else selected.get("expected_signature")
        if isinstance(selected.get("expected_signature"), dict)
        else _default_metric_signature(mode, selected, decision)
    )
    hypothesis = str(
        direction.get("hypothesis")
        or decision.get("hypothesis")
        or decision.get("core_hypothesis")
        or selected.get("hypothesis")
        or selected.get("description")
        or "S2 should instantiate and test this direction."
    )
    why_baseline_fails = str(
        direction.get("why_baseline_fails")
        or decision.get("why_baseline_fails")
        or selected.get("why_baseline_fails")
        or selected.get("motivation")
        or decision.get("rationale")
        or "The current baseline lacks the proposed control signal or mechanism axis."
    )
    return {
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "direction_id": direction_id,
        "title": str(title),
        "mechanism_type": mechanism_type or "generic_direction",
        "mechanism_axis": str(direction.get("mechanism_axis") or decision.get("mechanism_axis") or selected.get("mechanism_axis") or defaults.get("mechanism_axis") or "method"),
        "integration_point": str(direction.get("integration_point") or decision.get("integration_point") or selected.get("integration_point") or _infer_integration_point(expected_files) or defaults.get("integration_point") or "experiment_plan"),
        "control_signal": str(direction.get("control_signal") or decision.get("control_signal") or selected.get("control_signal") or defaults.get("control_signal") or "primary_metric"),
        "hypothesis": hypothesis,
        "why_baseline_fails": why_baseline_fails,
        "expected_metric_signature": expected_metric_signature,
        "required_evidence_refs": required_refs or [{"source_type": "artifact", "source_label": "evidence_bundle", "claim": "S1 evidence bundle supports this direction."}],
        "counterevidence_refs": counter_refs or [{"source_type": "artifact", "source_label": "risk", "claim": "S2 must verify the direction under baseline-comparable controls."}],
        "implementation_surface_refs": implementation_refs or [{"source_type": "artifact", "source_label": "S2", "claim": "S2 must identify the concrete implementation surface."}],
        "known_negative_memory_refs": _known_negative_refs(direction, decision, selected, negative_constraints, refs),
        "go_to_s2_conditions": _as_list(direction.get("go_to_s2_conditions")) or _as_list(decision.get("go_to_s2_conditions")) or [
            "required evidence refs resolve",
            "novelty audit passes or is explicitly skipped",
            "implementation surface refs are available for S2 planning",
        ],
        "return_to_s1_conditions": _as_list(direction.get("return_to_s1_conditions")) or _as_list(decision.get("return_to_s1_conditions")) or [
            "same-direction failure budget is exhausted",
            "S2 cannot instantiate an executable variant without violating forbidden patterns",
            "S3 shows repeated method-level metric collapse for this direction",
        ],
        "allowed_variants": _as_list(direction.get("allowed_variants")) or _as_list(decision.get("allowed_variants")) or _as_list(selected.get("s1_allowed_variants")),
        "forbidden_patterns": _as_list(direction.get("forbidden_patterns")) or _as_list(decision.get("forbidden_patterns")) or _as_list(selected.get("s1_forbidden_patterns")) or _as_list(negative_constraints.get("forbidden_patterns")),
        "target_datasets": _as_list(direction.get("target_datasets")) or _as_list(decision.get("target_datasets")),
        "expected_files": expected_files,
        "verification_commands": _as_list(direction.get("verification_commands")) or _as_list(decision.get("verification_commands")) or _as_list(selected.get("verification_commands")),
        "used_shared_memory_refs": refs,
        "source": {
            "mode": mode,
            "legacy_direction_decision": decision,
            "legacy_selected_idea_id": selected.get("id"),
        },
    }


def build_direction_scorecard(
    direction: dict[str, Any],
    *,
    evidence_bundle: dict[str, Any] | None = None,
    novelty_audit: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    audit = normalize_novelty_audit(novelty_audit, direction_id=str(direction.get("direction_id") or "unknown_direction"))
    evidence_items = (evidence_bundle or {}).get("items") if isinstance(evidence_bundle, dict) else []
    required_refs = _as_list(direction.get("required_evidence_refs"))
    counter_refs = _as_list(direction.get("counterevidence_refs"))
    surfaces = _as_list(direction.get("implementation_surface_refs"))
    return {
        "schema_version": DIRECTION_SCORECARD_SCHEMA_VERSION,
        "direction_id": direction.get("direction_id"),
        "title": direction.get("title"),
        "mechanism_axis": direction.get("mechanism_axis"),
        "integration_point": direction.get("integration_point"),
        "control_signal": direction.get("control_signal"),
        "evidence_item_count": len(evidence_items or []),
        "required_evidence_ref_count": len(required_refs),
        "counterevidence_ref_count": len(counter_refs),
        "implementation_surface_ref_count": len(surfaces),
        "novelty": {
            "status": audit.get("status"),
            "passed": audit.get("passed"),
            "threshold": audit.get("threshold"),
        },
        "go_to_s2_ready": bool(direction.get("direction_id") and required_refs and surfaces and audit.get("passed") is not False),
        "go_to_s2_conditions": _as_list(direction.get("go_to_s2_conditions")),
        "return_to_s1_conditions": _as_list(direction.get("return_to_s1_conditions")),
        "used_shared_memory_refs": _as_list(direction.get("used_shared_memory_refs")),
    }


def build_s1_direction_fingerprint(
    direction: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic identity for the selected C2C S1 direction."""

    features = {
        "direction_id": str(direction.get("direction_id") or ""),
        "mechanism_axis": str(direction.get("mechanism_axis") or ""),
        "integration_point": str(direction.get("integration_point") or ""),
        "control_signal": str(direction.get("control_signal") or ""),
        "mechanism_type": str(direction.get("mechanism_type") or ""),
        "expected_files": sorted(str(item) for item in _as_list(direction.get("expected_files"))),
        "implementation_surface_refs": sorted(_ref_label(item) for item in _as_list(direction.get("implementation_surface_refs"))),
    }
    token_text = " ".join(
        [
            str(direction.get("title") or ""),
            str(direction.get("hypothesis") or ""),
            str(direction.get("why_baseline_fails") or ""),
            " ".join(str(value) for value in features.values() if not isinstance(value, list)),
            " ".join(features["expected_files"]),
            " ".join(features["implementation_surface_refs"]),
        ]
    )
    tokens = _fingerprint_tokens(token_text)
    history = _direction_history_similarity(project_root, tokens) if project_root else []
    similarity = max((float(item.get("similarity") or 0.0) for item in history), default=0.0)
    fingerprint_payload = {
        "features": features,
        "tokens": sorted(tokens),
    }
    return {
        "schema_version": C2C_S1_DIRECTION_FINGERPRINT_SCHEMA_VERSION,
        "direction_id": features["direction_id"],
        "fingerprint": hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16],
        "features": features,
        "feature_tokens": sorted(tokens),
        "history": history[:10],
        "same_direction_similarity": round(similarity, 4),
    }


def build_s1_evidence_quality_score(
    direction: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    evidence_bundle: dict[str, Any] | None = None,
    evidence_ref_report: dict[str, Any] | None = None,
    novelty_audit: dict[str, Any] | list[Any] | None = None,
    direction_fingerprint: dict[str, Any] | None = None,
    direction_bundle_ref_report: dict[str, Any] | None = None,
    shared_memory_checked: bool = False,
) -> dict[str, Any]:
    """Score whether C2C S1 gathered enough evidence to enter S2."""

    payload = payload if isinstance(payload, dict) else {}
    evidence_bundle = evidence_bundle if isinstance(evidence_bundle, dict) else payload.get("evidence_bundle") if isinstance(payload.get("evidence_bundle"), dict) else {}
    evidence_ref_report = evidence_ref_report if isinstance(evidence_ref_report, dict) else {}
    resolved_refs = [item for item in evidence_ref_report.get("resolved") or [] if isinstance(item, dict)]
    unresolved_refs = [item for item in evidence_ref_report.get("errors") or [] if isinstance(item, dict)]

    coverage_keys: dict[str, set[str]] = {"paper": set(), "rebuttal": set(), "code": set(), "failure_memory": set()}
    coverage_contributors: dict[str, list[dict[str, Any]]] = {"paper": [], "rebuttal": [], "code": [], "failure_memory": []}
    for entry in resolved_refs:
        bucket = _evidence_bucket(entry)
        if bucket not in coverage_keys:
            continue
        key = _coverage_ref_identity(entry)
        if key not in coverage_keys[bucket]:
            coverage_keys[bucket].add(key)
            coverage_contributors[bucket].append(_compact_quality_ref(entry))
    for item in evidence_bundle.get("items") or []:
        if not isinstance(item, dict):
            continue
        bucket = _evidence_bucket(item)
        if bucket not in coverage_keys:
            continue
        key = _coverage_ref_identity(item)
        if key not in coverage_keys[bucket]:
            coverage_keys[bucket].add(key)
            coverage_contributors[bucket].append(_compact_quality_ref(item))

    negative_refs = _as_list(direction.get("known_negative_memory_refs")) or _as_list(payload.get("used_shared_memory_refs"))
    for ref in negative_refs:
        key = str(ref)
        if key and key not in coverage_keys["failure_memory"]:
            coverage_keys["failure_memory"].add(key)
            coverage_contributors["failure_memory"].append({"ref": key})

    support_coverage = {bucket: len(keys) for bucket, keys in coverage_keys.items()}
    counter_refs = _counter_refs(direction=direction, payload=payload, evidence_bundle=evidence_bundle)
    resolved_counter_keys = {
        _ref_identity(entry)
        for entry in resolved_refs
        if str(entry.get("kind") or "").endswith("counterevidence_refs") or str(entry.get("kind") or "") == "counterevidence_refs"
    }
    for item in evidence_bundle.get("items") or []:
        if isinstance(item, dict) and item.get("risks"):
            resolved_counter_keys.add(_ref_identity(item))

    surface_targets = _implementation_surface_targets(direction, payload)
    covered_surfaces = _covered_implementation_surfaces(surface_targets, resolved_refs, evidence_bundle)
    surface_coverage = (len(covered_surfaces) / len(surface_targets)) if surface_targets else 0.0
    novelty_score = _extract_novelty_score(normalize_novelty_audit(novelty_audit, direction_id=str(direction.get("direction_id") or "")))
    same_direction_similarity = float((direction_fingerprint or {}).get("same_direction_similarity") or 0.0)
    thresholds = {
        "unresolved_ref_count": 0,
        "support_coverage.paper": 2,
        "support_coverage.code": 2,
        "counterevidence.count": 1,
        "implementation_surface_coverage": 0.6,
        "novelty_score": 0.6,
    }
    unresolved_ref_count = int((evidence_ref_report.get("counts") or {}).get("unresolved") or len(unresolved_refs))
    counterevidence = {"count": len(counter_refs), "resolved_count": len(resolved_counter_keys)}
    failed_rules: list[str] = []
    if unresolved_ref_count != 0:
        failed_rules.append("unresolved_ref_count")
    if support_coverage["paper"] < thresholds["support_coverage.paper"]:
        failed_rules.append("support_coverage.paper")
    if support_coverage["code"] < thresholds["support_coverage.code"]:
        failed_rules.append("support_coverage.code")
    if counterevidence["count"] < thresholds["counterevidence.count"]:
        failed_rules.append("counterevidence.count")
    if surface_coverage < thresholds["implementation_surface_coverage"]:
        failed_rules.append("implementation_surface_coverage")
    if novelty_score < thresholds["novelty_score"]:
        failed_rules.append("novelty_score")
    if isinstance(direction_bundle_ref_report, dict) and direction_bundle_ref_report.get("status") not in {None, "pass"}:
        failed_rules.append("direction_refs_not_in_retrieved_bundle")
    return {
        "schema_version": C2C_S1_EVIDENCE_QUALITY_SCHEMA_VERSION,
        "direction_id": str(direction.get("direction_id") or ""),
        "support_coverage": support_coverage,
        "counterevidence": counterevidence,
        "implementation_surface_coverage": round(surface_coverage, 4),
        "implementation_surface": {
            "target_count": len(surface_targets),
            "covered_count": len(covered_surfaces),
            "targets": sorted(surface_targets),
            "covered": sorted(covered_surfaces),
        },
        "unresolved_ref_count": unresolved_ref_count,
        "shared_memory_checked": bool(shared_memory_checked),
        "novelty_score": round(novelty_score, 4),
        "same_direction_similarity": round(same_direction_similarity, 4),
        "direction_bundle_ref_report": direction_bundle_ref_report if isinstance(direction_bundle_ref_report, dict) else {},
        "thresholds": thresholds,
        "failed_rules": failed_rules,
        "coverage_contributors": {key: values[:20] for key, values in coverage_contributors.items()},
        "gate": "pass" if not failed_rules else "fail",
    }


def build_s1_evidence_retrieval_trace(
    direction: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    evidence_ref_report: dict[str, Any] | None = None,
    evidence_quality_score: dict[str, Any] | None = None,
    direction_fingerprint: dict[str, Any] | None = None,
    deterministic_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(deterministic_trace, dict) and deterministic_trace.get("schema_version"):
        trace = dict(deterministic_trace)
        trace["direction_id"] = str(direction.get("direction_id") or trace.get("direction_id") or "")
        score = evidence_quality_score if isinstance(evidence_quality_score, dict) else {}
        trace["quality_gate"] = {
            "gate": score.get("gate"),
            "failed_rules": _as_list(score.get("failed_rules")),
            "thresholds": score.get("thresholds") if isinstance(score.get("thresholds"), dict) else {},
        }
        trace["direction_fingerprint"] = {
            "fingerprint": (direction_fingerprint or {}).get("fingerprint"),
            "same_direction_similarity": (direction_fingerprint or {}).get("same_direction_similarity", 0.0),
            "artifact": "literature/c2c/direction_fingerprint.json",
        }
        trace.setdefault("deterministic", True)
        return trace
    payload = payload if isinstance(payload, dict) else {}
    evidence_ref_report = evidence_ref_report if isinstance(evidence_ref_report, dict) else {}
    score = evidence_quality_score if isinstance(evidence_quality_score, dict) else {}
    counts = evidence_ref_report.get("counts") if isinstance(evidence_ref_report.get("counts"), dict) else {}
    return {
        "schema_version": C2C_S1_EVIDENCE_RETRIEVAL_TRACE_SCHEMA_VERSION,
        "direction_id": str(direction.get("direction_id") or ""),
        "evidence_requests": _as_list(payload.get("evidence_requests")),
        "resolved_ref_count": int(counts.get("resolved") or len(evidence_ref_report.get("resolved") or [])),
        "unresolved_ref_count": int(counts.get("unresolved") or len(evidence_ref_report.get("errors") or [])),
        "resolved_refs": [_compact_quality_ref(item) for item in (evidence_ref_report.get("resolved") or []) if isinstance(item, dict)][:80],
        "unresolved_refs": [_compact_quality_ref(item) for item in (evidence_ref_report.get("errors") or []) if isinstance(item, dict)][:80],
        "coverage_contributors": score.get("coverage_contributors") if isinstance(score.get("coverage_contributors"), dict) else {},
        "quality_gate": {
            "gate": score.get("gate"),
            "failed_rules": _as_list(score.get("failed_rules")),
            "thresholds": score.get("thresholds") if isinstance(score.get("thresholds"), dict) else {},
        },
        "direction_fingerprint": {
            "fingerprint": (direction_fingerprint or {}).get("fingerprint"),
            "same_direction_similarity": (direction_fingerprint or {}).get("same_direction_similarity", 0.0),
            "artifact": "literature/c2c/direction_fingerprint.json",
        },
    }


def normalize_novelty_audit(value: dict[str, Any] | list[Any] | None, *, direction_id: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value.get("schema_version") == NOVELTY_AUDIT_SCHEMA_VERSION:
        return value
    audits = value if isinstance(value, list) else []
    latest = next((item for item in reversed(audits) if isinstance(item, dict)), {})
    if not latest and isinstance(value, dict):
        latest = value
    passed = latest.get("passed")
    if passed is None:
        passed = True
    status = str(latest.get("status") or ("ok" if latest else "skipped"))
    return {
        "schema_version": NOVELTY_AUDIT_SCHEMA_VERSION,
        "direction_id": direction_id,
        "status": status,
        "enabled": latest.get("enabled", bool(latest)),
        "passed": bool(passed),
        "threshold": latest.get("threshold"),
        "latest": latest,
        "audits": audits,
    }


def direction_to_legacy_idea(direction: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": direction.get("direction_id") or "selected_direction",
        "direction_id": direction.get("direction_id"),
        "s1_direction_id": direction.get("direction_id"),
        "title": direction.get("title") or direction.get("direction_id") or "Selected direction",
        "selected": True,
        "hypothesis": direction.get("hypothesis") or "",
        "novelty_score": 7,
        "feasibility_score": 7,
        "mechanism_type": direction.get("mechanism_type"),
        "description": direction.get("hypothesis") or "",
        "motivation": direction.get("why_baseline_fails") or "",
        "why_baseline_fails": direction.get("why_baseline_fails"),
        "expected_signature": direction.get("expected_metric_signature"),
        "expected_files": _as_list(direction.get("expected_files")),
        "verification_commands": _as_list(direction.get("verification_commands")),
        "evidence_refs": _as_list(direction.get("required_evidence_refs")),
        "counterevidence_refs": _as_list(direction.get("counterevidence_refs")),
        "code_refs": _as_list(direction.get("implementation_surface_refs")),
        "s1_allowed_variants": _as_list(direction.get("allowed_variants")),
        "s1_forbidden_patterns": _as_list(direction.get("forbidden_patterns")),
        "used_shared_memory_refs": _as_list(direction.get("used_shared_memory_refs")),
        "s1_direction_contract": {
            "schema_version": direction.get("schema_version"),
            "mechanism_axis": direction.get("mechanism_axis"),
            "integration_point": direction.get("integration_point"),
            "control_signal": direction.get("control_signal"),
        },
    }


def load_direction_or_legacy_idea(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direction = read_json(project_root / "literature" / "direction.json", default={}) or {}
    if isinstance(direction, dict) and direction.get("direction_id"):
        ideas = read_json(project_root / "literature" / "ideas.json", default=[]) or []
        if not isinstance(ideas, list) or not ideas:
            ideas = [direction_to_legacy_idea(direction)]
        return direction, [item for item in ideas if isinstance(item, dict)] or [direction_to_legacy_idea(direction)]
    ideas = read_json(project_root / "literature" / "ideas.json", default=[]) or []
    ideas = [item for item in ideas if isinstance(item, dict)] if isinstance(ideas, list) else []
    selected = next((idea for idea in ideas if idea.get("selected")), ideas[0] if ideas else {})
    payload = {"selected_ideas": [selected], "direction_decision": selected, "evidence_bundle": {"items": selected.get("evidence_refs") or []}}
    return build_direction_contract(payload), ideas or [direction_to_legacy_idea(build_direction_contract(payload))]


def build_planner_decision_artifact(
    *,
    direction: dict[str, Any],
    planner_summary: str | None,
    planning_mode: str | None,
    next_variant: dict[str, Any],
    used_shared_memory_refs: list[str] | None = None,
    source: str = "plan_agent",
) -> dict[str, Any]:
    return {
        "schema_version": PLANNER_DECISION_SCHEMA_VERSION,
        "direction_id": direction.get("direction_id"),
        "planner_summary": planner_summary or "",
        "planning_mode": planning_mode or "same_direction_variant",
        "used_shared_memory_refs": _as_list(used_shared_memory_refs),
        "next_variant": next_variant,
        "source": source,
    }


def build_variant_contract(
    *,
    direction: dict[str, Any],
    variant: dict[str, Any],
    plan: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    mode: str = "regular",
) -> dict[str, Any]:
    plan = plan or {}
    execution = execution if isinstance(execution, dict) else plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    contract = variant.get("experiment_contract") if isinstance(variant.get("experiment_contract"), dict) else {}
    ablation_plan = variant.get("ablation_plan") if isinstance(variant.get("ablation_plan"), dict) else {}
    expected_files = _as_list(contract.get("expected_files")) or _as_list(variant.get("expected_files")) or _as_list(direction.get("expected_files"))
    if not expected_files and mode != "c2c":
        expected_files = ["S2 implementation surface TBD"]
    expected_metric_signature = (
        variant.get("expected_metric_signature")
        if isinstance(variant.get("expected_metric_signature"), dict)
        else variant.get("expected_signature")
        if isinstance(variant.get("expected_signature"), dict)
        else direction.get("expected_metric_signature")
        if isinstance(direction.get("expected_metric_signature"), dict)
        else {}
    )
    ablation_switch = contract.get("ablation_switch") or ablation_plan.get("switch") or variant.get("ablation_switch")
    fingerprint = variant.get("variant_fingerprint") or ((variant.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(variant.get("s2_variant"), dict) else "")
    return {
        "schema_version": VARIANT_CONTRACT_SCHEMA_VERSION,
        "direction_id": direction.get("direction_id") or variant.get("s1_direction_id"),
        "variant_id": variant.get("id") or "selected_variant",
        "title": variant.get("title") or variant.get("id") or "Selected variant",
        "mode": mode,
        "variant_fingerprint": fingerprint,
        "mechanism_axis": variant.get("mechanism_axis") or ((variant.get("s2_variant") or {}).get("mechanism_axis") if isinstance(variant.get("s2_variant"), dict) else None) or direction.get("mechanism_axis"),
        "integration_point": variant.get("integration_point") or ((variant.get("s2_variant") or {}).get("integration_point") if isinstance(variant.get("s2_variant"), dict) else None) or direction.get("integration_point"),
        "control_signal": variant.get("control_signal") or ((variant.get("s2_variant") or {}).get("control_signal") if isinstance(variant.get("s2_variant"), dict) else None) or direction.get("control_signal"),
        "hypothesis": variant.get("hypothesis") or direction.get("hypothesis") or "",
        "why_next": variant.get("why_next") or variant.get("anti_repeat") or "",
        "expected_files": expected_files,
        "implementation_surface_refs": _as_list(variant.get("implementation_surface_refs")) or _as_list(variant.get("code_refs")) or _surface_refs_from_files(expected_files) or _as_list(direction.get("implementation_surface_refs")),
        "resource_budget": plan.get("resource_budget") if isinstance(plan.get("resource_budget"), dict) else {},
        "expected_metric_signature": expected_metric_signature,
        "ablation": {
            "switch": ablation_switch or "disable_selected_mechanism",
            "control": contract.get("ablation_control") or "candidate ablation-off control",
            "expected_signature": expected_metric_signature,
        },
        "acceptance": {
            "min_delta_to_pass": execution.get("min_delta_to_pass"),
            "max_dataset_regression": execution.get("max_dataset_regression"),
            "criteria": plan.get("acceptance_criteria") if isinstance(plan.get("acceptance_criteria"), dict) else {},
        },
        "failure_routing": {
            "go_to_s3_conditions": _as_list(variant.get("go_to_s3_conditions")) or ["S2 gate passes", "S2.5 patch manifest is executable or disabled"],
            "return_to_s2_conditions": _as_list(variant.get("return_to_s2_conditions")) or ["variant implementation fails", "ablation switch does not change mechanism trace"],
            "return_to_s1_conditions": _as_list(variant.get("return_to_s1_conditions")) or _as_list(direction.get("return_to_s1_conditions")),
        },
        "used_shared_memory_refs": _as_list(variant.get("used_shared_memory_refs")) or _as_list(direction.get("used_shared_memory_refs")),
    }


def build_variant_fingerprint_artifact(
    *,
    direction: dict[str, Any],
    variant: dict[str, Any],
    fingerprint: str | None = None,
    history_fingerprints: list[str] | None = None,
    mode: str = "regular",
) -> dict[str, Any]:
    current = fingerprint or variant.get("variant_fingerprint") or ((variant.get("s2_variant") or {}).get("variant_fingerprint") if isinstance(variant.get("s2_variant"), dict) else "")
    if not current:
        current = variant_fingerprint(variant, direction=direction)
    history = [str(item) for item in history_fingerprints or [] if item]
    s2_variant = variant.get("s2_variant") if isinstance(variant.get("s2_variant"), dict) else {}
    return {
        "schema_version": VARIANT_FINGERPRINT_SCHEMA_VERSION,
        "direction_id": direction.get("direction_id") or variant.get("s1_direction_id"),
        "variant_id": variant.get("id") or "selected_variant",
        "variant_fingerprint": current,
        "mechanism_axis": variant.get("mechanism_axis") or s2_variant.get("mechanism_axis") or direction.get("mechanism_axis"),
        "integration_point": variant.get("integration_point") or s2_variant.get("integration_point") or direction.get("integration_point"),
        "control_signal": variant.get("control_signal") or s2_variant.get("control_signal") or direction.get("control_signal"),
        "history_fingerprints": history,
        "is_repeat": bool(current and current in set(history)),
        "mode": mode,
    }


def variant_fingerprint(variant: dict[str, Any], *, direction: dict[str, Any] | None = None) -> str:
    direction = direction or {}
    contract = variant.get("experiment_contract") if isinstance(variant.get("experiment_contract"), dict) else {}
    s2_variant = variant.get("s2_variant") if isinstance(variant.get("s2_variant"), dict) else {}
    payload = {
        "direction_id": direction.get("direction_id") or variant.get("s1_direction_id"),
        "mechanism_type": variant.get("mechanism_type") or direction.get("mechanism_type"),
        "mechanism_axis": variant.get("mechanism_axis") or s2_variant.get("mechanism_axis") or direction.get("mechanism_axis"),
        "integration_point": variant.get("integration_point") or s2_variant.get("integration_point") or direction.get("integration_point"),
        "control_signal": variant.get("control_signal") or s2_variant.get("control_signal") or direction.get("control_signal"),
        "expected_files": sorted(_as_list(contract.get("expected_files")) or _as_list(variant.get("expected_files")) or _as_list(direction.get("expected_files"))),
        "config_keys": sorted(_flatten_keys(contract.get("config_overrides") or {})),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]


def _ref_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("source_label", "source_path", "chunk_id", "path", "file", "symbol", "claim"):
            if value.get(key):
                return str(value.get(key))
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)[:240]
    return str(value)


def _fingerprint_tokens(text: str) -> set[str]:
    words = []
    current = []
    for char in text.lower():
        if char.isalnum() or char in {"_", "-", "/"}:
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return {word for word in words if len(word) >= 3}


def _direction_history_similarity(project_root: Path | None, tokens: set[str]) -> list[dict[str, Any]]:
    if not project_root or not tokens:
        return []
    candidates: list[tuple[str, Any]] = [
        ("plan/direction_scorecard.json", read_json(project_root / "plan" / "direction_scorecard.json", default={})),
        ("plan/performance_feedback.json", read_json(project_root / "plan" / "performance_feedback.json", default={})),
        ("intake/c2c/negative_result_memory.json", read_json(project_root / "intake" / "c2c" / "negative_result_memory.json", default={})),
        ("literature/c2c/negative_result_memory.json", read_json(project_root / "literature" / "c2c" / "negative_result_memory.json", default={})),
    ]
    history = []
    for source, payload in candidates:
        for idx, item in enumerate(_history_records(payload)):
            item_text = json.dumps(item, sort_keys=True, ensure_ascii=True, default=str)
            similarity = _jaccard(tokens, _fingerprint_tokens(item_text))
            if similarity <= 0:
                continue
            history.append(
                {
                    "source": source,
                    "record_id": str(item.get("direction_id") or item.get("id") or item.get("memory_id") or item.get("variant_id") or idx),
                    "similarity": round(similarity, 4),
                }
            )
    return sorted(history, key=lambda item: float(item.get("similarity") or 0.0), reverse=True)


def _history_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    records: list[dict[str, Any]] = []
    for key in ["directions", "direction_history", "entries", "recent_entries", "memories", "failed_variants", "below_baseline_rows", "records"]:
        value = payload.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.extend(item for item in value.values() if isinstance(item, dict))
    for key in ["current_direction", "latest_direction", "direction", "summary"]:
        value = payload.get(key)
        if isinstance(value, dict):
            records.append(value)
    return records


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _evidence_bucket(entry: dict[str, Any]) -> str:
    ref = entry.get("ref") if isinstance(entry.get("ref"), dict) else entry
    source_type = str(ref.get("source_type") or entry.get("source_type") or "").lower()
    kind = str(entry.get("kind") or "").lower()
    label = " ".join(
        str(ref.get(key) or entry.get(key) or "").lower()
        for key in ("source_label", "source_path", "chunk_id", "path", "file", "claim", "summary", "reason")
    )
    if source_type in {"paper", "paper_chunk", "literature", "article", "reference"}:
        return "paper"
    if source_type in {"rebuttal", "review", "reviewer", "concern", "counterevidence"}:
        return "rebuttal"
    if source_type in {"failure_feedback", "failure_memory", "memory", "shared_memory", "negative_memory", "negative_result"}:
        return "failure_memory"
    if kind == "code_refs" or source_type == "code" or _looks_like_code_path(label):
        return "code"
    if any(token in label for token in ["rebuttal", "reviewer", "concern", "counterevidence"]):
        return "rebuttal"
    if any(token in label for token in ["failure", "negative_memory", "negative_result", "performance_feedback", "shared_memory", "memory"]):
        return "failure_memory"
    if any(token in label for token in ["paper", "arxiv", "pdf", "references/papers", "paper_chunks"]):
        return "paper"
    return "other"


def _ref_identity(entry: dict[str, Any]) -> str:
    ref = entry.get("ref") if isinstance(entry.get("ref"), dict) else entry
    parts = [
        str(entry.get("kind") or ""),
        str(entry.get("owner") or ""),
        str(ref.get("source_type") or ""),
        str(ref.get("chunk_id") or ""),
        str(ref.get("source_path") or ref.get("path") or ref.get("file") or ""),
        str(ref.get("source_label") or ref.get("label") or ref.get("source") or ""),
        str(ref.get("claim") or ref.get("summary") or ""),
    ]
    text = "|".join(part for part in parts if part)
    return text or json.dumps(entry, sort_keys=True, ensure_ascii=True, default=str)


def _coverage_ref_identity(entry: dict[str, Any]) -> str:
    ref = entry.get("ref") if isinstance(entry.get("ref"), dict) else entry
    source_type = str(ref.get("source_type") or entry.get("source_type") or "")
    chunk_id = str(ref.get("chunk_id") or "")
    source_path = str(ref.get("source_path") or ref.get("path") or ref.get("file") or "")
    source_label = str(ref.get("source_label") or ref.get("label") or ref.get("source") or "")
    if source_type.lower() == "code":
        label = source_path or source_label or chunk_id.removeprefix("code:")
        normalized = _normalize_surface_key(label)
    else:
        label = source_label or chunk_id or source_path
        normalized = label.strip().lower()
    text = "|".join(part for part in [source_type, normalized] if part)
    return text or _ref_identity(entry)


def _compact_quality_ref(entry: dict[str, Any]) -> dict[str, Any]:
    ref = entry.get("ref") if isinstance(entry.get("ref"), dict) else entry
    compact = {
        "kind": entry.get("kind"),
        "owner": entry.get("owner"),
        "source_type": ref.get("source_type") or entry.get("source_type"),
        "chunk_id": ref.get("chunk_id") or entry.get("chunk_id"),
        "source_path": ref.get("source_path") or ref.get("path") or entry.get("source_path") or entry.get("path"),
        "source_label": ref.get("source_label") or ref.get("label") or entry.get("source_label") or entry.get("label"),
        "claim": ref.get("claim") or ref.get("summary") or entry.get("claim") or entry.get("summary"),
        "status": entry.get("status"),
        "reason": entry.get("reason"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _counter_refs(
    *,
    direction: dict[str, Any],
    payload: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> list[str]:
    refs: list[str] = []
    for ref in _as_list(direction.get("counterevidence_refs")):
        ref_dict = ref if isinstance(ref, dict) else {"source_label": ref}
        if not _is_default_counter_ref(ref_dict):
            refs.append(_ref_identity(ref_dict))
    for idea in payload.get("selected_ideas") or []:
        if not isinstance(idea, dict):
            continue
        for ref in idea.get("counterevidence_refs") or []:
            if isinstance(ref, dict):
                refs.append(_ref_identity(ref))
    for item in evidence_bundle.get("items") or []:
        if isinstance(item, dict) and item.get("risks"):
            refs.append(_ref_identity(item))
    seen = set()
    result = []
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result


def _is_default_counter_ref(ref: dict[str, Any]) -> bool:
    source_type = str(ref.get("source_type") or "").lower()
    source_label = str(ref.get("source_label") or ref.get("label") or "").lower()
    claim = str(ref.get("claim") or "").lower()
    return source_type == "artifact" and source_label == "risk" and "s2 must verify" in claim


def _implementation_surface_targets(direction: dict[str, Any], payload: dict[str, Any]) -> set[str]:
    targets = set()
    for source in [
        direction.get("expected_files"),
        direction.get("implementation_surface_refs"),
        (payload.get("direction_decision") or {}).get("expected_files") if isinstance(payload.get("direction_decision"), dict) else [],
        (payload.get("direction_decision") or {}).get("implementation_surface_refs") if isinstance(payload.get("direction_decision"), dict) else [],
    ]:
        for item in _as_list(source):
            normalized = _normalize_surface_key(_ref_label(item))
            if normalized:
                targets.add(normalized)
    for idea in payload.get("selected_ideas") or []:
        if not isinstance(idea, dict):
            continue
        for field in ["expected_files", "implementation_surface_refs", "code_refs"]:
            for item in _as_list(idea.get(field)):
                normalized = _normalize_surface_key(_ref_label(item))
                if normalized:
                    targets.add(normalized)
    return targets


def _covered_implementation_surfaces(
    surface_targets: set[str],
    resolved_refs: list[dict[str, Any]],
    evidence_bundle: dict[str, Any],
) -> set[str]:
    if not surface_targets:
        return set()
    code_refs = [entry for entry in resolved_refs if _evidence_bucket(entry) == "code"]
    code_refs.extend(item for item in evidence_bundle.get("items") or [] if isinstance(item, dict) and _evidence_bucket(item) == "code")
    covered = set()
    for entry in code_refs:
        ref = entry.get("ref") if isinstance(entry.get("ref"), dict) else entry
        candidates = [
            ref.get("source_path"),
            ref.get("path"),
            ref.get("file"),
            ref.get("source_label"),
            ref.get("chunk_id"),
            entry.get("owner"),
        ]
        normalized_candidates = [_normalize_surface_key(str(value)) for value in candidates if value]
        for target in surface_targets:
            if any(candidate and (candidate == target or candidate.endswith("/" + target) or target.endswith("/" + candidate) or candidate in target or target in candidate) for candidate in normalized_candidates):
                covered.add(target)
    return covered


def _normalize_surface_key(value: str) -> str:
    text = value.strip().removeprefix("code:").split("#", 1)[0].split(":", 1)[0].strip()
    text = text.replace("\\", "/")
    text = text.removeprefix("./")
    return text.lower()


def _looks_like_code_path(value: str) -> bool:
    return any(token in value for token in [".py", ".toml", ".yaml", ".yml", ".json", "src/", "train", "aligner", "projector", "wrapper"])


def _extract_novelty_score(audit: dict[str, Any]) -> float:
    candidates = [
        audit.get("novelty_score"),
        (audit.get("latest") or {}).get("novelty_score") if isinstance(audit.get("latest"), dict) else None,
        ((audit.get("latest") or {}).get("audit") or {}).get("novelty_score") if isinstance((audit.get("latest") or {}).get("audit") if isinstance(audit.get("latest"), dict) else None, dict) else None,
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if score > 1.0:
                score = score / 10.0
            return max(0.0, min(1.0, score))
    if audit.get("passed") is False:
        return 0.0
    return 0.60


def _selected_idea(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, list):
        return {}
    items = [item for item in raw if isinstance(item, dict)]
    return next((item for item in items if item.get("selected")), items[0] if items else {})


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]
    if value in ("", {}, []):
        return []
    return [value]


def _refs_from_evidence_bundle(bundle: dict[str, Any], *, want_risk: bool) -> list[dict[str, Any]]:
    refs = []
    for item in bundle.get("items") or []:
        if not isinstance(item, dict):
            continue
        has_risk = bool(item.get("risks"))
        if has_risk != want_risk:
            continue
        refs.append(
            {
                "source_type": item.get("source_type") or "artifact",
                "source_label": item.get("chunk_id") or item.get("source_path") or "evidence_bundle",
                "source_path": item.get("source_path"),
                "claim": item.get("summary") or item.get("claim") or "",
            }
        )
    return refs


def _surface_refs_from_files(files: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": "code" if str(path).endswith(".py") else "artifact",
            "source_label": str(path),
            "source_path": str(path),
            "claim": "Candidate implementation surface",
        }
        for path in files
    ]


def _known_negative_refs(
    direction: dict[str, Any],
    decision: dict[str, Any],
    selected: dict[str, Any],
    negative_constraints: dict[str, Any],
    used_refs: list[str],
) -> list[Any]:
    refs = (
        _as_list(direction.get("known_negative_memory_refs"))
        or _as_list(decision.get("known_negative_memory_refs"))
        or _as_list(selected.get("known_negative_memory_refs"))
        or _as_list(negative_constraints.get("forbidden_idea_ids"))
    )
    return refs + [ref for ref in used_refs if ref not in refs]


def _default_metric_signature(mode: str, selected: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if mode == "c2c" or selected.get("mechanism_type"):
        return {
            "primary_metric": "three_dataset_mean",
            "expected_direction": "increase",
            "dataset_metrics": ["mmlu-redux_overall_accuracy", "ai2-arc_overall_accuracy", "openbookqa_overall_accuracy"],
            "diagnostics": _as_list(decision.get("failure_focus")) or ["mechanism trace changes when ablation switch is enabled"],
        }
    return {
        "primary_metric": "primary_metric",
        "expected_direction": "increase",
        "diagnostics": _as_list(decision.get("failure_focus")) or ["ablation-off control weakens the effect"],
    }


def _infer_integration_point(files: list[Any]) -> str:
    joined = " ".join(str(item).lower() for item in files)
    for name in ["aligner", "projector", "wrapper", "train_loss", "recipe"]:
        if name in joined:
            return name
    if "train" in joined:
        return "train_loss"
    return ""


def _snakeish(text: str) -> str:
    out = []
    last_underscore = False
    for char in text.lower():
        if char.isalnum():
            out.append(char)
            last_underscore = False
        elif not last_underscore:
            out.append("_")
            last_underscore = True
    return "".join(out).strip("_") or "selected_direction"


def _flatten_keys(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            keys.extend(_flatten_keys(item, prefix=next_prefix))
        return keys
    return [prefix] if prefix else []
