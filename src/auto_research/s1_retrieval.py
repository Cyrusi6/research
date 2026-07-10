"""Deterministic S1 evidence retrieval for C2C projects."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


REQUEST_PLAN_SCHEMA_VERSION = "c2c_s1_evidence_request_plan_v1"
DETERMINISTIC_BUNDLE_SCHEMA_VERSION = "c2c_s1_deterministic_evidence_bundle_v1"
DETERMINISTIC_TRACE_SCHEMA_VERSION = "c2c_s1_deterministic_retrieval_trace_v1"
RETRIEVER_VERSION = "c2c_s1_deterministic_keyword_v1"
SOURCE_TYPES = {"paper", "rebuttal", "code", "failure_memory", "feedback"}


def default_c2c_evidence_request_plan(*, topic: str = "c2c") -> dict[str, Any]:
    return {
        "schema_version": REQUEST_PLAN_SCHEMA_VERSION,
        "request_plan_id": _short_hash({"topic": topic, "version": REQUEST_PLAN_SCHEMA_VERSION}),
        "evidence_requests": [
            {
                "request_id": "paper_support",
                "source_type": "paper",
                "query": f"{topic} cache transfer mechanism evidence",
                "keywords": ["cache", "transfer", "mechanism"],
                "purpose": "support",
                "top_k": 2,
                "filters": {},
                "must_resolve": True,
            },
            {
                "request_id": "code_surface",
                "source_type": "code",
                "query": f"{topic} implementation surface aligner projector wrapper",
                "keywords": ["aligner", "projector", "wrapper", "cache"],
                "purpose": "implementation_surface",
                "top_k": 2,
                "filters": {},
                "must_resolve": True,
            },
            {
                "request_id": "counterevidence",
                "source_type": "rebuttal",
                "query": f"{topic} reviewer risk failure counterevidence",
                "keywords": ["risk", "failure", "coverage", "regression"],
                "purpose": "counterevidence",
                "top_k": 1,
                "filters": {},
                "must_resolve": True,
            },
            {
                "request_id": "failure_memory",
                "source_type": "failure_memory",
                "query": f"{topic} prior negative result failure memory",
                "keywords": ["failure", "negative", "regression", "collapse"],
                "purpose": "failure_memory",
                "top_k": 1,
                "filters": {},
                "must_resolve": False,
            },
        ],
        "candidate_direction_hypotheses": [
            {
                "hypothesis_id": "routing_control_signal",
                "mechanism_axis": "routing",
                "integration_point": "wrapper",
                "control_signal": "utility",
                "why_plausible": "C2C cache transfer may need an explicit runtime signal to route useful cache states.",
                "uncertainty_axes": ["paper_support", "code_surface", "counterevidence"],
            },
            {
                "hypothesis_id": "alignment_surface_signal",
                "mechanism_axis": "alignment",
                "integration_point": "aligner",
                "control_signal": "representation_match",
                "why_plausible": "Transfer quality may fail when the aligner/projector surface does not preserve compatible representations.",
                "uncertainty_axes": ["paper_support", "implementation_surface", "failure_memory"],
            },
        ],
        "uncertainty_axes": [
            {
                "axis_id": "mechanism_axis",
                "question": "Which high-level mechanism axis has the strongest support and least unresolved counterevidence?",
                "needed_sources": ["paper", "rebuttal", "failure_memory"],
            },
            {
                "axis_id": "implementation_surface",
                "question": "Which code entry point can S2 realistically edit without touching evaluation code?",
                "needed_sources": ["code"],
            },
        ],
        "discriminating_evidence_requests": [
            {
                "request_id": "paper_support",
                "distinguishes": ["routing_control_signal", "alignment_surface_signal"],
                "decision_if_supported": "Prefer the mechanism axis whose support evidence explains the baseline failure.",
                "decision_if_refuted": "Ask for additional paper/rebuttal evidence before choosing the direction.",
            },
            {
                "request_id": "code_surface",
                "distinguishes": ["wrapper", "aligner", "projector"],
                "decision_if_supported": "Prefer the integration point with direct editable code evidence and adjacent callgraph support.",
                "decision_if_refuted": "Do not advance to S2 until implementation-surface cards resolve.",
            },
        ],
        "must_have_before_direction": [
            {"source_type": "paper", "purpose": "support", "minimum": 2},
            {"source_type": "code", "purpose": "implementation_surface", "minimum": 2},
            {"source_type": "rebuttal", "purpose": "counterevidence", "minimum": 1},
        ],
        "required_source_coverage": {"paper": 2, "code": 2, "counterevidence": 1, "failure_memory": 0},
        "retrieval_budget": {"top_k_per_request": 2, "max_total_items": 12, "min_score": 0.0},
        "forbidden_outputs": ["direction_decision", "selected_ideas", "evidence_bundle", "expected_files"],
        "request_rationale": "Retrieve support, implementation, counterevidence, and prior failure evidence before S1 chooses a direction.",
    }


def normalize_c2c_evidence_request_plan(plan: dict[str, Any], *, topic: str = "c2c") -> dict[str, Any]:
    fallback = default_c2c_evidence_request_plan(topic=topic)
    if not isinstance(plan, dict):
        return fallback
    normalized = dict(plan)
    normalized["schema_version"] = str(normalized.get("schema_version") or REQUEST_PLAN_SCHEMA_VERSION)
    normalized["request_plan_id"] = str(normalized.get("request_plan_id") or _short_hash({"topic": topic, "requests": normalized.get("evidence_requests")}))
    requests = []
    for idx, request in enumerate(normalized.get("evidence_requests") or []):
        if not isinstance(request, dict):
            continue
        source_type = _normalize_request_source_type(request.get("source_type"))
        purpose = str(request.get("purpose") or request.get("desired_evidence") or _default_purpose(source_type))
        requests.append(
            {
                "request_id": str(request.get("request_id") or f"req_{idx + 1}_{source_type}"),
                "source_type": source_type,
                "query": str(request.get("query") or " ".join(str(item) for item in request.get("keywords") or []) or source_type),
                "keywords": _dedupe([str(item) for item in request.get("keywords") or [] if item]),
                "purpose": purpose,
                "top_k": max(1, int(request.get("top_k") or 2)),
                "filters": request.get("filters") if isinstance(request.get("filters"), dict) else {},
                "must_resolve": bool(request.get("must_resolve", source_type in {"paper", "code"} or purpose == "counterevidence")),
            }
        )
    normalized["evidence_requests"] = requests or fallback["evidence_requests"]
    normalized["candidate_direction_hypotheses"] = _normalize_candidate_direction_hypotheses(
        normalized.get("candidate_direction_hypotheses"),
        fallback=fallback["candidate_direction_hypotheses"],
    )
    normalized["uncertainty_axes"] = _normalize_uncertainty_axes(normalized.get("uncertainty_axes"), fallback=fallback["uncertainty_axes"])
    normalized["discriminating_evidence_requests"] = _normalize_discriminating_evidence_requests(
        normalized.get("discriminating_evidence_requests"),
        fallback=fallback["discriminating_evidence_requests"],
    )
    normalized["discriminating_evidence_requests"] = _align_discriminating_request_ids(normalized["discriminating_evidence_requests"], normalized["evidence_requests"])
    normalized["must_have_before_direction"] = _normalize_must_have_before_direction(
        normalized.get("must_have_before_direction"),
        fallback=fallback["must_have_before_direction"],
    )
    normalized.setdefault("required_source_coverage", fallback["required_source_coverage"])
    normalized.setdefault("retrieval_budget", fallback["retrieval_budget"])
    normalized["forbidden_outputs"] = ["direction_decision", "selected_ideas", "evidence_bundle", "expected_files"]
    normalized.setdefault("request_rationale", fallback["request_rationale"])
    return normalized


def validate_c2c_evidence_request_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["evidence request plan must be an object"]
    for field in ["direction_decision", "selected_ideas", "evidence_bundle", "expected_files"]:
        if field in plan:
            errors.append(f"evidence_request_plan must not include {field}")
    if not isinstance(plan.get("evidence_requests"), list) or not plan.get("evidence_requests"):
        errors.append("evidence_requests must be a non-empty list")
        return errors
    seen = set()
    source_types = set()
    purposes = set()
    for idx, request in enumerate(plan.get("evidence_requests") or []):
        if not isinstance(request, dict):
            errors.append(f"evidence_requests[{idx}] must be an object")
            continue
        request_id = str(request.get("request_id") or "")
        if not request_id:
            errors.append(f"evidence_requests[{idx}].request_id missing")
        elif request_id in seen:
            errors.append(f"duplicate request_id: {request_id}")
        seen.add(request_id)
        source_type = str(request.get("source_type") or "")
        source_types.add(source_type)
        if source_type not in SOURCE_TYPES:
            errors.append(f"evidence_requests[{idx}].source_type must be one of {sorted(SOURCE_TYPES)}")
        if not str(request.get("query") or "").strip():
            errors.append(f"evidence_requests[{idx}].query missing")
        purpose = str(request.get("purpose") or "")
        purposes.add(purpose)
        if not purpose:
            errors.append(f"evidence_requests[{idx}].purpose missing")
        if int(request.get("top_k") or 0) <= 0:
            errors.append(f"evidence_requests[{idx}].top_k must be positive")
    if "paper" not in source_types:
        errors.append("evidence_requests must include paper source coverage")
    if "code" not in source_types:
        errors.append("evidence_requests must include code source coverage")
    if "counterevidence" not in purposes and not (source_types & {"rebuttal", "failure_memory", "feedback"}):
        errors.append("evidence_requests must include counterevidence request coverage")
    hypotheses = plan.get("candidate_direction_hypotheses")
    if not isinstance(hypotheses, list) or len([item for item in hypotheses if isinstance(item, dict)]) < 2:
        errors.append("candidate_direction_hypotheses must contain at least two competing hypotheses")
    axes = plan.get("uncertainty_axes")
    if not isinstance(axes, list) or not axes:
        errors.append("uncertainty_axes must be a non-empty list")
    discriminators = plan.get("discriminating_evidence_requests")
    if not isinstance(discriminators, list) or not discriminators:
        errors.append("discriminating_evidence_requests must be a non-empty list")
    else:
        known_request_ids = {str(item.get("request_id") or "") for item in plan.get("evidence_requests") or [] if isinstance(item, dict)}
        for idx, item in enumerate(discriminators):
            if not isinstance(item, dict):
                errors.append(f"discriminating_evidence_requests[{idx}] must be an object")
                continue
            request_id = str(item.get("request_id") or "")
            if not request_id:
                errors.append(f"discriminating_evidence_requests[{idx}].request_id missing")
            elif request_id not in known_request_ids:
                errors.append(f"discriminating_evidence_requests[{idx}].request_id must reference evidence_requests")
            if not item.get("distinguishes"):
                errors.append(f"discriminating_evidence_requests[{idx}].distinguishes must be non-empty")
    must_have = plan.get("must_have_before_direction")
    if not isinstance(must_have, list) or not must_have:
        errors.append("must_have_before_direction must be a non-empty list")
    return errors


def _normalize_candidate_direction_hypotheses(value: Any, *, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or f"hypothesis_{idx + 1}").strip()
        if not hypothesis_id:
            continue
        normalized.append(
            {
                "hypothesis_id": hypothesis_id,
                "mechanism_axis": str(item.get("mechanism_axis") or item.get("axis") or "unknown"),
                "integration_point": str(item.get("integration_point") or item.get("surface") or "unknown"),
                "control_signal": str(item.get("control_signal") or item.get("signal") or "unknown"),
                "why_plausible": str(item.get("why_plausible") or item.get("rationale") or "Requires deterministic evidence before direction selection."),
                "uncertainty_axes": _dedupe([str(axis) for axis in item.get("uncertainty_axes") or [] if axis]),
            }
        )
    return normalized if len(normalized) >= 2 else [dict(item) for item in fallback]


def _normalize_uncertainty_axes(value: Any, *, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, str):
            axis_id = item.strip()
            if axis_id:
                normalized.append({"axis_id": axis_id, "question": f"Resolve uncertainty around {axis_id}.", "needed_sources": []})
            continue
        if not isinstance(item, dict):
            continue
        axis_id = str(item.get("axis_id") or item.get("id") or item.get("axis") or f"axis_{idx + 1}").strip()
        if not axis_id:
            continue
        normalized.append(
            {
                "axis_id": axis_id,
                "question": str(item.get("question") or item.get("rationale") or f"Resolve uncertainty around {axis_id}."),
                "needed_sources": _dedupe([_normalize_request_source_type(source) for source in item.get("needed_sources") or item.get("source_types") or [] if source]),
            }
        )
    return normalized or [dict(item) for item in fallback]


def _normalize_discriminating_evidence_requests(value: Any, *, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            request_id = item.strip()
            if request_id:
                normalized.append(
                    {
                        "request_id": request_id,
                        "distinguishes": [],
                        "decision_if_supported": "Prefer directions supported by this evidence.",
                        "decision_if_refuted": "Request more evidence before selecting a direction.",
                    }
                )
            continue
        if not isinstance(item, dict):
            continue
        request_id = str(item.get("request_id") or item.get("evidence_request_id") or "").strip()
        if not request_id:
            continue
        normalized.append(
            {
                "request_id": request_id,
                "distinguishes": _dedupe([str(value) for value in item.get("distinguishes") or item.get("compares") or [] if value]),
                "decision_if_supported": str(item.get("decision_if_supported") or item.get("if_supported") or "Prefer the direction supported by this evidence."),
                "decision_if_refuted": str(item.get("decision_if_refuted") or item.get("if_refuted") or "Request more evidence before selecting a direction."),
            }
        )
    return normalized or [dict(item) for item in fallback]


def _align_discriminating_request_ids(discriminators: list[dict[str, Any]], requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_request_ids = {str(item.get("request_id") or "") for item in requests if isinstance(item, dict)}
    aligned: list[dict[str, Any]] = []
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        request_id = str(entry.get("request_id") or "")
        if request_id not in known_request_ids:
            replacement = _replacement_discriminator_request_id(request_id, requests)
            if replacement:
                entry["request_id"] = replacement
        if str(entry.get("request_id") or "") in known_request_ids:
            aligned.append(entry)
    return aligned


def _replacement_discriminator_request_id(request_id: str, requests: list[dict[str, Any]]) -> str:
    request_id = request_id.lower()
    preferred_source = "code" if "code" in request_id or "surface" in request_id else "paper" if "paper" in request_id or "support" in request_id else ""
    preferred_purpose = "counterevidence" if "counter" in request_id or "risk" in request_id else "implementation_surface" if "surface" in request_id or "code" in request_id else ""
    for request in requests:
        if not isinstance(request, dict):
            continue
        if preferred_purpose and str(request.get("purpose") or "") == preferred_purpose:
            return str(request.get("request_id") or "")
    for request in requests:
        if not isinstance(request, dict):
            continue
        if preferred_source and str(request.get("source_type") or "") == preferred_source:
            return str(request.get("request_id") or "")
    return str((requests[0] or {}).get("request_id") or "") if requests else ""


def _normalize_must_have_before_direction(value: Any, *, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.append({"source_type": _normalize_request_source_type(item), "purpose": item, "minimum": 1})
            continue
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "source_type": _normalize_request_source_type(item.get("source_type") or item.get("source")),
                "purpose": str(item.get("purpose") or item.get("evidence_type") or "support"),
                "minimum": max(0, int(item.get("minimum") or item.get("min_count") or 1)),
            }
        )
    return normalized or [dict(item) for item in fallback]


def retrieve_s1_c2c_requested_evidence(
    evidence_request_plan: dict[str, Any],
    *,
    chunk_index: dict[str, Any] | None = None,
    paper_chunks: list[dict[str, Any]] | None = None,
    rebuttal_chunks: list[dict[str, Any]] | None = None,
    code_chunks: list[dict[str, Any]] | None = None,
    code_edges: list[dict[str, Any]] | None = None,
    code_retrieval_index: dict[str, Any] | None = None,
    implementation_surface_map: dict[str, Any] | None = None,
    negative_memory: dict[str, Any] | None = None,
    feedback: list[dict[str, Any]] | None = None,
    shared_memory: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = normalize_c2c_evidence_request_plan(evidence_request_plan)
    retriever_cfg = (((config or {}).get("ideation") or {}).get("c2c_s1_two_phase") or {}).get("retriever") or {}
    min_score = float(retriever_cfg.get("min_score", (plan.get("retrieval_budget") or {}).get("min_score", 0.0)) or 0.0)
    max_total_items = int(retriever_cfg.get("max_total_items", (plan.get("retrieval_budget") or {}).get("max_total_items", 12)) or 12)
    candidates = _candidate_pool(
        chunk_index=chunk_index or {},
        paper_chunks=paper_chunks or [],
        rebuttal_chunks=rebuttal_chunks or [],
        code_chunks=code_chunks or [],
        code_edges=code_edges or [],
        code_retrieval_index=code_retrieval_index or {},
        implementation_surface_map=implementation_surface_map or {},
        negative_memory=negative_memory or {},
        feedback=feedback or [],
        shared_memory=shared_memory or {},
    )
    candidate_counts = _candidate_counts(candidates)
    selected: list[dict[str, Any]] = []
    selected_keys = set()
    request_results = []
    rejected_top_candidates = []
    unfilled_requests = []
    code_neighborhood_expansions: list[dict[str, Any]] = []
    code_neighborhood_enabled = bool(retriever_cfg.get("enable_code_neighborhood_expansion", True))
    code_neighborhood_max = int(retriever_cfg.get("code_neighborhood_max_per_request", 2) or 2)
    for request in plan["evidence_requests"]:
        scored = [_score_candidate(request, candidate, implementation_surface_map or {}) for candidate in candidates]
        source_only = [
            item
            for item in scored
            if _request_source_matches(request, item["candidate"]) and not item.get("relevance_match")
        ]
        rejected_top_candidates.extend(
            _trace_candidate(request, item["candidate"], item, reason="source_only_match")
            for item in sorted(source_only, key=lambda value: -float(value.get("score") or 0.0))[:5]
        )
        scored = [
            item
            for item in scored
            if item["score"] >= min_score
            and _request_source_matches(request, item["candidate"])
            and item.get("relevance_match")
        ]
        scored.sort(key=lambda item: (-item["score"], item["candidate"]["source_type"], item["candidate"]["locator"]))
        request_selected = []
        for score_item in scored:
            candidate = score_item["candidate"]
            key = _candidate_key(candidate)
            if key in selected_keys:
                rejected_top_candidates.append(_trace_candidate(request, candidate, score_item, reason="dedupe"))
                continue
            evidence_item = _candidate_to_evidence_item(candidate, request, score_item)
            selected.append(evidence_item)
            selected_keys.add(key)
            request_selected.append(evidence_item)
            if len(request_selected) >= int(request.get("top_k") or 1) or len(selected) >= max_total_items:
                break
        if code_neighborhood_enabled and len(selected) < max_total_items:
            added_items, expansions = _expand_code_neighborhood(
                request,
                request_selected=request_selected,
                candidates=candidates,
                code_edges=code_edges or [],
                selected_keys=selected_keys,
                implementation_surface_map=implementation_surface_map or {},
                max_neighbors=code_neighborhood_max,
                remaining=max(0, max_total_items - len(selected)),
            )
            selected.extend(added_items)
            request_selected.extend(added_items)
            code_neighborhood_expansions.extend(expansions)
        if request.get("must_resolve") and not request_selected:
            unfilled_requests.append({"request_id": request.get("request_id"), "source_type": request.get("source_type"), "purpose": request.get("purpose"), "reason": "no_candidate_above_threshold"})
        request_results.append(
            {
                "request_id": request.get("request_id"),
                "source_type": request.get("source_type"),
                "purpose": request.get("purpose"),
                "query": request.get("query"),
                "candidate_count": len(scored),
                "selected_refs": [item["ref"] for item in request_selected],
                "unfilled": bool(request.get("must_resolve") and not request_selected),
            }
        )
        if len(selected) >= max_total_items:
            break
    bundle = {
        "schema_version": DETERMINISTIC_BUNDLE_SCHEMA_VERSION,
        "producer": "deterministic_retriever",
        "retriever_version": RETRIEVER_VERSION,
        "request_plan_id": plan.get("request_plan_id"),
        "items": selected,
    }
    coverage = _coverage(selected)
    trace = {
        "schema_version": DETERMINISTIC_TRACE_SCHEMA_VERSION,
        "retriever_version": RETRIEVER_VERSION,
        "request_plan_id": plan.get("request_plan_id"),
        "requests": request_results,
        "evidence_requests": plan.get("evidence_requests") or [],
        "candidate_counts": candidate_counts,
        "selected_refs": [item["ref"] for item in selected],
        "resolved_refs": [item["ref"] for item in selected],
        "unresolved_refs": [],
        "resolved_ref_count": len(selected),
        "unresolved_ref_count": 0,
        "unfilled_requests": unfilled_requests,
        "unfilled_must_resolve_requests": [item for item in unfilled_requests],
        "coverage": coverage,
        "coverage_contributors": _coverage_contributors(selected),
        "code_neighborhood_expansions": code_neighborhood_expansions,
        "candidate_direction_hypotheses": plan.get("candidate_direction_hypotheses") or [],
        "uncertainty_axes": plan.get("uncertainty_axes") or [],
        "discriminating_evidence_requests": plan.get("discriminating_evidence_requests") or [],
        "must_have_before_direction": plan.get("must_have_before_direction") or [],
        "rejected_top_candidates": rejected_top_candidates[:40],
        "deterministic": True,
        "retrieval_inputs_hash": _short_hash(
            {
                "request_plan": plan,
                "chunk_index": _compact_for_hash(chunk_index or {}),
                "paper_chunks": _compact_for_hash(paper_chunks or []),
                "rebuttal_chunks": _compact_for_hash(rebuttal_chunks or []),
                "code_chunks": _compact_for_hash(code_chunks or []),
                "code_edges": _compact_for_hash(code_edges or []),
                "code_retrieval_index": _compact_for_hash(code_retrieval_index or {}),
                "implementation_surface_map": _compact_for_hash(implementation_surface_map or {}),
                "negative_memory": _compact_for_hash(negative_memory or {}),
                "feedback": _compact_for_hash(feedback or []),
            }
        ),
        "quality_gate": {},
        "direction_fingerprint": {},
    }
    return bundle, trace


def bundle_ref_set(evidence_bundle: dict[str, Any]) -> set[str]:
    refs = set()
    for item in evidence_bundle.get("items") or []:
        if not isinstance(item, dict):
            continue
        ref = item.get("ref") if isinstance(item.get("ref"), dict) else item
        refs.add(canonical_ref_key(ref))
    return refs


def canonical_ref_key(ref: dict[str, Any]) -> str:
    if not isinstance(ref, dict):
        return ""
    payload = {
        "source_type": str(ref.get("source_type") or ""),
        "chunk_id": str(ref.get("chunk_id") or ""),
        "source_path": str(ref.get("source_path") or ref.get("path") or ref.get("file") or ""),
        "source_label": str(ref.get("source_label") or ref.get("label") or ref.get("source") or ""),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _candidate_pool(
    *,
    chunk_index: dict[str, Any],
    paper_chunks: list[dict[str, Any]],
    rebuttal_chunks: list[dict[str, Any]],
    code_chunks: list[dict[str, Any]],
    code_edges: list[dict[str, Any]],
    code_retrieval_index: dict[str, Any],
    implementation_surface_map: dict[str, Any],
    negative_memory: dict[str, Any],
    feedback: list[dict[str, Any]],
    shared_memory: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    for source_type, chunks in [("paper", paper_chunks), ("rebuttal", rebuttal_chunks), ("code", code_chunks)]:
        for chunk in chunks:
            if isinstance(chunk, dict):
                candidates.append(_chunk_candidate(chunk, source_type=source_type))
    for entry in chunk_index.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        source_type = _normalize_candidate_source_type(entry.get("source_type"))
        if source_type in {"paper", "rebuttal", "code"}:
            candidates.append(_chunk_candidate(entry, source_type=source_type))
    for edge in code_edges:
        if isinstance(edge, dict):
            candidates.append(_record_candidate(edge, source_type="code", default_label="code_edge"))
    for query in code_retrieval_index.get("default_queries") or []:
        if isinstance(query, dict):
            candidates.append(_record_candidate(query, source_type="code", default_label="code_retrieval_index"))
    surfaces = (implementation_surface_map.get("surfaces") or {}) if isinstance(implementation_surface_map, dict) else {}
    for surface_key, surface in surfaces.items():
        entries = surface if isinstance(surface, list) else [surface]
        for entry in entries:
            if isinstance(entry, dict):
                record = dict(entry)
                record.setdefault("path", str(surface_key))
                candidates.append(_record_candidate(record, source_type="code", default_label=str(surface_key)))
    for record in _flatten_memory_records(negative_memory):
        candidates.append(_record_candidate(record, source_type="failure_memory", default_label="negative_memory"))
    for record in feedback:
        if isinstance(record, dict):
            candidates.append(_record_candidate(record, source_type="feedback", default_label="feedback"))
    for record in _flatten_memory_records(shared_memory):
        candidates.append(_record_candidate(record, source_type="failure_memory", default_label="shared_memory"))
    unique = {}
    for candidate in candidates:
        unique.setdefault(_candidate_key(candidate), candidate)
    return list(unique.values())


def _chunk_candidate(chunk: dict[str, Any], *, source_type: str) -> dict[str, Any]:
    locator = str(chunk.get("chunk_id") or chunk.get("path") or chunk.get("source_path") or chunk.get("symbol") or source_type)
    text = _candidate_text(chunk)
    return {
        "source_type": source_type,
        "locator": locator,
        "chunk_id": chunk.get("chunk_id"),
        "source_path": chunk.get("source_path") or chunk.get("path") or chunk.get("file_path"),
        "source_label": chunk.get("chunk_id") or chunk.get("symbol") or chunk.get("path") or chunk.get("source_path"),
        "symbol": chunk.get("symbol"),
        "text": text,
        "keywords": _dedupe([*(chunk.get("keywords") or []), *(chunk.get("retrieval_keywords") or []), *(chunk.get("mechanism_tags") or [])]),
        "record": chunk,
    }


def _record_candidate(record: dict[str, Any], *, source_type: str, default_label: str) -> dict[str, Any]:
    locator = str(record.get("memory_id") or record.get("id") or record.get("chunk_id") or record.get("path") or record.get("source_path") or default_label)
    source_path = record.get("source_path") or record.get("path") or _default_record_source_path(source_type, default_label)
    return {
        "source_type": source_type,
        "locator": locator,
        "chunk_id": record.get("chunk_id"),
        "source_path": source_path,
        "source_label": record.get("memory_id") or record.get("id") or record.get("source_label") or record.get("chunk_id") or record.get("path") or default_label,
        "symbol": record.get("symbol"),
        "text": _candidate_text(record),
        "keywords": _dedupe([str(item) for item in record.get("keywords") or record.get("retrieval_keywords") or [] if item]),
        "record": record,
    }


def _default_record_source_path(source_type: str, default_label: str) -> str:
    if source_type == "failure_memory":
        return "intake/c2c/negative_result_memory.json" if "shared" not in default_label else "intake/shared_method_failure_memory.json"
    if source_type == "feedback":
        return "experiment/results/failure_feedback.json"
    return ""


def _candidate_to_evidence_item(candidate: dict[str, Any], request: dict[str, Any], score_item: dict[str, Any]) -> dict[str, Any]:
    evidence_id = "ev_" + _short_hash({"request_id": request.get("request_id"), "candidate": _candidate_key(candidate)})
    source_type = _bundle_source_type(candidate["source_type"])
    ref = {
        "source_type": source_type,
        "source_label": str(candidate.get("source_label") or candidate.get("locator") or evidence_id),
        "claim": _summary(candidate),
    }
    if candidate.get("chunk_id"):
        ref["chunk_id"] = str(candidate["chunk_id"])
    if candidate.get("source_path"):
        ref["source_path"] = str(candidate["source_path"])
    item = {
        "evidence_id": evidence_id,
        "ref": ref,
        "request_id": request.get("request_id"),
        "purpose": request.get("purpose"),
        "source_type": source_type,
        "locator": candidate.get("locator"),
        "chunk_id": ref.get("chunk_id"),
        "source_path": ref.get("source_path"),
        "source_label": ref.get("source_label"),
        "summary": ref["claim"],
        "text": _excerpt(candidate),
        "excerpt": _excerpt(candidate),
        "supports": [request.get("purpose")] if request.get("purpose") != "counterevidence" else [],
        "risks": ["counterevidence"] if request.get("purpose") == "counterevidence" or source_type in {"rebuttal", "failure_feedback"} else [],
        "score": round(float(score_item["score"]), 4),
        "score_components": score_item["components"],
        "why_selected": _why_selected(request, score_item),
        "source_hash": _short_hash(candidate.get("record") or candidate),
    }
    return {key: value for key, value in item.items() if value not in (None, "", [], {})}


def _expand_code_neighborhood(
    request: dict[str, Any],
    *,
    request_selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    code_edges: list[dict[str, Any]],
    selected_keys: set[str],
    implementation_surface_map: dict[str, Any],
    max_neighbors: int,
    remaining: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if remaining <= 0 or max_neighbors <= 0 or not request_selected:
        return [], []
    if str(request.get("source_type") or "") != "code" and str(request.get("purpose") or "") != "implementation_surface":
        return [], []
    edge_neighbor_ids = _edge_neighbor_ids(request_selected, code_edges)
    seed_paths = {str(item.get("source_path") or "") for item in request_selected if item.get("source_path")}
    scored_neighbors = []
    for candidate in candidates:
        if candidate.get("source_type") != "code" or _candidate_is_doc_like(candidate):
            continue
        if not candidate.get("chunk_id") and str(candidate.get("source_path") or "") in seed_paths:
            continue
        key = _candidate_key(candidate)
        if key in selected_keys:
            continue
        reason, seed_ref = _code_neighbor_reason(candidate, request_selected, edge_neighbor_ids=edge_neighbor_ids, seed_paths=seed_paths)
        if not reason:
            continue
        score_item = _score_candidate(request, candidate, implementation_surface_map)
        score_item["score"] = float(score_item.get("score") or 0.0) + 0.8
        score_item.setdefault("components", {})["code_neighborhood_boost"] = 0.8
        priority = 0 if reason in {"callgraph_edge", "same_file_neighbor_edge"} else 1
        scored_neighbors.append((priority, -float(score_item["score"]), str(candidate.get("source_path") or ""), str(candidate.get("locator") or ""), candidate, score_item, reason, seed_ref))
    scored_neighbors.sort(key=lambda item: item[:4])
    added: list[dict[str, Any]] = []
    expansions: list[dict[str, Any]] = []
    for _, _, _, _, candidate, score_item, reason, seed_ref in scored_neighbors[: min(max_neighbors, remaining)]:
        key = _candidate_key(candidate)
        if key in selected_keys:
            continue
        neighbor_request = dict(request)
        neighbor_request["request_id"] = f"{request.get('request_id')}:code_neighborhood"
        neighbor_request["purpose"] = "implementation_surface_neighbor"
        evidence_item = _candidate_to_evidence_item(candidate, neighbor_request, score_item)
        supports = list(evidence_item.get("supports") or [])
        supports.append("code_neighborhood")
        evidence_item["supports"] = _dedupe(supports)
        evidence_item["why_selected"] = f"Added deterministic code-neighborhood evidence via {reason} from {seed_ref}."
        selected_keys.add(key)
        added.append(evidence_item)
        expansions.append(
            {
                "request_id": request.get("request_id"),
                "seed_ref": seed_ref,
                "added_ref": evidence_item.get("ref"),
                "reason": reason,
            }
        )
    return added, expansions


def _edge_neighbor_ids(selected_items: list[dict[str, Any]], code_edges: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    seed_ids: set[str] = set()
    for item in selected_items:
        seed_ids.update(_code_item_ids(item))
    neighbors: dict[str, str] = {}
    for edge in code_edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if not src or not dst:
            continue
        edge_type = str(edge.get("edge_type") or "code_edge")
        if src in seed_ids:
            neighbors[dst] = {"edge_type": edge_type, "seed_id": src}
        if dst in seed_ids:
            neighbors[src] = {"edge_type": edge_type, "seed_id": dst}
    return neighbors


def _code_neighbor_reason(
    candidate: dict[str, Any],
    selected_items: list[dict[str, Any]],
    *,
    edge_neighbor_ids: dict[str, dict[str, str]],
    seed_paths: set[str],
) -> tuple[str, str]:
    candidate_ids = _candidate_code_ids(candidate)
    for candidate_id in candidate_ids:
        if candidate_id in edge_neighbor_ids:
            edge = edge_neighbor_ids[candidate_id]
            return _edge_type_reason(edge.get("edge_type") or ""), str(edge.get("seed_id") or "")
    candidate_path = str(candidate.get("source_path") or "")
    if candidate.get("chunk_id") and candidate_path and candidate_path in seed_paths:
        seed = next((item for item in selected_items if str(item.get("source_path") or "") == candidate_path), {})
        return "same_file_neighbor", str(seed.get("chunk_id") or seed.get("source_label") or candidate_path)
    return "", ""


def _code_item_ids(item: dict[str, Any]) -> set[str]:
    ref = item.get("ref") if isinstance(item.get("ref"), dict) else {}
    values = [
        item.get("chunk_id"),
        item.get("locator"),
        item.get("source_label"),
        item.get("source_path"),
        ref.get("chunk_id"),
        ref.get("source_label"),
        ref.get("source_path"),
    ]
    return {str(value) for value in values if value}


def _candidate_code_ids(candidate: dict[str, Any]) -> set[str]:
    values = [candidate.get("chunk_id"), candidate.get("locator"), candidate.get("source_label"), candidate.get("source_path"), candidate.get("symbol")]
    return {str(value) for value in values if value}


def _edge_type_reason(edge_type: str) -> str:
    if edge_type == "same_file_neighbor":
        return "same_file_neighbor_edge"
    if edge_type in {"resolved_call", "calls", "tested_by", "tests_symbol", "config_key_defined_in"}:
        return "callgraph_edge"
    return "code_edge"


def _score_candidate(request: dict[str, Any], candidate: dict[str, Any], implementation_surface_map: dict[str, Any]) -> dict[str, Any]:
    query_tokens = _tokens(str(request.get("query") or ""))
    keyword_tokens = set().union(*(_tokens(str(item)) for item in request.get("keywords") or [])) if request.get("keywords") else set()
    candidate_tokens = _tokens(candidate.get("text") or "") | set(_tokens(" ".join(candidate.get("keywords") or []))) | _tokens(candidate.get("locator") or "")
    keyword_overlap = len(keyword_tokens & candidate_tokens)
    query_token_score = len(query_tokens & candidate_tokens)
    source_type_match = 0.0 if _request_source_matches(request, candidate) else -10.0
    implementation_surface_boost = 0.0
    if candidate.get("source_type") == "code":
        surface_status = _candidate_surface_status(candidate, implementation_surface_map)
        if surface_status == "allowed":
            implementation_surface_boost = 4.0
        elif surface_status:
            implementation_surface_boost = 1.0
        if request.get("purpose") == "implementation_surface" and _candidate_is_doc_like(candidate):
            implementation_surface_boost -= 1.0
    counterevidence_boost = 1.0 if request.get("purpose") == "counterevidence" and candidate.get("source_type") in {"rebuttal", "failure_memory", "feedback"} else 0.0
    recent_failure_boost = 0.8 if candidate.get("source_type") in {"failure_memory", "feedback"} else 0.0
    relevance_match = keyword_overlap > 0 or query_token_score > 0
    score = source_type_match + keyword_overlap * 1.2 + query_token_score * 0.4 + implementation_surface_boost + counterevidence_boost + recent_failure_boost
    return {
        "candidate": candidate,
        "score": score,
        "relevance_match": relevance_match,
        "components": {
            "keyword_overlap": keyword_overlap,
            "query_token_score": query_token_score,
            "source_type_match": source_type_match,
            "implementation_surface_boost": implementation_surface_boost,
            "counterevidence_boost": counterevidence_boost,
            "recent_failure_boost": recent_failure_boost,
        },
    }


def _request_source_matches(request: dict[str, Any], candidate: dict[str, Any]) -> bool:
    source_type = str(request.get("source_type") or "")
    candidate_type = str(candidate.get("source_type") or "")
    purpose = str(request.get("purpose") or "")
    if source_type == candidate_type:
        return True
    if purpose == "counterevidence" and candidate_type in {"rebuttal", "failure_memory", "feedback"}:
        return True
    if source_type == "failure_memory" and candidate_type == "feedback":
        return True
    return False


def _candidate_surface_status(candidate: dict[str, Any], implementation_surface_map: dict[str, Any]) -> str:
    surfaces = implementation_surface_map.get("surfaces") if isinstance(implementation_surface_map, dict) else {}
    if not isinstance(surfaces, dict):
        return ""
    targets = _candidate_match_targets(candidate)
    status = ""
    for surface_key, surface_value in surfaces.items():
        entries = surface_value if isinstance(surface_value, list) else [surface_value]
        if _target_matches_surface(str(surface_key), targets):
            status = _stronger_surface_status(status, _surface_entry_status(surface_value))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_targets = [
                str(entry.get("path") or ""),
                str(entry.get("source_path") or ""),
                str(entry.get("chunk_id") or ""),
                str(entry.get("symbol") or ""),
            ]
            if any(_target_matches_surface(target, targets) for target in entry_targets):
                status = _stronger_surface_status(status, _surface_entry_status(entry))
    return status


def _candidate_match_targets(candidate: dict[str, Any]) -> set[str]:
    targets = set()
    for key in ["source_path", "path", "locator", "chunk_id", "source_label", "symbol"]:
        value = candidate.get(key)
        if value:
            targets.add(_normalize_surface_target(str(value)))
    return {item for item in targets if item}


def _target_matches_surface(target: str, candidate_targets: set[str]) -> bool:
    normalized = _normalize_surface_target(target)
    if not normalized:
        return False
    return any(
        item == normalized
        or item.endswith("/" + normalized)
        or normalized.endswith("/" + item)
        or normalized in item
        or item in normalized
        for item in candidate_targets
    )


def _surface_entry_status(entry: Any) -> str:
    if isinstance(entry, dict) and str(entry.get("edit_surface") or "").lower() == "allowed":
        return "allowed"
    return "surface"


def _stronger_surface_status(current: str, incoming: str) -> str:
    if incoming == "allowed" or current == "allowed":
        return "allowed"
    return incoming or current


def _candidate_is_doc_like(candidate: dict[str, Any]) -> bool:
    path = str(candidate.get("source_path") or candidate.get("path") or candidate.get("locator") or "").lower()
    return path.endswith((".md", ".rst", ".txt"))


def _normalize_surface_target(value: str) -> str:
    text = str(value).strip().replace("\\", "/").removeprefix("./")
    if "::" in text:
        text = text.split("::", 1)[0]
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.lower()


def _coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"paper": 0, "rebuttal": 0, "code": 0, "failure_memory": 0, "feedback": 0, "counterevidence": 0}
    seen_by_type: dict[str, set[str]] = {key: set() for key in counts}
    for item in items:
        source_type = str(item.get("source_type") or "")
        key = canonical_ref_key(item.get("ref") if isinstance(item.get("ref"), dict) else item)
        if source_type == "failure_feedback":
            source_type = "failure_memory"
        if source_type in seen_by_type:
            seen_by_type[source_type].add(key)
        if item.get("risks"):
            seen_by_type["counterevidence"].add(key)
    return {key: len(value) for key, value in seen_by_type.items()}


def _coverage_contributors(items: list[dict[str, Any]]) -> dict[str, Any]:
    contributors: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        ref = item.get("ref") if isinstance(item.get("ref"), dict) else {}
        if not ref:
            continue
        source_type = str(item.get("source_type") or ref.get("source_type") or "unknown")
        if source_type == "failure_feedback":
            source_type = "failure_memory"
        for key in [source_type, *[str(value) for value in item.get("supports") or [] if value], *[str(value) for value in item.get("risks") or [] if value]]:
            contributors.setdefault(key, [])
            if canonical_ref_key(ref) not in {canonical_ref_key(existing) for existing in contributors[key]}:
                contributors[key].append(ref)
    return contributors


def _candidate_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        source_type = str(candidate.get("source_type") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _trace_candidate(request: dict[str, Any], candidate: dict[str, Any], score_item: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "request_id": request.get("request_id"),
        "source_type": candidate.get("source_type"),
        "locator": candidate.get("locator"),
        "score": round(float(score_item.get("score") or 0.0), 4),
        "reason": reason,
    }


def _flatten_memory_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    if not value:
        return []
    records = []
    for key in ["memory_catalog", "recent_entries", "entries", "blocked_idea_patterns", "failed_variants", "results", "records"]:
        item = value.get(key)
        if isinstance(item, list):
            for entry in item:
                records.append(entry if isinstance(entry, dict) else {"id": str(entry), "text": str(entry)})
        elif isinstance(item, dict):
            records.extend(entry for entry in item.values() if isinstance(entry, dict))
    if not records:
        records.append(value)
    return records


def _candidate_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ["text", "content", "text_preview", "semantic_summary", "summary", "claim", "title", "section", "path", "symbol"]:
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def _summary(candidate: dict[str, Any]) -> str:
    text = _excerpt(candidate, max_chars=180)
    return text or f"Retrieved {candidate.get('source_type')} evidence from {candidate.get('locator')}"


def _excerpt(candidate: dict[str, Any], *, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(candidate.get("text") or "")).strip()
    return text[:max_chars]


def _why_selected(request: dict[str, Any], score_item: dict[str, Any]) -> str:
    components = score_item.get("components") or {}
    return f"Matched {request.get('source_type')} request {request.get('request_id')} with keyword_overlap={components.get('keyword_overlap', 0)} and query_token_score={components.get('query_token_score', 0)}."


def _bundle_source_type(source_type: str) -> str:
    if source_type in {"failure_memory", "feedback"}:
        return "failure_feedback"
    return source_type


def _normalize_request_source_type(value: Any) -> str:
    source_type = str(value or "paper").lower()
    if source_type in {"artifact", "memory", "negative_memory"}:
        return "failure_memory"
    if source_type not in SOURCE_TYPES:
        return "paper"
    return source_type


def _normalize_candidate_source_type(value: Any) -> str:
    source_type = str(value or "").lower()
    if source_type in {"paper", "rebuttal", "code"}:
        return source_type
    return "failure_memory" if source_type in {"failure_feedback", "memory", "negative_memory"} else source_type


def _default_purpose(source_type: str) -> str:
    if source_type == "code":
        return "implementation_surface"
    if source_type in {"rebuttal", "failure_memory", "feedback"}:
        return "counterevidence"
    return "support"


def _candidate_key(candidate: dict[str, Any]) -> str:
    return "|".join(
        str(candidate.get(key) or "")
        for key in ["source_type", "chunk_id", "source_path", "source_label", "locator"]
    )


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_./:-]+", text.lower()) if len(token) >= 3}


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _compact_for_hash(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= 40000:
        return value
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "truncated_chars": len(text)}


def _short_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
