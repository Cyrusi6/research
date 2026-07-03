"""Shared method-level failure memory across projects."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .failure_log import build_c2c_feedback_bundle
from .utils import ensure_dir, now_utc, repo_root, write_json


DEFAULT_SHARED_METHOD_MEMORY_PATH = ".auto-research/method_failure_memory.jsonl"
DEFAULT_SHARED_METHOD_MEMORY_SUMMARY_PATH = ".auto-research/method_failure_memory.md"
SHARED_METHOD_MEMORY_SCHEMA = "shared_method_failure_memory_v1"


def shared_method_memory_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = _shared_memory_config(config or {})
    return bool(cfg.get("enabled", True))


def shared_method_memory_path(config: dict[str, Any] | None = None) -> Path:
    cfg = _shared_memory_config(config or {})
    value = str(cfg.get("path") or DEFAULT_SHARED_METHOD_MEMORY_PATH)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path


def shared_method_memory_summary_path(config: dict[str, Any] | None = None) -> Path:
    cfg = _shared_memory_config(config or {})
    value = str(cfg.get("summary_path") or DEFAULT_SHARED_METHOD_MEMORY_SUMMARY_PATH)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path


def load_shared_method_memory(
    config: dict[str, Any] | None = None,
    *,
    limit: int | None = None,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not shared_method_memory_enabled(config):
        return _empty_shared_memory(config, disabled=True)
    path = shared_method_memory_path(config)
    entries = _load_jsonl(path)
    entries = _dedupe_entries(entries)
    entries = _score_entries_with_cross_project_context(entries)
    retrieval_context = _normalize_retrieval_context(query_context)
    entries = _score_entries_for_retrieval(entries, retrieval_context)
    entries.sort(key=_memory_retrieval_sort_key, reverse=True)
    quality_summary = _shared_memory_quality_summary(entries)
    max_entries = limit
    if max_entries is None:
        max_entries = int((_shared_memory_config(config or {}).get("prompt_limit") or 40))
    selected = entries[: max(0, int(max_entries))]
    memory_path = _display_path(path)
    catalog = [_memory_catalog_entry_for_prompt(entry, memory_path=memory_path) for entry in selected]
    return {
        "schema_version": SHARED_METHOD_MEMORY_SCHEMA,
        "enabled": True,
        "path": memory_path,
        "entry_count": len(entries),
        "entries": selected,
        "prompt_view": "catalog_with_full_entries_available",
        "ranking_policy": _shared_memory_ranking_policy(),
        "retrieval_context": retrieval_context,
        "retrieval_policy": _shared_memory_retrieval_policy(),
        "quality_summary": quality_summary,
        "retrieved_quality_summary": _shared_memory_quality_summary(selected),
        "full_memory_access": _shared_memory_access_hints(memory_path),
        "memory_catalog": catalog,
    }


def append_shared_c2c_method_failure(
    config: dict[str, Any] | None,
    *,
    project_root: Path,
    performance_feedback: dict[str, Any],
    direction_scorecard: dict[str, Any] | None = None,
    route: str | None = None,
    route_decision: dict[str, Any] | None = None,
    attempt_record: dict[str, Any] | None = None,
    source_paths: list[str] | None = None,
) -> dict[str, Any]:
    if not shared_method_memory_enabled(config):
        return {"status": "disabled"}
    if not _shared_method_memory_project_write_allowed(config or {}, project_root):
        return {"status": "skipped", "reason": "non_workspace_project_not_written_to_global_method_memory"}
    memory_effects = route_decision.get("memory_effects") if isinstance(route_decision, dict) else {}
    if isinstance(memory_effects, dict) and route_decision:
        if memory_effects.get("write_shared_method_memory") is not True:
            return {
                "status": "skipped",
                "reason": memory_effects.get("skip_reason") or "route_decision_skipped_method_memory",
                "route_decision": route_decision.get("decision"),
            }
    elif _is_implementation_failure(performance_feedback):
        return {"status": "skipped", "reason": "implementation_failure_is_not_method_memory"}
    proxy_calibration = _load_proxy_calibration_signal(project_root)

    method_entries = build_c2c_feedback_bundle(
        [
            {
                "kind": "c2c_performance_feedback",
                "source_path": "plan/performance_feedback.json",
                **performance_feedback,
            },
            *(
                [
                    {
                        "kind": "c2c_direction_scorecard",
                        "source_path": "plan/direction_scorecard.json",
                        **direction_scorecard,
                    }
                ]
                if isinstance(direction_scorecard, dict) and direction_scorecard
                else []
            ),
            *(
                [
                    {
                        "kind": "c2c_proxy_calibration",
                        "source_path": "experiment/results/proxy_calibration.json",
                        "proxy_calibration": proxy_calibration,
                    }
                ]
                if proxy_calibration
                else []
            ),
            *(
                [
                    {
                        "kind": "c2c_route_decision",
                        "source_path": "meta/route_decision.json",
                        "route_decision": _compact_route_decision(route_decision),
                    }
                ]
                if isinstance(route_decision, dict) and route_decision
                else []
            ),
            *(
                [
                    {
                        "kind": "c2c_attempt_record",
                        "source_path": "meta/attempt_ledger.json",
                        "attempt_record": _compact_attempt_record(attempt_record),
                    }
                ]
                if isinstance(attempt_record, dict) and attempt_record
                else []
            ),
        ],
        project_id=project_root.name,
        iteration=(performance_feedback.get("summary") or {}).get("iteration"),
        sources=source_paths
        or [
            "plan/performance_feedback.json",
            "plan/direction_scorecard.json",
            "experiment/results/proxy_calibration.json",
            "meta/route_decision.json",
            "meta/attempt_ledger.json",
        ],
        view="method",
    )
    if not method_entries.get("entries"):
        return {"status": "skipped", "reason": "no_method_level_evidence"}

    entry = {
        "schema_version": SHARED_METHOD_MEMORY_SCHEMA,
        "timestamp": now_utc(),
        "project_id": project_root.name,
        "topic": _project_topic(project_root),
        "kind": "shared_c2c_method_failure",
        "route": route or (performance_feedback.get("summary") or {}).get("route"),
        "failure_class": (route_decision or {}).get("failure_class") or "method_failure",
        "source_project_paths": source_paths
        or [
            "plan/performance_feedback.json",
            "plan/direction_scorecard.json",
            "experiment/results/main_results.json",
            "experiment/results/failure_feedback.json",
            "meta/route_decision.json",
            "meta/attempt_ledger.json",
        ],
        "summary": method_entries.get("summary") or {},
        "entries": method_entries.get("entries") or [],
        "proxy_calibration": proxy_calibration,
        "route_decision": _compact_route_decision(route_decision),
        "attempt_record": _compact_attempt_record(attempt_record),
        "source_repo_fingerprint": _source_repo_fingerprint(config, project_root),
        "direction_scorecard": (direction_scorecard or {}).get("current_direction")
        if isinstance(direction_scorecard, dict)
        else {},
    }
    entry = _compact_shared_method_memory_entry(entry)
    if not _entry_has_persistable_method_signal(entry):
        return {"status": "skipped", "reason": "no_persistable_method_outcome"}
    entry["memory_quality"] = _memory_quality(entry)
    entry["memory_id"] = _memory_id(entry)

    path = shared_method_memory_path(config)
    existing = _dedupe_entries([*_load_jsonl(path), entry])
    existing = _score_entries_with_cross_project_context(existing)
    final_entry = next((item for item in existing if item.get("memory_id") == entry["memory_id"]), entry)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in existing:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    _write_shared_method_memory_markdown(shared_method_memory_summary_path(config), existing)
    return {
        "status": "appended",
        "path": _display_path(path),
        "memory_id": entry["memory_id"],
        "memory_priority": (final_entry.get("memory_quality") or entry["memory_quality"]).get("priority"),
        "entry_count": len(existing),
    }


def write_project_shared_method_memory_snapshot(project_root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    memory = load_shared_method_memory(config, query_context=shared_method_memory_query_context(config, project_root=project_root))
    target = project_root / "intake" / "shared_method_failure_memory.json"
    write_json(target, memory)
    return {
        "status": "ok" if memory.get("enabled") else "disabled",
        "path": "intake/shared_method_failure_memory.json",
        "entry_count": memory.get("entry_count", 0),
    }


def shared_method_memory_for_prompt(
    config: dict[str, Any] | None = None,
    *,
    limit: int | None = None,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = load_shared_method_memory(config, limit=limit, query_context=query_context)
    selected = [entry for entry in memory.get("entries", []) if isinstance(entry, dict)]
    catalog = [
        _memory_catalog_entry_for_prompt(entry, memory_path=str(memory.get("path") or ""))
        for entry in selected[: max(0, int(limit or 12))]
    ]
    return {
        "schema_version": memory.get("schema_version"),
        "enabled": memory.get("enabled"),
        "path": memory.get("path"),
        "entry_count": memory.get("entry_count", 0),
        "prompt_view": "catalog_only",
        "ranking_policy": memory.get("ranking_policy") or _shared_memory_ranking_policy(),
        "retrieval_policy": memory.get("retrieval_policy") or _shared_memory_retrieval_policy(),
        "retrieval_context": memory.get("retrieval_context") or {},
        "quality_summary": memory.get("quality_summary") or {},
        "retrieved_quality_summary": _shared_memory_quality_summary(selected),
        "high_quality_memory_ids": _high_quality_memory_ids(selected, limit=8),
        "full_memory_access": _shared_memory_access_hints(str(memory.get("path") or "")),
        "memory_catalog": catalog,
        "recent_entries": catalog,
    }


def shared_method_memory_query_context(
    config: dict[str, Any] | None = None,
    *,
    project_root: Path | None = None,
    topic: str | None = None,
    selected_direction: dict[str, Any] | None = None,
    feedback: Any = None,
    negative_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    selected_direction = selected_direction if isinstance(selected_direction, dict) else {}
    baseline = ((cfg.get("c2c") or {}).get("baseline") or {}) if isinstance(cfg.get("c2c"), dict) else {}
    datasets = _query_datasets_from_sources([selected_direction, feedback, negative_memory, baseline])
    if not datasets and isinstance(baseline.get("datasets"), dict):
        datasets = sorted(str(key) for key in baseline["datasets"].keys())
    mechanisms = _query_mechanisms_from_sources([selected_direction, feedback, negative_memory])
    failure_modes = _query_failure_modes_from_sources([selected_direction, feedback, negative_memory])
    if not topic and project_root:
        topic = _project_topic(project_root)
    return _strip_empty(
        {
            "topic": topic or (cfg.get("research_topic") if isinstance(cfg, dict) else None),
            "datasets": datasets,
            "mechanism_types": mechanisms,
            "failure_modes": failure_modes,
            "source_repo_fingerprint": _source_repo_fingerprint(cfg, project_root),
        }
    )


def shared_method_memory_ids(memory: dict[str, Any] | None) -> list[str]:
    if not isinstance(memory, dict):
        return []
    ids: list[str] = []
    for key in ["entries", "recent_entries"]:
        for entry in memory.get(key) or []:
            if isinstance(entry, dict) and entry.get("memory_id"):
                ids.append(str(entry["memory_id"]))
    return sorted(set(ids))


def collect_used_shared_memory_refs(payload: Any, memory: dict[str, Any] | None = None) -> list[str]:
    explicit = _explicit_used_shared_memory_refs(payload)
    known_ids = set(shared_method_memory_ids(memory))
    if memory is not None:
        explicit = [item for item in explicit if item in known_ids]
        if explicit:
            return sorted(set(explicit))
        if not known_ids:
            return []
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return sorted(memory_id for memory_id in known_ids if memory_id in text)
    return sorted(set(explicit))


def _shared_memory_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = ((config.get("orchestration") or {}).get("shared_method_memory") or {})
    return cfg if isinstance(cfg, dict) else {}


def _shared_method_memory_project_write_allowed(config: dict[str, Any], project_root: Path) -> bool:
    cfg = _shared_memory_config(config)
    if cfg.get("allow_non_workspace_projects") is True:
        return True
    if _shared_memory_config_uses_explicit_path(config):
        return True
    try:
        resolved = Path(project_root).resolve()
        workspace_root = (repo_root() / "workspace").resolve()
        resolved.relative_to(workspace_root)
        return True
    except (OSError, ValueError):
        return False


def _shared_memory_config_uses_explicit_path(config: dict[str, Any]) -> bool:
    cfg = _shared_memory_config(config)
    path = cfg.get("path")
    summary_path = cfg.get("summary_path")
    if not path and not summary_path:
        return False
    default_path = str(Path(DEFAULT_SHARED_METHOD_MEMORY_PATH))
    default_summary = str(Path(DEFAULT_SHARED_METHOD_MEMORY_SUMMARY_PATH))
    return str(path or default_path) != default_path or str(summary_path or default_summary) != default_summary


def _empty_shared_memory(config: dict[str, Any] | None = None, *, disabled: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SHARED_METHOD_MEMORY_SCHEMA,
        "enabled": not disabled,
        "path": _display_path(shared_method_memory_path(config)),
        "entry_count": 0,
        "entries": [],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        compacted = _compact_shared_method_memory_entry(entry)
        if _entry_is_implementation_noise(compacted) or not _entry_has_persistable_method_signal(compacted):
            continue
        memory_id = str(compacted.get("memory_id") or _memory_id(compacted))
        copied = dict(compacted)
        copied["memory_id"] = memory_id
        copied["memory_quality"] = _memory_quality(copied)
        by_id[memory_id] = copied
    return sorted(by_id.values(), key=lambda item: str(item.get("timestamp") or ""))


def _compact_shared_method_memory_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    existing_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    direction_summary = direction.get("summary") if isinstance(direction.get("summary"), dict) else {}
    proxy_calibration = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    compact_entries = [
        item
        for item in (_compact_method_memory_evidence_item(item) for item in entry.get("entries") or [])
        if item
    ]
    method_context = _strip_empty(
        {
            "direction_id": direction.get("direction_id") or existing_context.get("direction_id"),
            "title": direction.get("title") or existing_context.get("title"),
            "mechanism_type": direction.get("mechanism_type") or existing_context.get("mechanism_type"),
            "attempt_count": direction_summary.get("attempt_count") or existing_context.get("attempt_count"),
            "same_direction_failure_count": direction_summary.get("same_direction_failure_count")
            or existing_context.get("same_direction_failure_count"),
            "same_direction_failure_budget": direction_summary.get("same_direction_failure_budget")
            or existing_context.get("same_direction_failure_budget"),
            "best_proxy_delta": direction_summary.get("best_proxy_delta")
            if direction_summary.get("best_proxy_delta") is not None
            else existing_context.get("best_proxy_delta"),
            "direction_quality": direction_summary.get("direction_quality") or existing_context.get("direction_quality"),
            "recommendation": (direction.get("s1_feedback") or {}).get("recommendation")
            if isinstance(direction.get("s1_feedback"), dict)
            else existing_context.get("recommendation"),
        }
    )
    compact_summary = _strip_empty(
        {
            "summary_text": summary.get("summary_text"),
            "failed_idea_ids": summary.get("failed_idea_ids") or [],
            "failed_titles": summary.get("failed_titles") or [],
            "failure_modes": summary.get("failure_modes") or [],
            "dataset_regressions": summary.get("dataset_regressions") or {},
            "dragging_datasets": summary.get("dragging_datasets") or [],
            "sample_type_failures": summary.get("sample_type_failures") or [],
            "mixed_gain_patterns": summary.get("mixed_gain_patterns") or [],
            "avoid_repeat_rules": summary.get("avoid_repeat_rules") or summary.get("blocked_idea_patterns") or [],
        }
    )
    compact = {
        "schema_version": SHARED_METHOD_MEMORY_SCHEMA,
        "timestamp": entry.get("timestamp"),
        "project_id": entry.get("project_id"),
        "topic": entry.get("topic"),
        "kind": entry.get("kind") or "shared_c2c_method_failure",
        "route": entry.get("route"),
        "failure_class": "method_failure",
        "source_project_paths": _method_memory_source_paths(entry.get("source_project_paths") or []),
        "summary": compact_summary,
        "method_context": method_context,
        "entries": compact_entries,
        "proxy_calibration": _compact_persisted_proxy_calibration(proxy_calibration),
        "route_decision": _compact_route_decision(entry.get("route_decision")),
        "attempt_record": _compact_attempt_record(entry.get("attempt_record")),
        "source_repo_fingerprint": entry.get("source_repo_fingerprint") if isinstance(entry.get("source_repo_fingerprint"), dict) else {},
    }
    if entry.get("memory_id"):
        compact["memory_id"] = entry.get("memory_id")
    if entry.get("memory_quality"):
        compact["memory_quality"] = entry.get("memory_quality")
    if entry.get("memory_retrieval"):
        compact["memory_retrieval"] = entry.get("memory_retrieval")
    return _strip_empty(compact)


def _method_memory_source_paths(paths: Any) -> list[str]:
    allowed_markers = [
        "plan/performance_feedback.json",
        "experiment/results/main_results.json",
        "experiment/results/proxy_calibration.json",
        "meta/route_decision.json",
        "meta/attempt_ledger.json",
    ]
    blocked_markers = [
        "direction_scorecard",
        "failure_feedback",
        "negative_memory",
        "s2_planner_memory",
        "patch_manifest",
    ]
    result: list[str] = []
    for path in paths if isinstance(paths, list) else []:
        text = str(path)
        lowered = text.lower()
        if any(marker in lowered for marker in blocked_markers):
            continue
        if allowed_markers and not any(marker in text for marker in allowed_markers):
            continue
        if text not in result:
            result.append(text)
    return result


def _compact_route_decision(route_decision: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(route_decision, dict) or not route_decision:
        return {}
    return _strip_empty(
        {
            "decision": route_decision.get("decision"),
            "next_stage": route_decision.get("next_stage"),
            "failure_class": route_decision.get("failure_class"),
            "reason_codes": route_decision.get("reason_codes") or [],
            "budget_effects": route_decision.get("budget_effects")
            if isinstance(route_decision.get("budget_effects"), dict)
            else {},
            "memory_effects": route_decision.get("memory_effects")
            if isinstance(route_decision.get("memory_effects"), dict)
            else {},
        }
    )


def _compact_attempt_record(attempt_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(attempt_record, dict) or not attempt_record:
        return {}
    return _strip_empty(
        {
            "iteration": attempt_record.get("iteration"),
            "direction_id": attempt_record.get("direction_id"),
            "variant_id": attempt_record.get("variant_id"),
            "patch_id": attempt_record.get("patch_id"),
            "stage": attempt_record.get("stage"),
            "failure_class": attempt_record.get("failure_class"),
            "route_decision": attempt_record.get("route_decision"),
            "consumes_same_direction_attempt": attempt_record.get("consumes_same_direction_attempt"),
            "consumes_patch_repair_attempt": attempt_record.get("consumes_patch_repair_attempt"),
            "consumes_resource_retry": attempt_record.get("consumes_resource_retry"),
            "writes_method_memory": attempt_record.get("writes_method_memory"),
        }
    )


def _compact_method_memory_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if item.get("kind") == "c2c_route_decision":
        compact = _compact_route_decision(item.get("route_decision"))
        return {"kind": "c2c_route_decision", "source_path": item.get("source_path"), "route_decision": compact} if compact else {}
    if item.get("kind") == "c2c_attempt_record":
        compact = _compact_attempt_record(item.get("attempt_record"))
        return {"kind": "c2c_attempt_record", "source_path": item.get("source_path"), "attempt_record": compact} if compact else {}
    if item.get("kind") == "c2c_direction_scorecard":
        return {}
    if item.get("kind") == "c2c_proxy_calibration":
        proxy_calibration = item.get("proxy_calibration") if isinstance(item.get("proxy_calibration"), dict) else {}
        compact = _compact_persisted_proxy_calibration(proxy_calibration)
        return {"kind": "c2c_proxy_calibration", "proxy_calibration": compact} if compact else {}
    attribution = item.get("failure_attribution") if isinstance(item.get("failure_attribution"), dict) else {}
    compact: dict[str, Any] = {
        "kind": item.get("kind"),
        "source_path": item.get("source_path"),
        "idea_id": item.get("idea_id") or item.get("id"),
        "title": item.get("title"),
        "decision": item.get("decision"),
        "mechanism_type": item.get("mechanism_type"),
        "route": item.get("route"),
        "failure_mode": item.get("failure_mode"),
        "reason": item.get("reason") if _method_memory_reason_is_allowed(item.get("reason")) else None,
        "metrics": item.get("metrics") if isinstance(item.get("metrics"), dict) else {},
        "dataset_regressions": item.get("dataset_regressions") if isinstance(item.get("dataset_regressions"), dict) else {},
        "dragging_datasets": item.get("dragging_datasets") or attribution.get("dragging_datasets") or [],
        "sample_type_failures": item.get("sample_type_failures") or attribution.get("sample_type_failures") or [],
        "mixed_gain_patterns": item.get("mixed_gain_patterns") or attribution.get("mixed_gain_patterns") or [],
        "ablation_evidence": item.get("ablation_evidence")
        if isinstance(item.get("ablation_evidence"), dict)
        else attribution.get("ablation_evidence") if isinstance(attribution.get("ablation_evidence"), dict) else {},
        "avoid_repeat_rule": item.get("avoid_repeat_rule"),
    }
    proxy = item.get("proxy_screen") if isinstance(item.get("proxy_screen"), dict) else {}
    if proxy:
        compact["proxy_screen"] = _strip_empty(
            {
                "status": proxy.get("status"),
                "metrics": proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {},
                "baseline_metrics": proxy.get("baseline_metrics") if isinstance(proxy.get("baseline_metrics"), dict) else {},
                "proxy_delta_vs_baseline": proxy.get("proxy_delta_vs_baseline"),
                "proxy_dataset_deltas": proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {},
                "proxy_dataset_regressions": proxy.get("proxy_dataset_regressions") if isinstance(proxy.get("proxy_dataset_regressions"), dict) else {},
                "proxy_worst_dataset_regression": proxy.get("proxy_worst_dataset_regression"),
                "proxy_score": proxy.get("proxy_score"),
                "soft_fail": proxy.get("soft_fail"),
                "soft_flags": proxy.get("soft_flags") or [],
            }
        )
    candidate_results = [
        child
        for child in (_compact_method_memory_evidence_item(child) for child in item.get("candidate_results") or [])
        if child
    ]
    if candidate_results:
        compact["candidate_results"] = candidate_results
    return _strip_empty(compact)


def _compact_persisted_proxy_calibration(proxy_calibration: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(proxy_calibration, dict) or not proxy_calibration:
        return {}
    if not (
        _proxy_calibration_false_positive_count(proxy_calibration)
        or _proxy_calibration_mispredicted_datasets(proxy_calibration)
        or _proxy_calibration_risky_mechanisms(proxy_calibration)
    ):
        return {}
    return _compact_proxy_calibration_for_prompt(proxy_calibration)


def _method_memory_reason_is_allowed(value: Any) -> bool:
    text = str(value or "").lower()
    blocked = [
        "runtimeerror",
        "traceback",
        "dtype",
        "py_compile",
        "no candidate metrics",
        "checkpoint",
        "evaluator",
        "gpu",
        "oom",
        "resource",
        "quota",
        "rate limit",
    ]
    return bool(text.strip()) and not any(marker in text for marker in blocked)


def _entry_has_persistable_method_signal(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    proxy_calibration = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    if _proxy_calibration_false_positive_count(proxy_calibration):
        return True
    if _entry_has_full_train_result(entry):
        return True
    for item in entry.get("entries") or []:
        if not isinstance(item, dict):
            continue
        if item.get("metrics") or item.get("dataset_regressions") or item.get("dragging_datasets"):
            return True
        proxy = item.get("proxy_screen") if isinstance(item.get("proxy_screen"), dict) else {}
        if proxy.get("metrics") or proxy.get("proxy_dataset_deltas") or proxy.get("proxy_delta_vs_baseline") is not None:
            return True
        for child in item.get("candidate_results") or []:
            if isinstance(child, dict) and _entry_has_persistable_method_signal({"entries": [child]}):
                return True
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    if summary.get("dataset_regressions") or summary.get("dragging_datasets"):
        return True
    return False


def _score_entries_with_cross_project_context(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus_context = _memory_corpus_context(entries)
    scored: list[dict[str, Any]] = []
    for entry in entries:
        copied = dict(entry)
        copied["memory_quality"] = _memory_quality(copied, corpus_context=corpus_context)
        scored.append(copied)
    return scored


def _shared_memory_ranking_policy() -> dict[str, Any]:
    return {
        "sort": "descending memory_quality.priority",
        "high_quality_signals": [
            "proxy_full_false_positive",
            "full_train_failure",
            "proxy_dataset_misprediction",
            "cross_project_mechanism_failure",
            "ablation_evidence",
            "dataset_regression",
            "repeated_failure",
        ],
        "use_guidance": [
            "Prefer memories with full_train_failure over ordinary cheap_proxy_failure.",
            "Treat proxy_full_false_positive as the strongest warning: cheap proxy said good but full train failed.",
            "Use mispredicted_datasets to avoid overtrusting proxy on those datasets.",
            "Use risky_mechanisms and cross_project_mechanism_failure to avoid mechanisms that repeatedly look good in proxy but fail later.",
            "Copy every influential memory_id into used_shared_memory_refs for auditability.",
        ],
    }


def _shared_memory_quality_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    signal_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    high_quality_ids: list[str] = []
    false_positive_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
        signals = [str(signal) for signal in quality.get("signals") or [] if signal]
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        evidence = quality.get("evidence") if isinstance(quality.get("evidence"), dict) else {}
        for dataset in evidence.get("mispredicted_datasets") or []:
            dataset_counts[str(dataset)] = dataset_counts.get(str(dataset), 0) + 1
        for mechanism in evidence.get("risky_mechanisms") or []:
            mechanism_counts[str(mechanism)] = mechanism_counts.get(str(mechanism), 0) + 1
        for mechanism in evidence.get("cross_project_mechanisms") or []:
            if isinstance(mechanism, dict) and mechanism.get("mechanism_type"):
                key = str(mechanism["mechanism_type"])
                mechanism_counts[key] = mechanism_counts.get(key, 0) + int(mechanism.get("project_count") or 1)
        memory_id = entry.get("memory_id")
        if memory_id and _is_high_quality_memory(quality):
            high_quality_ids.append(str(memory_id))
        if memory_id and "proxy_full_false_positive" in signals:
            false_positive_ids.append(str(memory_id))
    return _strip_empty(
        {
            "entry_count": len(entries),
            "high_quality_memory_ids": high_quality_ids[:12],
            "proxy_full_false_positive_memory_ids": false_positive_ids[:12],
            "signal_counts": dict(sorted(signal_counts.items())),
            "top_mispredicted_datasets": _rank_count_map(dataset_counts, limit=8),
            "top_risky_mechanisms": _rank_count_map(mechanism_counts, limit=8),
        }
    )


def _high_quality_memory_ids(entries: list[dict[str, Any]], *, limit: int) -> list[str]:
    ids: list[str] = []
    for entry in entries:
        quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
        if entry.get("memory_id") and _is_high_quality_memory(quality):
            ids.append(str(entry["memory_id"]))
    return ids[: max(0, int(limit))]


def _is_high_quality_memory(quality: dict[str, Any]) -> bool:
    signals = set(str(signal) for signal in quality.get("signals") or [])
    high_quality_signals = {
        "proxy_full_false_positive",
        "full_train_failure",
        "proxy_dataset_misprediction",
        "cross_project_mechanism_failure",
        "ablation_evidence",
        "dataset_regression",
        "repeated_failure",
    }
    if signals & high_quality_signals:
        return True
    try:
        return float(quality.get("priority") or 0.0) >= 5.0
    except (TypeError, ValueError):
        return False


def _rank_count_map(counts: dict[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"id": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[: max(0, int(limit))]
    ]


def _expand_shared_entries_for_feedback_bundle(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        if summary:
            expanded.append(
                {
                    "kind": "shared_method_memory_summary",
                    "timestamp": entry.get("timestamp"),
                    "project_id": entry.get("project_id"),
                    "route": entry.get("route"),
                    **summary,
                }
            )
        expanded.extend(item for item in entry.get("entries") or [] if isinstance(item, dict))
        direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
        if direction:
            expanded.append({"kind": "c2c_direction_scorecard", "current_direction": direction})
        proxy_calibration = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
        if proxy_calibration:
            expanded.append(
                {
                    "kind": "c2c_proxy_calibration",
                    "proxy_calibration": proxy_calibration,
                    "source_path": "experiment/results/proxy_calibration.json",
                }
            )
    return expanded or entries


def _memory_id(entry: dict[str, Any]) -> str:
    payload = {
        "project_id": entry.get("project_id"),
        "route": entry.get("route"),
        "summary": entry.get("summary") or {},
        "entries": entry.get("entries") or [],
        "method_context": entry.get("method_context") or {},
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _memory_priority_sort_key(entry: dict[str, Any]) -> tuple[float, str]:
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
    return (float(quality.get("priority") or 0.0), str(entry.get("timestamp") or ""))


def _memory_retrieval_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float, str]:
    retrieval = entry.get("memory_retrieval") if isinstance(entry.get("memory_retrieval"), dict) else {}
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
    return (
        1.0 if retrieval.get("has_relevance_match") else 0.0,
        float(retrieval.get("combined_score") or quality.get("priority") or 0.0),
        float(retrieval.get("relevance_score") or 0.0),
        float(quality.get("priority") or 0.0),
        str(entry.get("timestamp") or ""),
    )


def _score_entries_for_retrieval(entries: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for entry in entries:
        copied = dict(entry)
        quality = copied.get("memory_quality") if isinstance(copied.get("memory_quality"), dict) else _memory_quality(copied)
        relevance = _entry_relevance(copied, context)
        quality_priority = float(quality.get("priority") or 0.0)
        copied["memory_retrieval"] = {
            "quality_priority": round(quality_priority, 3),
            "relevance_score": round(float(relevance.get("score") or 0.0), 3),
            "combined_score": round(quality_priority + float(relevance.get("score") or 0.0), 3),
            "has_relevance_match": bool(relevance.get("matched_fields")),
            "matched_fields": relevance.get("matched_fields") or [],
            "matched_values": relevance.get("matched_values") or {},
        }
        scored.append(copied)
    return scored


def _shared_memory_retrieval_policy() -> dict[str, Any]:
    return {
        "mode": "quality_weighted_top_k_retrieval",
        "sort": "descending memory_retrieval.combined_score",
        "relevance_fields": ["topic", "datasets", "mechanism_types", "failure_modes", "source_repo_fingerprint"],
        "score": "memory_quality.priority + deterministic relevance bonuses",
        "notes": [
            "source_repo_fingerprint exact matches receive the largest relevance bonus.",
            "dataset and mechanism_type matches are preferred over unrelated high-priority memories.",
            "If no context is available, retrieval falls back to quality priority ordering.",
        ],
    }


def _normalize_retrieval_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    normalized = {
        "topic": str(context.get("topic") or "").strip(),
        "topic_tokens": sorted(_tokenize_text(context.get("topic") or "")),
        "datasets": sorted({str(item) for item in context.get("datasets") or [] if item}),
        "mechanism_types": sorted({str(item) for item in context.get("mechanism_types") or [] if item}),
        "failure_modes": sorted({str(item) for item in context.get("failure_modes") or [] if item}),
        "source_repo_fingerprint": context.get("source_repo_fingerprint") if isinstance(context.get("source_repo_fingerprint"), dict) else {},
    }
    return _strip_empty(normalized)


def _entry_relevance(entry: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if not context:
        return {"score": 0.0, "matched_fields": [], "matched_values": {}}
    score = 0.0
    matched: dict[str, list[str]] = {}
    entry_text_tokens = _entry_text_tokens(entry)
    topic_tokens = set(context.get("topic_tokens") or [])
    topic_overlap = sorted(topic_tokens & entry_text_tokens)
    if topic_overlap:
        score += min(2.0, 0.4 * len(topic_overlap))
        matched["topic"] = topic_overlap[:8]
    dataset_overlap = sorted(set(context.get("datasets") or []) & _entry_datasets(entry))
    if dataset_overlap:
        score += min(6.0, 3.0 + len(dataset_overlap))
        matched["datasets"] = dataset_overlap[:8]
    mechanism_overlap = sorted(set(context.get("mechanism_types") or []) & _entry_mechanism_types(entry))
    if mechanism_overlap:
        score += min(7.0, 4.0 + len(mechanism_overlap))
        matched["mechanism_types"] = mechanism_overlap[:8]
    failure_overlap = sorted(set(context.get("failure_modes") or []) & _entry_failure_modes(entry))
    if failure_overlap:
        score += min(4.0, 2.0 + 0.5 * len(failure_overlap))
        matched["failure_modes"] = failure_overlap[:8]
    if _source_repo_fingerprint_matches(context.get("source_repo_fingerprint"), entry.get("source_repo_fingerprint")):
        score += 8.0
        matched["source_repo_fingerprint"] = ["match"]
    return {
        "score": score,
        "matched_fields": sorted(matched),
        "matched_values": matched,
    }


def _memory_quality(entry: dict[str, Any], *, corpus_context: dict[str, Any] | None = None) -> dict[str, Any]:
    signals: list[str] = []
    components: dict[str, float] = {"base": 1.0}
    proxy_calibration = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    direction_summary = direction.get("summary") if isinstance(direction.get("summary"), dict) else {}
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    false_positive_count = _proxy_calibration_false_positive_count(proxy_calibration)
    mispredicted_datasets = _proxy_calibration_mispredicted_datasets(proxy_calibration)
    risky_mechanisms = _proxy_calibration_risky_mechanisms(proxy_calibration)
    failure_scope = _entry_failure_scope(entry, proxy_calibration=proxy_calibration)
    repeated_failure_count = _entry_repeated_failure_count(entry)
    has_dataset_regression = _entry_has_dataset_regression(entry, proxy_calibration=proxy_calibration)
    has_mean_delta = _entry_has_mean_delta(entry)
    has_ablation_evidence = _entry_has_ablation_evidence(entry)
    mechanisms = sorted(_entry_mechanism_types(entry))
    cross_project_mechanisms = _entry_cross_project_mechanisms(entry, corpus_context)

    if failure_scope == "full_train":
        components["full_train_failure"] = 4.0
        signals.append("full_train_failure")
    elif failure_scope == "cheap_proxy":
        components["cheap_proxy_failure"] = 1.5
        signals.append("cheap_proxy_failure")
    if false_positive_count:
        components["proxy_full_false_positive"] = 5.0
        signals.append("proxy_full_false_positive")
    if mispredicted_datasets:
        components["proxy_dataset_misprediction"] = 2.0
        signals.append("proxy_dataset_misprediction")
    if risky_mechanisms:
        components["proxy_risky_mechanism"] = 1.5
        signals.append("proxy_risky_mechanism")
    if has_dataset_regression:
        components["dataset_regression"] = 2.0
        signals.append("dataset_regression")
    elif has_mean_delta:
        components["mean_delta_only"] = 0.75
        signals.append("mean_delta_only")
    if repeated_failure_count >= 2:
        components["repeated_failure"] = round(min(3.0, 0.75 * (repeated_failure_count - 1)), 3)
        signals.append("repeated_failure")
    if has_ablation_evidence:
        components["ablation_evidence"] = 2.0
        signals.append("ablation_evidence")
    if cross_project_mechanisms:
        components["cross_project_mechanism_failure"] = 2.5
        signals.append("cross_project_mechanism_failure")
    direction_quality = direction_summary.get("direction_quality") or method_context.get("direction_quality")
    if (
        direction_summary.get("same_direction_failure_count")
        or method_context.get("same_direction_failure_count")
        or direction_quality == "poor_direction_evidence"
    ):
        components["direction_budget_evidence"] = 1.5
        signals.append("direction_budget_evidence")
    if summary.get("all_datasets_collapsed") or direction_summary.get("all_dataset_collapse_attempts") or method_context.get("all_dataset_collapse_attempts"):
        components["all_dataset_collapse"] = 1.0
        signals.append("all_dataset_collapse")
    priority = sum(float(value) for value in components.values())
    return {
        "priority": round(priority, 3),
        "signals": sorted(set(signals)),
        "score_components": {key: round(float(value), 3) for key, value in sorted(components.items())},
        "evidence": _strip_empty(
            {
                "failure_scope": failure_scope,
                "full_train_failure": failure_scope == "full_train",
                "cheap_proxy_failure": failure_scope == "cheap_proxy",
                "proxy_false_positive_count": false_positive_count,
                "mispredicted_datasets": mispredicted_datasets,
                "risky_mechanisms": risky_mechanisms,
                "repeated_failure_count": repeated_failure_count,
                "has_ablation_evidence": has_ablation_evidence,
                "has_dataset_regression": has_dataset_regression,
                "has_mean_delta": has_mean_delta,
                "mechanisms": mechanisms,
                "cross_project_mechanisms": cross_project_mechanisms,
            }
        ),
    }


def _memory_corpus_context(entries: list[dict[str, Any]]) -> dict[str, Any]:
    mechanism_projects: dict[str, set[str]] = {}
    mechanism_memory_ids: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        project_id = str(entry.get("project_id") or "")
        memory_id = str(entry.get("memory_id") or _memory_id(entry))
        for mechanism in _entry_mechanism_types(entry):
            if not mechanism or mechanism == "unknown":
                continue
            mechanism_projects.setdefault(mechanism, set()).add(project_id or memory_id)
            mechanism_memory_ids.setdefault(mechanism, set()).add(memory_id)
    return {
        "mechanism_project_counts": {key: len(value) for key, value in mechanism_projects.items()},
        "mechanism_memory_counts": {key: len(value) for key, value in mechanism_memory_ids.items()},
    }


def _entry_cross_project_mechanisms(entry: dict[str, Any], corpus_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(corpus_context, dict):
        return []
    project_counts = corpus_context.get("mechanism_project_counts") if isinstance(corpus_context.get("mechanism_project_counts"), dict) else {}
    memory_counts = corpus_context.get("mechanism_memory_counts") if isinstance(corpus_context.get("mechanism_memory_counts"), dict) else {}
    result: list[dict[str, Any]] = []
    for mechanism in sorted(_entry_mechanism_types(entry)):
        project_count = _int_or_none(project_counts.get(mechanism)) or 0
        if project_count < 2:
            continue
        result.append(
            {
                "mechanism_type": mechanism,
                "project_count": project_count,
                "memory_count": _int_or_none(memory_counts.get(mechanism)) or project_count,
            }
        )
    return result


def _entry_failure_scope(entry: dict[str, Any], *, proxy_calibration: dict[str, Any]) -> str:
    route_values = {
        str(value).lower()
        for value in [
            entry.get("route"),
            (entry.get("summary") or {}).get("route") if isinstance(entry.get("summary"), dict) else None,
        ]
        if value not in (None, "")
    }
    if any("full" in route or "s3_failure" in route for route in route_values):
        return "full_train"
    if _proxy_calibration_false_positive_count(proxy_calibration) or _entry_has_full_train_result(entry):
        return "full_train"
    if any("proxy" in route for route in route_values) or _entry_has_proxy_failure(entry):
        return "cheap_proxy"
    return "unknown"


def _entry_has_full_train_result(entry: dict[str, Any]) -> bool:
    for node in _walk_dicts(entry):
        metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
        if metrics and any(key in metrics for key in ["mean", "datasets", "dataset_scores", "accuracy"]):
            return True
        if _numeric_key_present(node, ["full_mean_delta", "full_delta_vs_baseline", "full_score", "full_mean"]):
            return True
        if isinstance(node.get("full_results"), dict) or isinstance(node.get("full_train_metrics"), dict):
            return True
    return False


def _entry_has_proxy_failure(entry: dict[str, Any]) -> bool:
    failure_statuses = {
        "rejected",
        "proxy_rejected",
        "repairable_proxy_risk",
        "proxy_repairable",
        "not_viable",
    }
    for node in _walk_dicts(entry):
        decision = str(node.get("decision") or "").lower()
        if decision in failure_statuses or decision.startswith("proxy_"):
            return True
        proxy_screen = node.get("proxy_screen") if isinstance(node.get("proxy_screen"), dict) else {}
        status = str(proxy_screen.get("status") or "").lower()
        if status in failure_statuses:
            return True
    return False


def _entry_repeated_failure_count(entry: dict[str, Any]) -> int:
    counts: list[int] = []
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    direction_summary = direction.get("summary") if isinstance(direction.get("summary"), dict) else {}
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    for source in [summary, direction, direction_summary, method_context]:
        for key in ["same_direction_failure_count", "attempt_count", "failure_count", "failed_attempt_count"]:
            value = _int_or_none(source.get(key))
            if value is not None:
                counts.append(value)
    attempts = direction.get("attempts") if isinstance(direction.get("attempts"), list) else []
    if attempts:
        counts.append(len(attempts))
    return max(counts) if counts else 1


def _entry_has_dataset_regression(entry: dict[str, Any], *, proxy_calibration: dict[str, Any]) -> bool:
    if _proxy_calibration_mispredicted_datasets(proxy_calibration):
        return True
    for node in _walk_dicts(entry):
        if _non_empty(node.get("dataset_regressions")) or _non_empty(node.get("dragging_datasets")):
            return True
        attribution = node.get("failure_attribution") if isinstance(node.get("failure_attribution"), dict) else {}
        if _non_empty(attribution.get("dragging_datasets")) or _non_empty(attribution.get("mixed_gain_patterns")):
            return True
        proxy_screen = node.get("proxy_screen") if isinstance(node.get("proxy_screen"), dict) else {}
        if _numeric_map_has_negative(proxy_screen.get("proxy_dataset_deltas")):
            return True
        if _non_empty(proxy_screen.get("proxy_dataset_regressions")):
            return True
    return False


def _entry_has_mean_delta(entry: dict[str, Any]) -> bool:
    delta_keys = [
        "best_proxy_delta",
        "delta_vs_baseline",
        "full_delta_vs_baseline",
        "full_mean_delta",
        "mean_delta",
        "proxy_delta_vs_baseline",
        "proxy_mean_delta",
        "proxy_mean_delta_max",
        "proxy_mean_delta_min",
    ]
    for node in _walk_dicts(entry):
        if _numeric_key_present(node, delta_keys):
            return True
        proxy_screen = node.get("proxy_screen") if isinstance(node.get("proxy_screen"), dict) else {}
        if _numeric_key_present(proxy_screen, delta_keys):
            return True
    return False


def _entry_has_ablation_evidence(entry: dict[str, Any]) -> bool:
    for node in _walk_dicts(entry):
        if _non_empty(node.get("ablation_evidence")):
            return True
        for key in ["ablation_summary", "ablation_results", "matched_coverage_ablation_result"]:
            value = node.get(key)
            if isinstance(value, dict) and _non_empty(value) and str(value.get("status") or "").lower() not in {
                "missing",
                "not_run",
                "pending",
            }:
                return True
    return False


def _entry_mechanism_types(entry: dict[str, Any]) -> set[str]:
    mechanisms: set[str] = set()
    for node in _walk_dicts(entry):
        for key in ["mechanism_type", "mechanism"]:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                mechanisms.add(value.strip())
        variant = node.get("s2_variant") if isinstance(node.get("s2_variant"), dict) else {}
        value = variant.get("mechanism_type")
        if isinstance(value, str) and value.strip():
            mechanisms.add(value.strip())
    mechanisms.update(_proxy_calibration_risky_mechanisms(entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}))
    return mechanisms or {"unknown"}


def _entry_datasets(entry: dict[str, Any]) -> set[str]:
    datasets: set[str] = set()
    for node in _walk_dicts(entry):
        for key in ["target_datasets", "datasets", "mispredicted_datasets", "dragging_datasets", "risky_datasets"]:
            value = node.get(key)
            if isinstance(value, dict):
                datasets.update(str(dataset) for dataset in value.keys() if dataset)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("dataset"):
                        datasets.add(str(item["dataset"]))
                    elif isinstance(item, str):
                        datasets.add(item)
        for key in ["dataset_regressions", "proxy_dataset_deltas", "proxy_dataset_regressions", "dataset_error_summary", "dataset_calibration"]:
            value = node.get(key)
            if isinstance(value, dict):
                datasets.update(str(dataset) for dataset in value.keys() if dataset)
    return datasets


def _entry_failure_modes(entry: dict[str, Any]) -> set[str]:
    modes: set[str] = set()
    for node in _walk_dicts(entry):
        for key in ["failure_mode", "latest_failure_mode", "route", "decision", "failure_class", "recommended_s2_action"]:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                modes.add(value.strip())
        for key in ["failure_modes", "signals"]:
            value = node.get(key)
            if isinstance(value, list):
                modes.update(str(item) for item in value if item)
        attribution = node.get("failure_attribution") if isinstance(node.get("failure_attribution"), dict) else {}
        for key in ["primary_failure", "failure_mode"]:
            value = attribution.get(key)
            if isinstance(value, str) and value.strip():
                modes.add(value.strip())
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else {}
    modes.update(str(signal) for signal in quality.get("signals") or [] if signal)
    return modes


def _entry_text_tokens(entry: dict[str, Any]) -> set[str]:
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    values = [
        entry.get("topic"),
        entry.get("project_id"),
        entry.get("route"),
        (entry.get("summary") or {}).get("summary_text") if isinstance(entry.get("summary"), dict) else None,
        (entry.get("direction_scorecard") or {}).get("title") if isinstance(entry.get("direction_scorecard"), dict) else None,
        (entry.get("direction_scorecard") or {}).get("mechanism_type") if isinstance(entry.get("direction_scorecard"), dict) else None,
        method_context.get("title"),
        method_context.get("mechanism_type"),
        method_context.get("direction_id"),
    ]
    for item in entry.get("entries") or []:
        if isinstance(item, dict):
            values.extend([item.get("title"), item.get("id"), item.get("idea_id"), item.get("mechanism_type"), item.get("reason")])
    return _tokenize_text(" ".join(str(item) for item in values if item))


def _query_datasets_from_sources(sources: list[Any]) -> list[str]:
    datasets: set[str] = set()
    for source in sources:
        if not isinstance(source, (dict, list)):
            continue
        if isinstance(source, dict):
            datasets.update(_entry_datasets(source))
        else:
            for item in source:
                if isinstance(item, dict):
                    datasets.update(_entry_datasets(item))
    return sorted(dataset for dataset in datasets if dataset and dataset != "unknown")


def _query_mechanisms_from_sources(sources: list[Any]) -> list[str]:
    mechanisms: set[str] = set()
    for source in sources:
        if isinstance(source, dict):
            mechanisms.update(_entry_mechanism_types(source))
        elif isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    mechanisms.update(_entry_mechanism_types(item))
    return sorted(mechanism for mechanism in mechanisms if mechanism and mechanism != "unknown")


def _query_failure_modes_from_sources(sources: list[Any]) -> list[str]:
    modes: set[str] = set()
    for source in sources:
        if isinstance(source, dict):
            modes.update(_entry_failure_modes(source))
        elif isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    modes.update(_entry_failure_modes(item))
    return sorted(mode for mode in modes if mode)


def _source_repo_fingerprint(config: dict[str, Any] | None, project_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    snapshot_path = c2c_cfg.get("snapshot_path")
    if not snapshot_path:
        return {}
    path = Path(str(snapshot_path)).expanduser()
    if not path.is_absolute() and project_root:
        path = project_root / path
    return _strip_empty(
        {
            "snapshot_path_name": path.name,
            "snapshot_path_hash": hashlib.sha256(str(path.resolve() if path.exists() else path).encode("utf-8")).hexdigest()[:16],
            "allowed_files_hash": hashlib.sha256(json.dumps(c2c_cfg.get("allowed_files") or [], sort_keys=True).encode("utf-8")).hexdigest()[:16],
            "allowed_prefixes_hash": hashlib.sha256(json.dumps(c2c_cfg.get("allowed_prefixes") or [], sort_keys=True).encode("utf-8")).hexdigest()[:16],
        }
    )


def _source_repo_fingerprint_matches(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    for key in ["snapshot_path_hash", "allowed_files_hash", "allowed_prefixes_hash"]:
        if left.get(key) and right.get(key) and left.get(key) == right.get(key):
            return True
    return False


def _tokenize_text(text: Any) -> set[str]:
    raw = str(text or "").lower()
    return {token for token in re_split_tokens(raw) if len(token) >= 3}


def re_split_tokens(text: str) -> list[str]:
    return re.split(r"[^a-z0-9_\\-]+", text)


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            result.append(current)
            stack.extend(item for key, item in current.items() if key != "memory_quality")
        elif isinstance(current, list):
            stack.extend(current)
    return result


def _numeric_key_present(value: dict[str, Any], keys: list[str]) -> bool:
    return any(isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool) for key in keys)


def _numeric_map_has_negative(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(isinstance(item, (int, float)) and not isinstance(item, bool) and float(item) < 0 for item in value.values())


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _load_proxy_calibration_signal(project_root: Path) -> dict[str, Any]:
    path = project_root / "experiment" / "results" / "proxy_calibration.json"
    payload = _read_json_silent(path)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    current = payload.get("current_iteration") if isinstance(payload.get("current_iteration"), dict) else {}
    if not summary and not current:
        return {}
    method_feedback = summary.get("method_feedback") if isinstance(summary.get("method_feedback"), dict) else {}
    candidates = current.get("candidates") if isinstance(current.get("candidates"), list) else []
    false_positive_candidates = [
        _compact_proxy_false_positive_candidate(item)
        for item in candidates
        if isinstance(item, dict) and item.get("proxy_false_positive")
    ]
    false_positive_candidates = [item for item in false_positive_candidates if item]
    result = {
        "schema_version": "c2c_proxy_full_calibration_memory_v1",
        "source_path": "experiment/results/proxy_calibration.json",
        "current_iteration": {
            "iteration": current.get("iteration"),
            "acceptance_passed": current.get("acceptance_passed"),
            "candidate_count": current.get("candidate_count"),
            "proxy_false_positive_count": current.get("proxy_false_positive_count"),
            "proxy_false_positive_rate": current.get("proxy_false_positive_rate"),
            "dataset_error_summary": current.get("dataset_error_summary") or {},
            "proxy_full_delta_correlation": current.get("proxy_full_delta_correlation"),
        },
        "summary": {
            "candidate_count": summary.get("candidate_count"),
            "proxy_false_positive_count": summary.get("proxy_false_positive_count"),
            "proxy_false_positive_rate": summary.get("proxy_false_positive_rate"),
            "false_positive_reasons": summary.get("false_positive_reasons") or {},
            "proxy_full_delta_correlation": summary.get("proxy_full_delta_correlation"),
            "dataset_error_summary": summary.get("dataset_error_summary") or {},
            "mechanism_false_positive_summary": summary.get("mechanism_false_positive_summary") or {},
            "method_feedback": {
                "risky_datasets": (method_feedback.get("risky_datasets") or [])[:8],
                "risky_mechanisms": (method_feedback.get("risky_mechanisms") or [])[:8],
                "risky_integration_points": (method_feedback.get("risky_integration_points") or [])[:8],
                "recommendations": (method_feedback.get("recommendations") or [])[:8],
            },
        },
        "false_positive_candidates": false_positive_candidates[:8],
    }
    return _strip_empty(result)


def _compact_proxy_false_positive_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return _strip_empty(
        {
            "id": candidate.get("id"),
            "title": candidate.get("title"),
            "mechanism_type": candidate.get("mechanism_type"),
            "mechanism_axis": candidate.get("mechanism_axis"),
            "integration_point": candidate.get("integration_point"),
            "control_signal": candidate.get("control_signal"),
            "decision": candidate.get("decision"),
            "false_positive_reason": candidate.get("false_positive_reason"),
            "proxy_mean_delta": candidate.get("proxy_mean_delta"),
            "full_mean_delta": candidate.get("full_mean_delta"),
            "mispredicted_datasets": candidate.get("mispredicted_datasets") or [],
            "dataset_calibration": candidate.get("dataset_calibration") or {},
        }
    )


def _proxy_calibration_false_positive_count(proxy_calibration: dict[str, Any]) -> int:
    if not isinstance(proxy_calibration, dict):
        return 0
    for key in ["current_false_positive_count", "overall_false_positive_count", "proxy_false_positive_count"]:
        try:
            count = int(proxy_calibration.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            return count
    current = proxy_calibration.get("current_iteration") if isinstance(proxy_calibration.get("current_iteration"), dict) else {}
    summary = proxy_calibration.get("summary") if isinstance(proxy_calibration.get("summary"), dict) else {}
    for source in [current, summary]:
        try:
            count = int(source.get("proxy_false_positive_count") or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            return count
    return 0


def _proxy_calibration_mispredicted_datasets(proxy_calibration: dict[str, Any]) -> list[str]:
    datasets: set[str] = set()
    if not isinstance(proxy_calibration, dict):
        return []
    datasets.update(str(dataset) for dataset in proxy_calibration.get("mispredicted_datasets") or [] if dataset)
    for section_key in ["current_iteration", "summary"]:
        section = proxy_calibration.get(section_key) if isinstance(proxy_calibration.get(section_key), dict) else {}
        for dataset, stats in (section.get("dataset_error_summary") or {}).items():
            if isinstance(stats, dict) and int(stats.get("misprediction_count") or 0) > 0:
                datasets.add(str(dataset))
    for item in proxy_calibration.get("false_positive_candidates") or []:
        if isinstance(item, dict):
            datasets.update(str(dataset) for dataset in item.get("mispredicted_datasets") or [] if dataset)
    return sorted(datasets)


def _proxy_calibration_risky_mechanisms(proxy_calibration: dict[str, Any]) -> list[str]:
    mechanisms: set[str] = set()
    if not isinstance(proxy_calibration, dict):
        return []
    mechanisms.update(str(item) for item in proxy_calibration.get("risky_mechanisms") or [] if item)
    summary = proxy_calibration.get("summary") if isinstance(proxy_calibration.get("summary"), dict) else {}
    for mechanism, stats in (summary.get("mechanism_false_positive_summary") or {}).items():
        if isinstance(stats, dict) and int(stats.get("false_positive_count") or 0) > 0:
            mechanisms.add(str(mechanism))
    feedback = summary.get("method_feedback") if isinstance(summary.get("method_feedback"), dict) else {}
    for item in feedback.get("risky_mechanisms") or []:
        if isinstance(item, dict) and item.get("mechanism_type"):
            mechanisms.add(str(item["mechanism_type"]))
    for item in proxy_calibration.get("false_positive_candidates") or []:
        if isinstance(item, dict) and item.get("mechanism_type"):
            mechanisms.add(str(item["mechanism_type"]))
    return sorted(mechanisms)


def _strip_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: item for key, item in ((key, _strip_empty(item)) for key, item in value.items()) if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [item for item in (_strip_empty(item) for item in value) if item not in (None, "", [], {})]
    return value


def _read_json_silent(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_implementation_failure(performance_feedback: dict[str, Any]) -> bool:
    summary = performance_feedback.get("summary") if isinstance(performance_feedback.get("summary"), dict) else {}
    return summary.get("failure_class") == "implementation_failure" or bool(summary.get("does_not_consume_same_direction_attempt"))


def _entry_is_implementation_noise(entry: dict[str, Any]) -> bool:
    if entry.get("failure_class") == "implementation_failure":
        return True
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    if summary.get("failure_class") == "implementation_failure" or summary.get("does_not_consume_same_direction_attempt"):
        return True
    text = json.dumps(entry, ensure_ascii=False, default=str).lower()
    implementation_markers = [
        "runtimeerror",
        "traceback",
        "dtype",
        "valid_mask",
        "py_compile",
        "patch_too_broad",
        "evaluator",
        "test-only",
        "proxy command",
        "codex_failed",
        "429 too many requests",
    ]
    return not _entry_has_method_signal(entry) and any(marker in text for marker in implementation_markers)


def _entry_has_method_signal(entry: dict[str, Any]) -> bool:
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    if _proxy_calibration_false_positive_count(entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}):
        return True
    if summary.get("dataset_regressions") or summary.get("dragging_datasets") or summary.get("best_proxy_delta") is not None:
        return True
    for item in entry.get("entries") or []:
        if not isinstance(item, dict):
            continue
        if item.get("proxy_screen") or item.get("dataset_regressions") or item.get("dragging_datasets"):
            return True
        attribution = item.get("failure_attribution") if isinstance(item.get("failure_attribution"), dict) else {}
        if attribution.get("dragging_datasets") or attribution.get("mixed_gain_patterns"):
            return True
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    direction_summary = direction.get("summary") if isinstance(direction.get("summary"), dict) else {}
    return bool(direction_summary or method_context.get("best_proxy_delta") is not None or method_context.get("direction_quality") == "poor_direction_evidence")


def _memory_catalog_entry_for_prompt(entry: dict[str, Any], *, memory_path: str) -> dict[str, Any]:
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
    retrieval = entry.get("memory_retrieval") if isinstance(entry.get("memory_retrieval"), dict) else {}
    summary = _memory_one_line_summary(entry, quality=quality)
    return _strip_empty(
        {
            "memory_id": entry.get("memory_id"),
            "one_line_summary": summary,
            "priority": quality.get("priority"),
            "signals": (quality.get("signals") or [])[:8],
            "retrieval": {
                "combined_score": retrieval.get("combined_score"),
                "relevance_score": retrieval.get("relevance_score"),
                "matched_fields": retrieval.get("matched_fields") or [],
                "matched_values": retrieval.get("matched_values") or {},
            },
            "project_id": entry.get("project_id"),
            "topic": entry.get("topic"),
            "route": entry.get("route"),
            "datasets": sorted(_entry_datasets(entry))[:8],
            "mechanism_types": sorted(_entry_mechanism_types(entry))[:8],
            "failure_modes": sorted(_entry_failure_modes(entry))[:8],
            "read_hint": {
                "jsonl_path": memory_path,
                "snapshot_path": "intake/shared_method_failure_memory.json",
                "query": f"memory_id == {entry.get('memory_id')}",
                "suggested_commands": [
                    f"rg -n '\"memory_id\": \"{entry.get('memory_id')}\"' {memory_path}" if memory_path else "",
                    "python -m json.tool intake/shared_method_failure_memory.json",
                ],
            },
        }
    )


def _shared_memory_access_hints(memory_path: str) -> dict[str, Any]:
    return _strip_empty(
        {
            "catalog_only_prompt": True,
            "jsonl_path": memory_path,
            "project_snapshot_path": "intake/shared_method_failure_memory.json",
            "instruction": "Use memory_catalog/recent_entries as an index. If a memory seems relevant, inspect the full JSONL or project snapshot by memory_id before relying on detailed evidence.",
        }
    )


def _memory_one_line_summary(entry: dict[str, Any], *, quality: dict[str, Any]) -> str:
    datasets = sorted(_entry_datasets(entry))
    mechanisms = sorted(mechanism for mechanism in _entry_mechanism_types(entry) if mechanism != "unknown")
    signals = [str(signal) for signal in quality.get("signals") or []]
    route = str(entry.get("route") or "method_failure")
    project = str(entry.get("project_id") or "unknown_project")
    fragments = [f"{project}: {route}"]
    if mechanisms:
        fragments.append(f"mechanism={', '.join(mechanisms[:2])}")
    if datasets:
        fragments.append(f"datasets={', '.join(datasets[:3])}")
    if "proxy_full_false_positive" in signals:
        fragments.append("cheap proxy looked positive but full train failed")
    elif "dataset_regression" in signals:
        fragments.append("explicit dataset regression")
    elif "ablation_evidence" in signals:
        fragments.append("ablation evidence available")
    elif "repeated_failure" in signals:
        fragments.append("repeated direction failure")
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    if summary.get("summary_text"):
        fragments.append(str(summary["summary_text"])[:180])
    return "; ".join(fragment for fragment in fragments if fragment)


def _compact_entry_for_prompt(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    direction = entry.get("direction_scorecard") if isinstance(entry.get("direction_scorecard"), dict) else {}
    method_context = entry.get("method_context") if isinstance(entry.get("method_context"), dict) else {}
    proxy_calibration = entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {}
    quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
    return {
        "memory_id": entry.get("memory_id"),
        "memory_quality": quality,
        "memory_retrieval": entry.get("memory_retrieval") if isinstance(entry.get("memory_retrieval"), dict) else {},
        "timestamp": entry.get("timestamp"),
        "project_id": entry.get("project_id"),
        "topic": entry.get("topic"),
        "route": entry.get("route"),
        "failed_idea_ids": summary.get("failed_idea_ids") or [],
        "dragging_datasets": summary.get("dragging_datasets") or [],
        "dataset_regressions": summary.get("dataset_regressions") or {},
        "avoid_repeat_rules": summary.get("avoid_repeat_rules") or summary.get("avoid_repeat_rule") or [],
        "failure_modes": summary.get("failure_modes") or [],
        "summary_text": summary.get("summary_text"),
        "proxy_calibration": _compact_proxy_calibration_for_prompt(proxy_calibration),
        "method_context": method_context
        or _strip_empty(
            {
                "direction_id": direction.get("direction_id"),
                "title": direction.get("title"),
                "mechanism_type": direction.get("mechanism_type"),
                "summary": direction.get("summary") or {},
                "s1_feedback": direction.get("s1_feedback") or {},
            }
        ),
    }


def _compact_proxy_calibration_for_prompt(proxy_calibration: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(proxy_calibration, dict) or not proxy_calibration:
        return {}
    summary = proxy_calibration.get("summary") if isinstance(proxy_calibration.get("summary"), dict) else {}
    current = proxy_calibration.get("current_iteration") if isinstance(proxy_calibration.get("current_iteration"), dict) else {}
    method_feedback = summary.get("method_feedback") if isinstance(summary.get("method_feedback"), dict) else {}
    return _strip_empty(
        {
            "current_false_positive_count": current.get("proxy_false_positive_count")
            if current.get("proxy_false_positive_count") is not None
            else proxy_calibration.get("current_false_positive_count"),
            "current_false_positive_rate": current.get("proxy_false_positive_rate")
            if current.get("proxy_false_positive_rate") is not None
            else proxy_calibration.get("current_false_positive_rate"),
            "overall_false_positive_count": summary.get("proxy_false_positive_count")
            if summary.get("proxy_false_positive_count") is not None
            else proxy_calibration.get("overall_false_positive_count"),
            "overall_false_positive_rate": summary.get("proxy_false_positive_rate")
            if summary.get("proxy_false_positive_rate") is not None
            else proxy_calibration.get("overall_false_positive_rate"),
            "proxy_full_delta_correlation": summary.get("proxy_full_delta_correlation")
            if summary.get("proxy_full_delta_correlation") is not None
            else proxy_calibration.get("proxy_full_delta_correlation"),
            "mispredicted_datasets": _proxy_calibration_mispredicted_datasets(proxy_calibration),
            "risky_mechanisms": _proxy_calibration_risky_mechanisms(proxy_calibration),
            "risky_integration_points": [
                item.get("integration_point")
                for item in method_feedback.get("risky_integration_points") or []
                if isinstance(item, dict) and item.get("integration_point")
            ][:8]
            or proxy_calibration.get("risky_integration_points")
            or [],
            "recommendations": (method_feedback.get("recommendations") or proxy_calibration.get("recommendations") or [])[:8],
            "false_positive_candidates": (proxy_calibration.get("false_positive_candidates") or [])[:4],
        }
    )


def _explicit_used_shared_memory_refs(payload: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(payload, dict):
        for key in ["used_shared_memory_refs", "shared_memory_refs", "used_method_memory_refs"]:
            value = payload.get(key)
            if isinstance(value, list):
                refs.extend(str(item) for item in value if item not in (None, ""))
        for key in ["direction_decision", "negative_constraints", "decision_chain", "planner_summary"]:
            refs.extend(_explicit_used_shared_memory_refs(payload.get(key)))
        for key in ["selected_ideas", "candidate_ideas", "variant_candidates", "selected_variant_candidates", "candidates"]:
            refs.extend(_explicit_used_shared_memory_refs(payload.get(key)))
    elif isinstance(payload, list):
        for item in payload:
            refs.extend(_explicit_used_shared_memory_refs(item))
    return [item for item in refs if item]


def _write_shared_method_memory_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = ["# Shared Method Failure Memory", ""]
    if not entries:
        lines.append("- No method-level failures recorded.")
    for entry in reversed(entries[-80:]):
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        lines.append(f"## {entry.get('memory_id')} | {entry.get('project_id')}")
        lines.append(f"- Timestamp: {entry.get('timestamp')}")
        lines.append(f"- Route: {entry.get('route')}")
        if entry.get("topic"):
            lines.append(f"- Topic: {entry.get('topic')}")
        if summary.get("summary_text"):
            lines.append(f"- Summary: {summary.get('summary_text')}")
        quality = entry.get("memory_quality") if isinstance(entry.get("memory_quality"), dict) else _memory_quality(entry)
        lines.append(f"- Priority: {quality.get('priority')} ({', '.join(quality.get('signals') or [])})")
        components = quality.get("score_components") if isinstance(quality.get("score_components"), dict) else {}
        if components:
            lines.append(f"- Quality components: {json.dumps(components, ensure_ascii=False, sort_keys=True)}")
        proxy_calibration = _compact_proxy_calibration_for_prompt(entry.get("proxy_calibration") if isinstance(entry.get("proxy_calibration"), dict) else {})
        if proxy_calibration:
            lines.append(f"- Proxy/full calibration: {json.dumps(proxy_calibration, ensure_ascii=False)}")
        if summary.get("failed_idea_ids"):
            lines.append(f"- Failed ideas: {', '.join(str(item) for item in summary.get('failed_idea_ids')[:8])}")
        if summary.get("dragging_datasets"):
            lines.append(f"- Dragging datasets: {json.dumps(summary.get('dragging_datasets'), ensure_ascii=False)}")
        rules = summary.get("avoid_repeat_rules") or []
        if rules:
            lines.append(f"- Avoid: {rules[0]}")
        lines.append("")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _project_topic(project_root: Path) -> str:
    for path in [project_root / "meta" / "registry.yaml", project_root / "meta" / "project_config.yaml"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for prefix in ["research_topic:", "topic:"]:
            for line in text.splitlines():
                if line.strip().startswith(prefix):
                    return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)
