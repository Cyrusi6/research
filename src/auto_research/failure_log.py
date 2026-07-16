"""Failure logging and cleanup for rejected ideas and stopped runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .research_state import ResearchEventLedger
from .utils import now_utc

C2C_FAILED_DECISIONS = {None, "not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}
C2C_FEEDBACK_VIEWS = {"all", "implementation", "method"}
C2C_RETRYABLE_PAUSE_TYPES = {
    "runtime_smoke_resource_retry",
    "s3_proxy_resource_retry",
    "codex_quota_or_rate_limit",
    "retryable_quota_or_rate_limit",
}
C2C_RESOURCE_FAILURE_CATEGORIES = {
    "resource_oom",
    "s3_proxy_resource_oom",
    "s3_proxy_gpu_resource_retry",
    "runtime_smoke_resource_retry",
}


class FailureLogManager:
    def __init__(self, config: dict[str, Any], *, external_root: Path):
        self.config = config
        self.external_root = external_root
        self.failure_md = external_root / config.get("experiment", {}).get("failure_log_filename", "failure.md")
        self.failure_jsonl = external_root / "failure.jsonl"

    def record_not_viable_ideas(
        self,
        *,
        project_id: str,
        baseline_metrics: dict[str, Any] | None,
        candidate_results: list[dict[str, Any]],
        cleanup: bool = True,
    ) -> list[str]:
        removed = []
        entries = []
        for item in candidate_results:
            if item.get("decision") != "not_viable":
                continue
            run_dir = Path(item.get("train_log", "")).parent
            metrics = item.get("metrics") or {}
            reason = self._reason_not_viable(metrics, baseline_metrics)
            entries.append(
                {
                    "timestamp": now_utc(),
                    "project_id": project_id,
                    "kind": "not_viable_idea",
                    "idea_id": item.get("id"),
                    "title": item.get("title"),
                    "direction": item.get("direction"),
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "reason": reason,
                    "cleanup_performed": cleanup and run_dir.exists(),
                }
            )
            if cleanup and run_dir.exists():
                shutil.rmtree(run_dir)
                removed.append(str(run_dir))
        if entries:
            self._append_entries(entries)
        return removed

    def record_stopped_validation(
        self,
        *,
        project_id: str,
        title: str,
        run_dir: Path,
        summary: dict[str, Any],
        reason: str,
        cleanup: bool = False,
    ) -> None:
        entry = {
            "timestamp": now_utc(),
            "project_id": project_id,
            "kind": "stopped_validation",
            "title": title,
            "run_dir": str(run_dir),
            "summary": summary,
            "reason": reason,
            "cleanup_performed": cleanup and run_dir.exists(),
        }
        if cleanup and run_dir.exists():
            shutil.rmtree(run_dir)
        self._append_entries([entry])

    def append_c2c_feedback(
        self,
        *,
        project_id: str,
        iteration: int,
        candidate: dict[str, Any] | None,
        acceptance: dict[str, Any] | None,
        failure_mode: str,
        reason: str,
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        metrics = (candidate or {}).get("metrics") or {}
        proxy_screen = (candidate or {}).get("proxy_screen") or {}
        entry = {
            "timestamp": now_utc(),
            "project_id": project_id,
            "iteration": iteration,
            "kind": "c2c_failure_feedback",
            "idea_id": (candidate or {}).get("id"),
            "title": (candidate or {}).get("title"),
            "decision": (candidate or {}).get("decision"),
            "failure_mode": failure_mode,
            "reason": reason,
            "metrics": metrics,
            "proxy_screen": proxy_screen,
            "proxy_metrics": (proxy_screen.get("metrics") or {}) if isinstance(proxy_screen, dict) else {},
            "acceptance": acceptance or {},
            "dataset_regressions": (candidate or {}).get("dataset_regressions") or {},
            "failure_attribution": (candidate or {}).get("failure_attribution") or {},
            "avoid_repeat_rule": self._c2c_avoid_repeat_rule(candidate, acceptance, reason),
            "artifacts": artifacts or [],
            "cleanup_performed": False,
        }
        self._append_entries([entry])
        return entry

    def _append_entries(self, entries: list[dict[str, Any]]) -> None:
        self.failure_md.parent.mkdir(parents=True, exist_ok=True)
        self.failure_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.failure_jsonl.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.failure_md.write_text(self._render_markdown(self._load_entries()), encoding="utf-8")

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.failure_jsonl.exists():
            return []
        entries = []
        for line in self.failure_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    @staticmethod
    def _reason_not_viable(metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return "The run did not produce trusted metrics."
        if not baseline_metrics:
            return "The run finished but no baseline was available for comparison."
        reasons = []
        if metrics.get("rsum", 0) < baseline_metrics.get("rsum", 0):
            reasons.append("rsum did not beat the matched baseline")
        if (metrics.get("t2i") or {}).get("R@1", 0) <= (baseline_metrics.get("t2i") or {}).get("R@1", 0):
            reasons.append("T2I R@1 did not improve")
        if metrics.get("similarity_time") and baseline_metrics.get("similarity_time"):
            if metrics["similarity_time"] > baseline_metrics["similarity_time"] * 1.15:
                reasons.append("inference time regressed too much")
        return "; ".join(reasons) or "The idea did not clear the screening threshold."

    @staticmethod
    def _render_markdown(entries: list[dict[str, Any]]) -> str:
        lines = ["# Failure Log", ""]
        if not entries:
            lines.append("- No failures recorded.")
            return "\n".join(lines) + "\n"
        for idx, entry in enumerate(entries, start=1):
            lines.append(f"## Entry {idx}")
            lines.append(f"- Timestamp: {entry.get('timestamp')}")
            lines.append(f"- Project: {entry.get('project_id')}")
            lines.append(f"- Kind: {entry.get('kind')}")
            if entry.get("title"):
                lines.append(f"- Title: {entry.get('title')}")
            if entry.get("idea_id"):
                lines.append(f"- Idea id: {entry.get('idea_id')}")
            if entry.get("direction"):
                lines.append(f"- Direction: {entry.get('direction')}")
            if entry.get("run_dir"):
                lines.append(f"- Run dir: {entry.get('run_dir')}")
            if entry.get("reason"):
                lines.append(f"- Reason: {entry.get('reason')}")
            if entry.get("failure_mode"):
                lines.append(f"- Failure mode: {entry.get('failure_mode')}")
            if entry.get("avoid_repeat_rule"):
                lines.append(f"- Avoid repeat rule: {entry.get('avoid_repeat_rule')}")
            attribution = entry.get("failure_attribution") or {}
            if attribution:
                lines.append(f"- Primary failure: {attribution.get('primary_failure')}")
                lines.append(f"- Dragging datasets: {json.dumps(attribution.get('dragging_datasets') or [], ensure_ascii=False)}")
                lines.append(f"- Sample type failures: {json.dumps(attribution.get('sample_type_failures') or [], ensure_ascii=False)}")
                lines.append(f"- Patch risk labels: {json.dumps((attribution.get('patch_risk') or {}).get('risk_labels') or [], ensure_ascii=False)}")
            metrics = entry.get("metrics") or {}
            if metrics:
                if "mean" in metrics:
                    lines.append(f"- Metrics: mean={metrics.get('mean')}, datasets={json.dumps(metrics.get('datasets') or {}, ensure_ascii=False)}")
                else:
                    lines.append(f"- Metrics: rsum={metrics.get('rsum')}, i2t_R1={(metrics.get('i2t') or {}).get('R@1')}, t2i_R1={(metrics.get('t2i') or {}).get('R@1')}")
            summary = entry.get("summary") or {}
            if summary:
                lines.append(f"- Summary: {json.dumps(summary, ensure_ascii=False)}")
            lines.append(f"- Cleanup performed: {entry.get('cleanup_performed')}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _c2c_avoid_repeat_rule(candidate: dict[str, Any] | None, acceptance: dict[str, Any] | None, reason: str) -> str:
        candidate = candidate or {}
        regressions = candidate.get("dataset_regressions") or {}
        if regressions:
            worst_dataset = max(regressions, key=lambda key: regressions[key])
            if regressions[worst_dataset] > float((acceptance or {}).get("max_dataset_regression", 2.0)):
                return f"Do not repeat this mechanism without addressing {worst_dataset} regression."
        attribution = candidate.get("failure_attribution") or {}
        if attribution.get("primary_failure") == "cheap_proxy_rejected_before_full_training":
            return "Do not rerun this mechanism in full S3 until cheap proxy risk is cleared."
        if attribution.get("primary_failure") == "repairable_proxy_risk_before_full_training":
            return "Repair the S2.5 patch and rerun cheap proxy before discarding this mechanism."
        if attribution.get("primary_failure") == "ablation_no_effect":
            return "Do not rerun until the ablation switch measurably changes enabled-vs-disabled behavior."
        dragging = attribution.get("dragging_datasets") or []
        if dragging and isinstance(dragging[0], dict):
            return f"Do not repeat this mechanism without addressing {dragging[0].get('dataset')} regression evidence."
        proxy_screen = candidate.get("proxy_screen") or {}
        if candidate.get("decision") == "proxy_rejected":
            proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
            if proxy_deltas:
                try:
                    worst_dataset = min(proxy_deltas, key=lambda key: float(proxy_deltas[key]))
                    return f"Do not send this mechanism to full S3 until cheap proxy regression on {worst_dataset} is repaired."
                except (TypeError, ValueError):
                    pass
            return "Do not send this mechanism to full S3 until cheap proxy risk is cleared."
        if candidate.get("decision") == "proxy_repairable":
            return "Repair the S2.5 patch and rerun cheap proxy before discarding this mechanism."
        if "no candidate metrics" in reason or not candidate.get("metrics"):
            return "Do not rerun without fixing preflight, checkpoint, or evaluator failures."
        return "Do not repeat below-baseline variants without a new mechanism and explicit ablation."


def build_c2c_feedback_bundle(
    entries: list[dict[str, Any]],
    *,
    project_id: str | None = None,
    iteration: int | None = None,
    traces: list[dict[str, Any]] | None = None,
    sources: list[str] | None = None,
    view: str = "all",
) -> dict[str, Any]:
    view = view if view in C2C_FEEDBACK_VIEWS else "all"
    entries = [
        cleaned
        for cleaned in (_strip_retryable_c2c_feedback_noise(entry) for entry in entries if isinstance(entry, dict))
        if cleaned
    ]
    traces = [
        cleaned
        for cleaned in (_strip_retryable_c2c_feedback_noise(trace) for trace in (traces or []) if isinstance(trace, dict))
        if cleaned
    ]
    normalized_entries = _dedupe_c2c_feedback_entries(
        [_normalize_feedback_entry(entry) for entry in entries if isinstance(entry, dict)]
    )
    normalized_traces = _dedupe_c2c_feedback_entries(
        [_normalize_feedback_entry(entry) for entry in traces if isinstance(entry, dict)]
    )
    if view == "method":
        normalized_entries = _dedupe_c2c_feedback_entries(_method_feedback_entries(normalized_entries))
        normalized_traces = []
    summary = _summarize_c2c_feedback(normalized_entries, iteration_traces=normalized_traces, project_id=project_id, iteration=iteration)
    summary_entry = {
        "timestamp": now_utc(),
        "project_id": project_id or summary.get("project_id"),
        "iteration": iteration or summary.get("latest_iteration"),
        "kind": "c2c_feedback_summary",
        "feedback_view": view,
        **summary,
    }
    return {
        "created_at": now_utc(),
        "feedback_view": view,
        "project_id": project_id or summary.get("project_id"),
        "iteration": iteration or summary.get("latest_iteration"),
        "sources": sorted({str(item) for item in (sources or []) if item}),
        "entries": normalized_entries,
        "iteration_traces": normalized_traces,
        "summary": summary,
        "summary_entry": summary_entry,
        "feedback_items": [summary_entry, *normalized_entries],
    }


def load_c2c_feedback_bundle(project_root: Path, *, view: str = "all") -> dict[str, Any]:
    project_root = Path(project_root)
    entries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    sources: list[str] = []

    feedback_sources = [
        project_root / "meta" / "negative_memory.jsonl",
        project_root / "plan" / "direction_scorecard.json",
        project_root / "plan" / "performance_feedback.json",
        project_root / "experiment" / "results" / "proxy_calibration.json",
        project_root / "experiment" / "results" / "failure_feedback.json",
    ]
    ledger_path = project_root / "meta" / "research_events.sqlite3"
    if ledger_path.exists():
        sources.append(_relative_path(project_root, ledger_path))
        entries.extend(_authoritative_feedback_entries(ResearchEventLedger(project_root).state()))
    feedback_dir = project_root / "literature" / "feedback"
    if feedback_dir.exists():
        feedback_sources.extend(sorted(feedback_dir.glob("failed_ideas_round_*.json")))
    trace_path = project_root / "meta" / "iteration_trace.jsonl"

    for path in feedback_sources:
        if not path.exists():
            continue
        sources.append(_relative_path(project_root, path))
        payload, payload_entries, payload_traces = _load_feedback_payloads(path)
        if payload_entries:
            entries.extend(payload_entries)
        elif isinstance(payload, dict):
            entries.append(payload)
        if payload_traces:
            traces.extend(payload_traces)

    if trace_path.exists():
        sources.append(_relative_path(project_root, trace_path))
        _, trace_entries, trace_traces = _load_feedback_payloads(trace_path, kind="traces")
        traces.extend(trace_traces or trace_entries)

    if view == "method":
        shared_memory_path = project_root / "intake" / "shared_method_failure_memory.json"
        if shared_memory_path.exists():
            sources.append(_relative_path(project_root, shared_memory_path))
            _, shared_entries, _ = _load_feedback_payloads(shared_memory_path)
            entries.extend(shared_entries)

    return build_c2c_feedback_bundle(entries, project_id=project_root.name, traces=traces, sources=sources, view=view)


def _authoritative_feedback_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = state.get("attempts") if isinstance(state.get("attempts"), dict) else {}
    entries: list[dict[str, Any]] = []
    for attempt_id, trial in sorted((state.get("trial_results") or {}).items()):
        if not isinstance(trial, dict) or trial.get("outcome_classification") != "rejected":
            continue
        attempt = attempts.get(attempt_id) if isinstance(attempts, dict) else {}
        entries.append(_trial_feedback_entry(attempt if isinstance(attempt, dict) else {}, trial))
    for attempt_id, outcome in sorted((state.get("proxy_outcomes") or {}).items()):
        if not isinstance(outcome, dict) or outcome.get("decision") != "PROPOSE_NEXT_VARIANT":
            continue
        attempt = attempts.get(attempt_id) if isinstance(attempts, dict) else {}
        entries.append(_proxy_feedback_entry(attempt if isinstance(attempt, dict) else {}, outcome))
    seen_failure_events: set[str] = set()
    for operation in (state.get("operation_events") or {}).values():
        if not isinstance(operation, dict):
            continue
        event_id = str(operation.get("event_id") or "")
        if event_id in seen_failure_events:
            continue
        payload = operation.get("payload") if isinstance(operation.get("payload"), dict) else {}
        evidence = payload.get("failure_evidence") if isinstance(payload.get("failure_evidence"), dict) else {}
        if not evidence or evidence.get("failure_class") in {"resource_pause", "oom_retry"}:
            continue
        attempt = attempts.get(evidence.get("attempt_id")) if isinstance(attempts, dict) else {}
        entries.append(_failure_feedback_entry(attempt if isinstance(attempt, dict) else {}, evidence, event_id))
        seen_failure_events.add(event_id)
    return entries


def _trial_feedback_entry(attempt: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    summary = trial.get("primary_metric_summary") if isinstance(trial.get("primary_metric_summary"), dict) else {}
    metric_id = str(summary.get("metric_id") or "primary_metric")
    observations = [
        item
        for item in trial.get("observations") or []
        if isinstance(item, dict) and item.get("metric_id") == metric_id and item.get("role") in {"baseline", "candidate"}
    ]
    role_datasets: dict[str, dict[str, list[float]]] = {"baseline": {}, "candidate": {}}
    for observation in observations:
        role = str(observation["role"])
        dataset = str(observation.get("dataset_id") or "")
        if dataset:
            role_datasets[role].setdefault(dataset, []).append(float(observation["metric_value"]))
    baseline_datasets = {key: sum(values) / len(values) for key, values in role_datasets["baseline"].items() if values}
    candidate_datasets = {key: sum(values) / len(values) for key, values in role_datasets["candidate"].items() if values}
    improvements = summary.get("paired_improvements") if isinstance(summary.get("paired_improvements"), dict) else {}
    dataset_regressions: dict[str, float] = {}
    for pair_id, value in improvements.items():
        dataset = str(pair_id).rsplit(":", 1)[0]
        regression = max(0.0, -float(value))
        if regression:
            dataset_regressions[dataset] = max(dataset_regressions.get(dataset, 0.0), regression)
    return {
        "kind": "authoritative_trial_feedback",
        "source": "research_event_ledger",
        "attempt_id": trial.get("attempt_id"),
        "idea_id": trial.get("variant_id") or attempt.get("variant_id"),
        "title": trial.get("variant_id") or attempt.get("variant_id"),
        "decision": "not_viable",
        "failure_mode": "method_rejected",
        "reason": "verified TrialResult failed one or more preregistered acceptance constraints",
        "metrics": {
            "mean": summary.get("candidate_mean"),
            "datasets": candidate_datasets,
        },
        "baseline_metrics": {
            "mean": summary.get("baseline_mean"),
            "datasets": baseline_datasets,
        },
        "dataset_regressions": dataset_regressions,
        "acceptance": {
            "passed": False,
            "delta": summary.get("delta"),
            "all_hard_constraints_passed": trial.get("all_hard_constraints_passed"),
            "constraint_results": trial.get("constraint_results") or [],
        },
    }


def _proxy_feedback_entry(attempt: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    deltas = {str(key): float(value) for key, value in (outcome.get("dataset_deltas") or {}).items()}
    regressions = {key: max(0.0, -value) for key, value in deltas.items() if value < 0}
    return {
        "kind": "authoritative_proxy_feedback",
        "source": "research_event_ledger",
        "attempt_id": outcome.get("attempt_id"),
        "idea_id": attempt.get("variant_id"),
        "title": attempt.get("variant_id"),
        "decision": "proxy_rejected",
        "failure_mode": "proxy_rejected",
        "reason": ", ".join(str(item) for item in outcome.get("reason_codes") or []),
        "proxy_screen": {
            "status": "rejected",
            "proxy_delta_vs_baseline": outcome.get("observed_delta"),
            "proxy_dataset_deltas": deltas,
            "proxy_dataset_regressions": regressions,
            "proxy_worst_dataset_regression": outcome.get("worst_dataset_regression"),
        },
        "dataset_regressions": regressions,
        "proxy_outcome_hash": outcome.get("evidence_set_hash"),
    }


def _failure_feedback_entry(attempt: dict[str, Any], evidence: dict[str, Any], event_id: str) -> dict[str, Any]:
    failure_class = str(evidence.get("failure_class") or "integrity_failure")
    return {
        "kind": "authoritative_attempt_failure_feedback",
        "source": "research_event_ledger",
        "source_event_id": event_id,
        "attempt_id": evidence.get("attempt_id"),
        "idea_id": attempt.get("variant_id"),
        "title": attempt.get("variant_id"),
        "decision": "proxy_repairable" if failure_class in {"implementation_failure", "activation_failure"} else "blocked",
        "failure_mode": failure_class,
        "reason": evidence.get("reason") or failure_class,
        "failure_class": failure_class,
        "source_phase": evidence.get("source_phase"),
        "evidence_id": evidence.get("evidence_id"),
        "receipt_hash": evidence.get("receipt_hash"),
    }


def is_retryable_c2c_candidate(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    if candidate.get("resource_retry") is True:
        return True
    if candidate.get("pause_type") in C2C_RETRYABLE_PAUSE_TYPES:
        return True
    if candidate.get("failure_category") in C2C_RESOURCE_FAILURE_CATEGORIES:
        return True
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    if proxy.get("resource_retry") is True:
        return True
    if proxy.get("status") == "resource_retry":
        return True
    if proxy.get("failure_category") in C2C_RESOURCE_FAILURE_CATEGORIES:
        return True
    command_failure = proxy.get("command_failure") if isinstance(proxy.get("command_failure"), dict) else {}
    if command_failure.get("category") in C2C_RESOURCE_FAILURE_CATEGORIES:
        return True
    validation = ((candidate.get("patch_result") or {}).get("validation") or {}) if isinstance(candidate.get("patch_result"), dict) else {}
    if validation.get("resource_retry") is True or validation.get("failure_category") in C2C_RESOURCE_FAILURE_CATEGORIES:
        return True
    for check in validation.get("checks") or []:
        if isinstance(check, dict) and (check.get("resource_retry") is True or check.get("failure_category") in C2C_RESOURCE_FAILURE_CATEGORIES):
            return True
    return False


def is_retryable_c2c_feedback_entry(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    saw_candidate_list = False
    for list_key in ["candidate_results", "feedback_entries", "entries", "feedback_items"]:
        candidates = [item for item in entry.get(list_key) or [] if isinstance(item, dict)]
        if candidates:
            saw_candidate_list = True
            if not all(is_retryable_c2c_candidate(item) or is_retryable_c2c_feedback_entry(item) for item in candidates):
                return False
    if saw_candidate_list:
        return True
    if entry.get("status") == "retryable_paused":
        return True
    if entry.get("pause_type") in C2C_RETRYABLE_PAUSE_TYPES:
        return True
    if entry.get("route") == "resource_retry" or entry.get("repair_route") == "resource_retry":
        return True
    if entry.get("failure_class") == "resource_retry":
        return True
    if entry.get("failure_mode") in {"resource_retry", "s3_proxy_resource_retry", "runtime_smoke_resource_retry"}:
        return True
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    if summary.get("failure_class") == "resource_retry" or summary.get("route") == "resource_retry":
        return True
    if is_retryable_c2c_candidate(entry):
        return True
    for key in ["entry", "candidate", "candidate_snapshot", "entry_snapshot", "best_candidate", "best_proxy_candidate"]:
        if is_retryable_c2c_candidate(entry.get(key)):
            return True
    return False


def _strip_retryable_c2c_feedback_noise(entry: dict[str, Any]) -> dict[str, Any] | None:
    if is_retryable_c2c_feedback_entry(entry):
        return None
    cleaned = dict(entry)
    removed_candidate = False
    for key in ["entry", "candidate", "candidate_snapshot", "entry_snapshot", "best_candidate", "best_proxy_candidate"]:
        nested = cleaned.get(key)
        if is_retryable_c2c_candidate(nested) or (isinstance(nested, dict) and is_retryable_c2c_feedback_entry(nested)):
            cleaned.pop(key, None)
            removed_candidate = True
    for key in ["candidate_results", "feedback_entries", "entries", "feedback_items"]:
        values = cleaned.get(key)
        if not isinstance(values, list):
            continue
        filtered = [
            item
            for item in values
            if not (isinstance(item, dict) and (is_retryable_c2c_candidate(item) or is_retryable_c2c_feedback_entry(item)))
        ]
        if filtered:
            cleaned[key] = filtered
        else:
            cleaned.pop(key, None)
        if len(filtered) != len(values):
            removed_candidate = True
    if removed_candidate:
        for key in ["failed_idea_ids", "failed_titles", "avoid_repeat_rules", "blocked_idea_patterns"]:
            cleaned.pop(key, None)
    if is_retryable_c2c_feedback_entry(cleaned):
        return None
    return cleaned


def _load_feedback_payloads(path: Path, *, kind: str = "entries") -> tuple[dict[str, Any] | list[dict[str, Any]] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    if kind == "traces":
                        traces.append(item)
                    else:
                        payloads.append(item)
            return None, payloads, traces
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, payloads, traces
    if isinstance(payload, list):
        payloads.extend(item for item in payload if isinstance(item, dict))
    elif isinstance(payload, dict):
        if (
            payload.get("schema_version") == "shared_method_failure_memory_v1"
            and isinstance(payload.get("feedback_items"), list)
        ):
            payloads.extend(item for item in payload.get("feedback_items", []) if isinstance(item, dict))
        elif isinstance(payload.get("entries"), list) or isinstance(payload.get("summary_entry"), dict):
            if isinstance(payload.get("summary_entry"), dict):
                payloads.append(payload["summary_entry"])
            payloads.extend(item for item in payload.get("entries") or [] if isinstance(item, dict))
        elif isinstance(payload.get("feedback_items"), list):
            payloads.extend(item for item in payload.get("feedback_items", []) if isinstance(item, dict))
        else:
            payloads.append(payload)
        if isinstance(payload.get("iteration_traces"), list):
            traces.extend(item for item in payload.get("iteration_traces") or [] if isinstance(item, dict))
    return payload, payloads, traces


def _normalize_feedback_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    candidate = normalized.get("candidate")
    if isinstance(candidate, dict):
        candidate_snapshot = {
            "id": candidate.get("id"),
            "title": candidate.get("title"),
            "decision": candidate.get("decision"),
            "metrics": candidate.get("metrics") or {},
            "dataset_regressions": candidate.get("dataset_regressions") or {},
            "failure_attribution": candidate.get("failure_attribution") or {},
            "proxy_screen": candidate.get("proxy_screen") or {},
        }
        normalized.setdefault("candidate_snapshot", candidate_snapshot)
    entry_payload = normalized.get("entry")
    if isinstance(entry_payload, dict):
        normalized.setdefault("entry_snapshot", {
            "idea_id": entry_payload.get("idea_id"),
            "title": entry_payload.get("title"),
            "decision": entry_payload.get("decision"),
            "reason": entry_payload.get("reason"),
            "avoid_repeat_rule": entry_payload.get("avoid_repeat_rule"),
            "failure_attribution": entry_payload.get("failure_attribution") or {},
        })
    return normalized


def _method_feedback_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_entries: list[dict[str, Any]] = []
    for entry in entries:
        sanitized = _sanitize_method_feedback_entry(entry)
        if sanitized:
            method_entries.append(sanitized)
    return method_entries


def _sanitize_method_feedback_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    sanitized = _sanitize_method_feedback_item(entry)
    if sanitized is None:
        sanitized = {}
        for key in ["timestamp", "project_id", "iteration", "kind", "source_path", "path"]:
            if entry.get(key) not in (None, "", [], {}):
                sanitized[key] = entry[key]

    for nested_key in ["entry", "candidate", "candidate_snapshot", "entry_snapshot"]:
        nested = _sanitize_method_feedback_item(entry.get(nested_key))
        if nested:
            sanitized[nested_key] = nested

    candidate_results = [
        item
        for item in (_sanitize_method_feedback_item(candidate) for candidate in entry.get("candidate_results") or [])
        if item
    ]
    if candidate_results:
        sanitized["candidate_results"] = candidate_results
        sanitized["failed_idea_ids"] = [item.get("id") or item.get("idea_id") for item in candidate_results if item.get("id") or item.get("idea_id")]
        sanitized["failed_titles"] = [item.get("title") for item in candidate_results if item.get("title")]

    posthoc = _sanitize_method_posthoc(entry.get("posthoc_review"))
    if posthoc:
        sanitized["posthoc_review"] = posthoc

    direction_scorecard = _sanitize_method_direction_scorecard(entry)
    if direction_scorecard:
        sanitized["direction_scorecard"] = direction_scorecard
        sanitized["kind"] = sanitized.get("kind") or "c2c_direction_scorecard"

    proxy_calibration = _sanitize_method_proxy_calibration(entry)
    if proxy_calibration:
        sanitized["proxy_calibration"] = proxy_calibration
        sanitized["kind"] = sanitized.get("kind") or "c2c_proxy_calibration"

    if not _has_method_feedback_evidence(sanitized):
        return None

    sanitized["feedback_view"] = "method"
    return {key: value for key, value in sanitized.items() if value not in (None, "", [], {})}


def _sanitize_method_feedback_item(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sanitized: dict[str, Any] = {}
    for key in [
        "timestamp",
        "project_id",
        "iteration",
        "kind",
        "idea_id",
        "id",
        "candidate_id",
        "title",
        "decision",
        "source_path",
        "path",
        "feedback_round_path",
        "feedback_view",
    ]:
        if value.get(key) not in (None, "", [], {}):
            sanitized[key] = value[key]

    metrics = _sanitize_method_metrics(value.get("metrics"))
    if metrics:
        sanitized["metrics"] = metrics

    dataset_regressions = _sanitize_float_map(value.get("dataset_regressions"))
    if dataset_regressions:
        sanitized["dataset_regressions"] = dataset_regressions

    proxy_screen = _sanitize_method_proxy_screen(value.get("proxy_screen"))
    if proxy_screen:
        sanitized["proxy_screen"] = proxy_screen

    attribution = _sanitize_method_failure_attribution(value.get("failure_attribution"))
    if attribution:
        sanitized["failure_attribution"] = attribution

    for key in ["dragging_datasets", "sample_type_failures", "mixed_gain_patterns"]:
        items = _sanitize_method_list(value.get(key))
        if items:
            sanitized[key] = items

    ablation = _sanitize_method_ablation(value.get("ablation_evidence"))
    if ablation:
        sanitized["ablation_evidence"] = ablation

    if value.get("kind") != "c2c_feedback_summary":
        for key in ["failed_idea_ids", "failed_titles"]:
            items = _sanitize_method_list(value.get(key))
            if items and _has_direct_method_evidence(sanitized):
                sanitized[key] = items

    failure_mode = value.get("failure_mode") or value.get("latest_failure_mode")
    if failure_mode and _is_method_feedback_text(failure_mode):
        sanitized["failure_mode"] = str(failure_mode)

    failure_modes = [str(item) for item in _sanitize_method_list(value.get("failure_modes")) if _is_method_feedback_text(item)]
    if failure_modes:
        sanitized["failure_modes"] = failure_modes

    reason = value.get("reason") or value.get("latest_reason")
    if reason and _is_method_feedback_text(reason) and _has_direct_method_evidence(sanitized):
        sanitized["reason"] = str(reason)

    avoid_rule = value.get("avoid_repeat_rule")
    if avoid_rule and _is_method_feedback_text(avoid_rule) and _has_direct_method_evidence(sanitized):
        sanitized["avoid_repeat_rule"] = str(avoid_rule)
    elif _has_direct_method_evidence(sanitized):
        rule = _method_avoid_repeat_rule(sanitized)
        if rule:
            sanitized["avoid_repeat_rule"] = rule

    avoid_rules = [str(item) for item in _sanitize_method_list(value.get("avoid_repeat_rules")) if _is_method_feedback_text(item)]
    if avoid_rules and _has_direct_method_evidence(sanitized):
        sanitized["avoid_repeat_rules"] = avoid_rules

    next_round = [str(item) for item in _sanitize_method_list(value.get("next_round_suggestions")) if _is_method_feedback_text(item)]
    if next_round and _has_direct_method_evidence(sanitized):
        sanitized["next_round_suggestions"] = next_round

    acceptance = _sanitize_method_acceptance(value.get("acceptance"))
    if acceptance and _has_direct_method_evidence(sanitized):
        sanitized["acceptance"] = acceptance

    return {key: item for key, item in sanitized.items() if item not in (None, "", [], {})} if _has_direct_method_evidence(sanitized) else None


def _sanitize_method_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    method_keys = {
        "mean",
        "datasets",
        "rsum",
        "i2t",
        "t2i",
        "similarity_time",
        "accuracy",
        "loss",
        "eval_loss",
    }
    if not any(key in value for key in method_keys):
        return {}
    return dict(value)


def _sanitize_method_proxy_screen(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed_keys = [
        "enabled",
        "status",
        "mode",
        "metrics",
        "baseline_metrics",
        "proxy_baseline",
        "proxy_delta_vs_baseline",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "proxy_worst_dataset_regression",
        "proxy_score",
        "proxy_decision_mode",
        "soft_fail",
        "soft_flags",
    ]
    sanitized = {key: value[key] for key in allowed_keys if value.get(key) not in (None, "", [], {})}
    metrics = _sanitize_method_metrics(value.get("metrics"))
    if metrics:
        sanitized["metrics"] = metrics
    for key in ["baseline_metrics", "proxy_baseline"]:
        metrics = _sanitize_method_metrics(value.get(key))
        if metrics:
            sanitized[key] = metrics
    if not _proxy_screen_has_method_evidence(sanitized):
        return {}
    reason = value.get("reason")
    if reason and _is_method_feedback_text(reason):
        sanitized["reason"] = str(reason)
    return sanitized


def _sanitize_method_failure_attribution(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ["dragging_datasets", "sample_type_failures", "mixed_gain_patterns"]:
        items = _sanitize_method_list(value.get(key))
        if items:
            sanitized[key] = items
    ablation = _sanitize_method_ablation(value.get("ablation_evidence"))
    if ablation:
        sanitized["ablation_evidence"] = ablation
    proxy_screen = _sanitize_method_proxy_screen(value.get("proxy_screen"))
    if proxy_screen:
        sanitized["proxy_screen"] = proxy_screen
    primary = value.get("primary_failure")
    if primary and sanitized and _is_method_feedback_text(primary):
        sanitized["primary_failure"] = str(primary)
    return sanitized


def _sanitize_method_ablation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed_keys = [
        "status",
        "enabled_minus_disabled_mean",
        "dataset_enabled_minus_disabled",
        "best_delta_enabled_vs_disabled",
        "best_dataset_enabled_minus_disabled",
        "delta_enabled_vs_disabled",
    ]
    sanitized = {key: value[key] for key in allowed_keys if value.get(key) not in (None, "", [], {})}
    return sanitized


def _sanitize_method_posthoc(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ["failure_modes", "next_round_suggestions", "avoid_repeat_rules"]:
        items = [str(item) for item in _sanitize_method_list(value.get(key)) if _is_method_feedback_text(item)]
        if items:
            sanitized[key] = items
    feedback_entries = [
        item
        for item in (_sanitize_method_feedback_item(entry) for entry in value.get("feedback_entries") or [])
        if item
    ]
    if feedback_entries:
        sanitized["feedback_entries"] = feedback_entries
    return sanitized if _has_method_feedback_evidence(sanitized) else {}


def _sanitize_method_acceptance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed_keys = [
        "passed",
        "delta",
        "proxy_best_mean",
        "proxy_delta",
        "proxy_score",
        "proxy_worst_dataset_regression",
        "min_delta_to_pass",
        "max_dataset_regression",
        "baseline_mean",
        "candidate_mean",
    ]
    return {key: value[key] for key in allowed_keys if value.get(key) not in (None, "", [], {})}


def _sanitize_method_direction_scorecard(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    current = value.get("current_direction") if isinstance(value.get("current_direction"), dict) else value
    summary = current.get("summary") if isinstance(current.get("summary"), dict) else {}
    s1_feedback = current.get("s1_feedback") if isinstance(current.get("s1_feedback"), dict) else {}
    allowed_summary = [
        "status",
        "attempt_count",
        "same_direction_failure_count",
        "same_direction_failure_budget",
        "best_proxy_delta",
        "positive_dataset_signal_attempts",
        "runtime_stable_attempts",
        "low_patch_risk_attempts",
        "all_dataset_collapse_attempts",
        "health_score",
        "direction_quality",
        "latest_recommended_s2_action",
    ]
    result = {
        "direction_id": current.get("direction_id") or value.get("current_direction_id"),
        "title": current.get("title"),
        "mechanism_type": current.get("mechanism_type"),
        "summary": {key: summary[key] for key in allowed_summary if summary.get(key) not in (None, "", [], {})},
        "s1_feedback": {
            key: s1_feedback[key]
            for key in ["recommendation", "conclusion", "avoid_repeat_rule"]
            if s1_feedback.get(key) not in (None, "", [], {})
        },
    }
    return {key: item for key, item in result.items() if item not in (None, "", [], {})} if result.get("summary") else {}


def _sanitize_method_proxy_calibration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("proxy_calibration"), dict):
        value = value["proxy_calibration"]
    summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
    current = value.get("current_iteration") if isinstance(value.get("current_iteration"), dict) else {}
    if not summary and not current:
        return {}
    allowed_summary = [
        "candidate_count",
        "proxy_false_positive_count",
        "proxy_false_positive_rate",
        "false_positive_reasons",
        "proxy_full_delta_correlation",
        "dataset_error_summary",
        "mechanism_false_positive_summary",
        "method_feedback",
    ]
    allowed_current = [
        "iteration",
        "acceptance_passed",
        "candidate_count",
        "proxy_false_positive_count",
        "proxy_false_positive_rate",
        "proxy_full_delta_correlation",
        "dataset_error_summary",
    ]
    result = {
        "summary": {key: summary[key] for key in allowed_summary if summary.get(key) not in (None, "", [], {})},
        "current_iteration": {key: current[key] for key in allowed_current if current.get(key) not in (None, "", [], {})},
    }
    false_positive_candidates = [
        item
        for item in value.get("false_positive_candidates") or []
        if isinstance(item, dict)
    ][:8]
    if false_positive_candidates:
        result["false_positive_candidates"] = false_positive_candidates
    if result["summary"].get("proxy_false_positive_rate", 0):
        result["avoid_repeat_rule"] = "Treat cheap proxy as suspect for mechanisms/datasets with recorded false-positive calibration."
    return {key: item for key, item in result.items() if item not in (None, "", [], {})}


def _sanitize_method_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]
    if value not in (None, "", [], {}):
        return [value]
    return []


def _sanitize_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _has_method_feedback_evidence(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if _has_direct_method_evidence(value):
        return True
    for key in ["entry", "candidate", "candidate_snapshot", "entry_snapshot", "posthoc_review"]:
        if _has_method_feedback_evidence(value.get(key)):
            return True
    for key in ["candidate_results", "feedback_entries", "entries"]:
        if any(_has_method_feedback_evidence(item) for item in value.get(key) or [] if isinstance(item, dict)):
            return True
    return False


def _has_direct_method_evidence(value: dict[str, Any]) -> bool:
    if value.get("metrics") or value.get("dataset_regressions"):
        return True
    if value.get("dragging_datasets") or value.get("sample_type_failures") or value.get("mixed_gain_patterns"):
        return True
    if value.get("ablation_evidence"):
        return True
    if value.get("direction_scorecard"):
        return True
    if value.get("proxy_calibration"):
        return True
    attribution = value.get("failure_attribution") or {}
    if isinstance(attribution, dict) and any(
        attribution.get(key) for key in ["dragging_datasets", "sample_type_failures", "mixed_gain_patterns", "ablation_evidence", "proxy_screen"]
    ):
        return True
    proxy_screen = value.get("proxy_screen") or {}
    return _proxy_screen_has_method_evidence(proxy_screen)


def _proxy_screen_has_method_evidence(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("metrics") or value.get("baseline_metrics") or value.get("proxy_baseline"):
        return True
    for key in ["proxy_dataset_deltas", "proxy_dataset_regressions"]:
        if value.get(key):
            return True
    for key in ["proxy_delta_vs_baseline", "proxy_worst_dataset_regression", "proxy_score"]:
        if value.get(key) is not None:
            return True
    return False


def _is_method_feedback_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    implementation_terms = [
        "proxy command",
        "command failure",
        "command failed",
        "runtimeerror",
        "traceback",
        "dtype",
        "bfloat",
        "float/bfloat",
        "valid_mask",
        "nameerror",
        "syntaxerror",
        "py_compile",
        "validation failed",
        "failed checks",
        "activation_check",
        "contract_failure",
        "patch_too_broad",
        "patch too broad",
        "patch risk",
        "risk_labels",
        "changed files",
        "evaluator",
        "script/evaluation",
        "no executable patch",
        "test-only",
        "only changes tests",
        "repair s2.5 patch",
        "fixing preflight",
        "checkpoint",
    ]
    return not any(term in text for term in implementation_terms)


def _method_avoid_repeat_rule(item: dict[str, Any]) -> str:
    regressions = item.get("dataset_regressions") or {}
    if regressions:
        worst_dataset = max(regressions, key=lambda key: regressions[key])
        return f"Do not repeat this method without addressing {worst_dataset} regression evidence."
    dragging = item.get("dragging_datasets") or (item.get("failure_attribution") or {}).get("dragging_datasets") or []
    if dragging and isinstance(dragging[0], dict) and dragging[0].get("dataset"):
        return f"Do not repeat this method without addressing {dragging[0].get('dataset')} regression evidence."
    proxy_screen = item.get("proxy_screen") or {}
    proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
    if proxy_deltas:
        try:
            worst_dataset = min(proxy_deltas, key=lambda key: float(proxy_deltas[key]))
            if float(proxy_deltas[worst_dataset]) < 0:
                return f"Do not repeat this method without addressing proxy regression on {worst_dataset}."
        except (TypeError, ValueError):
            pass
    ablation = item.get("ablation_evidence") or (item.get("failure_attribution") or {}).get("ablation_evidence") or {}
    if isinstance(ablation, dict) and ablation.get("status") == "no_effect":
        return "Do not repeat this method until the ablation switch changes enabled-vs-disabled behavior."
    proxy_delta = proxy_screen.get("proxy_delta_vs_baseline")
    if proxy_delta is not None:
        try:
            if float(proxy_delta) < 0:
                return "Do not repeat this method without changing the mechanism to improve proxy mean over baseline."
        except (TypeError, ValueError):
            pass
    return "Do not repeat this below-baseline method without a new mechanism and explicit ablation."


def _summarize_c2c_feedback(
    entries: list[dict[str, Any]],
    *,
    iteration_traces: list[dict[str, Any]] | None = None,
    project_id: str | None = None,
    iteration: int | None = None,
) -> dict[str, Any]:
    failed_ids: set[str] = set()
    failed_titles: set[str] = set()
    avoid_rules: set[str] = set()
    failure_modes: set[str] = set()
    next_round_suggestions: list[str] = []
    dataset_regressions: dict[str, float] = {}
    dragging_datasets: dict[str, dict[str, Any]] = {}
    sample_type_failures: set[str] = set()
    patch_risk_labels: set[str] = set()
    patch_risk_files: set[str] = set()
    mixed_gain_patterns: set[str] = set()
    latest: dict[str, Any] | None = None

    def add_failure_attribution(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for item in value.get("dragging_datasets") or []:
            if not isinstance(item, dict) or not item.get("dataset"):
                continue
            dataset = str(item["dataset"])
            existing = dragging_datasets.get(dataset, {})
            regression = _safe_feedback_float(item.get("regression"), default=0.0)
            if regression >= _safe_feedback_float(existing.get("regression"), default=float("-inf")):
                dragging_datasets[dataset] = dict(item)
        for item in value.get("sample_type_failures") or []:
            if isinstance(item, dict) and item.get("sample_family"):
                sample_type_failures.add(str(item["sample_family"]))
            elif item:
                sample_type_failures.add(str(item))
        for pattern in value.get("mixed_gain_patterns") or []:
            if pattern:
                mixed_gain_patterns.add(str(pattern))
        patch_risk = value.get("patch_risk") or {}
        for label in patch_risk.get("risk_labels") or []:
            if label:
                patch_risk_labels.add(str(label))
        for risk_file in patch_risk.get("risk_files") or []:
            if isinstance(risk_file, dict) and risk_file.get("path"):
                patch_risk_files.add(str(risk_file["path"]))
        proxy_screen = value.get("proxy_screen") or {}
        add_proxy_screen(proxy_screen)

    def add_proxy_screen(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for dataset, delta in (value.get("proxy_dataset_deltas") or {}).items():
            try:
                delta_f = float(delta)
            except (TypeError, ValueError):
                continue
            if delta_f < 0:
                existing = dragging_datasets.get(str(dataset), {})
                regression = round(abs(delta_f), 4)
                if regression >= _safe_feedback_float(existing.get("regression"), default=float("-inf")):
                    dragging_datasets[str(dataset)] = {
                        "dataset": str(dataset),
                        "sample_family": _c2c_feedback_sample_family(str(dataset)),
                        "delta": round(delta_f, 4),
                        "regression": regression,
                        "source": "proxy_screen",
                    }
                sample_type_failures.add(_c2c_feedback_sample_family(str(dataset)))
                dataset_regressions[str(dataset)] = max(dataset_regressions.get(str(dataset), float("-inf")), regression)
            elif delta_f > 0:
                mixed_gain_patterns.add("proxy_cross_dataset_tradeoff")
        command_failure = value.get("command_failure") or {}
        category = command_failure.get("category")
        if category:
            failure_modes.add(f"proxy_command_{category}")

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        current_iteration = entry.get("iteration")
        if current_iteration is not None and iteration is None:
            try:
                iteration = max(int(iteration or 0), int(current_iteration))
            except (TypeError, ValueError):
                pass
        if entry.get("avoid_repeat_rule"):
            avoid_rules.add(str(entry["avoid_repeat_rule"]))
        for rule in entry.get("avoid_repeat_rules") or []:
            if rule:
                avoid_rules.add(str(rule))
        if entry.get("idea_id") and entry.get("decision") in C2C_FAILED_DECISIONS:
            failed_ids.add(str(entry.get("idea_id")))
        if entry.get("title") and entry.get("decision") in C2C_FAILED_DECISIONS:
            failed_titles.add(str(entry.get("title")).strip().lower())
        for failed_id in entry.get("failed_idea_ids") or []:
            failed_ids.add(str(failed_id))
        for title in entry.get("failed_titles") or []:
            failed_titles.add(str(title).strip().lower())
        if entry.get("failure_mode"):
            failure_modes.add(str(entry["failure_mode"]))
        add_failure_attribution(entry.get("failure_attribution"))
        add_proxy_screen(entry.get("proxy_screen"))
        if entry.get("next_round_suggestions"):
            for suggestion in entry.get("next_round_suggestions") or []:
                if suggestion:
                    next_round_suggestions.append(str(suggestion))
        for mode in (entry.get("posthoc_review") or {}).get("failure_modes") or []:
            if mode:
                failure_modes.add(str(mode))
        for suggestion in (entry.get("posthoc_review") or {}).get("next_round_suggestions") or []:
            if suggestion:
                next_round_suggestions.append(str(suggestion))
        for rule in (entry.get("posthoc_review") or {}).get("avoid_repeat_rules") or []:
            if rule:
                avoid_rules.add(str(rule))
        for candidate in entry.get("candidate_results") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("id") or candidate.get("idea_id") or candidate.get("candidate_id")
            candidate_title = candidate.get("title")
            decision = candidate.get("decision")
            if candidate_id and decision in C2C_FAILED_DECISIONS:
                failed_ids.add(str(candidate_id))
            if candidate_title and decision in C2C_FAILED_DECISIONS:
                failed_titles.add(str(candidate_title).strip().lower())
            for rule in candidate.get("avoid_repeat_rules") or []:
                if rule:
                    avoid_rules.add(str(rule))
            if candidate.get("dataset_regressions"):
                for dataset, delta in candidate["dataset_regressions"].items():
                    try:
                        dataset_regressions[str(dataset)] = max(dataset_regressions.get(str(dataset), float("-inf")), float(delta))
                    except (TypeError, ValueError):
                        continue
            add_failure_attribution(candidate.get("failure_attribution"))
            add_proxy_screen(candidate.get("proxy_screen"))
        candidate = entry.get("candidate")
        if isinstance(candidate, dict):
            if candidate.get("id") and candidate.get("decision") in C2C_FAILED_DECISIONS:
                failed_ids.add(str(candidate["id"]))
            if candidate.get("title") and candidate.get("decision") in C2C_FAILED_DECISIONS:
                failed_titles.add(str(candidate["title"]).strip().lower())
            if candidate.get("dataset_regressions"):
                for dataset, delta in candidate["dataset_regressions"].items():
                    try:
                        dataset_regressions[str(dataset)] = max(dataset_regressions.get(str(dataset), float("-inf")), float(delta))
                    except (TypeError, ValueError):
                        continue
            add_failure_attribution(candidate.get("failure_attribution"))
            add_proxy_screen(candidate.get("proxy_screen"))
        if entry.get("dataset_regressions"):
            for dataset, delta in (entry.get("dataset_regressions") or {}).items():
                try:
                    dataset_regressions[str(dataset)] = max(dataset_regressions.get(str(dataset), float("-inf")), float(delta))
                except (TypeError, ValueError):
                    continue
        add_failure_attribution((entry.get("candidate_snapshot") or {}).get("failure_attribution"))
        add_proxy_screen((entry.get("candidate_snapshot") or {}).get("proxy_screen"))
        add_failure_attribution((entry.get("entry_snapshot") or {}).get("failure_attribution"))
        if latest is None or _feedback_sort_key(entry) >= _feedback_sort_key(latest):
            latest = entry

    iteration_traces = iteration_traces or []
    trace_summary = [
        {
            "timestamp": trace.get("timestamp"),
            "from_stage": trace.get("from_stage"),
            "to_stage": trace.get("to_stage"),
            "iteration": trace.get("iteration"),
            "reason": trace.get("reason"),
            "result_status": trace.get("result_status"),
        }
        for trace in iteration_traces[:12]
    ]
    latest_acceptance = (latest or {}).get("acceptance") or {}
    latest_posthoc = (latest or {}).get("posthoc_review") or {}
    summary_text_parts = []
    if latest:
        summary_text_parts.append(
            f"latest={latest.get('kind', 'feedback')}:{latest.get('failure_mode') or latest.get('decision') or 'n/a'}"
        )
        if latest.get("reason"):
            summary_text_parts.append(f"reason={latest.get('reason')}")
    if failed_ids:
        summary_text_parts.append(f"failed_idea_ids={','.join(sorted(failed_ids)[:4])}")
    if avoid_rules:
        summary_text_parts.append(f"avoid_repeat={sorted(avoid_rules)[0]}")
    if failure_modes:
        summary_text_parts.append(f"failure_modes={','.join(sorted(failure_modes)[:4])}")
    if dragging_datasets:
        summary_text_parts.append(f"dragging_datasets={','.join(sorted(dragging_datasets)[:4])}")
    if mixed_gain_patterns:
        summary_text_parts.append(f"mixed_patterns={','.join(sorted(mixed_gain_patterns)[:3])}")

    summary = {
        "project_id": project_id,
        "iteration": iteration,
        "entry_count": len(entries),
        "trace_count": len(iteration_traces),
        "failed_idea_ids": sorted(failed_ids),
        "failed_titles": sorted(failed_titles),
        "avoid_repeat_rules": sorted(avoid_rules),
        "blocked_idea_patterns": sorted(avoid_rules),
        "failure_modes": sorted(failure_modes),
        "next_round_suggestions": next_round_suggestions[:12],
        "dataset_regressions": {key: round(value, 4) for key, value in dataset_regressions.items()},
        "dragging_datasets": sorted(dragging_datasets.values(), key=lambda item: item.get("regression", 0.0), reverse=True),
        "sample_type_failures": sorted(sample_type_failures),
        "patch_risk_labels": sorted(patch_risk_labels),
        "patch_risk_files": sorted(patch_risk_files),
        "mixed_gain_patterns": sorted(mixed_gain_patterns),
        "latest_iteration": latest.get("iteration") if latest else None,
        "latest_idea_id": latest.get("idea_id") if latest else None,
        "latest_title": latest.get("title") if latest else None,
        "latest_reason": latest.get("reason") if latest else None,
        "latest_failure_mode": latest.get("failure_mode") if latest else None,
        "latest_decision": latest.get("decision") if latest else None,
        "latest_acceptance": latest_acceptance,
        "latest_posthoc_review": latest_posthoc,
        "iteration_traces": trace_summary,
        "summary_text": " | ".join(summary_text_parts),
    }
    if latest and latest.get("candidate_snapshot"):
        summary["latest_candidate_snapshot"] = latest.get("candidate_snapshot")
    if latest and latest.get("entry_snapshot"):
        summary["latest_entry_snapshot"] = latest.get("entry_snapshot")
    return summary


def _dedupe_c2c_feedback_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (
            entry.get("project_id"),
            entry.get("iteration"),
            entry.get("kind"),
            entry.get("source_path"),
            entry.get("idea_id"),
            entry.get("title"),
            entry.get("failure_mode"),
            entry.get("decision"),
            entry.get("reason"),
            json.dumps(entry.get("metrics") or {}, sort_keys=True, ensure_ascii=False),
            json.dumps(entry.get("acceptance") or {}, sort_keys=True, ensure_ascii=False),
            json.dumps(entry.get("proxy_calibration") or {}, sort_keys=True, ensure_ascii=False),
            json.dumps(entry.get("direction_scorecard") or {}, sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    deduped.sort(key=_feedback_sort_key)
    return deduped


def _feedback_sort_key(entry: dict[str, Any]) -> tuple[int, str, str, str]:
    iteration = entry.get("iteration")
    try:
        iteration_value = int(iteration or 0)
    except (TypeError, ValueError):
        iteration_value = 0
    timestamp = str(entry.get("timestamp") or "")
    kind = str(entry.get("kind") or "")
    title = str(entry.get("title") or entry.get("idea_id") or "")
    return iteration_value, timestamp, kind, title


def _safe_feedback_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _c2c_feedback_sample_family(dataset: str) -> str:
    mapping = {
        "mmlu-redux": "multi_domain_knowledge_reasoning",
        "ai2-arc": "science_reasoning_challenge",
        "openbookqa": "openbook_science_qa",
    }
    return mapping.get(dataset, "unknown")


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()
