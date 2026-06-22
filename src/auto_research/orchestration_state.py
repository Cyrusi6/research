"""Machine-readable orchestration runtime state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .constants import STAGE_ORDER
from .utils import ensure_dir, now_utc, read_json, sha256_file, write_json


STATE_SCHEMA_VERSION = 1


class OrchestrationStateManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.path = project_root / "orchestration" / "state.json"

    def initialize(self, registry: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        if self.path.exists() and not force:
            return self.load()
        state = _default_state(registry)
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        return read_json(self.path, default={}) or {}

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now_utc()
        ensure_dir(self.path.parent)
        write_json(self.path, state)

    def sync_from_registry(self, registry: dict[str, Any]) -> dict[str, Any]:
        state = self.load() or _default_state(registry)
        state.update(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "project_id": registry.get("project_id"),
                "research_topic": registry.get("research_topic"),
                "status": registry.get("status"),
                "current_stage": registry.get("current_stage"),
                "iteration": registry.get("iteration"),
                "max_iterations": registry.get("max_iterations"),
                "blocked_reason": registry.get("blocked_reason"),
                "pause_type": registry.get("pause_type"),
                "resume_instruction": registry.get("resume_instruction"),
            }
        )
        state.setdefault("revision_loop", {}).setdefault("count", 0)
        stages = state.setdefault("stages", {})
        for stage_key in STAGE_ORDER:
            registry_stage = registry.get("stages", {}).get(stage_key, {})
            stage = stages.setdefault(stage_key, _default_stage_state())
            stage.update(
                {
                    "status": registry_stage.get("status", stage.get("status", "pending")),
                    "started_at": registry_stage.get("started_at"),
                    "completed_at": registry_stage.get("completed_at"),
                    "judge_retries": registry_stage.get("judge_retries", 0),
                    "judge_passed": registry_stage.get("judge_passed", False),
                    "blocked_reason": registry_stage.get("blocked_reason"),
                    "pause_type": registry_stage.get("pause_type"),
                    "resume_instruction": registry_stage.get("resume_instruction"),
                    "contract_path": f"orchestration/stage_contracts/{stage_key}.json",
                }
            )
            if registry_stage.get("artifacts"):
                stage["artifacts"] = self._artifact_records(registry_stage.get("artifacts", []))
        self.save(state)
        return state

    def run_started(self, registry: dict[str, Any]) -> None:
        state = self.sync_from_registry(registry)
        state["status"] = "running"
        state["last_event"] = {
            "type": "run_started",
            "timestamp": now_utc(),
            "stage": registry.get("current_stage"),
            "iteration": registry.get("iteration"),
            "pid": os.getpid(),
        }
        state["running"] = {
            "pid": os.getpid(),
            "stage": registry.get("current_stage"),
            "iteration": registry.get("iteration"),
            "heartbeat_at": now_utc(),
        }
        self.save(state)

    def stage_started(self, registry: dict[str, Any], stage_key: str) -> None:
        state = self.sync_from_registry(registry)
        stage = state.setdefault("stages", {}).setdefault(stage_key, _default_stage_state())
        stage["status"] = "running"
        stage["attempts"] = int(stage.get("attempts") or 0) + 1
        stage["started_at"] = stage.get("started_at") or now_utc()
        stage["last_error"] = None
        state["current_stage"] = stage_key
        state["status"] = "running"
        state["running"] = {
            "pid": os.getpid(),
            "stage": stage_key,
            "iteration": registry.get("iteration"),
            "heartbeat_at": now_utc(),
        }
        state["last_event"] = {"type": "stage_started", "timestamp": now_utc(), "stage": stage_key}
        self.save(state)

    def gate_recorded(self, registry: dict[str, Any], stage_key: str, *, passed: bool, reason: str, report: dict[str, Any] | None = None) -> None:
        state = self.sync_from_registry(registry)
        gate = {"passed": bool(passed), "reason": reason, "timestamp": now_utc()}
        if report:
            gate.update(
                {
                    "status": report.get("status"),
                    "validator": report.get("validator"),
                    "report_path": report.get("report_path"),
                    "check_count": len(report.get("checks") or []),
                }
            )
        state.setdefault("stages", {}).setdefault(stage_key, _default_stage_state())["last_gate"] = gate
        state["last_gate"] = {"stage": stage_key, **gate}
        self.save(state)

    def stage_completed(self, registry: dict[str, Any], stage_key: str, *, artifacts: list[str] | None = None) -> None:
        state = self.sync_from_registry(registry)
        stage = state.setdefault("stages", {}).setdefault(stage_key, _default_stage_state())
        stage["status"] = "completed"
        stage["completed_at"] = now_utc()
        stage["last_error"] = None
        stage.pop("pause_type", None)
        stage.pop("resume_instruction", None)
        if artifacts is not None:
            stage["artifacts"] = self._artifact_records(artifacts)
        state["last_event"] = {"type": "stage_completed", "timestamp": now_utc(), "stage": stage_key}
        state["running"] = None
        self.save(state)

    def stage_blocked(self, registry: dict[str, Any], stage_key: str, reason: str) -> None:
        self._stage_stopped(registry, stage_key, status="blocked", reason=reason)

    def stage_failed(self, registry: dict[str, Any], stage_key: str, reason: str) -> None:
        self._stage_stopped(registry, stage_key, status="failed", reason=reason)

    def stage_retryable_paused(self, registry: dict[str, Any], stage_key: str, reason: str) -> None:
        self._stage_stopped(registry, stage_key, status="retryable_paused", reason=reason)

    def judge_retry(self, registry: dict[str, Any], stage_key: str, *, retries: int, reason: str) -> None:
        state = self.sync_from_registry(registry)
        stage = state.setdefault("stages", {}).setdefault(stage_key, _default_stage_state())
        stage["judge_retries"] = retries
        stage["last_error"] = reason
        state["last_event"] = {
            "type": "judge_retry",
            "timestamp": now_utc(),
            "stage": stage_key,
            "retries": retries,
            "reason": reason,
        }
        self.save(state)

    def failure_feedback_routed(self, registry: dict[str, Any], routed: dict[str, Any]) -> None:
        state = self.sync_from_registry(registry)
        state["failure_feedback"] = {
            "last_status": routed.get("status"),
            "last_reason": routed.get("reason"),
            "last_written_iteration": routed.get("next_iteration") or registry.get("iteration"),
            "routed_to": routed.get("next_stage") or ("S1_literature" if routed.get("next_iteration") else None),
            "routed_to_s1": routed.get("status") == "routed" and bool(routed.get("next_iteration")),
            "routed_to_s2": routed.get("status") == "routed" and routed.get("next_stage") == "S2_plan",
            "timestamp": now_utc(),
        }
        state["last_event"] = {"type": "failure_feedback_routed", "timestamp": now_utc(), **routed}
        self.save(state)

    def revision_loop_incremented(self, registry: dict[str, Any], *, reason: str) -> None:
        state = self.sync_from_registry(registry)
        loop = state.setdefault("revision_loop", {"count": 0, "history": []})
        loop["count"] = int(loop.get("count") or 0) + 1
        loop.setdefault("history", []).append({"timestamp": now_utc(), "iteration": registry.get("iteration"), "reason": reason})
        state["last_event"] = {"type": "revision_loop", "timestamp": now_utc(), "reason": reason}
        self.save(state)

    def mark_completed(self, registry: dict[str, Any]) -> None:
        state = self.sync_from_registry(registry)
        state["status"] = registry.get("status", "completed")
        state["current_stage"] = registry.get("current_stage", "DONE")
        state["running"] = None
        state["last_event"] = {"type": "run_completed", "timestamp": now_utc()}
        self.save(state)

    def _stage_stopped(self, registry: dict[str, Any], stage_key: str, *, status: str, reason: str) -> None:
        state = self.sync_from_registry(registry)
        stage = state.setdefault("stages", {}).setdefault(stage_key, _default_stage_state())
        stage["status"] = status
        stage["last_error"] = reason
        stage["blocked_reason"] = reason
        if status == "retryable_paused":
            stage["pause_type"] = registry.get("pause_type")
            stage["resume_instruction"] = registry.get("resume_instruction")
        else:
            stage.pop("pause_type", None)
            stage.pop("resume_instruction", None)
        state["status"] = status
        state["blocked_reason"] = reason
        if status == "retryable_paused":
            state["pause_type"] = registry.get("pause_type")
            state["resume_instruction"] = registry.get("resume_instruction")
        else:
            state.pop("pause_type", None)
            state.pop("resume_instruction", None)
        state["running"] = None
        state["last_event"] = {"type": f"stage_{status}", "timestamp": now_utc(), "stage": stage_key, "reason": reason}
        self.save(state)

    def _artifact_records(self, artifacts: list[str]) -> list[dict[str, Any]]:
        records = []
        for artifact in artifacts:
            path = self.project_root / artifact
            record = {"path": artifact, "exists": path.exists()}
            if path.exists() and path.is_file():
                record.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size})
            records.append(record)
        return records


def _default_state(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "project_id": registry.get("project_id"),
        "research_topic": registry.get("research_topic"),
        "status": registry.get("status", "initialized"),
        "current_stage": registry.get("current_stage"),
        "iteration": registry.get("iteration", 1),
        "max_iterations": registry.get("max_iterations"),
        "blocked_reason": registry.get("blocked_reason"),
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "running": None,
        "last_event": None,
        "last_gate": None,
        "failure_feedback": {},
        "revision_loop": {"count": 0, "history": []},
        "stages": {stage_key: _default_stage_state() for stage_key in STAGE_ORDER},
    }


def _default_stage_state() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "judge_retries": 0,
        "judge_passed": False,
        "started_at": None,
        "completed_at": None,
        "last_gate": None,
        "last_error": None,
        "blocked_reason": None,
        "artifacts": [],
        "contract_path": None,
    }
