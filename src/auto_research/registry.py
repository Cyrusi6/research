"""Registry state helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import STAGE_LABELS, STAGE_ORDER
from .utils import now_utc, read_yaml, write_yaml


def default_registry(*, project_id: str, topic: str, config: dict[str, Any]) -> dict[str, Any]:
    max_iterations = config.get("review", {}).get("max_iterations", 5)
    stages = {}
    for stage_key in STAGE_ORDER:
        stages[stage_key] = {
            "label": STAGE_LABELS[stage_key],
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "judge_passed": False,
            "judge_retries": 0,
            "artifacts": [],
            "blocked_reason": None,
        }
    stages["S5_review"]["decision"] = None
    stages["S5_review"]["weighted_score"] = None
    stages["S5_review"]["revision_dispatch"] = None
    return {
        "project_id": project_id,
        "research_topic": topic,
        "current_stage": STAGE_ORDER[0],
        "iteration": 1,
        "max_iterations": max_iterations,
        "status": "initialized",
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "references_dir": "references/papers",
        "artifacts_summary": {},
        "blocked_reason": None,
        "last_run_id": None,
        "invalidated_by": [],
        "stages": stages,
    }


def load_registry(path: Path) -> dict[str, Any]:
    return read_yaml(path)


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    payload = deepcopy(registry)
    payload["updated_at"] = now_utc()
    write_yaml(path, payload)


def stage_index(stage_key: str) -> int:
    return STAGE_ORDER.index(stage_key)


def begin_stage(registry: dict[str, Any], stage_key: str) -> None:
    stage = registry["stages"][stage_key]
    stage["status"] = "running"
    stage["started_at"] = stage["started_at"] or now_utc()
    stage["blocked_reason"] = None
    registry["current_stage"] = stage_key
    registry["status"] = "running"


def complete_stage(registry: dict[str, Any], stage_key: str, *, artifacts: list[str] | None = None) -> None:
    stage = registry["stages"][stage_key]
    stage["status"] = "completed"
    stage["completed_at"] = now_utc()
    stage["judge_passed"] = True
    stage["blocked_reason"] = None
    if artifacts is not None:
        stage["artifacts"] = artifacts
        registry.setdefault("artifacts_summary", {})[stage_key] = len(artifacts)
    next_idx = stage_index(stage_key) + 1
    registry["current_stage"] = STAGE_ORDER[next_idx] if next_idx < len(STAGE_ORDER) else "DONE"
    if registry["current_stage"] == "DONE":
        registry["status"] = "completed"


def fail_stage(registry: dict[str, Any], stage_key: str, reason: str) -> None:
    stage = registry["stages"][stage_key]
    stage["status"] = "failed"
    stage["blocked_reason"] = reason
    registry["status"] = "failed"
    registry["blocked_reason"] = reason
    registry["current_stage"] = stage_key


def block_stage(registry: dict[str, Any], stage_key: str, reason: str) -> None:
    stage = registry["stages"][stage_key]
    stage["status"] = "blocked"
    stage["blocked_reason"] = reason
    registry["status"] = "blocked"
    registry["blocked_reason"] = reason
    registry["current_stage"] = stage_key


def increment_judge_retry(registry: dict[str, Any], stage_key: str) -> int:
    stage = registry["stages"][stage_key]
    stage["judge_retries"] += 1
    return stage["judge_retries"]


def set_review_outcome(registry: dict[str, Any], *, decision: str, score: float, revision_dispatch: str | None) -> None:
    review_stage = registry["stages"]["S5_review"]
    review_stage["decision"] = decision
    review_stage["weighted_score"] = score
    review_stage["revision_dispatch"] = revision_dispatch


def invalidate_from(registry: dict[str, Any], stage_key: str, *, invalidated_by: str) -> None:
    start = stage_index(stage_key)
    for idx in range(start, len(STAGE_ORDER)):
        current = STAGE_ORDER[idx]
        registry["stages"][current]["status"] = "pending"
        registry["stages"][current]["judge_passed"] = False
        registry["stages"][current]["completed_at"] = None
    registry["current_stage"] = stage_key
    registry["status"] = "running"
    registry.setdefault("invalidated_by", []).append(invalidated_by)
