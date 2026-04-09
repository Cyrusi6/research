"""Failure logging and cleanup for rejected ideas and stopped runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .utils import now_utc


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
            metrics = entry.get("metrics") or {}
            if metrics:
                lines.append(f"- Metrics: rsum={metrics.get('rsum')}, i2t_R1={(metrics.get('i2t') or {}).get('R@1')}, t2i_R1={(metrics.get('t2i') or {}).get('R@1')}")
            summary = entry.get("summary") or {}
            if summary:
                lines.append(f"- Summary: {json.dumps(summary, ensure_ascii=False)}")
            lines.append(f"- Cleanup performed: {entry.get('cleanup_performed')}")
            lines.append("")
        return "\n".join(lines)
