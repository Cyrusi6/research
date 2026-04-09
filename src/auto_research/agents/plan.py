"""Planning stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..itr_ideas import build_quick_screen_execution
from ..resources import best_matching_run
from ..resources import best_itr_execution_plan, discover_local_mm_resources
from ..utils import compact_markdown
from .base import AgentContext


class PlanAgent:
    stage_key = "S2_plan"

    def __init__(self, context: AgentContext):
        self.context = context

    def run(self) -> dict[str, Any]:
        ideas_path = self.context.project_root / "literature" / "ideas.json"
        ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
        selected = next((idea for idea in ideas if idea.get("selected")), ideas[0])
        simulate = bool(self.context.config.get("experiment", {}).get("simulate"))
        resources = discover_local_mm_resources(self.context.config)
        datasets = self._datasets_for_topic(selected["title"], resources, selected)
        baselines = self._baselines_for_topic(selected, resources)
        execution = self._execution_for_ideas(selected, ideas, resources, simulate, project_id=self.context.project_root.name)
        plan = {
            "selected_idea": selected,
            "candidate_ideas": ideas,
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": f"{selected['title']} improves the primary benchmark over the strongest baseline.",
                    "type": "superiority",
                    "metric": "primary_metric",
                    "expected_margin": ">= 1.0%",
                    "null_hypothesis": "No improvement over baseline.",
                },
                {
                    "id": "H2",
                    "statement": "The core added component drives most of the improvement.",
                    "type": "ablation",
                    "metric": "primary_metric",
                    "expected_margin": "Removal degrades performance.",
                    "null_hypothesis": "The component is not necessary.",
                },
            ],
            "baselines": baselines,
            "datasets": datasets,
            "metrics": [
                {"name": "primary_metric", "primary": True, "higher_is_better": True},
                {"name": "inference_time", "primary": False, "higher_is_better": False},
            ],
            "statistical_testing": {
                "method": "bootstrap",
                "seeds": self.context.config.get("experiment", {}).get("random_seeds", [42, 123, 456]),
                "report": "mean ± std",
                "significance_level": 0.05,
            },
            "ablation_matrix": [
                {"experiment": "w/o core module", "tests_hypothesis": "H2", "modification": "Remove the main intervention."}
            ],
            "task_graph": {
                "parallel_group_1": [
                    {"task": "baseline_eval", "gpu": 1, "estimated_hours": 2, "depends_on": []},
                    {"task": "proposed_eval", "gpu": 1, "estimated_hours": 2, "depends_on": []},
                ],
                "final": [
                    {"task": "aggregate_results", "gpu": 0, "estimated_hours": 0.5, "depends_on": ["parallel_group_1"]},
                ],
            },
            "resource_budget": {
                "total_gpu_hours": 4 if simulate else (2 if execution.get("collector") in {"laps_eval", "itr_quick_screen"} else 12),
                "peak_concurrent_gpus": 1 if execution.get("collector") in {"laps_eval", "itr_quick_screen"} else 2,
                "estimated_wall_time": "2 hours" if simulate else ("1-2 hours" if execution.get("collector") in {"laps_eval", "itr_quick_screen"} else "1 day"),
                "storage": "~5GB",
            },
            "local_resources": resources,
            "execution": execution,
        }
        plan_record = self.context.artifacts.write_yaml(
            self.stage_key,
            "plan.yaml",
            plan,
            artifact_type="plan",
            summary="Structured experiment plan",
            source_paths=["literature/ideas.json"],
        )
        hypotheses = self._hypotheses_md(plan["hypotheses"])
        hypotheses_record = self.context.artifacts.write_text(
            self.stage_key,
            "hypotheses.md",
            hypotheses,
            artifact_type="hypotheses",
            summary="Formal hypotheses",
            source_paths=[plan_record["path"]],
        )
        task_graph = self.context.artifacts.write_text(
            self.stage_key,
            "task_graph.md",
            compact_markdown(self._task_graph_md(plan["task_graph"])),
            artifact_type="task_graph",
            summary="Human-readable task graph",
            source_paths=[plan_record["path"]],
        )
        budget = self.context.artifacts.write_text(
            self.stage_key,
            "resource_budget.md",
            compact_markdown(self._budget_md(plan["resource_budget"])),
            artifact_type="resource_budget",
            summary="Resource budget",
            source_paths=[plan_record["path"]],
        )
        return {"plan": plan, "artifacts": [plan_record["path"], hypotheses_record["path"], task_graph["path"], budget["path"]]}

    @staticmethod
    def _datasets_for_topic(topic: str, resources: dict[str, Any], selected: dict[str, Any]) -> list[dict[str, Any]]:
        topic_lc = topic.lower()
        datasets = []
        available = resources.get("datasets", {})
        if selected.get("primary_codebase") == "LAPS_change" and available.get("f30k", {}).get("available"):
            return [
                {
                    "name": "Flickr30K resumed screen",
                    "domain": "image-text retrieval",
                    "size": "31K images / 5 captions each",
                    "split": "official split, 1-epoch resumed screening",
                    "reason": "Small-budget but baseline-comparable screen from a strong local checkpoint.",
                    "data_path": str(Path(resources["codebases"]["laps_change"]["root"]) / "data"),
                    "image_root": available["f30k"]["image_root"],
                },
                {
                    "name": "Flickr30K full",
                    "domain": "image-text retrieval",
                    "size": "31K images / 5 captions each",
                    "split": "official split",
                    "reason": "Follow-up confirmation benchmark after a quick-screen pass.",
                    "data_path": available["f30k"]["data_path"],
                    "image_root": available["f30k"]["image_root"],
                },
            ]
        if "image" in topic_lc and "text" in topic_lc and "retrieval" in topic_lc:
            if available.get("f30k", {}).get("available"):
                datasets.append(
                    {
                        "name": "Flickr30K",
                        "domain": "image-text retrieval",
                        "size": "31K images / 5 captions each",
                        "split": "official split",
                        "reason": "Fast single-GPU baseline and ablation benchmark.",
                        "data_path": available["f30k"]["data_path"],
                        "image_root": available["f30k"]["image_root"],
                    }
                )
            if available.get("coco", {}).get("available"):
                datasets.append(
                    {
                        "name": "MSCOCO",
                        "domain": "image-text retrieval",
                        "size": "123K images",
                        "split": "official split",
                        "reason": "Standard larger benchmark with local images and captions available.",
                        "data_path": available["coco"]["data_path"],
                        "image_root": available["coco"]["image_root"],
                    }
                )
        if datasets:
            return datasets
        return [
            {
                "name": "TBD-benchmark",
                "domain": "ML",
                "size": "TBD",
                "split": "official split",
                "reason": "Placeholder until a project-specific benchmark is locked.",
            }
        ]

    @staticmethod
    def _baselines_for_topic(selected: dict[str, Any], resources: dict[str, Any]) -> list[dict[str, Any]]:
        baselines = []
        laps_anchor = best_matching_run(resources.get("reusable_runs", []), repo_family="LAPS_change", dataset="f30k")
        if resources.get("codebases", {}).get("laps_change", {}).get("available"):
            baselines.append(
                {
                    "name": "LAPS (local code)",
                    "reason": "Available local image-text retrieval codebase with train/eval scripts and matching datasets.",
                    "implementation": "local_repo",
                    "repo_root": resources["codebases"]["laps_change"]["root"],
                }
            )
        if laps_anchor:
            baselines.append(
                {
                    "name": "LAPS_change best F30K anchor",
                    "reason": "Best locally verified LAPS_change single-model result for F30K quick-screen control.",
                    "implementation": "local_checkpoint",
                    "checkpoint": laps_anchor["model_best_path"],
                    "rsum": laps_anchor.get("rsum"),
                }
            )
        if resources.get("checkpoints", {}).get("laps_coco_vit"):
            baselines.append(
                {
                    "name": "LAPS coco_vit checkpoint",
                    "reason": "Local official checkpoint suitable for fast baseline evaluation.",
                    "implementation": "local_checkpoint",
                    "checkpoint": resources["checkpoints"]["laps_coco_vit"],
                }
            )
        for name in selected.get("key_baselines", [])[:2]:
            if len(baselines) >= 2:
                break
            baselines.append({"name": name, "reason": "Relevant baseline from literature", "implementation": "official_repo"})
        while len(baselines) < 2:
            baselines.append({"name": "Classic baseline", "reason": "Widely recognized anchor", "implementation": "official_repo"})
        return baselines

    @staticmethod
    def _execution_for_ideas(selected: dict[str, Any], ideas: list[dict[str, Any]], resources: dict[str, Any], simulate: bool, *, project_id: str) -> dict[str, Any]:
        if simulate:
            return {"mode": "simulate", "commands": [], "blocked_reason": None}
        review = selected.get("idea_review", {})
        if review.get("decision") == "accept" and not selected.get("screening_recipe"):
            return {
                "mode": "manual",
                "commands": [],
                "blocked_reason": f"Selected high-ceiling idea `{selected['id']}` requires fresh implementation before screening; do not fallback to low-ceiling quick screens.",
            }
        if selected.get("primary_codebase") == "LAPS_change" and any(item.get("screening_recipe") for item in ideas):
            return build_quick_screen_execution(ideas, resources, project_id=project_id)
        if "image" in selected.get("title", "").lower() and "text" in selected.get("title", "").lower() and "retrieval" in selected.get("title", "").lower():
            return best_itr_execution_plan(resources)
        return {"mode": "manual", "commands": [], "blocked_reason": "No executable experiment commands supplied in plan yet."}

    @staticmethod
    def _hypotheses_md(hypotheses: list[dict[str, Any]]) -> str:
        lines = ["# Hypotheses", ""]
        for item in hypotheses:
            lines.append(f"## {item['id']}")
            lines.append(f"- Statement: {item['statement']}")
            lines.append(f"- Type: {item['type']}")
            lines.append(f"- Metric: {item['metric']}")
            lines.append(f"- Null hypothesis: {item['null_hypothesis']}")
            lines.append("")
        return compact_markdown("\n".join(lines))

    @staticmethod
    def _task_graph_md(task_graph: dict[str, Any]) -> str:
        lines = ["# Task Graph", ""]
        for group, tasks in task_graph.items():
            lines.append(f"## {group}")
            for task in tasks:
                lines.append(
                    f"- {task['task']}: {task['estimated_hours']}h, gpu={task['gpu']}, depends_on={task.get('depends_on', [])}"
                )
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _budget_md(resource_budget: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Resource Budget",
                "",
                f"- Total GPU hours: {resource_budget['total_gpu_hours']}",
                f"- Peak concurrent GPUs: {resource_budget['peak_concurrent_gpus']}",
                f"- Estimated wall time: {resource_budget['estimated_wall_time']}",
                f"- Storage: {resource_budget['storage']}",
            ]
        )
