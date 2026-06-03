"""Planning stage."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..adapters.runner import ExperimentRunner
from ..c2c import C2CAdapter, c2c_idea_novelty_report, c2c_implementation_scope_report, is_c2c_project, normalize_c2c_mechanism_fields
from ..code_patch import CodePatchAgent
from ..failure_log import load_c2c_feedback_bundle
from ..itr_ideas import build_quick_screen_execution
from ..resources import best_matching_run
from ..resources import best_itr_execution_plan, discover_local_mm_resources
from ..utils import compact_markdown, ensure_dir, now_utc, read_json, read_yaml, write_yaml
from .base import AgentContext


class PlanAgent:
    stage_key = "S2_plan"

    def __init__(self, context: AgentContext):
        self.context = context
        self.runner = ExperimentRunner(context.config)

    def run(self) -> dict[str, Any]:
        if is_c2c_project(self.context.config):
            return self._run_c2c_plan()
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

    def _run_c2c_plan(self) -> dict[str, Any]:
        adapter = C2CAdapter(self.context.project_root, self.context.config)
        ideas_path = self.context.project_root / "literature" / "ideas.json"
        s1_ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
        s1_selected = next((idea for idea in s1_ideas if idea.get("selected")), s1_ideas[0])
        baseline_path = self.context.project_root / "literature" / "c2c" / "baseline_evidence.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else adapter.baseline
        concern_path = self.context.project_root / "literature" / "c2c" / "rebuttal_concern_matrix.json"
        concern_matrix = json.loads(concern_path.read_text(encoding="utf-8")) if concern_path.exists() else {}
        negative_path = self.context.project_root / "literature" / "c2c" / "negative_result_memory.json"
        negative_memory = json.loads(negative_path.read_text(encoding="utf-8")) if negative_path.exists() else {}
        small_loop_cfg = self.context.config.get("c2c", {}).get("small_loop", {})
        feedback = self._load_c2c_plan_feedback()
        planner_memory = self._load_c2c_s2_planner_memory()
        planning_result = self._c2c_directional_planner(
            selected=s1_selected,
            s1_ideas=s1_ideas,
            baseline=baseline,
            concern_matrix=concern_matrix,
            negative_memory=negative_memory,
            feedback=feedback,
            planner_memory=planner_memory,
            adapter=adapter,
        )
        ideas = planning_result["ideas"]
        selected = next((idea for idea in ideas if idea.get("selected")), ideas[0])
        gpu_policy = dict(self.context.config.get("experiment", {}).get("gpu_policy", {}))
        gpu_policy.setdefault("max_gpus", 6)
        gpu_policy.setdefault("min_free_mb", 0)
        gpu_policy.setdefault("gpu_ids", small_loop_cfg.get("gpu_ids", "auto"))
        gpu_selection = self.runner.select_gpus(gpu_policy)
        selected_gpu_ids = gpu_selection.selected_ids
        actual_train_gpus = len(selected_gpu_ids)
        min_delta = float(small_loop_cfg.get("min_delta_to_pass", 0.1))
        max_regression = float(small_loop_cfg.get("max_dataset_regression", 2.0))
        datasets = [
            {
                "name": name,
                "domain": "LLM benchmark",
                "split": "C2C unified_evaluator configured split",
                "metric": "overall_accuracy",
                "reason": "Part of the configured original C2C three-dataset small-loop protocol.",
            }
            for name in self.context.config.get("c2c", {}).get("datasets", ["mmlu-redux", "ai2-arc", "openbookqa"])
        ]
        short_loop = {
            "collector": "c2c_small_loop",
            "mode": "small_loop",
            "workdir": str(adapter.repo_root),
            "python": adapter.env_python,
            "baseline": baseline,
            "allowed_files": adapter.allowed_files,
            "allowed_prefixes": adapter.allowed_prefixes,
            "train_samples": self.context.config.get("c2c", {}).get("small_loop", {}).get("train_samples", 2048),
            "eval_datasets": [item["name"] for item in datasets],
            "max_candidates": small_loop_cfg.get("max_candidates", 3),
            "min_delta_to_pass": min_delta,
            "max_dataset_regression": max_regression,
            "patch_source": "llm_json_guarded_edits",
            "gpu_policy": gpu_selection.policy,
            "selected_gpu_ids": selected_gpu_ids,
            "num_train_processes": actual_train_gpus,
            "eval_max_gpus": actual_train_gpus,
            "resource_snapshot": gpu_selection.snapshot,
            "acceptance_rule": (
                f"best_candidate.mean >= baseline.mean + {min_delta} and no dataset regresses more than {max_regression} points"
            ),
            "reviewer_concerns": concern_matrix.get("top_concerns", []),
            "blocked_patterns": (negative_memory.get("blocked_idea_patterns") or [])[:8],
            "failure_feedback_count": len(feedback),
        }
        plan = {
            "selected_idea": selected,
            "candidate_ideas": ideas,
            "hypotheses": [
                {
                    "id": "H1",
                    "statement": f"{selected['title']} improves the C2C three-dataset mean over {baseline.get('name')}.",
                    "type": "superiority",
                    "metric": "three_dataset_mean",
                    "expected_margin": f">= +{min_delta} over configured baseline mean={baseline.get('mean')}",
                    "null_hypothesis": "The candidate does not improve the mean benchmark score.",
                },
                {
                    "id": "H2",
                    "statement": "The gain comes from the selected mechanism and disappears or weakens when its ablation switch is disabled.",
                    "type": "ablation",
                    "metric": "mechanism_signature_and_dataset_accuracy_breakdown",
                    "expected_margin": f"Mechanism-enabled proxy/full results beat the disabled variant while no dataset regresses by more than {max_regression} points.",
                    "null_hypothesis": "Any score movement is unrelated to the proposed mechanism and survives disabling it.",
                },
            ],
            "baselines": [
                {
                    "name": baseline.get("name", "paper_original_rosetta_fuser"),
                    "reason": "Configured original C2C/Rosetta fuser baseline for effect-first discovery.",
                    "implementation": "original_baseline_snapshot",
                    "mean": baseline.get("mean"),
                    "datasets": baseline.get("datasets"),
                    "source": baseline.get("source"),
                },
                {
                    "name": "candidate ablation-off control",
                    "reason": "Disables only the proposed mechanism while keeping the configured original C2C protocol.",
                    "implementation": "generated_by_candidate_ablation_switch",
                },
            ],
            "datasets": datasets,
            "metrics": [
                {"name": "three_dataset_mean", "primary": True, "higher_is_better": True},
                {"name": "mmlu-redux_overall_accuracy", "primary": False, "higher_is_better": True},
                {"name": "ai2-arc_overall_accuracy", "primary": False, "higher_is_better": True},
                {"name": "openbookqa_overall_accuracy", "primary": False, "higher_is_better": True},
            ],
            "statistical_testing": {
                "method": "fixed-protocol comparison",
                "seeds": self.context.config.get("experiment", {}).get("random_seeds", [42, 123, 456]),
                "report": "three-dataset mean and per-dataset scores",
                "significance_level": None,
            },
            "acceptance_criteria": {
                "minimum_mean_delta": min_delta,
                "max_dataset_regression": max_regression,
                "primary_baseline": baseline.get("name", "paper_original_rosetta_fuser"),
                "must_emit": ["main_results.json", "ablation_results.json", "hypothesis_verification.md", "c2c_small_loop_results.json"],
                "forbidden_shortcut": "Do not accept candidates whose gain is explained only by adding another hard accept/reject gate or reducing transfer coverage.",
                "coverage_diagnostics_required": True,
                "matched_coverage_ablation_required": True,
            },
            "reviewer_risk_controls": {
                "top_concerns": concern_matrix.get("top_concerns", []),
                "blocked_idea_patterns": (negative_memory.get("blocked_idea_patterns") or [])[:8],
            },
            "implementation_scope": {
                "selected_scope": selected.get("implementation_scope"),
                "implementation_plan": selected.get("implementation_plan"),
                "scope_gate": selected.get("implementation_scope_gate"),
            },
            "directional_planning": planning_result["metadata"],
            "ablation_matrix": [
                {
                    "experiment": "disable selected mechanism",
                    "tests_hypothesis": "H2",
                    "modification": "Use the candidate ablation_switch to fall back to configured original C2C behavior.",
                    "switch": (selected.get("experiment_contract") or {}).get("ablation_switch")
                    or (selected.get("ablation_plan") or {}).get("switch"),
                    "expected_signature": selected.get("expected_signature"),
                    "coverage_diagnostics": selected.get("coverage_diagnostics"),
                },
                {
                    "experiment": "matched transfer coverage control",
                    "tests_hypothesis": "H2",
                    "modification": "Match baseline/control transfer coverage to candidate coverage before comparing mechanism benefit.",
                    "matched_coverage_ablation": selected.get("matched_coverage_ablation"),
                    "required_stats": (selected.get("coverage_diagnostics") or {}).get("stats"),
                }
            ],
            "task_graph": {
                "candidate_loop": [
                    {"task": "guarded_patch_generation", "gpu": 0, "estimated_hours": 0.1, "depends_on": []},
                    {"task": "preflight_compile_and_tests", "gpu": 0, "estimated_hours": 0.1, "depends_on": ["guarded_patch_generation"]},
                    {"task": "small2048_train", "gpu": actual_train_gpus, "estimated_hours": 2, "depends_on": ["preflight_compile_and_tests"]},
                    {"task": "three_dataset_eval", "gpu": actual_train_gpus, "estimated_hours": 3, "depends_on": ["small2048_train"]},
                ],
                "final": [],
            },
            "resource_budget": {
                "total_gpu_hours": 15 * actual_train_gpus,
                "peak_concurrent_gpus": actual_train_gpus,
                "estimated_wall_time": "small2048 training plus three evals per candidate",
                "storage": "C2C snapshot plus generated local/auto_research_runs artifacts",
                "gpu_policy": gpu_selection.policy,
                "selected_gpu_ids": selected_gpu_ids,
                "resource_snapshot": gpu_selection.snapshot,
            },
            "execution": short_loop,
        }
        code_patch_manifest = CodePatchAgent(self.context.project_root, self.context.config, self.context.artifacts).run(plan, ideas)
        plan["candidate_ideas"] = ideas
        plan["code_patch_manifest"] = {
            "status": code_patch_manifest.get("status"),
            "path": "plan/code_patches/patch_manifest.json" if code_patch_manifest.get("status") != "disabled" else "",
        }
        short_loop["patch_source"] = "s2_5_frozen_codex_patch" if code_patch_manifest.get("status") != "disabled" else "config_overrides_only"
        memory_record = self._append_c2c_s2_planner_memory(
            planning_result=planning_result,
            s1_selected=s1_selected,
            selected=selected,
            ideas=ideas,
            feedback=feedback,
            code_patch_manifest=code_patch_manifest,
        )
        planning_result["metadata"]["memory_path"] = "plan/s2_planner_memory.json"
        planning_result["metadata"]["memory_entry_count"] = memory_record["entry_count"]
        plan_record = self.context.artifacts.write_yaml(
            self.stage_key,
            "plan.yaml",
            plan,
            artifact_type="plan",
            summary="C2C structured small-loop experiment plan",
            source_paths=["literature/ideas.json", "literature/c2c/baseline_evidence.json"],
        )
        candidate_record = self.context.artifacts.write_json(
            self.stage_key,
            "candidate_ideas.json",
            ideas,
            artifact_type="c2c_candidate_ideas",
            summary="C2C candidate ideas with bounded edit scope",
            source_paths=[plan_record["path"]],
        )
        short_loop_record = self.context.artifacts.write_yaml(
            self.stage_key,
            "short_loop_plan.yaml",
            short_loop,
            artifact_type="c2c_short_loop_plan",
            summary="C2C small-loop execution contract",
            source_paths=[plan_record["path"]],
        )
        feedback_record = self.context.artifacts.write_json(
            self.stage_key,
            "plan_feedback.json",
            {
                "feedback": feedback,
                "directional_planning": planning_result["metadata"],
                "s2_planner_memory": {
                    "path": "plan/s2_planner_memory.json",
                    "entry_count": memory_record["entry_count"],
                },
                "selected_gpu_ids": selected_gpu_ids,
                "gpu_policy": gpu_selection.policy,
            },
            artifact_type="c2c_plan_feedback",
            summary="C2C failure feedback and resource selection used by S2",
            source_paths=[plan_record["path"]],
        )
        hypotheses_record = self.context.artifacts.write_text(
            self.stage_key,
            "hypotheses.md",
            self._hypotheses_md(plan["hypotheses"]),
            artifact_type="hypotheses",
            summary="C2C formal hypotheses",
            source_paths=[plan_record["path"]],
        )
        task_graph_record = self.context.artifacts.write_text(
            self.stage_key,
            "task_graph.md",
            compact_markdown(self._task_graph_md(plan["task_graph"])),
            artifact_type="task_graph",
            summary="C2C task graph",
            source_paths=[plan_record["path"]],
        )
        budget_record = self.context.artifacts.write_text(
            self.stage_key,
            "resource_budget.md",
            compact_markdown(self._budget_md(plan["resource_budget"])),
            artifact_type="resource_budget",
            summary="C2C resource budget",
            source_paths=[plan_record["path"]],
        )
        return {
            "plan": plan,
            "artifacts": [
                plan_record["path"],
                *code_patch_manifest.get("artifacts", []),
                memory_record["artifact"]["path"],
                candidate_record["path"],
                short_loop_record["path"],
                feedback_record["path"],
                hypotheses_record["path"],
                task_graph_record["path"],
                budget_record["path"],
            ],
        }

    def _c2c_directional_planner(
        self,
        *,
        selected: dict[str, Any],
        s1_ideas: list[dict[str, Any]],
        baseline: dict[str, Any],
        concern_matrix: dict[str, Any],
        negative_memory: dict[str, Any],
        feedback: list[dict[str, Any]],
        planner_memory: dict[str, Any],
        adapter: C2CAdapter,
    ) -> dict[str, Any]:
        cfg = (self.context.config.get("agents", {}).get("s2_directional_planner") or {})
        enabled = bool(cfg.get("enabled", self.context.config.get("c2c", {}).get("s2_directional_planner", {}).get("enabled", True)))
        fallback = self._normalize_c2c_plan_ideas(s1_ideas, baseline=baseline, selected_id=selected.get("id"))
        if not enabled:
            return {
                "ideas": fallback,
                "metadata": {
                    "enabled": False,
                    "status": "disabled",
                    "source": "s1_ideas",
                    "candidate_count": len(fallback),
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                },
            }
        resume_result = self._run_c2c_s2_resume_planner(
            selected=selected,
            s1_ideas=s1_ideas,
            baseline=baseline,
            concern_matrix=concern_matrix,
            negative_memory=negative_memory,
            feedback=feedback,
            planner_memory=planner_memory,
            adapter=adapter,
            max_candidates=int(cfg.get("max_candidates") or self.context.config.get("c2c", {}).get("small_loop", {}).get("max_candidates") or 3),
        )
        if resume_result.get("status") == "ok":
            return resume_result
        if not getattr(self.context.llm, "use_real_api", False):
            return {
                "ideas": fallback,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_no_real_llm",
                    "source": "s1_ideas",
                    "candidate_count": len(fallback),
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                    "resume_planner": resume_result.get("metadata") if isinstance(resume_result.get("metadata"), dict) else None,
                },
            }
        max_candidates = int(cfg.get("max_candidates") or self.context.config.get("c2c", {}).get("small_loop", {}).get("max_candidates") or 3)
        prompt = {
            "objective": "Generate concrete direction-conditioned S2 experiment candidates, not a new S1 direction.",
            "s1_selected_direction": _compact_for_plan_prompt(selected, 5000),
            "s1_candidate_pool": _compact_for_plan_prompt(s1_ideas, 6000),
            "baseline": baseline,
            "allowed_files": adapter.allowed_files,
            "allowed_prefixes": adapter.allowed_prefixes,
            "reviewer_concerns": (concern_matrix.get("top_concerns") or [])[:8] if isinstance(concern_matrix, dict) else [],
            "negative_memory": _compact_for_plan_prompt(negative_memory, 4000),
            "s2_planner_memory": _compact_for_plan_prompt(_c2c_s2_memory_for_prompt(planner_memory), int(cfg.get("memory_prompt_chars") or 6000)),
            "available_artifacts": _c2c_s2_available_artifacts(),
            "failure_feedback": _compact_for_plan_prompt(feedback, 9000),
            "requirements": [
                "Stay inside the S1 selected mechanism direction unless performance_feedback explicitly says return_to_s1_new_direction.",
                "If performance_feedback.summary.recommended_s2_action is present, follow it when choosing patch_repair, mechanism_repair, or new_same_direction_variant.",
                "If this is a same-direction repair, create 2-3 concrete variants that differ in integration strategy or mechanism strength.",
                "Prefer diversity across the five same-direction attempts, but treat diversity as a soft search heuristic rather than a hard gate.",
                "Use s2_planner_memory first to avoid recreating variants already tried in this S1 direction.",
                "If s2_planner_memory is insufficient, use failure_feedback and cite available_artifacts paths in failure_feedback_refs.",
                "Do not change evaluator, datasets, baseline, or cheap proxy protocol.",
                "Do not propose pure threshold/top-k/fallback tuning.",
                "Each candidate must include experiment_contract.config_overrides, ablation_switch, expected_files, implementation_plan, expected_signature, and failure_avoidance.",
                "Use performance feedback as performance evidence; ignore low-level S2.5 implementation errors unless they imply a method-level risk.",
            ],
            "return_shape": {
                "planner_summary": "short explanation",
                "planning_mode": "same_direction_variant|new_direction_after_budget|fallback",
                "candidates": [
                    {
                        "id": "snake_case_id",
                        "title": "short title",
                        "description": "mechanism variant",
                        "hypothesis": "proxy-testable hypothesis",
                        "mechanism_type": "same mechanism type as S1 direction when same-direction",
                        "experiment_contract": {},
                        "implementation_plan": {},
                        "failure_feedback_refs": [],
                        "failure_avoidance": [],
                        "selected": True,
                    }
                ],
            },
        }
        try:
            payload = self.context.llm.generate_json_with_schema(
                instructions=(
                    "You are the S2 experiment planner for an automated C2C research loop. "
                    "S1 sets the mechanism direction; your job is to turn that direction plus real performance feedback into executable candidates for S2.5. "
                    "Be concrete, code-aware, and effect-first. Return JSON only."
                ),
                prompt=json.dumps(prompt, ensure_ascii=False),
                default={},
                schema={"type": "object", "required": ["candidates"]},
                agent_name="c2c-s2-directional-planner",
            )
        except Exception as exc:
            return {
                "ideas": fallback,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_llm_error",
                    "source": "s1_ideas",
                    "reason": str(exc)[-500:],
                    "candidate_count": len(fallback),
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                },
            }
        planned = payload.get("candidates") if isinstance(payload, dict) else None
        normalized = self._normalize_c2c_plan_ideas(
            planned if isinstance(planned, list) else [],
            baseline=baseline,
            selected_id=None,
            s1_selected=selected,
            max_candidates=max_candidates,
        )
        accepted = [idea for idea in normalized if self._c2c_plan_candidate_acceptable(idea, selected)]
        if not accepted:
            return {
                "ideas": fallback,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_invalid_planner_output",
                    "source": "s1_ideas",
                    "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                    "rejected_count": len(normalized),
                    "candidate_count": len(fallback),
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                },
            }
        for idx, idea in enumerate(accepted):
            idea["selected"] = idx == 0
            idea.setdefault("s2_planner", {})
            idea["s2_planner"].update(
                {
                    "source": "directional_planner",
                    "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                    "s1_direction_id": selected.get("id"),
                    "s1_mechanism_type": selected.get("mechanism_type"),
                }
            )
        return {
            "ideas": accepted,
            "metadata": {
                "enabled": True,
                "status": "ok",
                "source": "directional_planner",
                "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                "candidate_count": len(accepted),
                "s1_direction_id": selected.get("id"),
                "memory_entry_count": len(planner_memory.get("entries") or []),
                "resume_planner": resume_result.get("metadata") if isinstance(resume_result.get("metadata"), dict) else None,
            },
        }

    def _run_c2c_s2_resume_planner(
        self,
        *,
        selected: dict[str, Any],
        s1_ideas: list[dict[str, Any]],
        baseline: dict[str, Any],
        concern_matrix: dict[str, Any],
        negative_memory: dict[str, Any],
        feedback: list[dict[str, Any]],
        planner_memory: dict[str, Any],
        adapter: C2CAdapter,
        max_candidates: int,
    ) -> dict[str, Any]:
        cfg = self.context.config.get("agents", {}).get("s2_directional_planner") or {}
        resume_enabled = bool(cfg.get("resume_enabled", self.context.config.get("c2c", {}).get("s2_directional_planner", {}).get("resume_enabled", True)))
        if not resume_enabled:
            return {"status": "disabled", "metadata": {"enabled": False, "reason": "resume planner disabled"}}
        if not shutil.which("codex"):
            return {"status": "unavailable", "metadata": {"enabled": True, "reason": "codex executable not found"}}
        llm_cfg = self.context.config.get("llm", {}) or {}
        codex_cfg = llm_cfg.get("codex_cli") or {}
        if codex_cfg.get("use_resume", True) is False:
            return {"status": "disabled", "metadata": {"enabled": False, "reason": "llm.codex_cli.use_resume is false"}}
        session_key = f"s2_planner:{selected.get('id') or selected.get('mechanism_type') or 'direction'}"
        session_record = _load_s2_codex_session_record(self.context.project_root, session_key)
        session_id = _session_id_from_record(session_record)
        prompt = _c2c_s2_resume_planner_prompt(
            selected=selected,
            s1_ideas=s1_ideas,
            baseline=baseline,
            concern_matrix=concern_matrix,
            negative_memory=negative_memory,
            feedback=feedback,
            planner_memory=planner_memory,
            adapter=adapter,
            max_candidates=max_candidates,
        )
        result = _run_s2_codex_json_planner(
            project_root=self.context.project_root,
            config=self.context.config,
            session_key=session_key,
            session_id=session_id,
            prompt=prompt,
        )
        metadata = {
            "enabled": True,
            "status": result.get("status"),
            "source": "codex_resume_planner",
            "session_key": session_key,
            "session_id": result.get("session_id") or session_id,
            "used_existing_session": bool(session_id),
            "reason": result.get("reason"),
            "memory_entry_count": len(planner_memory.get("entries") or []),
        }
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        planned = payload.get("candidates") if isinstance(payload, dict) else None
        normalized = self._normalize_c2c_plan_ideas(
            planned if isinstance(planned, list) else [],
            baseline=baseline,
            selected_id=None,
            s1_selected=selected,
            max_candidates=max_candidates,
        )
        accepted = [idea for idea in normalized if self._c2c_plan_candidate_acceptable(idea, selected)]
        if not accepted:
            health = _record_s2_planner_session_health(
                project_root=self.context.project_root,
                session_key=session_key,
                config=self.context.config,
                session_id=result.get("session_id") or session_id,
                status="invalid",
                candidate_ids=_s2_planner_candidate_ids(normalized),
                duplicate=False,
                reason="resume planner returned no acceptable candidates",
            )
            metadata["accepted_count"] = 0
            metadata["rejected_count"] = len(normalized)
            metadata["planner_summary"] = payload.get("planner_summary") if isinstance(payload, dict) else None
            metadata["session_health"] = health.get("health")
            metadata["session_reset"] = health.get("session_reset")
            metadata["session_reset_reason"] = health.get("session_reset_reason")
            return {"status": "invalid", "metadata": metadata}
        candidate_ids = _s2_planner_candidate_ids(accepted)
        duplicate_output = _s2_planner_duplicate_candidate_ids(candidate_ids, planner_memory, session_record)
        health = _record_s2_planner_session_health(
            project_root=self.context.project_root,
            session_key=session_key,
            config=self.context.config,
            session_id=result.get("session_id") or session_id,
            status="duplicate" if duplicate_output else "ok",
            candidate_ids=candidate_ids,
            duplicate=duplicate_output,
            reason="resume planner repeated candidate ids already present in memory" if duplicate_output else "resume planner produced acceptable candidates",
        )
        if duplicate_output and health.get("session_reset"):
            metadata.update(
                {
                    "accepted_count": len(accepted),
                    "duplicate_candidate_ids": candidate_ids,
                    "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                    "session_health": health.get("health"),
                    "session_reset": True,
                    "session_reset_reason": health.get("session_reset_reason"),
                }
            )
            return {"status": "invalid", "metadata": metadata}
        for idx, idea in enumerate(accepted):
            idea["selected"] = idx == 0
            idea.setdefault("s2_planner", {})
            idea["s2_planner"].update(
                {
                    "source": "codex_resume_planner",
                    "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                    "s1_direction_id": selected.get("id"),
                    "s1_mechanism_type": selected.get("mechanism_type"),
                    "session_key": session_key,
                }
            )
        metadata.update(
            {
                "status": "ok",
                "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                "candidate_count": len(accepted),
                "s1_direction_id": selected.get("id"),
                "duplicate_output": duplicate_output,
                "session_health": health.get("health"),
                "session_reset": health.get("session_reset"),
            }
        )
        return {"status": "ok", "ideas": accepted, "metadata": metadata}

    def _load_c2c_s2_planner_memory(self) -> dict[str, Any]:
        memory = read_json(
            self.context.project_root / "plan" / "s2_planner_memory.json",
            default={
                "schema_version": "c2c_s2_planner_memory_v1",
                "project_id": self.context.project_root.name,
                "entries": [],
                "compacted_summary": {},
            },
        )
        if not isinstance(memory, dict):
            return {"schema_version": "c2c_s2_planner_memory_v1", "project_id": self.context.project_root.name, "entries": [], "compacted_summary": {}}
        memory.setdefault("schema_version", "c2c_s2_planner_memory_v1")
        memory.setdefault("project_id", self.context.project_root.name)
        memory.setdefault("entries", [])
        memory.setdefault("compacted_summary", {})
        return memory

    def _append_c2c_s2_planner_memory(
        self,
        *,
        planning_result: dict[str, Any],
        s1_selected: dict[str, Any],
        selected: dict[str, Any],
        ideas: list[dict[str, Any]],
        feedback: list[dict[str, Any]],
        code_patch_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = (self.context.config.get("agents", {}).get("s2_directional_planner") or {})
        max_entries = max(1, int(cfg.get("memory_max_entries") or 12))
        memory = self._load_c2c_s2_planner_memory()
        entries = [item for item in memory.get("entries") or [] if isinstance(item, dict)]
        metadata = planning_result.get("metadata") if isinstance(planning_result.get("metadata"), dict) else {}
        entry = {
            "timestamp": now_utc(),
            "iteration": _registry_iteration(self.context.project_root),
            "s1_direction": {
                "id": s1_selected.get("id"),
                "title": s1_selected.get("title"),
                "mechanism_type": s1_selected.get("mechanism_type"),
            },
            "planning": {
                "status": metadata.get("status"),
                "source": metadata.get("source"),
                "planning_mode": metadata.get("planning_mode"),
                "planner_summary": metadata.get("planner_summary"),
            },
            "selected_candidate": _compact_s2_memory_idea(selected),
            "candidate_ids": [idea.get("id") for idea in ideas if isinstance(idea, dict) and idea.get("id")],
            "candidate_summaries": [_compact_s2_memory_idea(idea) for idea in ideas[:5] if isinstance(idea, dict)],
            "feedback_digest": _c2c_s2_feedback_digest(feedback),
            "patch_manifest": {
                "status": code_patch_manifest.get("status"),
                "valid_patch_count": code_patch_manifest.get("valid_patch_count"),
                "retryable_patch_count": code_patch_manifest.get("retryable_patch_count"),
                "candidate_count": code_patch_manifest.get("candidate_count"),
            },
            "available_artifacts": _c2c_s2_available_artifacts(),
        }
        entries.append(entry)
        entries = entries[-max_entries:]
        memory = {
            "schema_version": "c2c_s2_planner_memory_v1",
            "project_id": self.context.project_root.name,
            "updated_at": now_utc(),
            "entry_count": len(entries),
            "entries": entries,
            "compacted_summary": _compact_c2c_s2_memory_summary(entries),
        }
        record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner_memory.json",
            memory,
            artifact_type="c2c_s2_planner_memory",
            summary="Compact memory for direction-conditioned S2 planning",
            source_paths=["plan/performance_feedback.json", "experiment/results/failure_feedback.json", "experiment/results/main_results.json"],
        )
        return {"artifact": record, "entry_count": len(entries), "memory": memory}

    @staticmethod
    def _normalize_c2c_plan_ideas(
        ideas: list[dict[str, Any]],
        *,
        baseline: dict[str, Any],
        selected_id: str | None = None,
        s1_selected: dict[str, Any] | None = None,
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        normalized = []
        base_contract = (s1_selected or {}).get("experiment_contract") if isinstance((s1_selected or {}).get("experiment_contract"), dict) else {}
        base_impl = (s1_selected or {}).get("implementation_plan") if isinstance((s1_selected or {}).get("implementation_plan"), dict) else {}
        limit = max_candidates if max_candidates is not None else len(ideas)
        for idx, idea in enumerate(ideas[: max(0, limit)]):
            if not isinstance(idea, dict):
                continue
            item = dict(idea)
            if s1_selected:
                item.setdefault("mechanism_type", s1_selected.get("mechanism_type"))
                item.setdefault("paper_claim", s1_selected.get("paper_claim"))
                item.setdefault("why_baseline_fails", s1_selected.get("why_baseline_fails"))
                item.setdefault("coverage_diagnostics", s1_selected.get("coverage_diagnostics"))
                item.setdefault("matched_coverage_ablation", s1_selected.get("matched_coverage_ablation"))
                item.setdefault("expected_files", s1_selected.get("expected_files"))
                item.setdefault("evidence_refs", s1_selected.get("evidence_refs"))
                item.setdefault("counterevidence_refs", s1_selected.get("counterevidence_refs"))
                item.setdefault("code_refs", s1_selected.get("code_refs"))
                item.setdefault("s1_allowed_variants", s1_selected.get("s1_allowed_variants"))
                item.setdefault("s1_forbidden_patterns", s1_selected.get("s1_forbidden_patterns"))
                item.setdefault("s1_direction_id", s1_selected.get("s1_direction_id") or s1_selected.get("direction_id") or s1_selected.get("id"))
                item.setdefault("implementation_plan", base_impl)
                contract = item.get("experiment_contract") if isinstance(item.get("experiment_contract"), dict) else {}
                merged_contract = dict(base_contract)
                merged_contract.update(contract)
                item["experiment_contract"] = merged_contract
            item = normalize_c2c_mechanism_fields(item, baseline)
            item["novelty_gate"] = c2c_idea_novelty_report(item)
            item["implementation_scope_gate"] = c2c_implementation_scope_report(item)
            item["selected"] = bool(item.get("selected") or (selected_id and item.get("id") == selected_id) or (selected_id is None and idx == 0))
            normalized.append(item)
        if normalized and not any(item.get("selected") for item in normalized):
            normalized[0]["selected"] = True
        return normalized

    @staticmethod
    def _c2c_plan_candidate_acceptable(idea: dict[str, Any], selected: dict[str, Any]) -> bool:
        if not isinstance(idea, dict):
            return False
        if selected.get("mechanism_type") and idea.get("mechanism_type") != selected.get("mechanism_type"):
            return False
        if (idea.get("novelty_gate") or {}).get("status") != "pass":
            return False
        if (idea.get("implementation_scope_gate") or {}).get("status") != "pass":
            return False
        contract = idea.get("experiment_contract") if isinstance(idea.get("experiment_contract"), dict) else {}
        if not contract.get("ablation_switch"):
            return False
        if not (contract.get("expected_files") or idea.get("expected_files")):
            return False
        return True

    def _load_c2c_plan_feedback(self) -> list[dict[str, Any]]:
        bundle = load_c2c_feedback_bundle(self.context.project_root, view="implementation")
        feedback = [bundle["summary_entry"], *bundle["entries"], *bundle["iteration_traces"]]
        performance_path = self.context.project_root / "plan" / "performance_feedback.json"
        if performance_path.exists():
            try:
                performance_feedback = json.loads(performance_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                performance_feedback = {}
            if isinstance(performance_feedback, dict):
                feedback.append(
                    {
                        "kind": "c2c_performance_feedback",
                        "source_path": "plan/performance_feedback.json",
                        **performance_feedback,
                    }
                )
        return feedback

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


def _compact_for_plan_prompt(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return value
    return {"truncated_json": text[:max_chars], "original_chars": len(text)}


def _registry_iteration(project_root: Path) -> int | None:
    registry = read_yaml(project_root / "meta" / "registry.yaml", default={}) or {}
    try:
        return int(registry.get("iteration") or 1)
    except (TypeError, ValueError):
        return None


def _compact_s2_memory_idea(idea: dict[str, Any]) -> dict[str, Any]:
    contract = idea.get("experiment_contract") if isinstance(idea.get("experiment_contract"), dict) else {}
    ablation_plan = idea.get("ablation_plan") if isinstance(idea.get("ablation_plan"), dict) else {}
    planner = idea.get("s2_planner") if isinstance(idea.get("s2_planner"), dict) else {}
    return {
        "id": idea.get("id"),
        "title": idea.get("title"),
        "mechanism_type": idea.get("mechanism_type"),
        "selected": bool(idea.get("selected")),
        "hypothesis": _short_text(str(idea.get("hypothesis") or ""), 500),
        "failure_avoidance": list(idea.get("failure_avoidance") or [])[:5] if isinstance(idea.get("failure_avoidance"), list) else [],
        "ablation_switch": contract.get("ablation_switch") or ablation_plan.get("switch"),
        "expected_files": (contract.get("expected_files") or idea.get("expected_files") or [])[:8] if isinstance(contract.get("expected_files") or idea.get("expected_files") or [], list) else [],
        "planner_source": planner.get("source"),
        "planning_mode": planner.get("planning_mode"),
        "novelty_status": (idea.get("novelty_gate") or {}).get("status") if isinstance(idea.get("novelty_gate"), dict) else None,
        "scope_status": (idea.get("implementation_scope_gate") or {}).get("status") if isinstance(idea.get("implementation_scope_gate"), dict) else None,
    }


def _c2c_s2_feedback_digest(feedback: list[dict[str, Any]]) -> dict[str, Any]:
    summary = next((item for item in feedback if isinstance(item, dict) and item.get("kind") == "c2c_feedback_summary"), {})
    performance = next((item for item in reversed(feedback) if isinstance(item, dict) and item.get("kind") == "c2c_performance_feedback"), {})
    perf_summary = performance.get("summary") if isinstance(performance.get("summary"), dict) else {}
    return {
        "latest_reason": summary.get("latest_reason"),
        "latest_failure_mode": summary.get("latest_failure_mode"),
        "latest_decision": summary.get("latest_decision"),
        "failed_idea_ids": list(summary.get("failed_idea_ids") or [])[:6],
        "dragging_datasets": list(summary.get("dragging_datasets") or [])[:5],
        "dataset_regressions": summary.get("dataset_regressions") if isinstance(summary.get("dataset_regressions"), dict) else {},
        "proxy_delta": (summary.get("latest_acceptance") or {}).get("proxy_delta") if isinstance(summary.get("latest_acceptance"), dict) else None,
        "performance_next_action": perf_summary.get("next_action"),
        "same_direction_failure_count": perf_summary.get("same_direction_failure_count"),
        "same_direction_failure_budget": perf_summary.get("same_direction_failure_budget"),
    }


def _compact_c2c_s2_memory_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    recent = entries[-5:]
    tried_ids: list[str] = []
    for entry in entries:
        tried_ids.extend(str(item) for item in entry.get("candidate_ids") or [] if item)
    return {
        "recent_entry_count": len(recent),
        "latest_planning_status": ((recent[-1].get("planning") or {}).get("status") if recent else None),
        "latest_planning_mode": ((recent[-1].get("planning") or {}).get("planning_mode") if recent else None),
        "latest_selected_candidate": ((recent[-1].get("selected_candidate") or {}).get("id") if recent else None),
        "tried_candidate_ids": sorted(set(tried_ids))[-20:],
        "latest_feedback_digest": recent[-1].get("feedback_digest") if recent else {},
    }


def _c2c_s2_memory_for_prompt(memory: dict[str, Any]) -> dict[str, Any]:
    entries = [item for item in memory.get("entries") or [] if isinstance(item, dict)]
    return {
        "schema_version": memory.get("schema_version"),
        "entry_count": len(entries),
        "compacted_summary": memory.get("compacted_summary") or {},
        "recent_entries": entries[-5:],
        "note": "Use this compact memory first; read full artifact paths only if this memory is insufficient.",
    }


def _c2c_s2_available_artifacts() -> dict[str, str]:
    return {
        "planner_memory": "plan/s2_planner_memory.json",
        "performance_feedback": "plan/performance_feedback.json",
        "direction_scorecard": "plan/direction_scorecard.json",
        "failure_feedback": "experiment/results/failure_feedback.json",
        "main_results": "experiment/results/main_results.json",
        "proxy_calibration": "experiment/results/proxy_calibration.json",
        "iteration_trace": "meta/iteration_trace.jsonl",
        "candidate_ideas": "plan/candidate_ideas.json",
        "patch_manifest": "plan/code_patches/patch_manifest.json",
    }


def _short_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "...[truncated]"


def _c2c_s2_resume_planner_prompt(
    *,
    selected: dict[str, Any],
    s1_ideas: list[dict[str, Any]],
    baseline: dict[str, Any],
    concern_matrix: dict[str, Any],
    negative_memory: dict[str, Any],
    feedback: list[dict[str, Any]],
    planner_memory: dict[str, Any],
    adapter: C2CAdapter,
    max_candidates: int,
) -> str:
    payload = {
        "role": "S2 direction-conditioned experiment planner",
        "mode": "resume_session_planning",
        "instruction": (
            "Continue the same S2 planning thread for this S1 mechanism direction. "
            "You may inspect local code and artifacts if needed. Return only JSON."
        ),
        "s1_selected_direction": selected,
        "s1_candidate_pool_compact": _compact_for_plan_prompt(s1_ideas, 6000),
        "baseline": baseline,
        "allowed_files": adapter.allowed_files,
        "allowed_prefixes": adapter.allowed_prefixes,
        "reviewer_concerns": (concern_matrix.get("top_concerns") or [])[:8] if isinstance(concern_matrix, dict) else [],
        "negative_memory": _compact_for_plan_prompt(negative_memory, 4000),
        "s2_planner_memory": _c2c_s2_memory_for_prompt(planner_memory),
        "failure_feedback_compact": _compact_for_plan_prompt(feedback, 9000),
        "available_artifacts": _c2c_s2_available_artifacts(),
        "max_candidates": max_candidates,
        "requirements": [
            "Use s2_planner_memory first; inspect full artifact paths only when memory is insufficient.",
            "Stay in the S1 mechanism direction unless performance feedback says return_to_s1_new_direction.",
            "If performance_feedback.summary.recommended_s2_action is present, follow it when choosing patch_repair, mechanism_repair, or new_same_direction_variant.",
            "Generate concrete variants for S2.5, not a new broad S1 idea.",
            "Prefer diversity across the five same-direction attempts, but treat diversity as a soft search heuristic rather than a hard gate.",
            "Do not change evaluator, datasets, baseline, or cheap proxy protocol.",
            "Avoid variants already tried in s2_planner_memory unless the new plan explicitly fixes the recorded failure.",
            "Each candidate needs id,title,description,hypothesis,mechanism_type,experiment_contract,implementation_plan,failure_avoidance,expected_signature.",
            "Return JSON object with planner_summary, planning_mode, candidates.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _run_s2_codex_json_planner(
    *,
    project_root: Path,
    config: dict[str, Any],
    session_key: str,
    session_id: str | None,
    prompt: str,
) -> dict[str, Any]:
    llm_cfg = config.get("llm", {}) or {}
    codex_cfg = llm_cfg.get("codex_cli") or {}
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as handle:
        output_path = Path(handle.name)
    command = ["codex"]
    sandbox = str(codex_cfg.get("sandbox") or "read-only")
    approval_policy = str(codex_cfg.get("approval_policy") or "never")
    command.extend(["-s", sandbox, "-a", approval_policy, "exec", "--skip-git-repo-check", "--output-last-message", str(output_path)])
    if codex_cfg.get("json_events", True):
        command.append("--json")
    model = str(llm_cfg.get("model") or "")
    if model:
        command.extend(["-m", model])
    command.extend(["-C", str(project_root.resolve())])
    merged_prompt = (
        "Follow this task exactly. You are planning experiments only. Do not edit files. "
        "You may inspect files and artifacts. Return only valid JSON.\n\n"
        f"{prompt}"
    )
    if session_id:
        command.extend(["resume", session_id, "-"])
    else:
        command.append("-")
    started = now_utc()
    start_monotonic = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            input=merged_prompt,
            cwd=project_root,
            timeout=int((config.get("agents", {}).get("s2_directional_planner") or {}).get("resume_timeout_seconds") or llm_cfg.get("timeout_seconds") or 1800),
        )
        text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    except subprocess.TimeoutExpired as exc:
        text = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        _append_s2_planner_event(project_root, session_key, {"status": "timeout", "session_id": session_id, "started_at": started, "ended_at": now_utc()})
        return {"status": "timeout", "reason": f"codex timed out: {exc}", "rationale": text}
    finally:
        output_path.unlink(missing_ok=True)
    parsed_session_id = _parse_s2_codex_session_id(result.stderr, result.stdout) or session_id
    call = {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "session_key": session_key,
        "session_id": parsed_session_id,
        "previous_session_id": session_id,
        "started_at": started,
        "ended_at": now_utc(),
        "duration_seconds": round(time.monotonic() - start_monotonic, 3),
    }
    _append_s2_planner_event(project_root, session_key, call)
    if parsed_session_id:
        _save_s2_codex_session(project_root, session_key, parsed_session_id, config, call)
    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": (result.stderr[-1200:] or result.stdout[-1200:] or f"codex exited {result.returncode}"),
            "session_id": parsed_session_id,
        }
    payload = _parse_json_object(text)
    if not isinstance(payload, dict):
        return {"status": "invalid_json", "reason": "Codex planner did not return a JSON object", "rationale": text[-2000:], "session_id": parsed_session_id}
    return {"status": "ok", "payload": payload, "session_id": parsed_session_id}


def _load_s2_codex_session(project_root: Path, session_key: str) -> str | None:
    session = _load_s2_codex_session_record(project_root, session_key)
    if isinstance(session, dict) and session.get("session_id"):
        return str(session["session_id"])
    return None


def _load_s2_codex_session_record(project_root: Path, session_key: str) -> dict[str, Any]:
    payload = read_yaml(project_root / "meta" / "codex_sessions.yaml", default={"sessions": {}}) or {"sessions": {}}
    session = (payload.get("sessions") or {}).get(session_key) if isinstance(payload, dict) else None
    return session if isinstance(session, dict) else {}


def _session_id_from_record(session: dict[str, Any]) -> str | None:
    if isinstance(session, dict) and session.get("session_id"):
        return str(session["session_id"])
    return None


def _save_s2_codex_session(project_root: Path, session_key: str, session_id: str, config: dict[str, Any], call: dict[str, Any]) -> None:
    path = project_root / "meta" / "codex_sessions.yaml"
    payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    payload.setdefault("sessions", {})
    previous = payload["sessions"].get(session_key) if isinstance(payload["sessions"].get(session_key), dict) else {}
    health = previous.get("health") if isinstance(previous.get("health"), dict) else {}
    payload["sessions"][session_key] = {
        "session_id": session_id,
        "provider": "codex_cli",
        "model": (config.get("llm") or {}).get("model"),
        "updated_at": now_utc(),
        "purpose": "s2_directional_planning",
        "last_call": call,
        "health": health,
    }
    write_yaml(path, payload)


def _record_s2_planner_session_health(
    *,
    project_root: Path,
    session_key: str,
    config: dict[str, Any],
    session_id: str | None,
    status: str,
    candidate_ids: list[str],
    duplicate: bool,
    reason: str,
) -> dict[str, Any]:
    cfg = config.get("agents", {}).get("s2_directional_planner") or {}
    invalid_threshold = max(1, int(cfg.get("session_reset_invalid_streak") or 2))
    duplicate_threshold = max(1, int(cfg.get("session_reset_duplicate_streak") or 2))
    path = project_root / "meta" / "codex_sessions.yaml"
    payload = read_yaml(path, default={"sessions": {}}) or {"sessions": {}}
    payload.setdefault("sessions", {})
    session = payload["sessions"].get(session_key) if isinstance(payload["sessions"].get(session_key), dict) else {}
    health = session.get("health") if isinstance(session.get("health"), dict) else {}
    invalid_streak = int(health.get("invalid_streak") or 0)
    duplicate_streak = int(health.get("duplicate_streak") or 0)
    if status == "invalid":
        invalid_streak += 1
        duplicate_streak = 0
    elif duplicate:
        duplicate_streak += 1
        invalid_streak = 0
    else:
        invalid_streak = 0
        duplicate_streak = 0
    health.update(
        {
            "last_status": status,
            "invalid_streak": invalid_streak,
            "duplicate_streak": duplicate_streak,
            "last_candidate_ids": candidate_ids,
            "last_reason": reason,
            "updated_at": now_utc(),
        }
    )
    should_reset = bool(session_id) and (
        invalid_streak >= invalid_threshold or duplicate_streak >= duplicate_threshold
    )
    reset_reason = None
    if should_reset:
        reset_reason = "invalid_output_streak" if invalid_streak >= invalid_threshold else "duplicate_output_streak"
        health["reset_count"] = int(health.get("reset_count") or 0) + 1
        health["last_reset_at"] = now_utc()
        health["last_reset_reason"] = reset_reason
        health["last_reset_session_id"] = session_id
        payload["sessions"].pop(session_key, None)
        _append_s2_planner_event(
            project_root,
            session_key,
            {
                "status": "session_reset",
                "session_id": session_id,
                "reason": reset_reason,
                "health": health,
                "memory_preserved": True,
            },
        )
    else:
        session.update(
            {
                "session_id": session_id or session.get("session_id"),
                "provider": session.get("provider") or "codex_cli",
                "model": session.get("model") or (config.get("llm") or {}).get("model"),
                "updated_at": now_utc(),
                "purpose": "s2_directional_planning",
                "health": health,
            }
        )
        payload["sessions"][session_key] = session
    write_yaml(path, payload)
    return {"health": health, "session_reset": should_reset, "session_reset_reason": reset_reason}


def _s2_planner_candidate_ids(ideas: list[dict[str, Any]]) -> list[str]:
    return [str(idea.get("id")) for idea in ideas if isinstance(idea, dict) and idea.get("id")]


def _s2_planner_duplicate_candidate_ids(candidate_ids: list[str], planner_memory: dict[str, Any], session_record: dict[str, Any]) -> bool:
    if not candidate_ids:
        return False
    memory_ids: set[str] = set()
    for entry in planner_memory.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        memory_ids.update(str(item) for item in entry.get("candidate_ids") or [] if item)
        selected = entry.get("selected_candidate") if isinstance(entry.get("selected_candidate"), dict) else {}
        if selected.get("id"):
            memory_ids.add(str(selected["id"]))
    health = session_record.get("health") if isinstance(session_record.get("health"), dict) else {}
    memory_ids.update(str(item) for item in health.get("last_candidate_ids") or [] if item)
    return bool(memory_ids) and all(candidate_id in memory_ids for candidate_id in candidate_ids)


def _append_s2_planner_event(project_root: Path, session_key: str, event: dict[str, Any]) -> None:
    path = project_root / "plan" / "logs" / "s2_planner_codex_events.jsonl"
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"session_key": session_key, **event}, ensure_ascii=False) + "\n")


def _parse_s2_codex_session_id(stderr: str, stdout: str = "") -> str | None:
    match = re.search(r"session id:\s*([0-9a-fA-F-]+)", stderr or "")
    if match:
        return match.group(1)
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None
