"""Planning stage."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..adapters.runner import ExperimentRunner
from ..c2c import C2CAdapter, c2c_idea_novelty_report, c2c_implementation_scope_report, is_c2c_project, normalize_c2c_mechanism_fields
from ..code_patch import CodePatchAgent
from ..contract_store import ContractStore, canonical_contract_bytes
from ..direction_contracts import (
    build_planner_decision_artifact,
    build_variant_contract,
    build_variant_fingerprint_artifact,
    direction_planner_seed,
    load_direction,
)
from ..failure_log import load_c2c_feedback_bundle
from ..method_memory import collect_used_shared_memory_refs, shared_method_memory_for_prompt, shared_method_memory_query_context
from ..itr_ideas import build_quick_screen_execution
from ..llm import codex_subprocess_env
from ..resources import best_matching_run
from ..resources import best_itr_execution_plan, discover_local_mm_resources
from ..s2_planner_contracts import (
    build_s2_5_patch_gate_report,
    build_s2_candidate_pool,
    build_s2_implementation_contract,
    build_s2_planner_gate_report,
    build_s2_variant_scorecard,
)
from ..s2_feedback_policy import (
    build_s2_adaptive_policy,
    build_s2_feedback_context,
    build_s2_score_adjustment_report,
)
from ..utils import compact_markdown, ensure_dir, now_utc, read_json, read_yaml, sanitize_filename, write_yaml
from ..domain_contracts import TRIAL_SPEC_SCHEMA_VERSION, canonical_hash, canonical_json, validate_trial_spec
from ..evidence import EVIDENCE_SCHEMA_VERSIONS
from ..proxy_classifier import build_proxy_decision_policy
from ..phase_command_plan import build_phase_command_plan, store_phase_command_plan
from ..research_state import ResearchEventLedger
from .base import AgentContext


_AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS = {
    **EVIDENCE_SCHEMA_VERSIONS,
    "activation_evidence": "auto_research_activation_evidence_v4",
    "full_s3_readiness": "auto_research_full_s3_readiness_v4",
}


class PlanAgent:
    stage_key = "S2_plan"

    def __init__(self, context: AgentContext):
        self.context = context
        self.runner = ExperimentRunner(context.config)

    def run(self) -> dict[str, Any]:
        if is_c2c_project(self.context.config):
            return self._run_c2c_plan()
        direction = load_direction(self.context.project_root)
        ideas = [direction_planner_seed(direction)]
        selected = next((idea for idea in ideas if idea.get("selected")), ideas[0])
        research_state = ResearchEventLedger(self.context.project_root).state()
        direction_budget = (((research_state.get("directions") or {}).get(direction.get("direction_semantic_hash")) or {}).get("budget") or {})
        variant_index = int(direction_budget.get("consumed", 0)) + 1
        selected = dict(selected)
        selected["id"] = f"{direction['direction_id']}-variant-{variant_index}"
        selected["variant_id"] = selected["id"]
        selected["variation_coordinates"] = {"intervention": {"strength": variant_index}}
        selected["algorithm_operations"] = ["apply mediator-aware routing", f"set intervention strength to {variant_index}"]
        selected["config_overrides"] = {"intervention_strength": variant_index}
        selected.setdefault("s1_direction_id", direction.get("direction_id"))
        selected.setdefault("direction_id", direction.get("direction_id"))
        selected.setdefault("mechanism_axis", direction.get("mechanism_axis"))
        selected.setdefault("integration_point", direction.get("integration_point"))
        selected.setdefault("control_signal", direction.get("control_signal"))
        selected.setdefault("expected_signature", direction.get("expected_metric_signature"))
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
            "acceptance_criteria": {
                "minimum_mean_delta": 0.01,
                "must_emit": ["main_results.json", "ablation_results.json", "hypothesis_verification.md"],
            },
            "ablation_matrix": [
                {"experiment": "w/o core module", "tests_hypothesis": "H2", "modification": "Remove the main intervention."}
            ],
            "task_graph": {
                "parallel_group_1": [
                    {"task": "baseline_eval", "estimated_hours": 2, "depends_on": []},
                    {"task": "proposed_eval", "estimated_hours": 2, "depends_on": []},
                ],
                "final": [
                    {"task": "aggregate_results", "estimated_hours": 0.5, "depends_on": ["parallel_group_1"]},
                ],
            },
            "resource_budget": {},
            "local_resources": resources,
            "execution": execution,
        }
        planner_decision = build_planner_decision_artifact(
            direction=direction,
            planner_summary="Generic S2 plan derived from the selected S1 direction.",
            planning_mode="single_direction_plan",
            next_variant=selected,
            used_shared_memory_refs=direction.get("used_shared_memory_refs") if isinstance(direction, dict) else [],
            source="generic_plan_agent",
        )
        variant_fingerprint = _s2_variant_fingerprint(selected)
        selected["variant_fingerprint"] = selected.get("variant_fingerprint") or variant_fingerprint
        plan["project_root"] = str(self.context.project_root)
        variant_contract = build_variant_contract(direction=direction, variant=selected, plan=plan, mode="generic")
        ledger = ResearchEventLedger(self.context.project_root)
        ledger.select_direction(direction, event_id=f"direction:{direction['direction_spec_hash']}")
        ledger.plan_variant(
            variant_contract,
            feedback_from_attempt_ids=list((variant_contract.get("lineage") or {}).get("feedback_from_attempt_ids") or []),
            event_id=f"variant:{variant_contract['variant_spec_hash']}",
        )
        variant_fingerprint_artifact = build_variant_fingerprint_artifact(
            direction=direction,
            variant=selected,
            fingerprint=selected.get("variant_fingerprint"),
            history_fingerprints=[],
            mode="generic",
        )
        plan["planner_decision"] = planner_decision
        plan["variant_contract"] = variant_contract
        plan["variant_fingerprint"] = variant_fingerprint_artifact
        planner_record = self.context.artifacts.write_json(
            self.stage_key,
            "planner_decision.json",
            planner_decision,
            artifact_type="s2_planner_decision",
            summary="S2 planner decision for the selected direction",
            source_paths=["literature/direction.json"],
        )
        variant_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant.json",
            variant_contract,
            artifact_type="s2_variant_contract",
            summary="Executable S2 variant contract for S2.5/S3",
            source_paths=["literature/direction.json"],
        )
        variant_fingerprint_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_fingerprint.json",
            variant_fingerprint_artifact,
            artifact_type="s2_variant_fingerprint",
            summary="Stable S2 variant fingerprint",
            source_paths=["literature/direction.json"],
        )
        implementation_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/implementation_contract.json",
            {
                "schema_version": "auto_research_implementation_contract_v1",
                "direction_id": direction["direction_id"],
                "variant_id": variant_contract["variant_id"],
                "variant_spec_hash": variant_contract["variant_spec_hash"],
                "mode": "no_patch_required",
                "implementation_surface_ids": variant_contract["implementation_surface_ids"],
            },
            artifact_type="implementation_contract",
            summary="Generic implementation contract",
            source_paths=[variant_contract_record["path"]],
        )
        patch_manifest_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_manifest.json",
            {
                "schema_version": "auto_research_patch_manifest_v1",
                "status": "disabled",
                "selected_candidate_id": variant_contract["variant_id"],
                "variant_spec_hash": variant_contract["variant_spec_hash"],
                "reason": "generic simulated or externally implemented variant",
            },
            artifact_type="patch_manifest",
            summary="Generic no-op implementation manifest",
            source_paths=[implementation_contract_record["path"]],
        )
        patch_gate_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_gate_report.json",
            {
                "schema_version": "auto_research_patch_gate_v1",
                "gate": "pass",
                "variant_id": variant_contract["variant_id"],
                "variant_spec_hash": variant_contract["variant_spec_hash"],
                "checks": {"no_patch_required": True},
            },
            artifact_type="patch_gate_report",
            summary="Generic no-op implementation gate",
            source_paths=[patch_manifest_record["path"]],
        )
        plan_record = self.context.artifacts.write_json(
            self.stage_key,
            "trial_spec.json",
            _trial_spec_from_plan(plan, variant_contract, profile=_execution_profile(self.context.config), project_root=self.context.project_root),
            artifact_type="plan",
            summary="Structured experiment plan",
            source_paths=["literature/direction.json", "literature/direction.json"],
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
        return {
            "plan": plan,
            "artifacts": [
                planner_record["path"],
                variant_contract_record["path"],
                variant_fingerprint_record["path"],
                implementation_contract_record["path"],
                patch_manifest_record["path"],
                patch_gate_record["path"],
                plan_record["path"],
                hypotheses_record["path"],
                task_graph["path"],
            ],
        }

    def _run_c2c_plan(self) -> dict[str, Any]:
        adapter = C2CAdapter(self.context.project_root, self.context.config)
        patch_only_feedback = self._c2c_implementation_patch_only_feedback()
        if patch_only_feedback:
            return self._run_c2c_s2_5_patch_only(adapter, patch_only_feedback)
        return self._run_c2c_plan_regular(adapter)

    def _run_c2c_plan_regular(self, adapter: C2CAdapter | None = None) -> dict[str, Any]:
        adapter = adapter or C2CAdapter(self.context.project_root, self.context.config)
        s1_direction = load_direction(self.context.project_root)
        s1_ideas = [direction_planner_seed(s1_direction)]
        s1_selected = next((idea for idea in s1_ideas if idea.get("selected")), s1_ideas[0])
        s1_selected.setdefault("s1_direction_id", s1_direction.get("direction_id"))
        s1_selected.setdefault("direction_id", s1_direction.get("direction_id"))
        s1_selected.setdefault("mechanism_axis", s1_direction.get("mechanism_axis"))
        s1_selected.setdefault("integration_point", s1_direction.get("integration_point"))
        s1_selected.setdefault("control_signal", s1_direction.get("control_signal"))
        s1_selected.setdefault("expected_signature", s1_direction.get("expected_metric_signature"))
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
        variant_selection = planning_result.get("variant_selection") if isinstance(planning_result.get("variant_selection"), dict) else {}
        next_variant = variant_selection.get("next_variant") if isinstance(variant_selection.get("next_variant"), dict) else {}
        selected = next((idea for idea in ideas if idea.get("selected")), ideas[0])
        selected["selected"] = True
        selected.setdefault("s1_direction_id", s1_direction.get("direction_id"))
        selected.setdefault("direction_id", s1_direction.get("direction_id"))
        selected.setdefault("mechanism_axis", s1_direction.get("mechanism_axis"))
        selected.setdefault("integration_point", s1_direction.get("integration_point"))
        selected.setdefault("control_signal", s1_direction.get("control_signal"))
        if not _variant_expected_files(selected) and s1_direction.get("expected_files"):
            selected["expected_files"] = list(s1_direction.get("expected_files") or [])
        _sanitize_c2c_variant_expected_files(selected, s1_direction, self.context.config)
        _ensure_c2c_s2_config_overrides(selected)
        if isinstance(selected.get("experiment_contract"), dict) and not selected["experiment_contract"].get("expected_files") and selected.get("expected_files"):
            selected["experiment_contract"]["expected_files"] = list(selected.get("expected_files") or [])
        min_delta = float(small_loop_cfg.get("min_delta_to_pass", 0.1))
        max_regression = float(small_loop_cfg.get("max_dataset_regression", 2.0))
        c2c_config = self.context.config.get("c2c", {})
        simulate = bool((self.context.config.get("experiment") or {}).get("simulate"))
        sample_provenance = c2c_config.get("sample_provenance") if isinstance(c2c_config.get("sample_provenance"), dict) else {}
        datasets = []
        for name in c2c_config.get("datasets", ["mmlu-redux", "ai2-arc", "openbookqa"]):
            provenance = sample_provenance.get(name) if isinstance(sample_provenance.get(name), dict) else {}
            dataset = {
                "name": name,
                "domain": "LLM benchmark",
                "split": "C2C unified_evaluator configured split",
                "metric": "overall_accuracy",
                "reason": "Part of the configured original C2C three-dataset small-loop protocol.",
            }
            if simulate:
                dataset["sample_count"] = int(provenance.get("sample_count") or 1)
            else:
                dataset.update(
                    {
                        "source_revision": provenance.get("source_revision"),
                        "ordered_sample_ids": deepcopy(provenance.get("ordered_sample_ids")),
                        "sample_count": provenance.get("sample_count"),
                    }
                )
            datasets.append(dataset)
        evaluator_provenance = _c2c_evaluator_provenance(adapter.repo_root, c2c_config, simulate=simulate)
        short_loop = {
            "collector": "c2c_small_loop",
            "mode": "simulate" if simulate else "small_loop",
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
            "acceptance_rule": (
                f"best_candidate.mean >= baseline.mean + {min_delta} and no dataset regresses more than {max_regression} points"
            ),
            "reviewer_concerns": concern_matrix.get("top_concerns", []),
            "blocked_patterns": (negative_memory.get("blocked_idea_patterns") or [])[:8],
            "failure_feedback_count": len(feedback),
            **evaluator_provenance,
        }
        plan = {
            "selected_idea": selected,
            "candidate_ideas": ideas,
            "next_variant": next_variant,
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
            "variant_selection": variant_selection,
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
                    {"task": "guarded_patch_generation", "estimated_hours": 0.1, "depends_on": []},
                    {"task": "preflight_compile_and_tests", "estimated_hours": 0.1, "depends_on": ["guarded_patch_generation"]},
                    {"task": "small2048_train", "estimated_hours": 2, "depends_on": ["preflight_compile_and_tests"]},
                    {"task": "three_dataset_eval", "estimated_hours": 3, "depends_on": ["small2048_train"]},
                ],
                "final": [],
            },
            "resource_budget": {},
            "execution": short_loop,
        }
        plan["candidate_ideas"] = ideas
        history_fingerprints = []
        if isinstance(variant_selection.get("history_summary"), dict):
            history_fingerprints = [str(item) for item in variant_selection["history_summary"].get("fingerprints") or [] if item]
        pool_candidates = variant_selection.get("candidate_pool") if isinstance(variant_selection.get("candidate_pool"), list) else ideas
        _sanitize_c2c_variant_expected_files(selected, s1_direction, self.context.config)
        for item in ideas:
            if isinstance(item, dict):
                _sanitize_c2c_variant_expected_files(item, s1_direction, self.context.config)
        if isinstance(pool_candidates, list):
            for item in pool_candidates:
                if isinstance(item, dict):
                    _sanitize_c2c_variant_expected_files(item, s1_direction, self.context.config)
        candidate_pool = build_s2_candidate_pool(
            direction=s1_direction,
            candidates=[item for item in pool_candidates if isinstance(item, dict)],
            source=str(planning_result["metadata"].get("source") or "s2_directional_planner"),
            used_shared_memory_refs=planning_result["metadata"].get("used_shared_memory_refs") if isinstance(planning_result.get("metadata"), dict) else [],
        )
        evidence_quality = read_json(self.context.project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
        shared_memory = shared_method_memory_for_prompt(
            self.context.config,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                selected_direction=s1_direction,
                feedback=feedback,
            ),
        )
        feedback_context = build_s2_feedback_context(
            project_root=self.context.project_root,
            direction=s1_direction,
            config=self.context.config,
            shared_memory=shared_memory,
        )
        adaptive_policy = build_s2_adaptive_policy(feedback_context, self.context.config)
        scorecard = build_s2_variant_scorecard(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            selected_variant=next_variant or selected,
            evidence_quality=evidence_quality if isinstance(evidence_quality, dict) else {},
            variant_fingerprint={},
            planner_memory=planner_memory,
            feedback=feedback,
            feedback_context=feedback_context,
            adaptive_policy=adaptive_policy,
            config=self.context.config,
        )
        selected = _select_candidate_for_scorecard(
            ideas=[item for item in ideas if isinstance(item, dict)],
            pool_candidates=[item for item in pool_candidates if isinstance(item, dict)],
            selected_variant_id=str(scorecard.get("selected_variant_id") or ""),
            fallback=selected,
        )
        _mark_selected_variant(ideas, selected)
        selected.setdefault("s1_direction_id", s1_direction.get("direction_id"))
        selected.setdefault("direction_id", s1_direction.get("direction_id"))
        selected.setdefault("mechanism_axis", s1_direction.get("mechanism_axis"))
        selected.setdefault("integration_point", s1_direction.get("integration_point"))
        selected.setdefault("control_signal", s1_direction.get("control_signal"))
        selected.setdefault("expected_signature", s1_direction.get("expected_metric_signature"))
        if not _variant_expected_files(selected) and s1_direction.get("expected_files"):
            selected["expected_files"] = list(s1_direction.get("expected_files") or [])
        _sanitize_c2c_variant_expected_files(selected, s1_direction, self.context.config)
        _ensure_c2c_s2_config_overrides(selected)
        if bool((self.context.config.get("experiment") or {}).get("simulate")):
            research_state = ResearchEventLedger(self.context.project_root).state()
            budget = (((research_state.get("directions") or {}).get(s1_direction.get("direction_semantic_hash")) or {}).get("budget") or {})
            ordinal = int(budget.get("consumed", 0)) + 1
            selected = deepcopy(selected)
            selected["variation_coordinates"] = {"intervention": {"utility_temperature": ordinal}}
            selected["algorithm_operations"] = list(selected.get("algorithm_operations") or []) + [f"set utility temperature to {ordinal}"]
            selected["config_overrides"] = {**(selected.get("config_overrides") or {}), "utility_temperature": ordinal}
        if isinstance(selected.get("experiment_contract"), dict) and not selected["experiment_contract"].get("expected_files") and selected.get("expected_files"):
            selected["experiment_contract"]["expected_files"] = list(selected.get("expected_files") or [])
        if not selected.get("variant_fingerprint"):
            selected["variant_fingerprint"] = _s2_variant_fingerprint(selected)
        next_variant = selected
        if variant_selection:
            variant_selection["next_variant"] = next_variant
            variant_selection["selected_variant_id"] = selected.get("id")
            variant_selection["adaptive_policy_hash"] = adaptive_policy.get("policy_hash")
        plan["selected_idea"] = selected
        plan["next_variant"] = next_variant
        plan["implementation_scope"] = {
            "selected_scope": selected.get("implementation_scope"),
            "implementation_plan": selected.get("implementation_plan"),
            "scope_gate": selected.get("implementation_scope_gate"),
        }
        if plan.get("ablation_matrix"):
            plan["ablation_matrix"][0]["switch"] = (selected.get("experiment_contract") or {}).get("ablation_switch") or (selected.get("ablation_plan") or {}).get("switch")
            plan["ablation_matrix"][0]["expected_signature"] = selected.get("expected_signature")
            plan["ablation_matrix"][0]["coverage_diagnostics"] = selected.get("coverage_diagnostics")
            if len(plan["ablation_matrix"]) > 1:
                plan["ablation_matrix"][1]["matched_coverage_ablation"] = selected.get("matched_coverage_ablation")
                plan["ablation_matrix"][1]["required_stats"] = (selected.get("coverage_diagnostics") or {}).get("stats")
        planner_decision = build_planner_decision_artifact(
            direction=s1_direction,
            planner_summary=planning_result["metadata"].get("planner_summary"),
            planning_mode=planning_result["metadata"].get("planning_mode"),
            next_variant=next_variant,
            used_shared_memory_refs=planning_result["metadata"].get("used_shared_memory_refs") if isinstance(planning_result.get("metadata"), dict) else [],
            source=str(planning_result["metadata"].get("source") or "c2c_plan_agent"),
        )
        plan["project_root"] = str(self.context.project_root)
        variant_contract = build_variant_contract(direction=s1_direction, variant=selected, plan=plan, execution=short_loop, mode="c2c")
        variant_fingerprint_artifact = build_variant_fingerprint_artifact(
            direction=s1_direction,
            variant=selected,
            fingerprint=selected.get("variant_fingerprint"),
            history_fingerprints=history_fingerprints,
            mode="regular",
        )
        score_adjustment_report = build_s2_score_adjustment_report(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            scorecard=scorecard,
            adaptive_policy=adaptive_policy,
            feedback_context=feedback_context,
        )
        planner_gate_report = build_s2_planner_gate_report(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            scorecard=scorecard,
            next_variant=next_variant,
            variant_contract=variant_contract,
            variant_fingerprint=variant_fingerprint_artifact,
            adaptive_policy=adaptive_policy,
            score_adjustment_report=score_adjustment_report,
            config=self.context.config,
        )
        plan["planner_decision"] = planner_decision
        plan["variant_contract"] = variant_contract
        plan["variant_fingerprint"] = variant_fingerprint_artifact
        plan["s2_planner"] = {
            "candidate_pool": candidate_pool,
            "feedback_context": feedback_context,
            "adaptive_policy": adaptive_policy,
            "variant_scorecard": scorecard,
            "score_adjustment_report": score_adjustment_report,
            "planner_gate_report": planner_gate_report,
        }
        plan["variant_selection"] = variant_selection
        if variant_selection:
            variant_selection_record = self.context.artifacts.write_json(
                self.stage_key,
                "variant_selection_report.json",
                variant_selection,
                artifact_type="c2c_s2_next_variant",
                summary="Direction-conditioned S2 next variant selected for S2.5",
                source_paths=["literature/direction.json", "plan/performance_feedback.json", "plan/s2_planner_memory.json"],
            )
        else:
            variant_selection_record = None
        candidate_pool_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/candidate_pool.json",
            candidate_pool,
            artifact_type="c2c_s2_candidate_pool",
            summary="S2a candidate variant pool generated from the selected S1 direction",
            source_paths=["literature/direction.json", "plan/performance_feedback.json", "plan/s2_planner_memory.json"],
        )
        feedback_context_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/feedback_context.json",
            feedback_context,
            artifact_type="c2c_s2_feedback_context",
            summary="S2 feedback context assembled from route, attempt, proxy, and memory history",
            source_paths=[
                "meta/research_state.json",
                "meta/route_outcome.json",
                "experiment/results/c2c_proxy_calibration_policy.json",
                "experiment/results/c2c_proxy_decision_report.json",
                "plan/performance_feedback.json",
            ],
        )
        adaptive_policy_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/adaptive_policy.json",
            adaptive_policy,
            artifact_type="c2c_s2_adaptive_policy",
            summary="S2 adaptive variant selection policy derived from feedback context",
            source_paths=[feedback_context_record["path"]],
        )
        scorecard_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/variant_scorecard.json",
            scorecard,
            artifact_type="c2c_s2_variant_scorecard",
            summary="S2b adaptive deterministic variant scorecard and selected next variant",
            source_paths=[candidate_pool_record["path"], feedback_context_record["path"], adaptive_policy_record["path"], "literature/c2c/evidence_quality_score.json", "plan/s2_planner_memory.json"],
        )
        score_adjustment_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/score_adjustment_report.json",
            score_adjustment_report,
            artifact_type="c2c_s2_score_adjustment_report",
            summary="Per-variant adaptive score adjustments used by the S2 selector",
            source_paths=[candidate_pool_record["path"], adaptive_policy_record["path"], scorecard_record["path"]],
        )
        planner_gate_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/planner_gate_report.json",
            planner_gate_report,
            artifact_type="c2c_s2_planner_gate_report",
            summary="S2c planner gate report for S2.5 handoff readiness",
            source_paths=[candidate_pool_record["path"], scorecard_record["path"], score_adjustment_record["path"], "literature/direction.json"],
        )
        s2_next_variant_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/selected_variant_report.json",
            next_variant or selected,
            artifact_type="c2c_s2_selected_next_variant",
            summary="S2c selected next variant passed by deterministic planner gate",
            source_paths=[planner_gate_record["path"], scorecard_record["path"]],
        )
        planner_decision_record = self.context.artifacts.write_json(
            self.stage_key,
            "planner_decision.json",
            planner_decision,
            artifact_type="s2_planner_decision",
            summary="C2C S2 planner decision for the next variant",
            source_paths=["literature/direction.json", "plan/performance_feedback.json", "plan/s2_planner_memory.json"],
        )
        variant_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant.json",
            variant_contract,
            artifact_type="s2_variant_contract",
            summary="C2C executable variant contract for S2.5",
            source_paths=["literature/direction.json", "plan/variant.json", "plan/trial_spec.json"],
        )
        variant_fingerprint_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_fingerprint.json",
            variant_fingerprint_artifact,
            artifact_type="s2_variant_fingerprint",
            summary="C2C stable variant fingerprint and history check",
            source_paths=["literature/direction.json", "plan/variant.json", "plan/s2_planner_memory.json"],
        )
        if planner_gate_report.get("gate") == "pass":
            ledger = ResearchEventLedger(self.context.project_root)
            ledger.select_direction(s1_direction, event_id=f"direction:{s1_direction['direction_spec_hash']}")
            ledger.plan_variant(
                variant_contract,
                feedback_from_attempt_ids=list((variant_contract.get("lineage") or {}).get("feedback_from_attempt_ids") or []),
                event_id=f"variant:{variant_contract['variant_spec_hash']}",
            )
            implementation_contract = build_s2_implementation_contract(
                direction=s1_direction,
                selected_variant=next_variant or selected,
                variant_contract=variant_contract,
                planner_gate_report=planner_gate_report,
                config=self.context.config,
            )
            code_patch_manifest = CodePatchAgent(self.context.project_root, self.context.config, self.context.artifacts).run_selected_variant(
                plan,
                next_variant or selected,
                implementation_contract,
                planner_gate_report,
                variant_fingerprint_artifact,
            )
        else:
            code_patch_manifest = {
                "status": "planner_gate_failed",
                "artifacts": [],
                "selected_candidate_id": selected.get("id"),
                "valid_patch_count": 0,
                "valid_patch_ids": [],
                "planner_gate_report": planner_gate_report,
            }
        plan["code_patch_manifest"] = {
            "status": code_patch_manifest.get("status"),
            "path": "plan/code_patches/patch_manifest.json" if code_patch_manifest.get("status") not in {"disabled", "planner_gate_failed"} else "",
            "selected_candidate_id": code_patch_manifest.get("selected_candidate_id"),
            "valid_patch_count": code_patch_manifest.get("valid_patch_count"),
            "valid_patch_ids": code_patch_manifest.get("valid_patch_ids") or [],
            "planner_gate": planner_gate_report.get("gate"),
        }
        short_loop["patch_source"] = "s2_5_frozen_codex_patch" if code_patch_manifest.get("status") not in {"disabled", "planner_gate_failed"} else "config_overrides_only"
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
        plan_record = self.context.artifacts.write_json(
            self.stage_key,
            "trial_spec.json",
            _trial_spec_from_plan(plan, variant_contract, profile=_execution_profile(self.context.config), project_root=self.context.project_root),
            artifact_type="plan",
            summary="C2C structured small-loop experiment plan",
            source_paths=["literature/direction.json", "literature/c2c/baseline_evidence.json"],
        )
        candidate_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/candidate_pool_snapshot.json",
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
            },
            artifact_type="c2c_plan_feedback",
            summary="C2C failure feedback used by S2",
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
        return {
            "plan": plan,
            "artifacts": [
                plan_record["path"],
                *code_patch_manifest.get("artifacts", []),
                *([variant_selection_record["path"]] if variant_selection_record else []),
                candidate_pool_record["path"],
                feedback_context_record["path"],
                adaptive_policy_record["path"],
                scorecard_record["path"],
                score_adjustment_record["path"],
                s2_next_variant_record["path"],
                planner_gate_record["path"],
                planner_decision_record["path"],
                variant_contract_record["path"],
                variant_fingerprint_record["path"],
                memory_record["artifact"]["path"],
                candidate_record["path"],
                short_loop_record["path"],
                feedback_record["path"],
                hypotheses_record["path"],
                task_graph_record["path"],
            ],
        }

    def _c2c_implementation_patch_only_feedback(self) -> dict[str, Any] | None:
        feedback_path = self.context.project_root / "plan" / "performance_feedback.json"
        feedback = read_json(feedback_path, default={}) if feedback_path.exists() else {}
        summary = feedback.get("summary") if isinstance(feedback, dict) and isinstance(feedback.get("summary"), dict) else {}
        if summary.get("failure_class") != "implementation_failure":
            return None
        return feedback

    def _run_c2c_s2_5_patch_only(self, adapter: C2CAdapter, feedback: dict[str, Any]) -> dict[str, Any]:
        del adapter
        repair_dispatch = _load_s2_5_repair_dispatch(self.context.project_root)
        candidate_path = self.context.project_root / "plan" / "s2_planner/candidate_pool_snapshot.json"
        if not candidate_path.exists():
            return self._run_c2c_patch_only_blocked("implementation_failure requested S2.5 patch-only repair but plan/s2_planner/candidate_pool.json is missing", feedback, repair_dispatch)
        ideas = read_json(candidate_path, default=[])
        if not isinstance(ideas, list) or not ideas:
            return self._run_c2c_patch_only_blocked("implementation_failure requested S2.5 patch-only repair but candidate pool has no variants", feedback, repair_dispatch)
        target_candidate_id = str(repair_dispatch.get("selected_candidate_id") or "").strip()
        target_fingerprint = str(repair_dispatch.get("variant_fingerprint") or "").strip()
        candidate_objects = [idea for idea in ideas if isinstance(idea, dict)]
        target_ideas = _select_patch_only_repair_ideas(
            candidate_objects,
            target_candidate_id=target_candidate_id,
            target_fingerprint=target_fingerprint,
        )
        patch_ideas = [
            _with_patch_only_previous_failure(idea, feedback, repair_dispatch=repair_dispatch)
            for idea in target_ideas
        ]
        if not patch_ideas:
            return self._run_c2c_patch_only_blocked("implementation_failure requested S2.5 patch-only repair but target candidate is missing", feedback, repair_dispatch)
        plan_path = self.context.project_root / "plan" / "trial_spec.json"
        plan = read_yaml(plan_path, default={}) if plan_path.exists() else {}
        if not isinstance(plan, dict):
            plan = {}
        for idx, idea in enumerate(patch_ideas):
            idea["selected"] = idx == 0
        selected_patch_idea = next((idea for idea in patch_ideas if isinstance(idea, dict) and idea.get("selected")), patch_ideas[0])
        plan["candidate_ideas"] = patch_ideas
        plan["selected_idea"] = selected_patch_idea
        plan["s2_5_patch_only_repair"] = {
            "enabled": True,
            "repair_lane": "s2_5_only_implementation_repair",
            "skips_s2_planner": True,
            "reason": (feedback.get("reason") if isinstance(feedback, dict) else None),
            "failure_class": "implementation_failure",
            "performance_feedback_path": "plan/performance_feedback.json",
            "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
            "selected_candidate_id": selected_patch_idea.get("id"),
            "variant_fingerprint": _candidate_variant_fingerprint(selected_patch_idea),
            "same_candidate_required": True,
            "same_variant_fingerprint_required": bool(_candidate_variant_fingerprint(selected_patch_idea)),
            "reuse_persistent_codex_session": True,
            "repair_until": "patch_eligible_for_s3_or_implementation_blocked",
            "does_not_consume_same_direction_attempt": True,
        }
        if repair_dispatch:
            plan["s2_5_repair_dispatch"] = repair_dispatch
        s1_direction = load_direction(self.context.project_root)
        selected_patch_idea.setdefault("s1_direction_id", s1_direction.get("direction_id"))
        selected_patch_idea.setdefault("direction_id", s1_direction.get("direction_id"))
        selected_patch_idea.setdefault("mechanism_axis", s1_direction.get("mechanism_axis"))
        selected_patch_idea.setdefault("integration_point", s1_direction.get("integration_point"))
        selected_patch_idea.setdefault("control_signal", s1_direction.get("control_signal"))
        repair_files = [str(item) for item in repair_dispatch.get("changed_files") or [] if item] if isinstance(repair_dispatch, dict) else []
        if repair_files:
            selected_patch_idea["expected_files"] = repair_files
            if isinstance(selected_patch_idea.get("experiment_contract"), dict):
                selected_patch_idea["experiment_contract"]["expected_files"] = repair_files
        if not _variant_expected_files(selected_patch_idea) and s1_direction.get("expected_files"):
            selected_patch_idea["expected_files"] = list(s1_direction.get("expected_files") or [])
        _sanitize_c2c_variant_expected_files(selected_patch_idea, s1_direction, self.context.config)
        _ensure_c2c_s2_config_overrides(selected_patch_idea)
        if isinstance(selected_patch_idea.get("experiment_contract"), dict) and not selected_patch_idea["experiment_contract"].get("expected_files") and selected_patch_idea.get("expected_files"):
            selected_patch_idea["experiment_contract"]["expected_files"] = list(selected_patch_idea.get("expected_files") or [])
        if not selected_patch_idea.get("variant_fingerprint"):
            selected_patch_idea["variant_fingerprint"] = _candidate_variant_fingerprint(selected_patch_idea) or _s2_variant_fingerprint(selected_patch_idea)
        planner_decision = build_planner_decision_artifact(
            direction=s1_direction,
            planner_summary="S2 planner skipped; implementation failure routes directly to S2.5 repair.",
            planning_mode="s2_5_only_implementation_repair",
            next_variant=selected_patch_idea,
            used_shared_memory_refs=selected_patch_idea.get("used_shared_memory_refs") if isinstance(selected_patch_idea, dict) else [],
            source="s2_5_patch_only_repair",
        )
        plan["project_root"] = str(self.context.project_root)
        variant_contract = build_variant_contract(direction=s1_direction, variant=selected_patch_idea, plan=plan, mode="implementation_repair")
        variant_fingerprint_artifact = build_variant_fingerprint_artifact(
            direction=s1_direction,
            variant=selected_patch_idea,
            fingerprint=selected_patch_idea.get("variant_fingerprint"),
            history_fingerprints=[],
            mode="implementation_repair",
        )
        candidate_pool = build_s2_candidate_pool(
            direction=s1_direction,
            candidates=[selected_patch_idea],
            source="s2_5_patch_only_repair_reuse",
            used_shared_memory_refs=selected_patch_idea.get("used_shared_memory_refs") if isinstance(selected_patch_idea, dict) else [],
        )
        evidence_quality = read_json(self.context.project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
        shared_memory = shared_method_memory_for_prompt(
            self.context.config,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                selected_direction=s1_direction,
                feedback=feedback,
            ),
        )
        feedback_context = build_s2_feedback_context(
            project_root=self.context.project_root,
            direction=s1_direction,
            config=self.context.config,
            shared_memory=shared_memory,
        )
        adaptive_policy = build_s2_adaptive_policy(feedback_context, self.context.config)
        scorecard = build_s2_variant_scorecard(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            selected_variant=selected_patch_idea,
            evidence_quality=evidence_quality if isinstance(evidence_quality, dict) else {},
            variant_fingerprint=variant_fingerprint_artifact,
            planner_memory=self._load_c2c_s2_planner_memory(),
            feedback=self._load_c2c_plan_feedback(),
            feedback_context=feedback_context,
            adaptive_policy=adaptive_policy,
            config=self.context.config,
        )
        score_adjustment_report = build_s2_score_adjustment_report(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            scorecard=scorecard,
            adaptive_policy=adaptive_policy,
            feedback_context=feedback_context,
        )
        planner_gate_report = build_s2_planner_gate_report(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            scorecard=scorecard,
            next_variant=selected_patch_idea,
            variant_contract=variant_contract,
            variant_fingerprint=variant_fingerprint_artifact,
            adaptive_policy=adaptive_policy,
            score_adjustment_report=score_adjustment_report,
            config=self.context.config,
        )
        plan["planner_decision"] = planner_decision
        plan["variant_contract"] = variant_contract
        plan["variant_fingerprint"] = variant_fingerprint_artifact
        plan["s2_planner"] = {
            "candidate_pool": candidate_pool,
            "feedback_context": feedback_context,
            "adaptive_policy": adaptive_policy,
            "variant_scorecard": scorecard,
            "score_adjustment_report": score_adjustment_report,
            "planner_gate_report": planner_gate_report,
        }
        candidate_pool_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/candidate_pool.json",
            candidate_pool,
            artifact_type="c2c_s2_candidate_pool",
            summary="S2a candidate pool reused for S2.5-only implementation repair",
            source_paths=["plan/performance_feedback.json", "plan/s2_planner/candidate_pool.json"],
        )
        feedback_context_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/feedback_context.json",
            feedback_context,
            artifact_type="c2c_s2_feedback_context",
            summary="S2 feedback context for S2.5-only implementation repair",
            source_paths=["plan/performance_feedback.json", "meta/route_outcome.json", "meta/research_state.json"],
        )
        adaptive_policy_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/adaptive_policy.json",
            adaptive_policy,
            artifact_type="c2c_s2_adaptive_policy",
            summary="S2 adaptive policy recorded for S2.5-only implementation repair",
            source_paths=[feedback_context_record["path"]],
        )
        scorecard_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/variant_scorecard.json",
            scorecard,
            artifact_type="c2c_s2_variant_scorecard",
            summary="S2b deterministic scorecard for the S2.5-only repair variant",
            source_paths=[candidate_pool_record["path"], feedback_context_record["path"], adaptive_policy_record["path"], "plan/s2_5_repair_dispatch.json"],
        )
        score_adjustment_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/score_adjustment_report.json",
            score_adjustment_report,
            artifact_type="c2c_s2_score_adjustment_report",
            summary="Adaptive score adjustment report for the S2.5-only repair variant",
            source_paths=[candidate_pool_record["path"], adaptive_policy_record["path"], scorecard_record["path"]],
        )
        planner_gate_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/planner_gate_report.json",
            planner_gate_report,
            artifact_type="c2c_s2_planner_gate_report",
            summary="S2c planner gate report for S2.5-only implementation repair",
            source_paths=[candidate_pool_record["path"], scorecard_record["path"], score_adjustment_record["path"]],
        )
        s2_next_variant_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/selected_variant_report.json",
            selected_patch_idea,
            artifact_type="c2c_s2_selected_next_variant",
            summary="Locked selected variant reused by S2.5-only implementation repair",
            source_paths=[planner_gate_record["path"]],
        )
        legacy_next_variant_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_selection_report.json",
            planner_decision,
            artifact_type="c2c_s2_next_variant",
            summary="Compatibility next-variant mirror for S2.5-only implementation repair",
            source_paths=[s2_next_variant_record["path"]],
        )
        if planner_gate_report.get("gate") == "pass":
            implementation_contract = build_s2_implementation_contract(
                direction=s1_direction,
                selected_variant=selected_patch_idea,
                variant_contract=variant_contract,
                planner_gate_report=planner_gate_report,
                config=self.context.config,
            )
            code_patch_manifest = CodePatchAgent(self.context.project_root, self.context.config, self.context.artifacts).run_selected_variant(
                plan,
                selected_patch_idea,
                implementation_contract,
                planner_gate_report,
                variant_fingerprint_artifact,
            )
        else:
            code_patch_manifest = {
                "status": "planner_gate_failed",
                "artifacts": [],
                "selected_candidate_id": selected_patch_idea.get("id"),
                "valid_patch_count": 0,
                "valid_patch_ids": [],
                "planner_gate_report": planner_gate_report,
            }
        patch_eligible = code_patch_manifest.get("status") == "ok"
        plan["s2_5_patch_only_repair"]["patch_eligible_for_s3"] = patch_eligible
        plan["s2_5_patch_only_repair"]["implementation_blocked"] = not patch_eligible
        plan["candidate_ideas"] = patch_ideas
        plan["code_patch_manifest"] = {
            "status": code_patch_manifest.get("status"),
            "path": "plan/code_patches/patch_manifest.json" if code_patch_manifest.get("status") not in {"disabled", "planner_gate_failed"} else "",
            "selected_candidate_id": code_patch_manifest.get("selected_candidate_id"),
            "valid_patch_count": code_patch_manifest.get("valid_patch_count"),
            "valid_patch_ids": code_patch_manifest.get("valid_patch_ids") or [],
            "patch_eligible_for_s3": patch_eligible,
            "implementation_blocked": not patch_eligible,
            "planner_gate": planner_gate_report.get("gate"),
        }
        planner_decision_record = self.context.artifacts.write_json(
            self.stage_key,
            "planner_decision.json",
            planner_decision,
            artifact_type="s2_planner_decision",
            summary="S2.5-only implementation repair planner decision",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json", "plan/s2_planner/candidate_pool.json"],
        )
        variant_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant.json",
            variant_contract,
            artifact_type="s2_variant_contract",
            summary="S2.5-only implementation repair variant contract",
            source_paths=["plan/performance_feedback.json", "plan/s2_planner/candidate_pool.json"],
        )
        variant_fingerprint_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_fingerprint.json",
            variant_fingerprint_artifact,
            artifact_type="s2_variant_fingerprint",
            summary="S2.5-only implementation repair variant fingerprint",
            source_paths=["plan/performance_feedback.json", "plan/s2_planner/candidate_pool.json"],
        )
        plan_record = self.context.artifacts.write_json(
            self.stage_key,
            "trial_spec.json",
            _trial_spec_from_plan(plan, variant_contract, profile=_execution_profile(self.context.config), project_root=self.context.project_root),
            artifact_type="plan",
            summary="C2C plan updated after S2.5-only implementation repair",
            source_paths=["plan/performance_feedback.json", "plan/s2_planner/candidate_pool.json", "plan/code_patches/patch_manifest.json"],
        )
        patch_only_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_5_patch_only_repair.json",
            {
                "schema_version": "c2c_s2_5_patch_only_repair_v1",
                "created_at": now_utc(),
                "status": code_patch_manifest.get("status"),
                "patch_eligible_for_s3": patch_eligible,
                "implementation_blocked": not patch_eligible,
                "repair_lane": "s2_5_only_implementation_repair",
                "failure_class": "implementation_failure",
                "skipped_s2_planner": True,
                "candidate_count": len(patch_ideas),
                "selected_candidate_id": code_patch_manifest.get("selected_candidate_id"),
                "requested_candidate_id": target_candidate_id or None,
                "variant_fingerprint": _candidate_variant_fingerprint(selected_patch_idea),
                "requested_variant_fingerprint": target_fingerprint or None,
                "same_candidate_required": True,
                "same_variant_fingerprint_required": bool(target_fingerprint or _candidate_variant_fingerprint(selected_patch_idea)),
                "reuse_persistent_codex_session": True,
                "repair_until": "patch_eligible_for_s3_or_implementation_blocked",
                "valid_patch_count": code_patch_manifest.get("valid_patch_count"),
                "performance_feedback_summary": (feedback.get("summary") if isinstance(feedback, dict) else {}),
                "performance_feedback_path": "plan/performance_feedback.json",
                "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
                "s2_5_repair_dispatch": _compact_s2_5_repair_dispatch_for_plan(repair_dispatch),
                "patch_manifest_path": "plan/code_patches/patch_manifest.json",
            },
            artifact_type="c2c_s2_5_patch_only_repair",
            summary="S2.5-only patch repair for implementation failure; S2 planning was not rerun",
            source_paths=["plan/performance_feedback.json", "plan/s2_planner/candidate_pool.json", "plan/code_patches/patch_manifest.json"],
        )
        candidate_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/candidate_pool_snapshot.json",
            patch_ideas,
            artifact_type="c2c_candidate_ideas",
            summary="C2C candidate ideas reused for S2.5-only implementation repair",
            source_paths=["plan/performance_feedback.json"],
        )
        feedback_record = self.context.artifacts.write_json(
            self.stage_key,
            "plan_feedback.json",
            {
                "feedback": self._load_c2c_plan_feedback(),
                "directional_planning": {
                    "status": "skipped_s2_5_patch_only_repair",
                    "reason": "implementation_failure feedback routes directly to S2.5 patch repair",
                },
                "s2_5_patch_only_repair": {
                    "enabled": True,
                    "repair_lane": "s2_5_only_implementation_repair",
                    "skipped_s2_planner": True,
                    "selected_candidate_id": selected_patch_idea.get("id"),
                    "performance_feedback_path": "plan/performance_feedback.json",
                    "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
                    "patch_only_repair_path": patch_only_record["path"],
                },
            },
            artifact_type="c2c_plan_feedback",
            summary="C2C implementation-failure feedback used for S2.5-only patch repair",
            source_paths=["plan/performance_feedback.json", patch_only_record["path"]],
        )
        return {
            "plan": plan,
            "artifacts": [
                plan_record["path"],
                *code_patch_manifest.get("artifacts", []),
                candidate_pool_record["path"],
                feedback_context_record["path"],
                adaptive_policy_record["path"],
                scorecard_record["path"],
                score_adjustment_record["path"],
                s2_next_variant_record["path"],
                planner_gate_record["path"],
                legacy_next_variant_record["path"],
                planner_decision_record["path"],
                variant_contract_record["path"],
                variant_fingerprint_record["path"],
                patch_only_record["path"],
                candidate_record["path"],
                feedback_record["path"],
            ],
        }

    def _run_c2c_plan_without_patch_only_fallback(self, reason: str) -> dict[str, Any]:
        feedback_path = self.context.project_root / "plan" / "performance_feedback.json"
        feedback = read_json(feedback_path, default={}) if feedback_path.exists() else {}
        return self._run_c2c_patch_only_blocked(reason, feedback if isinstance(feedback, dict) else {}, _load_s2_5_repair_dispatch(self.context.project_root))

    def _run_c2c_patch_only_blocked(
        self,
        reason: str,
        feedback: dict[str, Any],
        repair_dispatch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        repair_dispatch = repair_dispatch if isinstance(repair_dispatch, dict) else {}
        plan_path = self.context.project_root / "plan" / "trial_spec.json"
        plan = read_yaml(plan_path, default={}) if plan_path.exists() else {}
        if not isinstance(plan, dict):
            plan = {}
        plan.setdefault("hypotheses", [{"id": "implementation_blocked", "statement": reason, "type": "implementation"}])
        plan.setdefault("baselines", [{"name": "current_baseline"}, {"name": "previous_patch"}])
        plan.setdefault("datasets", ["unknown"])
        plan.setdefault("task_graph", {})
        plan.setdefault("resource_budget", {})
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        execution.setdefault("collector", "c2c_small_loop")
        plan["execution"] = execution
        plan["s2_5_patch_only_repair"] = {
            "enabled": True,
            "repair_lane": "s2_5_only_implementation_repair",
            "skips_s2_planner": True,
            "failure_class": "implementation_failure",
            "status": "implementation_blocked",
            "reason": reason,
            "patch_eligible_for_s3": False,
            "implementation_blocked": True,
            "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
            "variant_fingerprint": repair_dispatch.get("variant_fingerprint"),
            "same_candidate_required": True,
            "same_variant_fingerprint_required": bool(repair_dispatch.get("variant_fingerprint")),
            "reuse_persistent_codex_session": True,
            "performance_feedback_path": "plan/performance_feedback.json" if feedback else "",
            "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
            "does_not_consume_same_direction_attempt": True,
        }
        manifest = {
            "status": "no_valid_patch",
            "created_at": now_utc(),
            "backend": "codex_persistent_cli",
            "repair_lane": "s2_5_only_implementation_repair",
            "failure_class": "implementation_failure",
            "patch_eligible_for_s3": False,
            "implementation_blocked": True,
            "reason": reason,
            "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
            "valid_patch_count": 0,
            "candidate_count": 0,
            "input_candidate_count": 0,
            "failed_patch_count": 0,
            "retryable_patch_count": 0,
            "candidates": [],
            "patches": [],
        }
        patch_manifest_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_manifest.json",
            manifest,
            artifact_type="c2c_code_patch_manifest",
            summary="S2.5-only implementation repair blocked before patch generation",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        plan["code_patch_manifest"] = {
            "status": "no_valid_patch",
            "path": "plan/code_patches/patch_manifest.json",
            "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
            "valid_patch_count": 0,
            "valid_patch_ids": [],
            "patch_eligible_for_s3": False,
            "implementation_blocked": True,
        }
        s1_direction = load_direction(self.context.project_root)
        repair_variant = {
            "id": repair_dispatch.get("selected_candidate_id") or "implementation_blocked",
            "title": "Implementation repair blocked",
            "selected": True,
            "s1_direction_id": s1_direction.get("direction_id"),
            "variant_fingerprint": repair_dispatch.get("variant_fingerprint") or "implementation_blocked",
            "mechanism_axis": s1_direction.get("mechanism_axis"),
            "integration_point": s1_direction.get("integration_point"),
            "control_signal": s1_direction.get("control_signal"),
            "hypothesis": s1_direction.get("hypothesis") or reason,
            "expected_files": s1_direction.get("expected_files") or ["S2.5 repair target unavailable"],
            "expected_signature": s1_direction.get("expected_metric_signature") or {},
        }
        _sanitize_c2c_variant_expected_files(repair_variant, s1_direction, self.context.config)
        planner_decision = build_planner_decision_artifact(
            direction=s1_direction,
            planner_summary="S2 planner skipped; implementation repair blocked before patch generation.",
            planning_mode="s2_5_only_implementation_repair_blocked",
            next_variant=repair_variant,
            used_shared_memory_refs=s1_direction.get("used_shared_memory_refs") if isinstance(s1_direction, dict) else [],
            source="s2_5_patch_only_blocked",
        )
        plan["project_root"] = str(self.context.project_root)
        variant_contract = build_variant_contract(direction=s1_direction, variant=repair_variant, plan=plan, mode="implementation_repair")
        variant_fingerprint_artifact = build_variant_fingerprint_artifact(
            direction=s1_direction,
            variant=repair_variant,
            fingerprint=repair_variant.get("variant_fingerprint"),
            history_fingerprints=[],
            mode="implementation_repair",
        )
        candidate_pool = build_s2_candidate_pool(
            direction=s1_direction,
            candidates=[repair_variant],
            source="s2_5_patch_only_blocked",
            used_shared_memory_refs=s1_direction.get("used_shared_memory_refs") if isinstance(s1_direction, dict) else [],
        )
        evidence_quality = read_json(self.context.project_root / "literature" / "c2c" / "evidence_quality_score.json", default={}) or {}
        scorecard = build_s2_variant_scorecard(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            selected_variant=repair_variant,
            evidence_quality=evidence_quality if isinstance(evidence_quality, dict) else {},
            variant_fingerprint=variant_fingerprint_artifact,
            planner_memory=self._load_c2c_s2_planner_memory(),
            feedback=self._load_c2c_plan_feedback(),
            config=self.context.config,
        )
        planner_gate_report = build_s2_planner_gate_report(
            direction=s1_direction,
            candidate_pool=candidate_pool,
            scorecard=scorecard,
            next_variant=repair_variant,
            variant_contract=variant_contract,
            variant_fingerprint=variant_fingerprint_artifact,
            config=self.context.config,
        )
        implementation_contract = build_s2_implementation_contract(
            direction=s1_direction,
            selected_variant=repair_variant,
            variant_contract=variant_contract,
            planner_gate_report=planner_gate_report,
            config=self.context.config,
        )
        patch_gate_report = build_s2_5_patch_gate_report(
            patch_manifest=manifest,
            implementation_contract=implementation_contract,
            planner_gate_report=planner_gate_report,
            variant_fingerprint=variant_fingerprint_artifact,
            config=self.context.config,
        )
        plan["planner_decision"] = planner_decision
        plan["variant_contract"] = variant_contract
        plan["variant_fingerprint"] = variant_fingerprint_artifact
        plan["s2_planner"] = {
            "candidate_pool": candidate_pool,
            "variant_scorecard": scorecard,
            "planner_gate_report": planner_gate_report,
        }
        candidate_pool_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/candidate_pool.json",
            candidate_pool,
            artifact_type="c2c_s2_candidate_pool",
            summary="S2a blocked implementation-repair candidate pool",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        scorecard_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/variant_scorecard.json",
            scorecard,
            artifact_type="c2c_s2_variant_scorecard",
            summary="S2b blocked implementation-repair variant scorecard",
            source_paths=[candidate_pool_record["path"]],
        )
        planner_gate_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/planner_gate_report.json",
            planner_gate_report,
            artifact_type="c2c_s2_planner_gate_report",
            summary="S2c blocked implementation-repair planner gate report",
            source_paths=[candidate_pool_record["path"], scorecard_record["path"]],
        )
        s2_next_variant_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_planner/selected_variant_report.json",
            repair_variant,
            artifact_type="c2c_s2_selected_next_variant",
            summary="Blocked implementation-repair selected variant",
            source_paths=[planner_gate_record["path"]],
        )
        legacy_next_variant_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_selection_report.json",
            planner_decision,
            artifact_type="c2c_s2_next_variant",
            summary="Compatibility next-variant mirror for blocked implementation repair",
            source_paths=[s2_next_variant_record["path"]],
        )
        implementation_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/implementation_contract.json",
            implementation_contract,
            artifact_type="c2c_s2_5_implementation_contract",
            summary="S2.5 implementation contract for blocked repair classification",
            source_paths=[planner_gate_record["path"]],
        )
        patch_gate_record = self.context.artifacts.write_json(
            self.stage_key,
            "code_patches/patch_gate_report.json",
            patch_gate_report,
            artifact_type="c2c_s2_5_patch_gate_report",
            summary="S2.5 patch gate report for blocked repair classification",
            source_paths=[implementation_contract_record["path"], patch_manifest_record["path"]],
        )
        planner_decision_record = self.context.artifacts.write_json(
            self.stage_key,
            "planner_decision.json",
            planner_decision,
            artifact_type="s2_planner_decision",
            summary="Blocked S2.5-only implementation repair planner decision",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        variant_contract_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant.json",
            variant_contract,
            artifact_type="s2_variant_contract",
            summary="Blocked S2.5-only implementation repair variant contract",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        variant_fingerprint_record = self.context.artifacts.write_json(
            self.stage_key,
            "variant_fingerprint.json",
            variant_fingerprint_artifact,
            artifact_type="s2_variant_fingerprint",
            summary="Blocked S2.5-only implementation repair variant fingerprint",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        plan_record = self.context.artifacts.write_json(
            self.stage_key,
            "trial_spec.json",
            _trial_spec_from_plan(plan, variant_contract, profile=_execution_profile(self.context.config), project_root=self.context.project_root),
            artifact_type="plan",
            summary="C2C S2.5-only implementation repair blocked",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json"],
        )
        blocked_record = self.context.artifacts.write_json(
            self.stage_key,
            "s2_5_patch_only_repair.json",
            {
                "schema_version": "c2c_s2_5_patch_only_repair_v1",
                "created_at": now_utc(),
                "status": "implementation_blocked",
                "repair_lane": "s2_5_only_implementation_repair",
                "failure_class": "implementation_failure",
                "skipped_s2_planner": True,
                "patch_eligible_for_s3": False,
                "implementation_blocked": True,
                "reason": reason,
                "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
                "variant_fingerprint": repair_dispatch.get("variant_fingerprint"),
                "reuse_persistent_codex_session": True,
                "does_not_consume_same_direction_attempt": True,
                "performance_feedback_summary": (feedback.get("summary") if isinstance(feedback, dict) else {}),
                "performance_feedback_path": "plan/performance_feedback.json" if feedback else "",
                "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
                "s2_5_repair_dispatch": _compact_s2_5_repair_dispatch_for_plan(repair_dispatch),
                "patch_manifest_path": "plan/code_patches/patch_manifest.json",
            },
            artifact_type="c2c_s2_5_patch_only_repair",
            summary="S2.5-only implementation repair blocked without rerunning S2 planner",
            source_paths=["plan/performance_feedback.json", "plan/s2_5_repair_dispatch.json", "plan/code_patches/patch_manifest.json"],
        )
        return {
            "plan": plan,
            "artifacts": [
                plan_record["path"],
                patch_manifest_record["path"],
                candidate_pool_record["path"],
                scorecard_record["path"],
                s2_next_variant_record["path"],
                planner_gate_record["path"],
                legacy_next_variant_record["path"],
                implementation_contract_record["path"],
                patch_gate_record["path"],
                planner_decision_record["path"],
                variant_contract_record["path"],
                variant_fingerprint_record["path"],
                blocked_record["path"],
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
        fallback = [idea for idea in fallback if idea.get("s1_direction_id") == selected.get("id") or idea.get("id") == selected.get("id")]
        if not fallback:
            fallback = self._normalize_c2c_plan_ideas([selected], baseline=baseline, selected_id=selected.get("id"), s1_selected=selected)
        if not enabled:
            s1_used_shared_memory_refs = list(selected.get("used_shared_memory_refs") or [])
            fallback_selection = _build_s2_variant_selection(
                fallback,
                selected=selected,
                planner_memory=planner_memory,
                feedback=feedback,
                max_selected=len(fallback),
                source="fallback_s1_ideas",
                used_shared_memory_refs=s1_used_shared_memory_refs,
            )
            return {
                "ideas": fallback_selection["selected_ideas"] or fallback,
                "variant_selection": fallback_selection,
                "metadata": {
                    "enabled": False,
                    "status": "disabled",
                    "source": "s1_ideas",
                    "candidate_count": len(fallback_selection["selected_ideas"] or fallback),
                    "candidate_pool_count": fallback_selection.get("candidate_pool_count"),
                    "next_variant_id": (fallback_selection.get("next_variant") or {}).get("id") if isinstance(fallback_selection.get("next_variant"), dict) else None,
                    "used_shared_memory_refs": s1_used_shared_memory_refs,
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
        resume_metadata = resume_result.get("metadata") if isinstance(resume_result.get("metadata"), dict) else {}
        resume_disabled = resume_result.get("status") == "disabled" or resume_metadata.get("enabled") is False
        if (
            getattr(self.context.llm, "use_real_api", False)
            and not resume_disabled
            and cfg.get("fallback_to_gpt_after_resume_failure", False) is not True
        ):
            s1_used_shared_memory_refs = list(selected.get("used_shared_memory_refs") or [])
            fallback_selection = _build_s2_variant_selection(
                fallback,
                selected=selected,
                planner_memory=planner_memory,
                feedback=feedback,
                max_selected=len(fallback),
                source="fallback_s1_ideas",
                used_shared_memory_refs=s1_used_shared_memory_refs,
            )
            return {
                "ideas": fallback_selection["selected_ideas"] or fallback,
                "variant_selection": fallback_selection,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_resume_planner_unavailable",
                    "source": "s1_ideas",
                    "reason": "Codex resume planner did not produce acceptable fresh candidates; skipped GPT fallback in real API mode.",
                    "candidate_count": len(fallback_selection["selected_ideas"] or fallback),
                    "candidate_pool_count": fallback_selection.get("candidate_pool_count"),
                    "next_variant_id": (fallback_selection.get("next_variant") or {}).get("id") if isinstance(fallback_selection.get("next_variant"), dict) else None,
                    "used_shared_memory_refs": s1_used_shared_memory_refs,
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                    "resume_planner": resume_result.get("metadata") if isinstance(resume_result.get("metadata"), dict) else None,
                },
            }
        if not getattr(self.context.llm, "use_real_api", False):
            s1_used_shared_memory_refs = list(selected.get("used_shared_memory_refs") or [])
            fallback_selection = _build_s2_variant_selection(
                fallback,
                selected=selected,
                planner_memory=planner_memory,
                feedback=feedback,
                max_selected=len(fallback),
                source="fallback_s1_ideas",
                used_shared_memory_refs=s1_used_shared_memory_refs,
            )
            return {
                "ideas": fallback_selection["selected_ideas"] or fallback,
                "variant_selection": fallback_selection,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_no_real_llm",
                    "source": "s1_ideas",
                    "candidate_count": len(fallback_selection["selected_ideas"] or fallback),
                    "candidate_pool_count": fallback_selection.get("candidate_pool_count"),
                    "next_variant_id": (fallback_selection.get("next_variant") or {}).get("id") if isinstance(fallback_selection.get("next_variant"), dict) else None,
                    "used_shared_memory_refs": s1_used_shared_memory_refs,
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                    "resume_planner": resume_result.get("metadata") if isinstance(resume_result.get("metadata"), dict) else None,
                },
            }
        max_candidates = int(cfg.get("max_candidates") or self.context.config.get("c2c", {}).get("small_loop", {}).get("max_candidates") or 3)
        shared_method_memory = shared_method_memory_for_prompt(
            self.context.config,
            limit=12,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                selected_direction=selected,
                feedback=feedback,
                negative_memory=negative_memory,
            ),
        )
        prompt = {
            "objective": "Generate exactly one next same-direction S2 experiment variant for this iteration, not a batch and not a new S1 direction.",
            "s1_selected_direction": _compact_for_plan_prompt(selected, 5000),
            "s1_candidate_pool": _compact_for_plan_prompt(s1_ideas, 6000),
            "baseline": baseline,
            "negative_memory": _compact_for_plan_prompt(negative_memory, 4000),
            "shared_method_failure_memory": _compact_for_plan_prompt(shared_method_memory, 7000),
            "s2_planner_memory": _compact_for_plan_prompt(_c2c_s2_memory_for_prompt(planner_memory), int(cfg.get("memory_prompt_chars") or 6000)),
            "available_artifacts": _c2c_s2_available_artifacts(),
            "failure_feedback": _compact_for_plan_prompt(feedback, 9000),
            "requirements": [
                "Stay inside the S1 selected mechanism direction unless performance_feedback explicitly says return_to_s1_new_direction.",
                "Propose one next_variant only. Do not output a batch of variants; if legacy variant_candidates/candidates are returned, only one will be executed.",
                "Use the current S1 direction, same-direction attempt count, latest method-level S3/proxy feedback, s2_planner_memory, and top shared method memory.",
                "Explain why this next_variant is the right next branch after the previous result.",
                "Make the next_variant materially different from previous same-direction attempts by changing at least one of mechanism_axis, integration_point, or control_signal.",
                "If performance_feedback.summary.s2_action_policy is present, follow its action and matched_rule when choosing mechanism_repair or new_same_direction_variant.",
                "Implementation_failure should not reach S2 planning; if it appears in feedback, treat it only as implementation noise and keep the same method hypothesis.",
                "Use s2_planner_memory first to avoid recreating variants already tried in this S1 direction.",
                "If s2_planner_memory is insufficient, use failure_feedback and cite available_artifacts paths in failure_feedback_refs.",
                "Do not change evaluator, datasets, baseline, or cheap proxy protocol.",
                "Do not propose pure threshold/top-k/fallback tuning.",
                "The next_variant must include id,title,why_next,mechanism_axis,integration_point,control_signal,expected_dataset_tradeoff,risk_budget,anti_repeat,experiment_contract.config_overrides,ablation_switch,expected_files,implementation_plan,expected_signature,and failure_avoidance.",
                "Use performance feedback as performance evidence; ignore low-level S2.5 implementation errors unless they imply a method-level risk.",
                "shared_method_failure_memory.memory_catalog/recent_entries is a lightweight retrieved error catalog, ranked by memory_retrieval.combined_score and memory_quality.priority; follow retrieval_policy/ranking_policy and prioritize high_quality_memory_ids, especially proxy_full_false_positive, full_train_failure, proxy_dataset_misprediction, cross_project_mechanism_failure, and ablation_evidence when choosing anti_repeat rules. If a catalog item seems decision-relevant, inspect the full memory via full_memory_access/read_hint before relying on detailed evidence.",
                "If shared_method_failure_memory affects the next_variant, anti_repeat rule, or forbidden pattern, copy the exact memory_id values into used_shared_memory_refs at top-level and inside next_variant. Use [] if none affected the decision.",
            ],
            "return_shape": {
                "planner_summary": "short explanation",
                "planning_mode": "same_direction_variant|new_direction_after_budget|fallback",
                "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this S2 decision, or []"],
                "next_variant": {
                    "id": "snake_case_id",
                    "title": "short title",
                    "why_next": "why this is the next branch after the latest S3/proxy result",
                    "mechanism_axis": "scoring|routing|span_selection|normalization|training_signal|fallback",
                    "integration_point": "aligner|projector|wrapper|train_loss|recipe",
                    "control_signal": "confidence|entropy|span_agreement|utility|pathology|semantic_similarity",
                    "expected_dataset_tradeoff": {"mmlu-redux": "up|flat|risk", "ai2-arc": "up|flat|risk", "openbookqa": "up|flat|risk"},
                    "risk_budget": {"max_changed_files": 2, "forbidden_files": ["script/evaluation/*"], "risk_notes": []},
                    "anti_repeat": "how this differs from prior failed variants",
                    "description": "mechanism variant",
                    "hypothesis": "proxy-testable hypothesis",
                    "mechanism_type": "same mechanism type as S1 direction when same-direction",
                    "experiment_contract": {},
                    "implementation_plan": {},
                    "failure_feedback_refs": [],
                    "used_shared_memory_refs": ["memory_id values that influenced this variant, or []"],
                    "failure_avoidance": [],
                },
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
                schema={"type": "object"},
                agent_name="c2c-s2-directional-planner",
            )
        except Exception as exc:
            s1_used_shared_memory_refs = list(selected.get("used_shared_memory_refs") or [])
            fallback_selection = _build_s2_variant_selection(
                fallback,
                selected=selected,
                planner_memory=planner_memory,
                feedback=feedback,
                max_selected=len(fallback),
                source="fallback_s1_ideas",
                used_shared_memory_refs=s1_used_shared_memory_refs,
            )
            return {
                "ideas": fallback_selection["selected_ideas"] or fallback,
                "variant_selection": fallback_selection,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_llm_error",
                    "source": "s1_ideas",
                    "reason": str(exc)[-500:],
                    "candidate_count": len(fallback_selection["selected_ideas"] or fallback),
                    "candidate_pool_count": fallback_selection.get("candidate_pool_count"),
                    "next_variant_id": (fallback_selection.get("next_variant") or {}).get("id") if isinstance(fallback_selection.get("next_variant"), dict) else None,
                    "used_shared_memory_refs": s1_used_shared_memory_refs,
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                },
            }
        planned = _payload_variant_candidates(payload)
        used_shared_memory_refs = sorted(set(collect_used_shared_memory_refs(payload, shared_method_memory) + list(selected.get("used_shared_memory_refs") or [])))
        normalized = self._normalize_c2c_plan_ideas(
            planned if isinstance(planned, list) else [],
            baseline=baseline,
            selected_id=None,
            s1_selected=selected,
            max_candidates=max(max_candidates, int(cfg.get("max_variant_candidates") or 5)),
        )
        accepted = [idea for idea in normalized if self._c2c_plan_candidate_acceptable(idea, selected)]
        if not accepted:
            s1_used_shared_memory_refs = list(selected.get("used_shared_memory_refs") or [])
            fallback_selection = _build_s2_variant_selection(
                fallback,
                selected=selected,
                planner_memory=planner_memory,
                feedback=feedback,
                max_selected=len(fallback),
                source="fallback_s1_ideas",
                used_shared_memory_refs=s1_used_shared_memory_refs,
            )
            return {
                "ideas": fallback_selection["selected_ideas"] or fallback,
                "variant_selection": fallback_selection,
                "metadata": {
                    "enabled": True,
                    "status": "fallback_invalid_planner_output",
                    "source": "s1_ideas",
                    "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                    "rejected_count": len(normalized),
                    "candidate_count": len(fallback_selection["selected_ideas"] or fallback),
                    "candidate_pool_count": fallback_selection.get("candidate_pool_count"),
                    "next_variant_id": (fallback_selection.get("next_variant") or {}).get("id") if isinstance(fallback_selection.get("next_variant"), dict) else None,
                    "used_shared_memory_refs": s1_used_shared_memory_refs,
                    "memory_entry_count": len(planner_memory.get("entries") or []),
                },
            }
        variant_selection = _build_s2_variant_selection(
            accepted,
            selected=selected,
            planner_memory=planner_memory,
            feedback=feedback,
            max_selected=max_candidates,
            source="directional_planner",
            used_shared_memory_refs=used_shared_memory_refs,
        )
        selected_ideas = variant_selection["selected_ideas"]
        for idx, idea in enumerate(selected_ideas):
            idea["selected"] = idx == 0
            idea["used_shared_memory_refs"] = list(idea.get("used_shared_memory_refs") or used_shared_memory_refs)
            idea.setdefault("s2_planner", {})
            idea["s2_planner"].update(
                {
                    "source": "directional_planner",
                    "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                    "s1_direction_id": selected.get("id"),
                    "s1_mechanism_type": selected.get("mechanism_type"),
                    "used_shared_memory_refs": list(idea.get("used_shared_memory_refs") or []),
                }
            )
        return {
            "ideas": selected_ideas,
            "variant_selection": variant_selection,
            "metadata": {
                "enabled": True,
                "status": "ok",
                "source": "directional_planner",
                "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                "candidate_count": len(selected_ideas),
                "candidate_pool_count": variant_selection.get("candidate_pool_count"),
                "next_variant_id": (variant_selection.get("next_variant") or {}).get("id") if isinstance(variant_selection.get("next_variant"), dict) else None,
                "s1_direction_id": selected.get("id"),
                "used_shared_memory_refs": used_shared_memory_refs,
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
        shared_method_memory = shared_method_memory_for_prompt(
            self.context.config,
            limit=12,
            query_context=shared_method_memory_query_context(
                self.context.config,
                project_root=self.context.project_root,
                selected_direction=selected,
                feedback=feedback,
                negative_memory=negative_memory,
            ),
        )
        used_shared_memory_refs = sorted(set(collect_used_shared_memory_refs(payload, shared_method_memory) + list(selected.get("used_shared_memory_refs") or [])))
        metadata["used_shared_memory_refs"] = used_shared_memory_refs
        planned = _payload_variant_candidates(payload)
        normalized = self._normalize_c2c_plan_ideas(
            planned if isinstance(planned, list) else [],
            baseline=baseline,
            selected_id=None,
            s1_selected=selected,
            max_candidates=max(max_candidates, int(cfg.get("max_variant_candidates") or 5)),
        )
        accepted = [idea for idea in normalized if self._c2c_plan_candidate_acceptable(idea, selected)]
        if not accepted:
            reject_report = self._c2c_plan_reject_report(normalized, selected)
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
            metadata["reject_report"] = reject_report
            metadata["planner_summary"] = payload.get("planner_summary") if isinstance(payload, dict) else None
            metadata["session_health"] = health.get("health")
            metadata["session_reset"] = health.get("session_reset")
            metadata["session_reset_reason"] = health.get("session_reset_reason")
            return {"status": "invalid", "metadata": metadata}
        variant_selection = _build_s2_variant_selection(
            accepted,
            selected=selected,
            planner_memory=planner_memory,
            feedback=feedback,
            max_selected=max_candidates,
            source="codex_resume_planner",
            used_shared_memory_refs=used_shared_memory_refs,
        )
        selected_ideas = variant_selection["selected_ideas"]
        candidate_ids = _s2_planner_candidate_ids(selected_ideas)
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
                    "accepted_count": len(selected_ideas),
                    "duplicate_candidate_ids": candidate_ids,
                    "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                    "session_health": health.get("health"),
                    "session_reset": True,
                    "session_reset_reason": health.get("session_reset_reason"),
                }
            )
            return {"status": "invalid", "metadata": metadata}
        for idx, idea in enumerate(selected_ideas):
            idea["selected"] = idx == 0
            idea["used_shared_memory_refs"] = list(idea.get("used_shared_memory_refs") or used_shared_memory_refs)
            idea.setdefault("s2_planner", {})
            idea["s2_planner"].update(
                {
                    "source": "codex_resume_planner",
                    "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                    "s1_direction_id": selected.get("id"),
                    "s1_mechanism_type": selected.get("mechanism_type"),
                    "session_key": session_key,
                    "used_shared_memory_refs": list(idea.get("used_shared_memory_refs") or []),
                }
            )
        metadata.update(
            {
                "status": "ok",
                "planner_summary": payload.get("planner_summary") if isinstance(payload, dict) else None,
                "planning_mode": payload.get("planning_mode") if isinstance(payload, dict) else None,
                "candidate_count": len(selected_ideas),
                "candidate_pool_count": variant_selection.get("candidate_pool_count"),
                "next_variant_id": (variant_selection.get("next_variant") or {}).get("id") if isinstance(variant_selection.get("next_variant"), dict) else None,
                "s1_direction_id": selected.get("id"),
                "used_shared_memory_refs": used_shared_memory_refs,
                "duplicate_output": duplicate_output,
                "session_health": health.get("health"),
                "session_reset": health.get("session_reset"),
            }
        )
        return {"status": "ok", "ideas": selected_ideas, "variant_selection": variant_selection, "metadata": metadata}

    def _load_c2c_s2_planner_memory(self) -> dict[str, Any]:
        state = ResearchEventLedger(self.context.project_root).state()
        variants = state.get("variants") if isinstance(state.get("variants"), dict) else {}
        entries = []
        for outcome in state.get("method_tried_history") or []:
            if not isinstance(outcome, dict) or outcome.get("method_evaluable") is not True:
                continue
            variant = variants.get(outcome.get("variant_spec_hash")) if isinstance(variants, dict) else {}
            entries.append(
                {
                    "selected_candidate": {
                        "id": outcome.get("variant_id"),
                        "variant_fingerprint": outcome.get("variant_spec_hash"),
                        "integration_point": ((variant or {}).get("implementation_surface_ids") or [None])[0],
                    },
                    "method_outcome": outcome.get("outcome_classification"),
                    "attempt_id": outcome.get("attempt_id"),
                }
            )
        return {
            "schema_version": "c2c_s2_planner_memory_v2",
            "project_id": self.context.project_root.name,
            "entries": entries,
            "compacted_summary": {"method_evaluable_outcomes": len(entries)},
        }

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
                "selected_candidate_id": code_patch_manifest.get("selected_candidate_id"),
                "valid_patch_ids": code_patch_manifest.get("valid_patch_ids") or [],
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
            source_paths=["meta/research_events.sqlite3", "plan/performance_feedback.json", "experiment/results/failure_feedback.json"],
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
                raw_mechanism = str(item.get("mechanism_type") or "")
                if not raw_mechanism or raw_mechanism == s1_selected.get("id") or raw_mechanism == s1_selected.get("direction_id"):
                    item["mechanism_type"] = s1_selected.get("mechanism_type")
                else:
                    item.setdefault("mechanism_type", s1_selected.get("mechanism_type"))
                _inherit_if_present(item, s1_selected, "paper_claim")
                _inherit_if_present(item, s1_selected, "why_baseline_fails")
                _inherit_if_present(item, s1_selected, "coverage_diagnostics")
                _inherit_if_present(item, s1_selected, "matched_coverage_ablation")
                _inherit_if_present(item, s1_selected, "expected_files")
                _inherit_if_present(item, s1_selected, "evidence_refs")
                _inherit_if_present(item, s1_selected, "counterevidence_refs")
                _inherit_if_present(item, s1_selected, "code_refs")
                _inherit_if_present(item, s1_selected, "s1_allowed_variants")
                _inherit_if_present(item, s1_selected, "s1_forbidden_patterns")
                item.setdefault("s1_direction_id", s1_selected.get("s1_direction_id") or s1_selected.get("direction_id") or s1_selected.get("id"))
                item.setdefault("implementation_plan", base_impl)
                contract = item.get("experiment_contract") if isinstance(item.get("experiment_contract"), dict) else {}
                merged_contract = dict(base_contract)
                merged_contract.update(contract)
                item["experiment_contract"] = merged_contract
            item = normalize_c2c_mechanism_fields(item, baseline)
            if s1_selected and item.get("mechanism_type") != s1_selected.get("mechanism_type"):
                item["mechanism_type"] = s1_selected.get("mechanism_type")
                item = normalize_c2c_mechanism_fields(item, baseline)
            _ensure_c2c_s2_config_overrides(item)
            item["novelty_gate"] = c2c_idea_novelty_report(item)
            item["implementation_scope_gate"] = c2c_implementation_scope_report(item)
            item["selected"] = bool(item.get("selected") or (selected_id and item.get("id") == selected_id) or (selected_id is None and idx == 0))
            normalized.append(item)
        if normalized and not any(item.get("selected") for item in normalized):
            normalized[0]["selected"] = True
        return normalized

    @staticmethod
    def _c2c_plan_candidate_acceptable(idea: dict[str, Any], selected: dict[str, Any]) -> bool:
        return not PlanAgent._c2c_plan_candidate_reject_reasons(idea, selected)

    @staticmethod
    def _c2c_plan_candidate_reject_reasons(idea: dict[str, Any], selected: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not isinstance(idea, dict):
            return ["not_object"]
        if selected.get("mechanism_type") and idea.get("mechanism_type") != selected.get("mechanism_type"):
            reasons.append("mechanism_type_mismatch")
        if (idea.get("novelty_gate") or {}).get("status") != "pass":
            reasons.append("novelty_gate_not_pass")
        if (idea.get("implementation_scope_gate") or {}).get("status") != "pass":
            reasons.append("implementation_scope_gate_not_pass")
        contract = idea.get("experiment_contract") if isinstance(idea.get("experiment_contract"), dict) else {}
        if not contract.get("ablation_switch"):
            reasons.append("missing_ablation_switch")
        if not (contract.get("expected_files") or idea.get("expected_files")):
            reasons.append("missing_expected_files")
        return reasons

    @staticmethod
    def _c2c_plan_reject_report(ideas: list[dict[str, Any]], selected: dict[str, Any]) -> list[dict[str, Any]]:
        report = []
        for idea in ideas:
            if not isinstance(idea, dict):
                report.append({"id": None, "reasons": ["not_object"]})
                continue
            report.append(
                {
                    "id": idea.get("id"),
                    "mechanism_type": idea.get("mechanism_type"),
                    "s1_direction_id": idea.get("s1_direction_id"),
                    "reasons": PlanAgent._c2c_plan_candidate_reject_reasons(idea, selected),
                    "novelty_status": (idea.get("novelty_gate") or {}).get("status") if isinstance(idea.get("novelty_gate"), dict) else None,
                    "scope_status": (idea.get("implementation_scope_gate") or {}).get("status") if isinstance(idea.get("implementation_scope_gate"), dict) else None,
                    "missing_required_fields": (idea.get("novelty_gate") or {}).get("missing_required_fields") if isinstance(idea.get("novelty_gate"), dict) else [],
                    "blocked_reasons": (idea.get("implementation_scope_gate") or {}).get("blocked_reasons") if isinstance(idea.get("implementation_scope_gate"), dict) else [],
                }
            )
        return report

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
                        "reason": "Fast local baseline and ablation benchmark.",
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
                resource_part = f", gpu={task['gpu']}" if "gpu" in task else ""
                lines.append(
                    f"- {task['task']}: {task['estimated_hours']}h{resource_part}, depends_on={task.get('depends_on', [])}"
                )
            lines.append("")
        return "\n".join(lines)

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
    variant = idea.get("s2_variant") if isinstance(idea.get("s2_variant"), dict) else {}
    return {
        "id": idea.get("id"),
        "title": idea.get("title"),
        "mechanism_type": idea.get("mechanism_type"),
        "selected": bool(idea.get("selected")),
        "variant_fingerprint": variant.get("variant_fingerprint") or idea.get("variant_fingerprint"),
        "mechanism_axis": variant.get("mechanism_axis") or idea.get("mechanism_axis"),
        "integration_point": variant.get("integration_point") or idea.get("integration_point"),
        "control_signal": variant.get("control_signal") or idea.get("control_signal"),
        "variant_score": ((variant.get("variant_score") or {}).get("score") if isinstance(variant.get("variant_score"), dict) else None),
        "hypothesis": _short_text(str(idea.get("hypothesis") or ""), 500),
        "failure_avoidance": list(idea.get("failure_avoidance") or [])[:5] if isinstance(idea.get("failure_avoidance"), list) else [],
        "used_shared_memory_refs": list(idea.get("used_shared_memory_refs") or [])[:12] if isinstance(idea.get("used_shared_memory_refs"), list) else [],
        "ablation_switch": contract.get("ablation_switch") or ablation_plan.get("switch"),
        "expected_files": (contract.get("expected_files") or idea.get("expected_files") or [])[:8] if isinstance(contract.get("expected_files") or idea.get("expected_files") or [], list) else [],
        "planner_source": planner.get("source"),
        "planning_mode": planner.get("planning_mode"),
        "novelty_status": (idea.get("novelty_gate") or {}).get("status") if isinstance(idea.get("novelty_gate"), dict) else None,
        "scope_status": (idea.get("implementation_scope_gate") or {}).get("status") if isinstance(idea.get("implementation_scope_gate"), dict) else None,
    }


def _payload_variant_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    next_variant = payload.get("next_variant") or payload.get("variant_candidate")
    if isinstance(next_variant, dict) and next_variant:
        return [next_variant]
    variants = payload.get("variant_candidates")
    if isinstance(variants, list) and variants:
        return [item for item in variants if isinstance(item, dict)]
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def _build_s2_variant_selection(
    variants: list[dict[str, Any]],
    *,
    selected: dict[str, Any],
    planner_memory: dict[str, Any],
    feedback: list[dict[str, Any]],
    max_selected: int,
    source: str,
    used_shared_memory_refs: list[str] | None = None,
) -> dict[str, Any]:
    variant_candidates = []
    history = _s2_variant_history(planner_memory)
    feedback_targets = _s2_feedback_targets(feedback)
    batch_counts = _s2_variant_batch_counts(variants)
    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            continue
        normalized = dict(variant)
        normalized.setdefault("s1_direction_id", selected.get("id") or selected.get("direction_id") or selected.get("s1_direction_id"))
        normalized.setdefault("direction_id", selected.get("direction_id") or selected.get("id") or selected.get("s1_direction_id"))
        variant_refs = collect_used_shared_memory_refs(normalized, {"recent_entries": [{"memory_id": item} for item in used_shared_memory_refs or []]})
        normalized["used_shared_memory_refs"] = variant_refs or list(used_shared_memory_refs or [])
        normalized.setdefault("mechanism_axis", _infer_variant_axis(normalized))
        normalized.setdefault("integration_point", _infer_integration_point(normalized))
        normalized.setdefault("control_signal", _infer_control_signal(normalized))
        normalized.setdefault("expected_dataset_tradeoff", _default_expected_tradeoff(feedback_targets))
        normalized.setdefault("risk_budget", _default_variant_risk_budget(normalized))
        normalized.setdefault("anti_repeat", _default_anti_repeat(normalized, history))
        normalized["variant_fingerprint"] = _s2_variant_fingerprint(normalized)
        normalized["variant_rank_input_index"] = idx
        normalized["s2_planner_source"] = source
        score = _score_s2_variant(normalized, history=history, feedback_targets=feedback_targets, batch_counts=batch_counts)
        normalized["variant_score"] = score
        normalized["s2_variant"] = _variant_metadata(normalized)
        variant_candidates.append(normalized)
    selected_variants = _select_s2_variants(variant_candidates, max_selected=1)
    selected_ideas = []
    selected_fingerprints = {item.get("variant_fingerprint") for item in selected_variants}
    for idx, variant in enumerate(variant_candidates):
        is_selected = variant.get("variant_fingerprint") in selected_fingerprints
        idea = _variant_to_candidate_idea(variant, selected=selected, is_selected=is_selected and len(selected_ideas) == 0)
        if is_selected:
            selected_ideas.append(idea)
        variant["selected_for_s2_5"] = is_selected
    next_variant = _compact_variant_for_artifact(selected_variants[0]) if selected_variants else {}
    return {
        "schema_version": "c2c_s2_next_variant_selection_v1",
        "source": source,
        "s1_direction_id": selected.get("id"),
        "s1_mechanism_type": selected.get("mechanism_type"),
        "used_shared_memory_refs": list(used_shared_memory_refs or []),
        "max_selected": 1,
        "candidate_pool_count": len(variant_candidates),
        "candidate_pool": [_compact_variant_for_artifact(item) for item in variant_candidates],
        "considered_variant_ids": [str(item.get("id")) for item in variant_candidates if item.get("id")],
        "next_variant": next_variant,
        "selected_variant": next_variant,
        "selected_variants": [_compact_variant_for_artifact(item) for item in selected_variants],
        "selected_ideas": selected_ideas,
        "feedback_targets": feedback_targets,
        "history_summary": {
            "fingerprints": sorted(history["fingerprints"])[:30],
            "mechanism_axes": sorted(history["mechanism_axes"])[:20],
            "integration_points": sorted(history["integration_points"])[:20],
            "control_signals": sorted(history["control_signals"])[:20],
            "risk_files": sorted(history["risk_files"])[:20],
            "failed_integration_points": dict(sorted(history["failed_integration_points"].items())),
            "failed_file_groups": dict(sorted(("|".join(key), value) for key, value in history["failed_file_groups"].items())),
        },
    }


def _variant_to_candidate_idea(variant: dict[str, Any], *, selected: dict[str, Any], is_selected: bool) -> dict[str, Any]:
    idea = dict(variant)
    idea["selected"] = is_selected
    idea["used_shared_memory_refs"] = list(variant.get("used_shared_memory_refs") or [])
    idea.setdefault("s1_direction_id", selected.get("id") or selected.get("direction_id"))
    idea["variant_fingerprint"] = variant.get("variant_fingerprint")
    idea["s2_variant"] = _variant_metadata(variant)
    planner = idea.get("s2_planner") if isinstance(idea.get("s2_planner"), dict) else {}
    planner.update(
        {
            "source": variant.get("s2_planner_source") or variant.get("source") or "s2_variant_selection",
            "variant_fingerprint": variant.get("variant_fingerprint"),
            "mechanism_axis": variant.get("mechanism_axis"),
            "integration_point": variant.get("integration_point"),
            "control_signal": variant.get("control_signal"),
            "variant_score": variant.get("variant_score"),
            "used_shared_memory_refs": list(variant.get("used_shared_memory_refs") or []),
        }
    )
    idea["s2_planner"] = planner
    return idea


def _variant_metadata(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_fingerprint": variant.get("variant_fingerprint"),
        "mechanism_axis": variant.get("mechanism_axis"),
        "integration_point": variant.get("integration_point"),
        "control_signal": variant.get("control_signal"),
        "expected_dataset_tradeoff": variant.get("expected_dataset_tradeoff") or {},
        "risk_budget": variant.get("risk_budget") or {},
        "anti_repeat": variant.get("anti_repeat"),
        "used_shared_memory_refs": list(variant.get("used_shared_memory_refs") or []),
        "variant_score": variant.get("variant_score") or {},
    }


def _compact_variant_for_artifact(variant: dict[str, Any]) -> dict[str, Any]:
    experiment_contract = variant.get("experiment_contract") if isinstance(variant.get("experiment_contract"), dict) else {}
    return {
        "id": variant.get("id"),
        "title": variant.get("title"),
        "direction_id": variant.get("direction_id") or variant.get("s1_direction_id"),
        "s1_direction_id": variant.get("s1_direction_id") or variant.get("direction_id"),
        "selected_for_s2_5": bool(variant.get("selected_for_s2_5")),
        "variant_fingerprint": variant.get("variant_fingerprint"),
        "mechanism_type": variant.get("mechanism_type"),
        "mechanism_axis": variant.get("mechanism_axis"),
        "integration_point": variant.get("integration_point"),
        "control_signal": variant.get("control_signal"),
        "expected_dataset_tradeoff": variant.get("expected_dataset_tradeoff") or {},
        "risk_budget": variant.get("risk_budget") or {},
        "anti_repeat": _short_text(str(variant.get("anti_repeat") or ""), 500),
        "used_shared_memory_refs": list(variant.get("used_shared_memory_refs") or []),
        "expected_files": experiment_contract.get("expected_files") or variant.get("expected_files") or [],
        "ablation_switch": experiment_contract.get("ablation_switch") or ((variant.get("ablation_plan") or {}).get("switch") if isinstance(variant.get("ablation_plan"), dict) else None) or variant.get("ablation_switch"),
        "experiment_contract": experiment_contract,
        "implementation_plan": variant.get("implementation_plan") if isinstance(variant.get("implementation_plan"), dict) else {},
        "failure_feedback_refs": list(variant.get("failure_feedback_refs") or [])[:10] if isinstance(variant.get("failure_feedback_refs"), list) else [],
        "variant_score": variant.get("variant_score") or {},
        "failure_avoidance": list(variant.get("failure_avoidance") or [])[:5] if isinstance(variant.get("failure_avoidance"), list) else [],
    }


def _select_s2_variants(variants: list[dict[str, Any]], *, max_selected: int) -> list[dict[str, Any]]:
    budget = max(1, min(max_selected or 1, len(variants)))
    ranked = sorted(
        variants,
        key=lambda item: (
            float((item.get("variant_score") or {}).get("score") or 0.0),
            float((item.get("variant_score") or {}).get("diversity_score") or 0.0),
            -int(item.get("variant_rank_input_index") or 0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_axes: set[str] = set()
    selected_points: set[str] = set()
    for item in ranked:
        axis = str(item.get("mechanism_axis") or "")
        point = str(item.get("integration_point") or "")
        if len(selected) < budget and selected and axis in selected_axes and point in selected_points:
            continue
        selected.append(item)
        selected_axes.add(axis)
        selected_points.add(point)
        if len(selected) >= budget:
            break
    if not selected and ranked:
        selected.append(ranked[0])
    return selected


def _score_s2_variant(
    variant: dict[str, Any],
    *,
    history: dict[str, Any],
    feedback_targets: dict[str, Any],
    batch_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    axis = str(variant.get("mechanism_axis") or "")
    point = str(variant.get("integration_point") or "")
    signal = str(variant.get("control_signal") or "")
    fingerprint = str(variant.get("variant_fingerprint") or "")
    expected_files = _variant_expected_files(variant)
    risk_budget = variant.get("risk_budget") if isinstance(variant.get("risk_budget"), dict) else {}
    score = 0.0
    reasons: list[str] = []
    diversity = 0.0
    if fingerprint not in history["fingerprints"]:
        score += 2.0
        diversity += 2.0
        reasons.append("new_fingerprint")
    else:
        score -= 4.0
        reasons.append("repeated_fingerprint")
    if axis and axis not in history["mechanism_axes"]:
        score += 1.2
        diversity += 1.2
        reasons.append("new_mechanism_axis")
    if point and point not in history["integration_points"]:
        score += 1.0
        diversity += 1.0
        reasons.append("new_integration_point")
    if signal and signal not in history["control_signals"]:
        score += 0.8
        diversity += 0.8
        reasons.append("new_control_signal")
    batch_counts = batch_counts or {}
    duplicate_tuple_penalty = 0.0
    for key, value in [("mechanism_axes", axis), ("integration_points", point), ("control_signals", signal)]:
        if value and (batch_counts.get(key) or {}).get(value, 0) > 1:
            duplicate_tuple_penalty += 0.4
            reasons.append(f"same_batch_duplicate_{key[:-1]}")
    if duplicate_tuple_penalty:
        score -= duplicate_tuple_penalty
    recent_risk_files = set(history["risk_files"]) | set(str(path) for path in feedback_targets.get("risk_files") or [] if path)
    if any(path in recent_risk_files for path in expected_files):
        score -= 1.2
        reasons.append("reuses_recent_risk_file")
    failed_point_count = int((history.get("failed_integration_points") or {}).get(point, 0) or 0)
    if failed_point_count >= 2:
        penalty = min(3.0, 0.8 * failed_point_count)
        score -= penalty
        reasons.append("reuses_repeatedly_failed_integration_point")
    file_group = tuple(sorted(expected_files))
    failed_file_group_count = int((history.get("failed_file_groups") or {}).get(file_group, 0) or 0)
    if file_group and failed_file_group_count:
        score -= min(2.0, 0.6 * failed_file_group_count)
        reasons.append("reuses_failed_file_group")
    if _variant_targets_dragging_dataset(variant, feedback_targets):
        score += 1.4
        reasons.append("targets_dragging_dataset")
    if _variant_preserves_positive_signal(variant, feedback_targets):
        score += 0.8
        reasons.append("preserves_positive_dataset_signal")
    if str(variant.get("mechanism_type") or "") in set(feedback_targets.get("proxy_risky_mechanisms") or []):
        score -= 0.8
        reasons.append("proxy_calibration_risky_mechanism")
    if point and point in set(feedback_targets.get("proxy_risky_integration_points") or []):
        score -= 0.8
        reasons.append("proxy_calibration_risky_integration_point")
    if _variant_overtrusts_proxy_risky_dataset(variant, feedback_targets):
        score -= 0.6
        reasons.append("proxy_calibration_risky_dataset_unaddressed")
    max_files = _safe_int(risk_budget.get("max_changed_files"), default=0)
    if max_files and max_files <= 2:
        score += 0.5
        reasons.append("bounded_risk_budget")
    if _variant_touches_forbidden_eval(variant):
        score -= 6.0
        reasons.append("forbidden_evaluator_risk")
    if _looks_like_hard_gate_variant(variant):
        score -= 1.5
        reasons.append("hard_gate_risk")
    return {
        "score": round(score, 4),
        "diversity_score": round(diversity, 4),
        "risk_score": round(max(0.0, 3.0 - score), 4),
        "reasons": reasons,
    }


def _s2_variant_history(planner_memory: dict[str, Any]) -> dict[str, Any]:
    history = {
        "fingerprints": set(),
        "mechanism_axes": set(),
        "integration_points": set(),
        "control_signals": set(),
        "risk_files": set(),
        "failed_integration_points": {},
        "failed_file_groups": {},
    }
    for entry in planner_memory.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        for idea in [entry.get("selected_candidate"), *(entry.get("candidate_summaries") or [])]:
            if not isinstance(idea, dict):
                continue
            for source_key, target_key in [
                ("variant_fingerprint", "fingerprints"),
                ("mechanism_axis", "mechanism_axes"),
                ("integration_point", "integration_points"),
                ("control_signal", "control_signals"),
            ]:
                value = idea.get(source_key)
                if value:
                    history[target_key].add(str(value))
        feedback = entry.get("feedback_digest") if isinstance(entry.get("feedback_digest"), dict) else {}
        for path in feedback.get("patch_risk_files") or []:
            if path:
                history["risk_files"].add(str(path))
        patch = entry.get("patch_manifest") if isinstance(entry.get("patch_manifest"), dict) else {}
        for path in patch.get("risk_files") or []:
            if path:
                history["risk_files"].add(str(path))
        failure_digest = entry.get("feedback_digest") if isinstance(entry.get("feedback_digest"), dict) else {}
        failed = _s2_entry_failed(entry) or bool(failure_digest.get("latest_failure_mode") or failure_digest.get("latest_decision") in {"proxy_rejected", "proxy_repairable", "not_viable", "failed_no_metrics"})
        if failed:
            selected = entry.get("selected_candidate") if isinstance(entry.get("selected_candidate"), dict) else {}
            point = selected.get("integration_point")
            if point:
                history["failed_integration_points"][str(point)] = int(history["failed_integration_points"].get(str(point), 0)) + 1
            files = tuple(sorted(_variant_expected_files(selected)))
            if files:
                history["failed_file_groups"][files] = int(history["failed_file_groups"].get(files, 0)) + 1
    return history


def _s2_entry_failed(entry: dict[str, Any]) -> bool:
    feedback = entry.get("feedback_digest") if isinstance(entry.get("feedback_digest"), dict) else {}
    decision = feedback.get("latest_decision")
    failure_mode = feedback.get("latest_failure_mode")
    return bool(decision in {"proxy_rejected", "proxy_repairable", "not_viable", "failed_no_metrics", "partial", "blocked"} or failure_mode)


def _s2_variant_batch_counts(variants: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {"mechanism_axes": {}, "integration_points": {}, "control_signals": {}}
    for raw in variants:
        if not isinstance(raw, dict):
            continue
        axis = str(raw.get("mechanism_axis") or _infer_variant_axis(raw) or "")
        point = str(raw.get("integration_point") or _infer_integration_point(raw) or "")
        signal = str(raw.get("control_signal") or _infer_control_signal(raw) or "")
        for key, value in [("mechanism_axes", axis), ("integration_points", point), ("control_signals", signal)]:
            if value:
                counts[key][value] = int(counts[key].get(value, 0)) + 1
    return counts


def _s2_feedback_targets(feedback: list[dict[str, Any]]) -> dict[str, Any]:
    dragging: set[str] = set()
    improved: set[str] = set()
    risk_files: set[str] = set()
    signals: set[str] = set()
    proxy_risky_datasets: set[str] = set()
    proxy_risky_mechanisms: set[str] = set()
    proxy_risky_integration_points: set[str] = set()

    def visit(item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        for dataset in item.get("dragging_datasets") or []:
            if isinstance(dataset, dict) and dataset.get("dataset"):
                dragging.add(str(dataset["dataset"]))
            elif isinstance(dataset, str):
                dragging.add(dataset)
        proxy = item.get("proxy_screen") if isinstance(item.get("proxy_screen"), dict) else {}
        for dataset, delta in (proxy.get("proxy_dataset_deltas") or {}).items():
            try:
                if float(delta) < 0:
                    dragging.add(str(dataset))
                elif float(delta) > 0:
                    improved.add(str(dataset))
            except (TypeError, ValueError):
                continue
        attribution = item.get("failure_attribution") if isinstance(item.get("failure_attribution"), dict) else {}
        patch_risk = attribution.get("patch_risk") if isinstance(attribution.get("patch_risk"), dict) else {}
        for risk_file in patch_risk.get("risk_files") or []:
            if isinstance(risk_file, dict) and risk_file.get("path"):
                risk_files.add(str(risk_file["path"]))
            elif isinstance(risk_file, str):
                risk_files.add(risk_file)
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        for signal in summary.get("repair_vs_variant_signals") or []:
            signals.add(str(signal))
        calibration = item.get("proxy_calibration") if isinstance(item.get("proxy_calibration"), dict) else {}
        calibration_summary = calibration.get("summary") if isinstance(calibration.get("summary"), dict) else {}
        method_feedback = calibration_summary.get("method_feedback") if isinstance(calibration_summary.get("method_feedback"), dict) else {}
        for dataset in method_feedback.get("risky_datasets") or []:
            if isinstance(dataset, dict) and dataset.get("dataset"):
                proxy_risky_datasets.add(str(dataset["dataset"]))
        for mechanism in method_feedback.get("risky_mechanisms") or []:
            if isinstance(mechanism, dict) and mechanism.get("mechanism_type"):
                proxy_risky_mechanisms.add(str(mechanism["mechanism_type"]))
        for point in method_feedback.get("risky_integration_points") or []:
            if isinstance(point, dict) and point.get("integration_point"):
                proxy_risky_integration_points.add(str(point["integration_point"]))
        for candidate_result in item.get("candidate_results") or []:
            if isinstance(candidate_result, dict):
                visit(candidate_result)

    for item in feedback:
        if isinstance(item, dict):
            visit(item)
    return {
        "dragging_datasets": sorted(dragging),
        "improved_datasets": sorted(improved),
        "risk_files": sorted(risk_files),
        "signals": sorted(signals),
        "proxy_risky_datasets": sorted(proxy_risky_datasets),
        "proxy_risky_mechanisms": sorted(proxy_risky_mechanisms),
        "proxy_risky_integration_points": sorted(proxy_risky_integration_points),
    }


def _s2_variant_fingerprint(variant: dict[str, Any]) -> str:
    payload = {
        "mechanism_type": variant.get("mechanism_type"),
        "mechanism_axis": variant.get("mechanism_axis"),
        "integration_point": variant.get("integration_point"),
        "control_signal": variant.get("control_signal"),
        "expected_files": sorted(_variant_expected_files(variant)),
        "config_keys": sorted(_flatten_variant_config_keys((variant.get("experiment_contract") or {}).get("config_overrides") or {})),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:16]


def _select_candidate_for_scorecard(
    *,
    ideas: list[dict[str, Any]],
    pool_candidates: list[dict[str, Any]],
    selected_variant_id: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    target = str(selected_variant_id or "").strip()
    for collection in [ideas, pool_candidates]:
        for item in collection:
            if isinstance(item, dict) and str(item.get("id") or item.get("variant_id") or "").strip() == target:
                return item
    return fallback


def _mark_selected_variant(ideas: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    selected_id = str(selected.get("id") or selected.get("variant_id") or "")
    for idea in ideas:
        if isinstance(idea, dict):
            idea["selected"] = bool(selected_id and str(idea.get("id") or idea.get("variant_id") or "") == selected_id)
    selected["selected"] = True


def _variant_expected_files(variant: dict[str, Any]) -> list[str]:
    contract = variant.get("experiment_contract") if isinstance(variant.get("experiment_contract"), dict) else {}
    files = contract.get("expected_files") or variant.get("expected_files") or []
    if isinstance(files, str):
        return [files]
    if isinstance(files, list):
        return [str(item) for item in files if item]
    return []


def _sanitize_c2c_variant_expected_files(variant: dict[str, Any], direction: dict[str, Any], config: dict[str, Any]) -> None:
    if not isinstance(variant, dict):
        return
    files = _filter_c2c_allowed_expected_files(_variant_expected_files(variant), config)
    if not files:
        files = _filter_c2c_allowed_expected_files(_direction_surface_files(direction), config)
    if not files:
        return
    variant["expected_files"] = files
    contract = variant.get("experiment_contract")
    if not isinstance(contract, dict):
        contract = {}
        variant["experiment_contract"] = contract
    contract["expected_files"] = files


def _direction_surface_files(direction: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for item in direction.get("expected_files") or []:
        path = _surface_file_path(item)
        if path:
            files.append(path)
    for item in direction.get("implementation_surface_refs") or []:
        path = _surface_file_path(item)
        if path:
            files.append(path)
    seen = set()
    result = []
    for path in files:
        normalized = path.strip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _surface_file_path(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["source_path", "path", "file", "source_label", "chunk_id"]:
            if value.get(key):
                return _normalize_surface_file(str(value.get(key)))
        return ""
    return _normalize_surface_file(str(value)) if value else ""


def _normalize_surface_file(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    for separator in ["::", "#"]:
        if separator in text:
            text = text.split(separator, 1)[0]
    for marker in ["/external/c2c_snapshot/", "/C2C/"]:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    else:
        for marker, prefix in [("/rosetta/", "rosetta"), ("/script/", "script"), ("/test/", "test"), ("/tests/", "tests")]:
            if marker in text:
                text = f"{prefix}/{text.split(marker, 1)[1]}"
                break
    return text.removeprefix("./").strip("/")


def _filter_c2c_allowed_expected_files(files: list[str], config: dict[str, Any]) -> list[str]:
    c2c_cfg = config.get("c2c") if isinstance(config.get("c2c"), dict) else {}
    allowed_files = {str(item).strip("/") for item in c2c_cfg.get("allowed_files") or [] if item}
    allowed_prefixes = [str(item).strip("/") for item in c2c_cfg.get("allowed_prefixes") or [] if item]
    if not allowed_files and not allowed_prefixes:
        return _dedupe_paths([str(item).strip("/") for item in files if item])
    kept = []
    for item in files:
        normalized = str(item).strip("/")
        if not normalized:
            continue
        if normalized in allowed_files or any(normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/") for prefix in allowed_prefixes):
            kept.append(normalized)
    return _dedupe_paths(kept)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _flatten_variant_config_keys(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            keys.extend(_flatten_variant_config_keys(item, prefix=next_prefix))
        return keys
    return [prefix] if prefix else []


def _infer_variant_axis(variant: dict[str, Any]) -> str:
    text = _variant_text(variant)
    if any(token in text for token in ["loss", "train", "objective"]):
        return "training_signal"
    if any(token in text for token in ["span", "alignment"]):
        return "span_selection"
    if any(token in text for token in ["normaliz", "scale", "residual"]):
        return "normalization"
    if any(token in text for token in ["route", "router", "cache"]):
        return "routing"
    if any(token in text for token in ["fallback", "abstain"]):
        return "fallback"
    return "scoring"


def _infer_integration_point(variant: dict[str, Any]) -> str:
    files = " ".join(_variant_expected_files(variant)).lower()
    text = f"{files} {_variant_text(variant)}"
    if "wrapper" in text:
        return "wrapper"
    if "projector" in text:
        return "projector"
    if "train" in text or "sft" in text:
        return "train_loss"
    if "recipe" in text:
        return "recipe"
    return "aligner"


def _infer_control_signal(variant: dict[str, Any]) -> str:
    text = _variant_text(variant)
    for signal in ["entropy", "confidence", "span_agreement", "utility", "pathology", "semantic_similarity"]:
        if signal.replace("_", " ") in text or signal in text:
            return signal
    if "agreement" in text:
        return "span_agreement"
    if "semantic" in text:
        return "semantic_similarity"
    return "utility"


def _default_expected_tradeoff(feedback_targets: dict[str, Any]) -> dict[str, str]:
    datasets = ["mmlu-redux", "ai2-arc", "openbookqa"]
    dragging = set(feedback_targets.get("dragging_datasets") or [])
    improved = set(feedback_targets.get("improved_datasets") or [])
    return {dataset: "up" if dataset in dragging else "flat" if dataset in improved else "unknown" for dataset in datasets}


def _default_variant_risk_budget(variant: dict[str, Any]) -> dict[str, Any]:
    files = _variant_expected_files(variant)
    return {
        "max_changed_files": min(max(len(files), 1), 2),
        "forbidden_files": ["script/evaluation/*", "experiment/results/*", "local/auto_research_runs/*"],
        "risk_notes": [],
    }


def _default_anti_repeat(variant: dict[str, Any], history: dict[str, set[str]]) -> str:
    axis = str(variant.get("mechanism_axis") or "")
    point = str(variant.get("integration_point") or "")
    signal = str(variant.get("control_signal") or "")
    new_bits = []
    if axis and axis not in history["mechanism_axes"]:
        new_bits.append(f"new mechanism_axis={axis}")
    if point and point not in history["integration_points"]:
        new_bits.append(f"new integration_point={point}")
    if signal and signal not in history["control_signals"]:
        new_bits.append(f"new control_signal={signal}")
    return "; ".join(new_bits) or "must change implementation details relative to previous same-direction candidates"


def _variant_targets_dragging_dataset(variant: dict[str, Any], feedback_targets: dict[str, Any]) -> bool:
    dragging = set(feedback_targets.get("dragging_datasets") or [])
    tradeoff = variant.get("expected_dataset_tradeoff") if isinstance(variant.get("expected_dataset_tradeoff"), dict) else {}
    return any(str(tradeoff.get(dataset, "")).lower() in {"up", "improve", "fix", "recover"} for dataset in dragging)


def _variant_preserves_positive_signal(variant: dict[str, Any], feedback_targets: dict[str, Any]) -> bool:
    improved = set(feedback_targets.get("improved_datasets") or [])
    tradeoff = variant.get("expected_dataset_tradeoff") if isinstance(variant.get("expected_dataset_tradeoff"), dict) else {}
    return any(str(tradeoff.get(dataset, "")).lower() in {"flat", "preserve", "up", "improve"} for dataset in improved)


def _variant_overtrusts_proxy_risky_dataset(variant: dict[str, Any], feedback_targets: dict[str, Any]) -> bool:
    risky = set(feedback_targets.get("proxy_risky_datasets") or [])
    if not risky:
        return False
    tradeoff = variant.get("expected_dataset_tradeoff") if isinstance(variant.get("expected_dataset_tradeoff"), dict) else {}
    text = _variant_text(variant)
    for dataset in risky:
        target = str(tradeoff.get(dataset, "")).lower()
        if target in {"up", "improve", "gain"} and dataset.lower() not in text:
            return True
    return False


def _variant_touches_forbidden_eval(variant: dict[str, Any]) -> bool:
    files = _variant_expected_files(variant)
    return any(path.startswith("script/evaluation/") or path.startswith("experiment/results/") for path in files)


def _looks_like_hard_gate_variant(variant: dict[str, Any]) -> bool:
    text = _variant_text(variant)
    return any(token in text for token in ["hard gate", "accept/reject", "threshold only", "top-k only", "confidence floor"])


def _variant_text(variant: dict[str, Any]) -> str:
    parts = [
        variant.get("id"),
        variant.get("title"),
        variant.get("description"),
        variant.get("hypothesis"),
        variant.get("mechanism_axis"),
        variant.get("integration_point"),
        variant.get("control_signal"),
        variant.get("anti_repeat"),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _inherit_if_present(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    if target.get(key):
        return
    value = source.get(key)
    if value:
        target[key] = value


def _ensure_c2c_s2_config_overrides(idea: dict[str, Any]) -> None:
    if not isinstance(idea, dict):
        return
    contract = idea.get("experiment_contract") if isinstance(idea.get("experiment_contract"), dict) else {}
    if not contract:
        contract = {}
        idea["experiment_contract"] = contract
    overrides = contract.get("config_overrides") if isinstance(contract.get("config_overrides"), dict) else {}
    candidate_id = sanitize_filename(str(idea.get("id") or idea.get("mechanism_type") or "c2c_candidate"))
    switch = str(contract.get("ablation_switch") or (idea.get("ablation_plan") or {}).get("switch") or f"ablation_disable_{candidate_id}")
    contract["ablation_switch"] = switch
    if overrides:
        return
    mode_key = {
        "utility_predicted_cache_routing": "cache_routing_mode",
        "counterfactual_training_objective": "counterfactual_cache_dropout",
        "semantic_span_graph_alignment": "soft_alignment_score_mode",
        "verifier_guided_cache_acceptance": "cache_acceptance_mode",
        "latent_bridge_memory": "bridge_memory_mode",
        "pathology_conditioned_controller": "cache_controller_mode",
    }.get(str(idea.get("mechanism_type") or ""), "auto_research_mechanism_mode")
    if mode_key == "counterfactual_cache_dropout":
        model_overrides: dict[str, Any] = {mode_key: True, switch: False}
    else:
        model_overrides = {mode_key: candidate_id, switch: False}
    contract["config_overrides"] = {
        "train": {"model": model_overrides},
        "eval": {"model": {"rosetta_config": dict(model_overrides)}},
    }


def _load_s2_5_repair_dispatch(project_root: Path) -> dict[str, Any]:
    dispatch_path = project_root / "plan" / "s2_5_repair_dispatch.json"
    if not dispatch_path.exists():
        return {}
    dispatch = read_json(dispatch_path, default={}) or {}
    if not isinstance(dispatch, dict):
        return {}
    if str(dispatch.get("mode") or dispatch.get("repair_lane") or "") != "s2_5_only_implementation_repair":
        return {}
    return dispatch


def _select_patch_only_repair_ideas(
    ideas: list[dict[str, Any]],
    *,
    target_candidate_id: str,
    target_fingerprint: str,
) -> list[dict[str, Any]]:
    if target_candidate_id:
        matched = [idea for idea in ideas if str(idea.get("id") or idea.get("candidate_id") or "").strip() == target_candidate_id]
        if matched:
            return matched
        return []
    if target_fingerprint:
        matched = [idea for idea in ideas if _candidate_variant_fingerprint(idea) == target_fingerprint]
        if matched:
            return matched
        return []
    selected = [idea for idea in ideas if idea.get("selected") is True]
    if selected:
        return selected[:1]
    return ideas[:1]


def _candidate_variant_fingerprint(idea: dict[str, Any]) -> str:
    variant = idea.get("s2_variant") if isinstance(idea.get("s2_variant"), dict) else {}
    return str(idea.get("variant_fingerprint") or variant.get("variant_fingerprint") or "").strip()


def _compact_s2_5_repair_dispatch_for_plan(dispatch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dispatch, dict) or not dispatch:
        return {}
    return {
        "mode": dispatch.get("mode") or dispatch.get("repair_lane"),
        "selected_candidate_id": dispatch.get("selected_candidate_id"),
        "variant_fingerprint": dispatch.get("variant_fingerprint"),
        "same_candidate_required": dispatch.get("same_candidate_required"),
        "same_variant_fingerprint_required": dispatch.get("same_variant_fingerprint_required"),
        "reuse_persistent_codex_session": dispatch.get("reuse_persistent_codex_session"),
        "implementation_failure_signals": dispatch.get("implementation_failure_signals") or [],
        "changed_files": dispatch.get("changed_files") or [],
        "tensor_checks": dispatch.get("tensor_checks") or {},
        "activation_forward_probe_diagnostics": dispatch.get("activation_forward_probe_diagnostics") or {},
        "patch_manifest": dispatch.get("patch_manifest") or {},
        "performance_feedback_path": dispatch.get("performance_feedback_path"),
        "patch_manifest_path": dispatch.get("patch_manifest_path"),
    }


def _with_patch_only_previous_failure(
    idea: dict[str, Any],
    feedback: dict[str, Any],
    *,
    repair_dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    patched = dict(idea)
    previous_failure = dict(patched.get("previous_patch_failure") or {})
    summary = feedback.get("summary") if isinstance(feedback.get("summary"), dict) else {}
    repair_dispatch = repair_dispatch if isinstance(repair_dispatch, dict) else {}
    candidate_results = [item for item in feedback.get("candidate_results") or [] if isinstance(item, dict)]
    candidate_id = str(patched.get("id") or "")
    matched = next((item for item in candidate_results if str(item.get("id") or "") == candidate_id), None)
    if matched is None and candidate_results:
        matched = candidate_results[0]
    proxy_screen = {}
    if isinstance(matched, dict) and isinstance(matched.get("proxy_screen"), dict):
        proxy_screen = dict(matched["proxy_screen"])
    if not proxy_screen:
        proxy_screen = {"reason": feedback.get("reason") or summary.get("repair_vs_variant_reason") or "implementation_failure"}
    default_contract = {
        "mode": "s2_5_only_implementation_repair",
        "source": "implementation_failure",
        "reason": feedback.get("reason") or proxy_screen.get("reason") or "implementation_failure",
        "force_new_codex_session": False,
        "reuse_persistent_codex_session": True,
        "same_candidate_required": True,
        "same_variant_fingerprint_required": bool(_candidate_variant_fingerprint(patched)),
        "selected_candidate_id": patched.get("id"),
        "variant_fingerprint": _candidate_variant_fingerprint(patched),
        "repair_until": "patch_eligible_for_s3_or_implementation_blocked",
        "repair_dispatch_path": "plan/s2_5_repair_dispatch.json" if repair_dispatch else "",
        "implementation_failure_signals": repair_dispatch.get("implementation_failure_signals") or [],
        "activation_forward_probe_diagnostics": repair_dispatch.get("activation_forward_probe_diagnostics") or {},
        "tensor_checks": repair_dispatch.get("tensor_checks") or {},
        "changed_files": repair_dispatch.get("changed_files") or [],
        "repair_priorities": [
            "Repair implementation only; do not change the S1/S2 method, candidate, or selected variant.",
            "Use s2_5_repair_dispatch.activation_forward_probe_diagnostics, tensor_checks, patch_manifest, and changed_files as primary evidence.",
            "Reuse the same Codex persistent session/worktree for this candidate when available.",
            "Repair config -> rosetta_config -> constructor/wrapper/projector/aligner forward -> tensor/output activation before S3.",
        ],
    }
    existing_contract = proxy_screen.get("proxy_effect_repair_contract")
    if isinstance(existing_contract, dict):
        contract = {**existing_contract, **{key: value for key, value in default_contract.items() if key not in existing_contract}}
        contract["mode"] = "s2_5_only_implementation_repair"
        contract["force_new_codex_session"] = False
        contract["reuse_persistent_codex_session"] = True
        contract["same_candidate_required"] = True
        contract["same_variant_fingerprint_required"] = default_contract["same_variant_fingerprint_required"]
        contract["repair_priorities"] = default_contract["repair_priorities"]
        for key in [
            "implementation_failure_signals",
            "activation_forward_probe_diagnostics",
            "tensor_checks",
            "changed_files",
        ]:
            if not contract.get(key):
                contract[key] = default_contract[key]
    else:
        contract = default_contract
    previous_failure.update(
        {
            "status": "implementation_failure",
            "reason": feedback.get("reason") or summary.get("repair_vs_variant_reason") or "implementation failure before method-level proxy evaluation",
            "failure_class": "implementation_failure",
            "does_not_consume_same_direction_attempt": True,
            "performance_feedback_summary": summary,
            "s2_5_repair_dispatch": _compact_s2_5_repair_dispatch_for_plan(repair_dispatch),
            "proxy_screen": proxy_screen,
            "proxy_effect_repair_contract": contract,
            "candidate_result": matched or {},
        }
    )
    patched["previous_patch_failure"] = previous_failure
    return patched


def _c2c_s2_feedback_digest(feedback: list[dict[str, Any]]) -> dict[str, Any]:
    summary = next((item for item in feedback if isinstance(item, dict) and item.get("kind") == "c2c_feedback_summary"), {})
    performance = next((item for item in reversed(feedback) if isinstance(item, dict) and item.get("kind") == "c2c_performance_feedback"), {})
    perf_summary = performance.get("summary") if isinstance(performance.get("summary"), dict) else {}
    patch_risk_files: list[str] = []

    def visit(item: dict[str, Any]) -> None:
        if not isinstance(item, dict):
            return
        attribution = item.get("failure_attribution") if isinstance(item.get("failure_attribution"), dict) else {}
        patch_risk = attribution.get("patch_risk") if isinstance(attribution.get("patch_risk"), dict) else {}
        for risk_file in patch_risk.get("risk_files") or []:
            if isinstance(risk_file, dict) and risk_file.get("path"):
                patch_risk_files.append(str(risk_file["path"]))
            elif isinstance(risk_file, str):
                patch_risk_files.append(risk_file)
        for candidate_result in item.get("candidate_results") or []:
            if isinstance(candidate_result, dict):
                visit(candidate_result)

    for item in feedback:
        if isinstance(item, dict):
            visit(item)
    return {
        "latest_reason": summary.get("latest_reason"),
        "latest_failure_mode": summary.get("latest_failure_mode"),
        "latest_decision": summary.get("latest_decision"),
        "failed_idea_ids": list(summary.get("failed_idea_ids") or [])[:6],
        "dragging_datasets": list(summary.get("dragging_datasets") or [])[:5],
        "dataset_regressions": summary.get("dataset_regressions") if isinstance(summary.get("dataset_regressions"), dict) else {},
        "proxy_delta": (summary.get("latest_acceptance") or {}).get("proxy_delta") if isinstance(summary.get("latest_acceptance"), dict) else None,
        "performance_next_action": perf_summary.get("next_action"),
        "recommended_s2_action": perf_summary.get("recommended_s2_action"),
        "s2_action_policy": perf_summary.get("s2_action_policy") if isinstance(perf_summary.get("s2_action_policy"), dict) else {},
        "same_direction_failure_count": perf_summary.get("same_direction_failure_count"),
        "same_direction_failure_budget": perf_summary.get("same_direction_failure_budget"),
        "patch_risk_files": sorted(set(patch_risk_files))[:10],
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
        "research_ledger": "meta/research_events.sqlite3",
        "research_state_projection": "meta/research_state.json",
        "latest_trial_result_projection": "experiment/results/trial_result.json",
        "iteration_trace": "meta/iteration_trace.jsonl",
        "candidate_ideas": "plan/s2_planner/candidate_pool.json",
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
    shared_method_memory = shared_method_memory_for_prompt(
        adapter.config,
        limit=12,
        query_context=shared_method_memory_query_context(
            adapter.config,
            selected_direction=selected,
            feedback=feedback,
            negative_memory=negative_memory,
        ),
    )
    payload = {
        "role": "S2 direction-conditioned experiment planner",
        "mode": "resume_session_planning",
        "instruction": (
            "Continue the same S2 planning thread for this S1 mechanism direction. "
            "Use prior attempts and latest method-level feedback to propose exactly one next_variant. "
            "You may inspect local code and artifacts if needed. Return only JSON."
        ),
        "s1_selected_direction": selected,
        "s1_candidate_pool_compact": _compact_for_plan_prompt(s1_ideas, 6000),
        "baseline": baseline,
        "allowed_files": adapter.allowed_files,
        "allowed_prefixes": adapter.allowed_prefixes,
        "reviewer_concerns": (concern_matrix.get("top_concerns") or [])[:8] if isinstance(concern_matrix, dict) else [],
        "negative_memory": _compact_for_plan_prompt(negative_memory, 4000),
        "shared_method_failure_memory": _compact_for_plan_prompt(shared_method_memory, 7000),
        "s2_planner_memory": _c2c_s2_memory_for_prompt(planner_memory),
        "failure_feedback_compact": _compact_for_plan_prompt(feedback, 9000),
        "available_artifacts": _c2c_s2_available_artifacts(),
        "max_next_variants": 1,
        "requirements": [
            "Use s2_planner_memory first; inspect full artifact paths only when memory is insufficient.",
            "Use shared_method_failure_memory as cross-project method-level negative evidence; ignore implementation/runtime errors from other projects.",
            "shared_method_failure_memory.memory_catalog/recent_entries is a lightweight retrieved error catalog, ranked by memory_retrieval.combined_score and memory_quality.priority; follow retrieval_policy/ranking_policy and prioritize high_quality_memory_ids, especially proxy_full_false_positive, full_train_failure, proxy_dataset_misprediction, cross_project_mechanism_failure, and ablation_evidence when choosing anti_repeat rules. If a catalog item seems decision-relevant, inspect the full memory via full_memory_access/read_hint before relying on detailed evidence.",
            "If shared_method_failure_memory affects the next_variant, anti_repeat rule, or forbidden pattern, copy the exact memory_id values into used_shared_memory_refs at top-level and inside next_variant. Use [] if none affected the decision.",
            "Stay in the S1 mechanism direction unless performance feedback says return_to_s1_new_direction.",
            "Propose one next_variant only. Do not output a batch; if legacy variant_candidates/candidates are returned, only one will be executed.",
            "If performance_feedback.summary.s2_action_policy is present, follow its action and matched_rule when choosing mechanism_repair or new_same_direction_variant.",
            "Implementation_failure should not reach S2 planning; if it appears in feedback, treat it only as implementation noise and keep the same method hypothesis.",
            "The next_variant must materially differ from recent memory by changing at least one of mechanism_axis, integration_point, or control_signal.",
            "Explain why_next: which proxy/full/dataset signal from the previous attempt motivates this branch.",
            "Do not change evaluator, datasets, baseline, or cheap proxy protocol.",
            "Avoid variants already tried in s2_planner_memory unless the new plan explicitly fixes the recorded failure.",
            "The next_variant needs id,title,why_next,mechanism_axis,integration_point,control_signal,expected_dataset_tradeoff,risk_budget,anti_repeat,description,hypothesis,mechanism_type,experiment_contract,implementation_plan,failure_avoidance,expected_signature.",
            "Return JSON object with planner_summary, planning_mode, used_shared_memory_refs, next_variant. You may include legacy candidates only for backward compatibility.",
        ],
        "return_shape": {
            "planner_summary": "short explanation",
            "planning_mode": "same_direction_variant|new_direction_after_budget|fallback",
            "used_shared_memory_refs": ["memory_id values from shared_method_failure_memory that influenced this S2 decision, or []"],
            "next_variant": {
                "id": "snake_case_id",
                "title": "short title",
                "why_next": "why this is the next branch after the latest S3/proxy result",
                "mechanism_axis": "scoring|routing|span_selection|normalization|training_signal|fallback",
                "integration_point": "aligner|projector|wrapper|train_loss|recipe",
                "control_signal": "confidence|entropy|span_agreement|utility|pathology|semantic_similarity",
                "used_shared_memory_refs": ["memory_id values that influenced this variant, or []"],
            },
        },
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
    reasoning_effort = str(llm_cfg.get("reasoning_effort") or "").strip()
    if reasoning_effort and reasoning_effort != "none":
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
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
            env=codex_subprocess_env(config),
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



def _execution_profile(config: dict[str, Any]) -> str:
    return str(((config.get("orchestration") or {}).get("profile") or "standard")).lower()


def _c2c_evaluator_provenance(repo_root: Path, c2c_config: dict[str, Any], *, simulate: bool) -> dict[str, Any]:
    if simulate:
        return {
            "evaluator_id": "c2c-small-loop-synthetic-evaluator",
            "evaluator_source_payloads": [{"adapter": "c2c_small_loop", "mode": "synthetic"}],
            "dependency_payloads": [{"dependencies": [], "mode": "synthetic"}],
        }
    configured = c2c_config.get("evaluator_provenance") if isinstance(c2c_config.get("evaluator_provenance"), dict) else {}
    evaluator_paths = configured.get("source_paths") or ["script/evaluation", "rosetta/model"]
    source_entries: list[dict[str, str]] = []
    for relative in evaluator_paths:
        candidate = (repo_root / str(relative)).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError("C2C evaluator provenance source path escapes target repository") from exc
        paths = [candidate] if candidate.is_file() else sorted(path for path in candidate.rglob("*") if path.is_file()) if candidate.is_dir() else []
        for path in paths:
            source_entries.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not source_entries:
        raise ValueError("real C2C TrialSpec requires readable evaluator source files")
    dependency_manifest = configured.get("dependencies")
    if not isinstance(dependency_manifest, list) or not dependency_manifest:
        raise ValueError("real C2C TrialSpec requires explicit evaluator dependency provenance")
    return {
        "evaluator_id": str(configured.get("evaluator_id") or "c2c-unified-evaluator"),
        "evaluator_source_files": source_entries,
        "dependency_payloads": deepcopy(dependency_manifest),
    }


def _store_trial_contracts(
    *,
    project_root: Path,
    raw_datasets: list[Any],
    datasets: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    execution: dict[str, Any],
    variant: dict[str, Any],
    provenance_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ensure_dir(project_root)
    store = ContractStore(project_root)
    sample_datasets: list[dict[str, Any]] = []
    for source_value, dataset in zip(raw_datasets, datasets):
        source = source_value if isinstance(source_value, dict) else {}
        declared_sample_ids = source.get("ordered_sample_ids")
        sample_payloads: list[bytes] = []
        if provenance_mode == "real":
            configured_path = source.get("sample_path") or source.get("source_path")
            sample_path = Path(configured_path) if configured_path else project_root / "samples" / f"{dataset['dataset_id']}.jsonl"
            if not sample_path.is_absolute():
                sample_path = project_root / sample_path
            try:
                raw_lines = [line for line in sample_path.read_bytes().splitlines(keepends=True) if line.strip()]
            except OSError as exc:
                raise ValueError(f"real TrialSpec dataset {dataset['dataset_id']} requires readable selected sample bytes") from exc
            for ordinal, line in enumerate(raw_lines):
                try:
                    payload = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"sample {dataset['dataset_id']}:{ordinal} is not canonicalizable JSON") from exc
                del payload
                sample_payloads.append(line)
        else:
            synthetic_ids = list(declared_sample_ids or [])
            if not synthetic_ids:
                synthetic_ids = [canonical_hash({"dataset_id": dataset["dataset_id"], "split": dataset["split"], "ordinal": ordinal, "mode": "synthetic"}) for ordinal in range(dataset["sample_count"])]
            sample_payloads = [canonical_contract_bytes({"synthetic_sample_id": sample_id}) for sample_id in synthetic_ids]
        if len(sample_payloads) != dataset["sample_count"]:
            raise ValueError(f"TrialSpec dataset {dataset['dataset_id']} selected sample bytes must match sample_count")
        sample_refs = [store.put_bytes(raw) for raw in sample_payloads]
        sample_ids = [reference["digest"] for reference in sample_refs]
        if declared_sample_ids is not None and list(declared_sample_ids) != sample_ids:
            raise ValueError(f"TrialSpec dataset {dataset['dataset_id']} ordered_sample_ids do not match selected sample bytes")
        source_revision = str(
            source.get("source_revision")
            or ("synthetic-v1" if provenance_mode == "synthetic" else "")
        )
        if not source_revision:
            raise ValueError(f"real TrialSpec dataset {dataset['dataset_id']} requires source_revision")
        sample_source = {
            "dataset_id": dataset["dataset_id"],
            "source_revision": source_revision,
            "split": dataset["split"],
            "ordered_sample_ids": sample_ids,
            "record_format": "jsonl-record-bytes-v1",
            "canonicalization_contract": "preserve-selected-record-bytes-v1",
        }
        content_digest = store.digest_referenced_bytes(sample_refs)
        dataset["sample_hash"] = content_digest
        sample_datasets.append(
            {
                **sample_source,
                "sample_count": dataset["sample_count"],
                "raw_sample_refs": sample_refs,
                "content_digest": content_digest,
            }
        )
    sample_manifest = {
        "schema_version": "auto_research_sample_manifest_v4",
        "manifest_id": f"{variant.get('variant_id')}:sample-manifest",
        "provenance_mode": provenance_mode,
        "datasets": sample_datasets,
    }
    sample_ref = store.put_contract(
        sample_manifest,
        contract_kind="sample_manifest",
        schema_file="sample_manifest_v4.schema.json",
    )

    source_blobs: list[dict[str, Any]] = []
    if provenance_mode == "real":
        configured_files = execution.get("evaluator_source_files") or []
        if not configured_files:
            configured_paths = execution.get("evaluator_source_paths") or []
            workdir = Path(str(execution.get("workdir") or project_root)).resolve()
            for relative in configured_paths:
                candidate = (workdir / str(relative)).resolve()
                try:
                    candidate.relative_to(workdir)
                except ValueError as exc:
                    raise ValueError("evaluator source path escapes execution workdir") from exc
                paths = [candidate] if candidate.is_file() else sorted(path for path in candidate.rglob("*") if path.is_file()) if candidate.is_dir() else []
                configured_files.extend({"path": path.relative_to(workdir).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths)
        workdir = Path(str(execution.get("workdir") or project_root)).resolve()
        for entry in configured_files:
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ValueError("real evaluator source entries require paths")
            path = (workdir / str(entry["path"])).resolve()
            try:
                path.relative_to(workdir)
            except ValueError as exc:
                raise ValueError("evaluator source path escapes execution workdir") from exc
            raw = path.read_bytes()
            if entry.get("sha256") and hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("evaluator source changed after provenance enumeration")
            source_blobs.append(store.put_bytes(raw))
        if not source_blobs:
            raise ValueError("real TrialSpec requires readable evaluator source files")
    else:
        payloads = execution.get("evaluator_source_payloads") or [{"collector": execution.get("collector"), "mode": "synthetic"}]
        source_blobs = [store.put_bytes(canonical_contract_bytes(item)) for item in payloads]

    dependency_payloads = execution.get("dependency_payloads")
    if dependency_payloads is None:
        dependency_payloads = execution.get("dependencies") or []
    dependency_blobs = [store.put_bytes(canonical_contract_bytes(item)) for item in dependency_payloads]
    evaluator_config = {"metrics": metrics, "runtime": execution}
    config_blob = store.put_bytes(canonical_contract_bytes(evaluator_config))
    evaluator_id = str(execution.get("evaluator_id") or execution.get("collector") or "synthetic-evaluator")
    if len(evaluator_id) < 8:
        evaluator_id = f"{evaluator_id}-evaluator"
    evaluator_manifest = {
        "schema_version": "auto_research_evaluator_manifest_v2",
        "evaluator_id": evaluator_id,
        "provenance_mode": provenance_mode,
        "source_blobs": source_blobs,
        "dependency_blobs": dependency_blobs,
        "config_blob": config_blob,
        "config_digest": store.digest_referenced_bytes([config_blob]),
        "source_digest": store.digest_referenced_bytes(source_blobs),
        "dependency_digest": store.digest_referenced_bytes(dependency_blobs),
    }
    evaluator_ref = store.put_contract(
        evaluator_manifest,
        contract_kind="evaluator_manifest",
        schema_file="evaluator_manifest_v2.schema.json",
    )
    return sample_manifest, sample_ref, evaluator_manifest, evaluator_ref


def _trial_spec_from_plan(
    plan: dict[str, Any],
    variant: dict[str, Any],
    *,
    profile: str = "standard",
    project_root: Path | None = None,
) -> dict[str, Any]:
    raw_datasets = deepcopy(plan.get("datasets") or [])
    raw_metrics = deepcopy(plan.get("metrics") or [])
    statistical = deepcopy(plan.get("statistical_testing") or {})
    execution = deepcopy(plan.get("execution") or {})
    acceptance = deepcopy(plan.get("acceptance_criteria") or {})
    ablations = deepcopy(plan.get("ablation_matrix") or [])
    datasets = []
    for item in raw_datasets:
        source = item if isinstance(item, dict) else {"name": item}
        dataset_id = str(source.get("dataset_id") or source.get("name") or "")
        split = str(source.get("split") or "")
        if not dataset_id or not split:
            raise ValueError("TrialSpec datasets require dataset identity and split")
        sample_count = source.get("sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
            if execution.get("mode") == "simulate" or execution.get("collector") == "c2c_small_loop":
                sample_count = 1
            else:
                raise ValueError(f"TrialSpec dataset {dataset_id} requires a pre-registered sample_count")
        sample_hash = source.get("sample_hash") or canonical_hash(
            {"dataset_id": dataset_id, "split": split, "sample_count": sample_count, "sample_contract": source}
        )
        datasets.append({"dataset_id": dataset_id, "split": split, "sample_count": sample_count, "sample_hash": sample_hash})
    if not datasets:
        raise ValueError("TrialSpec requires datasets")
    metrics = []
    for item in raw_metrics:
        if not isinstance(item, dict):
            raise ValueError("TrialSpec metric entries must be objects")
        metric_id = str(item.get("metric_id") or item.get("name") or "")
        if not metric_id:
            raise ValueError("TrialSpec metric requires an identity")
        metrics.append(
            {
                "metric_id": metric_id,
                "objective": "maximize" if item.get("higher_is_better", True) else "minimize",
                "aggregation": "mean",
                "role": "primary" if item.get("primary") is True else "secondary",
            }
        )
    primary = [item for item in metrics if item["role"] == "primary"]
    if len(primary) != 1:
        raise ValueError("TrialSpec requires exactly one primary metric")
    primary_metric_id = primary[0]["metric_id"]
    seeds = [int(item) for item in statistical.get("seeds") or []]
    if not seeds:
        raise ValueError("TrialSpec requires pre-registered seeds")
    proxy_terminal = profile == "bootstrap"
    c2c_proxy_full = execution.get("collector") == "c2c_small_loop" and not proxy_terminal
    required_phases = ["proxy"] if proxy_terminal else (["proxy", "full"] if c2c_proxy_full else ["full"])
    terminal_phases = ["proxy"] if proxy_terminal else ["full"]
    required_roles = ["baseline", "candidate"]
    if ablations and not proxy_terminal:
        required_roles.append("ablation")
    constraints = []
    minimum_delta = acceptance.get("minimum_mean_delta")
    if not isinstance(minimum_delta, (int, float)) or isinstance(minimum_delta, bool):
        raise ValueError("TrialSpec requires a typed minimum_mean_delta")
    constraints.append(
        {
            "constraint_id": "primary-minimum-mean-delta",
            "kind": "minimum_mean_delta",
            "hard": True,
            "metric_id": primary_metric_id,
            "threshold": float(minimum_delta),
            "objective": primary[0]["objective"],
        }
    )
    maximum_regression = acceptance.get("maximum_dataset_regression", acceptance.get("max_dataset_regression"))
    if isinstance(maximum_regression, (int, float)) and not isinstance(maximum_regression, bool):
        constraints.append(
            {
                "constraint_id": "per-dataset-maximum-regression",
                "kind": "per_dataset_maximum_regression",
                "hard": True,
                "metric_id": primary_metric_id,
                "threshold": float(maximum_regression),
                "objective": primary[0]["objective"],
            }
        )
    if ablations and not proxy_terminal:
        constraints.append(
            {
                "constraint_id": "required-ablation-contrast",
                "kind": "required_ablation_contrast",
                "hard": True,
                "metric_id": primary_metric_id,
                "threshold": float(acceptance.get("minimum_ablation_delta", 0.0)),
                "objective": primary[0]["objective"],
            }
        )
    if acceptance.get("matched_coverage_ablation_required") and not proxy_terminal:
        required_roles.extend(["matched_control", "coverage"])
        constraints.extend(
            [
                {"constraint_id": "matched-control", "kind": "matched_control_constraint", "hard": True, "metric_id": primary_metric_id, "threshold": 0.0, "objective": primary[0]["objective"]},
                {"constraint_id": "coverage", "kind": "coverage_constraint", "hard": True, "metric_id": primary_metric_id, "threshold": 1.0, "objective": "maximize"},
            ]
        )
    required_artifacts = ["proxy_results"] if "proxy" in required_phases else []
    if "full" in required_phases:
        required_artifacts.append("main_results")
    if ablations and not proxy_terminal:
        required_artifacts.append("ablation_results")
    if "coverage" in required_roles:
        required_artifacts.append("coverage_results")
    if "matched_control" in required_roles:
        required_artifacts.append("matched_control_results")
    evidence_requirements = [
        {"requirement_id": "activation", "kind": "activation_evidence", "required": True, "applicable_phases": ["proxy" if "proxy" in required_phases else "full"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["activation_evidence"]},
    ]
    if "proxy" in required_phases:
        evidence_requirements.append({"requirement_id": "proxy-results", "kind": "proxy_results", "required": True, "applicable_phases": ["proxy"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["proxy_results"]})
    if "full" in required_phases:
        evidence_requirements.append({"requirement_id": "main-results", "kind": "main_results", "required": True, "applicable_phases": ["full"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["main_results"]})
    if ablations and not proxy_terminal:
        evidence_requirements.append({"requirement_id": "ablation-results", "kind": "ablation_results", "required": True, "applicable_phases": ["full"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["ablation_results"]})
    if "coverage" in required_roles:
        evidence_requirements.append({"requirement_id": "coverage-results", "kind": "coverage_results", "required": True, "applicable_phases": ["full"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["coverage_results"]})
    if "matched_control" in required_roles:
        evidence_requirements.append({"requirement_id": "matched-control-results", "kind": "matched_control_results", "required": True, "applicable_phases": ["full"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["matched_control_results"]})
    if execution.get("collector") == "c2c_small_loop":
        proxy_evidence = [
            ("proxy-baseline", "proxy_baseline_fingerprint"),
            ("proxy-cache", "proxy_cache_report"),
        ]
        evidence_requirements.extend(
            {"requirement_id": requirement_id, "kind": kind, "required": True, "applicable_phases": ["proxy"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS[kind]}
            for requirement_id, kind in proxy_evidence
        )
        if proxy_terminal:
            evidence_requirements.append({"requirement_id": "bootstrap-completion", "kind": "bootstrap_completion", "required": True, "applicable_phases": ["proxy"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["bootstrap_completion"]})
        else:
            evidence_requirements.append({"requirement_id": "full-readiness", "kind": "full_s3_readiness", "required": True, "applicable_phases": ["proxy"], "schema_version": _AUTHORITATIVE_EVIDENCE_SCHEMA_VERSIONS["full_s3_readiness"]})
    runtime_config = deepcopy(execution)
    provenance_mode = "synthetic" if execution.get("mode") == "simulate" else "real"
    dataset_ids = [item["dataset_id"] for item in datasets]
    if project_root is None:
        raise ValueError("TrialSpec production requires a project_root for immutable ContractStore manifests")
    sample_manifest, sample_manifest_ref, evaluator_manifest, evaluator_manifest_ref = _store_trial_contracts(
        project_root=project_root,
        raw_datasets=raw_datasets,
        datasets=datasets,
        metrics=metrics,
        execution=execution,
        variant=variant,
        provenance_mode=provenance_mode,
    )
    proxy_decision_policy = None
    phase_contracts = []
    adapter_id = _phase_adapter_id(execution, provenance_mode=provenance_mode)
    adapter_version = str(execution.get("adapter_version") or "1")
    source_snapshot_hash = str(evaluator_manifest["source_digest"])
    for phase in required_phases:
        phase_requirements = [
            item
            for item in evidence_requirements
            if phase in item["applicable_phases"] or "always" in item["applicable_phases"]
        ]
        phase_commands = _frozen_phase_commands(execution, phase=phase, adapter_id=adapter_id)
        command_plan = build_phase_command_plan(
            phase=phase,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            provenance_mode="synthetic" if provenance_mode == "synthetic" else "production",
            variant_spec_hash=str(variant["variant_spec_hash"]),
            source_snapshot_hash=source_snapshot_hash,
            command_values=phase_commands,
            expected_evidence=phase_requirements,
            default_cwd=str(execution.get("workdir") or project_root),
            project_root=project_root,
            coverage_contract={
                "mode": "exact_cartesian",
                "datasets": dataset_ids,
                "seeds": seeds,
                "metrics": [primary_metric_id],
                "roles": ["baseline", "candidate"] if phase == "proxy" else list(dict.fromkeys(required_roles)),
            },
            readiness_checks=(
                [{
                    "check_id": "activation-mechanism",
                    "check_kind": "activation_delta",
                    "predicate": {"field_path": "surface_measurements.delta", "comparator": "delta_gte", "threshold": float(acceptance.get("minimum_activation_delta", 0.0))},
                    "required_coverage": {"mode": "exact", "expected_surface_ids": list(variant.get("implementation_surface_ids") or ["c2c-surface"])},
                }]
                + ([] if proxy_terminal or phase != "proxy" else [{
                    "check_id": "proxy-ready-for-full",
                    "check_kind": "raw_measurement",
                    "predicate": {
                        "field_path": (
                            "readiness_checks.proxy-ready-for-full.measurement"
                            if execution.get("collector") == "c2c_small_loop" and provenance_mode != "synthetic"
                            else "ready"
                        ),
                        "comparator": ("gte" if execution.get("collector") == "c2c_small_loop" and provenance_mode != "synthetic" else "eq"),
                        "threshold": (1.0 if execution.get("collector") == "c2c_small_loop" and provenance_mode != "synthetic" else True),
                    },
                    "required_coverage": {"mode": "exact", "expected_surface_ids": []},
                }])
            ) if any(item["kind"] == "activation_evidence" for item in phase_requirements) else (),
        )
        command_plan_ref, command_plan_hash = store_phase_command_plan(project_root, command_plan)
        phase_contracts.append(
            {
                "phase": phase,
                "datasets": dataset_ids,
                "seeds": seeds,
                "roles": ["baseline", "candidate"] if phase == "proxy" else list(dict.fromkeys(required_roles)),
                "metrics": [primary_metric_id],
                "evidence_kinds": [item["kind"] for item in phase_requirements],
                "terminal": phase in terminal_phases,
                "consumes_direction_budget": profile == "standard" and phase in terminal_phases,
                "command_plan": command_plan,
                "command_plan_ref": command_plan_ref,
                "command_plan_hash": command_plan_hash,
                "derivation_plan": command_plan["derivation_plan"],
                "derivation_plan_ref": command_plan["derivation_plan_ref"],
                "derivation_plan_hash": command_plan["derivation_plan_hash"],
                "readiness_check_plan": command_plan["readiness_check_plan"],
                "readiness_check_plan_ref": command_plan["readiness_check_plan_ref"],
                "readiness_check_plan_hash": command_plan["readiness_check_plan_hash"],
            }
        )
    if "proxy" in required_phases:
        proxy_phase = next(item for item in phase_contracts if item["phase"] == "proxy")
        if proxy_phase["readiness_check_plan_ref"] is None:
            raise ValueError("proxy TrialSpec requires a frozen ReadinessCheckPlan")
        proxy_evidence_kinds = sorted(
            item["kind"] for item in evidence_requirements
            if "proxy" in item["applicable_phases"] or "always" in item["applicable_phases"]
        )
        proxy_decision_policy = build_proxy_decision_policy(
            primary_metric_id=primary_metric_id,
            objective=primary[0]["objective"],
            aggregation="paired_mean",
            datasets=dataset_ids,
            seeds=seeds,
            metric_ids=[primary_metric_id],
            roles=["baseline", "candidate"],
            aggregate_improvement_threshold=float(minimum_delta),
            per_dataset_maximum_regression=float(maximum_regression if isinstance(maximum_regression, (int, float)) and not isinstance(maximum_regression, bool) else 0.0),
            activation_delta_threshold=float(acceptance.get("minimum_activation_delta", 0.0)),
            activation_surface_ids=list(variant.get("implementation_surface_ids") or ["c2c-surface"]),
            readiness_check_ids=[] if proxy_terminal else ["proxy-ready-for-full"],
            readiness_check_plan_ref=proxy_phase["readiness_check_plan_ref"],
            readiness_check_plan_hash=proxy_phase["readiness_check_plan_hash"],
            evidence_kinds=proxy_evidence_kinds,
            mode="terminal_bootstrap" if proxy_terminal else "gate_to_full",
        )
    trial_spec = {
        "schema_version": TRIAL_SPEC_SCHEMA_VERSION,
        "protocol": {
            "protocol_id": f"{variant.get('variant_id')}:registered-protocol",
            "required_phases": required_phases,
            "terminal_phases": terminal_phases,
            "proxy_terminal_allowed": proxy_terminal,
            "aggregation": "mean",
        },
        "sample_manifest": sample_manifest,
        "sample_manifest_ref": sample_manifest_ref,
        "datasets": datasets,
        "metrics": metrics,
        "primary_metric_id": primary_metric_id,
        "statistical_testing": {
            "method": "paired" if len(seeds) > 1 else "none",
            "seeds": seeds,
            "require_complete_seed_coverage": True,
        },
        "required_roles": list(dict.fromkeys(required_roles)),
        "acceptance_constraints": constraints,
        "execution_contract": {
            "runtime_config": runtime_config,
            "runtime_config_hash": canonical_hash(runtime_config),
            "evaluator_provenance": evaluator_manifest,
            "evaluator_manifest_ref": evaluator_manifest_ref,
            "evaluator_hash": canonical_hash(evaluator_manifest),
            "phase_command_plan_hashes": {
                item["phase"]: item["command_plan_hash"] for item in phase_contracts
            },
        },
        "required_artifacts": required_artifacts,
        "evidence_requirements": evidence_requirements,
        "phase_contracts": phase_contracts,
        "proxy_decision_policy": proxy_decision_policy,
    }
    validate_trial_spec(trial_spec)
    return trial_spec


def _phase_adapter_id(execution: dict[str, Any], *, provenance_mode: str) -> str:
    collector = str(execution.get("collector") or "generic").replace("_", "-")
    if provenance_mode == "synthetic":
        return "synthetic-phase-adapter"
    if collector == "c2c-small-loop":
        return "c2c-phase-adapter"
    return "generic-external-adapter"


def _frozen_phase_commands(execution: dict[str, Any], *, phase: str, adapter_id: str) -> list[Any]:
    configured = execution.get("phase_commands")
    if isinstance(configured, dict) and isinstance(configured.get(phase), list):
        commands = deepcopy(configured[phase])
    else:
        raw = execution.get("commands")
        if isinstance(raw, dict):
            phase_value = raw.get(phase)
            if isinstance(phase_value, list):
                commands = deepcopy(phase_value)
            elif phase_value:
                commands = [deepcopy(phase_value)]
            else:
                commands = []
        elif isinstance(raw, list):
            commands = deepcopy(raw) if phase == "full" else []
        elif raw:
            commands = [deepcopy(raw)] if phase == "full" else []
        else:
            commands = []
    if commands:
        return commands
    if adapter_id == "c2c-phase-adapter":
        return [["auto-research-adapter", "c2c", phase, "freeze-required"]]
    if adapter_id == "synthetic-phase-adapter":
        return [["auto-research-adapter", "synthetic", phase]]
    raise ValueError(f"{adapter_id} {phase} phase requires explicit executable commands")

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
