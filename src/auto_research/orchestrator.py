"""Pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactManager
from .c2c import build_c2c_project_config, snapshot_c2c_repo, write_c2c_project_config
from .code_patch import is_retryable_patch_manifest
from .config import apply_runtime_overrides, load_project_config, load_root_config
from .judges import gate_s0, gate_s1, gate_s2, gate_s3, gate_s4, gate_s5
from .importers import ConsensusImporter
from .llm import ModelClient
from .method_memory import append_shared_c2c_method_failure
from .orchestration_state import OrchestrationStateManager
from .reporting import build_project_report
from .s0_enrichment import DeepSeekS0SemanticEnricher
from .registry import begin_stage, block_stage, complete_stage, fail_stage, increment_judge_retry, invalidate_from, load_registry, pause_stage_retryable, save_registry, set_review_outcome
from .stage_contracts import StageContractManager
from .utils import now_utc, read_json, read_yaml, write_json
from .workspace import init_workspace
from .hitl import HITLManager
from .agents.base import AgentContext
from .agents.experiment import ExperimentAgent
from .agents.intake import IntakeAgent
from .agents.literature import LiteratureAgent
from .agents.plan import PlanAgent
from .agents.review import ReviewAgent
from .agents.writing import WritingAgent


class Orchestrator:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path.cwd()

    def init_project(self, topic: str, *, project_id: str | None = None, simulate: bool | None = None) -> str:
        config = apply_runtime_overrides(load_root_config(), simulate=simulate)
        paths = init_workspace(config, topic, project_id=project_id, simulate=simulate)
        self._log_session(paths.root, action="init_project", details={"topic": topic, "simulate": simulate})
        return paths.root.name

    def init_c2c_project(
        self,
        topic: str,
        *,
        target_repo: Path,
        ref_paper: Path,
        ref_rebuttal: Path,
        env_python: Path,
        project_id: str | None = None,
        simulate: bool | None = None,
    ) -> str:
        config = apply_runtime_overrides(load_root_config(), simulate=simulate)
        paths = init_workspace(config, topic, project_id=project_id, simulate=simulate)
        snapshot_rel = "external/c2c_snapshot"
        snapshot_manifest = snapshot_c2c_repo(target_repo, paths.root / snapshot_rel)
        write_c2c_project_config(
            paths.root,
            build_c2c_project_config(
                topic=topic,
                target_repo=target_repo.expanduser().resolve(),
                snapshot_path=snapshot_rel,
                ref_paper=ref_paper.expanduser().resolve(),
                ref_rebuttal=ref_rebuttal.expanduser().resolve(),
                env_python=env_python.expanduser().resolve(),
            ),
        )
        ArtifactManager(paths.root).write_json(
            "S3_experiment",
            "c2c/repo_snapshot_manifest.json",
            snapshot_manifest,
            artifact_type="c2c_repo_snapshot",
            summary="C2C source snapshot manifest",
        )
        self._log_session(
            paths.root,
            action="init_c2c_project",
            details={
                "topic": topic,
                "target_repo": str(target_repo),
                "snapshot": snapshot_rel,
                "ref_paper": str(ref_paper),
                "ref_rebuttal": str(ref_rebuttal),
                "env_python": str(env_python),
                "simulate": simulate,
            },
        )
        return paths.root.name

    def start(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        return self._run(project_root)

    def resume(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        self._log_session(project_root, action="resume", details={})
        return self._run(project_root)

    def run_review(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        config = load_project_config(project_root)
        context = self._context(project_root, config)
        review = ReviewAgent(context).run(iteration=load_registry(project_root / "meta" / "registry.yaml")["iteration"])
        return review

    def status(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        registry = load_registry(project_root / "meta" / "registry.yaml")
        stage_counts = {}
        artifacts = ArtifactManager(project_root)
        for stage_key in registry["stages"]:
            try:
                stage_counts[stage_key] = len(artifacts.list_stage_artifacts(stage_key))
            except FileNotFoundError:
                stage_counts[stage_key] = 0
        return {
            "project_id": registry["project_id"],
            "research_topic": registry["research_topic"],
            "status": registry["status"],
            "current_stage": registry["current_stage"],
            "iteration": registry["iteration"],
            "blocked_reason": registry.get("blocked_reason"),
            "pause_type": registry.get("pause_type"),
            "resume_instruction": registry.get("resume_instruction"),
            "artifact_counts": stage_counts,
        }

    def report(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        return build_project_report(project_root)

    def enrich_s0(
        self,
        project_id: str,
        *,
        limit: int | str = 8,
        source_types: list[str] | None = None,
        dry_run: bool = False,
        refresh: bool = False,
        workers: int = 1,
    ) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        config = load_project_config(project_root)
        result = DeepSeekS0SemanticEnricher(project_root, config).run(
            limit=limit,
            source_types=source_types,
            dry_run=dry_run,
            refresh=refresh,
            workers=workers,
        )
        self._log_session(
            project_root,
            action="enrich_s0",
            details={
                "limit": limit,
                "source_types": source_types,
                "dry_run": dry_run,
                "refresh": refresh,
                "workers": workers,
                "artifacts": result.get("artifacts", []),
            },
        )
        return result

    def catchup(self, project_id: str) -> str:
        project_root = self._project_root(project_id)
        registry = load_registry(project_root / "meta" / "registry.yaml")
        selected_idea = "not selected"
        ideas_path = project_root / "literature" / "ideas.json"
        if ideas_path.exists():
            ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
            selected_idea = next((idea["title"] for idea in ideas if idea.get("selected")), ideas[0]["title"])
        plan_summary = "No plan yet."
        plan_path = project_root / "plan" / "plan.yaml"
        if plan_path.exists():
            plan = read_yaml(plan_path, default={}) or {}
            selected = plan.get("selected_idea") if isinstance(plan.get("selected_idea"), dict) else {}
            next_variant = plan.get("next_variant") if isinstance(plan.get("next_variant"), dict) else {}
            plan_summary = "\n".join(
                item
                for item in [
                    f"Selected: {selected.get('id') or selected.get('title')}" if selected else "",
                    f"Next variant: {next_variant.get('id') or next_variant.get('title')}" if next_variant else "",
                ]
                if item
            ) or "Plan exists."
        review_summary = (project_root / "review" / "meta_review_round_1.md").read_text(encoding="utf-8") if (project_root / "review" / "meta_review_round_1.md").exists() else "No review yet."
        return "\n".join(
            [
                f"Project: {registry['project_id']}",
                f"Topic: {registry['research_topic']}",
                f"Selected idea: {selected_idea}",
                f"Current stage: {registry['current_stage']}",
                "",
                "Plan summary:",
                plan_summary[:600],
                "",
                "Latest review:",
                review_summary[:600],
            ]
        )

    def import_consensus(self, project_id: str, file_path: str, *, label: str | None = None) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        importer = ConsensusImporter(project_root)
        result = importer.import_file(Path(file_path), label=label)
        self._log_session(project_root, action="import_consensus", details={"file": file_path, "label": label})
        return result

    def _run(self, project_root: Path) -> dict[str, Any]:
        config = load_project_config(project_root)
        context = self._context(project_root, config)
        registry_path = project_root / "meta" / "registry.yaml"
        registry = load_registry(registry_path)
        state = OrchestrationStateManager(project_root)
        state.initialize(registry)
        contracts = StageContractManager(project_root)
        contracts.initialize_all(config=config, iteration=registry.get("iteration"))
        topic = registry["research_topic"]
        iteration = registry["iteration"]
        hitl = HITLManager(project_root, config)
        self._log_session(project_root, action="start_run", details={"stage": registry["current_stage"], "iteration": iteration})
        state.run_started(registry)

        while registry["current_stage"] != "DONE":
            stage_key = registry["current_stage"]
            begin_stage(registry, stage_key)
            save_registry(registry_path, registry)
            state.stage_started(registry, stage_key)
            contracts.stage_started(stage_key, iteration=registry.get("iteration"), config=config)
            if stage_key == "S0_intake":
                result = IntakeAgent(context).run(topic)
                if result.get("status") == "blocked":
                    block_stage(registry, stage_key, result.get("blocked_reason", "Static intake stage blocked."))
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, registry["blocked_reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=registry["blocked_reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    self._log_session(project_root, action="blocked", details={"stage": stage_key, "reason": registry["blocked_reason"]})
                    return {"status": "blocked", "stage": stage_key, "reason": registry["blocked_reason"]}
                gate_report = gate_s0(project_root, config)
            elif stage_key == "S1_literature":
                result = LiteratureAgent(context).run(topic)
                if result.get("status") == "blocked":
                    block_stage(registry, stage_key, result.get("blocked_reason", "Literature stage blocked."))
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, registry["blocked_reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=registry["blocked_reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    self._log_session(project_root, action="blocked", details={"stage": stage_key, "reason": registry["blocked_reason"]})
                    return {"status": "blocked", "stage": stage_key, "reason": registry["blocked_reason"]}
                gate_report = gate_s1(project_root, config)
            elif stage_key == "S2_plan":
                preflight_route = self._route_existing_s2_5_validation_failure_before_s2_agent(
                    project_root,
                    registry,
                    config=config,
                )
                if preflight_route:
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, preflight_route)
                    self._log_session(project_root, action="s2_5_validation_failure_preflight_to_repair", details=preflight_route)
                    context = self._context(project_root, load_project_config(project_root))
                result = PlanAgent(context).run()
                gate_report = gate_s2(project_root, config)
            elif stage_key == "S3_experiment":
                result = ExperimentAgent(context).run()
                if result.get("status") == "blocked":
                    resource_pause = self._s3_resource_retry_pause_details(project_root, result, result["blocked_reason"])
                    if resource_pause:
                        pause_stage_retryable(registry, stage_key, resource_pause["reason"], pause_type=resource_pause["pause_type"])
                        save_registry(registry_path, registry)
                        state.stage_retryable_paused(registry, stage_key, resource_pause["reason"])
                        contracts.stage_stopped(
                            stage_key,
                            status="retryable_paused",
                            reason=resource_pause["reason"],
                            artifacts=result.get("artifacts", []),
                            config=config,
                            iteration=registry.get("iteration"),
                        )
                        self._log_session(project_root, action="s3_resource_retry_paused", details=resource_pause)
                        return {
                            "status": "retryable_paused",
                            "stage": stage_key,
                            "reason": resource_pause["reason"],
                            "pause_type": resource_pause["pause_type"],
                            "resume_instruction": registry.get("resume_instruction"),
                        }
                    if self._should_route_s3_implementation_failure_to_s2(config, project_root, registry):
                        routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        self._log_session(project_root, action="s3_implementation_failure_to_s2", details=routed)
                        if routed["status"] == "routed":
                            contracts.stage_stopped(
                                stage_key,
                                status="feedback_routed",
                                reason=result["blocked_reason"],
                                artifacts=result.get("artifacts", []),
                                config=config,
                                iteration=registry.get("iteration"),
                            )
                            context = self._context(project_root, load_project_config(project_root))
                            continue
                    if self._should_route_s3_repairable_proxy_to_s2(config, project_root, registry):
                        routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        repair_action = "s3_repairable_proxy_to_s1" if routed.get("next_stage") == "S1_literature" else "s3_repairable_proxy_to_s2"
                        self._log_session(project_root, action=repair_action, details=routed)
                        if routed["status"] == "routed":
                            contracts.stage_stopped(
                                stage_key,
                                status="feedback_routed",
                                reason=result["blocked_reason"],
                                artifacts=result.get("artifacts", []),
                                config=config,
                                iteration=registry.get("iteration"),
                            )
                            context = self._context(project_root, load_project_config(project_root))
                            continue
                    if self._should_route_s3_repairable_proxy_to_s1(config, project_root, registry):
                        routed = self._route_s3_repairable_proxy_to_s1(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        self._log_session(project_root, action="s3_repairable_proxy_to_s1", details=routed)
                        if routed["status"] == "routed":
                            contracts.stage_stopped(
                                stage_key,
                                status="feedback_routed",
                                reason=result["blocked_reason"],
                                artifacts=result.get("artifacts", []),
                                config=config,
                                iteration=registry.get("iteration"),
                            )
                            context = self._context(project_root, load_project_config(project_root))
                            continue
                        state.stage_blocked(registry, stage_key, routed["reason"])
                        contracts.stage_stopped(stage_key, status="blocked", reason=routed["reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        return {"status": "blocked", "stage": stage_key, "reason": routed["reason"]}
                    if self._should_route_s3_proxy_rejected_to_s2(config, project_root, registry):
                        routed = self._route_s3_proxy_rejected_to_s2(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        self._log_session(project_root, action="s3_proxy_rejected_to_s2", details=routed)
                        if routed["status"] == "routed":
                            contracts.stage_stopped(
                                stage_key,
                                status="feedback_routed",
                                reason=result["blocked_reason"],
                                artifacts=result.get("artifacts", []),
                                config=config,
                                iteration=registry.get("iteration"),
                            )
                            context = self._context(project_root, load_project_config(project_root))
                            continue
                    if self._should_route_s3_proxy_rejected_to_s1(config, project_root, registry):
                        routed = self._route_s3_proxy_rejected_to_s1(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        self._log_session(project_root, action="s3_proxy_rejected_to_s1", details=routed)
                        if routed["status"] == "routed":
                            contracts.stage_stopped(
                                stage_key,
                                status="feedback_routed",
                                reason=result["blocked_reason"],
                                artifacts=result.get("artifacts", []),
                                config=config,
                                iteration=registry.get("iteration"),
                            )
                            context = self._context(project_root, load_project_config(project_root))
                            continue
                        state.stage_blocked(registry, stage_key, routed["reason"])
                        contracts.stage_stopped(stage_key, status="blocked", reason=routed["reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        return {"status": "blocked", "stage": stage_key, "reason": routed["reason"]}
                    block_stage(registry, stage_key, result["blocked_reason"])
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, result["blocked_reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=result["blocked_reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    self._log_session(project_root, action="blocked", details={"stage": stage_key, "reason": result["blocked_reason"]})
                    return {"status": "blocked", "stage": stage_key, "reason": result["blocked_reason"]}
                gate_report = gate_s3(project_root, config)
                ok, reason = gate_report.legacy_tuple()
                gate_record = self._write_gate_report(context.artifacts, stage_key, gate_report)
                result.setdefault("artifacts", []).append(gate_record["path"])
                gate_payload = gate_report.to_dict()
                gate_payload["report_path"] = gate_record["path"]
                state.gate_recorded(registry, stage_key, passed=ok, reason=reason, report=gate_payload)
                contracts.gate_recorded(stage_key, gate_payload, report_path=gate_record["path"])
                if not ok and self._should_route_s3_implementation_failure_to_s2(config, project_root, registry):
                    routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_implementation_failure_to_s2", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                if not ok and self._should_route_s3_repairable_proxy_to_s2(config, project_root, registry):
                    routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    repair_action = "s3_repairable_proxy_to_s1" if routed.get("next_stage") == "S1_literature" else "s3_repairable_proxy_to_s2"
                    self._log_session(project_root, action=repair_action, details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                if not ok and self._should_route_s3_repairable_proxy_to_s1(config, project_root, registry):
                    routed = self._route_s3_repairable_proxy_to_s1(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_repairable_proxy_to_s1", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                    state.stage_blocked(registry, stage_key, routed["reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=routed["reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": routed["reason"]}
                if not ok and self._should_route_s3_proxy_rejected_to_s2(config, project_root, registry):
                    routed = self._route_s3_proxy_rejected_to_s2(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_proxy_rejected_to_s2", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                if not ok and self._should_route_s3_proxy_rejected_to_s1(config, project_root, registry):
                    routed = self._route_s3_proxy_rejected_to_s1(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_proxy_rejected_to_s1", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                    state.stage_blocked(registry, stage_key, routed["reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=routed["reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": routed["reason"]}
                if not ok and self._should_route_s3_failure_to_s1(config, project_root, result):
                    routed = self._route_s3_failure_to_s1(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_failure_feedback_to_s1", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
                    state.stage_blocked(registry, stage_key, routed["reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=routed["reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": routed["reason"]}
            elif stage_key == "S4_writing":
                LiteratureAgent(context).run(topic, phase="related_work_audit")
                result = WritingAgent(context).run()
                gate_report = gate_s4(project_root, config)
            elif stage_key == "S5_review":
                result = ReviewAgent(context).run(iteration=registry["iteration"])
                set_review_outcome(registry, decision=result["decision"], score=result["score"], revision_dispatch=result["dispatch_path"])
                gate_report = gate_s5(project_root, config)
            else:
                fail_stage(registry, stage_key, f"Unknown stage {stage_key}")
                save_registry(registry_path, registry)
                state.stage_failed(registry, stage_key, f"Unknown stage {stage_key}")
                contracts.stage_stopped(stage_key, status="failed", reason=f"Unknown stage {stage_key}", config=config, iteration=registry.get("iteration"))
                return {"status": "failed", "stage": stage_key}

            if stage_key != "S3_experiment":
                ok, reason = gate_report.legacy_tuple()
                gate_record = self._write_gate_report(context.artifacts, stage_key, gate_report)
                result.setdefault("artifacts", []).append(gate_record["path"])
                gate_payload = gate_report.to_dict()
                gate_payload["report_path"] = gate_record["path"]
                state.gate_recorded(registry, stage_key, passed=ok, reason=reason, report=gate_payload)
                contracts.gate_recorded(stage_key, gate_payload, report_path=gate_record["path"])

            if ok:
                # --- HITL approval gate ---
                hitl_decision = self._maybe_request_approval(
                    hitl, stage_key, result, project_root, registry,
                )
                if hitl_decision is not None:
                    if hitl_decision.action == "reject":
                        block_stage(registry, stage_key, f"Rejected by human: {hitl_decision.guidance}")
                        save_registry(registry_path, registry)
                        state.stage_blocked(registry, stage_key, f"Rejected by human: {hitl_decision.guidance}")
                        contracts.stage_stopped(stage_key, status="blocked", reason=f"Rejected by human: {hitl_decision.guidance}", artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        self._log_session(project_root, action="hitl_rejected", details={"stage": stage_key, "guidance": hitl_decision.guidance})
                        return {"status": "blocked", "stage": stage_key, "reason": f"Rejected by human: {hitl_decision.guidance}"}
                    if hitl_decision.action == "guide":
                        # Re-run current stage with human guidance
                        self._log_session(project_root, action="hitl_guidance", details={"stage": stage_key, "guidance": hitl_decision.guidance})
                        registry["stages"][stage_key]["guidance"] = hitl_decision.guidance
                        save_registry(registry_path, registry)
                        state.judge_retry(registry, stage_key, retries=registry["stages"][stage_key].get("judge_retries", 0), reason=f"HITL guidance: {hitl_decision.guidance}")
                        continue  # re-run same stage
                    # approve – fall through to complete_stage
                # --- end HITL gate ---

                complete_stage(registry, stage_key, artifacts=result.get("artifacts", []))
                save_registry(registry_path, registry)
                state.stage_completed(registry, stage_key, artifacts=result.get("artifacts", []))
                contracts.stage_completed(stage_key, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                self._log_session(project_root, action="stage_completed", details={"stage": stage_key})
                stop_after = config.get("orchestration", {}).get("stop_after_stage")
                if stop_after == stage_key:
                    registry["current_stage"] = "DONE"
                    registry["status"] = "completed"
                    save_registry(registry_path, registry)
                    state.mark_completed(registry)
                    break
                if stage_key == "S5_review":
                    if result["decision"] == "ACCEPT":
                        registry["current_stage"] = "DONE"
                        registry["status"] = "completed"
                        save_registry(registry_path, registry)
                        state.mark_completed(registry)
                        break
                    if result["decision"] == "REVISE":
                        self._execute_targeted_revisions(project_root, context, registry)
                        save_registry(registry_path, registry)
                        state.revision_loop_incremented(registry, reason="review_revise")
                        continue
                continue

            retryable_pause = self._retryable_pause_details(project_root, stage_key, gate_report, reason)
            if retryable_pause:
                pause_stage_retryable(registry, stage_key, retryable_pause["reason"], pause_type=retryable_pause["pause_type"])
                save_registry(registry_path, registry)
                state.stage_retryable_paused(registry, stage_key, retryable_pause["reason"])
                contracts.stage_stopped(
                    stage_key,
                    status="retryable_paused",
                    reason=retryable_pause["reason"],
                    artifacts=result.get("artifacts", []),
                    config=config,
                    iteration=registry.get("iteration"),
                )
                self._log_session(project_root, action="retryable_paused", details=retryable_pause)
                return {
                    "status": "retryable_paused",
                    "stage": stage_key,
                    "reason": retryable_pause["reason"],
                    "pause_type": retryable_pause["pause_type"],
                    "resume_instruction": registry.get("resume_instruction"),
                }

            s2_5_implementation_repair = self._route_s2_5_validation_failure_to_repair(
                project_root,
                registry,
                result,
                gate_report,
                reason,
                config=config,
            )
            if s2_5_implementation_repair:
                save_registry(registry_path, registry)
                state.failure_feedback_routed(registry, s2_5_implementation_repair)
                self._log_session(project_root, action="s2_5_validation_failure_to_repair", details=s2_5_implementation_repair)
                contracts.stage_stopped(
                    stage_key,
                    status="feedback_routed",
                    reason=s2_5_implementation_repair["reason"],
                    artifacts=result.get("artifacts", []),
                    config=config,
                    iteration=registry.get("iteration"),
                )
                context = self._context(project_root, load_project_config(project_root))
                continue

            retries = increment_judge_retry(registry, stage_key)
            max_retries = config.get("orchestration", {}).get("judge_max_retries", 2)
            if retries <= max_retries:
                save_registry(registry_path, registry)
                state.judge_retry(registry, stage_key, retries=retries, reason=reason)
                self._log_session(project_root, action="judge_retry", details={"stage": stage_key, "retries": retries, "reason": reason})
                continue
            fail_stage(registry, stage_key, reason)
            save_registry(registry_path, registry)
            state.stage_failed(registry, stage_key, reason)
            contracts.stage_stopped(stage_key, status="failed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
            self._log_session(project_root, action="stage_failed", details={"stage": stage_key, "reason": reason})
            return {"status": "failed", "stage": stage_key, "reason": reason}
        state.mark_completed(registry)
        return {"status": registry["status"], "project_id": registry["project_id"]}

    @staticmethod
    def _retryable_pause_details(project_root: Path, stage_key: str, gate_report: Any, reason: str) -> dict[str, Any] | None:
        if stage_key != "S2_plan":
            return None
        patch_manifest_path = project_root / "plan" / "code_patches" / "patch_manifest.json"
        if not patch_manifest_path.exists():
            return None
        patch_manifest = read_json(patch_manifest_path, default={}) or {}
        if not is_retryable_patch_manifest(patch_manifest):
            return None
        gate_payload = gate_report.to_dict() if hasattr(gate_report, "to_dict") else {}
        retry_checks = [
            check
            for check in gate_payload.get("checks", [])
            if isinstance(check, dict)
            and check.get("status") == "NEEDS_RETRY"
            and check.get("name") in {"s2_5_patch_manifest_status", "s2_5_executable_patch"}
        ]
        if not retry_checks:
            return None
        retryable_count = patch_manifest.get("retryable_patch_count")
        manifest_status = patch_manifest.get("status")
        resource_retry = _patch_manifest_has_resource_retry(patch_manifest)
        if resource_retry:
            pause_type = "runtime_smoke_resource_retry"
            pause_reason = (
                "S2.5 runtime smoke could not obtain a GPU with enough free memory after automatic GPU selection, "
                f"OOM retry, and resource wait (patch_manifest.status={manifest_status}, retryable_patch_count={retryable_count}). "
                "No Codex repair or S1/S2/S2.5 attempt budget was consumed; wait for GPU memory to become available, then resume this project."
            )
        else:
            pause_type = "codex_quota_or_rate_limit"
            pause_reason = (
                "S2.5 patch generation hit a retryable Codex/backend quota or rate-limit failure "
                f"(patch_manifest.status={manifest_status}, retryable_patch_count={retryable_count}). "
                "No S1/S2/S2.5 attempt budget was consumed; wait for quota recovery, then resume this project."
            )
        return {
            "pause_type": pause_type,
            "reason": pause_reason,
            "stage": stage_key,
            "patch_manifest": "plan/code_patches/patch_manifest.json",
            "patch_manifest_status": manifest_status,
            "retryable_patch_count": retryable_count,
            "resource_retry": resource_retry,
            "gate_reason": reason,
            "gate_checks": retry_checks,
        }

    @staticmethod
    def _s3_resource_retry_pause_details(project_root: Path, result: dict[str, Any], reason: str) -> dict[str, Any] | None:
        if not _s3_result_has_resource_retry(project_root):
            return None
        return {
            "pause_type": "s3_proxy_resource_retry",
            "reason": (
                "S3 cheap proxy hit a GPU OOM/resource failure. This is not an S2.5 implementation repair: "
                "no Codex repair and no S2.5 repair budget should be consumed. Wait for GPU memory or resume "
                "so the proxy can retry with resource-aware GPU selection."
            ),
            "stage": "S3_experiment",
            "result_status": result.get("status"),
            "blocked_reason": reason,
            "main_results": "experiment/results/main_results.json",
        }

    @staticmethod
    def _route_existing_s2_5_validation_failure_before_s2_agent(
        project_root: Path,
        registry: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if registry.get("current_stage") != "S2_plan":
            return None
        patch_manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
        if not _patch_manifest_has_s2_5_implementation_failure(patch_manifest):
            return None
        route_key = _s2_5_patch_manifest_repair_route_key(project_root, registry, patch_manifest)
        dispatch = read_json(project_root / "plan" / "s2_5_repair_dispatch.json", default={}) or {}
        if _s2_5_repair_dispatch_matches_route_key(dispatch, route_key, registry):
            return None
        reason = "Existing S2.5 patch manifest has implementation failure; route directly to S2.5-only repair before rerunning S2 planner."
        return Orchestrator._route_s2_5_validation_failure_to_repair(
            project_root,
            registry,
            {"status": "preflight_routed"},
            {"checks": []},
            reason,
            config=config,
        )

    @staticmethod
    def _route_s2_5_validation_failure_to_repair(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        gate_report: Any,
        reason: str,
        *,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if registry.get("current_stage") != "S2_plan":
            return None
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True):
            return None
        patch_manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
        if not _patch_manifest_has_s2_5_implementation_failure(patch_manifest):
            return None
        iteration_key = str(registry.get("iteration") or 1)
        route_key = _s2_5_patch_manifest_repair_route_key(project_root, registry, patch_manifest)
        implementation_routes_by_candidate = registry.setdefault("implementation_repair_routes_by_candidate", {})
        previous_routes = int(implementation_routes_by_candidate.get(route_key) or 0)
        max_routes = _implementation_repair_route_budget(feedback_cfg)
        if previous_routes >= max_routes:
            return None
        implementation_routes_by_candidate[route_key] = previous_routes + 1
        route_count = implementation_routes_by_candidate[route_key]
        implementation_routes = registry.setdefault("implementation_repair_routes", {})
        implementation_routes[iteration_key] = int(implementation_routes.get(iteration_key) or 0) + 1
        performance_feedback = _s2_5_patch_manifest_performance_feedback(
            project_root,
            registry,
            patch_manifest,
            reason=reason,
            route_count=route_count,
            max_routes=max_routes,
            gate_report=gate_report,
        )
        repair_dispatch = _write_s2_5_only_repair_dispatch(
            project_root,
            registry,
            performance_feedback,
            reason=reason,
            route_count=route_count,
        )
        write_json(project_root / "plan" / "performance_feedback.json", performance_feedback)
        trace_path = project_root / "meta" / "iteration_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "timestamp": now_utc(),
            "from_stage": "S2_plan",
            "to_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "reason": reason,
            "result_status": result.get("status"),
            "repair_route": "s2_5_validation_failure",
            "repair_lane": "s2_5_only_implementation_repair",
            "failure_class": "implementation_failure",
            "route_count": route_count,
            "repair_route_key": route_key,
            "performance_feedback": performance_feedback.get("summary", {}),
            "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json",
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        invalidate_from(registry, "S2_plan", invalidated_by="s2_5_validation_failure_repair")
        for stage_key in ["S2_plan", "S3_experiment"]:
            registry["stages"][stage_key]["judge_retries"] = 0
        return {
            "status": "routed",
            "reason": reason,
            "next_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "repair_lane": "s2_5_only_implementation_repair",
            "skips_s2_planner": True,
            "failure_class": "implementation_failure",
            "route_count": route_count,
            "repair_route_key": route_key,
            "does_not_consume_same_direction_attempt": True,
            "performance_feedback_path": "plan/performance_feedback.json",
            "performance_feedback": performance_feedback.get("summary", {}),
            "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json",
            "s2_5_repair_dispatch": {
                "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
                "variant_fingerprint": repair_dispatch.get("variant_fingerprint"),
                "implementation_failure_signals": repair_dispatch.get("implementation_failure_signals") or [],
            },
        }

    @staticmethod
    def _should_route_s3_failure_to_s1(config: dict[str, Any], project_root: Path, result: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_s3_failure_to_s1", True):
            return False
        if _s3_feedback_failure_class(project_root) == "implementation_failure":
            return False
        return result.get("status") in {"not_viable", "partial", "failed"}

    @staticmethod
    def _should_route_s3_implementation_failure_to_s2(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True):
            return False
        route_enabled = feedback_cfg.get("route_implementation_failure_to_s2")
        if route_enabled is None:
            route_enabled = feedback_cfg.get("route_repairable_proxy_to_s2", True)
        if not route_enabled or _s3_feedback_failure_class(project_root) != "implementation_failure":
            return False
        route_key = _s3_implementation_repair_route_key(project_root, registry)
        implementation_routes = registry.setdefault("implementation_repair_routes_by_candidate", {})
        previous_routes = int(implementation_routes.get(route_key) or 0)
        return previous_routes < _implementation_repair_route_budget(feedback_cfg)

    @staticmethod
    def _should_route_s3_repairable_proxy_to_s2(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_repairable_proxy_to_s2", True):
            return False
        if not _s3_result_has_repairable_proxy_risk(project_root):
            return False
        if _s3_feedback_failure_class(project_root) == "implementation_failure":
            route_key = _s3_implementation_repair_route_key(project_root, registry)
            implementation_routes = registry.setdefault("implementation_repair_routes_by_candidate", {})
            previous_routes = int(implementation_routes.get(route_key) or 0)
            max_routes = _implementation_repair_route_budget(feedback_cfg)
            return previous_routes < max_routes
        repair_routes = registry.setdefault("repair_routes", {})
        iteration_key = str(registry.get("iteration") or 1)
        max_failures = _repairable_proxy_failure_budget(feedback_cfg)
        previous_failures = int(repair_routes.get(iteration_key) or 0)
        return previous_failures + 1 < max_failures

    @staticmethod
    def _should_route_s3_repairable_proxy_to_s1(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_s3_failure_to_s1", True):
            return False
        if not _s3_result_has_repairable_proxy_risk(project_root):
            return False
        if _s3_feedback_failure_class(project_root) == "implementation_failure":
            return False
        repair_routes = registry.get("repair_routes") or {}
        iteration_key = str(registry.get("iteration") or 1)
        max_failures = _repairable_proxy_failure_budget(feedback_cfg)
        previous_failures = int(repair_routes.get(iteration_key) or 0)
        return previous_failures + 1 >= max_failures

    @staticmethod
    def _should_route_s3_proxy_rejected_to_s2(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True):
            return False
        route_enabled = feedback_cfg.get("route_proxy_rejected_to_s2")
        if route_enabled is None:
            route_enabled = feedback_cfg.get("route_repairable_proxy_to_s2", False)
        if not route_enabled or not _s3_result_has_proxy_rejection(project_root):
            return False
        if _s3_feedback_failure_class(project_root) == "implementation_failure":
            return False
        iteration_key = str(registry.get("iteration") or 1)
        max_failures = _same_direction_proxy_failure_budget(feedback_cfg)
        previous_failures = _same_direction_proxy_failure_count(registry, iteration_key)
        return previous_failures + 1 < max_failures

    @staticmethod
    def _should_route_s3_proxy_rejected_to_s1(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_s3_failure_to_s1", True):
            return False
        if not _s3_result_has_proxy_rejection(project_root):
            return False
        if _s3_feedback_failure_class(project_root) == "implementation_failure":
            return False
        route_enabled = feedback_cfg.get("route_proxy_rejected_to_s2")
        if route_enabled is None:
            route_enabled = feedback_cfg.get("route_repairable_proxy_to_s2", False)
        if not route_enabled:
            return True
        iteration_key = str(registry.get("iteration") or 1)
        max_failures = _same_direction_proxy_failure_budget(feedback_cfg)
        previous_failures = _same_direction_proxy_failure_count(registry, iteration_key)
        return previous_failures + 1 >= max_failures

    @staticmethod
    def _route_s3_repairable_proxy_to_s2(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        iteration_key = str(registry.get("iteration") or 1)
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        failure_class = _s3_feedback_failure_class(project_root)
        if failure_class == "implementation_failure":
            route_key = _s3_implementation_repair_route_key(project_root, registry)
            implementation_routes_by_candidate = registry.setdefault("implementation_repair_routes_by_candidate", {})
            implementation_routes_by_candidate[route_key] = int(implementation_routes_by_candidate.get(route_key) or 0) + 1
            route_count = implementation_routes_by_candidate[route_key]
            implementation_routes = registry.setdefault("implementation_repair_routes", {})
            implementation_routes[iteration_key] = int(implementation_routes.get(iteration_key) or 0) + 1
            failure_count = 0
            failure_budget = _same_direction_proxy_failure_budget(feedback_cfg)
            repair_route = "implementation_failure"
        else:
            route_key = ""
            repair_routes = registry.setdefault("repair_routes", {})
            repair_routes[iteration_key] = int(repair_routes.get(iteration_key) or 0) + 1
            route_count = repair_routes[iteration_key]
            failure_count = route_count
            failure_budget = _repairable_proxy_failure_budget(feedback_cfg)
            repair_route = "repairable_proxy_risk"
        performance_feedback = _s3_proxy_performance_feedback(
            project_root,
            registry,
            result,
            reason,
            route_count=route_count,
            failure_count=failure_count,
            max_failures=failure_budget,
            route=repair_route,
        )
        if failure_class == "implementation_failure":
            performance_feedback.setdefault("summary", {})["failure_class"] = "implementation_failure"
            performance_feedback.setdefault("summary", {})["does_not_consume_same_direction_attempt"] = True
            performance_feedback["direction_scorecard"] = None
            performance_feedback["direction_scorecard_path"] = None
            repair_dispatch = _write_s2_5_only_repair_dispatch(
                project_root,
                registry,
                performance_feedback,
                reason=reason,
                route_count=route_count,
            )
        else:
            scorecard = _update_c2c_direction_scorecard(project_root, registry, performance_feedback)
            performance_feedback["direction_scorecard"] = scorecard.get("current_direction")
            performance_feedback["direction_scorecard_path"] = "plan/direction_scorecard.json"
            performance_feedback["shared_method_memory"] = append_shared_c2c_method_failure(
                config or {},
                project_root=project_root,
                performance_feedback=performance_feedback,
                direction_scorecard=scorecard,
                route=repair_route,
            )
            repair_dispatch = {}
        write_json(project_root / "plan" / "performance_feedback.json", performance_feedback)
        trace_path = project_root / "meta" / "iteration_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "timestamp": now_utc(),
            "from_stage": "S3_experiment",
            "to_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "reason": reason,
            "result_status": result.get("status"),
            "repair_route": repair_route,
            "failure_class": failure_class,
            "route_count": route_count,
            "repair_route_key": route_key or None,
            "same_direction_failure_count": failure_count,
            "performance_feedback": performance_feedback.get("summary", {}),
        }
        if repair_dispatch:
            trace["repair_lane"] = "s2_5_only_implementation_repair"
            trace["s2_5_repair_dispatch_path"] = "plan/s2_5_repair_dispatch.json"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        invalidate_from(
            registry,
            "S2_plan",
            invalidated_by="s2_5_only_implementation_repair" if repair_dispatch else "s3_repairable_proxy",
        )
        for stage_key in ["S2_plan", "S3_experiment"]:
            registry["stages"][stage_key]["judge_retries"] = 0
        return {
            "status": "routed",
            "reason": reason,
            "next_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "route_count": route_count,
            "failure_class": failure_class,
            "same_direction_failure_count": failure_count,
            "repair_route_key": route_key or None,
            "performance_feedback_path": "plan/performance_feedback.json",
            "performance_feedback": performance_feedback.get("summary", {}),
            **(
                {
                    "repair_lane": "s2_5_only_implementation_repair",
                    "skips_s2_planner": True,
                    "s2_5_repair_dispatch_path": "plan/s2_5_repair_dispatch.json",
                    "s2_5_repair_dispatch": {
                        "selected_candidate_id": repair_dispatch.get("selected_candidate_id"),
                        "variant_fingerprint": repair_dispatch.get("variant_fingerprint"),
                        "implementation_failure_signals": repair_dispatch.get("implementation_failure_signals") or [],
                    },
                }
                if repair_dispatch
                else {}
            ),
        }

    @staticmethod
    def _route_s3_repairable_proxy_to_s1(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        iteration_key = str(registry.get("iteration") or 1)
        repair_routes = registry.get("repair_routes") or {}
        route_count = int(repair_routes.get(iteration_key) or 0)
        failure_budget = _repairable_proxy_failure_budget(feedback_cfg)
        failure_count = max(route_count + 1, failure_budget)
        performance_feedback = _s3_proxy_performance_feedback(
            project_root,
            registry,
            result,
            reason,
            route_count=route_count,
            failure_count=failure_count,
            max_failures=failure_budget,
            route="repairable_proxy_risk",
        )
        scorecard = _update_c2c_direction_scorecard(project_root, registry, performance_feedback)
        performance_feedback["direction_scorecard"] = scorecard.get("current_direction")
        performance_feedback["direction_scorecard_path"] = "plan/direction_scorecard.json"
        performance_feedback["shared_method_memory"] = append_shared_c2c_method_failure(
            config or {},
            project_root=project_root,
            performance_feedback=performance_feedback,
            direction_scorecard=scorecard,
            route="repairable_proxy_risk",
        )
        write_json(project_root / "plan" / "performance_feedback.json", performance_feedback)
        routed = Orchestrator._route_s3_failure_to_s1(project_root, registry, result, reason, config=config)
        routed["next_stage"] = "S1_literature" if routed.get("status") == "routed" else routed.get("next_stage")
        routed["route_count"] = route_count
        routed["same_direction_failure_count"] = failure_count
        routed["same_direction_failure_budget"] = failure_budget
        routed["performance_feedback_path"] = "plan/performance_feedback.json"
        routed["performance_feedback"] = performance_feedback.get("summary", {})
        return routed

    @staticmethod
    def _route_s3_proxy_rejected_to_s2(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        iteration_key = str(registry.get("iteration") or 1)
        proxy_rejected_routes = registry.setdefault("proxy_rejected_routes", {})
        proxy_rejected_routes[iteration_key] = int(proxy_rejected_routes.get(iteration_key) or 0) + 1
        max_failures = _same_direction_proxy_failure_budget(feedback_cfg)
        failure_count = _same_direction_proxy_failure_count(registry, iteration_key)
        performance_feedback = _s3_proxy_performance_feedback(
            project_root,
            registry,
            result,
            reason,
            route_count=proxy_rejected_routes[iteration_key],
            failure_count=failure_count,
            max_failures=max_failures,
        )
        scorecard = _update_c2c_direction_scorecard(project_root, registry, performance_feedback)
        performance_feedback["direction_scorecard"] = scorecard.get("current_direction")
        performance_feedback["direction_scorecard_path"] = "plan/direction_scorecard.json"
        performance_feedback["shared_method_memory"] = append_shared_c2c_method_failure(
            config or {},
            project_root=project_root,
            performance_feedback=performance_feedback,
            direction_scorecard=scorecard,
            route="proxy_rejected_same_direction",
        )
        write_json(project_root / "plan" / "performance_feedback.json", performance_feedback)
        trace_path = project_root / "meta" / "iteration_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "timestamp": now_utc(),
            "from_stage": "S3_experiment",
            "to_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "reason": reason,
            "result_status": result.get("status"),
            "repair_route": "proxy_rejected_same_direction",
            "route_count": proxy_rejected_routes[iteration_key],
            "same_direction_failure_count": failure_count,
            "same_direction_failure_budget": max_failures,
            "performance_feedback": performance_feedback.get("summary", {}),
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        invalidate_from(registry, "S2_plan", invalidated_by="s3_proxy_performance_feedback")
        for stage_key in ["S2_plan", "S3_experiment"]:
            registry["stages"][stage_key]["judge_retries"] = 0
        return {
            "status": "routed",
            "reason": reason,
            "next_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "route_count": proxy_rejected_routes[iteration_key],
            "same_direction_failure_count": failure_count,
            "same_direction_failure_budget": max_failures,
            "performance_feedback_path": "plan/performance_feedback.json",
        }

    @staticmethod
    def _route_s3_proxy_rejected_to_s1(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        iteration_key = str(registry.get("iteration") or 1)
        route_count = int((registry.get("proxy_rejected_routes") or {}).get(iteration_key) or 0)
        max_failures = _same_direction_proxy_failure_budget(feedback_cfg)
        failure_count = route_count + 1
        performance_feedback = _s3_proxy_performance_feedback(
            project_root,
            registry,
            result,
            reason,
            route_count=route_count,
            failure_count=failure_count,
            max_failures=max_failures,
        )
        scorecard = _update_c2c_direction_scorecard(project_root, registry, performance_feedback)
        performance_feedback["direction_scorecard"] = scorecard.get("current_direction")
        performance_feedback["direction_scorecard_path"] = "plan/direction_scorecard.json"
        performance_feedback["shared_method_memory"] = append_shared_c2c_method_failure(
            config or {},
            project_root=project_root,
            performance_feedback=performance_feedback,
            direction_scorecard=scorecard,
            route="proxy_rejected_same_direction",
        )
        write_json(project_root / "plan" / "performance_feedback.json", performance_feedback)
        routed = Orchestrator._route_s3_failure_to_s1(project_root, registry, result, reason, config=config)
        routed["performance_feedback_path"] = "plan/performance_feedback.json"
        routed["same_direction_failure_count"] = failure_count
        routed["same_direction_failure_budget"] = max_failures
        return routed

    @staticmethod
    def _route_s3_failure_to_s1(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_record = Orchestrator._append_s3_full_failure_shared_memory(project_root, registry, result, reason, config=config or {})
        max_iterations = int(registry.get("max_iterations") or 1)
        if int(registry.get("iteration") or 1) >= max_iterations:
            block_stage(registry, "S3_experiment", f"Reached max_iterations={max_iterations} after S3 failure feedback: {reason}")
            return {"status": "blocked", "reason": registry["blocked_reason"], "shared_method_memory": memory_record}
        early_stop_reason = Orchestrator._s3_failure_early_stop_reason(project_root, registry, reason, config=config or {})
        if early_stop_reason:
            block_stage(registry, "S3_experiment", early_stop_reason)
            return {"status": "blocked", "reason": registry["blocked_reason"], "shared_method_memory": memory_record}
        trace_path = project_root / "meta" / "iteration_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "timestamp": now_utc(),
            "from_stage": "S3_experiment",
            "to_stage": "S1_literature",
            "iteration": registry.get("iteration"),
            "reason": reason,
            "result_status": result.get("status"),
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        registry["iteration"] = int(registry.get("iteration") or 1) + 1
        invalidate_from(registry, "S1_literature", invalidated_by="s3_failure_feedback")
        for stage_key in ["S1_literature", "S2_plan", "S3_experiment"]:
            registry["stages"][stage_key]["judge_retries"] = 0
        return {"status": "routed", "reason": reason, "next_iteration": registry["iteration"], "shared_method_memory": memory_record}

    @staticmethod
    def _append_s3_full_failure_shared_memory(
        project_root: Path,
        registry: dict[str, Any],
        result: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        performance_feedback = _s3_full_performance_feedback(project_root, registry, result, reason)
        summary = performance_feedback.get("summary") if isinstance(performance_feedback.get("summary"), dict) else {}
        if summary.get("failure_class") == "implementation_failure" or summary.get("does_not_consume_same_direction_attempt"):
            return {"status": "skipped", "reason": "implementation_failure_is_not_method_memory"}
        direction_scorecard = read_json(project_root / "plan" / "direction_scorecard.json", default={}) or {}
        return append_shared_c2c_method_failure(
            config or {},
            project_root=project_root,
            performance_feedback=performance_feedback,
            direction_scorecard=direction_scorecard if isinstance(direction_scorecard, dict) else {},
            route="full_s3_failure",
            source_paths=[
                "experiment/results/main_results.json",
                "experiment/results/failure_feedback.json",
                "experiment/results/proxy_calibration.json",
                "plan/direction_scorecard.json",
            ],
        )

    @staticmethod
    def _s3_failure_early_stop_reason(
        project_root: Path,
        registry: dict[str, Any],
        reason: str,
        *,
        config: dict[str, Any],
    ) -> str | None:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        early_cfg = feedback_cfg.get("early_stop", {})
        if not early_cfg.get("enabled", False):
            return None
        history = read_json(
            project_root / "experiment" / "results" / "c2c_iteration_history.json",
            default={},
        )
        iterations = history.get("iterations") or []
        if not isinstance(iterations, list) or not iterations:
            return None
        current_iteration = int(registry.get("iteration") or 1)
        min_iterations = int(early_cfg.get("min_iterations", 3) or 3)
        if current_iteration < min_iterations:
            return None
        patience = int(early_cfg.get("patience", min_iterations) or min_iterations)
        consecutive_not_viable = int(history.get("consecutive_not_viable") or 0)
        if consecutive_not_viable < patience:
            return None
        best_delta = history.get("best_delta_so_far")
        if best_delta is None:
            return None
        try:
            best_delta_value = float(best_delta)
            stop_if_below = float(early_cfg.get("stop_if_best_delta_below", 0.0))
        except (TypeError, ValueError):
            return None
        if best_delta_value >= stop_if_below:
            return None
        return (
            "Early-stopped S3 failure feedback after "
            f"{consecutive_not_viable} consecutive non-viable C2C iteration(s): "
            f"best_delta_so_far={best_delta_value:.4f} < {stop_if_below:.4f}; "
            f"latest reason: {reason}"
        )

    @staticmethod
    def _maybe_request_approval(
        hitl: HITLManager,
        stage_key: str,
        stage_result: dict[str, Any],
        project_root: Path,
        registry: dict[str, Any],
    ) -> HITLDecision | None:
        """Request human approval if HITL is configured for this stage. Returns None if HITL is disabled."""
        if not hitl.enabled:
            return None
        approval_stages = hitl.config.get("hitl", {}).get("approval_stages", [])
        if stage_key not in approval_stages:
            return None

        from .constants import STAGE_LABELS

        summary_lines = [
            f"项目: **{registry['research_topic']}**",
            f"迭代: 第 {registry['iteration']} 轮",
        ]
        stage_label = STAGE_LABELS.get(stage_key, stage_key)
        # Build stage-specific summary
        if stage_key == "S1_literature":
            ideas_path = project_root / "literature" / "ideas.json"
            if ideas_path.exists():
                ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
                for idea in ideas[:3]:
                    selected = " [选中]" if idea.get("selected") else ""
                    summary_lines.append(
                        f"• **{idea['title']}**{selected}: novelty={idea['novelty_score']}, feasibility={idea['feasibility_score']}"
                    )
        elif stage_key == "S2_plan":
            plan_path = project_root / "plan" / "plan.yaml"
            if plan_path.exists():
                plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
                summary_lines.append(f"• 假设: {len(plan.get('hypotheses', []))} 个")
                summary_lines.append(f"• 基线: {', '.join(_names_for_summary(plan.get('baselines', [])[:5]))}")
                summary_lines.append(f"• 数据集: {', '.join(_names_for_summary(plan.get('datasets', [])[:5]))}")
        elif stage_key == "S4_writing":
            compile_path = project_root / "paper" / "compile_report.json"
            if compile_path.exists():
                summary_lines.append("• LaTeX 编译: 通过")
            audit_path = project_root / "paper" / "claim_audit.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                summary_lines.append(f"• Claim 审计通过率: {audit.get('pass_rate', 0):.0%}")

        summary = "\n".join(summary_lines)
        hitl_config = hitl.config.get("hitl", {})
        timeout_min = int(hitl_config.get("timeout_minutes", 60))
        blocking = bool(hitl_config.get("blocking", True))

        return hitl.request_approval(
            stage_key,
            stage_label,
            summary,
            artifacts=stage_result.get("artifacts", [])[:10],
            blocking=blocking,
            timeout_minutes=timeout_min,
        )

    def _execute_targeted_revisions(self, project_root: Path, context: AgentContext, registry: dict[str, Any]) -> None:
        dispatch_path = project_root / "review" / "revision_dispatch.yaml"
        dispatch = yaml.safe_load(dispatch_path.read_text(encoding="utf-8"))
        revisions_by_id = {item["id"]: item for item in dispatch.get("revisions", [])}
        for step in dispatch.get("execution_order", []):
            revision_items = [revisions_by_id[revision_id] for revision_id in step.get("revisions", []) if revision_id in revisions_by_id]
            for agent_name in step.get("agents", []):
                if agent_name == "experiment-agent":
                    ExperimentAgent(context).run(revisions=revision_items)
                    invalidate_from(registry, "S4_writing", invalidated_by="review-revision")
                elif agent_name == "writing-agent":
                    WritingAgent(context).run()
        registry["iteration"] += 1
        registry["current_stage"] = "S5_review"
        registry["status"] = "running"

    @staticmethod
    def _write_gate_report(artifacts: ArtifactManager, stage_key: str, gate_report: Any) -> dict[str, Any]:
        payload = gate_report.to_dict()
        return artifacts.write_json(
            stage_key,
            "gate_report.json",
            payload,
            artifact_type="stage_gate_report",
            summary=f"{stage_key} executable gate report: {payload['status']}",
            source_paths=payload.get("artifacts_checked", []),
            created_by="stage-gate-validator",
            validator=payload.get("validator"),
        )

    @staticmethod
    def _context(project_root: Path, config: dict[str, Any]) -> AgentContext:
        return AgentContext(
            project_root=project_root,
            config=config,
            artifacts=ArtifactManager(project_root),
            llm=ModelClient(config, project_root=project_root),
        )

    @staticmethod
    def _log_session(project_root: Path, *, action: str, details: dict[str, Any]) -> None:
        payload = {"timestamp": now_utc(), "action": action, "details": details}
        path = project_root / "meta" / "session_log.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _project_root(self, project_id: str) -> Path:
        config = load_root_config()
        root = (Path(config["project"]["workspace_root"]) / project_id).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project {project_id} not found")
        return root


def _names_for_summary(items: list[Any]) -> list[str]:
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("id") or item))
        else:
            names.append(str(item))
    return names


def _same_direction_proxy_failure_budget(feedback_cfg: dict[str, Any]) -> int:
    raw = feedback_cfg.get("max_same_direction_proxy_failures")
    if raw is None:
        raw = feedback_cfg.get("max_same_direction_proxy_iterations")
    if raw is None:
        raw = feedback_cfg.get("max_proxy_rejected_routes_per_iteration")
    try:
        budget = int(raw if raw is not None else 1)
    except (TypeError, ValueError):
        budget = 1
    return max(1, budget)


def _repairable_proxy_failure_budget(feedback_cfg: dict[str, Any]) -> int:
    raw = feedback_cfg.get("max_same_direction_proxy_failures")
    if raw is None:
        raw = feedback_cfg.get("max_same_direction_proxy_iterations")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1
    try:
        max_routes = int(feedback_cfg.get("max_proxy_repair_routes_per_iteration", 1) or 1)
    except (TypeError, ValueError):
        max_routes = 1
    return max(1, max_routes + 1)


def _implementation_repair_route_budget(feedback_cfg: dict[str, Any]) -> int:
    try:
        return max(1, int(feedback_cfg.get("max_implementation_repair_routes_per_iteration", 8) or 8))
    except (TypeError, ValueError):
        return 8


def _implementation_repair_route_key(
    *,
    iteration_key: str,
    candidate_id: str = "",
    variant_fingerprint: str = "",
) -> str:
    candidate = candidate_id.strip() or "unknown_candidate"
    variant = variant_fingerprint.strip() or "unknown_variant"
    return f"{iteration_key}:{candidate}:{variant}"


def _s2_5_patch_manifest_repair_route_key(
    project_root: Path,
    registry: dict[str, Any],
    patch_manifest: dict[str, Any],
) -> str:
    iteration_key = str(registry.get("iteration") or 1)
    candidates = [
        _s2_5_patch_manifest_candidate_feedback(project_root, item)
        for item in patch_manifest.get("candidates") or []
        if isinstance(item, dict)
    ]
    selected_candidate = _select_s2_5_repair_candidate(project_root, [], candidates)
    return _implementation_repair_route_key(
        iteration_key=iteration_key,
        candidate_id=_repair_candidate_id(selected_candidate),
        variant_fingerprint=_repair_candidate_variant_fingerprint(selected_candidate),
    )


def _s2_5_repair_dispatch_matches_route_key(
    dispatch: dict[str, Any],
    route_key: str,
    registry: dict[str, Any],
) -> bool:
    if not isinstance(dispatch, dict) or not dispatch:
        return False
    if dispatch.get("status") not in {"active", None}:
        return False
    if str(dispatch.get("mode") or dispatch.get("repair_lane") or "") != "s2_5_only_implementation_repair":
        return False
    iteration_key = str(registry.get("iteration") or 1)
    dispatch_key = _implementation_repair_route_key(
        iteration_key=iteration_key,
        candidate_id=str(dispatch.get("selected_candidate_id") or ""),
        variant_fingerprint=str(dispatch.get("variant_fingerprint") or ""),
    )
    return dispatch_key == route_key


def _s3_implementation_repair_route_key(project_root: Path, registry: dict[str, Any]) -> str:
    iteration_key = str(registry.get("iteration") or 1)
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    implementation_failed = next((item for item in candidates if _candidate_is_implementation_failure(item)), None)
    candidate = implementation_failed or (candidates[0] if candidates else {})
    return _implementation_repair_route_key(
        iteration_key=iteration_key,
        candidate_id=_repair_candidate_id(candidate),
        variant_fingerprint=_repair_candidate_variant_fingerprint(candidate),
    )


def _same_direction_proxy_failure_count(registry: dict[str, Any], iteration_key: str) -> int:
    proxy_rejected_routes = registry.get("proxy_rejected_routes") or {}
    return int(proxy_rejected_routes.get(iteration_key) or 0)


def _s3_feedback_failure_class(project_root: Path) -> str:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [candidate for candidate in payload.get("candidate_results") or [] if isinstance(candidate, dict)]
    if not candidates:
        return "unknown"
    return _classify_s3_failure_from_candidates(candidates)


def _classify_s3_failure_from_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "unknown"
    if any(_candidate_has_resource_retry(candidate) for candidate in candidates):
        return "resource_retry"
    if any(_candidate_has_full_s3_metrics(candidate) for candidate in candidates):
        return "method_failure"
    if any(_candidate_is_implementation_failure(candidate) for candidate in candidates):
        return "implementation_failure"
    return "method_failure"


def _candidate_has_full_s3_metrics(candidate: dict[str, Any]) -> bool:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    if metrics.get("mean") is not None:
        return True
    if isinstance(metrics.get("datasets"), dict) and metrics.get("datasets"):
        return True
    if candidate.get("delta_vs_baseline") is not None:
        return True
    command_status = str(candidate.get("command_status") or "")
    if command_status in {"ok", "partial"} and isinstance(metrics, dict) and metrics:
        return True
    return False


def _candidate_has_proxy_metrics(candidate: dict[str, Any]) -> bool:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    metrics = proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {}
    if metrics.get("mean") is not None:
        return True
    return bool(isinstance(metrics.get("datasets"), dict) and metrics.get("datasets"))


def _s3_result_has_resource_retry(project_root: Path) -> bool:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [candidate for candidate in payload.get("candidate_results") or [] if isinstance(candidate, dict)]
    return any(_candidate_has_resource_retry(candidate) for candidate in candidates)


def _candidate_has_resource_retry(candidate: dict[str, Any]) -> bool:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    if proxy.get("resource_retry") is True:
        return True
    if proxy.get("status") == "resource_retry":
        return True
    if proxy.get("failure_category") in {"s3_proxy_resource_oom", "s3_proxy_gpu_resource_retry"}:
        return True
    command_failure = proxy.get("command_failure") if isinstance(proxy.get("command_failure"), dict) else {}
    if command_failure.get("category") == "resource_oom":
        return True
    return False


def _candidate_is_implementation_failure(candidate: dict[str, Any]) -> bool:
    decision = str(candidate.get("decision") or "")
    command_status = str(candidate.get("command_status") or "")
    if _candidate_has_resource_retry(candidate):
        return False
    if decision in {"proxy_rejected", "not_viable"} and _candidate_has_proxy_metrics(candidate):
        return False
    if decision in {"patch_rejected", "failed_no_metrics"}:
        return True
    if command_status in {"patch_rejected", "failed"}:
        return True
    attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
    primary = str(attribution.get("primary_failure") or "")
    if primary in {
        "proxy_eval_output_health_failure",
        "proxy_activation_smoke_no_effect",
        "ablation_no_effect",
        "failed_no_metrics",
        "patch_rejected",
    }:
        return True
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    if proxy.get("status") == "baseline_blocked":
        return False
    if proxy.get("proxy_eval_health_failure"):
        return True
    if proxy.get("command_failure"):
        return True
    activation_smoke = candidate.get("activation_smoke") if isinstance(candidate.get("activation_smoke"), dict) else proxy.get("activation_smoke")
    if isinstance(activation_smoke, dict) and activation_smoke.get("status") == "failed":
        trace = activation_smoke.get("mechanism_trace") if isinstance(activation_smoke.get("mechanism_trace"), dict) else {}
        return trace.get("status") != "wired"
    contract = proxy.get("proxy_effect_repair_contract") if isinstance(proxy.get("proxy_effect_repair_contract"), dict) else {}
    if contract.get("command_failure") or contract.get("proxy_eval_health_failure"):
        return True
    if str(contract.get("source") or "") in {"static_proxy", "proxy_command", "proxy_activation_smoke"}:
        return True
    patch_result = candidate.get("patch_result") if isinstance(candidate.get("patch_result"), dict) else {}
    if patch_result.get("status") == "rejected":
        return True
    if _candidate_implementation_failure_signals(candidate):
        return True
    runtime = _candidate_runtime_validation_status(candidate)
    if runtime.get("validation") == "failed" or runtime.get("runtime_smoke") == "failed":
        return True
    patch_risk = attribution.get("patch_risk") if isinstance(attribution.get("patch_risk"), dict) else {}
    if not patch_risk and isinstance(proxy.get("patch_risk"), dict):
        patch_risk = proxy.get("patch_risk") or {}
    labels = {str(label) for label in patch_risk.get("risk_labels") or []}
    if any("evaluation" in label or "evaluator" in label or "test_change" in label for label in labels):
        return True
    return False


def _candidate_implementation_failure_signals(candidate: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    for check in _candidate_validation_checks(candidate):
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        failure_category = str(check.get("failure_category") or "")
        stderr = str(check.get("stderr") or "")
        if name == "runtime_smoke:mechanism_activation_forward_probe":
            signals.append("activation_forward_probe_failed")
            probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
            probe_type = str(probe.get("probe_type") or "")
            fallback_reason = str(probe.get("fallback_reason") or "")
            failures = {str(item) for item in probe.get("failures") or [] if item}
            if probe_type == "repo_small_batch_forward_failed_static_trace":
                signals.append("repo_small_batch_forward_failed_static_trace")
            if fallback_reason in {"torch_import_failed", "projector_import_failed", "small_batch_forward_failed"}:
                signals.append(fallback_reason)
            for marker in {
                "enabled_disabled_forward_tensors_identical",
                "enabled_disabled_wrapper_cache_identical",
                "torch_import_failed",
                "projector_import_failed",
                "small_batch_forward_failed",
                "forward_probe_command_failed",
            }:
                if marker in failures or marker in stderr:
                    signals.append(marker)
            diagnostics = check.get("forward_probe_diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = ((probe.get("diagnostics") if isinstance(probe.get("diagnostics"), dict) else {}) or {})
            if isinstance(diagnostics, dict):
                if diagnostics.get("projector_output_identical") is True:
                    signals.append("projector_output_identical")
                if diagnostics.get("wrapper_cache_identical") is True:
                    signals.append("wrapper_cache_identical")
                if diagnostics.get("switch_seen_by_forward") is False:
                    signals.append("forward_missing_switch_read")
        if failure_category in {
            "mechanism_activation_forward_probe_failed",
            "mechanism_activation_wiring_failed",
            "missing_ablation_switch",
            "mechanism_activation_materialization_failed",
        }:
            signals.append(failure_category)
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    command_failure = proxy.get("command_failure") if isinstance(proxy.get("command_failure"), dict) else {}
    command_failure_category = str(command_failure.get("category") or "")
    if command_failure_category in {"distributed_child_failed"}:
        signals.append(command_failure_category)
    activation_smoke = candidate.get("activation_smoke") if isinstance(candidate.get("activation_smoke"), dict) else proxy.get("activation_smoke")
    if isinstance(activation_smoke, dict) and activation_smoke.get("status") == "failed":
        trace = activation_smoke.get("mechanism_trace") if isinstance(activation_smoke.get("mechanism_trace"), dict) else {}
        if trace.get("status") != "wired":
            signals.append("proxy_activation_smoke_no_effect")
    return sorted(set(signals))


def _candidate_validation_checks(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    patch_result = candidate.get("patch_result") if isinstance(candidate.get("patch_result"), dict) else {}
    validation = patch_result.get("validation") if isinstance(patch_result.get("validation"), dict) else {}
    checks = validation.get("checks")
    return [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []


def _patch_manifest_has_s2_5_implementation_failure(patch_manifest: dict[str, Any]) -> bool:
    if not isinstance(patch_manifest, dict) or patch_manifest.get("status") != "no_valid_patch":
        return False
    if patch_manifest.get("valid_patch_count"):
        return False
    for candidate in patch_manifest.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        status = str(candidate.get("status") or "")
        if status not in {"validation_failed", "codex_failed"}:
            continue
        if _patch_entry_has_resource_retry(candidate):
            continue
        if _candidate_is_implementation_failure(_candidate_from_patch_manifest_entry(candidate)):
            return True
    return False


def _patch_manifest_has_resource_retry(patch_manifest: dict[str, Any]) -> bool:
    if not isinstance(patch_manifest, dict):
        return False
    if patch_manifest.get("resource_retry") is True or patch_manifest.get("failure_category") == "runtime_smoke_resource_retry":
        return True
    entries = patch_manifest.get("patches") or patch_manifest.get("candidates") or []
    return any(_patch_entry_has_resource_retry(entry) for entry in entries if isinstance(entry, dict))


def _patch_entry_has_resource_retry(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("resource_retry") is True or entry.get("failure_category") == "runtime_smoke_resource_retry":
        return True
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    if validation.get("resource_retry") is True or validation.get("failure_category") == "runtime_smoke_resource_retry":
        return True
    for check in validation.get("checks") or []:
        if isinstance(check, dict) and (check.get("resource_retry") is True or check.get("failure_category") == "runtime_smoke_resource_retry"):
            return True
    return False


def _s2_5_patch_manifest_performance_feedback(
    project_root: Path,
    registry: dict[str, Any],
    patch_manifest: dict[str, Any],
    *,
    reason: str,
    route_count: int,
    max_routes: int,
    gate_report: Any,
) -> dict[str, Any]:
    manifest_candidates = [
        item for item in patch_manifest.get("candidates") or [] if isinstance(item, dict)
    ]
    candidate_results = [
        _s2_5_patch_manifest_candidate_feedback(project_root, item)
        for item in manifest_candidates
    ]
    failed = [item for item in candidate_results if _candidate_is_implementation_failure(item)]
    gate_payload = gate_report.to_dict() if hasattr(gate_report, "to_dict") else {}
    summary = {
        "schema_version": "c2c_s2_5_validation_failure_feedback_v1",
        "created_at": now_utc(),
        "source_stage": "S2_plan",
        "failure_class": "implementation_failure",
        "does_not_consume_same_direction_attempt": True,
        "reason": reason,
        "route_count": route_count,
        "max_implementation_repair_routes": max_routes,
        "same_direction_failure_count": 0,
        "patch_manifest_status": patch_manifest.get("status"),
        "selected_candidate_id": patch_manifest.get("selected_candidate_id"),
        "failed_candidate_count": len(failed),
        "s2_action_policy": {
            "matched_rule": "implementation_failure",
            "route": "s2_5_only_implementation_repair",
            "skips_s2_planner": True,
            "same_candidate_required": True,
            "does_not_consume_same_direction_attempt": True,
        },
        "repair_vs_variant_signals": sorted(
            {
                signal
                for candidate in failed
                for signal in (candidate.get("implementation_failure_signals") or [])
                if signal
            }
        ),
        "gate_checks": [
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "message": check.get("message"),
            }
            for check in gate_payload.get("checks", [])
            if isinstance(check, dict) and check.get("status") not in {"PASS", "pass", "ok"}
        ],
    }
    return {
        "schema_version": "c2c_performance_feedback_v1",
        "created_at": now_utc(),
        "iteration": registry.get("iteration"),
        "route": "s2_5_validation_failure",
        "summary": summary,
        "candidate_results": candidate_results,
        "direction_scorecard": None,
        "direction_scorecard_path": None,
    }


def _s2_5_patch_manifest_candidate_feedback(project_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    candidate = _candidate_from_patch_manifest_entry(entry)
    validation_path = str(entry.get("validation") or "")
    validation = {}
    if validation_path:
        validation = read_json(project_root / validation_path, default={}) or {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), list) else []
    patch_result = candidate.setdefault("patch_result", {})
    patch_result["validation"] = validation if isinstance(validation, dict) else {}
    patch_result["changed_files"] = list(entry.get("changed_files") or [])
    patch_result["status"] = entry.get("status")
    runtime = _candidate_runtime_validation_status(candidate)
    return {
        **candidate,
        "decision": "patch_rejected",
        "command_status": "patch_rejected",
        "reason": entry.get("reason") or validation.get("status") or entry.get("status"),
        "patch_status": entry.get("status"),
        "runtime_validation": runtime,
        "implementation_failure_signals": _candidate_implementation_failure_signals(candidate),
        "failed_checks": [
            {
                "name": check.get("name"),
                "failure_category": check.get("failure_category"),
                "returncode": check.get("returncode"),
                "repair_hint": check.get("repair_hint"),
            }
            for check in checks
            if isinstance(check, dict) and check.get("returncode") not in (0, None)
        ],
    }


def _candidate_from_patch_manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    validation = entry.get("validation")
    if isinstance(validation, dict):
        validation_payload = validation
    else:
        validation_payload = {}
    patch_result = {
        "status": entry.get("status"),
        "reason": entry.get("reason"),
        "changed_files": list(entry.get("changed_files") or []),
        "validation": validation_payload,
    }
    return {
        "id": entry.get("candidate_id"),
        "candidate_id": entry.get("candidate_id"),
        "title": entry.get("title") or entry.get("candidate_id"),
        "variant_fingerprint": entry.get("variant_fingerprint"),
        "s2_variant": entry.get("s2_variant") if isinstance(entry.get("s2_variant"), dict) else {},
        "decision": "patch_rejected",
        "command_status": "patch_rejected",
        "patch_result": patch_result,
        "failure_attribution": {
            "primary_failure": "s2_5_validation_failed",
            "patch_risk": {
                "risk_labels": ((entry.get("risk_check") or {}).get("risk_labels") if isinstance(entry.get("risk_check"), dict) else []) or [],
                "risk_files": ((entry.get("risk_check") or {}).get("risk_files") if isinstance(entry.get("risk_check"), dict) else []) or [],
            },
        },
    }


def _write_s2_5_only_repair_dispatch(
    project_root: Path,
    registry: dict[str, Any],
    performance_feedback: dict[str, Any],
    *,
    reason: str,
    route_count: int,
) -> dict[str, Any]:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    feedback_candidates = [
        item for item in performance_feedback.get("candidate_results") or [] if isinstance(item, dict)
    ]
    selected_candidate = _select_s2_5_repair_candidate(project_root, candidates, feedback_candidates)
    patch_manifest_path = project_root / "plan" / "code_patches" / "patch_manifest.json"
    patch_manifest = read_json(patch_manifest_path, default={}) if patch_manifest_path.exists() else {}
    selected_candidate_id = _repair_candidate_id(selected_candidate) or str(patch_manifest.get("selected_candidate_id") or "")
    variant_fingerprint = _repair_candidate_variant_fingerprint(selected_candidate)
    diagnostics = _candidate_activation_forward_probe_diagnostics(selected_candidate)
    tensor_checks = _candidate_forward_tensor_checks(selected_candidate, diagnostics)
    changed_files = _candidate_changed_files(selected_candidate)
    dispatch = {
        "schema_version": "c2c_s2_5_only_repair_dispatch_v1",
        "created_at": now_utc(),
        "mode": "s2_5_only_implementation_repair",
        "repair_lane": "s2_5_only_implementation_repair",
        "status": "active",
        "reason": reason,
        "iteration": registry.get("iteration"),
        "route_count": route_count,
        "failure_class": "implementation_failure",
        "does_not_consume_same_direction_attempt": True,
        "skips_s2_planner": True,
        "selected_candidate_id": selected_candidate_id or None,
        "variant_fingerprint": variant_fingerprint or None,
        "same_candidate_required": True,
        "same_variant_fingerprint_required": bool(variant_fingerprint),
        "reuse_persistent_codex_session": True,
        "do_not_replan_method": True,
        "repair_until": "patch_eligible_for_s3_or_implementation_blocked",
        "performance_feedback_path": "plan/performance_feedback.json",
        "patch_manifest_path": "plan/code_patches/patch_manifest.json" if patch_manifest else None,
        "main_results_path": "experiment/results/main_results.json",
        "changed_files": changed_files,
        "implementation_failure_signals": _candidate_implementation_failure_signals(selected_candidate),
        "activation_forward_probe_diagnostics": diagnostics,
        "tensor_checks": tensor_checks,
        "runtime_validation": _candidate_runtime_validation_status(selected_candidate),
        "patch_manifest": _compact_s2_5_repair_patch_manifest(patch_manifest),
        "repair_policy": {
            "only_fix_implementation": True,
            "same_candidate_required": True,
            "same_variant_fingerprint_required": bool(variant_fingerprint),
            "reuse_persistent_codex_session": True,
            "forbidden_actions": [
                "rerun_s2_planner",
                "change_s1_direction",
                "switch_candidate",
                "switch_variant_fingerprint",
                "weaken_validation_or_proxy_thresholds",
                "edit_evaluator_or_metric_code_to_bypass_failure",
            ],
        },
    }
    write_json(project_root / "plan" / "s2_5_repair_dispatch.json", dispatch)
    registry["s2_5_repair_dispatch"] = {
        "active": True,
        "path": "plan/s2_5_repair_dispatch.json",
        "mode": "s2_5_only_implementation_repair",
        "selected_candidate_id": selected_candidate_id or None,
        "variant_fingerprint": variant_fingerprint or None,
        "iteration": registry.get("iteration"),
    }
    return dispatch


def _select_s2_5_repair_candidate(
    project_root: Path,
    candidates: list[dict[str, Any]],
    feedback_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = read_json(project_root / "plan" / "code_patches" / "patch_manifest.json", default={}) or {}
    selected_id = str(manifest.get("selected_candidate_id") or "").strip()
    if selected_id:
        matched = next((item for item in candidates if _repair_candidate_id(item) == selected_id), None)
        if matched:
            return matched
    feedback_id = str((feedback_candidates[0] or {}).get("id") or "") if feedback_candidates else ""
    if feedback_id:
        matched = next((item for item in candidates if _repair_candidate_id(item) == feedback_id), None)
        if matched:
            return matched
        feedback_matched = next((item for item in feedback_candidates if _repair_candidate_id(item) == feedback_id), None)
        if feedback_matched:
            return feedback_matched
    implementation_failed = next((item for item in candidates if _candidate_is_implementation_failure(item)), None)
    if implementation_failed:
        return implementation_failed
    feedback_implementation_failed = next((item for item in feedback_candidates if _candidate_is_implementation_failure(item)), None)
    if feedback_implementation_failed:
        return feedback_implementation_failed
    return candidates[0] if candidates else {}


def _repair_candidate_id(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("id") or candidate.get("candidate_id") or "").strip()


def _repair_candidate_variant_fingerprint(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return ""
    variant = candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {}
    return str(candidate.get("variant_fingerprint") or variant.get("variant_fingerprint") or "").strip()


def _candidate_changed_files(candidate: dict[str, Any]) -> list[str]:
    patch_result = candidate.get("patch_result") if isinstance(candidate.get("patch_result"), dict) else {}
    return [str(item) for item in patch_result.get("changed_files") or [] if item][:20]


def _candidate_activation_forward_probe_diagnostics(candidate: dict[str, Any]) -> dict[str, Any]:
    for check in _candidate_validation_checks(candidate):
        if str(check.get("name") or "") != "runtime_smoke:mechanism_activation_forward_probe":
            continue
        diagnostics = check.get("forward_probe_diagnostics")
        if not isinstance(diagnostics, dict):
            probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
            diagnostics = probe.get("diagnostics") if isinstance(probe.get("diagnostics"), dict) else {}
        result = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
        if probe:
            result.setdefault("probe_type", probe.get("probe_type"))
            result.setdefault("fallback_reason", probe.get("fallback_reason"))
            result.setdefault("failures", probe.get("failures") or [])
            result.setdefault("mechanism_observed", probe.get("mechanism_observed"))
            if isinstance(probe.get("tensor_checks"), dict):
                result.setdefault("tensor_checks", probe.get("tensor_checks"))
        result.setdefault("failure_category", check.get("failure_category"))
        result.setdefault("returncode", check.get("returncode"))
        return {key: value for key, value in result.items() if value not in (None, "", [], {})}
    return {}


def _candidate_forward_tensor_checks(candidate: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    tensor_checks = diagnostics.get("tensor_checks") if isinstance(diagnostics.get("tensor_checks"), dict) else {}
    if tensor_checks:
        return tensor_checks
    for check in _candidate_validation_checks(candidate):
        if str(check.get("name") or "") != "runtime_smoke:mechanism_activation_forward_probe":
            continue
        probe = check.get("probe") if isinstance(check.get("probe"), dict) else {}
        candidate_checks = probe.get("tensor_checks") if isinstance(probe.get("tensor_checks"), dict) else {}
        if candidate_checks:
            return candidate_checks
    compact: dict[str, Any] = {}
    for key in ["changed_tensors", "identical_tensors", "projector_output_identical", "wrapper_cache_identical", "switch_seen_by_forward"]:
        if key in diagnostics:
            compact[key] = diagnostics[key]
    return compact


def _compact_s2_5_repair_patch_manifest(patch_manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch_manifest, dict) or not patch_manifest:
        return {}
    selected_patch = patch_manifest.get("selected_patch") if isinstance(patch_manifest.get("selected_patch"), dict) else {}
    return {
        "status": patch_manifest.get("status"),
        "selection_policy": patch_manifest.get("selection_policy") if isinstance(patch_manifest.get("selection_policy"), dict) else {},
        "selected_candidate_id": patch_manifest.get("selected_candidate_id"),
        "valid_patch_count": patch_manifest.get("valid_patch_count"),
        "retryable": patch_manifest.get("retryable"),
        "selected_patch": {
            key: selected_patch.get(key)
            for key in [
                "candidate_id",
                "status",
                "patch_json",
                "variant_fingerprint",
                "changed_files",
                "code_worktree",
            ]
            if selected_patch.get(key) not in (None, "", [], {})
        },
    }


def _s3_proxy_performance_feedback(
    project_root: Path,
    registry: dict[str, Any],
    result: dict[str, Any],
    reason: str,
    *,
    route_count: int,
    failure_count: int,
    max_failures: int,
    route: str = "proxy_rejected_same_direction",
) -> dict[str, Any]:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    blocked = [item for item in candidates if item.get("decision") in {"proxy_rejected", "proxy_repairable", "patch_rejected", "failed_no_metrics"}]
    rejected = [item for item in blocked if item.get("decision") == "proxy_rejected"]
    repairable = [item for item in blocked if item.get("decision") == "proxy_repairable"]
    failure_class = _classify_s3_failure_from_candidates(blocked)
    feedback = read_json(project_root / "experiment" / "results" / "failure_feedback.json", default={}) or {}
    candidate_summaries = [_proxy_rejected_candidate_summary(item) for item in blocked]
    action_route = "implementation_failure" if failure_class == "implementation_failure" else route
    action = _same_direction_s2_action_recommendation(
        candidate_summaries,
        route=action_route,
        failure_count=failure_count,
        max_failures=max_failures,
    )
    mean_deltas = [
        item["proxy_mean_delta"]
        for item in candidate_summaries
        if isinstance(item.get("proxy_mean_delta"), (int, float))
    ]
    all_datasets_collapsed = bool(candidate_summaries) and all(
        item.get("all_proxy_datasets_below_baseline") for item in candidate_summaries
    )
    next_action = "repair_or_variant_same_direction" if action["action"] != "return_to_s1_new_direction" else "return_to_s1_new_direction"
    recommended_s2_action = action["action"]
    summary = {
        "route": route,
        "failure_class": failure_class,
        "does_not_consume_same_direction_attempt": failure_class == "implementation_failure",
        "iteration": registry.get("iteration"),
        "route_count": route_count,
        "same_direction_failure_count": failure_count,
        "same_direction_failure_budget": max_failures,
        "proxy_blocked_candidates": len(blocked),
        "proxy_rejected_candidates": len(rejected),
        "proxy_repairable_candidates": len(repairable),
        "proxy_mean_delta_min": min(mean_deltas) if mean_deltas else None,
        "proxy_mean_delta_max": max(mean_deltas) if mean_deltas else None,
        "all_datasets_collapsed": all_datasets_collapsed,
        "next_action": next_action,
        "recommended_s2_action": recommended_s2_action,
        "s2_action_policy": action,
        "repair_vs_variant_reason": action["reason"],
        "repair_vs_variant_signals": action["signals"],
    }
    repair_instructions = [
        "Keep the current S1 mechanism direction; do not ask S1 for a new idea during this budget.",
        "Follow summary.recommended_s2_action when choosing patch repair, mechanism repair, or a new same-direction variant.",
        "Use dragging_datasets, proxy_mean_delta, all_datasets_collapsed, patch_risk_labels, and validation/runtime status as the repair evidence.",
        "Generate a same-direction S2.5 repair or variant that can pass cheap proxy without evaluator changes; variant diversity is a soft preference, not a hard reject rule.",
    ]
    if failure_class == "implementation_failure":
        repair_instructions = [
            "This is implementation_failure, not method_failure: do not return to S1 and do not consume the same-direction method attempt budget.",
            "Repair S2.5 patch eligibility first: produce a valid frozen patch_json, pass validation/runtime smoke, avoid evaluator/test-only changes, and wire ablation/eval activation.",
            "Do not infer the S1/S2 mechanism is bad until a legal patch reaches cheap proxy and fails on proxy metrics.",
            "Use candidate_results.runtime_validation, patch_risk_labels, command_failure, activation_smoke, and proxy_eval_health_failure as the primary repair evidence.",
        ]
    return {
        "schema_version": "c2c_performance_feedback_v1",
        "created_at": now_utc(),
        "reason": reason,
        "result_status": result.get("status"),
        "summary": {key: value for key, value in summary.items() if value is not None},
        "candidate_results": candidate_summaries,
        "failure_feedback_summary": feedback.get("summary") if isinstance(feedback, dict) else {},
        "acceptance": payload.get("acceptance") or {},
        "repair_instructions": repair_instructions,
    }


def _s3_full_performance_feedback(
    project_root: Path,
    registry: dict[str, Any],
    result: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={}) or {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    failure_class = _classify_s3_failure_from_candidates(candidates)
    feedback = read_json(project_root / "experiment" / "results" / "failure_feedback.json", default={}) or {}
    proxy_calibration = read_json(project_root / "experiment" / "results" / "proxy_calibration.json", default={}) or {}
    candidate_summaries = [_full_s3_candidate_summary(item, payload.get("baseline") or {}) for item in candidates]
    proxy_false_positive_count = _nested_int(proxy_calibration, ["current_iteration", "proxy_false_positive_count"])
    if proxy_false_positive_count is None:
        proxy_false_positive_count = _nested_int(proxy_calibration, ["summary", "proxy_false_positive_count"]) or 0
    summary = {
        "route": "full_s3_failure",
        "failure_class": failure_class,
        "does_not_consume_same_direction_attempt": failure_class == "implementation_failure",
        "iteration": registry.get("iteration"),
        "result_status": result.get("status"),
        "reason": reason,
        "full_s3_completed_candidates": len([item for item in candidate_summaries if item.get("metrics")]),
        "proxy_false_positive_count": proxy_false_positive_count,
        "proxy_false_positive_rate": _nested_value(proxy_calibration, ["current_iteration", "proxy_false_positive_rate"])
        or _nested_value(proxy_calibration, ["summary", "proxy_false_positive_rate"]),
        "proxy_full_delta_correlation": _nested_value(proxy_calibration, ["current_iteration", "proxy_full_delta_correlation"])
        or _nested_value(proxy_calibration, ["summary", "proxy_full_delta_correlation"]),
    }
    repair_instructions = [
        "Treat this as method-level full S3 evidence only if the patch was legal and full metrics were produced.",
        "If cheap proxy passed but full S3 failed, treat the proxy/full mismatch as high-priority calibration evidence.",
        "Future S1/S2 should avoid mechanisms, datasets, or integration points marked as proxy false positives unless they add stronger full-readiness evidence.",
    ]
    return {
        "schema_version": "c2c_performance_feedback_v1",
        "created_at": now_utc(),
        "reason": reason,
        "result_status": result.get("status"),
        "summary": {key: value for key, value in summary.items() if value is not None},
        "candidate_results": candidate_summaries,
        "failure_feedback_summary": feedback.get("summary") if isinstance(feedback, dict) else {},
        "acceptance": payload.get("acceptance") or {},
        "proxy_calibration_summary": proxy_calibration.get("summary") if isinstance(proxy_calibration, dict) else {},
        "proxy_calibration_current_iteration": proxy_calibration.get("current_iteration") if isinstance(proxy_calibration, dict) else {},
        "repair_instructions": repair_instructions,
    }


def _full_s3_candidate_summary(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    datasets = metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {}
    baseline_datasets = baseline.get("datasets") if isinstance(baseline, dict) and isinstance(baseline.get("datasets"), dict) else {}
    dataset_regressions: dict[str, float] = {}
    for dataset, baseline_score in baseline_datasets.items():
        if dataset not in datasets:
            continue
        try:
            delta = float(datasets[dataset]) - float(baseline_score)
        except (TypeError, ValueError):
            continue
        if delta < 0:
            dataset_regressions[str(dataset)] = round(abs(delta), 4)
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    variant = candidate.get("s2_variant") if isinstance(candidate.get("s2_variant"), dict) else {}
    return {
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "decision": candidate.get("decision"),
        "mechanism_type": candidate.get("mechanism_type") or variant.get("mechanism_type"),
        "mechanism_axis": candidate.get("mechanism_axis") or variant.get("mechanism_axis"),
        "integration_point": candidate.get("integration_point") or variant.get("integration_point"),
        "control_signal": candidate.get("control_signal") or variant.get("control_signal"),
        "metrics": metrics,
        "delta_vs_baseline": candidate.get("delta_vs_baseline"),
        "dataset_regressions": dataset_regressions,
        "proxy_screen": {
            key: proxy.get(key)
            for key in [
                "status",
                "proxy_delta_vs_baseline",
                "proxy_score",
                "proxy_dataset_deltas",
                "proxy_dataset_regressions",
                "proxy_decision_mode",
            ]
            if proxy.get(key) not in (None, "", [], {})
        },
        "failure_attribution": candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {},
    }


def _nested_value(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _nested_int(payload: dict[str, Any], path: list[str]) -> int | None:
    value = _nested_value(payload, path)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _same_direction_s2_action_recommendation(
    candidate_summaries: list[dict[str, Any]],
    *,
    route: str,
    failure_count: int = 0,
    max_failures: int = 5,
) -> dict[str, Any]:
    if route == "implementation_failure":
        signals = sorted(
            {
                str(signal)
                for item in candidate_summaries
                for signal in (item.get("implementation_failure_signals") or [])
                if signal
            }
        )
        return {
            "action": "patch_repair",
            "reason": "candidate did not reach a valid method-level cheap proxy evaluation; repair implementation eligibility first",
            "matched_rule": "implementation_failure",
            "signals": ["implementation_failure", *signals],
        }
    if failure_count >= max_failures:
        return {
            "action": "return_to_s1_new_direction",
            "reason": "same-direction proxy failure budget is exhausted",
            "matched_rule": "failure_budget_exhausted",
            "signals": ["same_direction_budget_exhausted"],
        }
    if route == "repairable_proxy_risk":
        return {
            "action": "patch_repair",
            "reason": "cheap proxy classified the candidate as repairable risk before full S3",
            "matched_rule": "repairable_proxy_risk",
            "signals": ["repairable_proxy_risk"],
        }
    if not candidate_summaries:
        return {
            "action": "new_same_direction_variant",
            "reason": "no usable blocked-candidate summary was available, so avoid blind patch repair",
            "matched_rule": "missing_candidate_summary",
            "signals": ["missing_candidate_summary"],
        }
    signals: list[str] = []
    for item in candidate_summaries:
        runtime = item.get("runtime_validation") if isinstance(item.get("runtime_validation"), dict) else {}
        if runtime.get("validation") == "failed" or runtime.get("runtime_smoke") == "failed":
            signals.append("runtime_or_validation_failed")
            signals.extend(_runtime_failure_signals(runtime))
        labels = {str(label) for label in item.get("patch_risk_labels") or []}
        if any("evaluation" in label or "evaluator" in label or "test_change" in label for label in labels):
            signals.append("evaluator_or_test_patch_risk")
        if len(item.get("patch_risk_files") or []) >= 4 or len(item.get("changed_files") or []) >= 5:
            signals.append("patch_too_broad")
    if signals:
        return {
            "action": "patch_repair",
            "reason": "patch/runtime risk should be fixed before spending another same-direction mechanism variant",
            "matched_rule": "patch_or_runtime_repair",
            "signals": sorted(set(signals)),
        }
    if all(item.get("all_proxy_datasets_below_baseline") for item in candidate_summaries):
        return {
            "action": "new_same_direction_variant",
            "reason": "all proxy datasets were below baseline, so local repair is unlikely to rescue this exact mechanism shape",
            "matched_rule": "all_dataset_collapse",
            "signals": ["all_proxy_datasets_below_baseline"],
        }
    single_dataset = _single_dataset_small_drop_signal(candidate_summaries)
    if single_dataset:
        return {
            "action": "mechanism_repair",
            "reason": "only one proxy dataset shows a bounded drop, so repair the same mechanism behavior instead of changing variant",
            "matched_rule": "single_dataset_small_drop",
            "signals": ["single_dataset_small_drop", *single_dataset],
        }
    if any(_candidate_has_positive_dataset_signal(item) for item in candidate_summaries):
        return {
            "action": "mechanism_repair",
            "reason": "at least one proxy dataset improved, so keep the mechanism shape and repair the failing dataset behavior",
            "matched_rule": "mixed_dataset_signal",
            "signals": ["mixed_dataset_signal"],
        }
    mean_deltas = [
        float(item["proxy_mean_delta"])
        for item in candidate_summaries
        if isinstance(item.get("proxy_mean_delta"), (int, float))
    ]
    if mean_deltas and max(mean_deltas) <= -2.0:
        return {
            "action": "new_same_direction_variant",
            "reason": "proxy mean delta is strongly negative without compensating positive dataset signal",
            "matched_rule": "strong_negative_proxy_delta",
            "signals": ["strong_negative_proxy_delta"],
        }
    return {
        "action": "mechanism_repair",
        "reason": "proxy failure is not clearly a runtime bug or full collapse, so try a focused mechanism repair first",
        "matched_rule": "focused_mechanism_repair_default",
        "signals": ["focused_mechanism_repair_default"],
    }


def _runtime_failure_signals(runtime: dict[str, Any]) -> list[str]:
    signals = []
    text = json.dumps(runtime, ensure_ascii=False, default=str).lower()
    for token in ["dtype", "device", "valid_mask", "first_batch", "first batch", "cuda", "cpu"]:
        if token in text:
            signals.append(f"runtime_{token.replace(' ', '_')}")
    return signals


def _single_dataset_small_drop_signal(candidate_summaries: list[dict[str, Any]]) -> list[str]:
    signals = []
    for item in candidate_summaries:
        regressions = item.get("proxy_dataset_regressions") if isinstance(item.get("proxy_dataset_regressions"), dict) else {}
        dragging = item.get("dragging_datasets") if isinstance(item.get("dragging_datasets"), list) else []
        dataset_names = {str(entry.get("dataset")) for entry in dragging if isinstance(entry, dict) and entry.get("dataset")}
        if not dataset_names and regressions:
            dataset_names = {str(key) for key, value in regressions.items() if _coerce_float(value) > 0}
        if len(dataset_names) != 1:
            continue
        worst = max([_coerce_float(value) for value in regressions.values()] or [0.0])
        if worst <= 2.0:
            signals.append(f"dataset:{next(iter(dataset_names))}")
    return signals[:3]


def _candidate_has_positive_dataset_signal(candidate_summary: dict[str, Any]) -> bool:
    deltas = candidate_summary.get("proxy_dataset_deltas") if isinstance(candidate_summary.get("proxy_dataset_deltas"), dict) else {}
    for value in deltas.values():
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _update_c2c_direction_scorecard(project_root: Path, registry: dict[str, Any], performance_feedback: dict[str, Any]) -> dict[str, Any]:
    path = project_root / "plan" / "direction_scorecard.json"
    scorecard = read_json(
        path,
        default={
            "schema_version": "c2c_direction_scorecard_v1",
            "project_id": project_root.name,
            "directions": {},
        },
    )
    if not isinstance(scorecard, dict):
        scorecard = {"schema_version": "c2c_direction_scorecard_v1", "project_id": project_root.name, "directions": {}}
    scorecard.setdefault("schema_version", "c2c_direction_scorecard_v1")
    scorecard.setdefault("project_id", project_root.name)
    scorecard.setdefault("directions", {})
    direction = _current_c2c_direction(project_root, performance_feedback)
    direction_id = direction["id"]
    directions = scorecard.setdefault("directions", {})
    current = directions.get(direction_id) if isinstance(directions.get(direction_id), dict) else {}
    attempts = [item for item in current.get("attempts") or [] if isinstance(item, dict)]
    attempt = _direction_score_attempt(project_root, registry, performance_feedback)
    attempts.append(attempt)
    max_attempts = 25
    attempts = attempts[-max_attempts:]
    summary = _direction_score_summary(attempts, performance_feedback)
    current = {
        "direction_id": direction_id,
        "title": direction.get("title"),
        "mechanism_type": direction.get("mechanism_type"),
        "updated_at": now_utc(),
        "attempt_count": len(attempts),
        "summary": summary,
        "attempts": attempts,
        "s1_feedback": _direction_score_s1_feedback(direction, summary),
        "source_paths": [
            "plan/performance_feedback.json",
            "experiment/results/main_results.json",
            "experiment/results/failure_feedback.json",
        ],
    }
    directions[direction_id] = current
    scorecard["updated_at"] = now_utc()
    scorecard["current_direction_id"] = direction_id
    scorecard["current_direction"] = current
    write_json(path, scorecard)
    return scorecard


def _current_c2c_direction(project_root: Path, performance_feedback: dict[str, Any]) -> dict[str, Any]:
    candidate_results = performance_feedback.get("candidate_results") if isinstance(performance_feedback.get("candidate_results"), list) else []
    candidate = next((item for item in candidate_results if isinstance(item, dict)), {})
    candidates_path = project_root / "plan" / "candidate_ideas.json"
    ideas = read_json(candidates_path, default=[]) or []
    selected = next((idea for idea in ideas if isinstance(idea, dict) and idea.get("selected")), None)
    if selected is None and ideas:
        selected = next((idea for idea in ideas if isinstance(idea, dict)), None)
    s1_path = project_root / "literature" / "ideas.json"
    s1_ideas = read_json(s1_path, default=[]) or []
    s1_selected = next((idea for idea in s1_ideas if isinstance(idea, dict) and idea.get("selected")), None)
    if selected is None:
        selected = s1_selected
    direction_id = (
        (selected or {}).get("s1_planner", {}).get("s1_direction_id")
        if isinstance((selected or {}).get("s1_planner"), dict)
        else None
    )
    direction_id = (
        direction_id
        or ((selected or {}).get("s2_planner") or {}).get("s1_direction_id")
        if isinstance((selected or {}).get("s2_planner"), dict)
        else direction_id
    )
    direction_id = direction_id or (selected or {}).get("id") or candidate.get("id") or "unknown_direction"
    mechanism_type = (selected or {}).get("mechanism_type") or (s1_selected or {}).get("mechanism_type") or candidate.get("mechanism_type")
    return {
        "id": str(direction_id),
        "title": (selected or {}).get("title") or (s1_selected or {}).get("title") or candidate.get("title"),
        "mechanism_type": mechanism_type,
    }


def _direction_score_attempt(project_root: Path, registry: dict[str, Any], performance_feedback: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in performance_feedback.get("candidate_results") or [] if isinstance(item, dict)]
    mean_deltas = [
        float(item["proxy_mean_delta"])
        for item in candidates
        if isinstance(item.get("proxy_mean_delta"), (int, float))
    ]
    positive_dataset_hits = [
        item.get("id")
        for item in candidates
        if _candidate_has_positive_dataset_signal(item)
    ]
    runtime_stable = all(_candidate_runtime_stable(item) for item in candidates) if candidates else False
    low_patch_risk = all(_candidate_low_patch_risk(item) for item in candidates) if candidates else False
    all_dataset_collapse = bool(candidates) and all(item.get("all_proxy_datasets_below_baseline") for item in candidates)
    return {
        "timestamp": now_utc(),
        "iteration": registry.get("iteration"),
        "route": (performance_feedback.get("summary") or {}).get("route"),
        "recommended_s2_action": (performance_feedback.get("summary") or {}).get("recommended_s2_action"),
        "candidate_ids": [item.get("id") for item in candidates if item.get("id")],
        "best_proxy_delta": max(mean_deltas) if mean_deltas else None,
        "proxy_mean_delta_min": min(mean_deltas) if mean_deltas else None,
        "has_positive_dataset_signal": bool(positive_dataset_hits),
        "positive_dataset_candidate_ids": [item for item in positive_dataset_hits if item],
        "runtime_stable": runtime_stable,
        "low_patch_risk": low_patch_risk,
        "all_datasets_collapsed": all_dataset_collapse,
        "dragging_datasets": _direction_dragging_datasets(candidates),
        "candidate_count": len(candidates),
    }


def _direction_score_summary(attempts: list[dict[str, Any]], performance_feedback: dict[str, Any]) -> dict[str, Any]:
    best_deltas = [
        float(item["best_proxy_delta"])
        for item in attempts
        if isinstance(item.get("best_proxy_delta"), (int, float))
    ]
    best_proxy_delta = max(best_deltas) if best_deltas else None
    positive_attempts = sum(1 for item in attempts if item.get("has_positive_dataset_signal"))
    runtime_stable_attempts = sum(1 for item in attempts if item.get("runtime_stable"))
    low_patch_risk_attempts = sum(1 for item in attempts if item.get("low_patch_risk"))
    collapse_attempts = sum(1 for item in attempts if item.get("all_datasets_collapsed"))
    budget = (performance_feedback.get("summary") or {}).get("same_direction_failure_budget")
    failure_count = (performance_feedback.get("summary") or {}).get("same_direction_failure_count")
    status = "active"
    if failure_count is not None and budget is not None:
        try:
            if int(failure_count) >= int(budget):
                status = "budget_exhausted"
        except (TypeError, ValueError):
            status = "active"
    health_score = _direction_health_score(
        attempt_count=len(attempts),
        best_proxy_delta=best_proxy_delta,
        positive_attempts=positive_attempts,
        runtime_stable_attempts=runtime_stable_attempts,
        low_patch_risk_attempts=low_patch_risk_attempts,
        collapse_attempts=collapse_attempts,
    )
    return {
        "status": status,
        "attempt_count": len(attempts),
        "same_direction_failure_count": failure_count,
        "same_direction_failure_budget": budget,
        "best_proxy_delta": best_proxy_delta,
        "positive_dataset_signal_attempts": positive_attempts,
        "runtime_stable_attempts": runtime_stable_attempts,
        "low_patch_risk_attempts": low_patch_risk_attempts,
        "all_dataset_collapse_attempts": collapse_attempts,
        "health_score": health_score,
        "direction_quality": _direction_quality_label(health_score, status=status),
        "latest_recommended_s2_action": (performance_feedback.get("summary") or {}).get("recommended_s2_action"),
    }


def _direction_health_score(
    *,
    attempt_count: int,
    best_proxy_delta: float | None,
    positive_attempts: int,
    runtime_stable_attempts: int,
    low_patch_risk_attempts: int,
    collapse_attempts: int,
) -> float:
    if attempt_count <= 0:
        return 0.0
    score = 0.0
    if best_proxy_delta is not None:
        score += max(-3.0, min(3.0, best_proxy_delta)) * 10.0
    score += 12.0 * positive_attempts
    score += 5.0 * runtime_stable_attempts
    score += 4.0 * low_patch_risk_attempts
    score -= 14.0 * collapse_attempts
    return round(score / max(1, attempt_count), 3)


def _direction_quality_label(health_score: float, *, status: str) -> str:
    if status == "budget_exhausted" and health_score < 0:
        return "poor_direction_evidence"
    if health_score >= 15:
        return "promising_direction"
    if health_score >= 0:
        return "mixed_direction_evidence"
    return "weak_direction_evidence"


def _direction_score_s1_feedback(direction: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    title = direction.get("title") or direction.get("direction_id") or direction.get("id") or "current direction"
    exhausted = summary.get("status") == "budget_exhausted"
    if exhausted:
        recommendation = "return_to_s1_new_direction"
    elif summary.get("direction_quality") == "promising_direction":
        recommendation = "keep_direction_with_targeted_repair"
    else:
        recommendation = "continue_same_direction_until_budget"
    conclusion = (
        f"Direction `{title}` has {summary.get('attempt_count')} attempts; "
        f"best_proxy_delta={summary.get('best_proxy_delta')}, "
        f"positive_dataset_signal_attempts={summary.get('positive_dataset_signal_attempts')}, "
        f"runtime_stable_attempts={summary.get('runtime_stable_attempts')}, "
        f"low_patch_risk_attempts={summary.get('low_patch_risk_attempts')}, "
        f"all_dataset_collapse_attempts={summary.get('all_dataset_collapse_attempts')}. "
        f"Recommendation: {recommendation}."
    )
    return {
        "kind": "c2c_direction_scorecard",
        "feedback_view": "method",
        "recommendation": recommendation,
        "conclusion": conclusion,
        "avoid_repeat_rule": (
            "Do not repeat this S1 direction without a mechanism-level change that addresses the recorded proxy collapse."
            if exhausted and summary.get("all_dataset_collapse_attempts")
            else ""
        ),
    }


def _candidate_runtime_stable(candidate_summary: dict[str, Any]) -> bool:
    runtime = candidate_summary.get("runtime_validation") if isinstance(candidate_summary.get("runtime_validation"), dict) else {}
    return runtime.get("runtime_smoke") in {"passed", "unknown"} and runtime.get("validation") in {"passed", "unknown"}


def _candidate_low_patch_risk(candidate_summary: dict[str, Any]) -> bool:
    labels = {str(item) for item in candidate_summary.get("patch_risk_labels") or []}
    if any("evaluation" in label or "evaluator" in label or "test_change" in label for label in labels):
        return False
    return len(candidate_summary.get("patch_risk_files") or []) <= 2 and len(candidate_summary.get("changed_files") or []) <= 4


def _direction_dragging_datasets(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for item in candidate.get("dragging_datasets") or []:
            if not isinstance(item, dict):
                continue
            dataset = item.get("dataset")
            if not dataset:
                continue
            current = merged.get(str(dataset))
            regression = item.get("regression")
            if current is None or _coerce_float(regression) > _coerce_float(current.get("regression")):
                merged[str(dataset)] = dict(item)
    return list(merged.values())


def _proxy_rejected_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    attribution = candidate.get("failure_attribution") if isinstance(candidate.get("failure_attribution"), dict) else {}
    patch_result = candidate.get("patch_result") if isinstance(candidate.get("patch_result"), dict) else {}
    patch_risk = attribution.get("patch_risk") if isinstance(attribution.get("patch_risk"), dict) else {}
    if not patch_risk and isinstance(proxy.get("patch_risk"), dict):
        patch_risk = proxy.get("patch_risk") or {}
    dataset_deltas = proxy.get("proxy_dataset_deltas") or {}
    dragging = attribution.get("dragging_datasets") or []
    if not dragging and isinstance(proxy.get("proxy_effect_repair_contract"), dict):
        dragging = proxy["proxy_effect_repair_contract"].get("dragging_datasets") or []
    runtime_status = _candidate_runtime_validation_status(candidate)
    command_failure = proxy.get("command_failure") if isinstance(proxy.get("command_failure"), dict) else {}
    baseline_failure = proxy.get("baseline_failure") if isinstance(proxy.get("baseline_failure"), dict) else {}
    return {
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "decision": candidate.get("decision"),
        "reason": proxy.get("reason") or attribution.get("primary_failure") or candidate.get("command_status"),
        "proxy_mean_delta": proxy.get("proxy_delta_vs_baseline"),
        "proxy_score": proxy.get("proxy_score"),
        "proxy_worst_dataset_regression": proxy.get("proxy_worst_dataset_regression"),
        "proxy_dataset_deltas": dataset_deltas,
        "proxy_dataset_regressions": proxy.get("proxy_dataset_regressions") or {},
        "dragging_datasets": dragging,
        "all_proxy_datasets_below_baseline": bool(dataset_deltas) and all(_coerce_float(value) < 0 for value in dataset_deltas.values()),
        "patch_risk_files": patch_risk.get("risk_files") or [],
        "patch_risk_labels": patch_risk.get("risk_labels") or [],
        "changed_files": patch_result.get("changed_files") or [],
        "runtime_validation": runtime_status,
        "command_failure": command_failure,
        "baseline_failure": baseline_failure,
        "baseline_status": proxy.get("baseline_status"),
        "implementation_failure_signals": _candidate_implementation_failure_signals(candidate),
    }


def _candidate_runtime_validation_status(candidate: dict[str, Any]) -> dict[str, Any]:
    patch_result = candidate.get("patch_result") if isinstance(candidate.get("patch_result"), dict) else {}
    validation = patch_result.get("validation") if isinstance(patch_result.get("validation"), dict) else {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), list) else []
    if not checks:
        return {
            "patch_status": patch_result.get("status"),
            "command_status": candidate.get("command_status"),
            "runtime_smoke": "unknown",
            "validation": "unknown",
            "implementation_failure_signals": _candidate_implementation_failure_signals(candidate),
        }
    runtime_checks = [check for check in checks if isinstance(check, dict) and str(check.get("name") or "").startswith("runtime_smoke:")]
    failed = [check for check in checks if isinstance(check, dict) and check.get("returncode") not in (0, None)]
    return {
        "patch_status": patch_result.get("status"),
        "command_status": candidate.get("command_status"),
        "runtime_smoke": _check_status(runtime_checks),
        "validation": "passed" if not failed else "failed",
        "failed_checks": [
            {"name": check.get("name"), "failure_category": check.get("failure_category"), "returncode": check.get("returncode")}
            for check in failed[:3]
        ],
        "implementation_failure_signals": _candidate_implementation_failure_signals(candidate),
    }


def _check_status(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "unknown"
    if all(check.get("returncode") == 0 for check in checks):
        return "passed"
    return "failed"


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _s3_result_has_repairable_proxy_risk(project_root: Path) -> bool:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={})
    for candidate in payload.get("candidate_results") or []:
        if not isinstance(candidate, dict):
            continue
        if _candidate_has_resource_retry(candidate):
            continue
        if candidate.get("decision") == "proxy_repairable":
            return True
        if (candidate.get("proxy_screen") or {}).get("status") == "repairable_proxy_risk":
            return True
        attribution = candidate.get("failure_attribution") or {}
        if attribution.get("primary_failure") == "repairable_proxy_risk_before_full_training":
            return True
    return False


def _s3_result_has_proxy_rejection(project_root: Path) -> bool:
    payload = read_json(project_root / "experiment" / "results" / "main_results.json", default={})
    candidates = [candidate for candidate in payload.get("candidate_results") or [] if isinstance(candidate, dict)]
    if not candidates:
        return False
    return all(candidate.get("decision") == "proxy_rejected" for candidate in candidates)
