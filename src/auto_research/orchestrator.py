"""Pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactManager
from .c2c import build_c2c_project_config, snapshot_c2c_repo, write_c2c_project_config
from .config import apply_runtime_overrides, load_project_config, load_root_config
from .judges import gate_s0, gate_s1, gate_s2, gate_s3, gate_s4, gate_s5
from .importers import ConsensusImporter
from .llm import ModelClient
from .orchestration_state import OrchestrationStateManager
from .registry import begin_stage, block_stage, complete_stage, fail_stage, increment_judge_retry, invalidate_from, load_registry, save_registry, set_review_outcome
from .stage_contracts import StageContractManager
from .utils import now_utc, read_json, write_json
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
            "artifact_counts": stage_counts,
        }

    def catchup(self, project_id: str) -> str:
        project_root = self._project_root(project_id)
        registry = load_registry(project_root / "meta" / "registry.yaml")
        selected_idea = "not selected"
        ideas_path = project_root / "literature" / "ideas.json"
        if ideas_path.exists():
            ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
            selected_idea = next((idea["title"] for idea in ideas if idea.get("selected")), ideas[0]["title"])
        plan_summary = (project_root / "plan" / "resource_budget.md").read_text(encoding="utf-8") if (project_root / "plan" / "resource_budget.md").exists() else "No plan yet."
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
                result = PlanAgent(context).run()
                gate_report = gate_s2(project_root, config)
            elif stage_key == "S3_experiment":
                result = ExperimentAgent(context).run()
                if result.get("status") == "blocked":
                    if self._should_route_s3_repairable_proxy_to_s2(config, project_root, registry):
                        routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, result["blocked_reason"], config=config)
                        save_registry(registry_path, registry)
                        state.failure_feedback_routed(registry, routed)
                        self._log_session(project_root, action="s3_repairable_proxy_to_s2", details=routed)
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
                if not ok and self._should_route_s3_repairable_proxy_to_s2(config, project_root, registry):
                    routed = self._route_s3_repairable_proxy_to_s2(project_root, registry, result, reason, config=config)
                    save_registry(registry_path, registry)
                    state.failure_feedback_routed(registry, routed)
                    self._log_session(project_root, action="s3_repairable_proxy_to_s2", details=routed)
                    if routed["status"] == "routed":
                        contracts.stage_stopped(stage_key, status="feedback_routed", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                        context = self._context(project_root, load_project_config(project_root))
                        continue
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
                if not ok and self._should_route_s3_failure_to_s1(config, result):
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
    def _should_route_s3_failure_to_s1(config: dict[str, Any], result: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_s3_failure_to_s1", True):
            return False
        return result.get("status") in {"not_viable", "partial", "failed"}

    @staticmethod
    def _should_route_s3_repairable_proxy_to_s2(config: dict[str, Any], project_root: Path, registry: dict[str, Any]) -> bool:
        feedback_cfg = config.get("orchestration", {}).get("failure_feedback", {})
        if not feedback_cfg.get("enabled", True) or not feedback_cfg.get("route_repairable_proxy_to_s2", True):
            return False
        if not _s3_result_has_repairable_proxy_risk(project_root):
            return False
        repair_routes = registry.setdefault("repair_routes", {})
        iteration_key = str(registry.get("iteration") or 1)
        max_routes = int(feedback_cfg.get("max_proxy_repair_routes_per_iteration", 1) or 1)
        return int(repair_routes.get(iteration_key) or 0) < max_routes

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
        repair_routes = registry.setdefault("repair_routes", {})
        iteration_key = str(registry.get("iteration") or 1)
        repair_routes[iteration_key] = int(repair_routes.get(iteration_key) or 0) + 1
        feedback_cfg = (config or {}).get("orchestration", {}).get("failure_feedback", {})
        max_routes = int(feedback_cfg.get("max_proxy_repair_routes_per_iteration", 1) or 1)
        performance_feedback = _s3_proxy_performance_feedback(
            project_root,
            registry,
            result,
            reason,
            route_count=repair_routes[iteration_key],
            failure_count=repair_routes[iteration_key],
            max_failures=max_routes,
            route="repairable_proxy_risk",
        )
        scorecard = _update_c2c_direction_scorecard(project_root, registry, performance_feedback)
        performance_feedback["direction_scorecard"] = scorecard.get("current_direction")
        performance_feedback["direction_scorecard_path"] = "plan/direction_scorecard.json"
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
            "repair_route": "repairable_proxy_risk",
            "route_count": repair_routes[iteration_key],
            "performance_feedback": performance_feedback.get("summary", {}),
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        invalidate_from(registry, "S2_plan", invalidated_by="s3_repairable_proxy")
        for stage_key in ["S2_plan", "S3_experiment"]:
            registry["stages"][stage_key]["judge_retries"] = 0
        return {
            "status": "routed",
            "reason": reason,
            "next_stage": "S2_plan",
            "iteration": registry.get("iteration"),
            "route_count": repair_routes[iteration_key],
            "performance_feedback_path": "plan/performance_feedback.json",
            "performance_feedback": performance_feedback.get("summary", {}),
        }

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
        max_iterations = int(registry.get("max_iterations") or 1)
        if int(registry.get("iteration") or 1) >= max_iterations:
            block_stage(registry, "S3_experiment", f"Reached max_iterations={max_iterations} after S3 failure feedback: {reason}")
            return {"status": "blocked", "reason": registry["blocked_reason"]}
        early_stop_reason = Orchestrator._s3_failure_early_stop_reason(project_root, registry, reason, config=config or {})
        if early_stop_reason:
            block_stage(registry, "S3_experiment", early_stop_reason)
            return {"status": "blocked", "reason": registry["blocked_reason"]}
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
        return {"status": "routed", "reason": reason, "next_iteration": registry["iteration"]}

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


def _same_direction_proxy_failure_count(registry: dict[str, Any], iteration_key: str) -> int:
    proxy_rejected_routes = registry.get("proxy_rejected_routes") or {}
    return int(proxy_rejected_routes.get(iteration_key) or 0)


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
    blocked = [item for item in candidates if item.get("decision") in {"proxy_rejected", "proxy_repairable"}]
    rejected = [item for item in blocked if item.get("decision") == "proxy_rejected"]
    repairable = [item for item in blocked if item.get("decision") == "proxy_repairable"]
    feedback = read_json(project_root / "experiment" / "results" / "failure_feedback.json", default={}) or {}
    candidate_summaries = [_proxy_rejected_candidate_summary(item) for item in blocked]
    action = _same_direction_s2_action_recommendation(candidate_summaries, route=route)
    mean_deltas = [
        item["proxy_mean_delta"]
        for item in candidate_summaries
        if isinstance(item.get("proxy_mean_delta"), (int, float))
    ]
    all_datasets_collapsed = bool(candidate_summaries) and all(
        item.get("all_proxy_datasets_below_baseline") for item in candidate_summaries
    )
    next_action = "repair_or_variant_same_direction" if failure_count < max_failures else "return_to_s1_new_direction"
    recommended_s2_action = action["action"] if failure_count < max_failures else "return_to_s1_new_direction"
    summary = {
        "route": route,
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
        "repair_vs_variant_reason": action["reason"],
        "repair_vs_variant_signals": action["signals"],
    }
    return {
        "schema_version": "c2c_performance_feedback_v1",
        "created_at": now_utc(),
        "reason": reason,
        "result_status": result.get("status"),
        "summary": {key: value for key, value in summary.items() if value is not None},
        "candidate_results": candidate_summaries,
        "failure_feedback_summary": feedback.get("summary") if isinstance(feedback, dict) else {},
        "acceptance": payload.get("acceptance") or {},
        "repair_instructions": [
            "Keep the current S1 mechanism direction; do not ask S1 for a new idea during this budget.",
            "Follow summary.recommended_s2_action when choosing patch repair, mechanism repair, or a new same-direction variant.",
            "Use dragging_datasets, proxy_mean_delta, all_datasets_collapsed, patch_risk_labels, and validation/runtime status as the repair evidence.",
            "Generate a same-direction S2.5 repair or variant that can pass cheap proxy without evaluator changes; variant diversity is a soft preference, not a hard reject rule.",
        ],
    }


def _same_direction_s2_action_recommendation(candidate_summaries: list[dict[str, Any]], *, route: str) -> dict[str, Any]:
    if route == "repairable_proxy_risk":
        return {
            "action": "patch_repair",
            "reason": "cheap proxy classified the candidate as repairable risk before full S3",
            "signals": ["repairable_proxy_risk"],
        }
    if not candidate_summaries:
        return {
            "action": "new_same_direction_variant",
            "reason": "no usable blocked-candidate summary was available, so avoid blind patch repair",
            "signals": ["missing_candidate_summary"],
        }
    signals: list[str] = []
    for item in candidate_summaries:
        runtime = item.get("runtime_validation") if isinstance(item.get("runtime_validation"), dict) else {}
        if runtime.get("validation") == "failed" or runtime.get("runtime_smoke") == "failed":
            signals.append("runtime_or_validation_failed")
        labels = {str(label) for label in item.get("patch_risk_labels") or []}
        if any("evaluation" in label or "evaluator" in label or "test_change" in label for label in labels):
            signals.append("evaluator_or_test_patch_risk")
        if len(item.get("patch_risk_files") or []) >= 4 or len(item.get("changed_files") or []) >= 5:
            signals.append("patch_too_broad")
    if signals:
        return {
            "action": "patch_repair",
            "reason": "patch/runtime risk should be fixed before spending another same-direction mechanism variant",
            "signals": sorted(set(signals)),
        }
    if all(item.get("all_proxy_datasets_below_baseline") for item in candidate_summaries):
        return {
            "action": "new_same_direction_variant",
            "reason": "all proxy datasets were below baseline, so local repair is unlikely to rescue this exact mechanism shape",
            "signals": ["all_proxy_datasets_below_baseline"],
        }
    if any(_candidate_has_positive_dataset_signal(item) for item in candidate_summaries):
        return {
            "action": "mechanism_repair",
            "reason": "at least one proxy dataset improved, so keep the mechanism shape and repair the failing dataset behavior",
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
            "signals": ["strong_negative_proxy_delta"],
        }
    return {
        "action": "mechanism_repair",
        "reason": "proxy failure is not clearly a runtime bug or full collapse, so try a focused mechanism repair first",
        "signals": ["focused_mechanism_repair_default"],
    }


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
