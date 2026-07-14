"""Pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactManager
from .c2c import build_c2c_project_config, snapshot_c2c_repo, write_c2c_project_config
from .c2c_e2e import (
    write_c2c_artifact_audit_report,
    write_c2c_e2e_readiness_report,
    write_c2c_e2e_run_manifest,
    write_c2c_execution_hooks_report,
    write_c2c_real_smoke_record,
    write_c2c_runtime_health_report,
    write_c2c_replay_plan,
    write_c2c_replay_result,
)
from .code_patch import is_retryable_patch_manifest
from .config import apply_runtime_overrides, load_project_config, load_root_config
from .judges import gate_s0, gate_s1, gate_s2, gate_s3, gate_s4, gate_s5
from .importers import ConsensusImporter
from .llm import ModelClient
from .method_memory import append_shared_c2c_method_failure
from .orchestration_state import OrchestrationStateManager
from .reporting import build_project_report
from .research_state import IntegrityError
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

    def doctor_c2c(self, project_id: str) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        config = load_project_config(project_root)
        runtime = write_c2c_runtime_health_report(project_root, config)
        execution_hooks = write_c2c_execution_hooks_report(project_root, config)
        readiness = write_c2c_e2e_readiness_report(project_root, config)
        smoke = write_c2c_real_smoke_record(project_root, config)
        self._log_session(project_root, action="doctor_c2c", details={"readiness_gate": readiness.get("gate")})
        return {
            "status": readiness.get("gate"),
            "project_id": project_id,
            "readiness_report": readiness,
            "runtime_health_report": runtime,
            "execution_hooks_report": execution_hooks,
            "real_smoke_record": smoke,
            "artifacts": [
                "meta/c2c_e2e_readiness_report.json",
                "meta/c2c_runtime_health_report.json",
                "meta/c2c_execution_hooks_report.json",
                "meta/c2c_real_smoke_record.json",
            ],
        }

    def audit_c2c(self, project_id: str, *, scope: str | None = None) -> dict[str, Any]:
        project_root = self._project_root(project_id)
        config = load_project_config(project_root)
        audit = write_c2c_artifact_audit_report(project_root, config, scope=scope)
        smoke = write_c2c_real_smoke_record(project_root, config)
        self._log_session(project_root, action="audit_c2c", details={"audit_gate": audit.get("gate"), "scope": audit.get("audit_scope")})
        return {
            "status": audit.get("gate"),
            "project_id": project_id,
            "artifact_audit_report": audit,
            "real_smoke_record": smoke,
            "artifacts": ["meta/c2c_artifact_audit_report.json", "meta/c2c_real_smoke_record.json"],
        }

    def replay_c2c(self, project_id: str, *, from_stage: str = "S3_experiment") -> dict[str, Any]:
        project_root = self._project_root(project_id)
        config = load_project_config(project_root)
        plan = write_c2c_replay_plan(project_root, replay_from=from_stage)
        result = write_c2c_replay_result(project_root, config)
        smoke = write_c2c_real_smoke_record(project_root, config)
        self._log_session(project_root, action="replay_c2c", details={"from_stage": from_stage, "status": result.get("status")})
        return {
            "status": result.get("status"),
            "project_id": project_id,
            "replay_plan": plan,
            "replay_result": result,
            "real_smoke_record": smoke,
            "artifacts": ["meta/c2c_replay_plan.json", "meta/c2c_replay_result.json", "meta/c2c_real_smoke_record.json"],
        }

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
        direction_path = project_root / "literature" / "direction.json"
        if direction_path.exists():
            direction = read_json(direction_path, default={}) or {}
            if isinstance(direction, dict) and direction.get("direction_id"):
                selected_idea = str(direction.get("research_question") or direction.get("direction_id"))
        plan_summary = "No plan yet."
        plan_path = project_root / "plan" / "trial_spec.json"
        if plan_path.exists():
            plan = read_json(plan_path, default={}) or {}
            variant = read_json(project_root / "plan" / "variant.json", default={}) or {}
            plan_summary = "\n".join(
                item
                for item in [
                    f"Variant: {variant.get('variant_id')}" if variant else "",
                    f"Hypothesis: {variant.get('hypothesis')}" if variant else "",
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
        e2e_block = self._prepare_c2c_e2e_run(project_root, config, registry_path, registry, state)
        if e2e_block:
            return e2e_block

        while registry["current_stage"] != "DONE":
            stage_key = registry["current_stage"]
            input_status = contracts.required_input_status(stage_key, iteration=registry.get("iteration"), config=config)
            if input_status["missing_inputs"]:
                reason = (
                    f"{stage_key} required inputs missing before agent execution: "
                    + ", ".join(input_status["missing_inputs"])
                    + ". Breaking contracts require rerunning from S1 when canonical S1-S3 artifacts are absent."
                )
                block_stage(registry, stage_key, reason)
                save_registry(registry_path, registry)
                state.stage_blocked(registry, stage_key, reason)
                self._log_session(project_root, action="stage_input_contract_blocked", details={"stage": stage_key, "missing_inputs": input_status["missing_inputs"]})
                return {"status": "blocked", "stage": stage_key, "reason": reason, "missing_inputs": input_status["missing_inputs"]}
            begin_stage(registry, stage_key)
            save_registry(registry_path, registry)
            state.stage_started(registry, stage_key)
            contracts.stage_started(stage_key, iteration=registry.get("iteration"), config=config)
            self._record_c2c_e2e_stage_event(project_root, config, stage_key, "started")
            if stage_key == "S0_intake":
                result = IntakeAgent(context).run(topic)
                if result.get("status") == "blocked":
                    block_stage(registry, stage_key, result.get("blocked_reason", "Static intake stage blocked."))
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, registry["blocked_reason"])
                    contracts.stage_stopped(stage_key, status="blocked", reason=registry["blocked_reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    self._record_c2c_e2e_stage_event(project_root, config, stage_key, "blocked", reason=registry["blocked_reason"])
                    self._finalize_c2c_e2e_run(project_root, config, registry, final_status="blocked")
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
                    self._record_c2c_e2e_stage_event(project_root, config, stage_key, "blocked", reason=registry["blocked_reason"])
                    self._finalize_c2c_e2e_run(project_root, config, registry, final_status="blocked")
                    self._log_session(project_root, action="blocked", details={"stage": stage_key, "reason": registry["blocked_reason"]})
                    return {"status": "blocked", "stage": stage_key, "reason": registry["blocked_reason"]}
                gate_report = gate_s1(project_root, config)
            elif stage_key == "S2_plan":
                result = PlanAgent(context).run()
                gate_report = gate_s2(project_root, config)
            elif stage_key == "S3_experiment":
                try:
                    result = ExperimentAgent(context).run()
                except IntegrityError as exc:
                    reason = f"S3 authoritative state rejected execution before commit: {exc}"
                    block_stage(registry, stage_key, reason)
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, reason)
                    contracts.stage_stopped(stage_key, status="blocked", reason=reason, config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": reason}
                gate_report = gate_s3(project_root, config)
                ok, reason = gate_report.legacy_tuple()
                gate_record = self._write_gate_report(context.artifacts, stage_key, gate_report)
                result.setdefault("artifacts", []).append(gate_record["path"])
                gate_payload = gate_report.to_dict()
                gate_payload["report_path"] = gate_record["path"]
                state.gate_recorded(registry, stage_key, passed=ok, reason=reason, report=gate_payload)
                contracts.gate_recorded(stage_key, gate_payload, report_path=gate_record["path"])
                route_outcome = result.get("route_outcome") if isinstance(result.get("route_outcome"), dict) else {}
                if not ok:
                    reason = reason or "S3 strict gate rejected TrialResult or event-derived state"
                    block_stage(registry, stage_key, reason)
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, reason)
                    contracts.stage_stopped(stage_key, status="blocked", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": reason, "route_outcome": route_outcome}
                next_action = route_outcome.get("next_action")
                if next_action in {"PROPOSE_NEXT_VARIANT", "REPAIR_IMPLEMENTATION"}:
                    complete_stage(registry, stage_key, artifacts=result.get("artifacts", []))
                    invalidate_from(registry, "S2_plan", invalidated_by=f"route_outcome:{next_action}")
                    save_registry(registry_path, registry)
                    state.stage_completed(registry, stage_key, artifacts=result.get("artifacts", []))
                    contracts.stage_completed(stage_key, artifacts=result.get("artifacts", []), status="outcome_recorded", reason=next_action, config=config, iteration=registry.get("iteration"))
                    context = self._context(project_root, load_project_config(project_root))
                    continue
                if next_action == "START_NEW_DIRECTION":
                    complete_stage(registry, stage_key, artifacts=result.get("artifacts", []))
                    invalidate_from(registry, "S1_literature", invalidated_by="route_outcome:START_NEW_DIRECTION")
                    save_registry(registry_path, registry)
                    state.stage_completed(registry, stage_key, artifacts=result.get("artifacts", []))
                    contracts.stage_completed(stage_key, artifacts=result.get("artifacts", []), status="direction_finished", reason=next_action, config=config, iteration=registry.get("iteration"))
                    context = self._context(project_root, load_project_config(project_root))
                    continue
                if next_action == "PAUSE_RESOURCE":
                    pause_stage_retryable(registry, stage_key, "Resource pause recorded by the attempt reducer", pause_type="s3_proxy_resource_retry")
                    save_registry(registry_path, registry)
                    state.stage_retryable_paused(registry, stage_key, registry["blocked_reason"])
                    contracts.stage_stopped(stage_key, status="retryable_paused", reason=registry["blocked_reason"], artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "retryable_paused", "stage": stage_key, "reason": registry["blocked_reason"], "attempt_id": (result.get("attempt") or {}).get("attempt_id")}
                if next_action == "BLOCK_INTEGRITY":
                    reason = "Integrity block recorded by the attempt reducer"
                    block_stage(registry, stage_key, reason)
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, reason)
                    contracts.stage_stopped(stage_key, status="blocked", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": reason}
                if next_action == "FINISH_RUN":
                    complete_stage(registry, stage_key, artifacts=result.get("artifacts", []))
                    registry["current_stage"] = "DONE"
                    registry["status"] = "completed"
                    save_registry(registry_path, registry)
                    state.stage_completed(registry, stage_key, artifacts=result.get("artifacts", []))
                    state.mark_completed(registry)
                    contracts.stage_completed(stage_key, artifacts=result.get("artifacts", []), status="completed", reason="FINISH_RUN", config=config, iteration=registry.get("iteration"))
                    break
                if next_action == "FINISH_DIRECTION":
                    result["direction_finished"] = True
                elif next_action not in {
                    "PROPOSE_NEXT_VARIANT",
                    "REPAIR_IMPLEMENTATION",
                    "START_NEW_DIRECTION",
                    "PAUSE_RESOURCE",
                    "BLOCK_INTEGRITY",
                    "FINISH_RUN",
                }:
                    reason = f"Unknown or illegal S3 RouteOutcome action: {next_action!r}"
                    block_stage(registry, stage_key, reason)
                    save_registry(registry_path, registry)
                    state.stage_blocked(registry, stage_key, reason)
                    contracts.stage_stopped(stage_key, status="blocked", reason=reason, artifacts=result.get("artifacts", []), config=config, iteration=registry.get("iteration"))
                    return {"status": "blocked", "stage": stage_key, "reason": reason}
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
                self._record_c2c_e2e_stage_event(project_root, config, stage_key, "failed", reason=f"Unknown stage {stage_key}")
                self._finalize_c2c_e2e_run(project_root, config, registry, final_status="failed")
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
                        self._record_c2c_e2e_stage_event(project_root, config, stage_key, "blocked", reason=f"Rejected by human: {hitl_decision.guidance}")
                        self._finalize_c2c_e2e_run(project_root, config, registry, final_status="blocked")
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
                self._record_c2c_e2e_stage_event(project_root, config, stage_key, "completed")
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
                self._record_c2c_e2e_stage_event(project_root, config, stage_key, "retryable_paused", reason=retryable_pause["reason"])
                self._finalize_c2c_e2e_run(project_root, config, registry, final_status="retryable_paused")
                self._log_session(project_root, action="retryable_paused", details=retryable_pause)
                return {
                    "status": "retryable_paused",
                    "stage": stage_key,
                    "reason": retryable_pause["reason"],
                    "pause_type": retryable_pause["pause_type"],
                    "resume_instruction": registry.get("resume_instruction"),
                }

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
            self._record_c2c_e2e_stage_event(project_root, config, stage_key, "failed", reason=reason)
            self._finalize_c2c_e2e_run(project_root, config, registry, final_status="failed")
            self._log_session(project_root, action="stage_failed", details={"stage": stage_key, "reason": reason})
            return {"status": "failed", "stage": stage_key, "reason": reason}
        state.mark_completed(registry)
        self._finalize_c2c_e2e_run(project_root, config, registry)
        return {"status": registry["status"], "project_id": registry["project_id"]}

    def _prepare_c2c_e2e_run(
        self,
        project_root: Path,
        config: dict[str, Any],
        registry_path: Path,
        registry: dict[str, Any],
        state: OrchestrationStateManager,
    ) -> dict[str, Any] | None:
        if not self._c2c_e2e_enabled(config, "readiness_gate_enabled"):
            return None
        write_c2c_e2e_run_manifest(
            project_root,
            config,
            command={
                "name": "start",
                "simulate": bool((config.get("experiment") or {}).get("simulate")),
                "stop_after_stage": (config.get("orchestration") or {}).get("stop_after_stage"),
                "max_iterations": registry.get("max_iterations") or (config.get("review") or {}).get("max_iterations"),
            },
        )
        write_c2c_runtime_health_report(project_root, config)
        if not self._real_c2c_run(config):
            return None
        write_c2c_execution_hooks_report(project_root, config)
        readiness = write_c2c_e2e_readiness_report(project_root, config)
        e2e_cfg = ((config.get("orchestration") or {}).get("c2c_e2e") or {}) if isinstance(config.get("orchestration"), dict) else {}
        if readiness.get("gate") == "fail" and e2e_cfg.get("block_real_run_on_readiness_fail", True):
            reason = "C2C real-run readiness failed: " + ", ".join(readiness.get("blocking_reasons") or [])
            stage_key = registry.get("current_stage") or "S0_intake"
            block_stage(registry, stage_key, reason)
            save_registry(registry_path, registry)
            state.stage_blocked(registry, stage_key, reason)
            self._record_c2c_e2e_stage_event(project_root, config, str(stage_key), "blocked", reason=reason)
            self._finalize_c2c_e2e_run(project_root, config, registry, final_status="blocked")
            self._log_session(project_root, action="c2c_e2e_readiness_blocked", details=readiness)
            return {
                "status": "blocked",
                "stage": stage_key,
                "reason": reason,
                "readiness_report_path": "meta/c2c_e2e_readiness_report.json",
            }
        return None

    @staticmethod
    def _c2c_e2e_enabled(config: dict[str, Any], key: str) -> bool:
        if not bool(((config or {}).get("c2c") or {}).get("enabled")):
            return False
        cfg = ((config.get("orchestration") or {}).get("c2c_e2e") or {}) if isinstance(config.get("orchestration"), dict) else {}
        return bool(cfg.get(key, False))

    @staticmethod
    def _real_c2c_run(config: dict[str, Any]) -> bool:
        return bool(((config or {}).get("c2c") or {}).get("enabled")) and not bool(((config or {}).get("experiment") or {}).get("simulate"))

    def _record_c2c_e2e_stage_event(
        self,
        project_root: Path,
        config: dict[str, Any],
        stage_key: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        if not self._c2c_e2e_enabled(config, "readiness_gate_enabled"):
            return
        write_c2c_e2e_run_manifest(
            project_root,
            config,
            stage_event={"stage": stage_key, "status": status, "reason": reason, "timestamp": now_utc()},
        )

    def _finalize_c2c_e2e_run(
        self,
        project_root: Path,
        config: dict[str, Any],
        registry: dict[str, Any],
        *,
        final_status: str | None = None,
    ) -> None:
        if not self._c2c_e2e_enabled(config, "readiness_gate_enabled"):
            return
        status = final_status or str(registry.get("status") or "completed")
        write_c2c_e2e_run_manifest(project_root, config, final_status=status)
        if self._real_c2c_run(config) and self._c2c_e2e_enabled(config, "artifact_audit_enabled"):
            write_c2c_artifact_audit_report(project_root, config)
        write_c2c_real_smoke_record(project_root, config)

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
            direction = read_json(project_root / "literature" / "direction.json", default={}) or {}
            invariants = direction.get("mechanism_invariants") if isinstance(direction.get("mechanism_invariants"), dict) else {}
            summary_lines.append(
                f"• **{direction.get('research_question') or direction.get('direction_id')}** [selected]: mediator={invariants.get('target_mediator')}"
            )
        elif stage_key == "S2_plan":
            trial_spec = read_json(project_root / "plan" / "trial_spec.json", default={}) or {}
            variant = read_json(project_root / "plan" / "variant.json", default={}) or {}
            summary_lines.append(f"• Variant: {variant.get('variant_id')}")
            summary_lines.append(f"• Hypothesis: {variant.get('hypothesis')}")
            summary_lines.append(f"• Datasets: {', '.join(_names_for_summary(trial_spec.get('datasets', [])[:5]))}")
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
