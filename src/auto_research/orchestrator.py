"""Pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ArtifactManager
from .config import apply_runtime_overrides, load_project_config, load_root_config
from .judges import judge_s1, judge_s2, judge_s3, judge_s4, judge_s5
from .importers import ConsensusImporter
from .llm import ModelClient
from .registry import begin_stage, block_stage, complete_stage, fail_stage, increment_judge_retry, invalidate_from, load_registry, save_registry, set_review_outcome
from .utils import now_utc
from .workspace import init_workspace
from .agents.base import AgentContext
from .agents.experiment import ExperimentAgent
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
        topic = registry["research_topic"]
        iteration = registry["iteration"]
        self._log_session(project_root, action="start_run", details={"stage": registry["current_stage"], "iteration": iteration})

        while registry["current_stage"] != "DONE":
            stage_key = registry["current_stage"]
            begin_stage(registry, stage_key)
            save_registry(registry_path, registry)
            if stage_key == "S1_literature":
                result = LiteratureAgent(context).run(topic)
                ok, reason = judge_s1(project_root)
            elif stage_key == "S2_plan":
                result = PlanAgent(context).run()
                ok, reason = judge_s2(project_root)
            elif stage_key == "S3_experiment":
                result = ExperimentAgent(context).run()
                if result.get("status") == "blocked":
                    block_stage(registry, stage_key, result["blocked_reason"])
                    save_registry(registry_path, registry)
                    self._log_session(project_root, action="blocked", details={"stage": stage_key, "reason": result["blocked_reason"]})
                    return {"status": "blocked", "stage": stage_key, "reason": result["blocked_reason"]}
                ok, reason = judge_s3(project_root)
            elif stage_key == "S4_writing":
                LiteratureAgent(context).run(topic, phase="related_work_audit")
                result = WritingAgent(context).run()
                ok, reason = judge_s4(project_root, config)
            elif stage_key == "S5_review":
                result = ReviewAgent(context).run(iteration=registry["iteration"])
                set_review_outcome(registry, decision=result["decision"], score=result["score"], revision_dispatch=result["dispatch_path"])
                ok, reason = judge_s5(project_root)
            else:
                fail_stage(registry, stage_key, f"Unknown stage {stage_key}")
                save_registry(registry_path, registry)
                return {"status": "failed", "stage": stage_key}

            if ok:
                complete_stage(registry, stage_key, artifacts=result.get("artifacts", []))
                save_registry(registry_path, registry)
                self._log_session(project_root, action="stage_completed", details={"stage": stage_key})
                if stage_key == "S5_review":
                    if result["decision"] == "ACCEPT":
                        registry["current_stage"] = "DONE"
                        registry["status"] = "completed"
                        save_registry(registry_path, registry)
                        break
                    if result["decision"] == "REVISE":
                        self._execute_targeted_revisions(project_root, context, registry)
                        save_registry(registry_path, registry)
                        continue
                continue

            retries = increment_judge_retry(registry, stage_key)
            max_retries = config.get("orchestration", {}).get("judge_max_retries", 2)
            if retries <= max_retries:
                save_registry(registry_path, registry)
                self._log_session(project_root, action="judge_retry", details={"stage": stage_key, "retries": retries, "reason": reason})
                continue
            fail_stage(registry, stage_key, reason)
            save_registry(registry_path, registry)
            self._log_session(project_root, action="stage_failed", details={"stage": stage_key, "reason": reason})
            return {"status": "failed", "stage": stage_key, "reason": reason}
        return {"status": registry["status"], "project_id": registry["project_id"]}

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
