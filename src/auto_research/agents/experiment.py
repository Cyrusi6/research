"""Experiment stage."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import yaml

from ..c2c import C2CAdapter, DEFAULT_BASELINE, c2c_candidate_config_overrides, c2c_proxy_screen_config, default_c2c_ideas
from ..code_patch import DynamicEditPolicy, FrozenPatchGuard, archive_patched_code_snapshot
from ..adapters.runner import ExperimentRunner
from ..failure_log import FailureLogManager, build_c2c_feedback_bundle
from ..itr_ideas import screening_summary_markdown
from ..utils import compact_markdown, ensure_dir, now_utc, read_json, read_yaml, sha256_file, write_json
from .base import AgentContext


class ExperimentAgent:
    stage_key = "S3_experiment"

    def __init__(self, context: AgentContext):
        self.context = context
        self.runner = ExperimentRunner(context.config)

    def run(self, *, mode: str = "full", revisions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        env = self.runner.env_report()
        env_record = self.context.artifacts.write_text(
            self.stage_key,
            "env_report.md",
            compact_markdown(self._env_report_md(env)),
            artifact_type="env_report",
            summary="Environment inspection",
        )
        self._scaffold_code()
        if mode == "env_check":
            return {"artifacts": [env_record["path"]], "status": "ok"}

        plan = yaml.safe_load((self.context.project_root / "plan" / "plan.yaml").read_text(encoding="utf-8"))
        execution = plan.get("execution", {})
        simulate = bool(self.context.config.get("experiment", {}).get("simulate"))
        if revisions:
            revision_record = self.context.artifacts.write_text(
                self.stage_key,
                "revision_notes.md",
                compact_markdown(self._revision_notes_md(revisions)),
                artifact_type="revision_notes",
                summary="Applied revision instructions",
            )
        else:
            revision_record = None

        if execution.get("collector") == "c2c_small_loop":
            return self._run_c2c_small_loop(plan, execution, env_record["path"], revision_record["path"] if revision_record else None)
        if simulate or execution.get("mode") == "simulate":
            return self._run_simulated(plan, env_record["path"], revision_record["path"] if revision_record else None)
        if execution.get("mode") == "reuse" and execution.get("collector") == "reused_runs":
            return self._collect_reused_run_results(execution, env_record["path"], revision_record["path"] if revision_record else None)
        if execution.get("collector") == "itr_quick_screen":
            return self._run_itr_quick_screen(execution, env_record["path"], revision_record["path"] if revision_record else None)

        commands = execution.get("commands") or []
        if not commands:
            blocked_reason = execution.get("blocked_reason") or "No execution commands defined."
            self.context.artifacts.write_text(
                self.stage_key,
                "self_heal_log.jsonl",
                json.dumps({"result": "blocked", "reason": blocked_reason}) + "\n",
                artifact_type="self_heal_log",
                summary="Blocked run",
            )
            return {"artifacts": [env_record["path"]], "status": "blocked", "blocked_reason": blocked_reason}

        execution_workdir = Path(execution.get("workdir") or self.context.project_root)
        log_path = self.context.project_root / "experiment" / "logs" / "command_runs.json"
        run_result = self.runner.run_plan_commands(commands, execution_workdir, log_path)
        self.context.artifacts.copy_into_stage(
            self.stage_key,
            log_path,
            "logs/command_runs.json",
            artifact_type="run_log",
            summary="Executed experiment commands",
        )
        if run_result["status"] != "ok":
            self.context.artifacts.write_text(
                self.stage_key,
                "self_heal_log.jsonl",
                json.dumps({"result": "failed", "runs": run_result["runs"]}) + "\n",
                artifact_type="self_heal_log",
                summary="Self-heal trace",
            )
            return {"artifacts": [env_record["path"]], "status": "failed", "blocked_reason": "Experiment commands failed."}
        collector = execution.get("collector")
        if collector == "laps_eval":
            collected = self._collect_laps_eval_results(log_path, env_record["path"], revision_record["path"] if revision_record else None)
            if collected:
                return collected
        return {"artifacts": [env_record["path"]], "status": "blocked", "blocked_reason": "Execution finished but no trusted result collector was available."}

    def _scaffold_code(self) -> None:
        self.context.artifacts.write_text(
            self.stage_key,
            "code/README.md",
            "# Experiment Code\n\nThis directory stores executable experiment code and tests.\n",
            artifact_type="code_readme",
            summary="Experiment code scaffold",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "code/tests/test_placeholder.py",
            "def test_placeholder():\n    assert True\n",
            artifact_type="test_stub",
            summary="Placeholder experiment test",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "configs/default.yaml",
            "seed: 42\n",
            artifact_type="config",
            summary="Default experiment config",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "run_all.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\necho 'Populate commands via plan.yaml execution.commands'\n",
            artifact_type="script",
            summary="Run scaffold",
        )

    def _run_simulated(self, plan: dict[str, Any], env_source: str, revision_source: str | None) -> dict[str, Any]:
        baseline_name = plan["baselines"][0]["name"]
        results = {
            "baseline": {"name": baseline_name, "primary_metric": {"mean": 78.4, "std": 0.5}},
            "proposed_method": {"name": plan["selected_idea"]["title"], "primary_metric": {"mean": 80.1, "std": 0.4}},
        }
        ablation = {
            "full_model": 80.1,
            "without_core_module": 78.9,
        }
        verification = compact_markdown(
            "\n".join(
                [
                    "# Hypothesis Verification",
                    "",
                    "- H1: supported. The proposed method improves the primary metric over the baseline.",
                    "- H2: supported. Removing the core module reduces the metric.",
                ]
            )
        )
        sources = [env_source]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            results,
            artifact_type="results",
            summary="Main experiment results",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            ablation,
            artifact_type="ablation",
            summary="Ablation results",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            verification,
            artifact_type="verification",
            summary="Hypothesis verification",
            source_paths=[main_record["path"], ablation_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Experiment Summary",
                        "",
                        f"- Baseline {baseline_name}: 78.4 ± 0.5",
                        f"- Proposed: 80.1 ± 0.4",
                        "- Ablation confirms the added module contributes most of the gain.",
                    ]
                )
            ),
            artifact_type="summary",
            summary="Experiment summary",
            source_paths=[main_record["path"], ablation_record["path"]],
        )
        table_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/tables/main_table.tex",
            "\\begin{tabular}{lc}\n\\toprule\nMethod & Primary \\\\\n\\midrule\nBaseline & 78.4 \\\\\nProposed & 80.1 \\\\\n\\bottomrule\n\\end{tabular}\n",
            artifact_type="table",
            summary="Main table",
            source_paths=[main_record["path"]],
        )
        ablation_table = self.context.artifacts.write_text(
            self.stage_key,
            "results/tables/ablation_table.tex",
            "\\begin{tabular}{lc}\n\\toprule\nVariant & Primary \\\\\n\\midrule\nFull & 80.1 \\\\\nwo module & 78.9 \\\\\n\\bottomrule\n\\end{tabular}\n",
            artifact_type="table",
            summary="Ablation table",
            source_paths=[ablation_record["path"]],
        )
        figure_record = self.context.artifacts.write_text(
            self.stage_key,
            "figures/main_comparison.txt",
            "Figure placeholder: proposed method improves over baseline.\n",
            artifact_type="figure_placeholder",
            summary="Comparison figure placeholder",
            source_paths=[main_record["path"]],
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "self_heal_log.jsonl",
            "",
            artifact_type="self_heal_log",
            summary="No self-heal actions needed",
        )
        artifacts = [
            main_record["path"],
            ablation_record["path"],
            verification_record["path"],
            summary_record["path"],
            table_record["path"],
            ablation_table["path"],
            figure_record["path"],
        ]
        return {"artifacts": artifacts, "status": "ok"}

    @staticmethod
    def _env_report_md(report: dict[str, Any]) -> str:
        lines = ["# Environment Report", ""]
        lines.append(f"- Python executable: {report.get('python')}")
        lines.append(f"- tmux available: {report.get('tmux')}")
        gpu = report.get("gpu") or []
        if gpu:
            lines.append("- GPUs:")
            for item in gpu:
                lines.append(f"  - {item}")
        else:
            lines.append("- GPUs: none detected")
        return "\n".join(lines)

    @staticmethod
    def _revision_notes_md(revisions: list[dict[str, Any]]) -> str:
        lines = ["# Revision Notes", ""]
        for revision in revisions:
            lines.append(f"- {revision['id']}: {revision['action']}")
        return "\n".join(lines)

    def _collect_laps_eval_results(self, log_path: Path, env_source: str, revision_source: str | None) -> dict[str, Any] | None:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        text = "\n".join((run.get("stdout") or "") + "\n" + (run.get("stderr") or "") for run in payload.get("runs", []))
        metrics = self._parse_laps_metrics(text)
        if not metrics:
            return None
        sources = [env_source, "experiment/logs/command_runs.json"]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            metrics,
            artifact_type="results",
            summary="Parsed LAPS evaluation results",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"note": "Baseline-only evaluation; ablation pending."},
            artifact_type="ablation",
            summary="Ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: not yet tested against a proposed method; baseline evaluation completed.",
                        "- H2: not yet tested; ablation pending.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Baseline-only verification status",
            source_paths=[main_record["path"]],
        )
        summary_lines = ["# Experiment Summary", ""]
        for model_name, scores in metrics.items():
            summary_lines.append(
                f"- {model_name}: i2t R@1={scores['i2t']['R@1']}, t2i R@1={scores['t2i']['R@1']}, rsum={scores['rsum']}"
            )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown("\n".join(summary_lines)),
            artifact_type="summary",
            summary="Parsed evaluation summary",
            source_paths=[main_record["path"]],
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"]],
            "status": "blocked",
            "blocked_reason": "Baseline evaluation completed from local LAPS checkpoints; add proposed-method or fine-tuning commands before continuing.",
        }

    def _collect_reused_run_results(self, execution: dict[str, Any], env_source: str, revision_source: str | None) -> dict[str, Any]:
        selected_runs = execution.get("selected_runs", [])
        copied_artifacts = []
        main_results = {}
        summary_lines = ["# Reused Baseline Summary", ""]
        for idx, run in enumerate(selected_runs, start=1):
            label = f"run_{idx}_{run.get('repo_family','unknown')}_{run.get('dataset','unknown')}_{run.get('encoder','unknown')}"
            log_path = Path(run["log_path"])
            if log_path.exists():
                copied = self.context.artifacts.copy_into_stage(
                    self.stage_key,
                    log_path,
                    f"reused/{label}/eval.log",
                    artifact_type="reused_eval_log",
                    summary="Imported existing evaluation log",
                )
                copied_artifacts.append(copied["path"])
            model_path = run.get("model_best_path")
            if model_path and Path(model_path).exists():
                copied = self.context.artifacts.copy_into_stage(
                    self.stage_key,
                    Path(model_path),
                    f"reused/{label}/model_best.pth",
                    artifact_type="reused_checkpoint",
                    summary="Imported existing checkpoint",
                )
                copied_artifacts.append(copied["path"])
            for results_path in run.get("results_paths", []):
                path = Path(results_path)
                if path.exists():
                    copied = self.context.artifacts.copy_into_stage(
                        self.stage_key,
                        path,
                        f"reused/{label}/{path.name}",
                        artifact_type="reused_similarity",
                        summary="Imported existing similarity file",
                    )
                    copied_artifacts.append(copied["path"])
            key = f"{run.get('repo_family')}:{run.get('dataset')}:{run.get('encoder')}"
            main_results[key] = {
                "rsum": run.get("rsum"),
                "i2t": run.get("i2t"),
                "t2i": run.get("t2i"),
                "source_log": run.get("log_path"),
                "source_repo": run.get("repo_family"),
            }
            summary_lines.append(
                f"- {key}: rsum={run.get('rsum')}, i2t R@1={run.get('i2t',{}).get('R@1')}, t2i R@1={run.get('t2i',{}).get('R@1')}"
            )

        sources = [env_source, *copied_artifacts]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            main_results,
            artifact_type="results",
            summary="Reused baseline results from local MM runs",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"status": "pending", "note": "Existing reusable runs provide baselines; ablation not yet generated for the new project."},
            artifact_type="ablation",
            summary="Ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: baseline references imported from existing local runs; proposed method not yet evaluated.",
                        "- H2: ablation remains pending for the new project-specific method.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Reused baseline verification status",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown("\n".join(summary_lines)),
            artifact_type="summary",
            summary="Summary of reused baseline runs",
            source_paths=[main_record["path"]],
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"], *copied_artifacts],
            "status": "blocked",
            "blocked_reason": "Reusable local baseline runs were imported successfully; add project-specific fine-tuning or new-method commands before continuing.",
        }

    def _run_itr_quick_screen(self, execution: dict[str, Any], env_source: str, revision_source: str | None) -> dict[str, Any]:
        workdir = Path(execution["workdir"])
        project_id = self.context.project_root.name
        control = execution["control"]
        candidates = execution.get("candidates", [])
        data_path = execution["data_path"]
        image_root = execution["image_root"]
        python_cmd = execution["python"]

        screen_specs = [{"id": "control", "title": "LAPS quick control baseline", "direction": "control", "screening_recipe": control}]
        screen_specs.extend(candidates)

        run_results = []
        copied_artifacts = []
        gpu_ids = [0, 1, 2, 3]

        for idx, idea in enumerate(screen_specs):
            recipe = idea["screening_recipe"]
            logger_rel = recipe["logger_name"]
            if not logger_rel.startswith("artifacts/"):
                logger_rel = f"artifacts/runs/{project_id}_{logger_rel}"
            recipe["logger_name"] = logger_rel
            run_dir = workdir / logger_rel
            train_log = run_dir / "train.log"
            eval_log = run_dir / "eval.log"

            metrics = self._parse_metrics_from_log(train_log) or self._parse_metrics_from_log(eval_log)
            status = "reused" if metrics else "pending"
            if not metrics:
                command = self._build_laps_train_command(
                    python_cmd=python_cmd,
                    data_path=data_path,
                    image_root=image_root,
                    logger_name=logger_rel,
                    gpu_id=gpu_ids[idx % len(gpu_ids)],
                    resume_path=recipe.get("resume"),
                    resume_strict=int(recipe.get("resume_strict", 1)),
                    learning_rate=float(recipe.get("learning_rate", 2e-4)),
                    batch_size=int(recipe.get("batch_size", 32)),
                    num_epochs=int(recipe.get("num_epochs", 1)),
                    changes=recipe.get("changes", {}),
                )
                command_result = self.runner.run_plan_commands(
                    [command],
                    workdir,
                    self.context.project_root / "experiment" / "logs" / f"{Path(logger_rel).name}_command.json",
                )
                status = command_result["status"]
                metrics = self._parse_metrics_from_log(train_log) or self._parse_metrics_from_log(eval_log)
            run_results.append(
                {
                    "id": idea["id"],
                    "title": idea["title"],
                    "direction": idea.get("direction", idea["id"]),
                    "run_dir": str(run_dir),
                    "train_log": str(train_log),
                    "metrics": metrics,
                    "status": status if metrics else "failed",
                }
            )

            if run_dir.exists():
                for rel_name in ["train.log", "Parameters.txt", "model_best.pth", "results_f30k.npy"]:
                    source = run_dir / rel_name
                    if source.exists():
                        copied = self.context.artifacts.copy_into_stage(
                            self.stage_key,
                            source,
                            f"screening/{Path(logger_rel).name}/{rel_name}",
                            artifact_type="itr_screen_artifact",
                            summary="Imported quick-screen artifact",
                        )
                        copied_artifacts.append(copied["path"])

        baseline = next((item for item in run_results if item["id"] == "control"), None)
        baseline_metrics = baseline.get("metrics") if baseline else None
        candidate_results = []
        for item in run_results:
            if item["id"] == "control":
                continue
            decision = self._screening_decision(item.get("metrics"), baseline_metrics)
            candidate_results.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "direction": item["direction"],
                    "status": item["status"],
                    "metrics": item.get("metrics"),
                    "decision": decision,
                    "train_log": item["train_log"],
                }
            )

        candidate_results.sort(
            key=lambda item: (
                0 if item["decision"] == "viable" else 1,
                -(item.get("metrics", {}) or {}).get("rsum", 0),
            )
        )

        if self.context.config.get("experiment", {}).get("cleanup_failed_idea_runs", True):
            failure_manager = FailureLogManager(self.context.config, external_root=workdir / "artifacts" / "runs")
            failure_manager.record_not_viable_ideas(
                project_id=project_id,
                baseline_metrics=baseline_metrics,
                candidate_results=candidate_results,
                cleanup=True,
            )

        summary_payload = {"baseline": baseline_metrics, "candidates": candidate_results}
        sources = [env_source, *copied_artifacts]
        if revision_source:
            sources.append(revision_source)
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            {
                "baseline_control": baseline_metrics,
                "candidate_results": candidate_results,
            },
            artifact_type="results",
            summary="Quick-screen results for RIS-driven image-text retrieval ideas",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            {"status": "pending", "note": "Quick-screen only; no full ablation package yet."},
            artifact_type="ablation",
            summary="Quick-screen ablation placeholder",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Hypothesis Verification",
                        "",
                        "- H1: quick-screen completed. Use the viable candidate list to decide the follow-up 3-epoch confirmation run.",
                        "- H2: ablation remains pending until one candidate is selected for confirmation.",
                    ]
                )
            ),
            artifact_type="verification",
            summary="Quick-screen hypothesis status",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            screening_summary_markdown(summary_payload),
            artifact_type="summary",
            summary="Idea screening summary",
            source_paths=[main_record["path"]],
        )
        ideas_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/idea_screening_summary.json",
            summary_payload,
            artifact_type="idea_screening",
            summary="Structured idea screening results",
            source_paths=[main_record["path"]],
        )
        viable = [item for item in candidate_results if item["decision"] == "viable"]
        blocked_reason = (
            f"Quick-screen completed; {len(viable)} viable idea(s) found. Review results and select one for a 3-epoch confirmation run."
            if viable
            else "Quick-screen completed but no candidate cleared the viability threshold."
        )
        return {
            "artifacts": [main_record["path"], ablation_record["path"], verification_record["path"], summary_record["path"], ideas_record["path"], *copied_artifacts],
            "status": "blocked",
            "blocked_reason": blocked_reason,
        }

    def _run_c2c_small_loop(
        self,
        plan: dict[str, Any],
        execution: dict[str, Any],
        env_source: str,
        revision_source: str | None,
    ) -> dict[str, Any]:
        adapter = C2CAdapter(self.context.project_root, self.context.config)
        candidates = list(plan.get("candidate_ideas") or default_c2c_ideas(plan.get("selected_idea", {}).get("title", "C2C"), adapter.baseline))
        candidates = candidates[: int(execution.get("max_candidates") or 3)]
        run_results = []
        command_logs = []
        copied_sources = [env_source]
        if revision_source:
            copied_sources.append(revision_source)

        simulate = bool(self.context.config.get("experiment", {}).get("simulate"))
        mock_results = bool(self.context.config.get("c2c", {}).get("small_loop", {}).get("mock_results"))
        baseline_mean = float((execution.get("baseline") or adapter.baseline).get("mean") or DEFAULT_BASELINE["mean"])
        min_delta = float(execution.get("min_delta_to_pass", self.context.config.get("c2c", {}).get("small_loop", {}).get("min_delta_to_pass", 0.1)))
        max_regression = float(execution.get("max_dataset_regression", self.context.config.get("c2c", {}).get("small_loop", {}).get("max_dataset_regression", 2.0)))
        original_files = self._snapshot_c2c_repo_state(adapter)
        gpu_policy = dict(execution.get("gpu_policy") or self.context.config.get("experiment", {}).get("gpu_policy", {}))
        if execution.get("selected_gpu_ids"):
            gpu_policy["gpu_ids"] = execution["selected_gpu_ids"]
        gpu_selection = self.runner.select_gpus(gpu_policy)

        for idx, candidate in enumerate(candidates):
            self._restore_c2c_repo_state(adapter, original_files)
            candidate_result = self._run_single_c2c_candidate(
                adapter=adapter,
                candidate=candidate,
                index=idx,
                simulate=simulate or mock_results,
                baseline_mean=baseline_mean,
                min_delta=min_delta,
                max_regression=max_regression,
                gpu_selection=gpu_selection,
            )
            run_results.append(candidate_result)
            command_logs.extend(candidate_result.get("command_logs", []))
            if candidate_result.get("decision") in {"failed_no_metrics", "patch_rejected", "proxy_rejected", "proxy_repairable"} and candidate_result.get("command_status") == "failed":
                break
            if candidate_result.get("decision") == "candidate_win":
                break
        self._restore_c2c_repo_state(adapter, original_files)

        best_proxy = self._best_c2c_proxy_candidate(run_results)
        best = self._best_c2c_candidate(run_results)
        comparison_candidate = best or best_proxy
        comparison = self._c2c_acceptance_comparison(comparison_candidate, execution.get("baseline") or adapter.baseline, min_delta, max_regression)
        main_payload = {
            "baseline": execution.get("baseline") or adapter.baseline,
            "best_candidate": best,
            "best_proxy_candidate": best_proxy,
            "candidate_results": run_results,
            "acceptance": comparison,
            "workflow_goal": self.context.config.get("c2c", {}).get("workflow_goal", "effect_first_discovery"),
            "gpu_selection": {
                "selected_gpu_ids": gpu_selection.selected_ids,
                "cuda_visible_devices": gpu_selection.cuda_visible_devices,
                "policy": gpu_selection.policy,
                "snapshot": gpu_selection.snapshot,
                "reason": gpu_selection.reason,
            },
        }
        main_payload["strong_reference_comparisons"] = self._c2c_strong_reference_comparisons(best, adapter)
        main_payload["paperization_readiness"] = _c2c_paperization_readiness(best, comparison)
        proxy_calibration = self._append_c2c_proxy_calibration(main_payload)
        main_payload["proxy_calibration"] = proxy_calibration.get("current_iteration")
        main_payload["proxy_calibration_summary"] = proxy_calibration.get("summary")
        history = self._append_c2c_iteration_history(main_payload)
        main_payload["iteration_history"] = {
            "path": "experiment/results/c2c_iteration_history.json",
            "entry_count": len(history.get("iterations", [])),
            "best_mean_so_far": history.get("best_mean_so_far"),
            "best_delta_so_far": history.get("best_delta_so_far"),
            "best_proxy_mean_so_far": history.get("best_proxy_mean_so_far"),
            "best_proxy_delta_so_far": history.get("best_proxy_delta_so_far"),
            "consecutive_not_viable": history.get("consecutive_not_viable"),
        }
        posthoc = None if comparison.get("passed") else self._c2c_posthoc_review(main_payload)
        main_payload["posthoc_review"] = posthoc or {"status": "skipped", "reason": "candidate accepted"}
        ablation_payload = self._c2c_ablation_payload(main_payload, adapter)
        main_payload["ablation_summary"] = {
            "status": ablation_payload.get("status"),
            "best_candidate_id": ablation_payload.get("best_candidate_id"),
            "best_supported": ablation_payload.get("best_supported"),
            "best_delta_enabled_vs_disabled": ablation_payload.get("best_delta_enabled_vs_disabled"),
        }
        main_payload = _compact_c2c_result_payload(main_payload)
        ablation_payload = _compact_c2c_result_payload(ablation_payload)
        sources = copied_sources
        main_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/main_results.json",
            main_payload,
            artifact_type="results",
            summary="C2C small-loop candidate results",
            source_paths=sources,
        )
        ablation_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/ablation_results.json",
            ablation_payload,
            artifact_type="ablation",
            summary="C2C automatic ablation results",
            source_paths=[main_record["path"]],
        )
        verification_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/hypothesis_verification.md",
            compact_markdown(self._c2c_verification_md(best, baseline_mean, run_results, min_delta, max_regression)),
            artifact_type="verification",
            summary="C2C hypothesis verification",
            source_paths=[main_record["path"]],
        )
        summary_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/summary.md",
            compact_markdown(self._c2c_summary_md(main_payload)),
            artifact_type="summary",
            summary="C2C small-loop summary",
            source_paths=[main_record["path"]],
        )
        loop_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/c2c_small_loop_results.json",
            main_payload,
            artifact_type="c2c_small_loop",
            summary="Structured C2C small-loop results",
            source_paths=[main_record["path"]],
        )
        posthoc_record = self.context.artifacts.write_json(
            self.stage_key,
            "results/posthoc_review.json",
            main_payload["posthoc_review"],
            artifact_type="c2c_posthoc_review",
            summary="GPT posthoc review of C2C training and evaluation outcome",
            source_paths=[main_record["path"]],
        )
        failure_analysis_record = self.context.artifacts.write_text(
            self.stage_key,
            "results/failure_analysis.md",
            compact_markdown(self._c2c_failure_analysis_md(main_payload, posthoc)),
            artifact_type="c2c_failure_analysis",
            summary="Failure analysis and next-round recommendations",
            source_paths=[main_record["path"], posthoc_record["path"]],
        )
        feedback_record = None
        if not comparison.get("passed"):
            feedback_record = self._write_c2c_failure_feedback(main_payload, artifacts=[main_record["path"], posthoc_record["path"]])
        self.context.artifacts.write_text(
            self.stage_key,
            "self_heal_log.jsonl",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in command_logs) + ("\n" if command_logs else ""),
            artifact_type="self_heal_log",
            summary="C2C command and patch trace",
        )
        if not best:
            blocked_reason = self._c2c_blocked_reason(run_results)
            if blocked_reason:
                return {
                    "artifacts": [
                        main_record["path"],
                        ablation_record["path"],
                        verification_record["path"],
                        summary_record["path"],
                        loop_record["path"],
                        posthoc_record["path"],
                        failure_analysis_record["path"],
                        *([feedback_record["path"]] if feedback_record else []),
                    ],
                    "status": "blocked",
                    "blocked_reason": blocked_reason,
                }
        return {
            "artifacts": [
                main_record["path"],
                ablation_record["path"],
                verification_record["path"],
                summary_record["path"],
                loop_record["path"],
                posthoc_record["path"],
                failure_analysis_record["path"],
                *([feedback_record["path"]] if feedback_record else []),
            ],
            "status": "ok" if comparison.get("passed") else "not_viable",
        }

    def _append_c2c_iteration_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        history_path = self.context.project_root / "experiment" / "results" / "c2c_iteration_history.json"
        history = read_json(
            history_path,
            default={
                "schema_version": "c2c_iteration_history_v1",
                "project_id": self.context.project_root.name,
                "iterations": [],
            },
        )
        iterations = [
            item
            for item in history.get("iterations", [])
            if isinstance(item, dict) and int(item.get("iteration") or -1) != self._registry_iteration()
        ]
        acceptance = payload.get("acceptance") or {}
        best = payload.get("best_candidate") or {}
        best_proxy = payload.get("best_proxy_candidate") or {}
        best_metrics = best.get("metrics") or {}
        best_proxy_screen = best_proxy.get("proxy_screen") or {}
        best_proxy_metrics = best_proxy_screen.get("metrics") or {}
        entry = {
            "timestamp": now_utc(),
            "iteration": self._registry_iteration(),
            "accepted": bool(acceptance.get("passed")),
            "acceptance": {
                "passed": acceptance.get("passed"),
                "reason": acceptance.get("reason"),
                "baseline_mean": acceptance.get("baseline_mean"),
                "best_mean": acceptance.get("best_mean"),
                "delta": acceptance.get("delta"),
                "proxy_best_mean": acceptance.get("proxy_best_mean"),
                "proxy_delta": acceptance.get("proxy_delta"),
                "proxy_score": acceptance.get("proxy_score"),
                "proxy_worst_dataset_regression": acceptance.get("proxy_worst_dataset_regression"),
                "worst_dataset_regression": acceptance.get("worst_dataset_regression"),
                "min_delta_to_pass": acceptance.get("min_delta_to_pass"),
                "max_dataset_regression": acceptance.get("max_dataset_regression"),
            },
            "best_candidate": {
                "id": best.get("id"),
                "title": best.get("title"),
                "decision": best.get("decision"),
                "metrics": best_metrics,
                "delta_vs_baseline": best.get("delta_vs_baseline"),
                "dataset_regressions": best.get("dataset_regressions") or {},
                "worst_dataset_regression": best.get("worst_dataset_regression"),
            },
            "best_proxy_candidate": {
                "id": best_proxy.get("id"),
                "title": best_proxy.get("title"),
                "decision": best_proxy.get("decision"),
                "proxy_metrics": best_proxy_metrics,
                "proxy_mean": best_proxy_metrics.get("mean"),
                "proxy_delta_vs_baseline": best_proxy_screen.get("proxy_delta_vs_baseline"),
                "proxy_score": best_proxy_screen.get("proxy_score"),
                "proxy_worst_dataset_regression": best_proxy_screen.get("proxy_worst_dataset_regression"),
                "proxy_dataset_deltas": best_proxy_screen.get("proxy_dataset_deltas") or {},
            },
            "candidate_count": len(payload.get("candidate_results") or []),
            "candidate_ids": [
                item.get("id")
                for item in payload.get("candidate_results") or []
                if isinstance(item, dict) and item.get("id")
            ],
        }
        iterations.append(entry)
        iterations.sort(key=lambda item: int(item.get("iteration") or 0))
        best_entries = [
            item
            for item in iterations
            if ((item.get("best_candidate") or {}).get("metrics") or {}).get("mean") is not None
        ]
        best_proxy_entries = [
            item
            for item in iterations
            if ((item.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean") is not None
        ]
        best_entry = max(
            best_entries,
            key=lambda item: float(((item.get("best_candidate") or {}).get("metrics") or {}).get("mean")),
            default=None,
        )
        best_proxy_entry = max(
            best_proxy_entries,
            key=lambda item: float(((item.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean")),
            default=None,
        )
        consecutive_not_viable = 0
        for item in reversed(iterations):
            if item.get("accepted"):
                break
            consecutive_not_viable += 1
        history = {
            "schema_version": "c2c_iteration_history_v1",
            "project_id": self.context.project_root.name,
            "updated_at": now_utc(),
            "iterations": iterations,
            "iteration_count": len(iterations),
            "best_iteration": best_entry.get("iteration") if best_entry else None,
            "best_candidate_id": (best_entry.get("best_candidate") or {}).get("id") if best_entry else None,
            "best_mean_so_far": ((best_entry.get("best_candidate") or {}).get("metrics") or {}).get("mean") if best_entry else None,
            "best_delta_so_far": (best_entry.get("best_candidate") or {}).get("delta_vs_baseline") if best_entry else None,
            "best_proxy_iteration": best_proxy_entry.get("iteration") if best_proxy_entry else None,
            "best_proxy_candidate_id": (best_proxy_entry.get("best_proxy_candidate") or {}).get("id") if best_proxy_entry else None,
            "best_proxy_mean_so_far": ((best_proxy_entry.get("best_proxy_candidate") or {}).get("proxy_metrics") or {}).get("mean") if best_proxy_entry else None,
            "best_proxy_delta_so_far": (best_proxy_entry.get("best_proxy_candidate") or {}).get("proxy_delta_vs_baseline") if best_proxy_entry else None,
            "consecutive_not_viable": consecutive_not_viable,
        }
        write_json(history_path, history)
        return history

    def _append_c2c_proxy_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        calibration_path = self.context.project_root / "experiment" / "results" / "proxy_calibration.json"
        calibration = read_json(
            calibration_path,
            default={
                "schema_version": "c2c_proxy_calibration_v1",
                "project_id": self.context.project_root.name,
                "iterations": [],
            },
        )
        if not isinstance(calibration, dict):
            calibration = {"schema_version": "c2c_proxy_calibration_v1", "project_id": self.context.project_root.name, "iterations": []}
        iterations = [
            item
            for item in calibration.get("iterations", [])
            if isinstance(item, dict) and int(item.get("iteration") or -1) != self._registry_iteration()
        ]
        current = _c2c_proxy_calibration_iteration(payload, iteration=self._registry_iteration())
        if current.get("candidate_count"):
            iterations.append(current)
        iterations.sort(key=lambda item: int(item.get("iteration") or 0))
        calibration = {
            "schema_version": "c2c_proxy_calibration_v1",
            "project_id": self.context.project_root.name,
            "updated_at": now_utc(),
            "iterations": iterations[-50:],
            "summary": _c2c_proxy_calibration_summary(iterations[-50:]),
            "current_iteration": current,
        }
        write_json(calibration_path, calibration)
        return calibration

    @staticmethod
    def _snapshot_c2c_core_files(adapter: C2CAdapter) -> dict[str, str]:
        return {key: value["content"] for key, value in ExperimentAgent._snapshot_c2c_repo_state(adapter).items() if value.get("existed")}

    @staticmethod
    def _restore_c2c_core_files(adapter: C2CAdapter, originals: dict[str, str]) -> None:
        for rel_path, content in originals.items():
            path = adapter.repo_root / rel_path
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _snapshot_c2c_repo_state(adapter: C2CAdapter) -> dict[str, dict[str, Any]]:
        policy = DynamicEditPolicy.from_config(adapter.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        originals: dict[str, dict[str, Any]] = {}
        for path in adapter.repo_root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(adapter.repo_root).as_posix()
            if not policy.allowed(rel_path, repo_root=adapter.repo_root):
                continue
            originals[rel_path] = {"existed": True, "content": path.read_text(encoding="utf-8", errors="ignore")}
        return originals

    @staticmethod
    def _restore_c2c_repo_state(adapter: C2CAdapter, originals: dict[str, dict[str, Any]]) -> None:
        policy = DynamicEditPolicy.from_config(adapter.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        for path in sorted(adapter.repo_root.rglob("*"), reverse=True):
            if not path.is_file():
                continue
            rel_path = path.relative_to(adapter.repo_root).as_posix()
            if policy.allowed(rel_path, repo_root=adapter.repo_root) and rel_path not in originals:
                path.unlink()
        for rel_path, item in originals.items():
            target = adapter.repo_root / rel_path
            ensure_dir(target.parent)
            target.write_text(str(item.get("content") or ""), encoding="utf-8")

    def _run_single_c2c_candidate(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        index: int,
        simulate: bool,
        baseline_mean: float,
        min_delta: float,
        max_regression: float,
        gpu_selection: Any,
    ) -> dict[str, Any]:
        original_repo_state = self._snapshot_c2c_repo_state(adapter)
        patch = self._load_frozen_c2c_patch(candidate)
        patch_result = self._apply_frozen_c2c_patch(candidate, adapter, patch)
        code_snapshot = archive_patched_code_snapshot(self.context.artifacts, adapter, candidate, patch_result) if patch_result.get("status") == "applied" else {"status": "skipped"}
        run_spec = adapter.materialize_candidate_configs(candidate, gpu_selection)
        has_executable_change = bool(patch_result.get("changed_files") or run_spec.get("has_executable_change"))
        patch_fingerprint = self._c2c_patch_fingerprint(adapter, patch_result, run_spec)
        reusable_state = self._load_reusable_c2c_proxy_state(run_spec, patch_fingerprint)
        run_state = {
            "candidate_id": candidate.get("id"),
            "run_id": run_spec["run_id"],
            "created_at": now_utc(),
            "preflight": None,
            "proxy_screen": None,
            "ablation": None,
            "train": None,
            "eval_by_dataset": {},
            "ablation_eval_by_dataset": {},
            "metrics": None,
            "ablation_metrics": None,
            "attempts": [],
            "recovery_actions": [],
            "frozen_hashes": run_spec.get("frozen_hashes", {}),
            "config_overrides": run_spec.get("config_overrides", {}),
            "has_executable_change": has_executable_change,
            "patch_fingerprint": patch_fingerprint,
            "code_snapshot": code_snapshot,
            "runtime_localization": run_spec.get("runtime_localization", {}),
        }
        logs = [
            {
                "candidate_id": candidate.get("id"),
                "event": "patch",
                "patch_summary": patch.get("summary", ""),
                "patch_result": patch_result,
                "code_snapshot": code_snapshot,
                "config_overrides": run_spec.get("config_overrides", {}),
                "has_executable_change": has_executable_change,
                "patch_fingerprint": patch_fingerprint,
                "frozen_hashes": run_spec.get("frozen_hashes", {}),
                "runtime_localization": run_spec.get("runtime_localization", {}),
            }
        ]
        command_status = "skipped"
        preflight = None
        if patch_result["status"] == "rejected":
            metrics = None
            command_status = "patch_rejected"
            self._save_c2c_run_state(run_spec, run_state)
        elif not simulate and not has_executable_change:
            metrics = None
            command_status = "blocked"
            reason = "candidate lacks frozen executable patch or config_overrides; refusing deterministic S3 no-op run"
            run_state["preflight"] = {"status": "blocked", "reason": reason}
            run_state["recovery_actions"].append({"action": "block_noop_candidate", "status": "blocked", "reason": reason})
            logs.append({"candidate_id": candidate.get("id"), "event": "blocked", "reason": reason})
            self._save_c2c_run_state(run_spec, run_state)
        elif simulate:
            metrics = adapter.write_mock_candidate_results(run_spec["run_id"], offset=max(min_delta + 0.15, 0.25) if index == 0 else -0.25)
            ablation = self._run_c2c_ablation_eval(
                adapter=adapter,
                candidate=candidate,
                run_spec=run_spec,
                gpu_selection=gpu_selection,
                retry_policy={"max_attempts": 1},
                simulate=True,
            )
            logs.append({"candidate_id": candidate.get("id"), "event": "mock_results", "metrics": metrics})
            logs.append({"candidate_id": candidate.get("id"), "event": "ablation", "status": ablation.get("status"), "ablation": ablation})
            command_status = "mocked"
            run_state["preflight"] = {"status": "skipped", "simulate": True}
            run_state["proxy_screen"] = {"enabled": False, "status": "skipped", "reason": "simulate mode"}
            run_state["metrics"] = metrics
            run_state["ablation"] = ablation
            run_state["ablation_metrics"] = ablation.get("metrics")
            self._save_c2c_run_state(run_spec, run_state)
        elif reusable_state:
            metrics = reusable_state.get("metrics")
            proxy_screen = reusable_state.get("proxy_screen") or {}
            preflight = reusable_state.get("preflight")
            run_state.update(reusable_state)
            run_state["patch_fingerprint"] = patch_fingerprint
            run_state["code_snapshot"] = code_snapshot
            logs.append(
                {
                    "candidate_id": candidate.get("id"),
                    "event": "reuse_proxy_screen",
                    "status": proxy_screen.get("status"),
                    "reason": "existing run_state has complete proxy_screen for matching frozen hashes and patch fingerprint",
                }
            )
            if proxy_screen.get("status") == "rejected":
                command_status = "proxy_rejected"
            elif proxy_screen.get("status") == "repairable_proxy_risk":
                command_status = "proxy_repairable"
            else:
                command_status = str(reusable_state.get("command_status") or "partial")
            self._save_c2c_run_state(run_spec, run_state)
        else:
            metrics = None
            preflight = adapter.preflight(run_spec, gpu_selection)
            run_state["preflight"] = preflight
            run_state["recovery_actions"].extend(preflight.get("recovery_actions", []))
            logs.append({"candidate_id": candidate.get("id"), "event": "preflight", "status": preflight.get("status"), "path": str(run_spec["preflight_path"])})
            if preflight.get("status") == "blocked":
                command_status = "blocked"
                self._save_c2c_run_state(run_spec, run_state)
            else:
                log_path = self.context.project_root / "experiment" / "logs" / f"c2c_{run_spec['run_id']}_commands.json"
                ensure_dir(log_path.parent)
                retry_policy = {
                    "max_attempts": int(self.context.config.get("experiment", {}).get("self_heal", {}).get("max_attempts", 1) or 1)
                }
                step_runs = []
                for step_idx, command in enumerate(run_spec["commands"]["preflight"]):
                    result = self.runner.run_step(
                        name=f"preflight_command_{step_idx}",
                        command=command,
                        working_dir=adapter.repo_root,
                        retry_policy={"max_attempts": 1},
                    )
                    step_runs.append(result)
                    run_state["attempts"].append(result)
                    if result["status"] != "ok":
                        command_status = "failed"
                        break
                if command_status != "failed":
                    baseline_proxy = self._ensure_c2c_proxy_baseline(
                        adapter,
                        run_spec,
                        gpu_selection,
                        retry_policy,
                        baseline_repo_state=original_repo_state,
                    )
                    if baseline_proxy.get("status") in {"failed", "blocked"}:
                        step_runs.extend(baseline_proxy.get("attempts") or [])
                        run_state["attempts"].extend(baseline_proxy.get("attempts") or [])
                        run_state["proxy_baseline"] = baseline_proxy
                        command_status = "failed"
                        logs.append({"candidate_id": candidate.get("id"), "event": "proxy_baseline", "status": baseline_proxy.get("status"), "proxy_baseline": baseline_proxy})
                if command_status != "failed":
                    proxy_screen = self._run_c2c_proxy_screen(
                        adapter=adapter,
                        candidate=candidate,
                        run_spec=run_spec,
                        patch_result=patch_result,
                        has_executable_change=has_executable_change,
                        baseline=adapter.baseline,
                    )
                    proxy_screen["patch_fingerprint"] = patch_fingerprint
                    run_state["proxy_screen"] = proxy_screen
                    logs.append({"candidate_id": candidate.get("id"), "event": "proxy_screen", "status": proxy_screen.get("status"), "proxy_screen": proxy_screen})
                    proxy_attempts = proxy_screen.get("attempts") or []
                    step_runs.extend(proxy_attempts)
                    run_state["attempts"].extend(proxy_attempts)
                    if proxy_screen.get("status") == "failed":
                        command_status = "failed"
                    elif proxy_screen.get("status") == "rejected":
                        command_status = "proxy_rejected"
                    elif proxy_screen.get("status") == "repairable_proxy_risk":
                        command_status = "proxy_repairable"
                if command_status not in {"failed", "proxy_rejected", "proxy_repairable"}:
                    train_result = self.runner.run_step(
                        name="train",
                        command=run_spec["commands"]["train"],
                        working_dir=adapter.repo_root,
                        retry_policy=retry_policy,
                    )
                    step_runs.append(train_result)
                    run_state["train"] = train_result
                    run_state["attempts"].append(train_result)
                    if train_result["status"] != "ok":
                        oom_action = self._c2c_oom_recovery_hint(train_result)
                        if oom_action and len(getattr(gpu_selection, "selected_ids", []) or []) > 1:
                            recovery_gpu_ids = [gpu_selection.selected_ids[0]]
                            recovery_command = adapter._candidate_commands(run_spec["train_config"], run_spec["eval_configs"], recovery_gpu_ids)["train"]
                            recovery_result = self.runner.run_step(
                                name="train_recovery_reduced_concurrency",
                                command=recovery_command,
                                working_dir=adapter.repo_root,
                                retry_policy={"max_attempts": 1},
                            )
                            step_runs.append(recovery_result)
                            run_state["attempts"].append(recovery_result)
                            oom_action.update({"recovery_gpu_ids": recovery_gpu_ids, "recovery_status": recovery_result["status"]})
                            run_state["recovery_actions"].append(oom_action)
                            if recovery_result["status"] == "ok":
                                train_result = recovery_result
                                run_state["train"] = recovery_result
                        if self._c2c_checkpoint_final_exists(run_spec):
                            command_status = "partial"
                            action = {"action": "skip_failed_train_with_existing_final_checkpoint", "status": "ok", "run_id": run_spec["run_id"]}
                            run_state["recovery_actions"].append(action)
                            logs.append({"candidate_id": candidate.get("id"), "event": "recovery", **action})
                        elif train_result["status"] != "ok":
                            command_status = "failed"
                    if command_status != "failed":
                        eval_commands = run_spec["commands"]["eval"]
                        eval_items = list(run_spec.get("eval_configs", {}).items())
                        for eval_idx, command in enumerate(eval_commands):
                            dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
                            result = self.runner.run_step(
                                name=f"eval_{dataset}",
                                command=command,
                                working_dir=adapter.repo_root,
                                retry_policy=retry_policy,
                            )
                            step_runs.append(result)
                            run_state["eval_by_dataset"][dataset] = result
                            run_state["attempts"].append(result)
                            if result["status"] != "ok":
                                command_status = "partial"
                    metrics = adapter.collect_candidate_metrics(run_spec["run_id"])
                    run_state["metrics"] = metrics
                    if metrics is not None and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                        ablation = self._run_c2c_ablation_eval(
                            adapter=adapter,
                            candidate=candidate,
                            run_spec=run_spec,
                            gpu_selection=gpu_selection,
                            retry_policy=retry_policy,
                            simulate=False,
                        )
                        run_state["ablation"] = ablation
                        run_state["ablation_metrics"] = ablation.get("metrics")
                        run_state["ablation_eval_by_dataset"] = ablation.get("eval_by_dataset") or {}
                        run_state["attempts"].extend(ablation.get("attempts") or [])
                        step_runs.extend(ablation.get("attempts") or [])
                        logs.append({"candidate_id": candidate.get("id"), "event": "ablation", "status": ablation.get("status"), "ablation": ablation})
                        if ablation.get("status") == "partial" and command_status == "ok":
                            command_status = "partial"
                    if command_status == "skipped":
                        command_status = "ok"
                    if metrics is None and command_status not in {"failed", "blocked", "proxy_rejected", "proxy_repairable"}:
                        command_status = "partial"
                write_json(
                    log_path,
                    {
                        "status": command_status,
                        "runs": _compact_attempts(step_runs, stdout_chars=4000, stderr_chars=4000),
                        "full_log_note": "stdout/stderr are stored as bounded tails to keep artifacts parseable",
                    },
                )
                logs.append({"candidate_id": candidate.get("id"), "event": "commands", "log_path": str(log_path), "status": command_status})
                self._save_c2c_run_state(run_spec, run_state)
        mean = (metrics or {}).get("mean")
        ablation_result = run_state.get("ablation") or {"enabled": False, "status": "skipped", "reason": "not run"}
        dataset_regressions = self._c2c_dataset_regressions(metrics, adapter.baseline)
        worst_regression = max(dataset_regressions.values()) if dataset_regressions else 0.0
        ablation_comparison = (ablation_result.get("comparison") or {}) if isinstance(ablation_result, dict) else {}
        require_ablation_support = bool(
            self.context.config.get("c2c", {}).get("small_loop", {}).get("require_ablation_support", False)
        )
        mechanism_supported = bool(ablation_comparison.get("mechanism_supported"))
        decision = (
            "candidate_win"
            if mean is not None and float(mean) >= baseline_mean + min_delta and worst_regression <= max_regression
            and (not require_ablation_support or mechanism_supported)
            else "not_viable"
        )
        if patch_result["status"] == "rejected":
            decision = "patch_rejected"
        elif preflight and preflight.get("status") == "blocked":
            decision = "blocked"
        elif command_status == "blocked":
            decision = "blocked"
        elif command_status == "proxy_rejected":
            decision = "proxy_rejected"
        elif command_status == "proxy_repairable":
            decision = "proxy_repairable"
        elif metrics is None:
            decision = "failed_no_metrics" if command_status == "failed" else "partial"
        result = {
            "id": candidate.get("id"),
            "title": candidate.get("title"),
            "hypothesis": candidate.get("hypothesis"),
            "mechanism_type": candidate.get("mechanism_type"),
            "run_id": run_spec["run_id"],
            "run_root": str(run_spec["run_root"]),
            "patch_result": _compact_patch_result_for_payload(patch_result),
            "code_snapshot": code_snapshot,
            "commands": _compact_command_plan(run_spec.get("commands") or {}),
            "command_status": command_status,
            "preflight": preflight,
            "proxy_screen": _compact_proxy_screen(run_state.get("proxy_screen")),
            "run_state_path": str(run_spec["run_state_path"]),
            "preflight_path": str(run_spec["preflight_path"]),
            "config_overrides": run_spec.get("config_overrides", {}),
            "runtime_localization": run_spec.get("runtime_localization", {}),
            "has_executable_change": has_executable_change,
            "patch_fingerprint": patch_fingerprint,
            "frozen_hashes": run_spec.get("frozen_hashes", {}),
            "metrics": metrics,
            "ablation": ablation_result,
            "delta_vs_baseline": round(float(mean) - baseline_mean, 4) if mean is not None else None,
            "dataset_regressions": dataset_regressions,
            "worst_dataset_regression": worst_regression,
            "acceptance_rule": {
                "min_delta_to_pass": min_delta,
                "max_dataset_regression": max_regression,
                "baseline_mean": baseline_mean,
                "require_ablation_support": require_ablation_support,
            },
            "mechanism_supported": mechanism_supported,
            "decision": decision,
            "command_logs": _compact_event_logs(logs),
        }
        result["failure_attribution"] = self._c2c_failure_attribution(result, adapter.baseline)
        return result

    @staticmethod
    def _c2c_patch_fingerprint(adapter: C2CAdapter, patch_result: dict[str, Any], run_spec: dict[str, Any]) -> str:
        changed_files = sorted(set(str(item) for item in (patch_result or {}).get("changed_files") or [] if item))
        file_hashes: dict[str, str] = {}
        for rel_path in changed_files:
            path = adapter.repo_root / rel_path
            file_hashes[rel_path] = sha256_file(path) if path.exists() and path.is_file() else "<missing>"
        payload = {
            "frozen_hashes": run_spec.get("frozen_hashes") or {},
            "patch_status": (patch_result or {}).get("status"),
            "patch_changed_files": changed_files,
            "patch_file_hashes": file_hashes,
        }
        return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))

    @staticmethod
    def _load_reusable_c2c_proxy_state(run_spec: dict[str, Any], patch_fingerprint: str | None = None) -> dict[str, Any] | None:
        state_path = Path(run_spec.get("run_state_path") or "")
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(state, dict):
            return None
        if state.get("frozen_hashes") != run_spec.get("frozen_hashes"):
            return None
        if not patch_fingerprint or state.get("patch_fingerprint") != patch_fingerprint:
            return None
        proxy_screen = state.get("proxy_screen") if isinstance(state.get("proxy_screen"), dict) else {}
        if proxy_screen.get("patch_fingerprint") != patch_fingerprint:
            return None
        if proxy_screen.get("status") not in {"rejected", "repairable_proxy_risk"}:
            return None
        if ((proxy_screen.get("metrics") or {}).get("mean")) is None:
            return None
        return state

    def _generate_c2c_patch(self, candidate: dict[str, Any], adapter: C2CAdapter) -> dict[str, Any]:
        del adapter
        return self._load_frozen_c2c_patch(candidate)

    def _load_frozen_c2c_patch(self, candidate: dict[str, Any]) -> dict[str, Any]:
        code_patch = candidate.get("code_patch") if isinstance(candidate.get("code_patch"), dict) else {}
        if code_patch and code_patch.get("status") != "ok":
            status = code_patch.get("status", "unknown")
            reason = code_patch.get("reason") or f"S2.5 code patch status is {status}; candidate is not eligible for deterministic S3."
            return {
                "summary": reason,
                "operations": [],
                "changed_files": [],
                "status": status,
                "fatal_patch_status": True,
            }
        if code_patch.get("status") != "ok" or not code_patch.get("patch_json"):
            return {"summary": "No valid frozen S2.5 patch attached to candidate.", "operations": [], "changed_files": [], "status": code_patch.get("status", "missing")}
        patch_path = self.context.project_root / str(code_patch["patch_json"])
        if not patch_path.exists():
            return {"summary": "Frozen S2.5 patch file is missing.", "operations": [], "changed_files": [], "status": "missing"}
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        validation_path = code_patch.get("validation")
        if validation_path:
            validation_file = self.context.project_root / str(validation_path)
            if validation_file.exists():
                try:
                    patch["validation"] = json.loads(validation_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    patch["validation"] = {"status": "invalid", "reason": f"Could not parse {validation_path}"}
        patch.setdefault("summary", patch.get("rationale", "Frozen S2.5 patch."))
        return patch

    def _apply_frozen_c2c_patch(self, candidate: dict[str, Any], adapter: C2CAdapter, patch: dict[str, Any]) -> dict[str, Any]:
        if patch.get("fatal_patch_status"):
            return {
                "status": "rejected",
                "errors": [patch.get("summary", "S2.5 patch is not eligible for S3.")],
                "changed_files": [],
                "reason": patch.get("summary", "S2.5 patch is not eligible for S3."),
                "candidate_id": candidate.get("id"),
                "patch_status": patch.get("status", "unknown"),
            }
        if not patch.get("operations"):
            return {"status": "skipped", "errors": [], "changed_files": [], "reason": patch.get("summary", "no frozen patch operations")}
        policy = DynamicEditPolicy.from_config(self.context.config.get("code_patch", {}).get("dynamic_whitelist") or {})
        result = FrozenPatchGuard(policy).apply(adapter.repo_root, patch)
        result["candidate_id"] = candidate.get("id")
        result["patch_status"] = patch.get("status", "ok")
        for key in ["risk_check", "activation_check", "mechanism_review", "quality_score", "validation"]:
            if isinstance(patch.get(key), dict):
                result[key] = patch[key]
        return result

    @staticmethod
    def _c2c_allowed_file_snippets(adapter: C2CAdapter, *, max_chars: int = 6000) -> list[dict[str, str]]:
        snippets = []
        for rel_path in adapter.allowed_files:
            path = adapter.repo_root / rel_path
            if path.exists():
                snippets.append({"path": rel_path, "text": path.read_text(encoding="utf-8", errors="ignore")[:max_chars]})
        return snippets

    @staticmethod
    def _save_c2c_run_state(run_spec: dict[str, Any], run_state: dict[str, Any]) -> None:
        run_state["updated_at"] = now_utc()
        write_json(Path(run_spec["run_state_path"]), run_state)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): ExperimentAgent._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ExperimentAgent._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [ExperimentAgent._jsonable(item) for item in value]
        return value

    @staticmethod
    def _c2c_checkpoint_final_exists(run_spec: dict[str, Any]) -> bool:
        final_path = Path(run_spec["run_root"]) / "checkpoints" / "final"
        return final_path.exists() and any(final_path.iterdir()) if final_path.is_dir() else final_path.exists()

    def _run_c2c_proxy_screen(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        patch_result: dict[str, Any],
        has_executable_change: bool,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        if not proxy_cfg.get("enabled", False):
            return {"enabled": False, "status": "skipped", "reason": "proxy_screen disabled"}

        proxy = self._c2c_static_proxy_screen(
            candidate=candidate,
            run_spec=run_spec,
            patch_result=patch_result,
            has_executable_change=has_executable_change,
        )
        if proxy["status"] in {"rejected", "repairable_proxy_risk"}:
            return proxy

        attempts: list[dict[str, Any]] = []
        rendered_commands = self._c2c_proxy_commands(adapter, run_spec, proxy_cfg)
        if rendered_commands:
            for idx, command in enumerate(rendered_commands):
                result = self.runner.run_step(
                    name=f"proxy_command_{idx}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={"max_attempts": 1, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, idx, rendered_commands)},
                )
                attempts.append(result)
                if result.get("status") != "ok" and proxy_cfg.get("reject_on_command_failure", True):
                    command_failure = self._c2c_proxy_command_failure(result)
                    reason = f"proxy command {idx} failed: {command_failure.get('summary')}"
                    repair_hint = command_failure.get("repair_hint")
                    proxy.update(
                        {
                            "status": "repairable_proxy_risk",
                            "reason": reason,
                            "repair_hint": repair_hint,
                            "repair_route": "S2_plan",
                            "repair_mode": "effect_first_proxy_repair",
                            "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                                reason=reason,
                                repair_hint=repair_hint,
                                evidence={"command_failure": command_failure},
                                patch_risk=proxy.get("patch_risk") or {},
                                source="proxy_command",
                            ),
                            "command_failure": command_failure,
                            "attempts": attempts,
                            "commands": rendered_commands,
                        }
                    )
                    return proxy

        metrics = adapter.collect_proxy_metrics(run_spec)
        baseline_metrics = adapter.proxy_baseline_metrics(run_spec)
        proxy["attempts"] = attempts
        proxy["commands"] = rendered_commands
        proxy["metrics"] = metrics
        proxy["baseline_metrics"] = baseline_metrics
        threshold_decision = self._c2c_proxy_metric_decision(
            metrics=metrics,
            baseline=baseline,
            proxy_cfg=proxy_cfg,
            proxy_baseline=baseline_metrics,
            patch_risk=proxy.get("patch_risk") or {},
        )
        if threshold_decision:
            proxy.update(threshold_decision)
            return proxy
        proxy["status"] = "passed"
        proxy["reason"] = proxy.get("reason") or "static and optional command proxy checks passed"
        return proxy

    @staticmethod
    def _c2c_proxy_command_timeout(proxy_cfg: dict[str, Any], idx: int, commands: list[str]) -> int | None:
        def coerce(value: Any) -> int | None:
            try:
                timeout = int(value)
            except (TypeError, ValueError):
                return None
            return timeout if timeout > 0 else None

        if idx < 0:
            return coerce(proxy_cfg.get("preflight_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        command = commands[idx] if 0 <= idx < len(commands) else ""
        lowered = str(command).lower()
        if "script/evaluation/" in lowered or "unified_evaluator.py" in lowered or "eval_" in lowered:
            return coerce(proxy_cfg.get("eval_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        if "script/train/" in lowered or "sft_train.py" in lowered or "torch.distributed" in lowered:
            return coerce(proxy_cfg.get("train_timeout_seconds")) or coerce(proxy_cfg.get("command_timeout_seconds"))
        return coerce(proxy_cfg.get("command_timeout_seconds"))

    def _run_c2c_ablation_eval(
        self,
        *,
        adapter: C2CAdapter,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        gpu_selection: Any,
        retry_policy: dict[str, Any],
        simulate: bool,
    ) -> dict[str, Any]:
        ablation_spec = adapter.materialize_ablation_eval_configs(candidate, run_spec, gpu_selection)
        if not ablation_spec.get("enabled"):
            return ablation_spec
        if simulate:
            metrics = adapter.write_mock_ablation_results(run_spec, offset=-0.05)
            enabled_metrics = adapter.collect_candidate_metrics(run_spec["run_id"])
            comparison = self._c2c_ablation_comparison(enabled_metrics, metrics)
            return self._jsonable({
                **ablation_spec,
                "status": "mocked",
                "metrics": metrics,
                "comparison": comparison,
                "attempts": [],
                "eval_by_dataset": {},
            })
        attempts: list[dict[str, Any]] = []
        eval_by_dataset: dict[str, Any] = {}
        eval_items = list((ablation_spec.get("eval_configs") or {}).items())
        status = "ok"
        for eval_idx, command in enumerate((ablation_spec.get("commands") or {}).get("eval") or []):
            dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
            result = self.runner.run_step(
                name=f"ablation_eval_{dataset}",
                command=command,
                working_dir=adapter.repo_root,
                retry_policy=retry_policy,
            )
            attempts.append(result)
            eval_by_dataset[dataset] = result
            if result.get("status") != "ok":
                status = "partial"
        metrics = adapter.collect_ablation_metrics(run_spec, ablation_spec)
        if metrics:
            write_json(Path(ablation_spec["metrics_path"]), metrics)
        else:
            status = "partial"
        comparison = self._c2c_ablation_comparison(adapter.collect_candidate_metrics(run_spec["run_id"]), metrics)
        return self._jsonable({
            **ablation_spec,
            "status": status,
            "metrics": metrics,
            "comparison": comparison,
            "attempts": attempts,
            "eval_by_dataset": eval_by_dataset,
        })

    @staticmethod
    def _c2c_ablation_comparison(enabled_metrics: dict[str, Any] | None, disabled_metrics: dict[str, Any] | None) -> dict[str, Any]:
        if not enabled_metrics or not disabled_metrics:
            return {"status": "insufficient_metrics", "enabled_mean": (enabled_metrics or {}).get("mean"), "disabled_mean": (disabled_metrics or {}).get("mean")}
        enabled_mean = enabled_metrics.get("mean")
        disabled_mean = disabled_metrics.get("mean")
        mean_delta = round(float(enabled_mean) - float(disabled_mean), 4) if enabled_mean is not None and disabled_mean is not None else None
        dataset_deltas = ExperimentAgent._c2c_dataset_deltas(enabled_metrics, disabled_metrics)
        return {
            "status": "ok",
            "enabled_mean": enabled_mean,
            "disabled_mean": disabled_mean,
            "enabled_minus_disabled_mean": mean_delta,
            "dataset_enabled_minus_disabled": dataset_deltas,
            "mechanism_supported": mean_delta is not None and mean_delta > 0,
        }

    @staticmethod
    def _c2c_ablation_payload(payload: dict[str, Any], adapter: C2CAdapter) -> dict[str, Any]:
        candidate_entries: list[dict[str, Any]] = []
        for candidate in payload.get("candidate_results") or []:
            if not isinstance(candidate, dict):
                continue
            ablation = candidate.get("ablation") or {}
            comparison = ablation.get("comparison") or {}
            contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
            ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
            declared_switch = ablation.get("switch") or contract.get("ablation_switch") or ablation_plan.get("switch")
            candidate_entries.append(
                {
                    "candidate_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "decision": candidate.get("decision"),
                    "command_status": candidate.get("command_status"),
                    "enabled": bool(ablation.get("enabled")),
                    "status": ablation.get("status", "missing"),
                    "switch": declared_switch,
                    "declared_switch": declared_switch,
                    "reached_ablation_stage": bool(ablation.get("comparison") or ablation.get("metrics") or ablation.get("attempts")),
                    "enabled_metrics": candidate.get("metrics"),
                    "disabled_metrics": ablation.get("metrics"),
                    "comparison": comparison,
                    "supported": comparison.get("mechanism_supported"),
                    "delta_enabled_vs_disabled": comparison.get("enabled_minus_disabled_mean"),
                    "dataset_enabled_minus_disabled": comparison.get("dataset_enabled_minus_disabled") or {},
                    "eval_configs": ablation.get("eval_configs") or {},
                    "metrics_path": ablation.get("metrics_path"),
                    "attempt_statuses": [
                        {
                            "step": attempt.get("step"),
                            "status": attempt.get("status"),
                            "returncode": attempt.get("returncode"),
                        }
                        for attempt in ablation.get("attempts") or []
                        if isinstance(attempt, dict)
                    ],
                    "reason": ablation.get("reason"),
                }
            )

        best = payload.get("best_candidate") or {}
        best_id = best.get("id")
        best_entry = next((item for item in candidate_entries if item.get("candidate_id") == best_id), None)
        completed = [item for item in candidate_entries if item.get("status") in {"ok", "mocked"} and item.get("disabled_metrics")]
        partial = [item for item in candidate_entries if item.get("enabled") and item.get("status") not in {"ok", "mocked", "skipped"}]
        eligible = [item for item in candidate_entries if item.get("enabled")]
        declared = [item for item in candidate_entries if item.get("declared_switch")]
        if completed:
            status = "ok"
            reason = "automatic ablation completed for at least one candidate"
        elif partial:
            status = "partial"
            reason = "ablation was materialized but did not produce complete disabled metrics"
        elif eligible:
            status = "pending"
            reason = "ablation was eligible but no disabled metrics were parsed"
        elif declared:
            status = "skipped"
            reason = "candidate ablation switches were declared, but no candidate reached full eval before ablation"
        else:
            status = "skipped"
            reason = "no candidate exposed an ablation_switch"

        best_comparison = (best_entry or {}).get("comparison") or {}
        proxy_baseline = None
        if best:
            proxy_baseline = ((best.get("proxy_screen") or {}).get("baseline_metrics") or {}).copy()
        return ExperimentAgent._jsonable(
            {
                "schema_version": "c2c_ablation_results_v1",
                "status": status,
                "reason": reason,
                "baseline": payload.get("baseline") or adapter.baseline,
                "proxy_baseline": proxy_baseline or None,
                "best_candidate_id": best_id,
                "best_supported": best_comparison.get("mechanism_supported"),
                "best_delta_enabled_vs_disabled": best_comparison.get("enabled_minus_disabled_mean"),
                "best_dataset_enabled_minus_disabled": best_comparison.get("dataset_enabled_minus_disabled") or {},
                "candidate_ablations": candidate_entries,
                "allowed_files": adapter.allowed_files,
            }
        )

    def _ensure_c2c_proxy_baseline(
        self,
        adapter: C2CAdapter,
        candidate_run_spec: dict[str, Any],
        gpu_selection: Any,
        retry_policy: dict[str, Any],
        baseline_repo_state: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        if not proxy_cfg.get("enabled", False) or not proxy_cfg.get("require_paired_baseline", True):
            return {"enabled": bool(proxy_cfg.get("enabled", False)), "status": "skipped", "reason": "paired proxy baseline disabled"}
        cached = adapter.proxy_baseline_metrics(candidate_run_spec)
        if cached and cached.get("source") != "configured_full_baseline_subset_fallback":
            return {"enabled": True, "status": "cached", "metrics": cached, "path": str((candidate_run_spec.get("proxy_screen") or {}).get("baseline_metrics_path") or "")}
        if not proxy_cfg.get("run_baseline_if_missing", True):
            if cached:
                return {"enabled": True, "status": "fallback", "metrics": cached, "reason": "using configured baseline subset fallback"}
            return {"enabled": True, "status": "blocked", "reason": "proxy baseline cache missing and run_baseline_if_missing is false"}

        patched_state = self._snapshot_c2c_repo_state(adapter) if baseline_repo_state is not None else None
        if baseline_repo_state is not None:
            self._restore_c2c_repo_state(adapter, baseline_repo_state)
        try:
            baseline_spec = adapter.materialize_proxy_baseline_configs(gpu_selection)
            attempts: list[dict[str, Any]] = []
            for step_idx, command in enumerate(baseline_spec["commands"]["preflight"]):
                result = self.runner.run_step(
                    name=f"proxy_baseline_preflight_{step_idx}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={"max_attempts": 1, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, -1, [])},
                )
                attempts.append(result)
                if result.get("status") != "ok":
                    return {"enabled": True, "status": "failed", "reason": f"proxy baseline preflight {step_idx} failed", "attempts": attempts}
            train_result = self.runner.run_step(
                name="proxy_baseline_train",
                command=baseline_spec["commands"]["train"],
                working_dir=adapter.repo_root,
                retry_policy={**retry_policy, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, 0, [baseline_spec["commands"]["train"]])},
            )
            attempts.append(train_result)
            if train_result.get("status") != "ok" and not self._c2c_checkpoint_final_exists(baseline_spec):
                return {"enabled": True, "status": "failed", "reason": "proxy baseline train failed", "attempts": attempts}
            eval_items = list(baseline_spec.get("eval_configs", {}).items())
            for eval_idx, command in enumerate(baseline_spec["commands"]["eval"]):
                dataset = eval_items[eval_idx][0] if eval_idx < len(eval_items) else f"dataset_{eval_idx}"
                result = self.runner.run_step(
                    name=f"proxy_baseline_eval_{dataset}",
                    command=command,
                    working_dir=adapter.repo_root,
                    retry_policy={**retry_policy, "timeout_seconds": self._c2c_proxy_command_timeout(proxy_cfg, 1 + eval_idx, baseline_spec["commands"]["eval"])},
                )
                attempts.append(result)
                if result.get("status") != "ok":
                    return {"enabled": True, "status": "failed", "reason": f"proxy baseline eval {dataset} failed", "attempts": attempts}
            metrics = adapter.collect_proxy_baseline_run_metrics(baseline_spec)
            if not metrics:
                if proxy_cfg.get("allow_configured_baseline_fallback", True):
                    fallback = adapter.proxy_baseline_metrics(candidate_run_spec)
                    if fallback:
                        return {
                            "enabled": True,
                            "status": "fallback",
                            "reason": "proxy baseline run produced no metrics; using configured baseline subset fallback",
                            "metrics": fallback,
                            "attempts": attempts,
                        }
                return {"enabled": True, "status": "failed", "reason": "proxy baseline run produced no metrics", "attempts": attempts}
            metrics = dict(metrics)
            metrics.setdefault("source", "proxy_baseline_run")
            write_json(Path(baseline_spec["metrics_path"]), metrics)
            return {"enabled": True, "status": "ok", "metrics": metrics, "path": str(baseline_spec["metrics_path"]), "attempts": attempts}
        finally:
            if patched_state is not None:
                self._restore_c2c_repo_state(adapter, patched_state)

    def _c2c_static_proxy_screen(
        self,
        *,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        patch_result: dict[str, Any],
        has_executable_change: bool,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.context.config)
        patch_risk = self._c2c_patch_risk(
            patch_result=patch_result,
            config_overrides=run_spec.get("config_overrides", {}),
            candidate=candidate,
        )
        signals = {
            "has_executable_change": has_executable_change,
            "changed_file_count": len((patch_result or {}).get("changed_files") or []),
            "config_override_keys": patch_risk.get("config_override_keys", []),
            "risk_labels": patch_risk.get("risk_labels", []),
            "mechanism_soft_issues": ((patch_result or {}).get("mechanism_review") or {}).get("soft_issues") or [],
            "quality_repair_needed": bool((((patch_result or {}).get("mechanism_review") or {}).get("quality_repair") or {}).get("needed")),
        }
        base = {
            "enabled": True,
            "status": "passed",
            "mode": proxy_cfg.get("mode", "static"),
            "signals": signals,
            "patch_risk": patch_risk,
            "quality_repair": _instrumentation_quality_repair_request(patch_result),
        }
        static_hard_gate = bool(proxy_cfg.get("static_hard_gate", True))
        if static_hard_gate and proxy_cfg.get("reject_if_no_executable_change", True) and not has_executable_change:
            base.update({"status": "rejected", "reason": "no executable patch or config override"})
            return base
        if static_hard_gate and proxy_cfg.get("reject_eval_code_changes", True) and "evaluation_code_changed" in set(patch_risk.get("risk_labels") or []):
            base.update(
                _repairable_proxy_risk(
                    "candidate changes evaluator code; repair S2.5 patch before full S3",
                    "move mechanism evidence out of script/evaluation and into model/train artifacts",
                    patch_risk=patch_risk,
                    source="static_proxy",
                )
            )
            return base
        if static_hard_gate and proxy_cfg.get("reject_test_only_changes", True):
            labels = set(patch_risk.get("risk_labels") or [])
            changed_files = patch_risk.get("changed_files") or []
            config_keys = patch_risk.get("config_override_keys") or []
            if changed_files and labels and labels <= {"test_change"} and not config_keys:
                base.update(
                    _repairable_proxy_risk(
                        "candidate only changes tests; repair S2.5 patch before full S3",
                        "add an executable model/train/recipe mechanism change",
                        patch_risk=patch_risk,
                        source="static_proxy",
                    )
                )
                return base
        max_risk_files = proxy_cfg.get("max_risk_files")
        if static_hard_gate and max_risk_files is not None and len(patch_risk.get("risk_files") or []) > int(max_risk_files):
            base.update(
                _repairable_proxy_risk(
                    f"patch risk file count exceeds proxy threshold {max_risk_files}",
                    "shrink patch to the core mechanism and one focused validation hook",
                    patch_risk=patch_risk,
                    source="static_proxy",
                )
            )
            return base
        return base

    @staticmethod
    def _c2c_proxy_commands(adapter: C2CAdapter, run_spec: dict[str, Any], proxy_cfg: dict[str, Any]) -> list[str]:
        commands = list(proxy_cfg.get("commands") or [])
        if not commands and proxy_cfg.get("mode") in {"command", "commands", "replay", "validation"}:
            proxy_commands = (run_spec.get("proxy_screen") or {}).get("commands") or {}
            commands.extend(proxy_commands.get("train") if isinstance(proxy_commands.get("train"), list) else [proxy_commands.get("train")] if proxy_commands.get("train") else [])
            commands.extend(proxy_commands.get("eval") or [])
        fields = {
            "repo_root": str(adapter.repo_root),
            "run_id": str(run_spec.get("run_id") or ""),
            "run_root": str(run_spec.get("run_root") or ""),
            "train_config": str(run_spec.get("train_config") or ""),
            "proxy_root": str((run_spec.get("proxy_screen") or {}).get("run_root") or ""),
            "proxy_train_config": str((run_spec.get("proxy_screen") or {}).get("train_config") or ""),
            "proxy_metrics": str((run_spec.get("proxy_screen") or {}).get("metrics_path") or ""),
        }
        rendered: list[str] = []
        for command in commands:
            if not command:
                continue
            try:
                rendered.append(str(command).format(**fields))
            except KeyError:
                rendered.append(str(command))
        return rendered

    @staticmethod
    def _c2c_proxy_metric_decision(
        *,
        metrics: dict[str, Any] | None,
        baseline: dict[str, Any],
        proxy_cfg: dict[str, Any],
        proxy_baseline: dict[str, Any] | None,
        patch_risk: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not metrics:
            if proxy_cfg.get("require_proxy_metrics"):
                return {"status": "rejected", "reason": "proxy metrics required but not found"}
            return None
        comparison_baseline = proxy_baseline or baseline
        if proxy_cfg.get("require_paired_baseline", True) and not proxy_baseline:
            return {"status": "rejected", "reason": "paired proxy baseline required but not found"}
        baseline_mean = float(comparison_baseline.get("mean") or baseline.get("mean") or 0.0)
        proxy_deltas = ExperimentAgent._c2c_dataset_deltas(metrics, comparison_baseline)
        regressions = ExperimentAgent._c2c_dataset_regressions(metrics, comparison_baseline)
        worst_regression = max(regressions.values()) if regressions else 0.0
        mean_delta = round(float(metrics["mean"]) - baseline_mean, 4) if metrics.get("mean") is not None else None
        proxy_score = ExperimentAgent._c2c_proxy_score(mean_delta, worst_regression, patch_risk or {}, proxy_cfg)
        evidence = {
            "proxy_baseline": comparison_baseline,
            "proxy_delta_vs_baseline": mean_delta,
            "proxy_dataset_deltas": proxy_deltas,
            "proxy_dataset_regressions": regressions,
            "proxy_worst_dataset_regression": round(worst_regression, 4),
            "proxy_score": proxy_score,
            "proxy_decision_mode": "paired_baseline" if proxy_baseline else "configured_full_baseline",
        }
        threshold = proxy_cfg.get("min_proxy_mean_delta")
        if threshold is not None and mean_delta is not None and mean_delta < float(threshold):
            repairable_margin = float(proxy_cfg.get("repairable_proxy_mean_margin", 0.0) or 0.0)
            if repairable_margin > 0 and mean_delta >= float(threshold) - repairable_margin:
                reason = f"proxy mean delta {mean_delta} below threshold {float(threshold)} but within repairable margin {repairable_margin}"
                repair_hint = "repair S2.5 patch using proxy dataset deltas before full S3"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                    ),
                    **evidence,
                }
            return {
                "status": "rejected",
                "reason": f"proxy mean delta {mean_delta} below hard threshold {float(threshold)}",
                **evidence,
            }
        max_regression = proxy_cfg.get("max_proxy_dataset_regression")
        if max_regression is not None and regressions:
            worst_dataset = max(regressions, key=lambda key: regressions[key])
            worst = float(regressions[worst_dataset])
            if worst > float(max_regression):
                repairable_margin = float(proxy_cfg.get("repairable_proxy_regression_margin", 0.0) or 0.0)
                if repairable_margin > 0 and worst <= float(max_regression) + repairable_margin:
                    reason = f"proxy dataset regression {worst_dataset}={round(worst, 4)} exceeds threshold {float(max_regression)} but within repairable margin {repairable_margin}"
                    repair_hint = f"repair S2.5 patch to bound {worst_dataset} regression before full S3"
                    return {
                        "status": "repairable_proxy_risk",
                        "reason": reason,
                        "repair_hint": repair_hint,
                        "repair_route": "S2_plan",
                        "repair_mode": "effect_first_proxy_repair",
                        "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                            reason=reason,
                            repair_hint=repair_hint,
                            evidence=evidence,
                            patch_risk=patch_risk or {},
                        ),
                        **evidence,
                    }
                return {
                    "status": "rejected",
                    "reason": f"proxy dataset regression {worst_dataset}={round(worst, 4)} exceeds hard threshold {float(max_regression)}",
                    **evidence,
                }
        min_proxy_score = proxy_cfg.get("min_proxy_score")
        if min_proxy_score is not None and proxy_score is not None and proxy_score < float(min_proxy_score):
            repairable_margin = float(proxy_cfg.get("repairable_proxy_score_margin", 0.0) or 0.0)
            if repairable_margin > 0 and proxy_score >= float(min_proxy_score) - repairable_margin:
                reason = f"proxy score {proxy_score} below threshold {float(min_proxy_score)} but within repairable margin {repairable_margin}"
                repair_hint = "repair S2.5 patch to reduce patch risk or dataset regression before full S3"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                    ),
                    **evidence,
                }
            return {
                "status": "rejected",
                "reason": f"proxy score {proxy_score} below hard threshold {float(min_proxy_score)}",
                **evidence,
            }

        soft_flags = []
        soft_delta = proxy_cfg.get("soft_proxy_mean_delta")
        if soft_delta is not None and mean_delta is not None and mean_delta < float(soft_delta):
            soft_flags.append(f"proxy mean delta {mean_delta} below soft threshold {float(soft_delta)}")
        soft_regression = proxy_cfg.get("soft_max_proxy_dataset_regression")
        if soft_regression is not None and worst_regression > float(soft_regression):
            soft_flags.append(f"proxy worst dataset regression {round(worst_regression, 4)} above soft threshold {float(soft_regression)}")
        soft_score = proxy_cfg.get("soft_min_proxy_score")
        if soft_score is not None and proxy_score is not None and proxy_score < float(soft_score):
            soft_flags.append(f"proxy score {proxy_score} below soft threshold {float(soft_score)}")
        if soft_flags:
            if proxy_cfg.get("repair_soft_proxy_fail", True):
                reason = "; ".join(soft_flags)
                repair_hint = "effect repair only: improve cheap-proxy mean/regression/runtime behavior before full S3; do not spend this repair on ablation, coverage, matched-coverage, or paperization diagnostics"
                return {
                    "status": "repairable_proxy_risk",
                    "reason": reason,
                    "repair_hint": repair_hint,
                    "repair_route": "S2_plan",
                    "repair_mode": "effect_first_proxy_repair",
                    "proxy_effect_repair_contract": _proxy_effect_repair_contract(
                        reason=reason,
                        repair_hint=repair_hint,
                        evidence=evidence,
                        patch_risk=patch_risk or {},
                        soft_flags=soft_flags,
                    ),
                    "soft_fail": True,
                    "soft_flags": soft_flags,
                    **evidence,
                }
            return {
                "status": "passed",
                "reason": "proxy passed hard thresholds with soft warnings",
                "soft_fail": True,
                "soft_flags": soft_flags,
                **evidence,
            }
        return {"status": "passed", "reason": "proxy passed hard and soft thresholds", **evidence}

    @staticmethod
    def _c2c_proxy_command_failure(step_result: dict[str, Any]) -> dict[str, Any]:
        text = "\n".join(
            (attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "")
            for attempt in step_result.get("attempts", [])
            if isinstance(attempt, dict)
        )
        lower = text.lower()
        timed_out = any(bool(attempt.get("timed_out")) for attempt in step_result.get("attempts", []) if isinstance(attempt, dict))
        if timed_out or step_result.get("returncode") == 124 or "timed out" in lower:
            category = "proxy_timeout"
            repair_hint = "reduce mechanism inference/training cost or add bounded early-exit guards before rerunning cheap proxy"
        elif "mat1 and mat2 must have the same dtype" in lower or "same dtype" in lower:
            category = "dtype_mismatch"
            repair_hint = "cast new mechanism tensors/modules to the active model dtype/device before matmul or linear layers"
        elif "must be real number, not list" in lower:
            category = "schema_shape_mismatch"
            repair_hint = "normalize dataset adapter fields to scalar/tensor shapes expected by the existing training path"
        elif "out of memory" in lower or "cuda oom" in lower or "cublas_status_alloc_failed" in lower:
            category = "resource_oom"
            repair_hint = "reduce proxy memory footprint or fix allocation behavior before full S3"
        elif "traceback" in lower:
            category = "runtime_exception"
            repair_hint = "fix the proxy training runtime exception in the S2.5 patch before full S3"
        else:
            category = "command_failed"
            repair_hint = "inspect proxy command stdout/stderr and repair the S2.5 patch before full S3"
        summary = category
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line and ("error" in line.lower() or "exception" in line.lower() or "runtimeerror" in line.lower() or "typeerror" in line.lower()):
                summary = f"{category}: {line[-240:]}"
                break
        return {
            "category": category,
            "summary": summary,
            "repair_hint": repair_hint,
            "returncode": step_result.get("returncode"),
            "step": step_result.get("step"),
            "elapsed_seconds": max(
                [
                    float(attempt.get("elapsed_seconds") or 0.0)
                    for attempt in step_result.get("attempts", [])
                    if isinstance(attempt, dict)
                ]
                or [0.0]
            ),
            "timeout_seconds": next(
                (
                    attempt.get("timeout_seconds")
                    for attempt in step_result.get("attempts", [])
                    if isinstance(attempt, dict) and attempt.get("timeout_seconds") is not None
                ),
                None,
            ),
        }

    @staticmethod
    def _c2c_oom_recovery_hint(step_result: dict[str, Any]) -> dict[str, Any] | None:
        text = "\n".join(
            (attempt.get("stdout") or "") + "\n" + (attempt.get("stderr") or "")
            for attempt in step_result.get("attempts", [])
        ).lower()
        if "out of memory" not in text and "cuda oom" not in text and "cublas_status_alloc_failed" not in text:
            return None
        return {
            "action": "retry_train_reduced_concurrency",
            "status": "attempted",
            "reason": "detected CUDA OOM signature; retry with fewer visible GPUs/processes without changing hyperparameters",
        }

    def _c2c_posthoc_review(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        fallback = self._c2c_deterministic_posthoc_review(payload)
        if self._c2c_proxy_only_failure(payload):
            fallback["status"] = "deterministic_proxy_feedback"
            fallback["reason"] = "cheap proxy rejected or repair-routed all candidates; skipped GPT posthoc to keep effect-first loop cheap"
            return fallback
        if not self.context.llm.use_real_api:
            fallback["status"] = "deterministic_no_llm"
            fallback["reason"] = "real GPT API unavailable; deterministic posthoc feedback generated"
            return fallback
        prompt = {
            "baseline": payload.get("baseline"),
            "acceptance": payload.get("acceptance"),
            "candidate_results": _compact_for_review(payload.get("candidate_results", []), 12000),
            "constraints": [
                "Do not modify metrics or acceptance.",
                "Only analyze causes and propose next-round S1/S2 constraints.",
                "Training and evaluation execution already completed deterministically.",
            ],
        }
        schema = {"type": "object", "required": ["status", "failure_modes", "next_round_suggestions", "avoid_repeat_rules"]}
        try:
            review = self.context.llm.generate_json_with_schema(
                instructions=(
                    "You are a posthoc experiment reviewer for C2C. Analyze failures, regressions, and likely next steps. "
                    "Return JSON only; do not alter reported results."
                ),
                prompt=json.dumps(prompt, ensure_ascii=False),
                default=fallback,
                schema=schema,
                agent_name="c2c-posthoc-reviewer",
            )
        except Exception as exc:
            fallback["status"] = "degraded"
            fallback["reason"] = f"GPT posthoc review unavailable: {_short_error(exc)}"
            fallback["gpt_error_type"] = exc.__class__.__name__
            return fallback
        return review if isinstance(review, dict) else None

    @staticmethod
    def _c2c_deterministic_posthoc_review(payload: dict[str, Any]) -> dict[str, Any]:
        acceptance = payload.get("acceptance") or {}
        baseline = payload.get("baseline") or {}
        best = payload.get("best_candidate") or {}
        candidate_results = payload.get("candidate_results") or []
        baseline_datasets = baseline.get("datasets") or {}
        failure_modes: list[str] = []
        suggestions: list[str] = []
        avoid_rules: list[str] = []
        feedback_entries: list[dict[str, Any]] = []
        proxy_failed = [
            item
            for item in candidate_results
            if isinstance(item, dict) and item.get("decision") in {"proxy_rejected", "proxy_repairable"}
        ]
        if proxy_failed and len(proxy_failed) == len(candidate_results):
            repairable_count = sum(1 for item in proxy_failed if item.get("decision") == "proxy_repairable")
            failure_modes.append(f"cheap proxy blocked all candidates before full S3: repairable={repairable_count}/{len(proxy_failed)}")
            suggestions.append("Route back to S2.5 and repair only the failing patch mechanism before any full S3 run.")
            suggestions.append("Require next S2.5 to cite proxy_dataset_deltas, command_failure category, and patch_risk_labels for each repaired candidate.")
            avoid_rules.append("Do not enter full S3 for a candidate until cheap proxy passes paired-baseline mean, worst-dataset regression, and proxy score checks.")

        best_metrics = best.get("metrics") or {}
        delta = acceptance.get("delta")
        if delta is None and best_metrics.get("mean") is not None and baseline.get("mean") is not None:
            try:
                delta = float(best_metrics["mean"]) - float(baseline["mean"])
            except (TypeError, ValueError):
                delta = None
        if delta is not None and float(delta) < float(acceptance.get("min_delta_to_pass", 0.1)):
            failure_modes.append(f"best mean did not clear baseline margin: delta={round(float(delta), 4)}")
            suggestions.append("Prioritize mechanisms with an explicit expected mean gain, not only regression protection.")
            avoid_rules.append("Do not repeat below-baseline configuration-only variants without a new mechanism and ablation.")

        best_ablation_evidence = ExperimentAgent._c2c_ablation_evidence(best)
        if best_ablation_evidence.get("status") == "no_effect":
            failure_modes.append(
                f"ablation switch did not change metrics: enabled_minus_disabled={best_ablation_evidence.get('enabled_minus_disabled_mean')}"
            )
            suggestions.append("Require S2.5 to prove the ablation switch changes the active inference path before full S3.")
            avoid_rules.append("Do not accept a mechanism whose ablation-disabled eval is identical to enabled eval.")

        regressions = best.get("dataset_regressions") or {}
        if regressions:
            worst_dataset = max(regressions, key=lambda key: regressions[key])
            worst_value = regressions.get(worst_dataset)
            failure_modes.append(f"worst dataset regression: {worst_dataset}={worst_value}")
            suggestions.append(f"Add a {worst_dataset}-specific guard or fallback before rerunning similar alignment gates.")
            avoid_rules.append(f"Do not increase alignment transfer strength without bounding {worst_dataset} regression.")

        for candidate in candidate_results:
            if not isinstance(candidate, dict):
                continue
            decision = candidate.get("decision")
            if decision not in {"not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}:
                continue
            candidate_metrics = candidate.get("metrics") or {}
            candidate_regressions = candidate.get("dataset_regressions") or {}
            attribution = candidate.get("failure_attribution") or {}
            proxy_screen = candidate.get("proxy_screen") or {}
            reason = (
                proxy_screen.get("reason")
                or candidate.get("blocked_reason")
                or acceptance.get("reason")
                or "candidate did not pass acceptance"
            )
            feedback_entries.append(
                {
                    "kind": "c2c_posthoc_feedback",
                    "idea_id": candidate.get("id"),
                    "title": candidate.get("title"),
                    "decision": decision,
                    "failure_mode": decision,
                    "reason": reason,
                    "metrics": candidate_metrics,
                    "dataset_regressions": candidate_regressions,
                    "failure_attribution": attribution,
                    "proxy_screen": proxy_screen,
                    "dragging_datasets": attribution.get("dragging_datasets") or [],
                    "sample_type_failures": attribution.get("sample_type_failures") or [],
                    "patch_risk": attribution.get("patch_risk") or {},
                    "mixed_gain_patterns": attribution.get("mixed_gain_patterns") or [],
                    "avoid_repeat_rule": _candidate_avoid_rule(candidate, baseline_datasets),
                }
            )

        return {
            "status": "deterministic_fallback",
            "reason": acceptance.get("reason") or "candidate did not pass acceptance",
            "failure_modes": failure_modes or [acceptance.get("reason") or "candidate did not pass acceptance"],
            "next_round_suggestions": suggestions or ["Generate a materially different mechanism before rerunning S3."],
            "avoid_repeat_rules": list(dict.fromkeys(avoid_rules or ["Do not repeat failed ideas without a new mechanism and explicit ablation."])),
            "feedback_entries": feedback_entries,
        }

    @staticmethod
    def _c2c_failure_analysis_md(payload: dict[str, Any], posthoc: dict[str, Any] | None) -> str:
        acceptance = payload.get("acceptance") or {}
        lines = ["# C2C Failure Analysis", ""]
        lines.append(f"- Acceptance passed: {acceptance.get('passed')}")
        lines.append(f"- Reason: {acceptance.get('reason')}")
        best = payload.get("best_candidate") or {}
        if best:
            lines.append(f"- Best candidate: {best.get('id')} decision={best.get('decision')} mean={(best.get('metrics') or {}).get('mean')}")
            lines.append(f"- Dataset regressions: {json.dumps(best.get('dataset_regressions') or {}, ensure_ascii=False)}")
            attribution = best.get("failure_attribution") or {}
            if attribution:
                lines.append(f"- Primary failure: {attribution.get('primary_failure')}")
                lines.append(f"- Dragging datasets: {json.dumps(attribution.get('dragging_datasets') or [], ensure_ascii=False)}")
                lines.append(f"- Sample families failed: {json.dumps(attribution.get('sample_type_failures') or [], ensure_ascii=False)}")
                lines.append(f"- Patch risk: {json.dumps((attribution.get('patch_risk') or {}).get('risk_files') or [], ensure_ascii=False)}")
                lines.append(f"- Mixed gain patterns: {json.dumps(attribution.get('mixed_gain_patterns') or [], ensure_ascii=False)}")
                lines.append(f"- Ablation evidence: {json.dumps(attribution.get('ablation_evidence') or {}, ensure_ascii=False)}")
        proxy_candidates = [
            item
            for item in payload.get("candidate_results", [])
            if isinstance(item, dict) and item.get("decision") in {"proxy_rejected", "proxy_repairable"}
        ]
        if proxy_candidates:
            lines.append("")
            lines.append("## Cheap Proxy Evidence")
            for item in proxy_candidates:
                proxy_screen = item.get("proxy_screen") or {}
                attribution = item.get("failure_attribution") or {}
                command_failure = proxy_screen.get("command_failure") or {}
                lines.append(f"- {item.get('id')}: decision={item.get('decision')} proxy_status={proxy_screen.get('status')}")
                if proxy_screen.get("reason"):
                    lines.append(f"  - Reason: {proxy_screen.get('reason')}")
                if command_failure:
                    lines.append(f"  - Command failure: {json.dumps(command_failure, ensure_ascii=False)}")
                if proxy_screen.get("proxy_dataset_deltas"):
                    lines.append(f"  - Proxy dataset deltas: {json.dumps(proxy_screen.get('proxy_dataset_deltas'), ensure_ascii=False)}")
                    lines.append(f"  - Proxy score: {proxy_screen.get('proxy_score')}")
                if attribution.get("dragging_datasets"):
                    lines.append(f"  - Dragging datasets: {json.dumps(attribution.get('dragging_datasets'), ensure_ascii=False)}")
                patch_labels = ((attribution.get("patch_risk") or {}).get("risk_labels") or [])
                if patch_labels:
                    lines.append(f"  - Patch risk labels: {json.dumps(patch_labels, ensure_ascii=False)}")
        lines.append("")
        lines.append("## Posthoc Review")
        if posthoc:
            for item in _posthoc_items(posthoc.get("failure_modes"), limit=8):
                lines.append(f"- Failure mode: {item}")
            for item in _posthoc_items(posthoc.get("next_round_suggestions"), limit=8):
                lines.append(f"- Next round: {item}")
            for item in _posthoc_items(posthoc.get("avoid_repeat_rules"), limit=8):
                lines.append(f"- Avoid: {item}")
        else:
            lines.append("- Posthoc review unavailable.")
        return "\n".join(lines)

    def _write_c2c_failure_feedback(self, payload: dict[str, Any], *, artifacts: list[str]) -> dict[str, Any]:
        best = payload.get("best_candidate")
        feedback_candidate = best
        if (not feedback_candidate or not feedback_candidate.get("metrics")) and isinstance(payload.get("best_proxy_candidate"), dict):
            feedback_candidate = payload.get("best_proxy_candidate")
        acceptance = payload.get("acceptance") or {}
        reason = acceptance.get("reason") or "C2C candidate did not clear acceptance"
        failure_mode = "not_viable"
        if feedback_candidate and feedback_candidate.get("decision") == "proxy_rejected":
            failure_mode = "proxy_rejected"
            reason = ((feedback_candidate.get("proxy_screen") or {}).get("reason") or reason)
        elif feedback_candidate and feedback_candidate.get("decision") == "proxy_repairable":
            failure_mode = "proxy_repairable"
            reason = ((feedback_candidate.get("proxy_screen") or {}).get("reason") or reason)
        elif not feedback_candidate or not feedback_candidate.get("metrics"):
            failure_mode = "no_metrics"
        elif feedback_candidate.get("decision") == "blocked":
            failure_mode = "blocked"
        manager = FailureLogManager(self.context.config, external_root=self.context.project_root / "meta")
        entry = manager.append_c2c_feedback(
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            candidate=feedback_candidate,
            acceptance=acceptance,
            failure_mode=failure_mode,
            reason=reason,
            artifacts=artifacts,
        )
        feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
        )
        method_feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
            view="method",
        )
        implementation_feedback_bundle = build_c2c_feedback_bundle(
            [entry, *((payload.get("posthoc_review") or {}).get("feedback_entries") or [])],
            project_id=self.context.project_root.name,
            iteration=self._registry_iteration(),
            traces=self._load_iteration_traces(),
            sources=artifacts,
            view="implementation",
        )
        meta_memory = self.context.project_root / "meta" / "negative_memory.jsonl"
        ensure_dir(meta_memory.parent)
        with meta_memory.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback_bundle["summary_entry"], ensure_ascii=False) + "\n")
        feedback_round_dir = self.context.project_root / "literature" / "feedback"
        ensure_dir(feedback_round_dir)
        round_path = feedback_round_dir / f"failed_ideas_round_{self._registry_iteration():03d}.json"
        write_json(round_path, feedback_bundle)
        failed_candidates = [
            item
            for item in payload.get("candidate_results", [])
            if item.get("decision") in {"not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}
        ]
        feedback_payload = {
            "created_at": now_utc(),
            "entry": entry,
            "summary": feedback_bundle["summary"],
            "method_feedback": method_feedback_bundle,
            "implementation_feedback": implementation_feedback_bundle,
            "candidate": _compact_candidate_result(feedback_candidate) if isinstance(feedback_candidate, dict) else feedback_candidate,
            "best_candidate": _compact_candidate_result(best) if isinstance(best, dict) else best,
            "best_proxy_candidate": _compact_candidate_result(payload.get("best_proxy_candidate")) if isinstance(payload.get("best_proxy_candidate"), dict) else payload.get("best_proxy_candidate"),
            "candidate_results": [
                _compact_candidate_result(item)
                for item in payload.get("candidate_results", [])
                if isinstance(item, dict)
            ],
            "failed_idea_ids": [item.get("id") for item in failed_candidates if item.get("id")],
            "failed_titles": [item.get("title") for item in failed_candidates if item.get("title")],
            "acceptance": acceptance,
            "posthoc_review": _compact_c2c_result_payload(payload.get("posthoc_review")),
            "avoid_repeat_rules": [
                item
                for item in [entry.get("avoid_repeat_rule"), *_posthoc_items((payload.get("posthoc_review") or {}).get("avoid_repeat_rules"))]
                if item
            ],
            "feedback_round_path": round_path.relative_to(self.context.project_root).as_posix(),
        }
        return self.context.artifacts.write_json(
            self.stage_key,
            "results/failure_feedback.json",
            feedback_payload,
            artifact_type="c2c_failure_feedback",
            summary="Failure feedback routed to next S1/S2 iteration",
            source_paths=artifacts,
        )

    def _load_iteration_traces(self) -> list[dict[str, Any]]:
        trace_path = self.context.project_root / "meta" / "iteration_trace.jsonl"
        if not trace_path.exists():
            return []
        traces: list[dict[str, Any]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                traces.append(item)
        return traces

    def _registry_iteration(self) -> int:
        registry_path = self.context.project_root / "meta" / "registry.yaml"
        payload = read_yaml(registry_path, default={}) or {}
        return int(payload.get("iteration", 1) or 1)

    @staticmethod
    def _best_c2c_candidate(run_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [
            item
            for item in run_results
            if (item.get("metrics") or {}).get("mean") is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda item: item["metrics"]["mean"])

    @staticmethod
    def _best_c2c_proxy_candidate(run_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [
            item
            for item in run_results
            if ((item.get("proxy_screen") or {}).get("metrics") or {}).get("mean") is not None
        ]
        if not scored:
            return None

        def key(item: dict[str, Any]) -> tuple[float, float]:
            proxy = item.get("proxy_screen") or {}
            metrics = proxy.get("metrics") or {}
            score = proxy.get("proxy_score")
            try:
                proxy_score = float(score)
            except (TypeError, ValueError):
                proxy_score = float("-inf")
            return (float(metrics.get("mean")), proxy_score)

        return max(scored, key=key)

    @staticmethod
    def _c2c_proxy_only_failure(payload: dict[str, Any]) -> bool:
        candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
        if not candidates:
            return False
        return all(item.get("decision") in {"proxy_rejected", "proxy_repairable"} for item in candidates)

    @staticmethod
    def _c2c_verification_md(
        best: dict[str, Any] | None,
        baseline_mean: float,
        run_results: list[dict[str, Any]],
        min_delta: float,
        max_regression: float,
    ) -> str:
        lines = ["# Hypothesis Verification", ""]
        threshold = baseline_mean + min_delta
        worst_regression = (best or {}).get("worst_dataset_regression", 0.0)
        if best and best.get("metrics", {}).get("mean", 0) >= threshold and worst_regression <= max_regression:
            lines.append(f"- H1: supported. Best candidate `{best['id']}` reached mean {best['metrics']['mean']} >= threshold {threshold:.4f}.")
        else:
            lines.append(f"- H1: not supported in this loop. No candidate cleared baseline {baseline_mean} + min_delta {min_delta}.")
        ablation = (best or {}).get("ablation") or {}
        comparison = ablation.get("comparison") or {}
        if ablation.get("status") in {"ok", "mocked"} and comparison.get("status") == "ok":
            delta = comparison.get("enabled_minus_disabled_mean")
            switch = ablation.get("switch") or "ablation_switch"
            if comparison.get("mechanism_supported"):
                lines.append(f"- H2: supported. Disabling `{switch}` reduced mean by {delta}, so the measured gain depends on the proposed mechanism.")
            else:
                lines.append(f"- H2: not supported. Disabling `{switch}` did not reduce mean; enabled-minus-disabled mean delta={delta}.")
        elif ablation.get("enabled"):
            lines.append(f"- H2: inconclusive. Ablation status={ablation.get('status')} reason={ablation.get('reason') or 'disabled metrics unavailable'}.")
        else:
            lines.append(f"- H2: skipped. Candidate did not expose an ablation switch ({ablation.get('reason') or 'not run'}).")
        lines.append("")
        lines.append("## Candidate Decisions")
        for item in run_results:
            mean = (item.get("metrics") or {}).get("mean")
            ablation_comparison = ((item.get("ablation") or {}).get("comparison") or {})
            lines.append(
                f"- {item.get('id')}: decision={item.get('decision')}, mean={mean}, "
                f"delta={item.get('delta_vs_baseline')}, ablation_delta={ablation_comparison.get('enabled_minus_disabled_mean')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _c2c_dataset_regressions(metrics: dict[str, Any] | None, baseline: dict[str, Any]) -> dict[str, float]:
        if not metrics:
            return {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        regressions = {}
        for dataset, base_score in baseline_scores.items():
            candidate_score = candidate_scores.get(dataset)
            if candidate_score is None:
                continue
            regressions[dataset] = round(max(0.0, float(base_score) - float(candidate_score)), 4)
        return regressions

    @staticmethod
    def _c2c_dataset_deltas(metrics: dict[str, Any] | None, baseline: dict[str, Any]) -> dict[str, float]:
        if not metrics:
            return {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        deltas = {}
        for dataset, base_score in baseline_scores.items():
            candidate_score = candidate_scores.get(dataset)
            if candidate_score is None:
                continue
            deltas[str(dataset)] = round(float(candidate_score) - float(base_score), 4)
        return deltas

    @staticmethod
    def _c2c_proxy_score(
        mean_delta: float | None,
        worst_regression: float,
        patch_risk: dict[str, Any],
        proxy_cfg: dict[str, Any],
    ) -> float | None:
        if mean_delta is None:
            return None
        regression_weight = float(proxy_cfg.get("proxy_score_regression_weight", 0.5) or 0.0)
        risk_penalty = float(proxy_cfg.get("risk_penalty_per_label", 0.05) or 0.0) * len(patch_risk.get("risk_labels") or [])
        return round(float(mean_delta) - regression_weight * float(worst_regression) - risk_penalty, 4)

    @staticmethod
    def _c2c_failure_attribution(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        metrics = candidate.get("metrics") or {}
        baseline_scores = baseline.get("datasets") or {}
        candidate_scores = metrics.get("datasets") or {}
        proxy_screen = candidate.get("proxy_screen") or {}
        proxy_baseline = proxy_screen.get("proxy_baseline") or proxy_screen.get("baseline_metrics") or {}
        proxy_scores = (proxy_screen.get("metrics") or {}).get("datasets") or {}
        proxy_baseline_scores = proxy_baseline.get("datasets") or {}
        if not candidate_scores and proxy_scores:
            candidate_scores = proxy_scores
            baseline_scores = proxy_baseline_scores or baseline_scores
        proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
        dragging: list[dict[str, Any]] = []
        improved: list[dict[str, Any]] = []
        sample_type_failures: list[dict[str, Any]] = []
        for dataset, base_score in baseline_scores.items():
            if dataset not in candidate_scores:
                continue
            try:
                delta = round(float(candidate_scores[dataset]) - float(base_score), 4)
                item = {
                    "dataset": str(dataset),
                    "sample_family": ExperimentAgent._c2c_dataset_sample_family(str(dataset)),
                    "baseline": float(base_score),
                    "score": float(candidate_scores[dataset]),
                    "delta": delta,
                }
                if proxy_scores:
                    item["source"] = "proxy_screen"
            except (TypeError, ValueError):
                continue
            if delta < 0:
                failed = dict(item)
                failed["regression"] = round(abs(delta), 4)
                dragging.append(failed)
                sample_type_failures.append(
                    {
                        "sample_family": failed["sample_family"],
                        "dataset": failed["dataset"],
                        "evidence": f"dataset delta {delta}",
                    }
                )
            elif delta > 0:
                improved.append(item)
        if not dragging and proxy_deltas:
            for dataset, delta_value in proxy_deltas.items():
                try:
                    delta = round(float(delta_value), 4)
                except (TypeError, ValueError):
                    continue
                baseline_score = proxy_baseline_scores.get(dataset)
                score = proxy_scores.get(dataset)
                item = {
                    "dataset": str(dataset),
                    "sample_family": ExperimentAgent._c2c_dataset_sample_family(str(dataset)),
                    "baseline": baseline_score,
                    "score": score,
                    "delta": delta,
                    "source": "proxy_screen",
                }
                if delta < 0:
                    failed = dict(item)
                    failed["regression"] = round(abs(delta), 4)
                    dragging.append(failed)
                    sample_type_failures.append(
                        {
                            "sample_family": failed["sample_family"],
                            "dataset": failed["dataset"],
                            "evidence": f"proxy dataset delta {delta}",
                        }
                    )
                elif delta > 0:
                    improved.append(item)
        dragging.sort(key=lambda item: item.get("regression", 0.0), reverse=True)
        improved.sort(key=lambda item: item.get("delta", 0.0), reverse=True)

        mixed_patterns = []
        dragging_names = {item["dataset"] for item in dragging}
        improved_names = {item["dataset"] for item in improved}
        if "openbookqa" in improved_names and "mmlu-redux" in dragging_names:
            mixed_patterns.append("openbookqa_gain_mmlu_redux_regression")
        if improved_names and dragging_names:
            mixed_patterns.append("cross_dataset_tradeoff")

        patch_risk = ExperimentAgent._c2c_patch_risk(
            patch_result=candidate.get("patch_result") or {},
            config_overrides=candidate.get("config_overrides") or {},
            candidate=candidate,
        )
        quality_repair = ((candidate.get("proxy_screen") or {}).get("quality_repair") or {})
        ablation_evidence = ExperimentAgent._c2c_ablation_evidence(candidate)
        primary_failure = "none"
        if ablation_evidence.get("status") == "no_effect":
            primary_failure = "ablation_no_effect"
        elif candidate.get("decision") == "proxy_rejected":
            primary_failure = "cheap_proxy_rejected_before_full_training"
        elif candidate.get("decision") == "proxy_repairable":
            primary_failure = "repairable_proxy_risk_before_full_training"
        elif dragging:
            primary_failure = f"{dragging[0]['dataset']}_regression"
        elif candidate.get("delta_vs_baseline") is not None and float(candidate.get("delta_vs_baseline") or 0.0) < 0:
            primary_failure = "mean_below_baseline"
        elif candidate.get("decision") in {"failed_no_metrics", "partial", "blocked", "patch_rejected"}:
            primary_failure = str(candidate.get("decision"))

        return {
            "primary_failure": primary_failure,
            "dragging_datasets": dragging,
            "improved_datasets": improved,
            "sample_type_failures": sample_type_failures,
            "mixed_gain_patterns": list(dict.fromkeys(mixed_patterns)),
            "patch_risk": patch_risk,
            "proxy_screen": _proxy_screen_for_failure_attribution(proxy_screen),
            "proxy_effect_repair_contract": proxy_screen.get("proxy_effect_repair_contract") or {},
            "ablation_evidence": ablation_evidence,
            "quality_repair": quality_repair,
        }

    @staticmethod
    def _c2c_ablation_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
        ablation = candidate.get("ablation") or {}
        comparison = ablation.get("comparison") or {}
        if not comparison:
            return {"status": "missing", "reason": (ablation or {}).get("reason") or "no ablation comparison"}
        delta = comparison.get("enabled_minus_disabled_mean")
        supported = bool(comparison.get("mechanism_supported"))
        dataset_deltas = comparison.get("dataset_enabled_minus_disabled") or {}
        no_effect = delta is not None and abs(float(delta)) <= 1e-4 and all(abs(float(v)) <= 1e-4 for v in dataset_deltas.values())
        if supported:
            status = "supported"
        elif no_effect:
            status = "no_effect"
        else:
            status = "unsupported"
        return {
            "status": status,
            "enabled_minus_disabled_mean": delta,
            "dataset_enabled_minus_disabled": dataset_deltas,
            "mechanism_supported": supported,
            "switch": ablation.get("switch"),
            "ablation_status": ablation.get("status"),
        }

    @staticmethod
    def _c2c_dataset_sample_family(dataset: str) -> str:
        mapping = {
            "mmlu-redux": "multi_domain_knowledge_reasoning",
            "ai2-arc": "science_reasoning_challenge",
            "openbookqa": "openbook_science_qa",
        }
        return mapping.get(dataset, dataset.replace("-", "_"))

    @staticmethod
    def _c2c_patch_risk(
        *,
        patch_result: dict[str, Any],
        config_overrides: dict[str, Any] | None = None,
        candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed_files = list((patch_result or {}).get("changed_files") or [])
        risk_files = []
        labels: set[str] = set()
        for rel_path in changed_files:
            reasons = []
            if str(rel_path).startswith("script/evaluation/"):
                reasons.append("evaluation code changed")
                labels.add("evaluation_code_changed")
            if str(rel_path).startswith("script/train/"):
                reasons.append("training loop changed")
                labels.add("training_loop_changed")
            if str(rel_path) == "rosetta/model/projector.py":
                reasons.append("projector mechanism changed")
                labels.add("projector_mechanism_changed")
            if str(rel_path) == "rosetta/model/aligner.py":
                reasons.append("alignment mechanism changed")
                labels.add("alignment_mechanism_changed")
            if str(rel_path).startswith("recipe/"):
                reasons.append("recipe or hyperparameter file changed")
                labels.add("recipe_changed")
            if str(rel_path).startswith("test/") or str(rel_path).startswith("tests/"):
                reasons.append("test-only change")
                labels.add("test_change")
            if reasons:
                risk_files.append({"path": str(rel_path), "reasons": reasons})
        override_keys = ExperimentAgent._flatten_config_keys(config_overrides or {})
        if override_keys:
            labels.add("config_override_changed")
        if (patch_result or {}).get("errors"):
            labels.add("patch_error")
        return {
            "changed_files": changed_files,
            "risk_files": risk_files,
            "risk_labels": sorted(labels),
            "config_override_keys": override_keys,
            "patch_errors": list((patch_result or {}).get("errors") or []),
            "candidate_id": (candidate or {}).get("id"),
        }

    @staticmethod
    def _flatten_config_keys(value: Any, *, prefix: str = "", limit: int = 40) -> list[str]:
        keys: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                keys.extend(ExperimentAgent._flatten_config_keys(item, prefix=next_prefix, limit=limit))
                if len(keys) >= limit:
                    return keys[:limit]
        elif prefix:
            keys.append(prefix)
        return keys[:limit]

    @staticmethod
    def _c2c_acceptance_comparison(
        best: dict[str, Any] | None,
        baseline: dict[str, Any],
        min_delta: float,
        max_regression: float,
    ) -> dict[str, Any]:
        baseline_mean = float(baseline.get("mean") or DEFAULT_BASELINE["mean"])
        if not best or not best.get("metrics"):
            proxy = (best or {}).get("proxy_screen") or {}
            if (proxy.get("metrics") or {}).get("mean") is not None:
                return {
                    "passed": False,
                    "baseline_mean": baseline_mean,
                    "best_mean": None,
                    "delta": None,
                    "proxy_best_mean": (proxy.get("metrics") or {}).get("mean"),
                    "proxy_delta": proxy.get("proxy_delta_vs_baseline"),
                    "proxy_score": proxy.get("proxy_score"),
                    "proxy_worst_dataset_regression": proxy.get("proxy_worst_dataset_regression"),
                    "min_delta_to_pass": min_delta,
                    "max_dataset_regression": max_regression,
                    "reason": proxy.get("reason") or "cheap proxy blocked candidate before full S3",
                }
            return {
                "passed": False,
                "baseline_mean": baseline_mean,
                "best_mean": None,
                "delta": None,
                "min_delta_to_pass": min_delta,
                "max_dataset_regression": max_regression,
                "reason": "no candidate metrics",
            }
        best_mean = float(best["metrics"]["mean"])
        delta = round(best_mean - baseline_mean, 4)
        worst_regression = float(best.get("worst_dataset_regression") or 0.0)
        passed = delta >= min_delta and worst_regression <= max_regression
        if passed and best.get("acceptance_rule", {}).get("require_ablation_support", False) and best.get("decision") != "candidate_win":
            passed = False
            reason = "mechanism ablation support not met"
        else:
            reason = "accepted" if passed else "mean delta or dataset regression threshold not met"
        return {
            "passed": passed,
            "baseline_mean": baseline_mean,
            "best_mean": best_mean,
            "delta": delta,
            "min_delta_to_pass": min_delta,
            "max_dataset_regression": max_regression,
            "worst_dataset_regression": worst_regression,
            "require_ablation_support": best.get("acceptance_rule", {}).get("require_ablation_support", False),
            "mechanism_supported": best.get("mechanism_supported"),
            "reason": reason,
        }

    @staticmethod
    def _c2c_strong_reference_comparisons(best: dict[str, Any] | None, adapter: C2CAdapter) -> list[dict[str, Any]]:
        metrics = (best or {}).get("metrics") or {}
        if not metrics:
            return []
        best_mean = metrics.get("mean")
        comparisons = []
        for reference in adapter.strong_references:
            if reference.get("enabled") is False:
                continue
            if reference.get("visible_to_ideation") is not False:
                continue
            ref_mean = reference.get("mean")
            delta = None
            if best_mean is not None and ref_mean is not None:
                delta = round(float(best_mean) - float(ref_mean), 4)
            comparisons.append(
                {
                    "name": reference.get("name"),
                    "reference_role": reference.get("reference_role", "s3_strong_reference_only"),
                    "visible_to_ideation": False,
                    "used_for_acceptance": False,
                    "candidate_id": (best or {}).get("id"),
                    "candidate_mean": best_mean,
                    "reference_mean": ref_mean,
                    "delta_vs_reference": delta,
                    "dataset_deltas": ExperimentAgent._c2c_dataset_deltas(metrics, reference),
                    "dataset_regressions": ExperimentAgent._c2c_dataset_regressions(metrics, reference),
                    "source": reference.get("source"),
                }
            )
        return comparisons

    @staticmethod
    def _c2c_summary_md(payload: dict[str, Any]) -> str:
        baseline = payload.get("baseline") or {}
        acceptance = payload.get("acceptance") or {}
        lines = [
            "# C2C Small-Loop Summary",
            "",
            f"- Baseline: {baseline.get('name')} mean={baseline.get('mean')}",
            f"- Acceptance: passed={acceptance.get('passed')} delta={acceptance.get('delta')} min_delta={acceptance.get('min_delta_to_pass')}",
        ]
        best = payload.get("best_candidate")
        if best:
            lines.append(f"- Best candidate: {best.get('id')} mean={best.get('metrics', {}).get('mean')} decision={best.get('decision')}")
            ablation_comparison = ((best.get("ablation") or {}).get("comparison") or {})
            if ablation_comparison:
                lines.append(
                    f"- Best ablation: status={(best.get('ablation') or {}).get('status')} "
                    f"enabled_minus_disabled={ablation_comparison.get('enabled_minus_disabled_mean')} "
                    f"supported={ablation_comparison.get('mechanism_supported')}"
                )
        else:
            lines.append("- Best candidate: none with parsed metrics")
        strong_refs = payload.get("strong_reference_comparisons") or []
        if strong_refs:
            lines.append("")
            lines.append("## S3-Only Strong References")
            for ref in strong_refs:
                lines.append(
                    f"- {ref.get('name')}: delta_vs_reference={ref.get('delta_vs_reference')} "
                    f"used_for_acceptance={ref.get('used_for_acceptance')}"
                )
        lines.append("")
        lines.append("## Runs")
        for item in payload.get("candidate_results", []):
            lines.append(
                f"- {item.get('id')}: status={item.get('command_status')}, decision={item.get('decision')}, metrics={item.get('metrics')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _c2c_blocked_reason(run_results: list[dict[str, Any]]) -> str | None:
        if not run_results:
            return "C2C small-loop produced no candidate runs."
        failed = [item for item in run_results if item.get("decision") in {"failed_no_metrics", "patch_rejected"}]
        if failed and len(failed) == len(run_results):
            return "C2C small-loop did not produce metrics; inspect experiment/logs/c2c_*_commands.json for the training or evaluation failure."
        proxy_rejected = [item for item in run_results if item.get("decision") == "proxy_rejected"]
        if proxy_rejected and len(proxy_rejected) == len(run_results):
            return "C2C cheap proxy rejected all candidates before full S3; inspect proxy_screen and failure_attribution fields."
        proxy_repairable = [item for item in run_results if item.get("decision") == "proxy_repairable"]
        if proxy_repairable and len(proxy_repairable) == len(run_results):
            return "C2C cheap proxy found repairable S2.5 patch risk for all candidates; reroute to S2.5 patch repair before full S3."
        return None

    @staticmethod
    def _parse_laps_metrics(text: str) -> dict[str, Any]:
        import re

        model_blocks = []
        pattern = re.compile(
            r"Evaluating\s+(runs/\S+).*?rsum:\s*([0-9.]+).*?i2t:\s*\[([0-9., ]+)\].*?t2i:\s*\[([0-9., ]+)\]",
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            model_blocks.append(match.groups())
        metrics = {}
        for name, rsum, i2t_raw, t2i_raw in model_blocks:
            i2t = [float(x.strip()) for x in i2t_raw.split(",")]
            t2i = [float(x.strip()) for x in t2i_raw.split(",")]
            metrics[name] = {
                "rsum": float(rsum),
                "i2t": {"R@1": i2t[0], "R@5": i2t[1], "R@10": i2t[2]},
                "t2i": {"R@1": t2i[0], "R@5": t2i[1], "R@10": t2i[2]},
            }
        return metrics

    @staticmethod
    def _parse_metrics_from_log(log_path: Path) -> dict[str, Any] | None:
        if not log_path.exists():
            return None
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        import re

        rsum = re.search(r"(?:rsum:\s*|Current rsum is\s*)([0-9.]+)", text)
        i2t = re.search(r"Image to text \(R@1, R@5, R@10\):\s*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
        t2i = re.search(r"Text to image \(R@1, R@5, R@10\):\s*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
        sim_t = re.search(r"calculate similarity time:\s*([0-9.]+)", text)
        if not (rsum and i2t and t2i):
            return None
        return {
            "rsum": float(rsum.group(1)),
            "i2t": {"R@1": float(i2t.group(1)), "R@5": float(i2t.group(2)), "R@10": float(i2t.group(3))},
            "t2i": {"R@1": float(t2i.group(1)), "R@5": float(t2i.group(2)), "R@10": float(t2i.group(3))},
            "similarity_time": float(sim_t.group(1)) if sim_t else None,
        }

    @staticmethod
    def _screening_decision(metrics: dict[str, Any] | None, baseline_metrics: dict[str, Any] | None) -> str:
        if not metrics or not baseline_metrics:
            return "failed"
        baseline_rsum = baseline_metrics.get("rsum", 0)
        delta_i2t = metrics["i2t"]["R@1"] - baseline_metrics["i2t"]["R@1"]
        delta_t2i = metrics["t2i"]["R@1"] - baseline_metrics["t2i"]["R@1"]
        time_ok = True
        if metrics.get("similarity_time") and baseline_metrics.get("similarity_time"):
            time_ok = metrics["similarity_time"] <= baseline_metrics["similarity_time"] * 1.15
        if time_ok and (metrics["rsum"] >= baseline_rsum or delta_i2t >= 0.5 or delta_t2i >= 0.5):
            return "viable"
        return "not_viable"

    @staticmethod
    def _build_laps_train_command(
        *,
        python_cmd: str,
        data_path: str,
        image_root: str,
        logger_name: str,
        gpu_id: int,
        resume_path: str | None,
        resume_strict: int,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        changes: dict[str, Any],
    ) -> str:
        args = {
            "dataset": "f30k",
            "data_path": data_path,
            "f30k_img_path": image_root,
            "gpu-id": 0,
            "logger_name": logger_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "workers": 8,
            "vit_type": "vit",
            "embed_size": 512,
            "loss": "trip",
            "size_augment": 0,
            "route_mode": "sum",
            "route_conflict_weight": 0.25,
            "attention_weight": 0.8,
            "sparse_ratio": 0.5,
            "aggr_ratio": 0.4,
            "use_cupr": 0,
            "seed": 0,
            "eval": 1,
            "save_results": 1,
            "learning_rate": learning_rate,
        }
        if resume_path:
            args["resume"] = resume_path
            args["resume_strict"] = resume_strict
        args.update(changes)
        rendered = []
        for key, value in args.items():
            rendered.append(f"--{key} {shlex.quote(str(value))}")
        return f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES={gpu_id} {python_cmd} train.py {' '.join(rendered)}"


def _compact_for_review(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return {"truncated_json": text[:max_chars]}


def _repairable_proxy_risk(
    reason: str,
    repair_hint: str,
    *,
    patch_risk: dict[str, Any] | None = None,
    source: str = "static_proxy",
) -> dict[str, Any]:
    return {
        "status": "repairable_proxy_risk",
        "reason": reason,
        "repair_hint": repair_hint,
        "repair_route": "S2_plan",
        "repair_mode": "effect_first_proxy_repair",
        "proxy_effect_repair_contract": _proxy_effect_repair_contract(
            reason=reason,
            repair_hint=repair_hint,
            evidence={},
            patch_risk=patch_risk or {},
            source=source,
        ),
    }


def _proxy_effect_repair_contract(
    *,
    reason: str,
    repair_hint: str | None,
    evidence: dict[str, Any],
    patch_risk: dict[str, Any],
    source: str = "cheap_proxy_metrics",
    soft_flags: list[str] | None = None,
) -> dict[str, Any]:
    dataset_deltas = evidence.get("proxy_dataset_deltas") or {}
    dataset_regressions = evidence.get("proxy_dataset_regressions") or {}
    dragging: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    for dataset, delta_value in dataset_deltas.items():
        try:
            delta = round(float(delta_value), 4)
        except (TypeError, ValueError):
            continue
        regression = dataset_regressions.get(dataset, max(0.0, -delta))
        item = {"dataset": str(dataset), "delta": delta, "regression": round(float(regression or 0.0), 4)}
        if delta <= 0:
            dragging.append(item)
        elif delta > 0:
            improved.append(item)
    dragging.sort(key=lambda item: (item.get("regression", 0.0), -item.get("delta", 0.0)), reverse=True)
    improved.sort(key=lambda item: item.get("delta", 0.0), reverse=True)

    risk_labels = list(patch_risk.get("risk_labels") or [])
    changed_files = list(patch_risk.get("changed_files") or [])
    config_keys = list(patch_risk.get("config_override_keys") or [])
    command_failure = evidence.get("command_failure") if isinstance(evidence.get("command_failure"), dict) else {}
    priorities = []
    if command_failure:
        priorities.append(f"Fix proxy runtime/smoke failure first: {_shorten_text(str(command_failure.get('category') or command_failure.get('summary') or 'command_failure'), 180)}")
    if dragging:
        priorities.append("Target dragging proxy datasets: " + ", ".join(item["dataset"] for item in dragging[:3]))
    if evidence.get("proxy_delta_vs_baseline") is not None:
        priorities.append("Raise proxy_delta_vs_baseline without increasing proxy_worst_dataset_regression.")
    if risk_labels:
        priorities.append("Reduce patch-risk labels that hurt proxy score: " + ", ".join(str(label) for label in risk_labels[:4]))
    priorities.append("Keep the same idea and produce an executable effect-first patch; do not switch to a new idea.")

    contract = {
        "mode": "effect_first_proxy_repair",
        "source": source,
        "goal": "Find a runnable patch with positive cheap-proxy signal and no evaluation pollution before spending full S3.",
        "reason": _shorten_text(str(reason), 500),
        "repair_hint": _shorten_text(str(repair_hint or ""), 500),
        "soft_flags": list(soft_flags or []),
        "proxy_delta_vs_baseline": evidence.get("proxy_delta_vs_baseline"),
        "proxy_score": evidence.get("proxy_score"),
        "proxy_worst_dataset_regression": evidence.get("proxy_worst_dataset_regression"),
        "proxy_dataset_deltas": dataset_deltas,
        "proxy_dataset_regressions": dataset_regressions,
        "dragging_datasets": dragging[:5],
        "improved_datasets": improved[:5],
        "patch_risk_labels": risk_labels,
        "changed_files": changed_files[:12],
        "config_override_keys": config_keys[:12],
        "repair_priorities": priorities[:6],
        "forbidden": [
            "Do not edit evaluator or metric computation files.",
            "Do not spend this repair on ablation, coverage, matched-coverage, or paperization-only diagnostics.",
            "Do not hide failure by weakening proxy thresholds, changing baseline metrics, or changing evaluation data.",
        ],
    }
    if command_failure:
        contract["command_failure"] = {
            "category": command_failure.get("category"),
            "summary": _shorten_text(str(command_failure.get("summary") or ""), 500),
            "repair_hint": _shorten_text(str(command_failure.get("repair_hint") or ""), 500),
        }
    return {key: value for key, value in contract.items() if value not in (None, "", [], {})}


def _instrumentation_quality_repair_request(patch_result: dict[str, Any] | None) -> dict[str, Any]:
    mechanism_review = (patch_result or {}).get("mechanism_review") or {}
    quality_repair = mechanism_review.get("quality_repair") or {}
    if not isinstance(quality_repair, dict):
        return {"needed": False}
    return {
        "needed": bool(quality_repair.get("needed")),
        "deferred": bool(quality_repair.get("deferred", True)),
        "repair_route": "paperization",
        "trigger": "after_effect_found",
        "mode": quality_repair.get("mode") or "paperization_after_effect",
        "issues": list(quality_repair.get("issues") or mechanism_review.get("soft_issues") or []),
        "constraints": list(quality_repair.get("constraints") or []),
        "ablation_switch": quality_repair.get("ablation_switch"),
        "acceptance_guard": {
            "rerun_same_proxy_subset": True,
            "reject_if_enabled_proxy_regresses": True,
            "max_enabled_mean_delta_drop": 0.2,
            "default_behavior_must_remain_enabled": True,
        },
    }


def _c2c_paperization_readiness(best: dict[str, Any] | None, acceptance: dict[str, Any] | None) -> dict[str, Any]:
    if not best or not (acceptance or {}).get("passed"):
        return {
            "status": "waiting_for_effect",
            "reason": "effect-first discovery has not found a full-S3 accepted patch yet",
            "next_stage": "",
            "tasks": [],
        }
    proxy_quality = ((best.get("proxy_screen") or {}).get("quality_repair") or {})
    patch_review = ((best.get("patch_result") or {}).get("mechanism_review") or {})
    issues = list(dict.fromkeys((proxy_quality.get("issues") or []) + (patch_review.get("soft_issues") or [])))
    tasks = []
    if "ablation_switch_not_wired" in issues:
        tasks.append("Add an ablation switch that disables the discovered mechanism without changing enabled behavior.")
    if "missing_coverage_diagnostics_evidence" in issues:
        tasks.append("Add coverage diagnostics for accepted spans/routes/pathology buckets.")
    if "missing_matched_coverage_evidence" in issues:
        tasks.append("Add matched-coverage control bookkeeping for paper analysis.")
    if not tasks:
        tasks.append("Audit ablation, coverage diagnostics, and reviewer-facing evidence before paper writing.")
    return {
        "status": "ready",
        "reason": "effect-first discovery found a full-S3 accepted patch",
        "next_stage": "paperization",
        "candidate_id": best.get("id"),
        "tasks": tasks,
        "constraints": [
            "Do not change default enabled scoring/routing/loss/data sampling.",
            "Do not edit evaluator or metric computation.",
            "Rerun the same cheap proxy and full S3 acceptance checks after paperization.",
        ],
    }


def _candidate_avoid_rule(candidate: dict[str, Any], baseline_datasets: dict[str, Any]) -> str:
    regressions = candidate.get("dataset_regressions") or {}
    if regressions:
        worst_dataset = max(regressions, key=lambda key: regressions[key])
        return f"Do not repeat {candidate.get('id') or 'this candidate'} without addressing {worst_dataset} regression."
    attribution = candidate.get("failure_attribution") or {}
    if attribution.get("primary_failure") == "cheap_proxy_rejected_before_full_training":
        return f"Do not send {candidate.get('id') or 'this candidate'} to full S3 until cheap proxy risk is cleared."
    if attribution.get("primary_failure") == "repairable_proxy_risk_before_full_training":
        return f"Repair the S2.5 patch for {candidate.get('id') or 'this candidate'} before discarding the idea."
    if attribution.get("primary_failure") == "ablation_no_effect":
        return f"Do not repeat {candidate.get('id') or 'this candidate'} until its ablation switch changes enabled-vs-disabled behavior."
    dragging = attribution.get("dragging_datasets") or []
    if dragging and isinstance(dragging[0], dict):
        return f"Do not repeat {candidate.get('id') or 'this candidate'} without addressing {dragging[0].get('dataset')} regression evidence."
    proxy_screen = candidate.get("proxy_screen") or {}
    proxy_deltas = proxy_screen.get("proxy_dataset_deltas") or {}
    if proxy_deltas:
        try:
            worst_dataset = min(proxy_deltas, key=lambda key: float(proxy_deltas[key]))
            if float(proxy_deltas[worst_dataset]) < 0:
                return f"Do not repeat {candidate.get('id') or 'this candidate'} without repairing proxy regression on {worst_dataset}."
        except (TypeError, ValueError):
            pass
    command_failure = proxy_screen.get("command_failure") or {}
    if command_failure.get("category"):
        return f"Do not rerun {candidate.get('id') or 'this candidate'} until proxy command failure {command_failure.get('category')} is fixed."
    metrics = candidate.get("metrics") or {}
    datasets = metrics.get("datasets") or {}
    if datasets and baseline_datasets:
        deltas: dict[str, float] = {}
        for dataset, value in datasets.items():
            try:
                deltas[str(dataset)] = float(value) - float(baseline_datasets.get(dataset, value))
            except (TypeError, ValueError):
                continue
        if deltas:
            worst_dataset = min(deltas, key=lambda key: deltas[key])
            if deltas[worst_dataset] < 0:
                return f"Do not repeat {candidate.get('id') or 'this candidate'} without a guard for {worst_dataset}."
    if not metrics:
        return "Do not rerun without fixing preflight, checkpoint, or evaluator failures."
        return "Do not repeat this candidate without a new mechanism and explicit ablation."


def _c2c_proxy_calibration_iteration(payload: dict[str, Any], *, iteration: int) -> dict[str, Any]:
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_datasets = baseline.get("datasets") if isinstance(baseline.get("datasets"), dict) else {}
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    candidates = [item for item in payload.get("candidate_results") or [] if isinstance(item, dict)]
    entries = [
        entry
        for entry in (_c2c_proxy_calibration_candidate(candidate, baseline_datasets, acceptance) for candidate in candidates)
        if entry
    ]
    false_positive_entries = [entry for entry in entries if entry.get("proxy_false_positive")]
    dataset_errors: dict[str, list[float]] = {}
    for entry in entries:
        for dataset, comparison in (entry.get("dataset_calibration") or {}).items():
            error = comparison.get("proxy_full_delta_error")
            if isinstance(error, (int, float)):
                dataset_errors.setdefault(dataset, []).append(float(error))
    return {
        "timestamp": now_utc(),
        "iteration": iteration,
        "acceptance_passed": bool(acceptance.get("passed")),
        "candidate_count": len(entries),
        "proxy_false_positive_count": len(false_positive_entries),
        "proxy_false_positive_rate": round(len(false_positive_entries) / len(entries), 4) if entries else 0.0,
        "dataset_error_summary": _proxy_dataset_error_summary(dataset_errors),
        "candidates": entries,
    }


def _c2c_proxy_calibration_candidate(candidate: dict[str, Any], baseline_datasets: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any] | None:
    proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
    if proxy.get("status") != "passed":
        return None
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    full_datasets = metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {}
    if not full_datasets:
        return None
    proxy_deltas = proxy.get("proxy_dataset_deltas") if isinstance(proxy.get("proxy_dataset_deltas"), dict) else {}
    if not proxy_deltas:
        proxy_baseline = proxy.get("proxy_baseline") or proxy.get("baseline_metrics") or {}
        proxy_scores = (proxy.get("metrics") or {}).get("datasets") or {}
        proxy_baseline_scores = proxy_baseline.get("datasets") if isinstance(proxy_baseline.get("datasets"), dict) else baseline_datasets
        proxy_deltas = {
            dataset: round(float(score) - float(proxy_baseline_scores[dataset]), 4)
            for dataset, score in proxy_scores.items()
            if dataset in proxy_baseline_scores and _is_number(score) and _is_number(proxy_baseline_scores[dataset])
        }
    full_deltas = {
        dataset: round(float(score) - float(baseline_datasets[dataset]), 4)
        for dataset, score in full_datasets.items()
        if dataset in baseline_datasets and _is_number(score) and _is_number(baseline_datasets[dataset])
    }
    dataset_calibration = {}
    for dataset in sorted(set(proxy_deltas) & set(full_deltas)):
        proxy_delta = float(proxy_deltas[dataset])
        full_delta = float(full_deltas[dataset])
        dataset_calibration[dataset] = {
            "proxy_delta": round(proxy_delta, 4),
            "full_delta": round(full_delta, 4),
            "proxy_full_delta_error": round(proxy_delta - full_delta, 4),
            "proxy_predicted_improvement": proxy_delta > 0,
            "full_improved": full_delta > 0,
            "proxy_mispredicted": (proxy_delta > 0) != (full_delta > 0),
        }
    full_passed = candidate.get("decision") == "candidate_win" and bool(acceptance.get("passed"))
    mechanism_type = candidate.get("mechanism_type")
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    return {
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "mechanism_type": mechanism_type or contract.get("mechanism_type"),
        "decision": candidate.get("decision"),
        "proxy_status": proxy.get("status"),
        "proxy_mean_delta": proxy.get("proxy_delta_vs_baseline"),
        "full_mean_delta": candidate.get("delta_vs_baseline"),
        "proxy_score": proxy.get("proxy_score"),
        "proxy_false_positive": not full_passed,
        "dataset_calibration": dataset_calibration,
        "mispredicted_datasets": [
            dataset
            for dataset, item in dataset_calibration.items()
            if item.get("proxy_mispredicted")
        ],
    }


def _c2c_proxy_calibration_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_entries = [
        candidate
        for iteration in iterations
        for candidate in iteration.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    false_positive_entries = [entry for entry in candidate_entries if entry.get("proxy_false_positive")]
    dataset_errors: dict[str, list[float]] = {}
    dataset_mispredictions: dict[str, int] = {}
    mechanism_counts: dict[str, dict[str, int]] = {}
    for entry in candidate_entries:
        mechanism = str(entry.get("mechanism_type") or "unknown")
        stats = mechanism_counts.setdefault(mechanism, {"count": 0, "false_positive_count": 0})
        stats["count"] += 1
        if entry.get("proxy_false_positive"):
            stats["false_positive_count"] += 1
        for dataset, comparison in (entry.get("dataset_calibration") or {}).items():
            error = comparison.get("proxy_full_delta_error")
            if isinstance(error, (int, float)):
                dataset_errors.setdefault(dataset, []).append(float(error))
            if comparison.get("proxy_mispredicted"):
                dataset_mispredictions[dataset] = dataset_mispredictions.get(dataset, 0) + 1
    return {
        "candidate_count": len(candidate_entries),
        "proxy_false_positive_count": len(false_positive_entries),
        "proxy_false_positive_rate": round(len(false_positive_entries) / len(candidate_entries), 4) if candidate_entries else 0.0,
        "dataset_error_summary": _proxy_dataset_error_summary(dataset_errors, mispredictions=dataset_mispredictions),
        "mechanism_false_positive_summary": {
            mechanism: {
                **stats,
                "false_positive_rate": round(stats["false_positive_count"] / stats["count"], 4) if stats["count"] else 0.0,
            }
            for mechanism, stats in sorted(mechanism_counts.items())
        },
    }


def _proxy_dataset_error_summary(dataset_errors: dict[str, list[float]], *, mispredictions: dict[str, int] | None = None) -> dict[str, Any]:
    mispredictions = mispredictions or {}
    return {
        dataset: {
            "mean_abs_proxy_full_delta_error": round(sum(abs(item) for item in values) / len(values), 4),
            "max_abs_proxy_full_delta_error": round(max(abs(item) for item in values), 4),
            "misprediction_count": mispredictions.get(dataset, 0),
            "count": len(values),
        }
        for dataset, values in sorted(dataset_errors.items())
        if values
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _compact_c2c_result_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if _looks_like_candidate_result(value):
            return _compact_candidate_result(value)
        return {str(key): _compact_c2c_result_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_c2c_result_payload(item) for item in value]
    return value


def _looks_like_candidate_result(value: dict[str, Any]) -> bool:
    return any(key in value for key in ["patch_result", "proxy_screen", "command_status", "run_state_path"]) and any(
        key in value for key in ["id", "candidate_id", "run_id"]
    )


def _compact_candidate_result(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = dict(candidate)
    if isinstance(compact.get("patch_result"), dict):
        compact["patch_result"] = _compact_patch_result_for_payload(compact["patch_result"])
    if isinstance(compact.get("proxy_screen"), dict):
        compact["proxy_screen"] = _compact_proxy_screen(compact["proxy_screen"])
    if isinstance(compact.get("commands"), dict):
        compact["commands"] = _compact_command_plan(compact["commands"])
    if isinstance(compact.get("command_logs"), list):
        compact["command_logs"] = _compact_event_logs(compact["command_logs"])
    if isinstance(compact.get("ablation"), dict):
        compact["ablation"] = _compact_ablation_payload(compact["ablation"])
    if isinstance(compact.get("failure_attribution"), dict):
        compact["failure_attribution"] = _compact_failure_attribution(compact["failure_attribution"])
    return compact


def _compact_patch_result_for_payload(patch_result: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "status",
        "reason",
        "candidate_id",
        "patch_status",
        "changed_files",
        "errors",
        "retryable",
        "failure_category",
    ]
    compact = {key: patch_result.get(key) for key in keep if key in patch_result}
    restore_state = patch_result.get("restore_state") or []
    if isinstance(restore_state, list) and restore_state:
        compact["restore_state_summary"] = [
            {
                "path": item.get("path"),
                "existed": item.get("existed"),
                "content_sha256": _sha256_text(str(item.get("content") or "")) if item.get("existed") else None,
                "content_chars": len(str(item.get("content") or "")) if item.get("existed") else 0,
            }
            for item in restore_state
            if isinstance(item, dict)
        ][:40]
        compact["restore_state_omitted"] = True
    for key in ["risk_check", "activation_check", "mechanism_review", "quality_score"]:
        if isinstance(patch_result.get(key), dict):
            compact[key] = _compact_c2c_result_payload(patch_result[key])
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


def _compact_proxy_screen(proxy_screen: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(proxy_screen, dict):
        return proxy_screen
    compact: dict[str, Any] = {}
    for key in [
        "enabled",
        "status",
        "mode",
        "reason",
        "repair_hint",
        "repair_route",
        "repair_mode",
        "proxy_effect_repair_contract",
        "metrics",
        "baseline_metrics",
        "proxy_baseline",
        "proxy_delta_vs_baseline",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "proxy_worst_dataset_regression",
        "proxy_score",
        "proxy_decision_mode",
        "soft_fail",
        "soft_flags",
        "signals",
        "patch_risk",
        "quality_repair",
        "patch_fingerprint",
    ]:
        if key in proxy_screen:
            compact[key] = proxy_screen[key]
    if isinstance(proxy_screen.get("command_failure"), dict):
        compact["command_failure"] = _compact_attempt(proxy_screen["command_failure"])
    if isinstance(proxy_screen.get("attempts"), list):
        compact["attempts"] = _compact_attempts(proxy_screen["attempts"])
    if isinstance(proxy_screen.get("commands"), list):
        compact["commands"] = [_shorten_text(str(command), 600) for command in proxy_screen["commands"]]
        compact["command_count"] = len(proxy_screen["commands"])
    if isinstance(compact.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(compact["proxy_effect_repair_contract"])
    return _compact_c2c_result_payload(compact)


def _proxy_screen_for_failure_attribution(proxy_screen: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "status",
        "reason",
        "repair_hint",
        "repair_route",
        "repair_mode",
        "proxy_delta_vs_baseline",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "proxy_worst_dataset_regression",
        "proxy_score",
        "soft_fail",
        "soft_flags",
    ]
    compact = {key: proxy_screen.get(key) for key in keep if key in proxy_screen}
    if isinstance(proxy_screen.get("command_failure"), dict):
        compact["command_failure"] = _compact_attempt(proxy_screen["command_failure"], stdout_chars=0, stderr_chars=0)
    if isinstance(proxy_screen.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(proxy_screen["proxy_effect_repair_contract"])
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_proxy_effect_repair_contract(contract: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "mode",
        "source",
        "goal",
        "reason",
        "repair_hint",
        "soft_flags",
        "proxy_delta_vs_baseline",
        "proxy_score",
        "proxy_worst_dataset_regression",
        "proxy_dataset_deltas",
        "proxy_dataset_regressions",
        "dragging_datasets",
        "improved_datasets",
        "patch_risk_labels",
        "changed_files",
        "config_override_keys",
        "repair_priorities",
        "forbidden",
    ]
    compact = {key: contract.get(key) for key in keep if key in contract}
    command_failure = contract.get("command_failure") or {}
    if isinstance(command_failure, dict):
        compact["command_failure"] = {
            key: _shorten_text(str(command_failure.get(key) or ""), 500)
            for key in ["category", "summary", "repair_hint"]
            if command_failure.get(key) not in (None, "")
        }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_attempts(attempts: list[Any], *, stdout_chars: int = 1000, stderr_chars: int = 1600) -> list[Any]:
    return [
        _compact_attempt(attempt, stdout_chars=stdout_chars, stderr_chars=stderr_chars)
        if isinstance(attempt, dict)
        else attempt
        for attempt in attempts
    ]


def _compact_attempt(attempt: dict[str, Any], *, stdout_chars: int = 1000, stderr_chars: int = 1600) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in [
        "step",
        "name",
        "status",
        "returncode",
        "category",
        "summary",
        "repair_hint",
        "elapsed_seconds",
        "timeout_seconds",
        "started_at",
        "completed_at",
    ]:
        if key in attempt:
            compact[key] = attempt[key]
    if "command" in attempt:
        compact["command"] = _shorten_text(str(attempt.get("command") or ""), 600)
    if "stdout" in attempt:
        compact["stdout_tail"] = str(attempt.get("stdout") or "")[-stdout_chars:]
        compact["stdout_chars"] = len(str(attempt.get("stdout") or ""))
    if "stderr" in attempt:
        compact["stderr_tail"] = str(attempt.get("stderr") or "")[-stderr_chars:]
        compact["stderr_chars"] = len(str(attempt.get("stderr") or ""))
    if isinstance(attempt.get("attempts"), list):
        compact["attempts"] = _compact_attempts(attempt["attempts"], stdout_chars=stdout_chars, stderr_chars=stderr_chars)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_command_plan(commands: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in commands.items():
        values = value if isinstance(value, list) else [value]
        compact[key] = [_shorten_text(str(item), 600) for item in values]
    return compact


def _compact_event_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_logs = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        compact = dict(item)
        if isinstance(compact.get("patch_result"), dict):
            compact["patch_result"] = _compact_patch_result_for_payload(compact["patch_result"])
        if isinstance(compact.get("proxy_screen"), dict):
            compact["proxy_screen"] = _compact_proxy_screen(compact["proxy_screen"])
        if isinstance(compact.get("ablation"), dict):
            compact["ablation"] = _compact_ablation_payload(compact["ablation"])
        compact_logs.append(compact)
    return compact_logs


def _compact_ablation_payload(ablation: dict[str, Any]) -> dict[str, Any]:
    compact = dict(ablation)
    if isinstance(compact.get("attempts"), list):
        compact["attempts"] = _compact_attempts(compact["attempts"])
    if isinstance(compact.get("eval_by_dataset"), dict):
        compact["eval_by_dataset"] = {
            dataset: _compact_attempt(attempt) if isinstance(attempt, dict) else attempt
            for dataset, attempt in compact["eval_by_dataset"].items()
        }
    if isinstance(compact.get("commands"), dict):
        compact["commands"] = _compact_command_plan(compact["commands"])
    return compact


def _compact_failure_attribution(attribution: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "primary_failure",
        "dragging_datasets",
        "improved_datasets",
        "sample_type_failures",
        "mixed_gain_patterns",
        "patch_risk",
        "ablation_evidence",
        "quality_repair",
        "proxy_effect_repair_contract",
    ]
    compact = {key: attribution.get(key) for key in keep if key in attribution}
    if isinstance(compact.get("proxy_screen"), dict):
        compact["proxy_screen"] = _compact_proxy_screen(compact["proxy_screen"])
    if isinstance(compact.get("proxy_effect_repair_contract"), dict):
        compact["proxy_effect_repair_contract"] = _compact_proxy_effect_repair_contract(compact["proxy_effect_repair_contract"])
    return compact


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "\n...[truncated]"


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _posthoc_items(value: Any, *, limit: int | None = None) -> list[str]:
    def render(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("failure_mode", "constraint", "action", "rule", "observed", "likely_root_cause", "reason"):
                if item.get(key):
                    return str(item[key])
            return json.dumps(item, ensure_ascii=False, sort_keys=True)
        return str(item)

    items: list[str] = []
    if isinstance(value, list):
        items.extend(render(item) for item in value)
    elif isinstance(value, dict):
        for group, group_value in value.items():
            if isinstance(group_value, list):
                items.extend(f"{group}: {render(item)}" for item in group_value)
            else:
                items.append(f"{group}: {render(group_value)}")
    elif value:
        items.append(render(value))
    items = [item for item in items if item]
    return items[:limit] if limit is not None else items


def _short_error(exc: Exception, *, limit: int = 500) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text
