"""Experiment stage."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..adapters.runner import ExperimentRunner
from ..failure_log import FailureLogManager
from ..itr_ideas import screening_summary_markdown
from ..utils import compact_markdown
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
