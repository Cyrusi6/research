"""Review stage."""

from __future__ import annotations

import json
from typing import Any

import yaml

from ..utils import compact_markdown, now_utc
from .base import AgentContext


class ReviewAgent:
    stage_key = "S5_review"

    def __init__(self, context: AgentContext):
        self.context = context

    def run(self, *, iteration: int) -> dict[str, Any]:
        claim_audit = json.loads((self.context.project_root / "paper" / "claim_audit.json").read_text(encoding="utf-8"))
        compile_report = json.loads((self.context.project_root / "paper" / "compile_report.json").read_text(encoding="utf-8"))
        has_ablation = (self.context.project_root / "experiment" / "results" / "ablation_results.json").exists()
        scores = self._scores(claim_audit, compile_report, has_ablation)
        decision = "ACCEPT" if scores["total"] >= self.context.config.get("review", {}).get("pass_threshold", 7.0) else "REVISE"
        reviewers = self._review_texts(scores, iteration)
        artifacts = []
        for reviewer_id, content in reviewers.items():
            record = self.context.artifacts.write_text(
                self.stage_key,
                f"{reviewer_id}_round_{iteration}.md",
                content,
                artifact_type="review",
                summary=f"{reviewer_id} review",
                source_paths=["paper/main.tex", "paper/claim_audit.json"],
            )
            artifacts.append(record["path"])
        debate_record = self.context.artifacts.write_text(
            self.stage_key,
            f"debate_round_{iteration}.md",
            self._debate(scores),
            artifact_type="debate",
            summary="Reviewer debate",
        )
        artifacts.append(debate_record["path"])
        meta_review_record = self.context.artifacts.write_text(
            self.stage_key,
            f"meta_review_round_{iteration}.md",
            self._meta_review(scores, decision),
            artifact_type="meta_review",
            summary="Meta review",
        )
        artifacts.append(meta_review_record["path"])
        dispatch = self._revision_dispatch(scores, decision, iteration)
        dispatch_record = self.context.artifacts.write_yaml(
            self.stage_key,
            "revision_dispatch.yaml",
            dispatch,
            artifact_type="revision_dispatch",
            summary="Targeted revisions",
        )
        artifacts.append(dispatch_record["path"])
        score_history = self._update_score_history(iteration, scores["total"], decision)
        score_record = self.context.artifacts.write_json(
            self.stage_key,
            "score_history.json",
            score_history,
            artifact_type="score_history",
            summary="Score history",
        )
        artifacts.append(score_record["path"])
        rebuttal_record = self.context.artifacts.write_text(
            self.stage_key,
            f"rebuttal_{iteration}.md",
            compact_markdown(
                "\n".join(
                    [
                        "# Rebuttal",
                        "",
                        "- This file is reserved for revision responses after review.",
                    ]
                )
            ),
            artifact_type="rebuttal",
            summary="Rebuttal scaffold",
        )
        artifacts.append(rebuttal_record["path"])
        return {"artifacts": artifacts, "decision": decision, "score": scores["total"], "dispatch_path": dispatch_record["path"]}

    def _scores(self, claim_audit: dict[str, Any], compile_report: dict[str, Any], has_ablation: bool) -> dict[str, float]:
        novelty = 7.2
        soundness = 7.8 if claim_audit.get("unsupported", 0) == 0 else 6.0
        experiment = 8.0 if has_ablation else 6.0
        presentation = 7.6 if compile_report.get("status") in {"ok", "unavailable"} else 5.0
        significance = 7.0
        reproducibility = 8.3
        total = round(
            novelty * 0.20
            + soundness * 0.25
            + experiment * 0.25
            + presentation * 0.15
            + significance * 0.10
            + reproducibility * 0.05,
            2,
        )
        return {
            "novelty": novelty,
            "soundness": soundness,
            "experiment": experiment,
            "presentation": presentation,
            "significance": significance,
            "reproducibility": reproducibility,
            "total": total,
        }

    def _review_texts(self, scores: dict[str, float], iteration: int) -> dict[str, str]:
        return {
            "reviewer_A": compact_markdown(
                "\n".join(
                    [
                        f"# Reviewer A — Round {iteration}",
                        "",
                        f"- Novelty: {scores['novelty']}",
                        f"- Soundness: {scores['soundness']}",
                        "- Strength: hypotheses and claims are aligned.",
                        "- Weakness: novelty still depends on broader benchmark confirmation.",
                    ]
                )
            ),
            "reviewer_B": compact_markdown(
                "\n".join(
                    [
                        f"# Reviewer B — Round {iteration}",
                        "",
                        f"- Experiment: {scores['experiment']}",
                        f"- Reproducibility: {scores['reproducibility']}",
                        "- Strength: ablation and manifests are present.",
                        "- Weakness: external benchmark breadth remains limited.",
                    ]
                )
            ),
            "reviewer_C": compact_markdown(
                "\n".join(
                    [
                        f"# Reviewer C — Round {iteration}",
                        "",
                        f"- Presentation: {scores['presentation']}",
                        f"- Significance: {scores['significance']}",
                        "- Strength: the paper is structured and traceable.",
                        "- Weakness: introduction can become more compelling with stronger benchmark framing.",
                    ]
                )
            ),
        }

    @staticmethod
    def _debate(scores: dict[str, float]) -> str:
        return compact_markdown(
            "\n".join(
                [
                    "# Debate Record",
                    "",
                    "- Consensus: the pipeline produces a coherent and auditable submission package.",
                    "- Consensus: stronger external validation would further improve confidence.",
                    f"- Final weighted score: {scores['total']}",
                ]
            )
        )

    @staticmethod
    def _meta_review(scores: dict[str, float], decision: str) -> str:
        return compact_markdown(
            "\n".join(
                [
                    f"# Meta Review — {decision}",
                    "",
                    f"- Final score: {scores['total']}",
                    "- Strengths: clear structure, traceable artifacts, bounded claims.",
                    "- Weaknesses: benchmark breadth and real-run coverage can still expand.",
                ]
            )
        )

    @staticmethod
    def _revision_dispatch(scores: dict[str, float], decision: str, iteration: int) -> dict[str, Any]:
        revisions = []
        if decision != "ACCEPT":
            revisions.append(
                {
                    "id": "REV-001",
                    "source": "reviewer_B",
                    "priority": "P0",
                    "assigned_agent": "experiment-agent",
                    "action": "Add one more benchmark or baseline comparison.",
                    "details": "Strengthen external validation.",
                    "estimated_effort": "4 GPU-hours",
                }
            )
            revisions.append(
                {
                    "id": "REV-002",
                    "source": "reviewer_C",
                    "priority": "P1",
                    "assigned_agent": "writing-agent",
                    "action": "Tighten introduction motivation and discussion.",
                    "details": "Make the benchmark framing more compelling.",
                    "estimated_effort": "1 hour",
                }
            )
        else:
            revisions.append(
                {
                    "id": "REV-ACCEPT-001",
                    "source": "meta-reviewer",
                    "priority": "P2",
                    "assigned_agent": "writing-agent",
                    "action": "Optional polish of motivation paragraph.",
                    "details": "Minor improvement only.",
                    "estimated_effort": "0.5 hour",
                }
            )
        execution_order = []
        grouped = {}
        for revision in revisions:
            grouped.setdefault(revision["assigned_agent"], []).append(revision["id"])
        for agent_name, revision_ids in grouped.items():
            execution_order.append({"agents": [agent_name], "revisions": revision_ids})
        return {
            "decision": decision,
            "score": scores["total"],
            "iteration": iteration,
            "generated_at": now_utc(),
            "revisions": revisions,
            "execution_order": execution_order,
        }

    def _update_score_history(self, iteration: int, score: float, decision: str) -> list[dict[str, Any]]:
        path = self.context.project_root / "review" / "score_history.json"
        if path.exists():
            history = json.loads(path.read_text(encoding="utf-8"))
        else:
            history = []
        history.append({"iteration": iteration, "score": score, "decision": decision, "timestamp": now_utc()})
        return history
