"""Idea review agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import compact_markdown
from .base import AgentContext


class IdeaReviewAgent:
    stage_key = "S1_literature"

    def __init__(self, context: AgentContext):
        self.context = context

    def review(self, *, ideas: list[dict[str, Any]], theme_map: dict[str, Any] | None = None) -> dict[str, Any]:
        target_rsum = float(self.context.config.get("ideation", {}).get("challenge_target_rsum", 507.0))
        reviewed = []
        for idea in ideas:
            title = idea.get("title", "")
            title_lc = title.lower()
            structural = 8 if any(key in title_lc for key in ["consistency", "relevance", "dual", "transformer", "token-guided"]) else 5
            feasibility = idea.get("feasibility_score", 5)
            challenge = self._challenge_score(title_lc)
            decision = "accept" if challenge >= 7 and structural >= 7 and feasibility >= 7 else "reject"
            reviewed.append(
                {
                    **idea,
                    "idea_review": {
                        "challenge_target_rsum": target_rsum,
                        "structural_score": structural,
                        "challenge_score": challenge,
                        "decision": decision,
                        "reason": self._reason(title_lc, challenge, structural),
                    },
                }
            )
        reviewed.sort(
            key=lambda item: (
                0 if item["idea_review"]["decision"] == "accept" else 1,
                -item["idea_review"]["challenge_score"],
                -item["idea_review"]["structural_score"],
            )
        )
        for idx, item in enumerate(reviewed):
            item["selected"] = idx == 0 and item["idea_review"]["decision"] == "accept"
        payload = {
            "target_rsum": target_rsum,
            "accepted_count": sum(1 for item in reviewed if item["idea_review"]["decision"] == "accept"),
            "ideas": reviewed,
        }
        review_record = self.context.artifacts.write_json(
            self.stage_key,
            "idea_review.json",
            payload,
            artifact_type="idea_review",
            summary="Idea review decisions",
        )
        self.context.artifacts.write_text(
            self.stage_key,
            "idea_review.md",
            self._render_markdown(payload),
            artifact_type="idea_review",
            summary="Human-readable idea review",
            source_paths=[review_record["path"]],
        )
        return payload

    @staticmethod
    def _challenge_score(title_lc: str) -> int:
        if any(key in title_lc for key in ["semantic importance consistency", "relevance", "dual transformer", "token-guided", "csic", "uarda", "tgdt"]):
            return 9
        if any(key in title_lc for key in ["triplet-infonce", "triplet-infonce hybrid", "triplet-infonce hybrid alignment", "triplet-infonce hybrid"]):
            return 6
        if any(key in title_lc for key in ["weighted route calibration", "dynamic attention", "global alignment", "multi-scale"]):
            return 5
        return 4

    @staticmethod
    def _reason(title_lc: str, challenge: int, structural: int) -> str:
        if challenge >= 8:
            return "This direction changes the relevance or alignment mechanism itself and is more likely to exceed the current ceiling."
        if structural < 7:
            return "This looks too close to a local calibration tweak and is unlikely to beat a 507-level target."
        return "This is implementable but does not look strong enough to challenge the target ceiling."

    @staticmethod
    def _render_markdown(payload: dict[str, Any]) -> str:
        lines = ["# Idea Review", ""]
        lines.append(f"- Target rsum: {payload['target_rsum']}")
        lines.append(f"- Accepted ideas: {payload['accepted_count']}")
        lines.append("")
        for item in payload["ideas"]:
            review = item["idea_review"]
            lines.append(f"## {item['title']}")
            lines.append(f"- Decision: {review['decision']}")
            lines.append(f"- Structural score: {review['structural_score']}")
            lines.append(f"- Challenge score: {review['challenge_score']}")
            lines.append(f"- Reason: {review['reason']}")
            lines.append("")
        return compact_markdown("\n".join(lines))
