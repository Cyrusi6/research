from pathlib import Path

from auto_research.artifacts import ArtifactManager
from auto_research.agents.base import AgentContext
from auto_research.agents.idea_review import IdeaReviewAgent
from auto_research.llm import ModelClient
from auto_research.workspace import init_workspace


def test_idea_review_rejects_low_ceiling_tweaks(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "ideation": {"challenge_target_rsum": 507.0},
        "llm": {"provider": "mock", "use_real_api": False},
    }
    paths = init_workspace(config, "idea review", project_id="proj_review", simulate=True)
    context = AgentContext(
        project_root=paths.root,
        config=config,
        artifacts=ArtifactManager(paths.root),
        llm=ModelClient(config, project_root=paths.root),
    )
    ideas = [
        {"id": "a", "title": "Weighted Route Calibration", "feasibility_score": 9, "selected": False},
        {"id": "b", "title": "Cross-Modal Semantic Importance Consistency", "feasibility_score": 8, "selected": False},
    ]
    payload = IdeaReviewAgent(context).review(ideas=ideas, theme_map={})

    decisions = {item["id"]: item["idea_review"]["decision"] for item in payload["ideas"]}
    assert decisions["a"] == "reject"
    assert decisions["b"] == "accept"
