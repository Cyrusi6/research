import json
from pathlib import Path

from auto_research.workspace import init_workspace


def test_init_workspace_creates_stage_dirs_and_reference_manifest(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 3},
    }

    paths = init_workspace(config, "test topic", project_id="proj_demo", simulate=True)

    assert paths.root == tmp_path / "proj_demo"
    assert (paths.root / "references" / "papers" / "manifest.json").exists()
    assert (paths.root / "intake" / "stage_manifest.json").exists()
    assert (paths.root / "literature" / "stage_manifest.json").exists()
    assert (paths.root / "paper" / "stage_manifest.json").exists()
    assert (paths.root / "meta" / "registry.yaml").exists()
    assert (paths.root / "meta" / "codex_sessions.yaml").exists()
    s1_contract = json.loads((paths.root / "orchestration" / "stage_contracts" / "S1_literature.json").read_text(encoding="utf-8"))
    assert s1_contract["schema_version"] == "stage_contract_v2"
    assert s1_contract["stage_key"] == "S1_literature"
    assert "meta/project_config.yaml" in s1_contract["required_inputs"]
    assert "experiment/results/failure_feedback.json" not in s1_contract["required_inputs"]
    assert "literature/direction.json" in s1_contract["declared_outputs"]
    state = json.loads((paths.root / "orchestration" / "state.json").read_text(encoding="utf-8"))
    assert state["project_id"] == "proj_demo"
    assert state["current_stage"] == "S0_intake"
    assert state["stages"]["S0_intake"]["status"] == "pending"
    assert state["stages"]["S1_literature"]["status"] == "pending"
