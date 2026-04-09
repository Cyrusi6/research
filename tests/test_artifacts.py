import json
from pathlib import Path

from auto_research.artifacts import ArtifactManager
from auto_research.workspace import init_workspace


def test_artifact_manager_registers_committed_file(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
    }
    paths = init_workspace(config, "artifact topic", project_id="proj_artifacts", simulate=True)
    manager = ArtifactManager(paths.root)

    record = manager.write_text(
        "S2_plan",
        "example.md",
        "# Example\n",
        artifact_type="note",
        summary="Example artifact",
    )

    manifest = json.loads((paths.root / "plan" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert record["path"] == "plan/example.md"
    assert any(item["path"] == "plan/example.md" for item in manifest["artifacts"])
