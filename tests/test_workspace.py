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
    assert (paths.root / "literature" / "stage_manifest.json").exists()
    assert (paths.root / "paper" / "stage_manifest.json").exists()
    assert (paths.root / "meta" / "registry.yaml").exists()
    assert (paths.root / "meta" / "codex_sessions.yaml").exists()
