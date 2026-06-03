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
    assert record["type"] == "note"
    assert record["artifact_type"] == "note"
    assert record["created_by"] == "plan-agent"
    assert record["validator"] == "s2_note_schema_v1"
    assert record["status"] == "committed"
    assert record["sha256"]
    assert manifest["schema_version"] == "stage_manifest_v1"
    assert manifest["stage_key"] == "S2_plan"
    assert manifest["artifact_count"] == 1
    assert any(item["path"] == "plan/example.md" for item in manifest["artifacts"])


def test_artifact_manifest_supports_override_source_and_dedup(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
    }
    paths = init_workspace(config, "artifact topic", project_id="proj_artifacts", simulate=True)
    manager = ArtifactManager(paths.root)

    first = manager.write_json(
        "S1_literature",
        "ideas.json",
        {"idea": "v1"},
        artifact_type="ideas",
        source_paths=["references/papers/demo.pdf"],
        created_by="literature-scout",
        validator="custom_ideas_schema_v1",
    )
    second = manager.write_json(
        "S1_literature",
        "ideas.json",
        {"idea": "v2"},
        artifact_type="ideas",
        source_paths=["literature/paper_cards.json"],
        created_by="method-inventor",
        validator="custom_ideas_schema_v2",
    )

    manifest = json.loads((paths.root / "literature" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert first["path"] == second["path"] == "literature/ideas.json"
    assert manifest["artifact_count"] == 1
    entry = manifest["artifacts"][0]
    assert entry["created_by"] == "method-inventor"
    assert entry["validator"] == "custom_ideas_schema_v2"
    assert entry["source_paths"] == ["literature/paper_cards.json"]


def test_artifact_manifest_upgrades_legacy_entries(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
    }
    paths = init_workspace(config, "artifact topic", project_id="proj_artifacts", simulate=True)
    legacy_path = paths.root / "plan" / "legacy.md"
    legacy_path.write_text("legacy\n", encoding="utf-8")
    (paths.root / "plan" / "stage_manifest.json").write_text(
        json.dumps(
            {
                "stage": "plan",
                "artifacts": [
                    {
                        "path": "plan/legacy.md",
                        "artifact_type": "plan",
                        "created_at": "2026-05-22T00:00:00+00:00",
                        "status": "committed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = ArtifactManager(paths.root).load_manifest("S2_plan")
    entry = manifest["artifacts"][0]
    assert manifest["schema_version"] == "stage_manifest_v1"
    assert manifest["artifact_count"] == 1
    assert entry["schema_version"] == "artifact_record_v1"
    assert entry["type"] == "plan"
    assert entry["created_by"] == "plan-agent"
    assert entry["validator"] == "s2_plan_schema_v1"
    assert entry["sha256"]
