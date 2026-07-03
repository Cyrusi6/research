import json
from pathlib import Path

from auto_research.stage_contracts import StageContractManager
from auto_research.workspace import init_workspace


def test_stage_contract_records_inputs_outputs_and_hashes(tmp_path: Path) -> None:
    config = {"project": {"workspace_root": str(tmp_path)}, "review": {"max_iterations": 2}}
    paths = init_workspace(config, "topic", project_id="proj_contract", simulate=True)
    manager = StageContractManager(paths.root)

    started = manager.stage_started("S2_plan", iteration=3)
    assert started["status"] == "running"
    assert started["iteration"] == 3
    assert any(item["path"] == "literature/direction.json" and not item["exists"] for item in started["resolved_inputs"])
    assert "literature/direction.json" in started["missing_inputs"]
    assert "literature/c2c/baseline_evidence.json" not in started["missing_inputs"]

    (paths.root / "plan" / "plan.yaml").write_text("hypotheses: []\n", encoding="utf-8")
    (paths.root / "plan" / "planner_decision.json").write_text("{}\n", encoding="utf-8")
    (paths.root / "plan" / "variant_contract.json").write_text("{}\n", encoding="utf-8")
    (paths.root / "plan" / "variant_fingerprint.json").write_text("{}\n", encoding="utf-8")
    completed = manager.stage_completed("S2_plan", artifacts=["plan/plan.yaml", "plan/planner_decision.json", "plan/variant_contract.json", "plan/variant_fingerprint.json"])

    assert completed["status"] == "completed"
    assert completed["output_hash"]
    assert any(item["path"] == "plan/plan.yaml" and item["sha256"] for item in completed["produced_outputs"])
    on_disk = json.loads((paths.root / "orchestration" / "stage_contracts" / "S2_plan.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "stage_contract_v2"


def test_stage_contract_keeps_c2c_inputs_conditional_for_generic_project(tmp_path: Path) -> None:
    config = {"project": {"workspace_root": str(tmp_path)}, "review": {"max_iterations": 2}}
    paths = init_workspace(config, "topic", project_id="proj_generic", simulate=True)
    manager = StageContractManager(paths.root)

    s3 = manager.stage_started("S3_experiment", iteration=1, config=config)

    assert "plan/plan.yaml" in s3["required_inputs"]
    assert "plan/candidate_ideas.json" not in s3["required_inputs"]
    assert "plan/short_loop_plan.yaml" not in s3["required_inputs"]
    assert "external/c2c_snapshot" not in s3["required_inputs"]
    assert s3["missing_inputs"] == ["plan/plan.yaml"]


def test_stage_contract_activates_c2c_small_loop_inputs(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "c2c": {"enabled": True},
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c", simulate=True)
    (paths.root / "plan").mkdir(exist_ok=True)
    (paths.root / "plan" / "plan.yaml").write_text("execution:\n  collector: c2c_small_loop\n", encoding="utf-8")
    manager = StageContractManager(paths.root)

    s3 = manager.stage_started("S3_experiment", iteration=1, config=config)

    assert "plan/candidate_ideas.json" in s3["required_inputs"]
    assert "plan/short_loop_plan.yaml" in s3["required_inputs"]
    assert "external/c2c_snapshot" in s3["required_inputs"]
    assert "plan/candidate_ideas.json" in s3["missing_inputs"]
