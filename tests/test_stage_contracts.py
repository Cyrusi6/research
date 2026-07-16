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

    (paths.root / "plan" / "trial_spec.json").write_text("{}\n", encoding="utf-8")
    (paths.root / "plan" / "planner_decision.json").write_text("{}\n", encoding="utf-8")
    (paths.root / "plan" / "variant.json").write_text("{}\n", encoding="utf-8")
    (paths.root / "plan" / "variant_fingerprint.json").write_text("{}\n", encoding="utf-8")
    completed = manager.stage_completed("S2_plan", artifacts=["plan/trial_spec.json", "plan/planner_decision.json", "plan/variant.json", "plan/variant_fingerprint.json"])

    assert completed["status"] == "completed"
    assert completed["output_hash"]
    assert any(item["path"] == "plan/trial_spec.json" and item["sha256"] for item in completed["produced_outputs"])
    on_disk = json.loads((paths.root / "orchestration" / "stage_contracts" / "S2_plan.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "stage_contract_v2"


def test_stage_contract_keeps_c2c_inputs_conditional_for_generic_project(tmp_path: Path) -> None:
    config = {"project": {"workspace_root": str(tmp_path)}, "review": {"max_iterations": 2}}
    paths = init_workspace(config, "topic", project_id="proj_generic", simulate=True)
    manager = StageContractManager(paths.root)

    s3 = manager.stage_started("S3_experiment", iteration=1, config=config)

    assert "plan/trial_spec.json" in s3["required_inputs"]
    assert "plan/s2_planner/candidate_pool.json" not in s3["required_inputs"]
    assert "external/c2c_snapshot" not in s3["required_inputs"]
    assert set(s3["missing_inputs"]) == {"literature/direction.json", "plan/variant.json", "plan/trial_spec.json", "plan/code_patches/implementation_contract.json", "plan/code_patches/patch_manifest.json"}


def test_stage_contract_declares_c2c_s1_quality_outputs(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "c2c": {"enabled": True},
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_s1_outputs", simulate=True)
    manager = StageContractManager(paths.root)

    s1 = manager.stage_started("S1_literature", iteration=1, config=config)

    assert "literature/c2c/evidence_request_plan.json" in s1["required_outputs"]
    assert "literature/c2c/direction_candidate_scorecard.json" in s1["required_outputs"]
    assert "literature/c2c/evidence_quality_score.json" in s1["required_outputs"]
    assert "literature/c2c/evidence_retrieval_trace.json" in s1["required_outputs"]
    assert "literature/c2c/direction_fingerprint.json" in s1["required_outputs"]


def test_stage_contract_declares_c2c_s2_planner_outputs(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "c2c": {"enabled": True},
        "code_patch": {"enabled": True},
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_s2_outputs", simulate=True)
    manager = StageContractManager(paths.root)

    s2 = manager.stage_started("S2_plan", iteration=1, config=config)

    assert "plan/s2_planner/candidate_pool.json" in s2["required_outputs"]
    assert "plan/s2_planner/feedback_context.json" in s2["required_outputs"]
    assert "plan/s2_planner/adaptive_policy.json" in s2["required_outputs"]
    assert "plan/s2_planner/variant_scorecard.json" in s2["required_outputs"]
    assert "plan/s2_planner/score_adjustment_report.json" in s2["required_outputs"]
    assert "plan/variant.json" in s2["required_outputs"]
    assert "plan/trial_spec.json" in s2["required_outputs"]
    assert "plan/s2_planner/planner_gate_report.json" in s2["required_outputs"]
    assert "plan/code_patches/implementation_contract.json" in s2["required_outputs"]
    assert "plan/code_patches/patch_gate_report.json" in s2["required_outputs"]


def test_stage_contract_activates_c2c_small_loop_inputs(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "c2c": {"enabled": True},
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c", simulate=True)
    (paths.root / "plan").mkdir(exist_ok=True)
    (paths.root / "plan" / "trial_spec.json").write_text(
        json.dumps({"execution_contract": {"runtime_config": {"collector": "c2c_small_loop"}}}),
        encoding="utf-8",
    )
    manager = StageContractManager(paths.root)

    s3 = manager.stage_started("S3_experiment", iteration=1, config=config)

    assert "external/c2c_snapshot" in s3["required_inputs"]
    assert "plan/s2_planner/candidate_pool.json" not in s3["required_inputs"]
    assert "external/c2c_snapshot" in s3["missing_inputs"]


def test_stage_contract_switches_c2c_s3_outputs_for_bootstrap(tmp_path: Path) -> None:
    config = {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 1},
        "c2c": {"enabled": True},
        "orchestration": {"profile": "bootstrap"},
    }
    paths = init_workspace(config, "topic", project_id="proj_c2c_bootstrap_contract", simulate=True)
    (paths.root / "plan" / "trial_spec.json").write_text(
        json.dumps({"execution_contract": {"runtime_config": {"collector": "c2c_small_loop"}}}),
        encoding="utf-8",
    )

    s3 = StageContractManager(paths.root).stage_started("S3_experiment", iteration=1, config=config)

    assert "meta/research_events.sqlite3" in s3["required_outputs"]
    assert "meta/research_state.json" in s3["required_outputs"]
    assert "meta/route_outcome.json" in s3["required_outputs"]
    assert "experiment/results/trial_result.json" in s3["required_outputs"]
    assert "experiment/results/bootstrap_proxy_completion.json" in s3["optional_outputs"]
    assert "experiment/results/bootstrap_proxy_completion.json" not in s3["required_outputs"]
    assert "experiment/results/c2c_full_s3_worthiness.json" not in s3["required_outputs"]
    assert "experiment/results/c2c_full_s3_decision.json" not in s3["required_outputs"]


def test_stage_contract_uses_authoritative_trial_projection_for_writing(tmp_path: Path) -> None:
    config = {"project": {"workspace_root": str(tmp_path)}, "review": {"max_iterations": 1}}
    paths = init_workspace(config, "topic", project_id="proj_s4_authority", simulate=True)

    s4 = StageContractManager(paths.root).stage_started("S4_writing", iteration=1, config=config)

    assert set(s4["required_inputs"]) == {
        "literature/survey.md",
        "meta/research_state.json",
        "experiment/results/trial_result.json",
    }
    assert "experiment/results/main_results.json" in s4["optional_inputs"]
    assert "experiment/results/ablation_results.json" in s4["optional_inputs"]
    assert "experiment/results/hypothesis_verification.md" in s4["optional_inputs"]
