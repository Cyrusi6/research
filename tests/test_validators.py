import json
from pathlib import Path

from auto_research.c2c import default_c2c_ideas
from auto_research.utils import sha256_file
from auto_research.validators import run_stage_gate
from auto_research.workspace import init_workspace


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "writing": {"claim_verification": {"min_pass_rate": 0.8}, "require_compile": False},
    }


def test_s1_gate_returns_structured_retry_for_missing_ideas(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate", simulate=True)

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path))
    payload = report.to_dict()

    assert payload["schema_version"] == "stage_gate_v1"
    assert payload["status"] == "NEEDS_RETRY"
    assert payload["passed"] is False
    assert payload["checks"][0]["name"] == "ideas_json_exists"


def test_s1_gate_retries_failed_novelty_audit(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s1_novelty_gate", simulate=True)
    (paths.root / "references" / "papers").mkdir(parents=True, exist_ok=True)
    (paths.root / "references" / "papers" / "manifest.json").write_text(json.dumps({"papers": [{"id": "p"}]}), encoding="utf-8")
    (paths.root / "literature").mkdir(parents=True, exist_ok=True)
    (paths.root / "literature" / "ideas.json").write_text(
        json.dumps(
            [
                {
                    "id": "repeat_direction",
                    "title": "Repeat direction",
                    "selected": True,
                    "hypothesis": "repeat",
                    "novelty_score": 7,
                    "feasibility_score": 7,
                    "evidence_refs": [{"source_type": "paper", "source_label": "p", "claim": "x"}],
                    "counterevidence_refs": [{"source_type": "paper", "source_label": "p", "claim": "y"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    (paths.root / "literature" / "novelty_audit.json").write_text(
        json.dumps(
            [
                {
                    "status": "ok",
                    "enabled": True,
                    "passed": False,
                    "threshold": 0.6,
                    "audit": {"novelty_score": 0.2, "max_similarity_score": 0.9},
                }
            ]
        ),
        encoding="utf-8",
    )

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "s1_novelty_audit" and check["status"] == "NEEDS_RETRY" for check in report["checks"])


def test_s2_gate_passes_c2c_plan_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate", simulate=True)
    plan_dir = paths.root / "plan"
    (plan_dir / "plan.yaml").write_text(
        """
hypotheses:
  - id: h1
baselines:
  - name: b1
  - name: b2
datasets:
  - name: d1
task_graph: {}
resource_budget: {}
execution:
  collector: c2c_small_loop
  min_delta_to_pass: 0.1
  max_dataset_regression: 2.0
acceptance_criteria:
  minimum_mean_delta: 0.1
  coverage_diagnostics_required: true
  matched_coverage_ablation_required: true
ablation_matrix:
  - experiment: disable selected mechanism
  - experiment: matched transfer coverage control
    matched_coverage_ablation:
      required: true
reviewer_risk_controls:
  top_concerns: []
""",
        encoding="utf-8",
    )
    (plan_dir / "short_loop_plan.yaml").write_text("run: true\n", encoding="utf-8")
    ideas = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})
    (plan_dir / "candidate_ideas.json").write_text(json.dumps(ideas), encoding="utf-8")

    config = _config(tmp_path)
    config["code_patch"] = {"validation": {"gate_mode": "strict"}}
    report = run_stage_gate("S2_plan", paths.root, config).to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "c2c_runtime_resource_selection_deferred" for check in report["checks"])
    assert any(check["name"] == "c2c_mechanism_novelty_gate" and check["status"] == "PASS" for check in report["checks"])
    assert any(check["name"] == "c2c_implementation_scope_gate" and check["status"] == "PASS" for check in report["checks"])


def test_s2_gate_retries_c2c_local_tuning_idea(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_local_tuning", simulate=True)
    plan_dir = paths.root / "plan"
    (plan_dir / "plan.yaml").write_text(
        """
hypotheses:
  - id: h1
baselines:
  - name: b1
  - name: b2
datasets:
  - name: d1
task_graph: {}
resource_budget: {}
execution:
  collector: c2c_small_loop
  min_delta_to_pass: 0.1
  max_dataset_regression: 2.0
acceptance_criteria:
  minimum_mean_delta: 0.1
reviewer_risk_controls:
  top_concerns: []
""",
        encoding="utf-8",
    )
    (plan_dir / "short_loop_plan.yaml").write_text("run: true\n", encoding="utf-8")
    (plan_dir / "candidate_ideas.json").write_text(
        json.dumps(
            [
                {
                    "id": "local_topk_tuning",
                    "title": "Local top-k tuning",
                    "selected": True,
                    "novelty_score": 6,
                    "feasibility_score": 9,
                    "experiment_contract": {
                        "primary_metric": "three_dataset_mean",
                        "baseline": "base",
                        "config_overrides": {
                            "train": {"model": {"soft_alignment_top_k": 2, "soft_alignment_confidence_floor": 0.2}},
                            "eval": {"model": {"rosetta_config": {"soft_alignment_top_k": 2}}},
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config = _config(tmp_path)
    config["code_patch"] = {"validation": {"gate_mode": "strict"}}
    report = run_stage_gate("S2_plan", paths.root, config).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    novelty = next(check for check in report["checks"] if check["name"] == "c2c_mechanism_novelty_gate")
    assert novelty["status"] == "NEEDS_RETRY"


def test_s2_gate_retries_large_scope_without_decomposition(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_large_scope", simulate=True)
    plan_dir = paths.root / "plan"
    (plan_dir / "plan.yaml").write_text(
        """
hypotheses:
  - id: h1
baselines:
  - name: b1
  - name: b2
datasets:
  - name: d1
task_graph: {}
resource_budget: {}
execution:
  collector: c2c_small_loop
  min_delta_to_pass: 0.1
  max_dataset_regression: 2.0
acceptance_criteria:
  minimum_mean_delta: 0.1
reviewer_risk_controls:
  top_concerns: []
""",
        encoding="utf-8",
    )
    (plan_dir / "short_loop_plan.yaml").write_text("run: true\n", encoding="utf-8")
    idea = default_c2c_ideas("topic", {"name": "base", "mean": 50.0, "datasets": {}})[0]
    idea["selected"] = True
    idea["implementation_scope"] = "large"
    idea["implementation_plan"] = {"scope": "large", "integration_points": [], "smoke_tests": []}
    idea["integration_points"] = []
    idea["smoke_tests"] = []
    idea["decomposition_plan"] = []
    (plan_dir / "candidate_ideas.json").write_text(json.dumps([idea]), encoding="utf-8")

    config = _config(tmp_path)
    config["code_patch"] = {"validation": {"gate_mode": "strict"}}
    report = run_stage_gate("S2_plan", paths.root, config).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    scope = next(check for check in report["checks"] if check["name"] == "c2c_implementation_scope_gate")
    assert scope["status"] == "NEEDS_RETRY"
    assert "missing decomposition_plan" in scope["details"]["blocked"][0]["blocked_reasons"]


def test_s3_gate_fails_below_acceptance_threshold(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate", simulate=True)
    results_dir = paths.root / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ablation_results.json").write_text("{}\n", encoding="utf-8")
    (results_dir / "hypothesis_verification.md").write_text("ok\n", encoding="utf-8")
    (results_dir / "main_results.json").write_text(
        json.dumps(
            {
                "baseline": {"mean": 50.0},
                "candidate_results": [],
                "acceptance": {"passed": False, "reason": "below baseline"},
                "best_candidate": {"metrics": {"mean": 49.9}},
            }
        ),
        encoding="utf-8",
    )

    report = run_stage_gate("S3_experiment", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "FAIL"
    assert "did not clear acceptance" in report["reason"]


def test_s3_gate_fails_when_s2_5_artifact_lock_changes(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s3_artifact_lock", simulate=True)
    results_dir = paths.root / "experiment" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = paths.root / "plan" / "code_patches" / "winner"
    patch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.root / "plan" / "code_patches" / "patch_manifest.json"
    patch_path = patch_dir / "patch.json"
    contract_path = patch_dir / "implementation_contract.json"

    manifest_path.write_text(json.dumps({"status": "ok", "selected_candidate_id": "winner"}), encoding="utf-8")
    patch_path.write_text(json.dumps({"candidate_id": "winner", "operations": []}), encoding="utf-8")
    contract_path.write_text(json.dumps({"candidate_id": "winner"}), encoding="utf-8")
    locked_patch_sha = sha256_file(patch_path)
    patch_path.write_text(json.dumps({"candidate_id": "winner", "operations": [{"op": "changed"}]}), encoding="utf-8")

    (results_dir / "ablation_results.json").write_text("{}\n", encoding="utf-8")
    (results_dir / "hypothesis_verification.md").write_text("ok\n", encoding="utf-8")
    (results_dir / "main_results.json").write_text(
        json.dumps({"candidate_results": [{"id": "winner", "metrics": {"mean": 51.0}}]}),
        encoding="utf-8",
    )
    (results_dir / "s3_candidate_selection.json").write_text(
        json.dumps(
            {
                "selected_candidate_id": "winner",
                "patch_manifest": {
                    "rel_path": "plan/code_patches/patch_manifest.json",
                    "sha256": sha256_file(manifest_path),
                },
                "selected_patch": {
                    "rel_path": "plan/code_patches/winner/patch.json",
                    "sha256": locked_patch_sha,
                },
                "selected_implementation_contract": {
                    "rel_path": "plan/code_patches/winner/implementation_contract.json",
                    "sha256": sha256_file(contract_path),
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_stage_gate("S3_experiment", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "FAIL"
    lock_check = next(check for check in report["checks"] if check["name"] == "s3_s2_5_artifact_lock_sha256")
    assert lock_check["status"] == "FAIL"
    assert lock_check["details"]["mismatches"][0]["name"] == "selected_patch"
