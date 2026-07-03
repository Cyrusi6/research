import json
from pathlib import Path

from auto_research.utils import sha256_file
from auto_research.validators import run_stage_gate


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path)},
        "c2c": {"small_loop": {"proxy_screen": {"enabled": True}}},
        "review": {"max_iterations": 2},
    }


def _write_required_outputs(root: Path, *, proxy_screen: dict | None = None) -> None:
    results = root / "experiment" / "results"
    results.mkdir(parents=True, exist_ok=True)
    candidate = {
        "id": "v1",
        "decision": "candidate_win",
        "command_status": "ok",
        "metrics": {"mean": 51.0, "datasets": {"mmlu-redux": 51.0}},
        "worst_dataset_regression": 0.0,
        "proxy_screen": proxy_screen or {"enabled": True, "status": "passed"},
    }
    (results / "main_results.json").write_text(
        json.dumps(
            {
                "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
                "candidate_results": [candidate],
                "best_candidate": candidate,
                "acceptance": {"passed": True, "baseline_mean": 50.0, "min_delta_to_pass": 0.1, "max_dataset_regression": 2.0},
            }
        ),
        encoding="utf-8",
    )
    (results / "ablation_results.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (results / "hypothesis_verification.md").write_text("ok\n", encoding="utf-8")


def _write_proxy_contracts(root: Path, *, decision: str = "proxy_pass", include_worthiness: bool = False, worthiness_score: float = 0.7) -> None:
    results = root / "experiment" / "results"
    results.mkdir(parents=True, exist_ok=True)
    policy_hash = "policyhash"
    baseline_hash = "baselinehash"
    (results / "c2c_proxy_baseline_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_proxy_baseline_fingerprint_v1",
                "created_at": "2026-07-03T00:00:00+00:00",
                "fingerprint_hash": baseline_hash,
                "inputs": {
                    "snapshot_path": "external/c2c_snapshot",
                    "snapshot_manifest_hash": "a",
                    "plan_hash": "b",
                    "execution_hash": "c",
                    "proxy_config_hash": "d",
                    "eval_recipe_hash": "e",
                    "dataset_signature": {"eval_datasets": ["mmlu-redux"], "eval_limit": 64, "dataset_root": "/tmp"},
                    "baseline_model_signature": {"model_map_hash": "f", "baseline_config_hash": "g"},
                    "evaluator_file_hashes": {},
                    "s2_5_locks": {},
                    "env_signature": {"python": "/usr/bin/python3"},
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "c2c_proxy_cache_report.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_proxy_cache_report_v1",
                "created_at": "2026-07-03T00:00:00+00:00",
                "cache_status": "hit",
                "reason": "fingerprint_hash_match",
                "expected_fingerprint_hash": baseline_hash,
                "actual_fingerprint_hash": baseline_hash,
                "baseline_cache_path": "experiment/results/c2c_proxy_baseline.json",
                "action": "reuse",
            }
        ),
        encoding="utf-8",
    )
    (results / "c2c_proxy_calibration_policy.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_proxy_calibration_policy_v1",
                "created_at": "2026-07-03T00:00:00+00:00",
                "history_count": 0,
                "min_history_for_adjustment": 3,
                "adjustments_active": False,
                "mechanism_false_positive_rate": 0.0,
                "integration_point_false_positive_rate": 0.0,
                "dataset_misprediction_risk": {},
                "recommended_min_proxy_mean_delta": -0.3,
                "recommended_max_proxy_dataset_regression": 1.5,
                "neutral_proxy_allowed": True,
                "reason_codes": [],
                "policy_hash": "calibrationhash",
            }
        ),
        encoding="utf-8",
    )
    (results / "c2c_effective_proxy_policy.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_effective_proxy_policy_v1",
                "created_at": "2026-07-03T00:00:00+00:00",
                "static_policy": {"min_proxy_mean_delta": -0.3, "max_proxy_dataset_regression": 1.5, "allow_neutral_proxy_full_s3": True},
                "calibration_adjustments": {},
                "effective_policy": {
                    "min_proxy_mean_delta": -0.3,
                    "max_proxy_dataset_regression": 1.5,
                    "allow_neutral_proxy_full_s3": True,
                    "neutral_proxy_min_delta": -0.1,
                    "neutral_proxy_max_dataset_regression": 0.25,
                    "full_s3_worthiness_min_score": 0.60,
                },
                "reason_codes": [],
                "policy_hash": policy_hash,
            }
        ),
        encoding="utf-8",
    )
    mean = 49.95 if decision == "neutral_proxy_full_s3" else 50.4
    delta = -0.05 if decision == "neutral_proxy_full_s3" else 0.4
    report = {
        "schema_version": "c2c_proxy_decision_report_v1",
        "created_at": "2026-07-03T00:00:00+00:00",
        "candidate_id": "v1",
        "variant_id": "v1",
        "variant_fingerprint": "fp",
        "baseline_fingerprint_hash": baseline_hash,
        "effective_policy_hash": policy_hash,
        "proxy_metrics": {"mean": mean, "datasets": {"mmlu-redux": mean}},
        "paired_baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "deltas": {"mean_delta": delta, "worst_dataset_regression": 0.0, "dataset_deltas": {"mmlu-redux": delta}},
        "static_checks": {"patch_gate_passed": True, "no_eval_code_change": True, "has_executable_change": True, "activation_smoke_passed": True},
        "decision": decision,
        "failure_class": None,
        "route_hint": "run_full_s3",
        "reason_codes": [],
    }
    if include_worthiness:
        report["full_s3_worthiness"] = {"path": "experiment/results/c2c_full_s3_worthiness.json", "score": worthiness_score, "decision": "run_full_s3", "threshold": 0.6}
        (results / "c2c_full_s3_worthiness.json").write_text(
            json.dumps(
                {
                    "schema_version": "c2c_full_s3_worthiness_v1",
                    "created_at": "2026-07-03T00:00:00+00:00",
                    "candidate_id": "v1",
                    "score": worthiness_score,
                    "threshold": 0.6,
                    "components": {},
                    "neutral_proxy_budget_remaining": True,
                    "decision": "run_full_s3",
                    "reason_codes": [],
                }
            ),
            encoding="utf-8",
        )
    (results / "c2c_proxy_decision_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_s3_gate_passes_valid_proxy_contracts(tmp_path: Path) -> None:
    _write_required_outputs(tmp_path)
    _write_proxy_contracts(tmp_path, decision="proxy_pass")

    report = run_stage_gate("S3_experiment", tmp_path, _config(tmp_path)).to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "proxy_decision_allows_full_s3" for check in report["checks"])


def test_s3_gate_retries_neutral_proxy_without_worthiness(tmp_path: Path) -> None:
    _write_required_outputs(tmp_path)
    _write_proxy_contracts(tmp_path, decision="neutral_proxy_full_s3", include_worthiness=False)

    report = run_stage_gate("S3_experiment", tmp_path, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "neutral_proxy_requires_worthiness_score" for check in report["checks"])


def test_s3_candidate_selection_requires_planner_and_patch_gate_locks(tmp_path: Path) -> None:
    _write_required_outputs(tmp_path, proxy_screen={"enabled": False, "status": "skipped"})
    results = tmp_path / "experiment" / "results"
    patch_dir = tmp_path / "plan" / "code_patches" / "v1"
    patch_dir.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "plan" / "code_patches" / "patch_manifest.json"
    patch = patch_dir / "patch.json"
    contract = patch_dir / "implementation_contract.json"
    manifest.write_text(json.dumps({"status": "ok", "selected_candidate_id": "v1"}), encoding="utf-8")
    patch.write_text(json.dumps({"candidate_id": "v1"}), encoding="utf-8")
    contract.write_text(json.dumps({"candidate_id": "v1"}), encoding="utf-8")
    (results / "s3_candidate_selection.json").write_text(
        json.dumps(
            {
                "selected_candidate_id": "v1",
                "patch_manifest": {"rel_path": "plan/code_patches/patch_manifest.json", "sha256": sha256_file(manifest)},
                "selected_patch": {"rel_path": "plan/code_patches/v1/patch.json", "sha256": sha256_file(patch)},
                "selected_implementation_contract": {"rel_path": "plan/code_patches/v1/implementation_contract.json", "sha256": sha256_file(contract)},
            }
        ),
        encoding="utf-8",
    )

    report = run_stage_gate("S3_experiment", tmp_path, {"project": {"workspace_root": str(tmp_path)}}).to_dict()

    assert report["status"] == "FAIL"
    lock_check = next(check for check in report["checks"] if check["name"] == "s3_s2_5_artifact_lock_sha256")
    assert {item["name"] for item in lock_check["details"]["mismatches"]} >= {
        "selected_patch_gate_report",
        "selected_planner_gate_report",
        "selected_variant_scorecard",
    }
