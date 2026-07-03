import json
from pathlib import Path

from auto_research.s3_proxy_contracts import (
    build_c2c_effective_proxy_policy,
    build_c2c_full_s3_worthiness_score,
    build_c2c_proxy_baseline_fingerprint,
    build_c2c_proxy_cache_report,
    build_c2c_proxy_calibration_policy,
    build_c2c_proxy_decision_report,
)


def _proxy_cfg(**overrides):
    cfg = {
        "enabled": True,
        "min_proxy_mean_delta": -0.3,
        "max_proxy_dataset_regression": 1.5,
        "allow_neutral_proxy_full_s3": True,
        "neutral_proxy_min_delta": -0.1,
        "neutral_proxy_max_dataset_regression": 0.25,
        "eval_datasets": ["mmlu-redux", "ai2-arc", "openbookqa"],
        "eval_limit": 64,
        "adaptive_policy": {
            "enabled": True,
            "min_history_for_adjustment": 3,
            "false_positive_rate_threshold": 0.5,
            "tighten_min_proxy_delta_to": -0.1,
            "tighten_max_dataset_regression_to": 0.75,
        },
        "full_s3_worthiness": {"enabled": True, "min_score": 0.60, "neutral_proxy_budget_per_direction": 1},
    }
    cfg.update(overrides)
    return cfg


def _config(proxy_cfg=None):
    return {
        "c2c": {
            "snapshot_path": "external/c2c_snapshot",
            "env_python": "/usr/bin/python3",
            "model_map": {"base": "/models/base"},
            "baseline": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "dataset_root": "/datasets/c2c",
            "small_loop": {"proxy_screen": proxy_cfg or _proxy_cfg()},
        },
        "experiment": {"gpu_policy": {"gpu_ids": [0]}},
    }


def _write_plan_locks(root: Path) -> None:
    patch_dir = root / "plan" / "code_patches" / "v1"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (root / "plan" / "s2_planner").mkdir(parents=True, exist_ok=True)
    (root / "plan" / "plan.yaml").write_text("execution:\n  collector: c2c_small_loop\n", encoding="utf-8")
    (patch_dir / "patch.json").write_text(json.dumps({"candidate_id": "v1", "operations": []}), encoding="utf-8")
    (patch_dir / "implementation_contract.json").write_text(json.dumps({"candidate_id": "v1"}), encoding="utf-8")
    (root / "plan" / "code_patches" / "patch_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "selected_candidate_id": "v1",
                "selected_patch": {
                    "candidate_id": "v1",
                    "patch_json": "plan/code_patches/v1/patch.json",
                    "implementation_contract": "plan/code_patches/v1/implementation_contract.json",
                },
            }
        ),
        encoding="utf-8",
    )


def test_proxy_baseline_fingerprint_cache_hit(tmp_path: Path) -> None:
    _write_plan_locks(tmp_path)
    cache = tmp_path / "experiment" / "results" / "c2c_proxy_baseline.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean": 50.0}), encoding="utf-8")

    fingerprint = build_c2c_proxy_baseline_fingerprint(
        project_root=tmp_path,
        config=_config(),
        proxy_config=_proxy_cfg(),
        execution={"collector": "c2c_small_loop"},
    )
    report = build_c2c_proxy_cache_report(
        expected_fingerprint=fingerprint,
        actual_fingerprint=fingerprint,
        baseline_cache_path=cache,
        baseline_cache_exists=True,
    )

    assert report["cache_status"] == "hit"
    assert report["action"] == "reuse"


def test_proxy_config_change_invalidates_cache(tmp_path: Path) -> None:
    _write_plan_locks(tmp_path)
    cache = tmp_path / "experiment" / "results" / "c2c_proxy_baseline.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean": 50.0}), encoding="utf-8")

    old = build_c2c_proxy_baseline_fingerprint(project_root=tmp_path, config=_config(_proxy_cfg(eval_limit=64)), proxy_config=_proxy_cfg(eval_limit=64))
    new = build_c2c_proxy_baseline_fingerprint(project_root=tmp_path, config=_config(_proxy_cfg(eval_limit=32)), proxy_config=_proxy_cfg(eval_limit=32))
    report = build_c2c_proxy_cache_report(
        expected_fingerprint=new,
        actual_fingerprint=old,
        baseline_cache_path=cache,
        baseline_cache_exists=True,
    )

    assert report["cache_status"] == "invalidated"
    assert report["action"] == "rerun_baseline"


def test_effective_policy_tightens_after_false_positive_history(tmp_path: Path) -> None:
    history = {
        "summary": {
            "candidate_count": 3,
            "proxy_false_positive_rate": 0.7,
            "dataset_error_summary": {"openbookqa": {"count": 4, "misprediction_count": 2}},
            "method_feedback": {
                "risky_mechanisms": [{"mechanism_type": "routing", "false_positive_rate": 0.7}],
                "risky_integration_points": [{"integration_point": "wrapper", "false_positive_rate": 0.25}],
            },
        }
    }
    calibration = build_c2c_proxy_calibration_policy(
        project_root=tmp_path,
        config=_config(),
        direction_fingerprint={"mechanism_type": "routing", "integration_point": "wrapper"},
        calibration_history=history,
    )
    effective = build_c2c_effective_proxy_policy(static_proxy_config=_proxy_cfg(), calibration_policy=calibration)

    assert calibration["adjustments_active"] is True
    assert effective["effective_policy"]["min_proxy_mean_delta"] == -0.1
    assert effective["effective_policy"]["max_proxy_dataset_regression"] == 0.75
    assert "openbookqa_proxy_regression_risk" in effective["reason_codes"]


def test_proxy_decision_report_passes_positive_delta() -> None:
    policy = build_c2c_effective_proxy_policy(static_proxy_config=_proxy_cfg(), calibration_policy={})
    report = build_c2c_proxy_decision_report(
        candidate={"id": "v1", "variant_fingerprint": "fp"},
        proxy_screen={
            "status": "passed",
            "metrics": {"mean": 50.4, "datasets": {"mmlu-redux": 50.4}},
            "baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
            "activation_smoke": {"status": "passed"},
            "signals": {"has_executable_change": True},
        },
        baseline_fingerprint={"fingerprint_hash": "basefp"},
        effective_proxy_policy=policy,
        patch_gate_report={"gate": "pass"},
    )

    assert report["decision"] == "proxy_pass"
    assert report["route_hint"] == "run_full_s3"
    assert report["deltas"]["mean_delta"] == 0.4


def test_neutral_proxy_requires_worthy_full_s3_score() -> None:
    policy = build_c2c_effective_proxy_policy(static_proxy_config=_proxy_cfg(), calibration_policy={})
    proxy = {
        "status": "passed",
        "metrics": {"mean": 49.95, "datasets": {"mmlu-redux": 49.95}},
        "baseline_metrics": {"mean": 50.0, "datasets": {"mmlu-redux": 50.0}},
        "activation_smoke": {"status": "passed"},
        "signals": {"has_executable_change": True},
    }
    worthiness = build_c2c_full_s3_worthiness_score(
        candidate={"id": "v1", "novelty_score": 0.7},
        proxy_screen=proxy,
        effective_proxy_policy=policy,
        patch_gate_report={"gate": "pass"},
        variant_scorecard={"selected_variant_id": "v1", "ranking": [{"variant_id": "v1", "score": 0.72, "decision": "selected"}]},
    )
    report = build_c2c_proxy_decision_report(
        candidate={"id": "v1", "variant_fingerprint": "fp"},
        proxy_screen=proxy,
        baseline_fingerprint={"fingerprint_hash": "basefp"},
        effective_proxy_policy=policy,
        patch_gate_report={"gate": "pass"},
        full_s3_worthiness=worthiness,
    )

    assert worthiness["decision"] == "run_full_s3"
    assert report["decision"] == "neutral_proxy_full_s3"
