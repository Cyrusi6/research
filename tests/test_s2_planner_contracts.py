from auto_research.s2_planner_contracts import (
    build_s2_5_patch_gate_report,
    build_s2_candidate_pool,
    build_s2_implementation_contract,
    build_s2_planner_gate_report,
    build_s2_variant_scorecard,
)


def _direction() -> dict:
    return {
        "direction_id": "utility_predicted_cache_routing",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "hypothesis": "Route transferred cache states with utility.",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
        "expected_files": ["rosetta/model/wrapper.py"],
    }


def _variant(**overrides) -> dict:
    variant = {
        "id": "wrapper_utility_variant",
        "title": "Wrapper utility variant",
        "direction_id": "utility_predicted_cache_routing",
        "s1_direction_id": "utility_predicted_cache_routing",
        "variant_fingerprint": "fp_wrapper_utility",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "expected_files": ["rosetta/model/wrapper.py"],
        "ablation_switch": "disable_wrapper_utility",
        "experiment_contract": {
            "expected_files": ["rosetta/model/wrapper.py"],
            "ablation_switch": "disable_wrapper_utility",
        },
        "implementation_plan": {"scope": "small"},
        "risk_budget": {"forbidden_files": ["script/evaluation/*"]},
    }
    variant.update(overrides)
    return variant


def _contract(variant: dict) -> dict:
    return {
        "direction_id": "utility_predicted_cache_routing",
        "variant_id": variant["id"],
        "variant_fingerprint": variant["variant_fingerprint"],
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "expected_files": variant.get("expected_files") or [],
        "resource_budget": {"max_hours": 4},
        "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
        "ablation": {"switch": variant.get("ablation_switch"), "control": "ablation-off"},
    }


def _fingerprint(**overrides) -> dict:
    payload = {
        "direction_id": "utility_predicted_cache_routing",
        "variant_id": "wrapper_utility_variant",
        "variant_fingerprint": "fp_wrapper_utility",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "is_repeat": False,
        "mode": "regular",
    }
    payload.update(overrides)
    return payload


def _scorecard_bundle(variant: dict | None = None, fingerprint: dict | None = None) -> tuple[dict, dict, dict, dict]:
    direction = _direction()
    variant = variant or _variant()
    fingerprint = fingerprint or _fingerprint(variant_id=variant["id"], variant_fingerprint=variant["variant_fingerprint"])
    pool = build_s2_candidate_pool(direction=direction, candidates=[variant], source="unit_test")
    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=pool,
        selected_variant=variant,
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint=fingerprint,
        planner_memory={"entries": []},
        feedback=[],
        config={"c2c": {"allowed_prefixes": ["rosetta/model"]}},
    )
    gate = build_s2_planner_gate_report(
        direction=direction,
        candidate_pool=pool,
        scorecard=scorecard,
        next_variant=variant,
        variant_contract=_contract(variant),
        variant_fingerprint=fingerprint,
        config={"c2c": {"allowed_prefixes": ["rosetta/model"]}},
    )
    return direction, pool, scorecard, gate


def test_s2_planner_contracts_pass_for_patch_ready_variant() -> None:
    _, pool, scorecard, gate = _scorecard_bundle()

    assert pool["candidate_count"] == 1
    assert scorecard["selected_variant_id"] == "wrapper_utility_variant"
    assert scorecard["ranking"][0]["decision"] == "selected"
    assert gate["gate"] == "pass"


def test_s2_planner_gate_fails_repeated_fingerprint() -> None:
    fingerprint = _fingerprint(is_repeat=True, history_fingerprints=["fp_wrapper_utility"])
    _, _, _, gate = _scorecard_bundle(fingerprint=fingerprint)

    assert gate["gate"] == "fail"
    assert "variant_fingerprint repeats a previous same-direction variant" in gate["errors"]


def test_s2_planner_gate_fails_missing_ablation_switch() -> None:
    variant = _variant(ablation_switch="", experiment_contract={"expected_files": ["rosetta/model/wrapper.py"]})
    _, _, _, gate = _scorecard_bundle(variant=variant)

    assert gate["gate"] == "fail"
    assert "next_variant.ablation_switch must be present" in gate["errors"]


def test_s2_patch_gate_passes_matching_executable_patch() -> None:
    direction = _direction()
    variant = _variant()
    planner_gate = _scorecard_bundle(variant=variant)[3]
    implementation_contract = build_s2_implementation_contract(
        direction=direction,
        selected_variant=variant,
        variant_contract=_contract(variant),
        planner_gate_report=planner_gate,
        config={"c2c": {"allowed_prefixes": ["rosetta/model"]}},
    )
    patch_manifest = {
        "status": "ok",
        "selected_candidate_id": variant["id"],
        "selected_patch": {"candidate_id": variant["id"]},
        "patches": [
            {
                "candidate_id": variant["id"],
                "status": "ok",
                "changed_files": ["rosetta/model/wrapper.py"],
                "has_executable_change": True,
                "validation": {
                    "status": "ok",
                    "activation_check": {"status": "ok"},
                    "risk_check": {"status": "ok"},
                    "mechanism_review": {"status": "ok"},
                },
            }
        ],
    }

    patch_gate = build_s2_5_patch_gate_report(
        patch_manifest=patch_manifest,
        implementation_contract=implementation_contract,
        planner_gate_report=planner_gate,
        variant_fingerprint=_fingerprint(),
        config={"code_patch": {"validation": {"runtime_smoke": {"enabled": False}}}},
    )

    assert patch_gate["gate"] == "pass"
    assert patch_gate["checks"]["selected_variant_matches_planner"] is True


def test_s2_patch_gate_fails_no_executable_change_as_repairable() -> None:
    direction = _direction()
    variant = _variant()
    planner_gate = _scorecard_bundle(variant=variant)[3]
    implementation_contract = build_s2_implementation_contract(
        direction=direction,
        selected_variant=variant,
        variant_contract=_contract(variant),
        planner_gate_report=planner_gate,
        config={},
    )
    patch_manifest = {
        "status": "no_valid_patch",
        "selected_candidate_id": variant["id"],
        "patches": [{"candidate_id": variant["id"], "status": "failed", "changed_files": [], "has_executable_change": False}],
    }

    patch_gate = build_s2_5_patch_gate_report(
        patch_manifest=patch_manifest,
        implementation_contract=implementation_contract,
        planner_gate_report=planner_gate,
        variant_fingerprint=_fingerprint(),
        config={},
    )

    assert patch_gate["gate"] == "fail"
    assert patch_gate["failure_class"] == "no_executable_change"
    assert patch_gate["repairable"] is True
