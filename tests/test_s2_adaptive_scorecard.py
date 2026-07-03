from auto_research.s2_feedback_policy import build_s2_adaptive_policy, build_s2_score_adjustment_report
from auto_research.s2_planner_contracts import build_s2_candidate_pool, build_s2_planner_gate_report, build_s2_variant_scorecard


def _direction() -> dict:
    return {
        "direction_id": "direction_x",
        "mechanism_axis": "routing",
        "integration_point": "projector",
        "control_signal": "utility",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
    }


def _variant(variant_id: str, integration_point: str, *, diagnostics: list[str] | None = None) -> dict:
    return {
        "id": variant_id,
        "title": variant_id,
        "direction_id": "direction_x",
        "s1_direction_id": "direction_x",
        "variant_fingerprint": f"fp_{variant_id}",
        "mechanism_axis": "routing",
        "integration_point": integration_point,
        "control_signal": "utility",
        "expected_files": [f"rosetta/model/{integration_point}.py"],
        "ablation_switch": f"disable_{variant_id}",
        "experiment_contract": {
            "expected_files": [f"rosetta/model/{integration_point}.py"],
            "ablation_switch": f"disable_{variant_id}",
            "diagnostics_required": diagnostics or [],
        },
    }


def _config() -> dict:
    return {"c2c": {"allowed_prefixes": ["rosetta/model"], "s2_adaptive_policy": {"enabled": True}}}


def test_no_history_policy_keeps_base_and_adjusted_scores_equal() -> None:
    direction = _direction()
    candidates = [_variant("old_point", "projector"), _variant("new_point", "wrapper")]
    pool = build_s2_candidate_pool(direction=direction, candidates=candidates, source="unit")
    context = {"direction_id": "direction_x", "recent_failures": [], "attempt_counters": {}, "dragging_datasets": []}
    policy = build_s2_adaptive_policy(context, _config())

    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=pool,
        selected_variant=candidates[0],
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint={},
        planner_memory={"entries": []},
        feedback=[],
        feedback_context=context,
        adaptive_policy=policy,
        config=_config(),
    )

    assert policy["history_sufficient"] is False
    assert all(row["base_score"] == row["adjusted_score"] for row in scorecard["ranking"])


def test_adaptive_scorecard_selects_new_integration_after_proxy_failure() -> None:
    direction = _direction()
    old_variant = _variant("old_point", "projector")
    new_variant = _variant("new_point", "wrapper")
    pool = build_s2_candidate_pool(direction=direction, candidates=[old_variant, new_variant], source="unit")
    context = {
        "direction_id": "direction_x",
        "recent_failures": [
            {
                "mechanism_axis": "routing",
                "integration_point": "projector",
                "control_signal": "utility",
                "failure_class": "proxy_false_positive",
            }
        ],
        "attempt_counters": {"same_direction_proxy_failures": 1, "max_same_direction_proxy_failures": 2},
        "dragging_datasets": [],
    }
    policy = build_s2_adaptive_policy(context, _config())

    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=pool,
        selected_variant=old_variant,
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint={},
        planner_memory={"entries": []},
        feedback=[],
        feedback_context=context,
        adaptive_policy=policy,
        config=_config(),
    )
    report = build_s2_score_adjustment_report(
        direction=direction,
        candidate_pool=pool,
        scorecard=scorecard,
        adaptive_policy=policy,
        feedback_context=context,
    )

    assert scorecard["selected_variant_id"] == "new_point"
    assert report["selected_variant_id"] == "new_point"
    assert {row["variant_id"] for row in report["adjustments"]} == {"old_point", "new_point"}


def test_planner_gate_passes_when_adaptive_selector_changes_variant() -> None:
    direction = _direction()
    old_variant = _variant("old_point", "projector")
    new_variant = _variant("new_point", "wrapper")
    pool = build_s2_candidate_pool(direction=direction, candidates=[old_variant, new_variant], source="unit")
    context = {
        "direction_id": "direction_x",
        "recent_failures": [{"mechanism_axis": "routing", "integration_point": "projector", "failure_class": "proxy_negative"}],
        "attempt_counters": {"same_direction_proxy_failures": 1, "max_same_direction_proxy_failures": 2},
        "dragging_datasets": [],
    }
    policy = build_s2_adaptive_policy(context, _config())
    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=pool,
        selected_variant=old_variant,
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint={},
        planner_memory={"entries": []},
        feedback=[],
        feedback_context=context,
        adaptive_policy=policy,
        config=_config(),
    )
    report = build_s2_score_adjustment_report(direction=direction, candidate_pool=pool, scorecard=scorecard, adaptive_policy=policy, feedback_context=context)
    selected = new_variant
    contract = {
        "direction_id": "direction_x",
        "variant_id": "new_point",
        "variant_fingerprint": "fp_new_point",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "expected_files": ["rosetta/model/wrapper.py"],
        "resource_budget": {},
        "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
        "ablation": {"switch": "disable_new_point", "control": "off"},
    }
    fingerprint = {"variant_fingerprint": "fp_new_point", "is_repeat": False, "mode": "regular"}

    gate = build_s2_planner_gate_report(
        direction=direction,
        candidate_pool=pool,
        scorecard=scorecard,
        next_variant=selected,
        variant_contract=contract,
        variant_fingerprint=fingerprint,
        adaptive_policy=policy,
        score_adjustment_report=report,
        config=_config(),
    )

    assert gate["gate"] == "pass"
    assert gate["selected_variant_id"] == "new_point"


def test_force_new_direction_makes_planner_gate_fail_to_s1() -> None:
    direction = _direction()
    variant = _variant("old_point", "projector")
    pool = build_s2_candidate_pool(direction=direction, candidates=[variant], source="unit")
    context = {
        "direction_id": "direction_x",
        "recent_failures": [{"mechanism_axis": "routing", "integration_point": "projector", "failure_class": "full_s3_method_failure"}],
        "attempt_counters": {"same_direction_full_s3_failures": 1, "max_same_direction_full_s3_failures": 1},
        "dragging_datasets": [],
    }
    policy = build_s2_adaptive_policy(context, _config())
    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=pool,
        selected_variant=variant,
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint={},
        planner_memory={"entries": []},
        feedback=[],
        feedback_context=context,
        adaptive_policy=policy,
        config=_config(),
    )
    report = build_s2_score_adjustment_report(direction=direction, candidate_pool=pool, scorecard=scorecard, adaptive_policy=policy, feedback_context=context)
    gate = build_s2_planner_gate_report(
        direction=direction,
        candidate_pool=pool,
        scorecard=scorecard,
        next_variant=variant,
        variant_contract={
            "direction_id": "direction_x",
            "variant_id": "old_point",
            "variant_fingerprint": "fp_old_point",
            "mechanism_axis": "routing",
            "integration_point": "projector",
            "control_signal": "utility",
            "expected_files": ["rosetta/model/projector.py"],
            "resource_budget": {},
            "expected_metric_signature": {"primary_metric": "three_dataset_mean"},
            "ablation": {"switch": "disable_old_point", "control": "off"},
        },
        variant_fingerprint={"variant_fingerprint": "fp_old_point", "is_repeat": False, "mode": "regular"},
        adaptive_policy=policy,
        score_adjustment_report=report,
        config=_config(),
    )

    assert policy["route_constraints"]["force_new_direction"] is True
    assert gate["gate"] == "fail"
    assert gate["return_to"] == "S1_literature"
