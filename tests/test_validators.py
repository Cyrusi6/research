import json
from pathlib import Path

from auto_research.c2c import default_c2c_ideas
from auto_research.s2_planner_contracts import (
    build_s2_5_patch_gate_report,
    build_s2_candidate_pool,
    build_s2_implementation_contract,
    build_s2_planner_gate_report,
    build_s2_variant_scorecard,
)
from auto_research.s2_feedback_policy import build_s2_adaptive_policy, build_s2_feedback_context, build_s2_score_adjustment_report
from auto_research.utils import sha256_file
from auto_research.validators import run_stage_gate
from auto_research.workspace import init_workspace


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"workspace_root": str(tmp_path)},
        "review": {"max_iterations": 2},
        "writing": {"claim_verification": {"min_pass_rate": 0.8}, "require_compile": False},
    }


def _direction_payload(direction_id: str = "utility_predicted_cache_routing") -> dict:
    return {
        "schema_version": "auto_research_direction_v1",
        "direction_id": direction_id,
        "title": "Utility Predicted Cache Routing",
        "mechanism_type": "utility_predicted_cache_routing",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "hypothesis": "Route transferred cache states with a utility signal.",
        "why_baseline_fails": "The baseline lacks downstream utility control.",
        "expected_metric_signature": {"primary_metric": "three_dataset_mean", "expected_direction": "increase"},
        "required_evidence_refs": [{"source_type": "paper", "source_label": "p", "claim": "support"}],
        "counterevidence_refs": [{"source_type": "failure_feedback", "source_label": "risk", "claim": "risk"}],
        "implementation_surface_refs": [{"source_type": "code", "source_label": "rosetta/model/wrapper.py", "claim": "surface"}],
        "known_negative_memory_refs": [],
        "go_to_s2_conditions": ["evidence resolved"],
        "return_to_s1_conditions": ["same-direction budget exhausted"],
        "expected_files": ["rosetta/model/wrapper.py"],
        "verification_commands": ["py_compile"],
        "used_shared_memory_refs": [],
    }


def _write_s1_contract(paths, *, direction_id: str = "utility_predicted_cache_routing", novelty_passed: bool = True) -> None:
    (paths.root / "references" / "papers").mkdir(parents=True, exist_ok=True)
    (paths.root / "references" / "papers" / "manifest.json").write_text(json.dumps({"papers": [{"id": "p"}]}), encoding="utf-8")
    literature = paths.root / "literature"
    literature.mkdir(parents=True, exist_ok=True)
    direction = _direction_payload(direction_id)
    (literature / "direction.json").write_text(json.dumps(direction), encoding="utf-8")
    (literature / "evidence_bundle.json").write_text(
        json.dumps({"schema_version": "auto_research_evidence_bundle_v1", "items": [{"source_type": "paper", "source_path": "p", "summary": "support"}]}),
        encoding="utf-8",
    )
    (literature / "direction_scorecard.json").write_text(
        json.dumps(
            {
                "schema_version": "auto_research_direction_scorecard_v1",
                "direction_id": direction_id,
                "evidence_item_count": 1,
                "required_evidence_ref_count": 1,
                "counterevidence_ref_count": 1,
                "implementation_surface_ref_count": 1,
                "novelty": {"status": "ok", "passed": novelty_passed},
                "go_to_s2_ready": novelty_passed,
                "go_to_s2_conditions": ["evidence resolved"],
                "return_to_s1_conditions": ["same-direction budget exhausted"],
            }
        ),
        encoding="utf-8",
    )
    (literature / "novelty_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "auto_research_novelty_audit_v1",
                "direction_id": direction_id,
                "status": "ok",
                "enabled": True,
                "passed": novelty_passed,
                "threshold": 0.6,
                "latest": {"status": "ok", "passed": novelty_passed, "audit": {"novelty_score": 0.8 if novelty_passed else 0.2}},
                "audits": [],
            }
        ),
        encoding="utf-8",
    )


def _write_c2c_s1_gate_context(paths, *, quality_gate: str = "pass", mutate_quality=None) -> None:
    intake = paths.root / "intake" / "c2c"
    literature = paths.root / "literature"
    c2c_literature = literature / "c2c"
    intake.mkdir(parents=True, exist_ok=True)
    c2c_literature.mkdir(parents=True, exist_ok=True)
    for rel, payload in {
        "static_bundle.json": {"schema_version": "c2c_static_bundle_v1"},
        "evidence_brief.json": {"summary": "brief"},
        "repo_card.json": {"repo": "c2c"},
        "paper_cards.json": [{"id": "p"}],
        "rebuttal_concern_matrix.json": {"top_concerns": []},
        "negative_result_memory.json": {"blocked": []},
        "baseline_evidence.json": {"name": "base", "mean": 50.0},
    }.items():
        (intake / rel).write_text(json.dumps(payload), encoding="utf-8")
    (intake / "result_ledger.csv").write_text("id,metric\n", encoding="utf-8")
    (intake / "paper_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "paper:p1", "source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl"})
        + "\n"
        + json.dumps({"chunk_id": "paper:p2", "source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl"})
        + "\n",
        encoding="utf-8",
    )
    (intake / "code_file_manifest.json").write_text(
        json.dumps({"files": [{"path": "rosetta/model/wrapper.py"}, {"path": "rosetta/model/aligner.py"}]}),
        encoding="utf-8",
    )
    (intake / "code_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "code:wrapper", "source_type": "code", "path": "rosetta/model/wrapper.py"})
        + "\n"
        + json.dumps({"chunk_id": "code:aligner", "source_type": "code", "path": "rosetta/model/aligner.py"})
        + "\n",
        encoding="utf-8",
    )
    (literature / "idea_debate.json").write_text(json.dumps({"strategy": "legacy_debate", "selected_ideas": []}), encoding="utf-8")
    (literature / "negative_constraints.json").write_text(json.dumps({"forbidden_patterns": []}), encoding="utf-8")
    (c2c_literature / "baseline_evidence.json").write_text(json.dumps({"name": "base", "mean": 50.0}), encoding="utf-8")
    (c2c_literature / "rebuttal_concern_matrix.json").write_text(json.dumps({"top_concerns": []}), encoding="utf-8")
    paper_ref = {"source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "source_label": "paper:p1", "claim": "support"}
    paper_ref2 = {"source_type": "paper", "source_path": "intake/c2c/paper_chunks.jsonl", "source_label": "paper:p2", "claim": "support 2"}
    code_ref = {"source_type": "code", "source_path": "rosetta/model/wrapper.py", "source_label": "rosetta/model/wrapper.py", "claim": "surface"}
    code_ref2 = {"source_type": "code", "source_path": "rosetta/model/aligner.py", "source_label": "rosetta/model/aligner.py", "claim": "surface 2"}
    counter_ref = {"source_type": "failure_feedback", "source_path": "intake/c2c/negative_result_memory.json", "source_label": "feedback:risk", "claim": "risk"}
    request_plan = {
        "schema_version": "c2c_s1_evidence_request_plan_v1",
        "request_plan_id": "req_plan",
        "evidence_requests": [
            {"request_id": "paper_support", "source_type": "paper", "query": "paper support", "keywords": ["paper"], "purpose": "support", "top_k": 2, "filters": {}, "must_resolve": True},
            {"request_id": "code_surface", "source_type": "code", "query": "code surface", "keywords": ["wrapper"], "purpose": "implementation_surface", "top_k": 2, "filters": {}, "must_resolve": True},
            {"request_id": "counter", "source_type": "failure_memory", "query": "risk", "keywords": ["risk"], "purpose": "counterevidence", "top_k": 1, "filters": {}, "must_resolve": True},
        ],
        "required_source_coverage": {"paper": 2, "code": 2, "counterevidence": 1},
        "retrieval_budget": {"top_k_per_request": 2, "max_total_items": 8, "min_score": 0.0},
        "forbidden_outputs": ["direction_decision", "selected_ideas", "evidence_bundle", "expected_files"],
        "request_rationale": "cover support, implementation, and counterevidence",
    }
    bundle = {
        "schema_version": "c2c_s1_deterministic_evidence_bundle_v1",
        "producer": "deterministic_retriever",
        "retriever_version": "c2c_s1_deterministic_keyword_v1",
        "request_plan_id": "req_plan",
        "items": [
            {"evidence_id": "ev_p1", "ref": paper_ref, "request_id": "paper_support", "purpose": "support", "source_type": "paper", "locator": "paper:p1", "source_path": paper_ref["source_path"], "source_label": paper_ref["source_label"], "summary": "support", "score": 3.0, "score_components": {}, "why_selected": "match", "source_hash": "h1"},
            {"evidence_id": "ev_p2", "ref": paper_ref2, "request_id": "paper_support", "purpose": "support", "source_type": "paper", "locator": "paper:p2", "source_path": paper_ref2["source_path"], "source_label": paper_ref2["source_label"], "summary": "support 2", "score": 3.0, "score_components": {}, "why_selected": "match", "source_hash": "h2"},
            {"evidence_id": "ev_code", "ref": code_ref, "request_id": "code_surface", "purpose": "implementation_surface", "source_type": "code", "locator": "rosetta/model/wrapper.py", "source_path": code_ref["source_path"], "source_label": code_ref["source_label"], "summary": "surface", "score": 3.0, "score_components": {}, "why_selected": "match", "source_hash": "h3"},
            {"evidence_id": "ev_code2", "ref": code_ref2, "request_id": "code_surface", "purpose": "implementation_surface", "source_type": "code", "locator": "rosetta/model/aligner.py", "source_path": code_ref2["source_path"], "source_label": code_ref2["source_label"], "summary": "surface 2", "score": 3.0, "score_components": {}, "why_selected": "match", "source_hash": "h4"},
            {"evidence_id": "ev_risk", "ref": counter_ref, "request_id": "counter", "purpose": "counterevidence", "source_type": "failure_feedback", "locator": "feedback:risk", "source_path": counter_ref["source_path"], "source_label": counter_ref["source_label"], "summary": "risk", "risks": ["risk"], "score": 3.0, "score_components": {}, "why_selected": "match", "source_hash": "h5"},
        ],
    }
    direction_decision = {
        "direction_id": "utility_predicted_cache_routing",
        "mechanism_direction": "Utility Predicted Cache Routing",
        "mechanism_type": "utility_predicted_cache_routing",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "core_hypothesis": "Route transferred cache states with a utility signal.",
        "why_baseline_fails": "The baseline lacks downstream utility control.",
        "why_this_direction": "Retrieved evidence supports utility routing.",
        "required_evidence_refs": [paper_ref, paper_ref2],
        "counterevidence_refs": [counter_ref],
        "implementation_surface_refs": [code_ref, code_ref2],
        "expected_files": ["rosetta/model/wrapper.py", "rosetta/model/aligner.py"],
        "allowed_variants": ["soft utility routing"],
        "forbidden_patterns": ["hard gate"],
        "failure_focus": ["coverage"],
        "verification_commands": ["py_compile"],
    }
    idea = {
        "id": "utility_predicted_cache_routing",
        "title": "Utility Predicted Cache Routing",
        "selected": True,
        "hypothesis": direction_decision["core_hypothesis"],
        "novelty_score": 7,
        "feasibility_score": 7,
        "mechanism_type": "utility_predicted_cache_routing",
        "reviewer_risk_response": "watch coverage",
        "expected_files": direction_decision["expected_files"],
        "verification_commands": ["py_compile"],
        "evidence_refs": [paper_ref, paper_ref2],
        "counterevidence_refs": [counter_ref],
        "code_refs": [code_ref, code_ref2],
    }
    (literature / "ideas.json").write_text(json.dumps([idea]), encoding="utf-8")
    (literature / "idea_debate.json").write_text(json.dumps({"strategy": "codex_two_phase_evidence_direction", "selected_ideas": [idea]}), encoding="utf-8")
    (c2c_literature / "evidence_request_plan.json").write_text(json.dumps(request_plan), encoding="utf-8")
    (c2c_literature / "evidence_requests.json").write_text(json.dumps(request_plan["evidence_requests"]), encoding="utf-8")
    (c2c_literature / "evidence_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (c2c_literature / "direction_decision.json").write_text(json.dumps(direction_decision), encoding="utf-8")
    (c2c_literature / "evidence_session.json").write_text(json.dumps({"schema_version": "c2c_s1_two_phase_session_v1", "status": "ok", "phases": []}), encoding="utf-8")
    (c2c_literature / "evidence_ref_report.json").write_text(json.dumps({"schema_version": "s1_evidence_ref_report_v1", "status": "pass", "counts": {"resolved": 5, "unresolved": 0}, "resolved": [], "errors": []}), encoding="utf-8")
    quality = {
        "schema_version": "c2c_s1_evidence_quality_v1",
        "direction_id": "utility_predicted_cache_routing",
        "support_coverage": {"paper": 3, "rebuttal": 1, "code": 2, "failure_memory": 1},
        "counterevidence": {"count": 2, "resolved_count": 2},
        "implementation_surface_coverage": 0.75,
        "implementation_surface": {"target_count": 4, "covered_count": 3, "targets": [], "covered": []},
        "unresolved_ref_count": 0,
        "shared_memory_checked": True,
        "novelty_score": 0.68,
        "same_direction_similarity": 0.21,
        "direction_bundle_ref_report": {"schema_version": "s1_direction_bundle_ref_report_v1", "status": "pass", "counts": {"errors": 0}, "errors": []},
        "thresholds": {},
        "failed_rules": [] if quality_gate == "pass" else ["support_coverage.paper"],
        "coverage_contributors": {},
        "gate": quality_gate,
    }
    if mutate_quality:
        mutate_quality(quality)
    (c2c_literature / "evidence_quality_score.json").write_text(json.dumps(quality), encoding="utf-8")
    (c2c_literature / "evidence_retrieval_trace.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_s1_deterministic_retrieval_trace_v1",
                "direction_id": "utility_predicted_cache_routing",
                "retriever_version": "c2c_s1_deterministic_keyword_v1",
                "request_plan_id": "req_plan",
                "requests": [],
                "evidence_requests": [],
                "candidate_counts": {"paper": 2, "code": 2, "failure_memory": 1},
                "selected_refs": [paper_ref, paper_ref2, code_ref, code_ref2, counter_ref],
                "resolved_ref_count": 6,
                "unresolved_ref_count": quality["unresolved_ref_count"],
                "resolved_refs": [],
                "unresolved_refs": [],
                "unfilled_must_resolve_requests": [],
                "coverage": {"paper": 3, "code": 2, "counterevidence": 2},
                "coverage_contributors": {},
                "deterministic": True,
                "retrieval_inputs_hash": "tracehash",
                "quality_gate": {"gate": quality["gate"], "failed_rules": quality["failed_rules"], "thresholds": {}},
                "direction_fingerprint": {"fingerprint": "fp", "same_direction_similarity": 0.21, "artifact": "literature/c2c/direction_fingerprint.json"},
            }
        ),
        encoding="utf-8",
    )
    (c2c_literature / "direction_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_s1_direction_fingerprint_v1",
                "direction_id": "utility_predicted_cache_routing",
                "fingerprint": "fp",
                "features": {"mechanism_axis": "routing"},
                "feature_tokens": ["routing"],
                "history": [],
                "same_direction_similarity": 0.21,
            }
        ),
        encoding="utf-8",
    )


def _write_s2_contracts(
    paths,
    *,
    direction_id: str = "utility_predicted_cache_routing",
    mutate_contract=None,
    mutate_fingerprint=None,
    mutate_next_variant=None,
    mutate_scorecard=None,
    mutate_planner_gate=None,
) -> None:
    plan_dir = paths.root / "plan"
    planner_dir = plan_dir / "s2_planner"
    planner_dir.mkdir(parents=True, exist_ok=True)
    direction = _direction_payload(direction_id)
    next_variant = {
        "id": "wrapper_utility_variant",
        "title": "Wrapper utility variant",
        "direction_id": direction_id,
        "s1_direction_id": direction_id,
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
        "implementation_plan": {"scope": "small", "integration_points": ["wrapper"]},
        "failure_feedback_refs": [],
    }
    planner = {
        "schema_version": "auto_research_planner_decision_v1",
        "direction_id": direction_id,
        "planner_summary": "Plan one utility variant.",
        "planning_mode": "same_direction_variant",
        "used_shared_memory_refs": [],
        "next_variant": next_variant,
    }
    contract = {
        "schema_version": "auto_research_variant_contract_v1",
        "direction_id": direction_id,
        "variant_id": "wrapper_utility_variant",
        "title": "Wrapper utility variant",
        "mode": "regular",
        "variant_fingerprint": "fp_wrapper_utility",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "hypothesis": direction["hypothesis"],
        "why_next": "Targets wrapper routing.",
        "expected_files": ["rosetta/model/wrapper.py"],
        "implementation_surface_refs": direction["implementation_surface_refs"],
        "resource_budget": {},
        "expected_metric_signature": direction["expected_metric_signature"],
        "ablation": {"switch": "disable_wrapper_utility", "control": "ablation-off"},
        "acceptance": {"min_delta_to_pass": 0.1, "max_dataset_regression": 2.0},
        "failure_routing": {
            "go_to_s3_conditions": ["gate passes"],
            "return_to_s2_conditions": ["patch invalid"],
            "return_to_s1_conditions": ["budget exhausted"],
        },
        "used_shared_memory_refs": [],
    }
    fingerprint = {
        "schema_version": "auto_research_variant_fingerprint_v1",
        "direction_id": direction_id,
        "variant_id": "wrapper_utility_variant",
        "variant_fingerprint": "fp_wrapper_utility",
        "mechanism_axis": "routing",
        "integration_point": "wrapper",
        "control_signal": "utility",
        "history_fingerprints": [],
        "is_repeat": False,
        "mode": "regular",
    }
    if mutate_contract:
        mutate_contract(contract)
    if mutate_fingerprint:
        mutate_fingerprint(fingerprint)
    if mutate_next_variant:
        mutate_next_variant(next_variant)
    candidate_pool = build_s2_candidate_pool(direction=direction, candidates=[next_variant], source="test_fixture")
    feedback_context = build_s2_feedback_context(project_root=paths.root, direction=direction, config={})
    adaptive_policy = build_s2_adaptive_policy(feedback_context, {})
    scorecard = build_s2_variant_scorecard(
        direction=direction,
        candidate_pool=candidate_pool,
        selected_variant=next_variant,
        evidence_quality={"support_coverage": {"paper": 2, "code": 2}},
        variant_fingerprint=fingerprint,
        planner_memory={"entries": []},
        feedback=[],
        feedback_context=feedback_context,
        adaptive_policy=adaptive_policy,
        config={},
    )
    score_adjustment_report = build_s2_score_adjustment_report(
        direction=direction,
        candidate_pool=candidate_pool,
        scorecard=scorecard,
        adaptive_policy=adaptive_policy,
        feedback_context=feedback_context,
    )
    if mutate_scorecard:
        mutate_scorecard(scorecard)
    planner_gate = build_s2_planner_gate_report(
        direction=direction,
        candidate_pool=candidate_pool,
        scorecard=scorecard,
        next_variant=next_variant,
        variant_contract=contract,
        variant_fingerprint=fingerprint,
        adaptive_policy=adaptive_policy,
        score_adjustment_report=score_adjustment_report,
        config={},
    )
    if mutate_planner_gate:
        mutate_planner_gate(planner_gate)
    (paths.root / "literature").mkdir(parents=True, exist_ok=True)
    (paths.root / "literature" / "direction.json").write_text(json.dumps(direction), encoding="utf-8")
    (plan_dir / "next_variant.json").write_text(json.dumps(planner), encoding="utf-8")
    (plan_dir / "planner_decision.json").write_text(json.dumps(planner), encoding="utf-8")
    (plan_dir / "variant_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (plan_dir / "variant_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    (planner_dir / "candidate_pool.json").write_text(json.dumps(candidate_pool), encoding="utf-8")
    (planner_dir / "feedback_context.json").write_text(json.dumps(feedback_context), encoding="utf-8")
    (planner_dir / "adaptive_policy.json").write_text(json.dumps(adaptive_policy), encoding="utf-8")
    (planner_dir / "variant_scorecard.json").write_text(json.dumps(scorecard), encoding="utf-8")
    (planner_dir / "score_adjustment_report.json").write_text(json.dumps(score_adjustment_report), encoding="utf-8")
    (planner_dir / "next_variant.json").write_text(json.dumps(next_variant), encoding="utf-8")
    (planner_dir / "planner_gate_report.json").write_text(json.dumps(planner_gate), encoding="utf-8")


def _write_s2_patch_gate_artifacts(
    paths,
    *,
    status: str = "ok",
    selected_candidate_id: str = "wrapper_utility_variant",
    changed_files: list[str] | None = None,
    has_executable_change: bool = True,
    mutate_patch_gate=None,
) -> None:
    plan_dir = paths.root / "plan"
    patch_dir = plan_dir / "code_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    direction = json.loads((paths.root / "literature" / "direction.json").read_text(encoding="utf-8"))
    next_variant = json.loads((plan_dir / "s2_planner" / "next_variant.json").read_text(encoding="utf-8"))
    variant_contract = json.loads((plan_dir / "variant_contract.json").read_text(encoding="utf-8"))
    variant_fingerprint = json.loads((plan_dir / "variant_fingerprint.json").read_text(encoding="utf-8"))
    planner_gate = json.loads((plan_dir / "s2_planner" / "planner_gate_report.json").read_text(encoding="utf-8"))
    implementation_contract = build_s2_implementation_contract(
        direction=direction,
        selected_variant=next_variant,
        variant_contract=variant_contract,
        planner_gate_report=planner_gate,
        config={},
    )
    changed_files = changed_files if changed_files is not None else ["rosetta/model/wrapper.py"]
    entry = {
        "candidate_id": selected_candidate_id,
        "status": "ok" if status == "ok" else "failed",
        "changed_files": changed_files,
        "has_executable_change": has_executable_change,
        "validation": {
            "status": "ok" if status == "ok" else "validation_failed",
            "activation_check": {"status": "ok"},
            "risk_check": {"status": "ok"},
            "mechanism_review": {"status": "ok"},
        },
    }
    manifest = {
        "status": status,
        "selected_candidate_id": selected_candidate_id,
        "valid_patch_count": 1 if status == "ok" else 0,
        "valid_patch_ids": [selected_candidate_id] if status == "ok" else [],
        "candidates": [entry],
        "patches": [entry],
        "selected_patch": {"candidate_id": selected_candidate_id},
    }
    patch_gate = build_s2_5_patch_gate_report(
        patch_manifest=manifest,
        implementation_contract=implementation_contract,
        planner_gate_report=planner_gate,
        variant_fingerprint=variant_fingerprint,
        config={},
    )
    if mutate_patch_gate:
        mutate_patch_gate(patch_gate)
    (patch_dir / "patch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (patch_dir / "implementation_contract.json").write_text(json.dumps(implementation_contract), encoding="utf-8")
    (patch_dir / "patch_gate_report.json").write_text(json.dumps(patch_gate), encoding="utf-8")


def _write_minimal_c2c_plan(paths) -> None:
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


def test_s1_gate_returns_structured_retry_for_missing_ideas(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate", simulate=True)

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path))
    payload = report.to_dict()

    assert payload["schema_version"] == "stage_gate_v1"
    assert payload["status"] == "NEEDS_RETRY"
    assert payload["passed"] is False
    assert payload["checks"][0]["name"] == "direction_json_exists"


def test_s1_gate_retries_failed_novelty_audit(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s1_novelty_gate", simulate=True)
    _write_s1_contract(paths, direction_id="repeat_direction", novelty_passed=False)

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "s1_novelty_audit" and check["status"] == "NEEDS_RETRY" for check in report["checks"])


def test_s1_gate_passes_direction_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s1_direction_pass", simulate=True)
    _write_s1_contract(paths)

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "direction_schema" and check["status"] == "PASS" for check in report["checks"])
    assert any(check["name"] == "novelty_audit_schema" and check["status"] == "PASS" for check in report["checks"])


def test_s1_gate_passes_c2c_evidence_quality_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s1_c2c_quality_pass", simulate=True)
    _write_s1_contract(paths)
    _write_c2c_s1_gate_context(paths)

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "c2c_s1_evidence_quality_schema" and check["status"] == "PASS" for check in report["checks"])
    assert any(check["name"] == "s1_evidence_quality_gate" and check["status"] == "PASS" for check in report["checks"])


def test_s1_gate_retries_thin_c2c_evidence_quality_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_s1_c2c_quality_thin", simulate=True)
    _write_s1_contract(paths)
    _write_c2c_s1_gate_context(
        paths,
        quality_gate="fail",
        mutate_quality=lambda quality: quality.update(
            {
                "support_coverage": {"paper": 1, "rebuttal": 1, "code": 2, "failure_memory": 1},
                "failed_rules": ["support_coverage.paper"],
            }
        ),
    )

    report = run_stage_gate("S1_literature", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    quality = next(check for check in report["checks"] if check["name"] == "s1_evidence_quality_gate")
    assert quality["status"] == "NEEDS_RETRY"
    assert "support_coverage.paper" in quality["details"]["failed_rules"]


def test_s2_gate_passes_c2c_plan_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate", simulate=True)
    _write_s1_contract(paths)
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
    _write_s2_contracts(paths)

    config = _config(tmp_path)
    config["code_patch"] = {"validation": {"gate_mode": "strict"}}
    report = run_stage_gate("S2_plan", paths.root, config).to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "c2c_runtime_resource_selection_deferred" for check in report["checks"])
    assert any(check["name"] == "c2c_mechanism_novelty_gate" and check["status"] == "PASS" for check in report["checks"])
    assert any(check["name"] == "c2c_implementation_scope_gate" and check["status"] == "PASS" for check in report["checks"])
    assert any(check["name"] == "s2_planner_gate_contract" and check["status"] == "PASS" for check in report["checks"])


def test_s2_gate_retries_missing_variant_contract(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_missing_variant_contract", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    (paths.root / "plan" / "variant_contract.json").unlink()

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "variant_contract_json_exists" for check in report["checks"])


def test_s2_gate_retries_missing_s2_candidate_pool(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_missing_s2_pool", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    (paths.root / "plan" / "s2_planner" / "candidate_pool.json").unlink()

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "s2_candidate_pool_json_exists" for check in report["checks"])


def test_s2_gate_retries_adaptive_policy_hash_mismatch(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_s2_policy_hash", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    scorecard_path = paths.root / "plan" / "s2_planner" / "variant_scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["policy_hash"] = "wrong_hash"
    scorecard_path.write_text(json.dumps(scorecard), encoding="utf-8")

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    planner_gate = next(check for check in report["checks"] if check["name"] == "s2_planner_gate_contract")
    assert "variant_scorecard.policy_hash must match adaptive_policy.policy_hash" in planner_gate["details"]["errors"]


def test_s2_gate_retries_score_adjustment_missing_candidate_coverage(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_s2_adjustment_coverage", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    report_path = paths.root / "plan" / "s2_planner" / "score_adjustment_report.json"
    adjustment = json.loads(report_path.read_text(encoding="utf-8"))
    adjustment["adjustments"] = []
    report_path.write_text(json.dumps(adjustment), encoding="utf-8")

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    planner_gate = next(check for check in report["checks"] if check["name"] == "s2_planner_gate_contract")
    assert any("score_adjustment_report.adjustments must cover every candidate" in error for error in planner_gate["details"]["errors"])


def test_s2_gate_retries_s2_next_variant_direction_mismatch(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_s2_next_direction_mismatch", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths, mutate_next_variant=lambda variant: variant.update({"direction_id": "other_direction"}))

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    planner_gate = next(check for check in report["checks"] if check["name"] == "s2_planner_gate_contract")
    assert "next_variant.direction_id must match literature/direction.json direction_id" in planner_gate["details"]["errors"]


def test_s2_gate_retries_s2_next_variant_missing_ablation_switch(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_s2_next_missing_ablation", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)

    def remove_ablation(variant: dict) -> None:
        variant.pop("ablation_switch", None)
        variant["experiment_contract"].pop("ablation_switch", None)

    _write_s2_contracts(
        paths,
        mutate_next_variant=remove_ablation,
        mutate_contract=lambda contract: contract["ablation"].pop("switch"),
    )

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    planner_gate = next(check for check in report["checks"] if check["name"] == "s2_planner_gate_contract")
    assert "next_variant.ablation_switch must be present" in planner_gate["details"]["planner_gate_errors"]


def test_s2_gate_retries_direction_mismatch(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_direction_mismatch", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths, mutate_contract=lambda contract: contract.update({"direction_id": "other_direction"}))

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    handoff = next(check for check in report["checks"] if check["name"] == "s2_variant_handoff_contract")
    assert any("direction_id mismatch" in error for error in handoff["details"]["errors"])


def test_s2_gate_retries_missing_ablation_switch(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_missing_ablation", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths, mutate_contract=lambda contract: contract["ablation"].pop("switch"))

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    handoff = next(check for check in report["checks"] if check["name"] == "s2_variant_handoff_contract")
    assert "variant_contract.ablation.switch must be present" in handoff["details"]["errors"]


def test_s2_gate_retries_missing_metric_signature(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_missing_metric_signature", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths, mutate_contract=lambda contract: contract.update({"expected_metric_signature": {}}))

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    handoff = next(check for check in report["checks"] if check["name"] == "s2_variant_handoff_contract")
    assert "variant_contract.expected_metric_signature must be a non-empty object" in handoff["details"]["errors"]


def test_s2_gate_retries_repeated_variant_fingerprint(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_repeated_fingerprint", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(
        paths,
        mutate_fingerprint=lambda fingerprint: fingerprint.update({"history_fingerprints": ["fp_wrapper_utility"], "is_repeat": True}),
    )

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    handoff = next(check for check in report["checks"] if check["name"] == "s2_variant_handoff_contract")
    assert "variant_fingerprint repeats a previous same-direction variant" in handoff["details"]["errors"]


def test_s2_gate_retries_missing_patch_gate_when_manifest_exists(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_missing_patch_gate", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    _write_s2_patch_gate_artifacts(paths)
    (paths.root / "plan" / "code_patches" / "patch_gate_report.json").unlink()

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    assert any(check["name"] == "s2_5_patch_gate_report_json_exists" for check in report["checks"])


def test_s2_gate_retries_patch_gate_no_executable_change(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_patch_no_exec", simulate=True)
    _write_s1_contract(paths)
    _write_minimal_c2c_plan(paths)
    _write_s2_contracts(paths)
    _write_s2_patch_gate_artifacts(paths, status="no_valid_patch", has_executable_change=False, changed_files=[])

    report = run_stage_gate("S2_plan", paths.root, _config(tmp_path)).to_dict()

    assert report["status"] == "NEEDS_RETRY"
    patch_gate = next(check for check in report["checks"] if check["name"] == "s2_5_patch_gate_contract")
    assert patch_gate["status"] == "NEEDS_RETRY"
    patch_status = next(check for check in report["checks"] if check["name"] == "s2_5_patch_manifest_status")
    assert patch_status["status"] == "NEEDS_RETRY"


def test_s2_gate_retries_c2c_local_tuning_idea(tmp_path: Path) -> None:
    paths = init_workspace(_config(tmp_path), "topic", project_id="proj_gate_local_tuning", simulate=True)
    _write_s1_contract(paths)
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
    _write_s2_contracts(paths)
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
    _write_s1_contract(paths)
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
    _write_s2_contracts(paths)
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
