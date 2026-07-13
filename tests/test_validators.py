import json
from pathlib import Path

from auto_research.c2c import default_c2c_ideas
from auto_research.direction_contracts import build_variant_contract
from auto_research.s2_planner_contracts import (
    build_s2_5_patch_gate_report,
    build_s2_candidate_pool,
    build_s2_implementation_contract,
    build_s2_planner_gate_report,
    build_s2_variant_scorecard,
)
from auto_research.s2_feedback_policy import build_s2_adaptive_policy, build_s2_feedback_context, build_s2_score_adjustment_report
from auto_research.utils import sha256_file, write_json
from auto_research.validators import C2CE2EGateValidator, run_stage_gate
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
        "candidate_direction_hypotheses": [
            {"hypothesis_id": "routing_control_signal", "mechanism_axis": "routing", "integration_point": "wrapper", "control_signal": "utility", "why_plausible": "paper and code refs support wrapper routing", "uncertainty_axes": ["mechanism_axis"]},
            {"hypothesis_id": "alignment_surface_signal", "mechanism_axis": "alignment", "integration_point": "aligner", "control_signal": "representation_match", "why_plausible": "aligner code refs may support an alternate surface", "uncertainty_axes": ["implementation_surface"]},
        ],
        "uncertainty_axes": [{"axis_id": "mechanism_axis", "question": "which mechanism is best supported", "needed_sources": ["paper", "code"]}],
        "discriminating_evidence_requests": [
            {"request_id": "paper_support", "distinguishes": ["routing_control_signal", "alignment_surface_signal"], "decision_if_supported": "prefer supported mechanism", "decision_if_refuted": "request more evidence"},
            {"request_id": "code_surface", "distinguishes": ["wrapper", "aligner"], "decision_if_supported": "prefer editable surface", "decision_if_refuted": "block S2 handoff"},
        ],
        "must_have_before_direction": [
            {"source_type": "paper", "purpose": "support", "minimum": 2},
            {"source_type": "code", "purpose": "implementation_surface", "minimum": 2},
            {"source_type": "failure_memory", "purpose": "counterevidence", "minimum": 1},
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
    (c2c_literature / "direction_candidate_scorecard.json").write_text(
        json.dumps(
            {
                "schema_version": "c2c_s1_direction_candidate_scorecard_v1",
                "direction_id": "utility_predicted_cache_routing",
                "selected_direction_id": "utility_predicted_cache_routing",
                "candidates": [
                    {
                        "direction_id": "utility_predicted_cache_routing",
                        "mechanism_axis": "routing",
                        "integration_point": "wrapper",
                        "control_signal": "utility",
                        "score": 1.0,
                        "selected": True,
                        "evidence_refs": [paper_ref, paper_ref2],
                        "counterevidence_refs": [counter_ref],
                        "implementation_surface_refs": [code_ref, code_ref2],
                        "why_selected": "best supported by deterministic refs",
                        "why_not_selected": [],
                    },
                    {
                        "direction_id": "alignment_surface_signal",
                        "mechanism_axis": "alignment",
                        "integration_point": "aligner",
                        "control_signal": "representation_match",
                        "score": 0.42,
                        "selected": False,
                        "evidence_refs": [paper_ref],
                        "counterevidence_refs": [counter_ref],
                        "implementation_surface_refs": [code_ref2],
                        "why_selected": "",
                        "why_not_selected": ["less direct wrapper evidence"],
                    },
                ],
                "comparison_axes": ["evidence_support", "implementation_surface"],
                "coverage": {"candidate_count": 2, "non_selected_count": 1, "bundle_ref_count": 5},
            }
        ),
        encoding="utf-8",
    )
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


def test_c2c_e2e_gate_validates_readiness_audit_and_replay_reports(tmp_path: Path) -> None:
    project = tmp_path / "proj_e2e_gate"
    write_json(
        project / "meta" / "c2c_e2e_readiness_report.json",
        {
            "schema_version": "c2c_e2e_readiness_report_v1",
            "project_id": "proj_e2e_gate",
            "mode": "real",
            "gate": "pass",
            "checks": {
                "target_repo_exists": True,
                "ref_paper_exists": True,
                "ref_rebuttal_exists": True,
                "env_python_executable": True,
                "workspace_writable": True,
                "worktree_root_writable": True,
                "llm_config_ready": True,
                "dataset_paths_ready": True,
                "gpu_policy_ready": True,
                "s0_cache_compatible": True,
                "baseline_cache_valid_or_invalidated": True,
                "real_execution_hooks_ready": True,
            },
            "warnings": [],
            "blocking_reasons": [],
            "recommended_action": "run_c2c",
        },
    )
    write_json(
        project / "meta" / "c2c_artifact_audit_report.json",
        {
            "schema_version": "c2c_artifact_audit_report_v1",
            "project_id": "proj_e2e_gate",
            "gate": "pass",
            "audit_scope": "completed",
            "expected_stages": [],
            "skipped_stages": [],
            "summary": {"checked_artifacts": 1, "missing": 0, "schema_failures": 0, "missing_manifest_hash": 0, "hash_mismatches": 0, "stale_artifacts": 0},
            "by_stage": {},
            "blocking_reasons": [],
        },
    )
    write_json(
        project / "meta" / "c2c_replay_result.json",
        {
            "schema_version": "c2c_replay_result_v1",
            "project_id": "proj_e2e_gate",
            "status": "match",
            "replayed_decisions": {},
            "expected_decisions": {},
            "mismatches": [],
        },
    )
    write_json(
        project / "meta" / "c2c_real_smoke_record.json",
        {
            "schema_version": "c2c_real_smoke_record_v1",
            "project_id": "proj_e2e_gate",
            "readiness_gate": "pass",
            "execution_hooks_gate": "pass",
            "run_manifest_final_status": "completed",
            "artifact_audit_gate": "pass",
            "replay_status": "match",
            "last_stage": "S3_experiment",
            "s1_evidence_gate": "pass",
            "s2_planner_gate": "pass",
            "s2_5_patch_gate": "pass",
            "s3_proxy_decision": "proxy_pass",
            "route_decision": "complete",
            "blocking_reasons": [],
            "warnings": [],
        },
    )

    report = C2CE2EGateValidator(project, {}).validate().to_dict()

    assert report["status"] == "PASS"
    assert any(check["name"] == "c2c_real_smoke_record_schema" and check["status"] == "PASS" for check in report["checks"])
